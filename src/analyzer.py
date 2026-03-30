# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================

职责：
1. 封装 LLM 调用逻辑（通过 LiteLLM 统一调用 Gemini/Anthropic/OpenAI 等）
2. 结合技术面和消息面生成分析报告
3. 解析 LLM 响应为结构化 AnalysisResult
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.agent.skills.defaults import CORE_TRADING_SKILL_POLICY_ZH
from src.config import (
    Config,
    extra_litellm_params,
    get_api_keys_for_model,
    get_config,
    get_configured_llm_models,
    resolve_news_window_days,
)
from src.storage import persist_llm_usage
from src.data.stock_mapping import STOCK_NAME_MAP
from src.report_language import (
    get_signal_level,
    get_no_data_text,
    get_placeholder_text,
    get_unknown_text,
    infer_decision_type_from_advice,
    localize_chip_health,
    localize_confidence_level,
    normalize_report_language,
)
from src.schemas.report_schema import AnalysisReportSchema
from src.market_context import get_market_role, get_market_guidelines

logger = logging.getLogger(__name__)


def _resolve_report_language() -> str:
    """Resolve report language from OS env first, then config. Always returns 'en' or 'zh'."""
    _lang = os.environ.get("REPORT_LANGUAGE", "").strip().lower()
    if _lang in ("en", "english", "en-us", "en_us"):
        return "en"
    config_lang = (getattr(get_config(), "report_language", "en") or "en").strip().lower()
    if config_lang in ("en", "english", "en-us", "en_us"):
        return "en"
    return "zh"


def check_content_integrity(result: "AnalysisResult") -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if result.sentiment_score is None:
        missing.append("sentiment_score")
    advice = result.operation_advice
    if not advice or not isinstance(advice, str) or not advice.strip():
        missing.append("operation_advice")
    summary = result.analysis_summary
    if not summary or not isinstance(summary, str) or not summary.strip():
        missing.append("analysis_summary")
    dash = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = dash.get("core_conclusion")
    core = core if isinstance(core, dict) else {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    intel = intel if isinstance(intel, dict) else None
    if intel is None or "risk_alerts" not in intel:
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan")
        battle = battle if isinstance(battle, dict) else {}
        sp = battle.get("sniper_points")
        sp = sp if isinstance(sp, dict) else {}
        stop_loss = sp.get("stop_loss")
        if stop_loss is None or (isinstance(stop_loss, str) and not stop_loss.strip()):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    placeholder = get_placeholder_text(getattr(result, "report_language", "zh"))
    for field in missing_fields:
        if field == "sentiment_score":
            result.sentiment_score = 50
        elif field == "operation_advice":
            result.operation_advice = result.operation_advice or placeholder
        elif field == "analysis_summary":
            result.analysis_summary = result.analysis_summary or placeholder
        elif field == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            if "core_conclusion" not in result.dashboard:
                result.dashboard["core_conclusion"] = {}
            result.dashboard["core_conclusion"]["one_sentence"] = (
                result.dashboard["core_conclusion"].get("one_sentence") or placeholder
            )
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            if "intelligence" not in result.dashboard:
                result.dashboard["intelligence"] = {}
            if "risk_alerts" not in result.dashboard["intelligence"]:
                result.dashboard["intelligence"]["risk_alerts"] = []
        elif field == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            if "battle_plan" not in result.dashboard:
                result.dashboard["battle_plan"] = {}
            if "sniper_points" not in result.dashboard["battle_plan"]:
                result.dashboard["battle_plan"]["sniper_points"] = {}
            result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = placeholder


_CHIP_KEYS: tuple = ("profit_ratio", "avg_cost", "concentration", "chip_health")


def _is_value_placeholder(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (int, float)) and v == 0:
        return True
    s = str(v).strip().lower()
    return s in ("", "n/a", "na", "数据缺失", "未知", "data unavailable", "unknown", "tbd")


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _derive_chip_health(profit_ratio: float, concentration_90: float, language: str = "zh") -> str:
    if profit_ratio >= 0.9:
        return localize_chip_health("警惕", language)
    if concentration_90 >= 0.25:
        return localize_chip_health("警惕", language)
    if concentration_90 < 0.15 and 0.3 <= profit_ratio < 0.9:
        return localize_chip_health("健康", language)
    return localize_chip_health("一般", language)


def _build_chip_structure_from_data(chip_data: Any, language: str = "zh") -> Dict[str, Any]:
    if hasattr(chip_data, "profit_ratio"):
        pr = _safe_float(chip_data.profit_ratio)
        ac = chip_data.avg_cost
        c90 = _safe_float(chip_data.concentration_90)
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        pr = _safe_float(d.get("profit_ratio"))
        ac = d.get("avg_cost")
        c90 = _safe_float(d.get("concentration_90"))
    chip_health = _derive_chip_health(pr, c90, language=language)
    return {
        "profit_ratio": f"{pr:.1%}",
        "avg_cost": ac if (ac is not None and _safe_float(ac) != 0.0) else "N/A",
        "concentration": f"{c90:.2%}",
        "chip_health": chip_health,
    }


def fill_chip_structure_if_needed(result: "AnalysisResult", chip_data: Any) -> None:
    if not result or not chip_data:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        cs = dp.get("chip_structure") or {}
        filled = _build_chip_structure_from_data(
            chip_data,
            language=getattr(result, "report_language", "zh"),
        )
        merged = dict(cs)
        for k in _CHIP_KEYS:
            if _is_value_placeholder(merged.get(k)):
                merged[k] = filled[k]
        if merged != cs:
            dp["chip_structure"] = merged
            logger.info("[chip_structure] Filled placeholder chip fields from data source")
    except Exception as e:
        logger.warning("[chip_structure] Fill failed, skipping: %s", e)


_PRICE_POS_KEYS = ("ma5", "ma10", "ma20", "bias_ma5", "bias_status", "current_price", "support_level", "resistance_level")


def fill_price_position_if_needed(
    result: "AnalysisResult",
    trend_result: Any = None,
    realtime_quote: Any = None,
) -> None:
    if not result:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        pp = dp.get("price_position") or {}
        computed: Dict[str, Any] = {}
        if trend_result:
            tr = trend_result if isinstance(trend_result, dict) else (
                trend_result.__dict__ if hasattr(trend_result, "__dict__") else {}
            )
            computed["ma5"] = tr.get("ma5")
            computed["ma10"] = tr.get("ma10")
            computed["ma20"] = tr.get("ma20")
            computed["bias_ma5"] = tr.get("bias_ma5")
            computed["current_price"] = tr.get("current_price")
            support_levels = tr.get("support_levels") or []
            resistance_levels = tr.get("resistance_levels") or []
            if support_levels:
                computed["support_level"] = support_levels[0]
            if resistance_levels:
                computed["resistance_level"] = resistance_levels[0]
        if realtime_quote:
            rq = realtime_quote if isinstance(realtime_quote, dict) else (
                realtime_quote.to_dict() if hasattr(realtime_quote, "to_dict") else {}
            )
            if _is_value_placeholder(computed.get("current_price")):
                computed["current_price"] = rq.get("price")
        filled = False
        for k in _PRICE_POS_KEYS:
            if _is_value_placeholder(pp.get(k)) and not _is_value_placeholder(computed.get(k)):
                pp[k] = computed[k]
                filled = True
        if filled:
            dp["price_position"] = pp
            logger.info("[price_position] Filled placeholder fields from computed data")
    except Exception as e:
        logger.warning("[price_position] Fill failed, skipping: %s", e)


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager=None
) -> str:
    if context:
        if context.get('stock_name'):
            name = context['stock_name']
            if name and not name.startswith('股票'):
                return name
        if 'realtime' in context and context['realtime'].get('name'):
            return context['realtime']['name']
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as e:
            logger.debug(f"Cannot init DataFetcherManager: {e}")
    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as e:
            logger.debug(f"Failed to get stock name from data source: {e}")
    return f'Stock {stock_code}'


@dataclass
class AnalysisResult:
    code: str
    name: str
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "Medium"
    report_language: str = "en"
    dashboard: Optional[Dict[str, Any]] = None
    trend_analysis: str = ""
    short_term_outlook: str = ""
    medium_term_outlook: str = ""
    technical_analysis: str = ""
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""
    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None
    current_price: Optional[float] = None
    change_pct: Optional[float] = None
    model_used: Optional[str] = None
    query_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'sentiment_score': self.sentiment_score,
            'trend_prediction': self.trend_prediction,
            'operation_advice': self.operation_advice,
            'decision_type': self.decision_type,
            'confidence_level': self.confidence_level,
            'report_language': self.report_language,
            'dashboard': self.dashboard,
            'trend_analysis': self.trend_analysis,
            'short_term_outlook': self.short_term_outlook,
            'medium_term_outlook': self.medium_term_outlook,
            'technical_analysis': self.technical_analysis,
            'ma_analysis': self.ma_analysis,
            'volume_analysis': self.volume_analysis,
            'pattern_analysis': self.pattern_analysis,
            'fundamental_analysis': self.fundamental_analysis,
            'sector_position': self.sector_position,
            'company_highlights': self.company_highlights,
            'news_summary': self.news_summary,
            'market_sentiment': self.market_sentiment,
            'hot_topics': self.hot_topics,
            'analysis_summary': self.analysis_summary,
            'key_points': self.key_points,
            'risk_warning': self.risk_warning,
            'buy_reason': self.buy_reason,
            'market_snapshot': self.market_snapshot,
            'search_performed': self.search_performed,
            'success': self.success,
            'error_message': self.error_message,
            'current_price': self.current_price,
            'change_pct': self.change_pct,
            'model_used': self.model_used,
        }

    def get_core_conclusion(self) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            return self.dashboard['core_conclusion'].get('one_sentence', self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = False) -> str:
        if self.dashboard and 'core_conclusion' in self.dashboard:
            pos_advice = self.dashboard['core_conclusion'].get('position_advice', {})
            if has_position:
                return pos_advice.get('has_position', self.operation_advice)
            return pos_advice.get('no_position', self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('sniper_points', {})
        return {}

    def get_checklist(self) -> List[str]:
        if self.dashboard and 'battle_plan' in self.dashboard:
            return self.dashboard['battle_plan'].get('action_checklist', [])
        return []

    def get_risk_alerts(self) -> List[str]:
        if self.dashboard and 'intelligence' in self.dashboard:
            return self.dashboard['intelligence'].get('risk_alerts', [])
        return []

    def get_emoji(self) -> str:
        _, emoji, _ = get_signal_level(
            self.operation_advice,
            self.sentiment_score,
            self.report_language,
        )
        return emoji

    def get_confidence_stars(self) -> str:
        star_map = {
            "高": "⭐⭐⭐", "high": "⭐⭐⭐",
            "中": "⭐⭐", "medium": "⭐⭐",
            "低": "⭐", "low": "⭐",
        }
        return star_map.get(str(self.confidence_level or "").strip().lower(), "⭐⭐")


# ============================================================
# SYSTEM_PROMPT — fully English, with English checklist items
# Placeholders {market_placeholder} and {guidelines_placeholder}
# are filled at runtime by _get_analysis_system_prompt()
# ============================================================
_SYSTEM_PROMPT_LINES = [
    "You are a professional trend-trading investment analyst covering {market_placeholder}.",
    "",
    "{guidelines_placeholder}",
    "",
    "## Trading Philosophy",
    "- Strict entry: Do not chase highs. Do not buy if bias rate > 5%.",
    "- Trend trading: Only trade bullish alignment MA5 > MA10 > MA20.",
    "- Efficiency first: Focus on stocks with concentrated chip distribution.",
    "- Entry preference: Buy on low-volume pullbacks to MA5/MA10 support.",
    "",
    "## Output Format: Decision Dashboard JSON",
    "",
    "Output ONLY valid JSON in the exact structure below:",
    "",
    '{',
    '    "stock_name": "Company name",',
    '    "sentiment_score": <integer 0-100>,',
    '    "trend_prediction": "Strong Bullish / Bullish / Sideways / Bearish / Strong Bearish",',
    '    "operation_advice": "Strong Buy / Buy / Hold / Reduce / Sell / Watch",',
    '    "decision_type": "buy / hold / sell",',
    '    "confidence_level": "High / Medium / Low",',
    '',
    '    "dashboard": {',
    '        "core_conclusion": {',
    '            "one_sentence": "One-line decision (max 20 words)",',
    '            "signal_type": "Buy Signal / Hold & Watch / Sell Signal / Risk Warning",',
    '            "time_sensitivity": "Act Now / Today / This Week / No Rush",',
    '            "position_advice": {',
    '                "no_position": "Advice for those not holding",',
    '                "has_position": "Advice for existing holders"',
    '            }',
    '        },',
    '',
    '        "data_perspective": {',
    '            "trend_status": {',
    '                "ma_alignment": "Description of MA alignment",',
    '                "is_bullish": true,',
    '                "trend_score": <0-100>',
    '            },',
    '            "price_position": {',
    '                "current_price": <number>,',
    '                "ma5": <number>,',
    '                "ma10": <number>,',
    '                "ma20": <number>,',
    '                "bias_ma5": <percent number>,',
    '                "bias_status": "Safe / Caution / Danger",',
    '                "support_level": <number>,',
    '                "resistance_level": <number>',
    '            },',
    '            "volume_analysis": {',
    '                "volume_ratio": <number>,',
    '                "volume_status": "High Volume / Low Volume / Normal",',
    '                "turnover_rate": <percent number>,',
    '                "volume_meaning": "Interpretation of volume action"',
    '            },',
    '            "chip_structure": {',
    '                "profit_ratio": <percent>,',
    '                "avg_cost": <number>,',
    '                "concentration": <percent>,',
    '                "chip_health": "Healthy / Average / Caution"',
    '            }',
    '        },',
    '',
    '        "intelligence": {',
    '            "latest_news": "Recent important news summary",',
    '            "risk_alerts": ["Risk 1: description", "Risk 2: description"],',
    '            "positive_catalysts": ["Catalyst 1: description", "Catalyst 2: description"],',
    '            "earnings_outlook": "Earnings expectation based on announcements",',
    '            "sentiment_summary": "One-sentence market sentiment summary"',
    '        },',
    '',
    '        "battle_plan": {',
    '            "sniper_points": {',
    '                "ideal_buy": "Ideal entry: $XX.XX (near MA5)",',
    '                "secondary_buy": "Secondary entry: $XX.XX (near MA10)",',
    '                "stop_loss": "Stop loss: $XX.XX (break below MA20)",',
    '                "take_profit": "Target: $XX.XX (prior high / key level)"',
    '            },',
    '            "position_strategy": {',
    '                "suggested_position": "Suggested position size: X/10",',
    '                "entry_plan": "Staged entry strategy description",',
    '                "risk_control": "Risk control strategy description"',
    '            },',
    '            "action_checklist": [',
    '                "Check 1: Bullish alignment (MA5 > MA10 > MA20) — result",',
    '                "Check 2: Bias rate reasonable (< 5%) — result",',
    '                "Check 3: Volume confirmation — result",',
    '                "Check 4: No major negative news — result",',
    '                "Check 5: Chip structure healthy — result",',
    '                "Check 6: Valuation reasonable (P/E) — result"',
    '            ]',
    '        }',
    '    },',
    '',
    '    "analysis_summary": "100-word comprehensive analysis",',
    '    "key_points": "3-5 key points, comma separated",',
    '    "risk_warning": "Risk warning",',
    '    "buy_reason": "Reasoning referencing trading philosophy",',
    '    "trend_analysis": "Price action analysis",',
    '    "short_term_outlook": "1-3 day outlook",',
    '    "medium_term_outlook": "1-2 week outlook",',
    '    "technical_analysis": "Technical indicators summary",',
    '    "ma_analysis": "Moving average analysis",',
    '    "volume_analysis": "Volume analysis",',
    '    "pattern_analysis": "Candlestick pattern analysis",',
    '    "fundamental_analysis": "Fundamental analysis",',
    '    "sector_position": "Sector and industry analysis",',
    '    "company_highlights": "Company highlights / risks",',
    '    "news_summary": "News summary",',
    '    "market_sentiment": "Market sentiment",',
    '    "hot_topics": "Related hot topics",',
    '    "search_performed": true,',
    '    "data_sources": "Data sources description"',
    '}',
    '',
    "## Scoring Criteria",
    "",
    "Strong Buy (80-100): MA bullish alignment + bias < 2% + volume confirmation + healthy chips + positive catalyst",
    "Buy (60-79): Bullish or weak bullish + bias < 5% + normal volume + one minor condition missing",
    "Watch (40-59): Bias > 5% (chasing risk) OR unclear trend OR risk event",
    "Sell/Reduce (0-39): Bearish alignment OR break below MA20 OR heavy sell-off OR major negative news",
    "",
    "## Core Principles",
    "",
    "1. Lead with the conclusion: one sentence tells the user what to do.",
    "2. Separate advice for holders vs non-holders.",
    "3. Give specific price levels — no vague statements.",
    "4. Checklist items must use English labels exactly as shown above.",
    "5. Highlight risk alerts prominently.",
]

_SYSTEM_PROMPT_EN = "\n".join(_SYSTEM_PROMPT_LINES)


class GeminiAnalyzer:
    """Gemini AI 分析器"""

    SYSTEM_PROMPT = _SYSTEM_PROMPT_EN

    def __init__(self, api_key: Optional[str] = None):
        self._router = None
        self._litellm_available = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys), AI analysis will be unavailable")

    def _get_analysis_system_prompt(self, report_language: str, stock_code: str = "") -> str:
        """Build the analyzer system prompt with output-language guidance."""
        lang = normalize_report_language(report_language)
        market_role = get_market_role(stock_code, lang)
        market_guidelines = get_market_guidelines(stock_code, lang)
        base_prompt = self.SYSTEM_PROMPT.replace(
            "{market_placeholder}", market_role
        ).replace(
            "{guidelines_placeholder}", market_guidelines
        )
        if lang == "en":
            return base_prompt + (
                "\n\n## Output Language (highest priority)\n"
                "- Keep all JSON keys unchanged.\n"
                "- decision_type must remain buy|hold|sell.\n"
                "- ALL human-readable values must be in English.\n"
                "- Checklist items must use English labels: 'Check 1: Bullish alignment', etc.\n"
                "- Do not use any Chinese characters anywhere in your response.\n"
            )
        return base_prompt + (
            "\n\n## 输出语言（最高优先级）\n"
            "- 所有 JSON 键名保持不变。\n"
            "- decision_type 必须保持为 buy|hold|sell。\n"
            "- 所有面向用户的人类可读文本值必须使用中文。\n"
        )

    def _has_channel_config(self, config: Config) -> bool:
        return bool(config.llm_model_list) and not all(
            e.get('model_name', '').startswith('__legacy_') for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return
        self._litellm_available = True
        if self._has_channel_config(config):
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique_models = list(dict.fromkeys(
                e['litellm_params']['model'] for e in model_list
            ))
            logger.info(f"Analyzer LLM: Router initialized — {len(model_list)} deployment(s), models: {unique_models}")
            return
        keys = get_api_keys_for_model(litellm_model, config)
        if len(keys) > 1:
            extra_params = extra_litellm_params(litellm_model, config)
            legacy_model_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {"model": litellm_model, "api_key": k, **extra_params},
                }
                for k in keys
            ]
            self._router = Router(
                model_list=legacy_model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            logger.info(f"Analyzer LLM: Legacy Router initialized with {len(keys)} keys for {litellm_model}")
        elif keys:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model})")
        else:
            logger.info(f"Analyzer LLM: litellm initialized (model={litellm_model}, API key from environment)")

    def is_available(self) -> bool:
        return self._router is not None or self._litellm_available

    def _call_litellm(
        self,
        prompt: str,
        generation_config: dict,
        *,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        config = get_config()
        max_tokens = (
            generation_config.get('max_output_tokens')
            or generation_config.get('max_tokens')
            or 8192
        )
        temperature = generation_config.get('temperature', 0.7)
        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]
        use_channel_router = self._has_channel_config(config)
        last_error = None
        effective_system_prompt = system_prompt or self.SYSTEM_PROMPT
        for model in models_to_try:
            try:
                model_short = model.split("/")[-1] if "/" in model else model
                call_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": effective_system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                extra = get_thinking_extra_body(model_short)
                if extra:
                    call_kwargs["extra_body"] = extra
                _router_model_names = set(get_configured_llm_models(config.llm_model_list))
                if use_channel_router and self._router and model in _router_model_names:
                    response = self._router.completion(**call_kwargs)
                elif self._router and model == config.litellm_model and not use_channel_router:
                    response = self._router.completion(**call_kwargs)
                else:
                    keys = get_api_keys_for_model(model, config)
                    if keys:
                        call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)
                if response and response.choices and response.choices[0].message.content:
                    usage: Dict[str, Any] = {}
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens or 0,
                            "completion_tokens": response.usage.completion_tokens or 0,
                            "total_tokens": response.usage.total_tokens or 0,
                        }
                    return (response.choices[0].message.content, model, usage)
                raise ValueError("LLM returned empty response")
            except Exception as e:
                logger.warning(f"[LiteLLM] {model} failed: {e}")
                last_error = e
                continue
        raise Exception(f"All LLM models failed (tried {len(models_to_try)} model(s)). Last error: {last_error}")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> Optional[str]:
        try:
            result = self._call_litellm(
                prompt,
                generation_config={"max_tokens": max_tokens, "temperature": temperature},
                system_prompt=system_prompt,
            )
            if isinstance(result, tuple):
                text, model_used, usage = result
                persist_llm_usage(usage, model_used, call_type="market_review")
                return text
            return result
        except Exception as exc:
            logger.error("[generate_text] LLM call failed: %s", exc)
            return None

    def analyze(
        self,
        context: Dict[str, Any],
        news_context: Optional[str] = None
    ) -> AnalysisResult:
        code = context.get('code', 'Unknown')
        report_language = normalize_report_language(_resolve_report_language())
        system_prompt = self._get_analysis_system_prompt(report_language, stock_code=code)
        config = get_config()
        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug(f"[LLM] Waiting {request_delay:.1f}s before request...")
            time.sleep(request_delay)
        name = context.get('stock_name')
        if not name or name.startswith('股票'):
            if 'realtime' in context and context['realtime'].get('name'):
                name = context['realtime']['name']
            else:
                name = STOCK_NAME_MAP.get(code, f'Stock {code}')
        if not self.is_available():
            return AnalysisResult(
                code=code, name=name, sentiment_score=50,
                trend_prediction='Sideways', operation_advice='Hold',
                confidence_level='Low',
                analysis_summary='AI analysis unavailable — no API key configured.',
                risk_warning='Configure an LLM API key and retry.',
                success=False, error_message='LLM API key not configured',
                model_used=None, report_language=report_language,
            )
        try:
            prompt = self._format_prompt(context, name, news_context, report_language=report_language)
            config = get_config()
            model_name = config.litellm_model or "unknown"
            logger.info(f"========== AI Analysis {name}({code}) | lang={report_language} ==========")
            logger.info(f"[LLM] Model: {model_name} | Prompt length: {len(prompt)} chars")
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            logger.info(f"[LLM Prompt preview]\n{prompt_preview}")
            generation_config = {
                "temperature": config.llm_temperature,
                "max_output_tokens": 8192,
            }
            current_prompt = prompt
            retry_count = 0
            max_retries = config.report_integrity_retry if config.report_integrity_enabled else 0
            while True:
                start_time = time.time()
                response_text, model_used, llm_usage = self._call_litellm(
                    current_prompt, generation_config, system_prompt=system_prompt,
                )
                elapsed = time.time() - start_time
                logger.info(f"[LLM] Response OK, {elapsed:.2f}s, {len(response_text)} chars")
                result = self._parse_response(response_text, code, name)
                result.raw_response = response_text
                result.search_performed = bool(news_context)
                result.market_snapshot = self._build_market_snapshot(context)
                result.model_used = model_used
                result.report_language = report_language
                if not config.report_integrity_enabled:
                    break
                pass_integrity, missing_fields = self._check_content_integrity(result)
                if pass_integrity:
                    break
                if retry_count < max_retries:
                    current_prompt = self._build_integrity_retry_prompt(
                        prompt, response_text, missing_fields, report_language=report_language,
                    )
                    retry_count += 1
                    logger.info("[LLM integrity] Missing %s, retry %d", missing_fields, retry_count)
                else:
                    self._apply_placeholder_fill(result, missing_fields)
                    logger.warning("[LLM integrity] Missing %s, filled with placeholders", missing_fields)
                    break
            persist_llm_usage(llm_usage, model_used, call_type="analysis", stock_code=code)
            logger.info(f"[LLM] {name}({code}) done: {result.trend_prediction}, score {result.sentiment_score}")
            return result
        except Exception as e:
            logger.error(f"AI analysis {name}({code}) failed: {e}")
            return AnalysisResult(
                code=code, name=name, sentiment_score=50,
                trend_prediction='Sideways', operation_advice='Hold',
                confidence_level='Low',
                analysis_summary=f'Analysis failed: {str(e)[:100]}',
                risk_warning='Analysis failed. Please retry later.',
                success=False, error_message=str(e),
                model_used=None, report_language=report_language,
            )

    def _format_prompt(
        self,
        context: Dict[str, Any],
        name: str,
        news_context: Optional[str] = None,
        report_language: str = "en",
    ) -> str:
        code = context.get('code', 'Unknown')
        report_language = normalize_report_language(report_language)
        stock_name = context.get('stock_name', name)
        if not stock_name or stock_name == f'股票{code}':
            stock_name = STOCK_NAME_MAP.get(code, f'Stock {code}')
        today = context.get('today', {})
        unknown_text = get_unknown_text(report_language)
        no_data_text = get_no_data_text(report_language)

        prompt = (
            f"# Decision Dashboard Analysis Request\n\n"
            f"## Stock Info\n"
            f"| Field | Value |\n"
            f"|-------|-------|\n"
            f"| Code | **{code}** |\n"
            f"| Name | **{stock_name}** |\n"
            f"| Date | {context.get('date', unknown_text)} |\n\n"
            f"---\n\n"
            f"## Technical Data\n\n"
            f"### Today's Price Action\n"
            f"| Indicator | Value |\n"
            f"|-----------|-------|\n"
            f"| Close | {today.get('close', 'N/A')} |\n"
            f"| Open | {today.get('open', 'N/A')} |\n"
            f"| High | {today.get('high', 'N/A')} |\n"
            f"| Low | {today.get('low', 'N/A')} |\n"
            f"| Change % | {today.get('pct_chg', 'N/A')}% |\n"
            f"| Volume | {self._format_volume(today.get('volume'))} |\n"
            f"| Turnover | {self._format_amount(today.get('amount'))} |\n\n"
            f"### Moving Averages\n"
            f"| MA | Value |\n"
            f"|----|-------|\n"
            f"| MA5 | {today.get('ma5', 'N/A')} |\n"
            f"| MA10 | {today.get('ma10', 'N/A')} |\n"
            f"| MA20 | {today.get('ma20', 'N/A')} |\n"
            f"| MA Status | {context.get('ma_status', unknown_text)} |\n"
        )

        if 'realtime' in context:
            rt = context['realtime']
            prompt += (
                f"\n### Realtime Data\n"
                f"| Indicator | Value |\n"
                f"|-----------|-------|\n"
                f"| Price | {rt.get('price', 'N/A')} |\n"
                f"| Volume Ratio | {rt.get('volume_ratio', 'N/A')} |\n"
                f"| Turnover Rate | {rt.get('turnover_rate', 'N/A')}% |\n"
                f"| P/E | {rt.get('pe_ratio', 'N/A')} |\n"
                f"| P/B | {rt.get('pb_ratio', 'N/A')} |\n"
                f"| Market Cap | {self._format_amount(rt.get('total_mv'))} |\n"
                f"| 60d Change | {rt.get('change_60d', 'N/A')}% |\n"
            )

        fundamental_context = context.get("fundamental_context") if isinstance(context, dict) else None
        earnings_block = (fundamental_context.get("earnings", {}) if isinstance(fundamental_context, dict) else {})
        earnings_data = (earnings_block.get("data", {}) if isinstance(earnings_block, dict) else {})
        financial_report = (earnings_data.get("financial_report", {}) if isinstance(earnings_data, dict) else {})
        dividend_metrics = (earnings_data.get("dividend", {}) if isinstance(earnings_data, dict) else {})
        if isinstance(financial_report, dict) or isinstance(dividend_metrics, dict):
            financial_report = financial_report if isinstance(financial_report, dict) else {}
            dividend_metrics = dividend_metrics if isinstance(dividend_metrics, dict) else {}
            prompt += (
                f"\n### Financials & Dividends\n"
                f"| Field | Value |\n"
                f"|-------|-------|\n"
                f"| Report Period | {financial_report.get('report_date', 'N/A')} |\n"
                f"| Revenue | {financial_report.get('revenue', 'N/A')} |\n"
                f"| Net Profit | {financial_report.get('net_profit_parent', 'N/A')} |\n"
                f"| Operating CF | {financial_report.get('operating_cash_flow', 'N/A')} |\n"
                f"| ROE | {financial_report.get('roe', 'N/A')} |\n"
                f"| TTM Dividend/Share | {dividend_metrics.get('ttm_cash_dividend_per_share', 'N/A')} |\n"
                f"| TTM Yield | {dividend_metrics.get('ttm_dividend_yield_pct', 'N/A')} |\n"
            )

        if 'chip' in context:
            chip = context['chip']
            profit_ratio = chip.get('profit_ratio', 0)
            prompt += (
                f"\n### Chip Distribution\n"
                f"| Indicator | Value |\n"
                f"|-----------|-------|\n"
                f"| Profit Ratio | {profit_ratio:.1%} |\n"
                f"| Avg Cost | {chip.get('avg_cost', 'N/A')} |\n"
                f"| Concentration 90% | {chip.get('concentration_90', 0):.2%} |\n"
                f"| Concentration 70% | {chip.get('concentration_70', 0):.2%} |\n"
                f"| Chip Status | {chip.get('chip_status', unknown_text)} |\n"
            )

        if 'trend_analysis' in context:
            trend = context['trend_analysis']
            bias = trend.get('bias_ma5', 0)
            bias_warning = "DANGER: > 5%, do not chase!" if bias > 5 else "Safe range"
            prompt += (
                f"\n### Trend Analysis\n"
                f"| Indicator | Value | Note |\n"
                f"|-----------|-------|------|\n"
                f"| Trend Status | {trend.get('trend_status', unknown_text)} | |\n"
                f"| MA Alignment | {trend.get('ma_alignment', unknown_text)} | |\n"
                f"| Trend Score | {trend.get('trend_strength', 0)}/100 | |\n"
                f"| Bias MA5 | {bias:+.2f}% | {bias_warning} |\n"
                f"| Bias MA10 | {trend.get('bias_ma10', 0):+.2f}% | |\n"
                f"| Volume Status | {trend.get('volume_status', unknown_text)} | |\n"
                f"| System Signal | {trend.get('buy_signal', unknown_text)} | |\n"
                f"| Signal Score | {trend.get('signal_score', 0)}/100 | |\n"
            )

        if 'yesterday' in context:
            prompt += (
                f"\n### Volume & Price Change\n"
                f"- Volume vs yesterday: {context.get('volume_change_ratio', 'N/A')}x\n"
                f"- Price vs yesterday: {context.get('price_change_ratio', 'N/A')}%\n"
            )

        news_window_days: Optional[int] = None
        context_window = context.get("news_window_days")
        try:
            if context_window is not None:
                parsed_window = int(context_window)
                if parsed_window > 0:
                    news_window_days = parsed_window
        except (TypeError, ValueError):
            news_window_days = None
        if news_window_days is None:
            prompt_config = get_config()
            news_window_days = resolve_news_window_days(
                news_max_age_days=getattr(prompt_config, "news_max_age_days", 3),
                news_strategy_profile=getattr(prompt_config, "news_strategy_profile", "short"),
            )

        prompt += "\n\n---\n\n## News & Sentiment\n"
        if news_context:
            prompt += (
                f"\nRecent {news_window_days}-day news for **{stock_name}({code})**:\n\n"
                f"```\n{news_context}\n```\n"
            )
        else:
            prompt += "\nNo recent news found. Base analysis on technical data only.\n"

        if context.get('data_missing'):
            prompt += (
                "\n**DATA WARNING**: Real-time data unavailable. "
                "Ignore N/A fields. Do not invent data. "
                "State 'Data unavailable' where needed.\n"
            )

        prompt += f"\n\n---\n\n## Task\n\nGenerate the Decision Dashboard JSON for **{stock_name}({code})**.\n"

        if context.get('is_index_etf'):
            prompt += (
                "\n> ETF/INDEX CONSTRAINT: This is an index-tracking ETF or market index.\n"
                "> Risk analysis: focus on index trend, tracking error, market liquidity only.\n"
                "> Do NOT include fund company litigation, reputation, or management changes in risk_alerts.\n"
            )

        prompt += (
            "\n\n### Output Requirements (highest priority)\n"
            "- Output valid JSON only — no markdown fences, no commentary.\n"
            "- decision_type must be exactly: buy, hold, or sell.\n"
            "- ALL text values must be in English — no Chinese characters.\n"
            "- Checklist items must follow this exact English format:\n"
            "  'Check 1: Bullish alignment (MA5 > MA10 > MA20) — [result]'\n"
            "  'Check 2: Bias rate reasonable (< 5%) — [result]'\n"
            "  'Check 3: Volume confirmation — [result]'\n"
            "  'Check 4: No major negative news — [result]'\n"
            "  'Check 5: Chip structure healthy — [result]'\n"
            "  'Check 6: Valuation reasonable (P/E) — [result]'\n"
            "- Give specific price levels for all sniper points.\n"
            "- News items in risk_alerts/positive_catalysts must include dates (YYYY-MM-DD).\n"
            f"- Ignore any news older than {news_window_days} days.\n"
        )

        return prompt

    def _format_volume(self, volume: Optional[float]) -> str:
        if volume is None:
            return 'N/A'
        if volume >= 1e8:
            return f"{volume / 1e8:.2f}B shares"
        elif volume >= 1e4:
            return f"{volume / 1e4:.2f}M shares"
        else:
            return f"{volume:.0f} shares"

    def _format_amount(self, amount: Optional[float]) -> str:
        if amount is None:
            return 'N/A'
        if amount >= 1e8:
            return f"{amount / 1e8:.2f}B"
        elif amount >= 1e4:
            return f"{amount / 1e4:.2f}M"
        else:
            return f"{amount:.0f}"

    def _format_percent(self, value: Optional[float]) -> str:
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return 'N/A'

    def _format_price(self, value: Optional[float]) -> str:
        if value is None:
            return 'N/A'
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return 'N/A'

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        today = context.get('today', {}) or {}
        realtime = context.get('realtime', {}) or {}
        yesterday = context.get('yesterday', {}) or {}
        prev_close = yesterday.get('close')
        close = today.get('close')
        high = today.get('high')
        low = today.get('low')
        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                amplitude = None
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                change_amount = None
        snapshot = {
            "date": context.get('date', 'Unknown'),
            "close": self._format_price(close),
            "open": self._format_price(today.get('open')),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get('pct_chg')),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get('volume')),
            "amount": self._format_amount(today.get('amount')),
        }
        if realtime:
            snapshot.update({
                "price": self._format_price(realtime.get('price')),
                "volume_ratio": realtime.get('volume_ratio', 'N/A'),
                "turnover_rate": self._format_percent(realtime.get('turnover_rate')),
                "source": getattr(realtime.get('source'), 'value', realtime.get('source', 'N/A')),
            })
        return snapshot

    def _check_content_integrity(self, result: AnalysisResult) -> Tuple[bool, List[str]]:
        return check_content_integrity(result)

    def _build_integrity_complement_prompt(self, missing_fields: List[str], report_language: str = "en") -> str:
        report_language = normalize_report_language(report_language)
        lines = ["### Completion requirements: fill the missing mandatory fields and output the full JSON again:"]
        for f in missing_fields:
            if f == "sentiment_score":
                lines.append("- sentiment_score: integer 0-100")
            elif f == "operation_advice":
                lines.append("- operation_advice: action advice in English")
            elif f == "analysis_summary":
                lines.append("- analysis_summary: concise analysis summary")
            elif f == "dashboard.core_conclusion.one_sentence":
                lines.append("- dashboard.core_conclusion.one_sentence: one-line decision")
            elif f == "dashboard.intelligence.risk_alerts":
                lines.append("- dashboard.intelligence.risk_alerts: risk alert list (can be empty array)")
            elif f == "dashboard.battle_plan.sniper_points.stop_loss":
                lines.append("- dashboard.battle_plan.sniper_points.stop_loss: stop-loss price level")
        return "\n".join(lines)

    def _build_integrity_retry_prompt(
        self,
        base_prompt: str,
        previous_response: str,
        missing_fields: List[str],
        report_language: str = "en",
    ) -> str:
        complement = self._build_integrity_complement_prompt(missing_fields, report_language=report_language)
        previous_output = previous_response.strip()
        prefix = "### Previous output below. Complete missing fields and return the full JSON. Do not omit existing fields:"
        return "\n\n".join([base_prompt, prefix, previous_output, complement])

    def _apply_placeholder_fill(self, result: AnalysisResult, missing_fields: List[str]) -> None:
        apply_placeholder_fill(result, missing_fields)

    def _parse_response(
        self,
        response_text: str,
        code: str,
        name: str
    ) -> AnalysisResult:
        try:
            report_language = normalize_report_language(_resolve_report_language())
            cleaned_text = response_text
            if '```json' in cleaned_text:
                cleaned_text = cleaned_text.replace('```json', '').replace('```', '')
            elif '```' in cleaned_text:
                cleaned_text = cleaned_text.replace('```', '')
            json_start = cleaned_text.find('{')
            json_end = cleaned_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = cleaned_text[json_start:json_end]
                json_str = self._fix_json_string(json_str)
                data = json.loads(json_str)
                try:
                    AnalysisReportSchema.model_validate(data)
                except Exception as e:
                    logger.warning("LLM report schema validation failed: %s", str(e)[:100])
                dashboard = data.get('dashboard', None)
                ai_stock_name = data.get('stock_name')
                if ai_stock_name and (name.startswith('股票') or name == code or 'Unknown' in name or name.startswith('Stock ')):
                    name = ai_stock_name
                decision_type = data.get('decision_type', '')
                if not decision_type:
                    op = data.get('operation_advice', 'Hold')
                    decision_type = infer_decision_type_from_advice(op, default='hold')
                return AnalysisResult(
                    code=code, name=name,
                    sentiment_score=int(data.get('sentiment_score', 50)),
                    trend_prediction=data.get('trend_prediction', 'Sideways'),
                    operation_advice=data.get('operation_advice', 'Hold'),
                    decision_type=decision_type,
                    confidence_level=localize_confidence_level(
                        data.get('confidence_level', 'Medium'), report_language,
                    ),
                    report_language=report_language,
                    dashboard=dashboard,
                    trend_analysis=data.get('trend_analysis', ''),
                    short_term_outlook=data.get('short_term_outlook', ''),
                    medium_term_outlook=data.get('medium_term_outlook', ''),
                    technical_analysis=data.get('technical_analysis', ''),
                    ma_analysis=data.get('ma_analysis', ''),
                    volume_analysis=data.get('volume_analysis', ''),
                    pattern_analysis=data.get('pattern_analysis', ''),
                    fundamental_analysis=data.get('fundamental_analysis', ''),
                    sector_position=data.get('sector_position', ''),
                    company_highlights=data.get('company_highlights', ''),
                    news_summary=data.get('news_summary', ''),
                    market_sentiment=data.get('market_sentiment', ''),
                    hot_topics=data.get('hot_topics', ''),
                    analysis_summary=data.get('analysis_summary', 'Analysis completed'),
                    key_points=data.get('key_points', ''),
                    risk_warning=data.get('risk_warning', ''),
                    buy_reason=data.get('buy_reason', ''),
                    search_performed=data.get('search_performed', False),
                    data_sources=data.get('data_sources', 'Technical data'),
                    success=True,
                )
            else:
                logger.warning("Could not extract JSON from response, using text fallback")
                return self._parse_text_response(response_text, code, name)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode failed: {e}, using text fallback")
            return self._parse_text_response(response_text, code, name)

    def _fix_json_string(self, json_str: str) -> str:
        import re
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = json_str.replace('True', 'true').replace('False', 'false')
        json_str = repair_json(json_str)
        return json_str

    def _parse_text_response(self, response_text: str, code: str, name: str) -> AnalysisResult:
        report_language = normalize_report_language(_resolve_report_language())
        sentiment_score = 50
        trend = 'Sideways'
        advice = 'Hold'
        text_lower = response_text.lower()
        positive_keywords = ['bullish', 'buy', 'breakout', 'uptrend', 'strong']
        negative_keywords = ['bearish', 'sell', 'downtrend', 'break down', 'weak']
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
        if positive_count > negative_count + 1:
            sentiment_score = 65
            trend = 'Bullish'
            advice = 'Buy'
            decision_type = 'buy'
        elif negative_count > positive_count + 1:
            sentiment_score = 35
            trend = 'Bearish'
            advice = 'Sell'
            decision_type = 'sell'
        else:
            decision_type = 'hold'
        summary = response_text[:500] if response_text else 'No analysis result'
        return AnalysisResult(
            code=code, name=name,
            sentiment_score=sentiment_score, trend_prediction=trend,
            operation_advice=advice, decision_type=decision_type,
            confidence_level='Low',
            analysis_summary=summary,
            key_points='JSON parsing failed; treat as best-effort output.',
            risk_warning='Result may be inaccurate. Cross-check with other sources.',
            raw_response=response_text, success=True,
            report_language=report_language,
        )

    def batch_analyze(self, contexts: List[Dict[str, Any]], delay_between: float = 2.0) -> List[AnalysisResult]:
        results = []
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug(f"Waiting {delay_between}s before next analysis...")
                time.sleep(delay_between)
            result = self.analyze(context)
            results.append(result)
        return results


def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_context = {
        'code': '600519', 'date': '2026-01-09',
        'today': {
            'open': 1800.0, 'high': 1850.0, 'low': 1780.0,
            'close': 1820.0, 'volume': 10000000,
            'amount': 18200000000, 'pct_chg': 1.5,
            'ma5': 1810.0, 'ma10': 1800.0, 'ma20': 1790.0,
        },
        'ma_status': 'Bullish alignment',
    }
    analyzer = GeminiAnalyzer()
    if analyzer.is_available():
        result = analyzer.analyze(test_context)
        print(f"Result: {result.to_dict()}")
    else:
        print("No API key configured, skipping test")
