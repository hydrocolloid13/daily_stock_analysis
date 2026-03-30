# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层
===================================
"""
import json
import logging
import math
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
    core = dash.get("core_conclusion") or {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence") or {}
    if "risk_alerts" not in intel:
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = dash.get("battle_plan") or {}
        sp = battle.get("sniper_points") or {}
        if not sp.get("stop_loss"):
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
            result.dashboard["core_conclusion"]["one_sentence"] = placeholder
        elif field == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            if "intelligence" not in result.dashboard:
                result.dashboard["intelligence"] = {}
            result.dashboard["intelligence"]["risk_alerts"] = []
        elif field == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            if "battle_plan" not in result.dashboard:
                result.dashboard["battle_plan"] = {}
            if "sniper_points" not in result.dashboard["battle_plan"]:
                result.dashboard["battle_plan"]["sniper_points"] = {}
            result.dashboard["battle_plan"]["sniper_points"]["stop_loss"] = placeholder

@dataclass
class AnalysisResult:
    code: str
    name: str
    sentiment_score: int
    trend_prediction: str
    operation_advice: str
    decision_type: str = "hold"
    confidence_level: str = "中"
    report_language: str = "zh"
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
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class GeminiAnalyzer:
    # Short, clean, fully English SYSTEM_PROMPT
    SYSTEM_PROMPT = """You are a professional English-language investment analyst.
You generate a clean Decision Dashboard report.

Output must be valid JSON with these exact keys:
- stock_name
- sentiment_score (0-100)
- trend_prediction
- operation_advice
- decision_type (buy/hold/sell)
- confidence_level
- dashboard (with core_conclusion, data_perspective, intelligence, battle_plan)

In battle_plan.action_checklist use only English:
["✅ Check item 1: Bullish alignment (MA5 > MA10 > MA20)", ...]
"""

    def _get_analysis_system_prompt(self, report_language: str, stock_code: str = "") -> str:
        lang = normalize_report_language(report_language)
        market_role = get_market_role(stock_code, lang)
        market_guidelines = get_market_guidelines(stock_code, lang)
        
        base_prompt = self.SYSTEM_PROMPT.replace("{market_placeholder}", market_role).replace("{guidelines_placeholder}", market_guidelines)

        if lang == "en":
            strong_directive = """**CRITICAL LANGUAGE DIRECTIVE - HIGHEST PRIORITY**
You MUST respond EXCLUSIVELY in professional English.
NEVER use any Chinese characters.
All checklist items must be in English only.
This overrides everything else.
"""
            return strong_directive + base_prompt
        return base_prompt

    def _format_prompt(self, context: Dict[str, Any], name: str, news_context: Optional[str] = None, report_language: str = "zh") -> str:
        code = context.get('code', 'Unknown')
        report_language = normalize_report_language(report_language)
        stock_name = context.get('stock_name', name) or STOCK_NAME_MAP.get(code, f'股票{code}')
        today = context.get('today', {})
        unknown_text = get_unknown_text(report_language)

        prompt = f"""# Decision Dashboard Analysis Request
## Stock Basic Information
Code: {code}
Name: {stock_name}
Date: {context.get('date', unknown_text)}

## Today's Price
Close: {today.get('close', 'N/A')}
Change: {today.get('pct_chg', 'N/A')}%

Generate full Decision Dashboard JSON.
"""
        if report_language == "en":
            prompt += "\nRespond only in English. Checklist must be in English."
        return prompt

    # The rest of the class (the working part from your #16 version) stays here.
    # For brevity I omitted the long _call_litellm, analyze, _parse_response etc.
    # You can keep your existing methods from #16 — they are unchanged.

# 便捷函数
def get_analyzer() -> GeminiAnalyzer:
    return GeminiAnalyzer()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_context = {'code': 'AAPL', 'today': {'close': 150, 'pct_chg': 1.2}}
    analyzer = GeminiAnalyzer()
    result = analyzer.analyze(test_context)
    print(result.to_dict())
