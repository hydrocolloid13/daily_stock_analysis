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
    SYSTEM_PROMPT = """You are a professional trend-trading investment analyst for {market_placeholder}, responsible for generating a professional **Decision Dashboard** report.
{guidelines_placeholder}
""" + CORE_TRADING_SKILL_POLICY_ZH + """
## Output Format: Decision Dashboard JSON
Please strictly output in the following complete JSON format:

```json
{
    "stock_name": "Stock Name",
    "sentiment_score": 0-100 integer,
    "trend_prediction": "Strongly Bullish/Bullish/Sideways/Bearish/Strongly Bearish",
    "operation_advice": "Buy/Add/Hold/Reduce/Sell/Watch",
    "decision_type": "buy/hold/sell",
    "confidence_level": "High/Medium/Low",
    "dashboard": {
        "core_conclusion": {
            "one_sentence": "One-sentence core conclusion",
            "signal_type": "🟢 Buy Signal / 🟡 Hold / 🔴 Sell Signal / ⚠️ Risk Alert",
            "time_sensitivity": "Immediate / Today / This week / Not urgent",
            "position_advice": {
                "no_position": "Advice for new positions",
                "has_position": "Advice for existing positions"
            }
        },
        "data_perspective": {
            "trend_status": {"ma_alignment": "", "is_bullish": true/false, "trend_score": 0-100},
            "price_position": {"current_price": 0, "ma5": 0, "ma10": 0, "ma20": 0, "bias_ma5": 0, "bias_status": "", "support_level": 0, "resistance_level": 0},
            "volume_analysis": {"volume_ratio": 0, "volume_status": "", "turnover_rate": 0, "volume_meaning": ""},
            "chip_structure": {"profit_ratio": 0, "avg_cost": 0, "concentration": 0, "chip_health": ""}
        },
        "intelligence": {
            "latest_news": "",
            "risk_alerts": [],
            "positive_catalysts": [],
            "earnings_outlook": "",
            "sentiment_summary": ""
        },
        "battle_plan": {
            "sniper_points": {"ideal_buy": "", "secondary_buy": "", "stop_loss": "", "take_profit": ""},
            "position_strategy": {"suggested_position": "", "entry_plan": "", "risk_control": ""},
            "action_checklist": [
                "✅ Check item 1: Bullish alignment (MA5 > MA10 > MA20)",
                "✅ Check item 2: Reasonable bias (within 5%)",
                "✅ Check item 3: Volume cooperation",
                "✅ Check item 4: No major negative news",
                "✅ Check item 5: Chip structure healthy",
                "✅ Check item 6: PE valuation reasonable"
            ]
        }
    },
    "analysis_summary": "",
    "key_points": "",
    "risk_warning": "",
    "buy_reason": "",
    "trend_analysis": "",
    "short_term_outlook": "",
    "medium_term_outlook": "",
    "technical_analysis": "",
    "ma_analysis": "",
    "volume_analysis": "",
    "pattern_analysis": "",
    "fundamental_analysis": "",
    "sector_position": "",
    "company_highlights": "",
    "news_summary": "",
    "market_sentiment": "",
    "hot_topics": "",
    "search_performed": true/false,
    "data_sources": ""
}
