# -*- coding: utf-8 -*-
"""
===================================
A股自选股智能分析系统 - AI分析层 (English + Macro + Correlation)
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
import yfinance as yf   # Macro data

logger = logging.getLogger(__name__)

# ==================== HELPER FUNCTIONS (unchanged) ====================
# (All the original helper functions are kept exactly as before)

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

# ... (chip_structure, price_position, get_stock_name_multi_source, AnalysisResult class stay unchanged)

# ==================== MAIN ANALYZER CLASS ====================

class GeminiAnalyzer:
    SYSTEM_PROMPT = """你是一位专注于趋势交易的{market_placeholder}投资分析师，负责生成专业的【决策仪表盘】分析报告。
{guidelines_placeholder}
""" + CORE_TRADING_SKILL_POLICY_ZH + """
## 输出格式：决策仪表盘 JSON
请严格按照以下 JSON 格式输出，这是一个完整的【决策仪表盘】：
```json
{
    "stock_name": "股票名称",
    "sentiment_score": 0-100,
    "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
    "operation_advice": "买入/加仓/持有/减仓/卖出/观望",
    "decision_type": "buy/hold/sell",
    "confidence_level": "高/中/低",
    "dashboard": { ... }   # full dashboard structure as before
}
