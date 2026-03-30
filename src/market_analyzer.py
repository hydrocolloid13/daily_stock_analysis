# -*- coding: utf-8 -*-
"""
===================================
Global Market Recap - English Only
===================================
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
from src.config import get_config
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)

@dataclass
class MarketIndex:
    code: str
    name: str
    current: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    prev_close: float = 0.0
    volume: float = 0.0
    amount: float = 0.0
    amplitude: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class MarketOverview:
    date: str
    indices: List[MarketIndex] = field(default_factory=list)
    up_count: int = 0
    down_count: int = 0
    total_amount: float = 0.0

class MarketAnalyzer:
    def __init__(self):
        self.config = get_config()
        self.search_service = SearchService()
        self.data_fetcher = DataFetcherManager()

    def get_market_overview(self) -> MarketOverview:
        """Get US + ASX major indices only (English version)."""
        indices_data = self.data_fetcher.get_major_indices()
        indices = []
        for idx in indices_data:
            indices.append(MarketIndex(
                code=idx.get('code', ''),
                name=idx.get('name', ''),
                current=idx.get('close', 0),
                change=idx.get('change', 0),
                change_pct=idx.get('change_pct', 0),
                open=idx.get('open', 0),
                high=idx.get('high', 0),
                low=idx.get('low', 0),
                prev_close=idx.get('prev_close', 0),
                volume=idx.get('volume', 0),
                amount=idx.get('amount', 0),
                amplitude=idx.get('amplitude', 0),
            ))
        return MarketOverview(
            date=datetime.now().strftime('%Y-%m-%d'),
            indices=indices,
            up_count=len([i for i in indices if i.change_pct > 0]),
            down_count=len([i for i in indices if i.change_pct < 0]),
            total_amount=sum(i.amount for i in indices if i.amount),
        )

    def search_market_news(self) -> List[str]:
        """Search recent market news (English only)."""
        return self.search_service.search_market_news(days=1)

    def _generate_english_review(self, overview: MarketOverview, news: List[str]) -> str:
        """Pure English template - no Chinese strings at all."""
        indices_text = "\n".join([
            f"- **{idx.name}**: {idx.current:.2f} ({idx.change_pct:+.2f}%)" 
            for idx in overview.indices[:6]
        ])

        return f"""## {overview.date} Global Market Recap

### 1. Market Summary
Today's global markets (S&P 500, Nasdaq, ASX, and major indices) showed { "broad weakness" if overview.down_count > overview.up_count else "mixed sentiment" } with {overview.down_count} declining indices out of {len(overview.indices)}.

### 2. Index Commentary
{indices_text}

### 3. Market Stats
- Up: {overview.up_count} | Down: {overview.down_count}
- Total turnover: {overview.total_amount/1e9:.1f} billion (USD equivalent)

### 4. Sector Performance
Leading sectors were limited; most sectors (especially high-beta and cyclical) underperformed significantly.

### 5. Outlook
Short-term outlook remains cautious. Expect continued volatility until clear support levels hold or positive catalysts appear.

### 6. Risk Alerts
- High downside risk if key supports break
- Increased volatility likely in the near term

### 7. Strategy Plan
Maintain defensive positioning and strict risk control. Wait for confirmation of stabilization before adding exposure.

This report is for reference only and does not constitute investment advice.
---
*Recap time: {datetime.now().strftime('%H:%M')}*
"""

    def generate_market_review(self, overview: MarketOverview, news: List[str]) -> str:
        """Force English only."""
        return self._generate_english_review(overview, news)

    def run_daily_review(self) -> str:
        logger.info("========== Starting Global Market Recap (English only) ==========")
        overview = self.get_market_overview()
        news = self.search_market_news()
        report = self.generate_market_review(overview, news)
        logger.info("========== Global Market Recap completed (fully English) ==========")
        return report
