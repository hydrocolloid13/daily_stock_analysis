# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================
职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
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
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    date: str
    indices: List[MarketIndex] = field(default_factory=list)
    up_count: int = 0
    down_count: int = 0
    flat_count: int = 0
    limit_up_count: int = 0
    limit_down_count: int = 0
    total_amount: float = 0.0
    top_sectors: List[Dict] = field(default_factory=list)
    bottom_sectors: List[Dict] = field(default_factory=list)


class MarketAnalyzer:
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
    ):
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)

    def get_market_overview(self) -> MarketOverview:
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        overview.indices = self._get_main_indices()
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
        return overview

    def _get_main_indices(self) -> List[MarketIndex]:
        indices = []
        try:
            data_list = self.data_manager.get_main_indices(region=self.region)
            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)
        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")
        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        try:
            stats = self.data_manager.get_market_stats()
            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        try:
            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)
            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")

    def search_market_news(self) -> List[Dict]:
        if not self.search_service:
            return []
        all_news = []
        search_queries = self.profile.news_queries
        try:
            market_name = "大盘" if self.region == "cn" else "US market"
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        return all_news

    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        if not self.analyzer or not self.analyzer.is_available():
            return self._generate_template_review(overview, news)
       
        prompt = self._build_review_prompt(overview, news)
        review = self.analyzer.generate_text(prompt, max_tokens=2048, temperature=0.7)
        if review:
            return self._inject_data_into_review(review, overview)
        return self._generate_template_review(overview, news)

    def _inject_data_into_review(self, review: str, overview: MarketOverview) -> str:
        import re
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        if stats_block:
            review = self._insert_after_section(review, r'###\s*1\.', stats_block)
        if indices_block:
            review = self._insert_after_section(review, r'###\s*2\.', indices_block)
        if sector_block:
            review = self._insert_after_section(review, r'###\s*4\.', sector_block)
        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        import re
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        next_heading = re.search(r'\n###\s', text[start:])
        insert_pos = start + next_heading.start() if next_heading else len(text)
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        if not (overview.up_count or overview.down_count or overview.total_amount):
            return ""
        return f"> 📈 Up: **{overview.up_count}** | Down: **{overview.down_count}** | Flat: **{overview.flat_count}** | Total turnover: **{overview.total_amount:.0f}** CNY bn"

    def _build_indices_block(self, overview: MarketOverview) -> str:
        if not overview.indices:
            return ""
        lines = ["| Index | Latest | Change | Turnover (bn) |", "|-------|--------|--------|---------------|"]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_str = f"{(idx.amount or 0)/1e8:.0f}" if idx.amount else "N/A"
            lines.append(f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | {amount_str} |")
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        lines = []
        if overview.top_sectors:
            top = " | ".join([f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]])
            lines.append(f"> 🔥 Leading: {top}")
        if overview.bottom_sectors:
            bot = " | ".join([f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]])
            lines.append(f"> 💧 Lagging: {bot}")
        return "\n".join(lines)

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """Build review prompt with VERY STRONG English forcing (always English now)."""
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])

        news_text = ""
        for i, n in enumerate(news[:6], 1):
            title = getattr(n, 'title', n.get('title', ''))[:50]
            snippet = getattr(n, 'snippet', n.get('snippet', ''))[:100]
            news_text += f"{i}. {title}\n {snippet}\n"

        strong_directive = """**CRITICAL LANGUAGE DIRECTIVE - HIGHEST PRIORITY (READ FIRST)**
You are a professional English-language investment analyst writing for English-speaking investors in Sydney.
**YOU MUST RESPOND EXCLUSIVELY IN PROFESSIONAL ENGLISH ONLY.**
- NEVER use any Chinese characters anywhere in your response.
- All text, headings, advice, summaries, risk alerts, and explanations must be in clear, natural English.
- Translate all concepts, company names, sectors, and terms into standard financial English.
This instruction has the absolute highest priority and overrides everything else in the prompt.
"""

        return strong_directive + f"""You are a professional US/A/H market analyst covering S&P 500, ASX, and China A-shares.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)

---
# Today's Market Data
## Date
{overview.date}

## Major Indices
{indices_text}

## Market Stats
- Up: {overview.up_count} | Down: {overview.down_count} | Flat: {overview.flat_count}
- Total turnover: {overview.total_amount:.0f} CNY bn

## Sector Performance
Leading: {top_sectors_text or "N/A"}
Lagging: {bottom_sectors_text or "N/A"}

## Market News
{news_text or "No relevant news found"}

{self.strategy.to_prompt_block()}

---
# Output Template (follow this structure exactly)

## {overview.date} Global Market Recap

### 1. Market Summary
(2-3 sentences on overall performance across US, Australia, and China markets)

### 2. Index Commentary
(Analyse S&P 500, ASX 200, Shanghai Composite, etc.)

### 3. Fund Flows & Volume
(Interpret turnover and flow implications)

### 4. Sector & Theme Highlights
(Analyse leading/lagging sectors)

### 5. Outlook
(Short-term view for the coming days)

### 6. Risk Alerts
(Key risks to watch)

### 7. Strategy Plan
(Provide risk-on/neutral/risk-off stance, position sizing, and one invalidation trigger)

---
Output the report content directly, no extra commentary.
"""

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """Fallback template - fully in English"""
        mood_code = self.profile.mood_index_code
        mood_index = next(
            (idx for idx in overview.indices if idx.code == mood_code or idx.code.endswith(mood_code)),
            None,
        )
        market_mood = "Sideways"
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = "Strong Rise"
            elif mood_index.change_pct > 0:
                market_mood = "Mild Rise"
            elif mood_index.change_pct > -1:
                market_mood = "Mild Decline"
            else:
                market_mood = "Sharp Decline"

        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        top_text = ", ".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = ", ".join([s['name'] for s in overview.bottom_sectors[:3]])

        stats_section = ""
        if self.profile.has_market_stats:
            stats_section = f"""
### 3. Market Stats
- Up: {overview.up_count} | Down: {overview.down_count}
- Total turnover: {overview.total_amount:.0f} billion CNY
"""

        sector_section = ""
        if self.profile.has_sector_rankings and (top_text or bottom_text):
            sector_section = f"""
### 4. Sector Performance
- Leading: {top_text or "N/A"}
- Lagging: {bottom_text or "N/A"}
"""

        strategy_summary = self.strategy.to_markdown_block()

        return f"""## {overview.date} Global Market Recap

### 1. Market Summary
Today's global markets showed **{market_mood}** overall.

### 2. Major Indices
{indices_text}
{stats_section}
{sector_section}

### 6. Risk Alerts
Markets carry risk. The above is for reference only and does not constitute investment advice.

{strategy_summary}
---
*Report generated at: {datetime.now().strftime('%H:%M')}*
"""

    def run_daily_review(self) -> str:
        logger.info("========== 开始大盘复盘分析 ==========")
        overview = self.get_market_overview()
        news = self.search_market_news()
        report = self.generate_market_review(overview, news)
        logger.info("========== 大盘复盘分析完成 ==========")
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
   
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
   
    analyzer = MarketAnalyzer()
    overview = analyzer.get_market_overview()
    report = analyzer._generate_template_review(overview, [])
    print(report)
