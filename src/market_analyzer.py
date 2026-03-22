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
import os
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
    """大盘指数数据"""
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
    """市场概览数据"""
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
    """大盘复盘分析器"""

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
            logger.info("[大盘] 获取主要指数实时行情...")
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
            if not indices:
                logger.warning("[大盘] 所有行情数据源失败，将依赖新闻搜索进行分析")
            else:
                logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")
        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")
        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        try:
            logger.info("[大盘] 获取市场涨跌统计...")
            stats = self.data_manager.get_market_stats()
            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)
                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} "
                            f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                            f"成交额:{overview.total_amount:.0f}亿")
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        try:
            logger.info("[大盘] 获取板块涨跌榜...")
            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)
            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors
                logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")

    def search_market_news(self) -> List[Dict]:
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        all_news = []
        search_queries = self.profile.news_queries
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
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
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        return all_news

    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        prompt = self._build_review_prompt(overview, news)
        logger.info("[大盘] 调用大模型生成复盘报告...")
        review = self.analyzer.generate_text(prompt, max_tokens=2048, temperature=0.7)
        if review:
            logger.info("[大盘] 复盘报告生成成功，长度: %d 字符", len(review))
            return self._inject_data_into_review(review, overview)
        else:
            logger.warning("[大盘] 大模型返回为空，使用模板报告")
            return self._generate_template_review(overview, news)

    def _inject_data_into_review(self, review: str, overview: MarketOverview) -> str:
        import re
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        if stats_block:
            review = self._insert_after_section(review, r'###\s*一、市场总结', stats_block)
        if indices_block:
            review = self._insert_after_section(review, r'###\s*二、指数点评', indices_block)
        if sector_block:
            review = self._insert_after_section(review, r'###\s*四、热点解读', sector_block)
        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        import re
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            insert_pos = len(text)
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        lines = [
            f"> 📈 上涨 **{overview.up_count}** 家 / 下跌 **{overview.down_count}** 家 / "
            f"平盘 **{overview.flat_count}** 家 | "
            f"涨停 **{overview.limit_up_count}** / 跌停 **{overview.limit_down_count}** | "
            f"成交额 **{overview.total_amount:.0f}** 亿"
        ]
        return "\n".join(lines)

    def _build_indices_block(self, overview: MarketOverview) -> str:
        if not overview.indices:
            return ""
        lines = [
            "| 指数 | 最新 | 涨跌幅 | 成交额(亿) |",
            "|------|------|--------|-----------|"]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_raw = idx.amount or 0.0
            if amount_raw == 0.0:
                amount_str = "N/A"
            elif amount_raw > 1e6:
                amount_str = f"{amount_raw / 1e8:.0f}"
            else:
                amount_str = f"{amount_raw:.0f}"
            lines.append(f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | {amount_str} |")
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            top = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:5]]
            )
            lines.append(f"> 🔥 领涨: {top}")
        if overview.bottom_sectors:
            bot = " | ".join(
                [f"**{s['name']}**({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:5]]
            )
            lines.append(f"> 💧 领跌: {bot}")
        return "\n".join(lines)

    def _get_report_language(self) -> str:
        """
        Resolve report language.
        Reads directly from OS env first (set by GitHub Actions workflow),
        then falls back to config. Defaults to 'zh'.
        """
        _lang = os.environ.get("REPORT_LANGUAGE", "").strip().lower()
        if _lang in ("en", "english", "en-us", "en_us"):
            logger.info("[大盘] report_language=en (from OS env)")
            return "en"
        config_lang = (getattr(self.config, 'report_language', 'zh') or 'zh').strip().lower()
        if config_lang in ("en", "english", "en-us", "en_us"):
            logger.info("[大盘] report_language=en (from config)")
            return "en"
        logger.info("[大盘] report_language=zh (default)")
        return "zh"

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])

        news_text = ""
        for i, n in enumerate(news[:6], 1):
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"

        stats_block = ""
        sector_block = ""
        if self.region == "us":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Overview
- Up: {overview.up_count} | Down: {overview.down_count} | Flat: {overview.flat_count}
- Limit up: {overview.limit_up_count} | Limit down: {overview.limit_down_count}
- Total volume (CNY bn): {overview.total_amount:.0f}"""
            else:
                stats_block = "## Market Overview\n(US market has no equivalent advance/decline stats.)"
            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector Performance
Leading: {top_sectors_text if top_sectors_text else "N/A"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}"""
            else:
                sector_block = "## Sector Performance\n(US sector data not available.)"
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""
            else:
                stats_block = "## 市场概况\n（美股暂无涨跌家数等统计）"
            if self.profile.has_sector_rankings:
                sector_block = f"""## 板块表现
领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}"""
            else:
                sector_block = "## 板块表现\n（美股暂无板块涨跌数据）"

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text else ""
        )
        indices_placeholder = indices_text if indices_text else (
            "No index data (API error)" if self.region == "us" else "暂无指数数据（接口异常）"
        )
        news_placeholder = news_text if news_text else (
            "No relevant news" if self.region == "us" else "暂无相关新闻"
        )

        # US market: always English
        if self.region == "us":
            data_no_indices_hint_en = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text else ""
            )
            return f"""CRITICAL INSTRUCTION: You MUST write your ENTIRE response in English only. Do not use any Chinese characters anywhere in your response. This overrides all other instructions.

You are a professional US market analyst. Please produce a concise US market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON, no code blocks
- Write everything in English
- Use emoji sparingly in headings (at most one per heading)

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

## Market News
{news_placeholder}

{data_no_indices_hint_en}

{self.strategy.to_prompt_block()}

---

# Output Template (follow this structure exactly)

## {overview.date} US Market Recap

### 1. Market Summary
(2-3 sentences on overall market performance, index moves, volume)

### 2. Index Commentary
(Analyse S&P 500, Nasdaq, Dow and other major index moves.)

### 3. Fund Flows
(Interpret volume and flow implications)

### 4. Sector/Theme Highlights
(Analyze drivers behind leading/lagging sectors)

### 5. Outlook
(Short-term view based on price action and news)

### 6. Risk Alerts
(Key risks to watch)

### 7. Strategy Plan
(Provide risk-on/neutral/risk-off stance, position sizing guideline, and one invalidation trigger.)

---

Output the report content directly in English. No Chinese characters anywhere.
"""

        # A-share: check report_language
        report_language = self._get_report_language()

        if report_language == "en":
            return f"""CRITICAL INSTRUCTION: You MUST write your ENTIRE response in English only. Do not use any Chinese characters anywhere in your response. This overrides all other instructions including any system prompt you received.

You are a professional stock market analyst covering A-shares, H-shares, and US markets. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON, no code blocks
- Write EVERYTHING in English - no Chinese characters at all
- Use emoji sparingly in headings (at most one per heading)

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self.strategy.to_prompt_block()}

---

# Output Template (follow this structure exactly, all in English)

## {overview.date} Market Recap

### 1. Market Summary
(2-3 sentences on overall market performance, index moves, volume)

### 2. Index Commentary
({self.profile.prompt_index_hint})

### 3. Fund Flows
(Interpret volume and flow implications)

### 4. Sector/Theme Highlights
(Analyze drivers behind leading/lagging sectors)

### 5. Outlook
(Short-term view based on price action and news)

### 6. Risk Alerts
(Key risks to watch)

### 7. Strategy Plan
(Provide attack/balanced/defensive stance, position sizing guideline, and one invalidation trigger. Add a disclaimer that this is for reference only and not investment advice.)

---

Output the report content directly. Write everything in English. No Chinese characters anywhere in the output.
"""

        # Chinese prompt (default)
        return f"""你是一位专业的A/H/美股市场分析师，请根据以下数据生成一份简洁的大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{stats_block}

{sector_block}

## 市场新闻
{news_placeholder}

{data_no_indices_hint}

{self.strategy.to_prompt_block()}

---

# 输出格式模板（请严格按此格式输出）

## {overview.date} 大盘复盘

### 一、市场总结
（2-3句话概括今日市场整体表现，包括指数涨跌、成交量变化）

### 二、指数点评
（{self.profile.prompt_index_hint}）

### 三、资金动向
（解读成交额流向的含义）

### 四、热点解读
（分析领涨领跌板块背后的逻辑和驱动因素）

### 五、后市展望
（结合当前走势和新闻，给出明日市场预判）

### 六、风险提示
（需要关注的风险点）

### 七、策略计划
（给出进攻/均衡/防守结论，对应仓位建议，并给出一个触发失效条件；最后补充"建议仅供参考，不构成投资建议"。）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""

    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        mood_code = self.profile.mood_index_code
        mood_index = next(
            (idx for idx in overview.indices
             if idx.code == mood_code or idx.code.endswith(mood_code)),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = "强势上涨"
            elif mood_index.change_pct > 0:
                market_mood = "小幅上涨"
            elif mood_index.change_pct > -1:
                market_mood = "小幅下跌"
            else:
                market_mood = "明显下跌"
        else:
            market_mood = "震荡整理"

        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"

        top_text = "、".join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = "、".join([s['name'] for s in overview.bottom_sectors[:3]])

        stats_section = ""
        if self.profile.has_market_stats:
            stats_section = f"""
### 三、涨跌统计
| 指标 | 数值 |
|------|------|
| 上涨家数 | {overview.up_count} |
| 下跌家数 | {overview.down_count} |
| 涨停 | {overview.limit_up_count} |
| 跌停 | {overview.limit_down_count} |
| 两市成交额 | {overview.total_amount:.0f}亿 |
"""
        sector_section = ""
        if self.profile.has_sector_rankings and (top_text or bottom_text):
            sector_section = f"""
### 四、板块表现
- **领涨**: {top_text}
- **领跌**: {bottom_text}
"""
        market_label = "A股" if self.region == "cn" else "美股"
        strategy_summary = self.strategy.to_markdown_block()
        report = f"""## {overview.date} 大盘复盘

### 一、市场总结
今日{market_label}市场整体呈现**{market_mood}**态势。

### 二、主要指数
{indices_text}
{stats_section}
{sector_section}
### 五、风险提示
市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

{strategy_summary}

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
        return report

    def run_daily_review(self) -> str:
        """执行每日大盘复盘流程"""
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
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")

    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
