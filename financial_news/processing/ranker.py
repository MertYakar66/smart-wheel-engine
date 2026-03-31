"""
Story Ranker - Score Stories by Macro/SP500 Relevance

Ranking formula:
- 35% macro relevance (Fed, inflation, labor, growth, oil, geopolitics)
- 25% SP500 relevance (index heavyweights, sector breadth)
- 20% source quality (official > company_ir > licensed > aggregator)
- 10% corroboration (multi-source confirmation)
- 10% recency (time decay)

Boosts:
- Index heavyweights (AAPL, MSFT, NVDA, etc.)
- Macro release days
- Watchlist overlap
- Oil/geopolitics with market transmission

Penalties:
- Stale stories (>24h)
- Single-source
- Commentary without new facts
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging

from financial_news.schema import (
    Story, Category, CategoryType, SourceType,
    DEFAULT_CATEGORIES,
)

logger = logging.getLogger(__name__)


# Index heavyweights and their approximate weights
SP500_HEAVYWEIGHTS = {
    "AAPL": 0.07, "MSFT": 0.07, "NVDA": 0.05, "AMZN": 0.04, "GOOGL": 0.04,
    "META": 0.03, "TSLA": 0.02, "BRK.B": 0.02, "UNH": 0.01, "JPM": 0.01,
    "JNJ": 0.01, "V": 0.01, "XOM": 0.01, "PG": 0.01, "MA": 0.01,
    "HD": 0.01, "CVX": 0.01, "MRK": 0.01, "ABBV": 0.01, "LLY": 0.01,
}

# Sectors and their tickers
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communications": "XLC",
}


@dataclass
class RankingWeights:
    """Configurable ranking weights"""
    macro_weight: float = 0.35
    sp500_weight: float = 0.25
    source_quality_weight: float = 0.20
    corroboration_weight: float = 0.10
    recency_weight: float = 0.10

    # Boosts
    heavyweight_boost: float = 0.15
    macro_day_boost: float = 0.10
    watchlist_boost: float = 0.20
    oil_geo_boost: float = 0.10

    # Penalties
    stale_penalty: float = 0.20
    single_source_penalty: float = 0.10


@dataclass
class RankingResult:
    """Result of ranking a story"""
    story_id: str
    total_score: float
    macro_score: float
    sp500_score: float
    source_score: float
    corroboration_score: float
    recency_score: float
    boosts_applied: List[str]
    penalties_applied: List[str]


class StoryRanker:
    """
    Rank stories by trading relevance.

    Uses weighted scoring with boosts and penalties.
    """

    def __init__(
        self,
        weights: Optional[RankingWeights] = None,
        categories: Optional[List[Category]] = None,
        watchlist: Optional[Set[str]] = None,
    ):
        self.weights = weights or RankingWeights()
        self.categories = {c.category_id: c for c in (categories or DEFAULT_CATEGORIES)}
        self.watchlist = watchlist or set()

        # Source quality scores
        self.source_quality = {
            SourceType.OFFICIAL: 1.0,
            SourceType.COMPANY_IR: 0.8,
            SourceType.LICENSED: 0.6,
            SourceType.AGGREGATOR: 0.4,
        }

    def rank_stories(
        self,
        stories: List[Story],
        is_macro_day: bool = False,
    ) -> List[RankingResult]:
        """
        Rank stories by relevance.

        Args:
            stories: Stories to rank
            is_macro_day: Whether today has major macro releases

        Returns:
            Sorted list of RankingResults (highest first)
        """
        results = []

        for story in stories:
            result = self._score_story(story, is_macro_day)
            results.append(result)

        # Sort by total score descending
        results.sort(key=lambda r: r.total_score, reverse=True)

        return results

    def _score_story(self, story: Story, is_macro_day: bool) -> RankingResult:
        """Score a single story."""
        boosts = []
        penalties = []

        # 1. Macro relevance score
        macro_score = self._score_macro_relevance(story)

        # 2. SP500 relevance score
        sp500_score = self._score_sp500_relevance(story)

        # 3. Source quality score
        source_score = self._score_source_quality(story)

        # 4. Corroboration score
        corr_score = self._score_corroboration(story)

        # 5. Recency score
        recency_score = self._score_recency(story)

        # Calculate base score
        base_score = (
            self.weights.macro_weight * macro_score +
            self.weights.sp500_weight * sp500_score +
            self.weights.source_quality_weight * source_score +
            self.weights.corroboration_weight * corr_score +
            self.weights.recency_weight * recency_score
        )

        # Apply boosts
        total_boost = 0.0

        # Heavyweight boost
        if self._has_heavyweight(story):
            total_boost += self.weights.heavyweight_boost
            boosts.append("heavyweight")

        # Macro day boost
        if is_macro_day and macro_score > 0.5:
            total_boost += self.weights.macro_day_boost
            boosts.append("macro_day")

        # Watchlist boost
        if self._matches_watchlist(story):
            total_boost += self.weights.watchlist_boost
            boosts.append("watchlist")

        # Oil/geopolitics with market transmission
        if self._has_oil_geo_transmission(story):
            total_boost += self.weights.oil_geo_boost
            boosts.append("oil_geo")

        # Apply penalties
        total_penalty = 0.0

        # Stale penalty
        hours_old = (datetime.utcnow() - story.last_updated_at).total_seconds() / 3600
        if hours_old > 24:
            total_penalty += self.weights.stale_penalty
            penalties.append("stale")

        # Single source penalty
        if story.source_count <= 1:
            total_penalty += self.weights.single_source_penalty
            penalties.append("single_source")

        # Final score
        total_score = min(1.0, max(0.0, base_score + total_boost - total_penalty))

        return RankingResult(
            story_id=story.story_id,
            total_score=total_score,
            macro_score=macro_score,
            sp500_score=sp500_score,
            source_score=source_score,
            corroboration_score=corr_score,
            recency_score=recency_score,
            boosts_applied=boosts,
            penalties_applied=penalties,
        )

    def _score_macro_relevance(self, story: Story) -> float:
        """Score macro relevance based on categories."""
        macro_categories = {
            CategoryType.FED_RATES.value,
            CategoryType.INFLATION.value,
            CategoryType.LABOR.value,
            CategoryType.GROWTH_CONSUMER.value,
            CategoryType.OIL_ENERGY.value,
            CategoryType.GEOPOLITICS.value,
        }

        score = 0.0
        for cat_id, cat_score in story.category_scores.items():
            if cat_id in macro_categories:
                score += cat_score

        # Also check affected factors
        macro_factors = {"rates", "inflation", "growth", "oil", "risk"}
        for factor in story.affected_factors:
            if factor.lower() in macro_factors:
                score += 0.1

        return min(1.0, score)

    def _score_sp500_relevance(self, story: Story) -> float:
        """Score SP500 relevance based on tickers."""
        score = 0.0

        # Check for index heavyweights
        for ticker in story.tickers:
            if ticker in SP500_HEAVYWEIGHTS:
                score += SP500_HEAVYWEIGHTS[ticker] * 5  # Amplify weight

        # Check for sector ETFs
        for sector in story.affected_sectors:
            if sector in SECTOR_ETFS:
                score += 0.1

        # Corporate events category
        if CategoryType.SP500_CORPORATE.value in story.category_scores:
            score += story.category_scores[CategoryType.SP500_CORPORATE.value] * 0.5

        return min(1.0, score)

    def _score_source_quality(self, story: Story) -> float:
        """Score based on source quality."""
        # For now, use source_count as proxy
        # In production, would track actual source types
        if story.source_count >= 3:
            return 1.0
        elif story.source_count == 2:
            return 0.8
        else:
            return 0.5

    def _score_corroboration(self, story: Story) -> float:
        """Score based on multi-source confirmation."""
        if story.source_count >= 5:
            return 1.0
        elif story.source_count >= 3:
            return 0.8
        elif story.source_count >= 2:
            return 0.6
        else:
            return 0.3

    def _score_recency(self, story: Story) -> float:
        """Score based on recency with time decay."""
        hours_old = (datetime.utcnow() - story.last_updated_at).total_seconds() / 3600

        if hours_old <= 1:
            return 1.0
        elif hours_old <= 4:
            return 0.9
        elif hours_old <= 12:
            return 0.7
        elif hours_old <= 24:
            return 0.5
        elif hours_old <= 48:
            return 0.3
        else:
            return 0.1

    def _has_heavyweight(self, story: Story) -> bool:
        """Check if story involves SP500 heavyweights."""
        return any(t in SP500_HEAVYWEIGHTS for t in story.tickers)

    def _matches_watchlist(self, story: Story) -> bool:
        """Check if story matches user watchlist."""
        return bool(self.watchlist & set(story.tickers))

    def _has_oil_geo_transmission(self, story: Story) -> bool:
        """Check if oil/geopolitics story has market transmission."""
        # Check categories
        oil_geo_cats = {
            CategoryType.OIL_ENERGY.value,
            CategoryType.GEOPOLITICS.value,
        }
        has_oil_geo = any(c in oil_geo_cats for c in story.category_scores)

        # Check for market transmission factors
        transmission_factors = {"oil", "rates", "risk", "defense", "shipping"}
        has_transmission = any(f in transmission_factors for f in story.affected_factors)

        return has_oil_geo and has_transmission

    def get_top_stories(
        self,
        stories: List[Story],
        limit: int = 15,
        category: Optional[CategoryType] = None,
        is_macro_day: bool = False,
    ) -> List[Story]:
        """
        Get top ranked stories.

        Args:
            stories: Stories to rank
            limit: Maximum stories to return
            category: Filter to specific category
            is_macro_day: Whether today has major releases

        Returns:
            Top stories sorted by rank
        """
        # Filter by category if specified
        if category:
            stories = [
                s for s in stories
                if category.value in s.category_scores
            ]

        # Rank
        results = self.rank_stories(stories, is_macro_day)

        # Get story IDs
        story_map = {s.story_id: s for s in stories}
        top_stories = []
        for result in results[:limit]:
            if result.story_id in story_map:
                story = story_map[result.story_id]
                story.impact_score = result.total_score
                top_stories.append(story)

        return top_stories
