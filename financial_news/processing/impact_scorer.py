"""
Impact Scoring for News Stories

Scores stories based on potential market impact using:
- Source diversity (more independent sources = higher confidence)
- Entity importance (major companies, central banks)
- Topic severity (rate decisions > routine filings)
- Keyword signals (guidance cut, SEC investigation)
- Abnormal news volume

Optional: price/volume anomaly detection if market data available.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import logging

from financial_news.models import Article, Story, TopicCategory

logger = logging.getLogger(__name__)


# High-impact keywords (signals potential market moves)
HIGH_IMPACT_KEYWORDS = {
    "critical": [
        "guidance cut", "guidance lowered", "profit warning",
        "sec investigation", "fbi investigation", "fraud",
        "bankruptcy", "chapter 11", "default",
        "emergency rate cut", "surprise rate hike",
        "hostile takeover", "activist investor",
        "ceo resign", "cfo resign", "ceo fired",
        "data breach", "cyberattack",
        "product recall", "fda warning",
    ],
    "high": [
        "earnings miss", "revenue miss", "guidance",
        "downgrade", "upgrade", "price target",
        "layoffs", "restructuring", "cost cutting",
        "merger", "acquisition", "takeover",
        "dividend cut", "buyback",
        "fed decision", "rate decision", "fomc",
        "sanctions", "tariffs", "trade war",
    ],
    "medium": [
        "beat estimates", "exceeded expectations",
        "new product", "partnership", "contract",
        "expansion", "investment",
        "stock split", "secondary offering",
        "analyst coverage", "initiated coverage",
    ],
}

# Topic importance weights
TOPIC_WEIGHTS = {
    TopicCategory.CENTRAL_BANKS: 1.0,  # Fed decisions are always major
    TopicCategory.MACRO_RATES: 0.9,
    TopicCategory.EARNINGS: 0.8,
    TopicCategory.M_AND_A: 0.8,
    TopicCategory.GEOPOLITICS: 0.7,
    TopicCategory.REGULATION: 0.7,
    TopicCategory.MACRO_INFLATION: 0.7,
    TopicCategory.TECH_AI: 0.6,
    TopicCategory.COMMODITIES_OIL: 0.6,
    TopicCategory.CRYPTO: 0.5,
}

# Major entities (always newsworthy)
MAJOR_ENTITIES = {
    # Central banks
    "federal reserve", "fed", "ecb", "bank of japan", "bank of england",
    "people's bank of china",

    # Mega-cap companies
    "apple", "microsoft", "google", "amazon", "nvidia", "meta",
    "berkshire hathaway", "jpmorgan", "exxon",

    # Key indices
    "s&p 500", "dow jones", "nasdaq", "russell 2000",
}


class ImpactScorer:
    """
    Scores news stories by potential market impact.

    Impact score is 0-1 scale where:
    - 0.0-0.3: Low impact (routine news)
    - 0.3-0.6: Medium impact (notable)
    - 0.6-0.8: High impact (likely to move markets)
    - 0.8-1.0: Critical (emergency/breaking)
    """

    def __init__(
        self,
        enable_volume_scoring: bool = True,
        market_data_provider: Optional[object] = None,
    ):
        """
        Initialize impact scorer.

        Args:
            enable_volume_scoring: Score based on news volume anomalies
            market_data_provider: Optional provider for price/volume data
        """
        self.enable_volume_scoring = enable_volume_scoring
        self.market_data_provider = market_data_provider

        # Track news volume for anomaly detection
        self._ticker_volume: Dict[str, List[datetime]] = {}
        self._topic_volume: Dict[str, List[datetime]] = {}

    def score_story(self, story: Story, articles: List[Article] = None) -> float:
        """
        Calculate impact score for a story.

        Args:
            story: Story to score
            articles: Optional list of articles in story (for detailed analysis)

        Returns:
            Impact score (0-1)
        """
        scores = []

        # 1. Source diversity score (0-1)
        # More independent sources = more likely to be significant
        source_score = min(1.0, story.source_count / 5)
        scores.append(("source_diversity", source_score, 0.2))

        # 2. Keyword impact score (0-1)
        text = f"{story.headline} {story.summary}".lower()
        keyword_score = self._score_keywords(text)
        scores.append(("keywords", keyword_score, 0.25))

        # 3. Topic importance score (0-1)
        topic_score = self._score_topics(story.topics)
        scores.append(("topics", topic_score, 0.2))

        # 4. Entity importance score (0-1)
        entity_score = self._score_entities(story.entities)
        scores.append(("entities", entity_score, 0.15))

        # 5. News volume anomaly (0-1)
        if self.enable_volume_scoring:
            volume_score = self._score_volume_anomaly(story)
            scores.append(("volume", volume_score, 0.1))

        # 6. Recency boost (0-1)
        # Breaking news gets a boost
        recency_score = self._score_recency(story.first_seen_at)
        scores.append(("recency", recency_score, 0.1))

        # Calculate weighted average
        total_weight = sum(s[2] for s in scores)
        weighted_sum = sum(s[1] * s[2] for s in scores)
        impact_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Apply any critical keyword boost
        if keyword_score > 0.8:
            impact_score = min(1.0, impact_score * 1.2)

        logger.debug(
            f"Impact score for '{story.headline[:50]}': {impact_score:.3f} "
            f"(components: {[(s[0], f'{s[1]:.2f}') for s in scores]})"
        )

        return round(impact_score, 3)

    def _score_keywords(self, text: str) -> float:
        """Score based on high-impact keyword presence"""
        # Check critical keywords (highest weight)
        for keyword in HIGH_IMPACT_KEYWORDS["critical"]:
            if keyword in text:
                return 1.0

        # Check high-impact keywords
        high_matches = sum(1 for kw in HIGH_IMPACT_KEYWORDS["high"] if kw in text)
        if high_matches >= 2:
            return 0.8
        if high_matches == 1:
            return 0.6

        # Check medium keywords
        medium_matches = sum(1 for kw in HIGH_IMPACT_KEYWORDS["medium"] if kw in text)
        if medium_matches >= 2:
            return 0.4
        if medium_matches == 1:
            return 0.2

        return 0.1

    def _score_topics(self, topics: List[TopicCategory]) -> float:
        """Score based on topic importance"""
        if not topics:
            return 0.3

        # Use highest-weighted topic
        max_weight = max(TOPIC_WEIGHTS.get(t, 0.3) for t in topics)
        return max_weight

    def _score_entities(self, entities: List) -> float:
        """Score based on entity importance"""
        if not entities:
            return 0.2

        # Check for major entities
        entity_names = [e.name.lower() for e in entities]

        for major in MAJOR_ENTITIES:
            if any(major in name for name in entity_names):
                return 1.0

        # Score based on entity count (more entities = more significant)
        entity_score = min(1.0, len(entities) / 5)

        # Boost for company entities with tickers
        has_tickers = any(e.ticker for e in entities)
        if has_tickers:
            entity_score = min(1.0, entity_score * 1.2)

        return entity_score

    def _score_volume_anomaly(self, story: Story) -> float:
        """
        Score based on abnormal news volume.

        High volume on a ticker/topic suggests significance.
        """
        volume_scores = []

        # Check ticker volume
        for ticker in story.tickers:
            count = self._get_recent_count(ticker, self._ticker_volume)
            # Simple heuristic: >5 articles in past hour is unusual
            if count > 10:
                volume_scores.append(1.0)
            elif count > 5:
                volume_scores.append(0.7)
            elif count > 2:
                volume_scores.append(0.4)

        if volume_scores:
            return max(volume_scores)

        return 0.2

    def _get_recent_count(
        self,
        key: str,
        volume_dict: Dict[str, List[datetime]],
        window_hours: int = 1,
    ) -> int:
        """Count articles in recent time window"""
        if key not in volume_dict:
            return 0

        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [t for t in volume_dict[key] if t > cutoff]
        return len(recent)

    def _score_recency(self, first_seen: datetime) -> float:
        """Score based on how recent the story is"""
        age_hours = (datetime.utcnow() - first_seen).total_seconds() / 3600

        if age_hours < 0.5:  # < 30 minutes
            return 1.0
        elif age_hours < 1:  # < 1 hour
            return 0.8
        elif age_hours < 3:  # < 3 hours
            return 0.6
        elif age_hours < 6:  # < 6 hours
            return 0.4
        elif age_hours < 12:  # < 12 hours
            return 0.2
        else:
            return 0.1

    def record_article(self, article: Article) -> None:
        """Record article for volume tracking"""
        now = datetime.utcnow()

        # Track by ticker
        for ticker in article.tickers:
            if ticker not in self._ticker_volume:
                self._ticker_volume[ticker] = []
            self._ticker_volume[ticker].append(now)

        # Track by topic
        for topic in article.topics:
            key = topic.value
            if key not in self._topic_volume:
                self._topic_volume[key] = []
            self._topic_volume[key].append(now)

    def cleanup_old_records(self, max_age_hours: int = 24) -> None:
        """Remove old volume records"""
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        for key in self._ticker_volume:
            self._ticker_volume[key] = [
                t for t in self._ticker_volume[key] if t > cutoff
            ]

        for key in self._topic_volume:
            self._topic_volume[key] = [
                t for t in self._topic_volume[key] if t > cutoff
            ]

    def score_articles(self, articles: List[Article]) -> List[Article]:
        """Score multiple articles and record for volume tracking"""
        for article in articles:
            # Simple article-level impact
            text = f"{article.title} {article.snippet or ''}".lower()
            keyword_score = self._score_keywords(text)
            topic_score = self._score_topics(article.topics)

            article.impact_score = 0.6 * keyword_score + 0.4 * topic_score

            # Record for volume tracking
            self.record_article(article)

        return articles
