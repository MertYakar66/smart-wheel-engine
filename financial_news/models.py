"""
Core data models for the Financial News Platform

Following Bloomberg's approach: news is structured, "tickerized" events
with rich metadata, so feeds can be built from queries + categories.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Severity(Enum):
    """News impact severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ArticleSource(Enum):
    """Supported article sources"""

    GDELT = "gdelt"
    SEC_EDGAR = "sec_edgar"
    RSS = "rss"
    LICENSED = "licensed"


class TopicCategory(Enum):
    """Primary topic categories (Bloomberg-style taxonomy)"""

    MACRO_RATES = "macro_rates"
    MACRO_INFLATION = "macro_inflation"
    MACRO_EMPLOYMENT = "macro_employment"
    EARNINGS = "earnings"
    M_AND_A = "m_and_a"
    IPO = "ipo"
    COMMODITIES_OIL = "commodities_oil"
    COMMODITIES_GOLD = "commodities_gold"
    COMMODITIES_AGRI = "commodities_agri"
    GEOPOLITICS = "geopolitics"
    REGULATION = "regulation"
    TECH_AI = "tech_ai"
    TECH_SEMIS = "tech_semis"
    CRYPTO = "crypto"
    FOREX = "forex"
    FIXED_INCOME = "fixed_income"
    CENTRAL_BANKS = "central_banks"
    CHINA = "china"
    EUROPE = "europe"
    EMERGING_MARKETS = "emerging_markets"


@dataclass
class Entity:
    """Extracted entity from news article"""

    name: str
    entity_type: str  # company, person, country, product, etc.
    ticker: str | None = None  # Stock ticker if applicable
    figi: str | None = None  # Financial Instrument Global Identifier
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "ticker": self.ticker,
            "figi": self.figi,
            "confidence": self.confidence,
        }


@dataclass
class Article:
    """
    Canonical article representation.

    Stores metadata + links only (no full text for licensed content).
    """

    article_id: str
    canonical_url: str
    source: ArticleSource
    source_name: str  # Publisher name
    published_at_utc: datetime
    title: str
    snippet: str | None = None  # Short excerpt if permitted
    language: str = "en"
    country: str = "US"

    # Extracted metadata
    entities: list[Entity] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    topics: list[TopicCategory] = field(default_factory=list)

    # Scoring
    relevance_scores: dict[str, float] = field(default_factory=dict)
    impact_score: float = 0.0

    # Provenance
    retrieval_provider: str | None = None
    fetched_at_utc: datetime = field(default_factory=datetime.utcnow)

    @staticmethod
    def generate_id(url: str, published_at: datetime, source: str) -> str:
        """Generate unique article ID from URL + timestamp + source"""
        content = f"{url}|{published_at.isoformat()}|{source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "canonical_url": self.canonical_url,
            "source": self.source.value,
            "source_name": self.source_name,
            "published_at_utc": self.published_at_utc.isoformat(),
            "title": self.title,
            "snippet": self.snippet,
            "language": self.language,
            "country": self.country,
            "entities": [e.to_dict() for e in self.entities],
            "tickers": self.tickers,
            "topics": [t.value for t in self.topics],
            "relevance_scores": self.relevance_scores,
            "impact_score": self.impact_score,
            "retrieval_provider": self.retrieval_provider,
            "fetched_at_utc": self.fetched_at_utc.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Article":
        entities = [Entity(**e) for e in data.get("entities", [])]
        topics = [TopicCategory(t) for t in data.get("topics", [])]
        return cls(
            article_id=data["article_id"],
            canonical_url=data["canonical_url"],
            source=ArticleSource(data["source"]),
            source_name=data["source_name"],
            published_at_utc=datetime.fromisoformat(data["published_at_utc"]),
            title=data["title"],
            snippet=data.get("snippet"),
            language=data.get("language", "en"),
            country=data.get("country", "US"),
            entities=entities,
            tickers=data.get("tickers", []),
            topics=topics,
            relevance_scores=data.get("relevance_scores", {}),
            impact_score=data.get("impact_score", 0.0),
            retrieval_provider=data.get("retrieval_provider"),
            fetched_at_utc=datetime.fromisoformat(data["fetched_at_utc"])
            if data.get("fetched_at_utc")
            else datetime.utcnow(),
        )


@dataclass
class Story:
    """
    A clustered story combining multiple articles about the same event.

    This is the key abstraction - users see stories, not individual articles.
    """

    story_id: str
    lead_article_id: str  # Most representative article
    headline: str  # Generated summary headline
    summary: str  # "What happened" - 1-2 sentences
    why_it_matters: str  # "Why it matters" - mechanism explanation

    first_seen_at: datetime
    last_updated_at: datetime

    # Aggregated metadata
    entities: list[Entity] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    topics: list[TopicCategory] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)

    # Coverage
    article_ids: list[str] = field(default_factory=list)
    source_count: int = 0  # Number of independent sources

    # Scoring
    impact_score: float = 0.0
    confidence_score: float = 0.0  # Based on source diversity

    # For tracking changes
    previous_summary: str | None = None
    change_description: str | None = None  # "What changed since last brief"

    @staticmethod
    def generate_id(lead_article_id: str, first_seen: datetime) -> str:
        content = f"{lead_article_id}|{first_seen.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "story_id": self.story_id,
            "lead_article_id": self.lead_article_id,
            "headline": self.headline,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "entities": [e.to_dict() for e in self.entities],
            "tickers": self.tickers,
            "topics": [t.value for t in self.topics],
            "regions": self.regions,
            "article_ids": self.article_ids,
            "source_count": self.source_count,
            "impact_score": self.impact_score,
            "confidence_score": self.confidence_score,
            "previous_summary": self.previous_summary,
            "change_description": self.change_description,
        }


@dataclass
class Category:
    """
    A saved query that defines a news category.

    Bloomberg concept: categories are queries + rules, not static buckets.
    """

    category_id: str
    name: str  # e.g., "Macro: Rates & Inflation"
    description: str

    # Query configuration
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # Company/person names
    tickers: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)  # Keywords to exclude

    # Filters
    topics: list[TopicCategory] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # Specific sources only

    # Ranking weights
    recency_weight: float = 0.4
    impact_weight: float = 0.3
    relevance_weight: float = 0.3

    # Digest configuration
    max_stories_per_brief: int = 10
    digest_tone: str = "professional"  # professional, casual, technical

    # State
    is_active: bool = True
    last_successful_fetch: datetime | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "entities": self.entities,
            "tickers": self.tickers,
            "exclusions": self.exclusions,
            "topics": [t.value for t in self.topics],
            "regions": self.regions,
            "languages": self.languages,
            "sources": self.sources,
            "recency_weight": self.recency_weight,
            "impact_weight": self.impact_weight,
            "relevance_weight": self.relevance_weight,
            "max_stories_per_brief": self.max_stories_per_brief,
            "digest_tone": self.digest_tone,
            "is_active": self.is_active,
            "last_successful_fetch": self.last_successful_fetch.isoformat()
            if self.last_successful_fetch
            else None,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UserProfile:
    """User profile with watchlist and category subscriptions"""

    user_id: str
    email: str | None = None

    # Subscriptions
    category_ids: list[str] = field(default_factory=list)
    watchlist_tickers: list[str] = field(default_factory=list)

    # Preferences
    timezone: str = "America/Toronto"
    morning_brief_time: str = "07:00"
    evening_brief_time: str = "19:00"
    enable_push_notifications: bool = False
    enable_email_digest: bool = True

    # State
    last_seen_story_ids: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "category_ids": self.category_ids,
            "watchlist_tickers": self.watchlist_tickers,
            "timezone": self.timezone,
            "morning_brief_time": self.morning_brief_time,
            "evening_brief_time": self.evening_brief_time,
            "enable_push_notifications": self.enable_push_notifications,
            "enable_email_digest": self.enable_email_digest,
            "last_seen_story_ids": list(self.last_seen_story_ids),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Brief:
    """Morning or Evening brief for a user"""

    brief_id: str
    user_id: str
    brief_type: str  # "morning" or "evening"
    generated_at: datetime

    stories: list[Story] = field(default_factory=list)
    new_stories_count: int = 0
    updated_stories_count: int = 0

    # Generated content
    executive_summary: str | None = None
    market_outlook: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "user_id": self.user_id,
            "brief_type": self.brief_type,
            "generated_at": self.generated_at.isoformat(),
            "stories": [s.to_dict() for s in self.stories],
            "new_stories_count": self.new_stories_count,
            "updated_stories_count": self.updated_stories_count,
            "executive_summary": self.executive_summary,
            "market_outlook": self.market_outlook,
        }


# Default categories following the user's taxonomy
DEFAULT_CATEGORIES = [
    Category(
        category_id="macro_rates",
        name="Macro: Rates & Inflation",
        description="Federal Reserve policy, interest rates, inflation data, bond markets",
        keywords=[
            "fed",
            "federal reserve",
            "interest rate",
            "inflation",
            "cpi",
            "pce",
            "fomc",
            "powell",
            "treasury yield",
        ],
        topics=[
            TopicCategory.MACRO_RATES,
            TopicCategory.MACRO_INFLATION,
            TopicCategory.CENTRAL_BANKS,
        ],
    ),
    Category(
        category_id="earnings",
        name="Corporate Earnings",
        description="Quarterly earnings reports, guidance, revenue surprises",
        keywords=[
            "earnings",
            "eps",
            "revenue",
            "quarterly results",
            "guidance",
            "beat",
            "miss",
            "outlook",
        ],
        topics=[TopicCategory.EARNINGS],
    ),
    Category(
        category_id="tech_ai",
        name="Technology: AI & Chips",
        description="Artificial intelligence developments, semiconductor industry, tech regulation",
        keywords=[
            "ai",
            "artificial intelligence",
            "nvidia",
            "openai",
            "semiconductor",
            "chip",
            "gpu",
            "llm",
            "machine learning",
        ],
        tickers=["NVDA", "AMD", "INTC", "GOOGL", "MSFT", "META", "TSMC"],
        topics=[TopicCategory.TECH_AI, TopicCategory.TECH_SEMIS],
    ),
    Category(
        category_id="commodities_energy",
        name="Commodities: Energy",
        description="Oil, natural gas, OPEC decisions, energy sector",
        keywords=[
            "oil",
            "crude",
            "brent",
            "wti",
            "opec",
            "natural gas",
            "lng",
            "energy",
            "petroleum",
        ],
        topics=[TopicCategory.COMMODITIES_OIL],
    ),
    Category(
        category_id="geopolitics",
        name="Geopolitics & Trade",
        description="International relations, trade agreements, sanctions, conflicts",
        keywords=[
            "tariff",
            "sanction",
            "trade war",
            "geopolitical",
            "conflict",
            "diplomacy",
            "embargo",
        ],
        topics=[TopicCategory.GEOPOLITICS],
    ),
    Category(
        category_id="china",
        name="China Markets",
        description="Chinese economy, markets, regulatory changes, US-China relations",
        keywords=["china", "chinese", "beijing", "shanghai", "ccp", "yuan", "renminbi"],
        topics=[TopicCategory.CHINA],
        regions=["CN", "HK"],
    ),
    Category(
        category_id="crypto",
        name="Crypto & Digital Assets",
        description="Cryptocurrency markets, blockchain, digital asset regulation",
        keywords=["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft", "stablecoin"],
        topics=[TopicCategory.CRYPTO],
    ),
    Category(
        category_id="m_and_a",
        name="M&A & Corporate Actions",
        description="Mergers, acquisitions, spinoffs, buybacks",
        keywords=["merger", "acquisition", "buyout", "takeover", "spinoff", "buyback", "ipo"],
        topics=[TopicCategory.M_AND_A, TopicCategory.IPO],
    ),
]
