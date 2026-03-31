"""
Canonical Database Schema for Macro + SP500 Event Intelligence System

This schema is built around scheduled official releases, not generic web crawling.
It supports:
- Official source registry with trust tiers
- 8 focused categories (Fed, Inflation, Labor, Growth, Oil, Geopolitics, SP500, Market Regime)
- Event calendar for macro releases and Fed events
- Story clustering and ranking
- AM/PM brief generation
- Run logging for rate limiting and debugging

Tables:
1. sources - Official source registry with trust tiers
2. categories - 8 core categories
3. category_rules - Deterministic classification rules
4. event_calendar - Scheduled macro releases and Fed events
5. articles - Normalized article/filing/release records
6. article_entities - Extracted entities (M2M)
7. stories - Clustered story records
8. story_articles - Story-article mapping (M2M)
9. story_categories - Story-category scores (M2M)
10. briefs - Generated morning/evening briefs
11. run_logs - Pipeline execution logs
12. user_watchlists - User ticker watchlists
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import hashlib
import json


# =============================================================================
# ENUMS
# =============================================================================

class SourceType(Enum):
    """Source trust tiers - determines processing priority"""
    OFFICIAL = "official"           # Fed, BLS, BEA, Census, EIA, SEC, Treasury, OPEC
    COMPANY_IR = "company_ir"       # Company investor relations, press releases
    LICENSED = "licensed"           # Licensed news providers
    AGGREGATOR = "aggregator"       # News aggregators, broad discovery


class SourceProvider(Enum):
    """Specific source providers"""
    # Tier 1 - Official
    FEDERAL_RESERVE = "federal_reserve"
    BLS = "bls"                     # Bureau of Labor Statistics
    BEA = "bea"                     # Bureau of Economic Analysis
    CENSUS = "census"               # Census Bureau
    EIA = "eia"                     # Energy Information Administration
    SEC_EDGAR = "sec_edgar"         # SEC EDGAR filings
    TREASURY = "treasury"           # Treasury / OFAC
    STATE_DEPT = "state_dept"       # State Department
    OPEC = "opec"                   # OPEC official

    # Tier 2 - Company Official
    COMPANY_IR = "company_ir"       # Company investor relations
    COMPANY_PR = "company_pr"       # Company press releases

    # Tier 3 - Licensed/Aggregator
    RSS_FEED = "rss_feed"
    NEWS_API = "news_api"


class CategoryType(Enum):
    """8 Core Categories - focused on trading-relevant events"""
    FED_RATES = "fed_rates"                 # FOMC, Fed, Powell, rates, policy
    INFLATION = "inflation"                  # CPI, PCE, shelter, services
    LABOR = "labor"                          # NFP, unemployment, JOLTS, wages
    GROWTH_CONSUMER = "growth_consumer"      # GDP, retail, housing, durable goods
    OIL_ENERGY = "oil_energy"               # EIA, OPEC, oil, gas, refinery
    GEOPOLITICS = "geopolitics"             # Sanctions, conflicts with market impact
    SP500_CORPORATE = "sp500_corporate"     # Earnings, 8-K, M&A, guidance
    MARKET_REGIME = "market_regime"         # Derived: risk-on/off, sector rotation


class EventType(Enum):
    """Types of scheduled events"""
    # Fed events
    FOMC_DECISION = "fomc_decision"
    FOMC_MINUTES = "fomc_minutes"
    FED_SPEECH = "fed_speech"
    FED_TESTIMONY = "fed_testimony"

    # Inflation releases
    CPI = "cpi"
    CORE_CPI = "core_cpi"
    PCE = "pce"
    CORE_PCE = "core_pce"
    PPI = "ppi"

    # Labor releases
    NFP = "nfp"                     # Nonfarm Payrolls / Employment Situation
    UNEMPLOYMENT = "unemployment"
    JOLTS = "jolts"
    JOBLESS_CLAIMS = "jobless_claims"
    ADP_EMPLOYMENT = "adp_employment"

    # Growth releases
    GDP = "gdp"
    GDP_REVISION = "gdp_revision"
    RETAIL_SALES = "retail_sales"
    DURABLE_GOODS = "durable_goods"
    HOUSING_STARTS = "housing_starts"
    EXISTING_HOME_SALES = "existing_home_sales"
    NEW_HOME_SALES = "new_home_sales"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"

    # Energy releases
    EIA_PETROLEUM = "eia_petroleum"     # Weekly Petroleum Status Report
    EIA_NATURAL_GAS = "eia_natural_gas"
    OPEC_MEETING = "opec_meeting"

    # Corporate events
    EARNINGS_RELEASE = "earnings_release"
    SEC_FILING = "sec_filing"


class EntityType(Enum):
    """Types of extracted entities"""
    COMPANY = "company"
    TICKER = "ticker"
    PERSON = "person"
    COUNTRY = "country"
    CENTRAL_BANK = "central_bank"
    COMMODITY = "commodity"
    SECTOR = "sector"
    INDEX = "index"
    MACRO_INDICATOR = "macro_indicator"


class ImportanceLevel(Enum):
    """Event importance for scheduling"""
    CRITICAL = "critical"   # FOMC, NFP, CPI - triggers mini-runs
    HIGH = "high"           # GDP, PCE, JOLTS
    MEDIUM = "medium"       # Housing, durable goods
    LOW = "low"             # Secondary indicators


class BriefType(Enum):
    """Types of generated briefs"""
    MORNING = "morning"     # 06:30 ET
    EVENING = "evening"     # 18:30 ET
    EVENT = "event"         # Post-release mini-brief


class RunStatus(Enum):
    """Pipeline run status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"     # Some sources failed
    FAILED = "failed"


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Source:
    """
    Official source registry with trust tiers.

    Determines:
    - Processing priority (official > company_ir > licensed > aggregator)
    - Rate limiting requirements
    - Category routing
    """
    source_id: str
    name: str
    provider: SourceProvider
    source_type: SourceType

    # Trust and priority
    priority_weight: float = 1.0        # Higher = more trusted

    # Rate limiting (SEC EDGAR: 10 req/sec)
    rate_limit_per_second: float = 10.0
    rate_limit_per_minute: float = 100.0

    # API configuration
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None

    # Category routing - which categories this source feeds
    default_categories: List[CategoryType] = field(default_factory=list)

    # State
    is_active: bool = True
    last_successful_fetch: Optional[datetime] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "provider": self.provider.value,
            "source_type": self.source_type.value,
            "priority_weight": self.priority_weight,
            "rate_limit_per_second": self.rate_limit_per_second,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "base_url": self.base_url,
            "default_categories": [c.value for c in self.default_categories],
            "is_active": self.is_active,
            "last_successful_fetch": self.last_successful_fetch.isoformat() if self.last_successful_fetch else None,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class Category:
    """
    8 Core Categories for macro + SP500 trading intelligence.

    Categories are not generic buckets - they are query configurations
    with deterministic routing rules.
    """
    category_id: str
    category_type: CategoryType
    name: str
    description: str

    # Ranking profile
    macro_weight: float = 0.35          # Weight for macro relevance
    sp500_weight: float = 0.25          # Weight for SP500 relevance
    source_quality_weight: float = 0.20 # Weight for source tier
    recency_weight: float = 0.10        # Weight for time decay
    corroboration_weight: float = 0.10  # Weight for multi-source confirmation

    # Brief configuration
    max_stories_per_brief: int = 10

    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category_id": self.category_id,
            "category_type": self.category_type.value,
            "name": self.name,
            "description": self.description,
            "macro_weight": self.macro_weight,
            "sp500_weight": self.sp500_weight,
            "source_quality_weight": self.source_quality_weight,
            "recency_weight": self.recency_weight,
            "corroboration_weight": self.corroboration_weight,
            "max_stories_per_brief": self.max_stories_per_brief,
            "is_active": self.is_active,
        }


@dataclass
class CategoryRule:
    """
    Deterministic classification rules for categories.

    Classification order:
    1. Source-to-category mapping (Fed pages -> FED_RATES)
    2. Keyword matching with required/exclude lists
    3. Entity-based boosts
    """
    rule_id: str
    category_id: str

    # Source-based routing (highest priority)
    source_whitelist: List[str] = field(default_factory=list)  # source_ids

    # Keyword matching
    include_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)  # Must have at least one

    # Entity matching
    required_entity_types: List[EntityType] = field(default_factory=list)
    ticker_whitelist: List[str] = field(default_factory=list)

    # Confidence thresholds
    min_confidence: float = 0.5
    keyword_match_boost: float = 0.2

    priority: int = 100  # Lower = higher priority
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category_id": self.category_id,
            "source_whitelist": self.source_whitelist,
            "include_keywords": self.include_keywords,
            "exclude_keywords": self.exclude_keywords,
            "required_keywords": self.required_keywords,
            "required_entity_types": [e.value for e in self.required_entity_types],
            "ticker_whitelist": self.ticker_whitelist,
            "min_confidence": self.min_confidence,
            "keyword_match_boost": self.keyword_match_boost,
            "priority": self.priority,
            "is_active": self.is_active,
        }


@dataclass
class ScheduledEvent:
    """
    Event calendar for macro releases and Fed events.

    This is critical - the system runs around known release schedules,
    not generic polling.
    """
    event_id: str
    source_id: str
    event_type: EventType
    category_id: str

    # Scheduling
    scheduled_at: datetime
    timezone: str = "America/New_York"

    # Importance determines mini-run triggers
    importance: ImportanceLevel = ImportanceLevel.MEDIUM

    # Pre/post run offsets for event-aware ingestion
    pre_run_offset_minutes: int = 10    # Run ingestion X minutes before
    post_run_offset_minutes: int = 5    # Run ingestion X minutes after

    # Event metadata
    title: str = ""
    description: str = ""

    # For recurring events
    is_recurring: bool = False
    recurrence_rule: Optional[str] = None  # iCal RRULE format

    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_id": self.source_id,
            "event_type": self.event_type.value,
            "category_id": self.category_id,
            "scheduled_at": self.scheduled_at.isoformat(),
            "timezone": self.timezone,
            "importance": self.importance.value,
            "pre_run_offset_minutes": self.pre_run_offset_minutes,
            "post_run_offset_minutes": self.post_run_offset_minutes,
            "title": self.title,
            "description": self.description,
            "is_recurring": self.is_recurring,
            "recurrence_rule": self.recurrence_rule,
            "is_active": self.is_active,
        }


@dataclass
class Entity:
    """Extracted entity from article/filing"""
    entity_id: str
    entity_type: EntityType
    value: str                          # The entity value (e.g., "AAPL", "Federal Reserve")

    # Normalized identifiers
    ticker: Optional[str] = None        # Stock ticker if applicable
    figi: Optional[str] = None          # Financial Instrument Global Identifier

    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type.value,
            "value": self.value,
            "ticker": self.ticker,
            "figi": self.figi,
            "confidence": self.confidence,
        }


@dataclass
class Article:
    """
    Normalized article/filing/release record.

    This is the canonical representation for all ingested items:
    - Fed releases
    - BLS/BEA data releases
    - SEC filings (8-K, 10-Q, etc.)
    - Company press releases
    - News articles
    """
    article_id: str
    source_id: str
    canonical_url: str

    # Content
    title: str
    snippet: Optional[str] = None       # Short excerpt if permitted
    content_hash: Optional[str] = None  # For deduplication

    # Timestamps
    published_at: datetime = field(default_factory=datetime.utcnow)
    ingested_at: datetime = field(default_factory=datetime.utcnow)

    # Metadata
    language: str = "en"
    country: str = "US"

    # For SEC filings
    filing_type: Optional[str] = None   # 8-K, 10-Q, 10-K, etc.
    cik: Optional[str] = None           # SEC Central Index Key
    accession_number: Optional[str] = None

    # For macro releases
    release_type: Optional[str] = None  # CPI, NFP, GDP, etc.
    release_period: Optional[str] = None  # "2026-Q1", "2026-02", etc.

    # Extracted data (populated by processing)
    entities: List[Entity] = field(default_factory=list)
    tickers: List[str] = field(default_factory=list)
    categories: List[CategoryType] = field(default_factory=list)

    # Scoring
    impact_score: float = 0.0

    # Raw metadata from source
    metadata_json: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(url: str, source_id: str) -> str:
        """Generate unique article ID from URL + source"""
        content = f"{url}|{source_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def generate_content_hash(title: str, content: str) -> str:
        """Generate content hash for deduplication"""
        return hashlib.sha256(f"{title}|{content}".encode()).hexdigest()[:32]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_id": self.article_id,
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "snippet": self.snippet,
            "content_hash": self.content_hash,
            "published_at": self.published_at.isoformat(),
            "ingested_at": self.ingested_at.isoformat(),
            "language": self.language,
            "country": self.country,
            "filing_type": self.filing_type,
            "cik": self.cik,
            "accession_number": self.accession_number,
            "release_type": self.release_type,
            "release_period": self.release_period,
            "entities": [e.to_dict() for e in self.entities],
            "tickers": self.tickers,
            "categories": [c.value for c in self.categories],
            "impact_score": self.impact_score,
            "metadata_json": self.metadata_json,
        }


@dataclass
class Story:
    """
    Clustered story combining multiple articles about the same event.

    Users see stories, not individual articles. Each story has:
    - What happened (summary)
    - Why it matters (mechanism explanation)
    - Affected assets
    - Confidence based on source diversity
    """
    story_id: str
    lead_article_id: str

    # Generated content
    headline: str
    summary: str                        # "What happened" - 1-2 sentences
    why_it_matters: str                 # "Why it matters" - mechanism explanation

    # Timestamps
    first_seen_at: datetime
    last_updated_at: datetime

    # Aggregated metadata
    tickers: List[str] = field(default_factory=list)
    affected_sectors: List[str] = field(default_factory=list)
    affected_factors: List[str] = field(default_factory=list)  # rates, oil, growth, etc.

    # Coverage
    article_ids: List[str] = field(default_factory=list)
    source_count: int = 0               # Number of independent sources

    # Scoring
    impact_score: float = 0.0
    confidence_score: float = 0.0       # Based on source diversity

    # Category scores (story can belong to multiple categories)
    category_scores: Dict[str, float] = field(default_factory=dict)

    # For tracking changes between briefs
    previous_summary: Optional[str] = None
    change_description: Optional[str] = None

    # Status
    is_developing: bool = False         # Story is still evolving

    @staticmethod
    def generate_id(lead_article_id: str, first_seen: datetime) -> str:
        content = f"{lead_article_id}|{first_seen.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "story_id": self.story_id,
            "lead_article_id": self.lead_article_id,
            "headline": self.headline,
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "tickers": self.tickers,
            "affected_sectors": self.affected_sectors,
            "affected_factors": self.affected_factors,
            "article_ids": self.article_ids,
            "source_count": self.source_count,
            "impact_score": self.impact_score,
            "confidence_score": self.confidence_score,
            "category_scores": self.category_scores,
            "previous_summary": self.previous_summary,
            "change_description": self.change_description,
            "is_developing": self.is_developing,
        }


@dataclass
class Brief:
    """
    Morning or Evening brief.

    Morning Book (06:30 ET):
    - Overnight and premarket developments
    - Today's macro calendar

    Evening Book (18:30 ET):
    - Post-close changes
    - Tomorrow's scheduled events
    """
    brief_id: str
    user_id: str
    brief_type: BriefType
    generated_at: datetime

    # Content
    stories: List[Story] = field(default_factory=list)

    # Generated sections
    executive_summary: Optional[str] = None
    macro_watch: Optional[str] = None       # Key macro events today/tomorrow
    oil_geo_watch: Optional[str] = None     # Oil/geopolitics watch
    sp500_watch: Optional[str] = None       # SP500 event watch
    calendar_summary: Optional[str] = None  # Upcoming releases

    # Stats
    new_stories_count: int = 0
    updated_stories_count: int = 0

    # Model used for generation
    model_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "user_id": self.user_id,
            "brief_type": self.brief_type.value,
            "generated_at": self.generated_at.isoformat(),
            "stories": [s.to_dict() for s in self.stories],
            "executive_summary": self.executive_summary,
            "macro_watch": self.macro_watch,
            "oil_geo_watch": self.oil_geo_watch,
            "sp500_watch": self.sp500_watch,
            "calendar_summary": self.calendar_summary,
            "new_stories_count": self.new_stories_count,
            "updated_stories_count": self.updated_stories_count,
            "model_used": self.model_used,
        }


@dataclass
class RunLog:
    """
    Pipeline execution log.

    Critical for:
    - Rate limiting tracking
    - Debugging failures
    - Performance monitoring
    """
    run_id: str
    job_name: str                       # "morning_ingest", "cpi_release", etc.

    started_at: datetime
    ended_at: Optional[datetime] = None

    status: RunStatus = RunStatus.PENDING

    # Stats
    items_fetched: int = 0
    items_processed: int = 0
    stories_created: int = 0
    stories_updated: int = 0

    # Source-level stats
    source_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Trigger info
    triggered_by: str = "scheduler"     # "scheduler", "event", "manual"
    event_id: Optional[str] = None      # If triggered by scheduled event

    def duration_seconds(self) -> Optional[float]:
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_name": self.job_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status.value,
            "items_fetched": self.items_fetched,
            "items_processed": self.items_processed,
            "stories_created": self.stories_created,
            "stories_updated": self.stories_updated,
            "source_stats": self.source_stats,
            "errors": self.errors,
            "triggered_by": self.triggered_by,
            "event_id": self.event_id,
            "duration_seconds": self.duration_seconds(),
        }


@dataclass
class UserWatchlist:
    """User ticker watchlist with weights"""
    user_id: str
    ticker: str
    weight: float = 1.0                 # Higher = more important
    added_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "ticker": self.ticker,
            "weight": self.weight,
            "added_at": self.added_at.isoformat(),
        }


# =============================================================================
# DEFAULT DATA
# =============================================================================

# Default Sources - Official Tier 1
DEFAULT_SOURCES: List[Source] = [
    Source(
        source_id="fed",
        name="Federal Reserve",
        provider=SourceProvider.FEDERAL_RESERVE,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.federalreserve.gov",
        default_categories=[CategoryType.FED_RATES],
        rate_limit_per_second=5.0,
    ),
    Source(
        source_id="bls",
        name="Bureau of Labor Statistics",
        provider=SourceProvider.BLS,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.bls.gov",
        default_categories=[CategoryType.INFLATION, CategoryType.LABOR],
        rate_limit_per_second=5.0,
    ),
    Source(
        source_id="bea",
        name="Bureau of Economic Analysis",
        provider=SourceProvider.BEA,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.bea.gov",
        default_categories=[CategoryType.GROWTH_CONSUMER, CategoryType.INFLATION],
        rate_limit_per_second=5.0,
    ),
    Source(
        source_id="census",
        name="Census Bureau",
        provider=SourceProvider.CENSUS,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.census.gov",
        default_categories=[CategoryType.GROWTH_CONSUMER],
        rate_limit_per_second=5.0,
    ),
    Source(
        source_id="eia",
        name="Energy Information Administration",
        provider=SourceProvider.EIA,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.eia.gov",
        default_categories=[CategoryType.OIL_ENERGY],
        rate_limit_per_second=5.0,
    ),
    Source(
        source_id="sec_edgar",
        name="SEC EDGAR",
        provider=SourceProvider.SEC_EDGAR,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.sec.gov/cgi-bin/browse-edgar",
        default_categories=[CategoryType.SP500_CORPORATE],
        rate_limit_per_second=10.0,  # SEC enforces 10 req/sec
        rate_limit_per_minute=100.0,
    ),
    Source(
        source_id="treasury",
        name="US Treasury / OFAC",
        provider=SourceProvider.TREASURY,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://home.treasury.gov",
        default_categories=[CategoryType.GEOPOLITICS, CategoryType.FED_RATES],
        rate_limit_per_second=5.0,
    ),
    Source(
        source_id="opec",
        name="OPEC",
        provider=SourceProvider.OPEC,
        source_type=SourceType.OFFICIAL,
        priority_weight=1.0,
        base_url="https://www.opec.org",
        default_categories=[CategoryType.OIL_ENERGY],
        rate_limit_per_second=2.0,
    ),
]


# Default Categories - 8 Core
DEFAULT_CATEGORIES: List[Category] = [
    Category(
        category_id="fed_rates",
        category_type=CategoryType.FED_RATES,
        name="Fed & Rates",
        description="FOMC decisions, Fed speeches, Treasury yields, monetary policy",
        macro_weight=0.40,
        sp500_weight=0.20,
    ),
    Category(
        category_id="inflation",
        category_type=CategoryType.INFLATION,
        name="Inflation",
        description="CPI, PCE, shelter inflation, services inflation, disinflation trends",
        macro_weight=0.45,
        sp500_weight=0.15,
    ),
    Category(
        category_id="labor",
        category_type=CategoryType.LABOR,
        name="Labor Market",
        description="Nonfarm payrolls, unemployment, JOLTS, wage growth",
        macro_weight=0.40,
        sp500_weight=0.20,
    ),
    Category(
        category_id="growth_consumer",
        category_type=CategoryType.GROWTH_CONSUMER,
        name="Growth & Consumer",
        description="GDP, retail sales, housing, durable goods, consumer spending",
        macro_weight=0.35,
        sp500_weight=0.25,
    ),
    Category(
        category_id="oil_energy",
        category_type=CategoryType.OIL_ENERGY,
        name="Oil & Energy",
        description="EIA inventories, OPEC decisions, refinery data, energy sanctions",
        macro_weight=0.30,
        sp500_weight=0.30,
    ),
    Category(
        category_id="geopolitics",
        category_type=CategoryType.GEOPOLITICS,
        name="Geopolitics",
        description="Sanctions, conflicts with market transmission (oil, shipping, semis, defense)",
        macro_weight=0.25,
        sp500_weight=0.35,
    ),
    Category(
        category_id="sp500_corporate",
        category_type=CategoryType.SP500_CORPORATE,
        name="SP500 Corporate Events",
        description="Earnings, 8-K filings, M&A, guidance changes, management",
        macro_weight=0.15,
        sp500_weight=0.45,
    ),
    Category(
        category_id="market_regime",
        category_type=CategoryType.MARKET_REGIME,
        name="Market Regime",
        description="Derived: risk-on/off, sector rotation, factor leadership",
        macro_weight=0.30,
        sp500_weight=0.30,
    ),
]


# Default Category Rules
DEFAULT_CATEGORY_RULES: List[CategoryRule] = [
    # Fed & Rates - Source-based routing
    CategoryRule(
        rule_id="fed_source",
        category_id="fed_rates",
        source_whitelist=["fed"],
        priority=10,
    ),
    CategoryRule(
        rule_id="fed_keywords",
        category_id="fed_rates",
        include_keywords=["fomc", "federal reserve", "powell", "fed chair", "dot plot",
                         "policy statement", "fed minutes", "balance sheet", "treasury yield",
                         "rate hike", "rate cut", "monetary policy", "quantitative tightening"],
        exclude_keywords=["fedex"],
        priority=20,
    ),

    # Inflation
    CategoryRule(
        rule_id="inflation_keywords",
        category_id="inflation",
        include_keywords=["cpi", "consumer price index", "pce", "personal consumption expenditure",
                         "inflation", "core inflation", "shelter inflation", "services inflation",
                         "disinflation", "deflation", "price index"],
        required_keywords=["inflation", "cpi", "pce", "price index"],
        priority=20,
    ),

    # Labor
    CategoryRule(
        rule_id="labor_source",
        category_id="labor",
        source_whitelist=["bls"],
        include_keywords=["employment", "payroll", "jobs", "unemployment", "jolts", "labor"],
        priority=10,
    ),
    CategoryRule(
        rule_id="labor_keywords",
        category_id="labor",
        include_keywords=["nonfarm payrolls", "unemployment rate", "jobless claims", "jolts",
                         "average hourly earnings", "labor market", "job openings", "hiring",
                         "layoffs", "wage growth"],
        priority=20,
    ),

    # Growth & Consumer
    CategoryRule(
        rule_id="growth_source_bea",
        category_id="growth_consumer",
        source_whitelist=["bea"],
        priority=10,
    ),
    CategoryRule(
        rule_id="growth_source_census",
        category_id="growth_consumer",
        source_whitelist=["census"],
        priority=10,
    ),
    CategoryRule(
        rule_id="growth_keywords",
        category_id="growth_consumer",
        include_keywords=["gdp", "gross domestic product", "retail sales", "consumer spending",
                         "durable goods", "housing starts", "building permits", "new home sales",
                         "existing home sales", "consumer confidence", "personal income",
                         "trade deficit", "trade balance", "inventories"],
        priority=20,
    ),

    # Oil & Energy
    CategoryRule(
        rule_id="oil_source_eia",
        category_id="oil_energy",
        source_whitelist=["eia"],
        priority=10,
    ),
    CategoryRule(
        rule_id="oil_source_opec",
        category_id="oil_energy",
        source_whitelist=["opec"],
        priority=10,
    ),
    CategoryRule(
        rule_id="oil_keywords",
        category_id="oil_energy",
        include_keywords=["crude oil", "brent", "wti", "opec", "petroleum", "refinery",
                         "gasoline", "distillates", "oil inventory", "oil production",
                         "natural gas", "lng", "energy sanctions", "oil supply",
                         "production cut", "strait of hormuz", "oil tanker"],
        priority=20,
    ),

    # Geopolitics (must have market transmission)
    CategoryRule(
        rule_id="geopolitics_source",
        category_id="geopolitics",
        source_whitelist=["treasury", "state_dept"],
        include_keywords=["sanction", "tariff", "export control"],
        priority=10,
    ),
    CategoryRule(
        rule_id="geopolitics_keywords",
        category_id="geopolitics",
        include_keywords=["sanctions", "tariff", "trade war", "export controls", "embargo",
                         "military", "conflict", "invasion", "shipping disruption",
                         "strait", "pipeline", "defense", "risk-off"],
        # Must be linked to market-relevant factors
        required_keywords=["oil", "shipping", "semiconductor", "defense", "sanction",
                          "tariff", "treasury", "dollar", "yield", "risk"],
        priority=20,
    ),

    # SP500 Corporate Events
    CategoryRule(
        rule_id="sp500_source_sec",
        category_id="sp500_corporate",
        source_whitelist=["sec_edgar"],
        priority=10,
    ),
    CategoryRule(
        rule_id="sp500_keywords",
        category_id="sp500_corporate",
        include_keywords=["earnings", "eps", "revenue", "guidance", "8-k", "10-q", "10-k",
                         "quarterly results", "beat estimates", "miss estimates",
                         "acquisition", "merger", "ceo", "management change", "buyback",
                         "dividend", "spinoff", "investigation", "sec filing"],
        priority=20,
    ),
]
