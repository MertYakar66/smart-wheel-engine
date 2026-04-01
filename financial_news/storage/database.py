"""
SQLite Database Implementation for Macro + SP500 Event Intelligence System

Implements the canonical schema with:
- Sources table with trust tiers
- Categories table with 8 core categories
- Event calendar for macro releases
- Articles, stories, briefs
- Run logs for rate limiting and debugging
"""

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from financial_news.schema import (
    DEFAULT_CATEGORIES,
    DEFAULT_CATEGORY_RULES,
    DEFAULT_SOURCES,
    Article,
    Category,
    CategoryRule,
    CategoryType,
    Entity,
    EntityType,
    EventType,
    ImportanceLevel,
    RunLog,
    RunStatus,
    ScheduledEvent,
    Source,
    SourceProvider,
    SourceType,
    Story,
)

logger = logging.getLogger(__name__)


class NewsDatabase:
    """
    SQLite database for the news intelligence system.

    Uses SQLite for simplicity - can be migrated to PostgreSQL for production.
    """

    def __init__(self, db_path: str = "data/news_intel.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    priority_weight REAL DEFAULT 1.0,
                    rate_limit_per_second REAL DEFAULT 10.0,
                    rate_limit_per_minute REAL DEFAULT 100.0,
                    base_url TEXT,
                    api_key_env_var TEXT,
                    default_categories TEXT,
                    is_active INTEGER DEFAULT 1,
                    last_successful_fetch TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Categories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    category_id TEXT PRIMARY KEY,
                    category_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    macro_weight REAL DEFAULT 0.35,
                    sp500_weight REAL DEFAULT 0.25,
                    source_quality_weight REAL DEFAULT 0.20,
                    recency_weight REAL DEFAULT 0.10,
                    corroboration_weight REAL DEFAULT 0.10,
                    max_stories_per_brief INTEGER DEFAULT 10,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Category rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS category_rules (
                    rule_id TEXT PRIMARY KEY,
                    category_id TEXT NOT NULL,
                    source_whitelist TEXT,
                    include_keywords TEXT,
                    exclude_keywords TEXT,
                    required_keywords TEXT,
                    required_entity_types TEXT,
                    ticker_whitelist TEXT,
                    min_confidence REAL DEFAULT 0.5,
                    keyword_match_boost REAL DEFAULT 0.2,
                    priority INTEGER DEFAULT 100,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (category_id) REFERENCES categories(category_id)
                )
            """)

            # Event calendar table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_calendar (
                    event_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    category_id TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    timezone TEXT DEFAULT 'America/New_York',
                    importance TEXT DEFAULT 'medium',
                    pre_run_offset_minutes INTEGER DEFAULT 10,
                    post_run_offset_minutes INTEGER DEFAULT 5,
                    title TEXT,
                    description TEXT,
                    is_recurring INTEGER DEFAULT 0,
                    recurrence_rule TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(source_id),
                    FOREIGN KEY (category_id) REFERENCES categories(category_id)
                )
            """)

            # Articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    article_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    canonical_url TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    snippet TEXT,
                    content_hash TEXT,
                    published_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    country TEXT DEFAULT 'US',
                    filing_type TEXT,
                    cik TEXT,
                    accession_number TEXT,
                    release_type TEXT,
                    release_period TEXT,
                    categories TEXT,
                    tickers TEXT,
                    impact_score REAL DEFAULT 0.0,
                    metadata_json TEXT,
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                )
            """)

            # Article entities (M2M)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS article_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    ticker TEXT,
                    figi TEXT,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (article_id) REFERENCES articles(article_id),
                    UNIQUE(article_id, entity_type, value)
                )
            """)

            # Stories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    story_id TEXT PRIMARY KEY,
                    lead_article_id TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    summary TEXT,
                    why_it_matters TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_updated_at TEXT NOT NULL,
                    tickers TEXT,
                    affected_sectors TEXT,
                    affected_factors TEXT,
                    source_count INTEGER DEFAULT 1,
                    impact_score REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    category_scores TEXT,
                    previous_summary TEXT,
                    change_description TEXT,
                    is_developing INTEGER DEFAULT 0,
                    FOREIGN KEY (lead_article_id) REFERENCES articles(article_id)
                )
            """)

            # Story articles (M2M)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS story_articles (
                    story_id TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (story_id, article_id),
                    FOREIGN KEY (story_id) REFERENCES stories(story_id),
                    FOREIGN KEY (article_id) REFERENCES articles(article_id)
                )
            """)

            # Story categories (M2M with scores)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS story_categories (
                    story_id TEXT NOT NULL,
                    category_id TEXT NOT NULL,
                    score REAL DEFAULT 0.0,
                    PRIMARY KEY (story_id, category_id),
                    FOREIGN KEY (story_id) REFERENCES stories(story_id),
                    FOREIGN KEY (category_id) REFERENCES categories(category_id)
                )
            """)

            # Briefs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS briefs (
                    brief_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    brief_type TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    story_ids TEXT,
                    executive_summary TEXT,
                    macro_watch TEXT,
                    oil_geo_watch TEXT,
                    sp500_watch TEXT,
                    calendar_summary TEXT,
                    new_stories_count INTEGER DEFAULT 0,
                    updated_stories_count INTEGER DEFAULT 0,
                    model_used TEXT
                )
            """)

            # Run logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_logs (
                    run_id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT DEFAULT 'pending',
                    items_fetched INTEGER DEFAULT 0,
                    items_processed INTEGER DEFAULT 0,
                    stories_created INTEGER DEFAULT 0,
                    stories_updated INTEGER DEFAULT 0,
                    source_stats TEXT,
                    errors TEXT,
                    triggered_by TEXT DEFAULT 'scheduler',
                    event_id TEXT
                )
            """)

            # User watchlists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_watchlists (
                    user_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, ticker)
                )
            """)

            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_articles_content_hash ON articles(content_hash)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_calendar_scheduled ON event_calendar(scheduled_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_stories_updated ON stories(last_updated_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_logs_started ON run_logs(started_at)"
            )

            conn.commit()

            # Initialize default data
            self._init_default_data(cursor, conn)

    def _init_default_data(self, cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
        """Initialize default sources, categories, and rules"""
        # Check if already initialized
        cursor.execute("SELECT COUNT(*) FROM sources")
        if cursor.fetchone()[0] > 0:
            return

        # Insert default sources
        for source in DEFAULT_SOURCES:
            cursor.execute(
                """
                INSERT OR IGNORE INTO sources
                (source_id, name, provider, source_type, priority_weight,
                 rate_limit_per_second, rate_limit_per_minute, base_url,
                 default_categories, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    source.source_id,
                    source.name,
                    source.provider.value,
                    source.source_type.value,
                    source.priority_weight,
                    source.rate_limit_per_second,
                    source.rate_limit_per_minute,
                    source.base_url,
                    json.dumps([c.value for c in source.default_categories]),
                    1 if source.is_active else 0,
                ),
            )

        # Insert default categories
        for category in DEFAULT_CATEGORIES:
            cursor.execute(
                """
                INSERT OR IGNORE INTO categories
                (category_id, category_type, name, description,
                 macro_weight, sp500_weight, source_quality_weight,
                 recency_weight, corroboration_weight, max_stories_per_brief, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    category.category_id,
                    category.category_type.value,
                    category.name,
                    category.description,
                    category.macro_weight,
                    category.sp500_weight,
                    category.source_quality_weight,
                    category.recency_weight,
                    category.corroboration_weight,
                    category.max_stories_per_brief,
                    1 if category.is_active else 0,
                ),
            )

        # Insert default category rules
        for rule in DEFAULT_CATEGORY_RULES:
            cursor.execute(
                """
                INSERT OR IGNORE INTO category_rules
                (rule_id, category_id, source_whitelist, include_keywords,
                 exclude_keywords, required_keywords, required_entity_types,
                 ticker_whitelist, min_confidence, keyword_match_boost, priority, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rule.rule_id,
                    rule.category_id,
                    json.dumps(rule.source_whitelist),
                    json.dumps(rule.include_keywords),
                    json.dumps(rule.exclude_keywords),
                    json.dumps(rule.required_keywords),
                    json.dumps([e.value for e in rule.required_entity_types]),
                    json.dumps(rule.ticker_whitelist),
                    rule.min_confidence,
                    rule.keyword_match_boost,
                    rule.priority,
                    1 if rule.is_active else 0,
                ),
            )

        conn.commit()
        logger.info(
            f"Initialized database with {len(DEFAULT_SOURCES)} sources, "
            f"{len(DEFAULT_CATEGORIES)} categories, {len(DEFAULT_CATEGORY_RULES)} rules"
        )

    # =========================================================================
    # SOURCE OPERATIONS
    # =========================================================================

    def get_source(self, source_id: str) -> Source | None:
        """Get a source by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_source(row)
        return None

    def get_all_sources(self, active_only: bool = True) -> list[Source]:
        """Get all sources"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute(
                    "SELECT * FROM sources WHERE is_active = 1 ORDER BY priority_weight DESC"
                )
            else:
                cursor.execute("SELECT * FROM sources ORDER BY priority_weight DESC")
            return [self._row_to_source(row) for row in cursor.fetchall()]

    def update_source_fetch_status(self, source_id: str, success: bool) -> None:
        """Update source after fetch attempt"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if success:
                cursor.execute(
                    """
                    UPDATE sources
                    SET last_successful_fetch = ?, consecutive_failures = 0
                    WHERE source_id = ?
                """,
                    (datetime.utcnow().isoformat(), source_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE sources
                    SET consecutive_failures = consecutive_failures + 1
                    WHERE source_id = ?
                """,
                    (source_id,),
                )
            conn.commit()

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        """Convert database row to Source object"""
        return Source(
            source_id=row["source_id"],
            name=row["name"],
            provider=SourceProvider(row["provider"]),
            source_type=SourceType(row["source_type"]),
            priority_weight=row["priority_weight"],
            rate_limit_per_second=row["rate_limit_per_second"],
            rate_limit_per_minute=row["rate_limit_per_minute"],
            base_url=row["base_url"],
            default_categories=[
                CategoryType(c) for c in json.loads(row["default_categories"] or "[]")
            ],
            is_active=bool(row["is_active"]),
            last_successful_fetch=datetime.fromisoformat(row["last_successful_fetch"])
            if row["last_successful_fetch"]
            else None,
            consecutive_failures=row["consecutive_failures"],
        )

    # =========================================================================
    # CATEGORY OPERATIONS
    # =========================================================================

    def get_category(self, category_id: str) -> Category | None:
        """Get a category by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories WHERE category_id = ?", (category_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_category(row)
        return None

    def get_all_categories(self, active_only: bool = True) -> list[Category]:
        """Get all categories"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute("SELECT * FROM categories WHERE is_active = 1")
            else:
                cursor.execute("SELECT * FROM categories")
            return [self._row_to_category(row) for row in cursor.fetchall()]

    def get_category_rules(self, category_id: str) -> list[CategoryRule]:
        """Get classification rules for a category"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM category_rules
                WHERE category_id = ? AND is_active = 1
                ORDER BY priority ASC
            """,
                (category_id,),
            )
            return [self._row_to_category_rule(row) for row in cursor.fetchall()]

    def get_all_category_rules(self) -> list[CategoryRule]:
        """Get all active category rules"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM category_rules WHERE is_active = 1 ORDER BY priority ASC")
            return [self._row_to_category_rule(row) for row in cursor.fetchall()]

    def _row_to_category(self, row: sqlite3.Row) -> Category:
        """Convert database row to Category object"""
        return Category(
            category_id=row["category_id"],
            category_type=CategoryType(row["category_type"]),
            name=row["name"],
            description=row["description"],
            macro_weight=row["macro_weight"],
            sp500_weight=row["sp500_weight"],
            source_quality_weight=row["source_quality_weight"],
            recency_weight=row["recency_weight"],
            corroboration_weight=row["corroboration_weight"],
            max_stories_per_brief=row["max_stories_per_brief"],
            is_active=bool(row["is_active"]),
        )

    def _row_to_category_rule(self, row: sqlite3.Row) -> CategoryRule:
        """Convert database row to CategoryRule object"""
        return CategoryRule(
            rule_id=row["rule_id"],
            category_id=row["category_id"],
            source_whitelist=json.loads(row["source_whitelist"] or "[]"),
            include_keywords=json.loads(row["include_keywords"] or "[]"),
            exclude_keywords=json.loads(row["exclude_keywords"] or "[]"),
            required_keywords=json.loads(row["required_keywords"] or "[]"),
            required_entity_types=[
                EntityType(e) for e in json.loads(row["required_entity_types"] or "[]")
            ],
            ticker_whitelist=json.loads(row["ticker_whitelist"] or "[]"),
            min_confidence=row["min_confidence"],
            keyword_match_boost=row["keyword_match_boost"],
            priority=row["priority"],
            is_active=bool(row["is_active"]),
        )

    # =========================================================================
    # EVENT CALENDAR OPERATIONS
    # =========================================================================

    def add_scheduled_event(self, event: ScheduledEvent) -> None:
        """Add a scheduled event to the calendar"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO event_calendar
                (event_id, source_id, event_type, category_id, scheduled_at,
                 timezone, importance, pre_run_offset_minutes, post_run_offset_minutes,
                 title, description, is_recurring, recurrence_rule, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.source_id,
                    event.event_type.value,
                    event.category_id,
                    event.scheduled_at.isoformat(),
                    event.timezone,
                    event.importance.value,
                    event.pre_run_offset_minutes,
                    event.post_run_offset_minutes,
                    event.title,
                    event.description,
                    1 if event.is_recurring else 0,
                    event.recurrence_rule,
                    1 if event.is_active else 0,
                ),
            )
            conn.commit()

    def get_upcoming_events(self, hours: int = 24) -> list[ScheduledEvent]:
        """Get events scheduled in the next N hours"""
        now = datetime.utcnow()
        end = now + timedelta(hours=hours)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM event_calendar
                WHERE is_active = 1
                  AND scheduled_at >= ?
                  AND scheduled_at <= ?
                ORDER BY scheduled_at ASC
            """,
                (now.isoformat(), end.isoformat()),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_events_needing_prerun(self, minutes_ahead: int = 15) -> list[ScheduledEvent]:
        """Get events that need pre-run ingestion"""
        now = datetime.utcnow()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Get events where we're within the pre-run window
            cursor.execute(
                """
                SELECT * FROM event_calendar
                WHERE is_active = 1
                  AND datetime(scheduled_at, '-' || pre_run_offset_minutes || ' minutes') <= ?
                  AND scheduled_at > ?
                ORDER BY scheduled_at ASC
            """,
                (now.isoformat(), now.isoformat()),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_events_needing_postrun(self, minutes_since: int = 30) -> list[ScheduledEvent]:
        """Get events that need post-run ingestion"""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=minutes_since)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM event_calendar
                WHERE is_active = 1
                  AND scheduled_at >= ?
                  AND scheduled_at <= ?
                ORDER BY scheduled_at ASC
            """,
                (window_start.isoformat(), now.isoformat()),
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]

    def _row_to_event(self, row: sqlite3.Row) -> ScheduledEvent:
        """Convert database row to ScheduledEvent object"""
        return ScheduledEvent(
            event_id=row["event_id"],
            source_id=row["source_id"],
            event_type=EventType(row["event_type"]),
            category_id=row["category_id"],
            scheduled_at=datetime.fromisoformat(row["scheduled_at"]),
            timezone=row["timezone"],
            importance=ImportanceLevel(row["importance"]),
            pre_run_offset_minutes=row["pre_run_offset_minutes"],
            post_run_offset_minutes=row["post_run_offset_minutes"],
            title=row["title"],
            description=row["description"],
            is_recurring=bool(row["is_recurring"]),
            recurrence_rule=row["recurrence_rule"],
            is_active=bool(row["is_active"]),
        )

    # =========================================================================
    # ARTICLE OPERATIONS
    # =========================================================================

    def save_article(self, article: Article) -> None:
        """Save an article to the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO articles
                (article_id, source_id, canonical_url, title, snippet, content_hash,
                 published_at, ingested_at, language, country, filing_type, cik,
                 accession_number, release_type, release_period, categories, tickers,
                 impact_score, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    article.article_id,
                    article.source_id,
                    article.canonical_url,
                    article.title,
                    article.snippet,
                    article.content_hash,
                    article.published_at.isoformat(),
                    article.ingested_at.isoformat(),
                    article.language,
                    article.country,
                    article.filing_type,
                    article.cik,
                    article.accession_number,
                    article.release_type,
                    article.release_period,
                    json.dumps([c.value for c in article.categories]),
                    json.dumps(article.tickers),
                    article.impact_score,
                    json.dumps(article.metadata_json),
                ),
            )

            # Save entities
            for entity in article.entities:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO article_entities
                    (article_id, entity_type, value, ticker, figi, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        article.article_id,
                        entity.entity_type.value,
                        entity.value,
                        entity.ticker,
                        entity.figi,
                        entity.confidence,
                    ),
                )

            conn.commit()

    def save_articles(self, articles: list[Article]) -> int:
        """Save multiple articles, returns count saved"""
        count = 0
        for article in articles:
            try:
                self.save_article(article)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save article {article.article_id}: {e}")
        return count

    def get_article(self, article_id: str) -> Article | None:
        """Get an article by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles WHERE article_id = ?", (article_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_article(row, cursor)
        return None

    def get_recent_articles(self, hours: int = 24, source_id: str | None = None) -> list[Article]:
        """Get recent articles"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if source_id:
                cursor.execute(
                    """
                    SELECT * FROM articles
                    WHERE published_at >= ? AND source_id = ?
                    ORDER BY published_at DESC
                """,
                    (cutoff, source_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM articles
                    WHERE published_at >= ?
                    ORDER BY published_at DESC
                """,
                    (cutoff,),
                )
            return [self._row_to_article(row, cursor) for row in cursor.fetchall()]

    def article_exists(self, canonical_url: str) -> bool:
        """Check if article already exists by URL"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM articles WHERE canonical_url = ?", (canonical_url,))
            return cursor.fetchone() is not None

    def _row_to_article(self, row: sqlite3.Row, cursor: sqlite3.Cursor) -> Article:
        """Convert database row to Article object"""
        # Get entities
        cursor.execute(
            """
            SELECT * FROM article_entities WHERE article_id = ?
        """,
            (row["article_id"],),
        )
        entities = [
            Entity(
                entity_id=f"{row['article_id']}_{e['entity_type']}_{e['value']}",
                entity_type=EntityType(e["entity_type"]),
                value=e["value"],
                ticker=e["ticker"],
                figi=e["figi"],
                confidence=e["confidence"],
            )
            for e in cursor.fetchall()
        ]

        return Article(
            article_id=row["article_id"],
            source_id=row["source_id"],
            canonical_url=row["canonical_url"],
            title=row["title"],
            snippet=row["snippet"],
            content_hash=row["content_hash"],
            published_at=datetime.fromisoformat(row["published_at"]),
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            language=row["language"],
            country=row["country"],
            filing_type=row["filing_type"],
            cik=row["cik"],
            accession_number=row["accession_number"],
            release_type=row["release_type"],
            release_period=row["release_period"],
            entities=entities,
            tickers=json.loads(row["tickers"] or "[]"),
            categories=[CategoryType(c) for c in json.loads(row["categories"] or "[]")],
            impact_score=row["impact_score"],
            metadata_json=json.loads(row["metadata_json"] or "{}"),
        )

    # =========================================================================
    # STORY OPERATIONS
    # =========================================================================

    def save_story(self, story: Story) -> None:
        """Save a story to the database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO stories
                (story_id, lead_article_id, headline, summary, why_it_matters,
                 first_seen_at, last_updated_at, tickers, affected_sectors,
                 affected_factors, source_count, impact_score, confidence_score,
                 category_scores, previous_summary, change_description, is_developing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    story.story_id,
                    story.lead_article_id,
                    story.headline,
                    story.summary,
                    story.why_it_matters,
                    story.first_seen_at.isoformat(),
                    story.last_updated_at.isoformat(),
                    json.dumps(story.tickers),
                    json.dumps(story.affected_sectors),
                    json.dumps(story.affected_factors),
                    story.source_count,
                    story.impact_score,
                    story.confidence_score,
                    json.dumps(story.category_scores),
                    story.previous_summary,
                    story.change_description,
                    1 if story.is_developing else 0,
                ),
            )

            # Save story-article mappings
            for article_id in story.article_ids:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO story_articles (story_id, article_id)
                    VALUES (?, ?)
                """,
                    (story.story_id, article_id),
                )

            # Save story-category scores
            for category_id, score in story.category_scores.items():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO story_categories (story_id, category_id, score)
                    VALUES (?, ?, ?)
                """,
                    (story.story_id, category_id, score),
                )

            conn.commit()

    def get_recent_stories(self, hours: int = 24, category_id: str | None = None) -> list[Story]:
        """Get recent stories, optionally filtered by category"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            if category_id:
                cursor.execute(
                    """
                    SELECT s.* FROM stories s
                    JOIN story_categories sc ON s.story_id = sc.story_id
                    WHERE s.last_updated_at >= ? AND sc.category_id = ?
                    ORDER BY s.impact_score DESC, s.last_updated_at DESC
                """,
                    (cutoff, category_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM stories
                    WHERE last_updated_at >= ?
                    ORDER BY impact_score DESC, last_updated_at DESC
                """,
                    (cutoff,),
                )
            return [self._row_to_story(row, cursor) for row in cursor.fetchall()]

    def _row_to_story(self, row: sqlite3.Row, cursor: sqlite3.Cursor) -> Story:
        """Convert database row to Story object"""
        # Get article IDs
        cursor.execute(
            "SELECT article_id FROM story_articles WHERE story_id = ?", (row["story_id"],)
        )
        article_ids = [r["article_id"] for r in cursor.fetchall()]

        return Story(
            story_id=row["story_id"],
            lead_article_id=row["lead_article_id"],
            headline=row["headline"],
            summary=row["summary"],
            why_it_matters=row["why_it_matters"],
            first_seen_at=datetime.fromisoformat(row["first_seen_at"]),
            last_updated_at=datetime.fromisoformat(row["last_updated_at"]),
            tickers=json.loads(row["tickers"] or "[]"),
            affected_sectors=json.loads(row["affected_sectors"] or "[]"),
            affected_factors=json.loads(row["affected_factors"] or "[]"),
            article_ids=article_ids,
            source_count=row["source_count"],
            impact_score=row["impact_score"],
            confidence_score=row["confidence_score"],
            category_scores=json.loads(row["category_scores"] or "{}"),
            previous_summary=row["previous_summary"],
            change_description=row["change_description"],
            is_developing=bool(row["is_developing"]),
        )

    # =========================================================================
    # RUN LOG OPERATIONS
    # =========================================================================

    def create_run_log(
        self,
        run_id: str,
        job_name: str,
        triggered_by: str = "scheduler",
        event_id: str | None = None,
    ) -> RunLog:
        """Create a new run log entry"""
        run_log = RunLog(
            run_id=run_id,
            job_name=job_name,
            started_at=datetime.utcnow(),
            triggered_by=triggered_by,
            event_id=event_id,
            status=RunStatus.RUNNING,
        )

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO run_logs
                (run_id, job_name, started_at, status, triggered_by, event_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    run_log.run_id,
                    run_log.job_name,
                    run_log.started_at.isoformat(),
                    run_log.status.value,
                    run_log.triggered_by,
                    run_log.event_id,
                ),
            )
            conn.commit()

        return run_log

    def update_run_log(self, run_log: RunLog) -> None:
        """Update a run log entry"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE run_logs
                SET ended_at = ?, status = ?, items_fetched = ?, items_processed = ?,
                    stories_created = ?, stories_updated = ?, source_stats = ?, errors = ?
                WHERE run_id = ?
            """,
                (
                    run_log.ended_at.isoformat() if run_log.ended_at else None,
                    run_log.status.value,
                    run_log.items_fetched,
                    run_log.items_processed,
                    run_log.stories_created,
                    run_log.stories_updated,
                    json.dumps(run_log.source_stats),
                    json.dumps(run_log.errors),
                    run_log.run_id,
                ),
            )
            conn.commit()

    def get_recent_runs(self, hours: int = 24) -> list[RunLog]:
        """Get recent run logs"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM run_logs
                WHERE started_at >= ?
                ORDER BY started_at DESC
            """,
                (cutoff,),
            )
            return [self._row_to_run_log(row) for row in cursor.fetchall()]

    def _row_to_run_log(self, row: sqlite3.Row) -> RunLog:
        """Convert database row to RunLog object"""
        return RunLog(
            run_id=row["run_id"],
            job_name=row["job_name"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            status=RunStatus(row["status"]),
            items_fetched=row["items_fetched"],
            items_processed=row["items_processed"],
            stories_created=row["stories_created"],
            stories_updated=row["stories_updated"],
            source_stats=json.loads(row["source_stats"] or "{}"),
            errors=json.loads(row["errors"] or "[]"),
            triggered_by=row["triggered_by"],
            event_id=row["event_id"],
        )

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {}

            cursor.execute("SELECT COUNT(*) FROM sources WHERE is_active = 1")
            stats["active_sources"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM categories WHERE is_active = 1")
            stats["active_categories"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM articles")
            stats["total_articles"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM stories")
            stats["total_stories"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM event_calendar WHERE is_active = 1")
            stats["scheduled_events"] = cursor.fetchone()[0]

            # Recent activity
            cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM articles WHERE ingested_at >= ?", (cutoff_24h,))
            stats["articles_24h"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM run_logs WHERE started_at >= ?", (cutoff_24h,))
            stats["runs_24h"] = cursor.fetchone()[0]

            return stats
