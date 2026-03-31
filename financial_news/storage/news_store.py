"""
News Store - Persistent storage for the Financial News Platform

Uses SQLite for simplicity and portability.
Designed for compliance with content licensing restrictions:
- Stores metadata + URLs only
- No caching of full article text from restricted sources
- Automatic cleanup of old data
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging

from financial_news.models import (
    Article, Story, Category, UserProfile, Brief,
    ArticleSource, TopicCategory, Entity, DEFAULT_CATEGORIES
)

logger = logging.getLogger(__name__)


class NewsStore:
    """
    SQLite-based storage for news platform.

    Tables:
    - articles: Article metadata
    - stories: Clustered stories
    - categories: Category configurations
    - users: User profiles
    - briefs: Generated briefs
    """

    def __init__(self, db_path: str = "data/news_store.db"):
        """
        Initialize news store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self) -> None:
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    article_id TEXT PRIMARY KEY,
                    canonical_url TEXT UNIQUE,
                    source TEXT,
                    source_name TEXT,
                    published_at_utc TEXT,
                    title TEXT,
                    snippet TEXT,
                    language TEXT DEFAULT 'en',
                    country TEXT DEFAULT 'US',
                    entities_json TEXT,
                    tickers_json TEXT,
                    topics_json TEXT,
                    impact_score REAL DEFAULT 0.0,
                    retrieval_provider TEXT,
                    fetched_at_utc TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Stories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stories (
                    story_id TEXT PRIMARY KEY,
                    lead_article_id TEXT,
                    headline TEXT,
                    summary TEXT,
                    why_it_matters TEXT,
                    first_seen_at TEXT,
                    last_updated_at TEXT,
                    entities_json TEXT,
                    tickers_json TEXT,
                    topics_json TEXT,
                    regions_json TEXT,
                    article_ids_json TEXT,
                    source_count INTEGER DEFAULT 1,
                    impact_score REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    previous_summary TEXT,
                    change_description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Categories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    category_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    config_json TEXT,
                    is_active INTEGER DEFAULT 1,
                    last_successful_fetch TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT,
                    config_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Briefs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS briefs (
                    brief_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    brief_type TEXT,
                    generated_at TEXT,
                    content_json TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)

            # Indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_articles_published
                ON articles(published_at_utc)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stories_updated
                ON stories(last_updated_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_briefs_user
                ON briefs(user_id, generated_at)
            """)

            # Insert default categories if not exist
            for cat in DEFAULT_CATEGORIES:
                cursor.execute("""
                    INSERT OR IGNORE INTO categories (category_id, name, description, config_json)
                    VALUES (?, ?, ?, ?)
                """, (cat.category_id, cat.name, cat.description, json.dumps(cat.to_dict())))

            logger.info(f"Database initialized at {self.db_path}")

    # ==================== Articles ====================

    def save_article(self, article: Article) -> None:
        """Save article to database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO articles
                (article_id, canonical_url, source, source_name, published_at_utc,
                 title, snippet, language, country, entities_json, tickers_json,
                 topics_json, impact_score, retrieval_provider, fetched_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                article.article_id,
                article.canonical_url,
                article.source.value,
                article.source_name,
                article.published_at_utc.isoformat(),
                article.title,
                article.snippet,
                article.language,
                article.country,
                json.dumps([e.to_dict() for e in article.entities]),
                json.dumps(article.tickers),
                json.dumps([t.value for t in article.topics]),
                article.impact_score,
                article.retrieval_provider,
                article.fetched_at_utc.isoformat(),
            ))

    def save_articles(self, articles: List[Article]) -> int:
        """Save multiple articles"""
        count = 0
        for article in articles:
            try:
                self.save_article(article)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save article {article.article_id}: {e}")
        return count

    def get_article(self, article_id: str) -> Optional[Article]:
        """Get article by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles WHERE article_id = ?", (article_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_article(row)
        return None

    def get_articles_by_ids(self, article_ids: List[str]) -> List[Article]:
        """Get multiple articles by IDs"""
        if not article_ids:
            return []

        placeholders = ",".join("?" * len(article_ids))
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM articles WHERE article_id IN ({placeholders})",
                article_ids
            )
            return [self._row_to_article(row) for row in cursor.fetchall()]

    def get_recent_articles(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[Article]:
        """Get articles within time range"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM articles
                WHERE published_at_utc BETWEEN ? AND ?
                ORDER BY published_at_utc DESC
                LIMIT ?
            """, (start_time.isoformat(), end_time.isoformat(), limit))
            return [self._row_to_article(row) for row in cursor.fetchall()]

    def _row_to_article(self, row: sqlite3.Row) -> Article:
        """Convert database row to Article object"""
        entities = [Entity(**e) for e in json.loads(row["entities_json"] or "[]")]
        topics = [TopicCategory(t) for t in json.loads(row["topics_json"] or "[]")]

        return Article(
            article_id=row["article_id"],
            canonical_url=row["canonical_url"],
            source=ArticleSource(row["source"]),
            source_name=row["source_name"],
            published_at_utc=datetime.fromisoformat(row["published_at_utc"]),
            title=row["title"],
            snippet=row["snippet"],
            language=row["language"],
            country=row["country"],
            entities=entities,
            tickers=json.loads(row["tickers_json"] or "[]"),
            topics=topics,
            impact_score=row["impact_score"],
            retrieval_provider=row["retrieval_provider"],
            fetched_at_utc=datetime.fromisoformat(row["fetched_at_utc"]),
        )

    # ==================== Stories ====================

    def save_story(self, story: Story) -> None:
        """Save story to database"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO stories
                (story_id, lead_article_id, headline, summary, why_it_matters,
                 first_seen_at, last_updated_at, entities_json, tickers_json,
                 topics_json, regions_json, article_ids_json, source_count,
                 impact_score, confidence_score, previous_summary, change_description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                story.story_id,
                story.lead_article_id,
                story.headline,
                story.summary,
                story.why_it_matters,
                story.first_seen_at.isoformat(),
                story.last_updated_at.isoformat(),
                json.dumps([e.to_dict() for e in story.entities]),
                json.dumps(story.tickers),
                json.dumps([t.value for t in story.topics]),
                json.dumps(story.regions),
                json.dumps(story.article_ids),
                story.source_count,
                story.impact_score,
                story.confidence_score,
                story.previous_summary,
                story.change_description,
            ))

    def save_stories(self, stories: List[Story]) -> int:
        """Save multiple stories"""
        count = 0
        for story in stories:
            try:
                self.save_story(story)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to save story {story.story_id}: {e}")
        return count

    def get_story(self, story_id: str) -> Optional[Story]:
        """Get story by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stories WHERE story_id = ?", (story_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_story(row)
        return None

    def get_recent_stories(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> List[Story]:
        """Get recent stories"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM stories
                WHERE last_updated_at > ?
                ORDER BY impact_score DESC, last_updated_at DESC
                LIMIT ?
            """, (cutoff, limit))
            return [self._row_to_story(row) for row in cursor.fetchall()]

    def get_stories_by_topic(
        self,
        topic: TopicCategory,
        hours: int = 24,
        limit: int = 20,
    ) -> List[Story]:
        """Get stories by topic"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # SQLite JSON search
            cursor.execute("""
                SELECT * FROM stories
                WHERE last_updated_at > ?
                AND topics_json LIKE ?
                ORDER BY impact_score DESC
                LIMIT ?
            """, (cutoff, f'%"{topic.value}"%', limit))
            return [self._row_to_story(row) for row in cursor.fetchall()]

    def _row_to_story(self, row: sqlite3.Row) -> Story:
        """Convert database row to Story object"""
        entities = [Entity(**e) for e in json.loads(row["entities_json"] or "[]")]
        topics = [TopicCategory(t) for t in json.loads(row["topics_json"] or "[]")]

        return Story(
            story_id=row["story_id"],
            lead_article_id=row["lead_article_id"],
            headline=row["headline"],
            summary=row["summary"] or "",
            why_it_matters=row["why_it_matters"] or "",
            first_seen_at=datetime.fromisoformat(row["first_seen_at"]),
            last_updated_at=datetime.fromisoformat(row["last_updated_at"]),
            entities=entities,
            tickers=json.loads(row["tickers_json"] or "[]"),
            topics=topics,
            regions=json.loads(row["regions_json"] or "[]"),
            article_ids=json.loads(row["article_ids_json"] or "[]"),
            source_count=row["source_count"],
            impact_score=row["impact_score"],
            confidence_score=row["confidence_score"],
            previous_summary=row["previous_summary"],
            change_description=row["change_description"],
        )

    # ==================== Categories ====================

    def get_category(self, category_id: str) -> Optional[Category]:
        """Get category by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM categories WHERE category_id = ?", (category_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_category(row)
        return None

    def get_all_categories(self, active_only: bool = True) -> List[Category]:
        """Get all categories"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute("SELECT * FROM categories WHERE is_active = 1")
            else:
                cursor.execute("SELECT * FROM categories")
            return [self._row_to_category(row) for row in cursor.fetchall()]

    def save_category(self, category: Category) -> None:
        """Save category"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO categories
                (category_id, name, description, config_json, is_active, last_successful_fetch)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                category.category_id,
                category.name,
                category.description,
                json.dumps(category.to_dict()),
                1 if category.is_active else 0,
                category.last_successful_fetch.isoformat() if category.last_successful_fetch else None,
            ))

    def _row_to_category(self, row: sqlite3.Row) -> Category:
        """Convert database row to Category object"""
        config = json.loads(row["config_json"])
        topics = [TopicCategory(t) for t in config.get("topics", [])]

        return Category(
            category_id=row["category_id"],
            name=row["name"],
            description=row["description"],
            keywords=config.get("keywords", []),
            entities=config.get("entities", []),
            tickers=config.get("tickers", []),
            exclusions=config.get("exclusions", []),
            topics=topics,
            regions=config.get("regions", []),
            languages=config.get("languages", []),
            sources=config.get("sources", []),
            recency_weight=config.get("recency_weight", 0.4),
            impact_weight=config.get("impact_weight", 0.3),
            relevance_weight=config.get("relevance_weight", 0.3),
            max_stories_per_brief=config.get("max_stories_per_brief", 10),
            digest_tone=config.get("digest_tone", "professional"),
            is_active=bool(row["is_active"]),
            last_successful_fetch=datetime.fromisoformat(row["last_successful_fetch"]) if row["last_successful_fetch"] else None,
        )

    # ==================== Users ====================

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get user by ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                config = json.loads(row["config_json"])
                return UserProfile(
                    user_id=row["user_id"],
                    email=row["email"],
                    category_ids=config.get("category_ids", []),
                    watchlist_tickers=config.get("watchlist_tickers", []),
                    timezone=config.get("timezone", "America/Toronto"),
                    morning_brief_time=config.get("morning_brief_time", "07:00"),
                    evening_brief_time=config.get("evening_brief_time", "19:00"),
                    enable_push_notifications=config.get("enable_push_notifications", False),
                    enable_email_digest=config.get("enable_email_digest", True),
                    last_seen_story_ids=set(config.get("last_seen_story_ids", [])),
                )
        return None

    def save_user(self, user: UserProfile) -> None:
        """Save user profile"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO users (user_id, email, config_json)
                VALUES (?, ?, ?)
            """, (
                user.user_id,
                user.email,
                json.dumps(user.to_dict()),
            ))

    # ==================== Briefs ====================

    def save_brief(self, brief: Brief) -> None:
        """Save generated brief"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO briefs
                (brief_id, user_id, brief_type, generated_at, content_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                brief.brief_id,
                brief.user_id,
                brief.brief_type,
                brief.generated_at.isoformat(),
                json.dumps(brief.to_dict()),
            ))

    def get_user_briefs(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Brief]:
        """Get recent briefs for a user"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT content_json FROM briefs
                WHERE user_id = ?
                ORDER BY generated_at DESC
                LIMIT ?
            """, (user_id, limit))

            briefs = []
            for row in cursor.fetchall():
                data = json.loads(row["content_json"])
                # Reconstruct Brief (simplified)
                briefs.append(Brief(
                    brief_id=data["brief_id"],
                    user_id=data["user_id"],
                    brief_type=data["brief_type"],
                    generated_at=datetime.fromisoformat(data["generated_at"]),
                    stories=[],  # Would need to reconstruct from data
                    new_stories_count=data["new_stories_count"],
                    updated_stories_count=data["updated_stories_count"],
                    executive_summary=data.get("executive_summary"),
                    market_outlook=data.get("market_outlook"),
                ))
            return briefs

    # ==================== Cleanup ====================

    def cleanup_old_data(self, max_age_days: int = 30) -> Dict[str, int]:
        """Remove old data from database"""
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()

        deleted = {"articles": 0, "stories": 0, "briefs": 0}

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM articles WHERE published_at_utc < ?", (cutoff,))
            deleted["articles"] = cursor.rowcount

            cursor.execute("DELETE FROM stories WHERE last_updated_at < ?", (cutoff,))
            deleted["stories"] = cursor.rowcount

            cursor.execute("DELETE FROM briefs WHERE generated_at < ?", (cutoff,))
            deleted["briefs"] = cursor.rowcount

        logger.info(f"Cleaned up old data: {deleted}")
        return deleted

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {}

            for table in ["articles", "stories", "categories", "users", "briefs"]:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            return stats
