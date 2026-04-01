"""
News Publisher

Handles publishing finalized stories to various destinations:
- Website feed (API)
- Database storage
- File output (JSON/HTML)

Integrates with the existing financial_news storage layer.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from news_pipeline.models import FinalizedStory, PublishedFeedItem

logger = logging.getLogger(__name__)


class NewsPublisher:
    """
    Publishes finalized stories to configured destinations.

    Supports multiple output modes:
    - api: POST to website API endpoint
    - database: Store in SQLite (integrates with financial_news)
    - file: Write to JSON/HTML files
    """

    def __init__(
        self,
        mode: str = "database",
        api_endpoint: str | None = None,
        api_key: str | None = None,
        db_path: str = "data/news_pipeline.db",
        output_dir: str = "data/published",
    ):
        """
        Initialize publisher.

        Args:
            mode: Publishing mode (api, database, file)
            api_endpoint: Website API endpoint for mode=api
            api_key: API authentication key
            db_path: SQLite database path for mode=database
            output_dir: Directory for file output for mode=file
        """
        self.mode = mode
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.db_path = db_path
        self.output_dir = output_dir

        self._db_conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Initialize the publisher."""
        if self.mode == "database":
            self._init_database()
        elif self.mode == "file":
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Publisher initialized in {self.mode} mode")

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db_conn = sqlite3.connect(self.db_path)
        cursor = self._db_conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS published_stories (
                story_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                what_happened TEXT,
                why_it_matters TEXT,
                bullet_points TEXT,
                affected_assets TEXT,
                category TEXT,
                verification_confidence INTEGER,
                published_at TEXT,
                is_breaking INTEGER DEFAULT 0,
                priority INTEGER DEFAULT 5,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_published_at
            ON published_stories(published_at)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category
            ON published_stories(category)
        """)

        self._db_conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    async def publish(self, story: FinalizedStory) -> PublishedFeedItem | None:
        """
        Publish a finalized story.

        Args:
            story: The story to publish

        Returns:
            PublishedFeedItem if successful, None otherwise
        """
        try:
            if self.mode == "api":
                return await self._publish_api(story)
            elif self.mode == "database":
                return await self._publish_database(story)
            elif self.mode == "file":
                return await self._publish_file(story)
            else:
                logger.error(f"Unknown publish mode: {self.mode}")
                return None

        except Exception as e:
            logger.error(f"Publishing failed for {story.story_id}: {e}")
            return None

    async def _publish_api(self, story: FinalizedStory) -> PublishedFeedItem | None:
        """Publish to website API."""
        if not self.api_endpoint:
            logger.error("API endpoint not configured")
            return None

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = await client.post(
                    self.api_endpoint,
                    json=story.to_dict(),
                    headers=headers,
                    timeout=30.0,
                )

                if response.status_code in (200, 201):
                    data = response.json()
                    feed_url = data.get("feed_url", self.api_endpoint)

                    logger.info(f"Published {story.story_id} to API")

                    return PublishedFeedItem(
                        story_id=story.story_id,
                        title=story.title,
                        what_happened=story.what_happened,
                        why_it_matters=story.why_it_matters,
                        bullet_points=story.bullet_points,
                        affected_assets=story.affected_assets,
                        category=story.category,
                        verification_confidence=story.verification_confidence,
                        published_at=datetime.utcnow(),
                        feed_url=feed_url,
                    )
                else:
                    logger.error(f"API publish failed: {response.status_code}")
                    return None

        except ImportError:
            logger.error("httpx package required for API publishing")
            return None

    async def _publish_database(self, story: FinalizedStory) -> PublishedFeedItem | None:
        """Publish to SQLite database."""
        if not self._db_conn:
            self._init_database()

        cursor = self._db_conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO published_stories
            (story_id, title, what_happened, why_it_matters, bullet_points,
             affected_assets, category, verification_confidence, published_at,
             is_breaking, priority, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                story.story_id,
                story.title,
                story.what_happened,
                story.why_it_matters,
                json.dumps(story.bullet_points),
                json.dumps(story.affected_assets),
                story.category,
                story.verification_confidence,
                datetime.utcnow().isoformat(),
                1 if story.is_breaking else 0,
                story.priority,
                json.dumps(story.tags),
            ),
        )

        self._db_conn.commit()
        logger.info(f"Published {story.story_id} to database")

        return PublishedFeedItem(
            story_id=story.story_id,
            title=story.title,
            what_happened=story.what_happened,
            why_it_matters=story.why_it_matters,
            bullet_points=story.bullet_points,
            affected_assets=story.affected_assets,
            category=story.category,
            verification_confidence=story.verification_confidence,
            published_at=datetime.utcnow(),
            feed_url=f"sqlite://{self.db_path}",
        )

    async def _publish_file(self, story: FinalizedStory) -> PublishedFeedItem | None:
        """Publish to JSON file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{story.story_id}.json"
        filepath = Path(self.output_dir) / filename

        with open(filepath, "w") as f:
            json.dump(story.to_dict(), f, indent=2)

        logger.info(f"Published {story.story_id} to {filepath}")

        return PublishedFeedItem(
            story_id=story.story_id,
            title=story.title,
            what_happened=story.what_happened,
            why_it_matters=story.why_it_matters,
            bullet_points=story.bullet_points,
            affected_assets=story.affected_assets,
            category=story.category,
            verification_confidence=story.verification_confidence,
            published_at=datetime.utcnow(),
            feed_url=f"file://{filepath.absolute()}",
        )

    async def get_recent_stories(
        self,
        limit: int = 20,
        category: str | None = None,
    ) -> list[dict]:
        """
        Get recently published stories from database.

        Args:
            limit: Maximum stories to return
            category: Filter by category (optional)

        Returns:
            List of story dictionaries
        """
        if self.mode != "database" or not self._db_conn:
            return []

        cursor = self._db_conn.cursor()

        if category:
            cursor.execute(
                """
                SELECT * FROM published_stories
                WHERE category = ?
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (category, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM published_stories
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (limit,),
            )

        columns = [desc[0] for desc in cursor.description]
        stories = []

        for row in cursor.fetchall():
            story = dict(zip(columns, row, strict=True))
            # Parse JSON fields
            story["bullet_points"] = json.loads(story.get("bullet_points", "[]"))
            story["affected_assets"] = json.loads(story.get("affected_assets", "[]"))
            story["tags"] = json.loads(story.get("tags", "[]"))
            story["is_breaking"] = bool(story.get("is_breaking"))
            stories.append(story)

        return stories

    def close(self) -> None:
        """Close database connection."""
        if self._db_conn:
            self._db_conn.close()
            self._db_conn = None


# Factory function for use as publish callback
def create_publisher_callback(
    mode: str = "database",
    **kwargs,
) -> callable:
    """
    Create a publisher callback for the orchestrator.

    Args:
        mode: Publishing mode
        **kwargs: Additional publisher arguments

    Returns:
        Async callback function
    """
    publisher = NewsPublisher(mode=mode, **kwargs)
    initialized = False

    async def callback(feed_item: PublishedFeedItem) -> bool:
        nonlocal initialized
        if not initialized:
            await publisher.initialize()
            initialized = True

        # Convert feed item back to finalized story for publishing
        story = FinalizedStory(
            story_id=feed_item.story_id,
            title=feed_item.title,
            what_happened=feed_item.what_happened,
            why_it_matters=feed_item.why_it_matters,
            bullet_points=feed_item.bullet_points,
            affected_assets=feed_item.affected_assets,
            category=feed_item.category,
            verification_confidence=feed_item.verification_confidence,
            finalized_at=feed_item.published_at,
        )

        result = await publisher.publish(story)
        return result is not None

    return callback
