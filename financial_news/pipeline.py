"""
News Pipeline Orchestrator

Coordinates the full news processing pipeline:
1. Fetch articles from sources
2. Extract entities and classify
3. Cluster into stories
4. Score impact
5. Generate briefs
6. Store and deliver

Runs on schedule (AM/PM) per the Bloomberg-style approach.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from financial_news.models import Article, Brief, Category, Story, TopicCategory
from financial_news.processing import BriefGenerator, EntityExtractor, ImpactScorer, StoryClusterer
from financial_news.sources import GDELTFetcher, RSSFetcher, SECEdgarFetcher
from financial_news.storage import NewsStore

logger = logging.getLogger(__name__)


class NewsPipeline:
    """
    Main orchestrator for the financial news platform.

    Implements the full pipeline from data fetching to brief delivery.
    """

    def __init__(
        self,
        db_path: str = "data/news_store.db",
        local_agent: Any | None = None,
        cloud_client: Any | None = None,
    ):
        """
        Initialize news pipeline.

        Args:
            db_path: Path to SQLite database
            local_agent: Local LLM agent for processing
            cloud_client: Optional cloud API client for high-quality summaries
        """
        # Storage
        self.store = NewsStore(db_path)

        # Data sources
        self.gdelt = GDELTFetcher()
        self.sec_edgar = SECEdgarFetcher()
        self.rss = RSSFetcher()

        # Processing
        self.entity_extractor = EntityExtractor()
        self.story_clusterer = StoryClusterer()
        self.impact_scorer = ImpactScorer()
        self.brief_generator = BriefGenerator(
            local_agent=local_agent,
            cloud_client=cloud_client,
        )

        # State
        self._is_running = False
        self._last_run: dict[str, datetime] = {}

    async def run_full_pipeline(
        self,
        categories: list[Category] | None = None,
        hours_lookback: int = 12,
    ) -> dict[str, Any]:
        """
        Run the complete news pipeline.

        Args:
            categories: Categories to process (default: all active)
            hours_lookback: Hours to look back for articles

        Returns:
            Pipeline run statistics
        """
        if self._is_running:
            logger.warning("Pipeline already running")
            return {"status": "already_running"}

        self._is_running = True
        start_time = datetime.utcnow()
        stats = {
            "start_time": start_time.isoformat(),
            "articles_fetched": 0,
            "articles_processed": 0,
            "stories_created": 0,
            "stories_updated": 0,
            "errors": [],
        }

        try:
            # Get categories to process
            if not categories:
                categories = self.store.get_all_categories(active_only=True)

            logger.info(f"Starting pipeline for {len(categories)} categories")

            # Calculate time window
            end_time = datetime.utcnow()
            start_time_window = end_time - timedelta(hours=hours_lookback)

            # Step 1: Fetch articles from all sources
            all_articles = await self._fetch_all_sources(categories, start_time_window, end_time)
            stats["articles_fetched"] = len(all_articles)

            # Step 2: Process articles (entity extraction, classification)
            processed_articles = await self._process_articles(all_articles)
            stats["articles_processed"] = len(processed_articles)

            # Step 3: Cluster into stories
            stories = self._cluster_articles(processed_articles)
            stats["stories_created"] = len(stories)

            # Step 4: Score impact
            for story in stories:
                story.impact_score = self.impact_scorer.score_story(story)

            # Step 5: Generate summaries for top stories
            top_stories = sorted(stories, key=lambda s: s.impact_score, reverse=True)[:20]
            for story in top_stories:
                article_ids = story.article_ids[:5]
                articles = self.store.get_articles_by_ids(article_ids)
                await self.brief_generator.generate_story_summary(
                    story, articles, use_cloud=(top_stories.index(story) < 5)
                )

            # Step 6: Save to database
            self.store.save_articles(processed_articles)
            self.store.save_stories(stories)

            # Update category fetch times
            for category in categories:
                category.last_successful_fetch = datetime.utcnow()
                self.store.save_category(category)

            stats["status"] = "success"
            stats["duration_seconds"] = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                f"Pipeline completed: {stats['articles_fetched']} articles, "
                f"{stats['stories_created']} stories in {stats['duration_seconds']:.1f}s"
            )

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            stats["status"] = "error"
            stats["errors"].append(str(e))

        finally:
            self._is_running = False
            self._last_run["full"] = datetime.utcnow()

        return stats

    async def _fetch_all_sources(
        self,
        categories: list[Category],
        start_time: datetime,
        end_time: datetime,
    ) -> list[Article]:
        """Fetch articles from all configured sources"""
        all_articles = []

        for category in categories:
            try:
                # Fetch from GDELT
                gdelt_articles = await self.gdelt.fetch_articles(category, start_time, end_time)
                all_articles.extend(gdelt_articles)

                # Fetch from SEC EDGAR (for relevant categories)
                if any(
                    t in [TopicCategory.EARNINGS, TopicCategory.M_AND_A, TopicCategory.IPO]
                    for t in category.topics
                ):
                    sec_articles = await self.sec_edgar.fetch_articles(
                        category, start_time, end_time
                    )
                    all_articles.extend(sec_articles)

                # Fetch from RSS
                rss_articles = await self.rss.fetch_articles(category, start_time, end_time)
                all_articles.extend(rss_articles)

            except Exception as e:
                logger.warning(f"Error fetching for category {category.name}: {e}")

        # Deduplicate by URL
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            if article.canonical_url not in seen_urls:
                seen_urls.add(article.canonical_url)
                unique_articles.append(article)

        logger.info(f"Fetched {len(unique_articles)} unique articles")
        return unique_articles

    async def _process_articles(self, articles: list[Article]) -> list[Article]:
        """Process articles: entity extraction and classification"""
        processed = await self.entity_extractor.process_batch(articles)

        # Score articles
        processed = self.impact_scorer.score_articles(processed)

        return processed

    def _cluster_articles(self, articles: list[Article]) -> list[Story]:
        """Cluster articles into stories"""
        # Get existing stories for merging
        existing_stories = self.store.get_recent_stories(hours=48)

        stories = self.story_clusterer.cluster_articles(articles, existing_stories)

        return stories

    async def generate_brief_for_user(
        self,
        user_id: str,
        brief_type: str = "morning",
    ) -> Brief | None:
        """
        Generate a brief for a specific user.

        Args:
            user_id: User ID
            brief_type: "morning" or "evening"

        Returns:
            Generated Brief or None if user not found
        """
        user = self.store.get_user(user_id)
        if not user:
            logger.warning(f"User {user_id} not found")
            return None

        # Get relevant stories
        stories = []
        for category_id in user.category_ids:
            category = self.store.get_category(category_id)
            if category:
                for topic in category.topics:
                    topic_stories = self.store.get_stories_by_topic(topic)
                    stories.extend(topic_stories)

        # Also get stories matching watchlist tickers
        if user.watchlist_tickers:
            # Would need ticker-based query
            pass

        # Deduplicate and rank
        seen_ids = set()
        unique_stories = []
        for story in stories:
            if story.story_id not in seen_ids:
                seen_ids.add(story.story_id)
                unique_stories.append(story)

        # Sort by impact
        unique_stories.sort(key=lambda s: s.impact_score, reverse=True)

        # Generate brief
        brief = await self.brief_generator.generate_brief(user, unique_stories[:15], brief_type)

        # Update user's seen stories
        user.last_seen_story_ids.update(s.story_id for s in brief.stories)
        self.store.save_user(user)

        # Save brief
        self.store.save_brief(brief)

        return brief

    async def get_category_feed(
        self,
        category_id: str,
        limit: int = 20,
    ) -> list[Story]:
        """
        Get ranked stories for a category.

        Args:
            category_id: Category ID
            limit: Maximum stories to return

        Returns:
            Ranked list of stories
        """
        category = self.store.get_category(category_id)
        if not category:
            return []

        stories = []
        for topic in category.topics:
            topic_stories = self.store.get_stories_by_topic(topic, limit=limit)
            stories.extend(topic_stories)

        # Deduplicate
        seen_ids = set()
        unique_stories = []
        for story in stories:
            if story.story_id not in seen_ids:
                seen_ids.add(story.story_id)
                unique_stories.append(story)

        # Rank using category weights
        def rank_story(story: Story) -> float:
            recency_hours = (datetime.utcnow() - story.last_updated_at).total_seconds() / 3600
            recency_score = max(0, 1 - recency_hours / 24)

            return (
                category.recency_weight * recency_score
                + category.impact_weight * story.impact_score
                + category.relevance_weight * story.confidence_score
            )

        unique_stories.sort(key=rank_story, reverse=True)

        return unique_stories[:limit]

    async def health_check(self) -> dict[str, Any]:
        """Check health of all components"""
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "sources": {},
            "database": False,
            "last_run": self._last_run.get("full"),
        }

        # Check sources
        health["sources"]["gdelt"] = await self.gdelt.health_check()
        health["sources"]["sec_edgar"] = await self.sec_edgar.health_check()
        health["sources"]["rss"] = await self.rss.health_check()

        # Check database
        try:
            stats = self.store.get_stats()
            health["database"] = True
            health["db_stats"] = stats
        except Exception as e:
            health["database"] = False
            health["db_error"] = str(e)

        return health

    async def close(self) -> None:
        """Close all connections"""
        await self.gdelt.close()
        await self.sec_edgar.close()
        await self.rss.close()


# Scheduler for AM/PM runs
class PipelineScheduler:
    """
    Schedules pipeline runs for morning and evening briefs.

    Toronto timezone:
    - Morning: 06:30 ET
    - Evening: 18:30 ET
    """

    def __init__(self, pipeline: NewsPipeline, timezone: str = "America/Toronto"):
        self.pipeline = pipeline
        self.timezone = timezone
        self._running = False

    async def start(self) -> None:
        """Start the scheduler"""
        from datetime import time

        import pytz

        tz = pytz.timezone(self.timezone)
        morning_time = time(6, 30)
        evening_time = time(18, 30)

        self._running = True
        logger.info(f"Starting scheduler ({self.timezone})")

        while self._running:
            now = datetime.now(tz)
            current_time = now.time()

            # Check if it's time to run
            if (
                current_time.hour == morning_time.hour
                and current_time.minute == morning_time.minute
            ):
                logger.info("Running morning pipeline")
                await self.pipeline.run_full_pipeline()

            elif (
                current_time.hour == evening_time.hour
                and current_time.minute == evening_time.minute
            ):
                logger.info("Running evening pipeline")
                await self.pipeline.run_full_pipeline()

            # Sleep until next minute
            await asyncio.sleep(60)

    def stop(self) -> None:
        """Stop the scheduler"""
        self._running = False


# CLI entry point
async def main():
    """Command-line entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Financial News Pipeline")
    parser.add_argument("--run-now", action="store_true", help="Run pipeline immediately")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule")
    parser.add_argument("--health-check", action="store_true", help="Check health")
    parser.add_argument("--hours", type=int, default=12, help="Hours lookback")

    args = parser.parse_args()

    pipeline = NewsPipeline()

    if args.health_check:
        health = await pipeline.health_check()
        print(health)

    elif args.run_now:
        stats = await pipeline.run_full_pipeline(hours_lookback=args.hours)
        print(stats)

    elif args.schedule:
        scheduler = PipelineScheduler(pipeline)
        try:
            await scheduler.start()
        except KeyboardInterrupt:
            scheduler.stop()

    await pipeline.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
