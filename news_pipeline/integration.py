"""
Integration with Financial News Infrastructure

Bridges the multi-model pipeline with the existing financial_news system:
- Uses financial_news storage for persistence
- Integrates with the existing scheduler
- Leverages existing connectors as fallback sources
- Syncs with the verification_engine

This allows the multi-model pipeline to coexist with and enhance
the existing financial news infrastructure.
"""

import logging
from datetime import datetime

from news_pipeline.models import (
    CandidateStory,
    FinalizedStory,
    PublishedFeedItem,
)

logger = logging.getLogger(__name__)


class FinancialNewsIntegration:
    """
    Integration layer between news_pipeline and financial_news.

    Provides:
    - Storage sync with NewsStore
    - Candidate import from existing connectors
    - Verification sync with verification_engine
    - Scheduler integration for AM/PM runs
    """

    def __init__(
        self,
        db_path: str = "data/news_store.db",
        use_existing_storage: bool = True,
    ):
        """
        Initialize integration.

        Args:
            db_path: Path to financial_news database
            use_existing_storage: Whether to use existing NewsStore
        """
        self.db_path = db_path
        self.use_existing_storage = use_existing_storage
        self._store = None
        self._verification_engine = None

    def _get_store(self):
        """Lazy load NewsStore from financial_news."""
        if self._store is None and self.use_existing_storage:
            try:
                from financial_news.storage import NewsStore

                self._store = NewsStore(self.db_path)
                logger.info("Connected to financial_news storage")
            except ImportError:
                logger.warning("financial_news.storage not available")
        return self._store

    def _get_verification_engine(self):
        """Lazy load VerificationEngine from financial_news."""
        if self._verification_engine is None:
            try:
                from financial_news.verification_engine import VerificationEngine

                self._verification_engine = VerificationEngine(self.db_path)
                logger.info("Connected to verification_engine")
            except ImportError:
                logger.warning("financial_news.verification_engine not available")
        return self._verification_engine

    async def import_candidates_from_connectors(
        self,
        categories: list[str],
        hours_lookback: int = 12,
    ) -> list[CandidateStory]:
        """
        Import candidate stories from existing financial_news connectors.

        Uses GDELT, SEC Edgar, and RSS feeds as additional sources
        alongside Grok's web search discovery.

        Args:
            categories: Categories to fetch
            hours_lookback: Hours to look back

        Returns:
            List of CandidateStory objects
        """
        candidates = []

        try:
            from financial_news.sources import GDELTFetcher, RSSFetcher, SECEdgarFetcher

            # Fetch from each source
            fetchers = [
                ("GDELT", GDELTFetcher()),
                ("SEC", SECEdgarFetcher()),
                ("RSS", RSSFetcher()),
            ]

            for source_name, fetcher in fetchers:
                try:
                    articles = await fetcher.fetch_recent(hours=hours_lookback)
                    for article in articles:
                        # Convert to CandidateStory
                        candidates.append(
                            CandidateStory(
                                story_id=article.get("id", "")[:8],
                                headline=article.get("title", ""),
                                source_name=source_name,
                                source_url=article.get("url", ""),
                                snippet=article.get("summary", "")[:500],
                                tickers=article.get("entities", {}).get("tickers", []),
                                category=self._map_category(article.get("category")),
                                published_at=article.get("published_at"),
                                source_type=self._get_source_type(source_name),
                                discovered_at=datetime.utcnow(),
                            )
                        )
                except Exception as e:
                    logger.warning(f"Failed to fetch from {source_name}: {e}")

            logger.info(f"Imported {len(candidates)} candidates from connectors")

        except ImportError:
            logger.warning("financial_news.sources not available")

        return candidates

    async def sync_to_storage(self, story: FinalizedStory) -> bool:
        """
        Sync a finalized story to the financial_news storage.

        Args:
            story: The finalized story to sync

        Returns:
            True if synced successfully
        """
        store = self._get_store()
        if not store:
            return False

        try:
            # Convert to financial_news Story format
            from financial_news.models import Story

            fn_story = Story(
                story_id=story.story_id,
                title=story.title,
                summary=story.what_happened,
                why_it_matters=story.why_it_matters,
                bullet_points=story.bullet_points,
                tickers=story.affected_assets,
                category=story.category,
                confidence_score=story.verification_confidence / 10.0,
                is_breaking=story.is_breaking,
                priority=story.priority,
                created_at=story.finalized_at,
            )

            await store.save_story(fn_story)
            logger.info(f"Synced story {story.story_id} to financial_news storage")
            return True

        except Exception as e:
            logger.error(f"Failed to sync to storage: {e}")
            return False

    async def sync_verification_result(
        self,
        story_id: str,
        confidence: int,
        status: str,
        notes: str = "",
    ) -> bool:
        """
        Sync verification result to verification_engine database.

        Args:
            story_id: Story identifier
            confidence: Verification confidence (0-10)
            status: Verification status
            notes: Additional notes

        Returns:
            True if synced successfully
        """
        engine = self._get_verification_engine()
        if not engine:
            return False

        try:
            from financial_news.verification_engine import VerificationStatus

            # Map status
            status_map = {
                "verified": VerificationStatus.VERIFIED,
                "partial": VerificationStatus.VERIFIED,
                "unverified": VerificationStatus.NEEDS_REVIEW,
                "contradicted": VerificationStatus.REJECTED,
            }

            engine.update_verification_status(
                candidate_id=story_id,
                status=status_map.get(status, VerificationStatus.NEEDS_REVIEW),
                confidence=confidence,
                notes=notes,
            )

            logger.info(f"Synced verification for {story_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to sync verification: {e}")
            return False

    def _map_category(self, category: str | None) -> str:
        """Map financial_news category to pipeline category."""
        if not category:
            return "other"

        category_map = {
            "federal_reserve": "fed",
            "monetary_policy": "fed",
            "corporate_earnings": "earnings",
            "quarterly_results": "earnings",
            "energy": "oil",
            "crude_oil": "oil",
            "international": "geopolitics",
            "politics": "geopolitics",
        }

        return category_map.get(category.lower(), category.lower())

    def _get_source_type(self, source_name: str) -> str:
        """Get source type based on source name."""
        official_sources = {"SEC", "Fed", "Treasury"}
        mainstream_sources = {"GDELT", "RSS"}

        if source_name in official_sources:
            return "official"
        elif source_name in mainstream_sources:
            return "mainstream"
        return "unknown"


class SchedulerIntegration:
    """
    Integration with financial_news scheduler.

    Allows the multi-model pipeline to be scheduled alongside
    the existing AM/PM news runs.
    """

    def __init__(self, scheduler=None):
        """
        Initialize scheduler integration.

        Args:
            scheduler: Existing scheduler instance (optional)
        """
        self._scheduler = scheduler

    def register_pipeline_job(
        self,
        orchestrator,
        schedule: str = "am_pm",
    ) -> bool:
        """
        Register the pipeline as a scheduled job.

        Args:
            orchestrator: NewsPipelineOrchestrator instance
            schedule: Schedule type (am_pm, hourly, etc.)

        Returns:
            True if registered successfully
        """
        try:
            from financial_news.scheduler import NewsScheduler

            if not self._scheduler:
                self._scheduler = NewsScheduler()

            async def pipeline_job():
                from news_pipeline import DiscoveryRequest

                request = DiscoveryRequest(
                    time_window="last_6h" if schedule == "hourly" else "overnight",
                )
                return await orchestrator.run(request)

            self._scheduler.add_job(
                name="multi_model_pipeline",
                func=pipeline_job,
                schedule=schedule,
            )

            logger.info(f"Registered pipeline job with schedule: {schedule}")
            return True

        except ImportError:
            logger.warning("financial_news.scheduler not available")
            return False
        except Exception as e:
            logger.error(f"Failed to register pipeline job: {e}")
            return False


def create_integrated_publisher(
    db_path: str = "data/news_store.db",
    sync_verification: bool = True,
) -> callable:
    """
    Create a publisher that integrates with financial_news.

    Args:
        db_path: Path to financial_news database
        sync_verification: Whether to sync verification results

    Returns:
        Publisher callback function
    """
    integration = FinancialNewsIntegration(db_path=db_path)

    async def publish_callback(feed_item: PublishedFeedItem) -> bool:
        # Convert to FinalizedStory
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

        # Sync to storage
        synced = await integration.sync_to_storage(story)

        # Optionally sync verification
        if sync_verification:
            await integration.sync_verification_result(
                story_id=story.story_id,
                confidence=story.verification_confidence,
                status="verified" if story.verification_confidence >= 7 else "partial",
            )

        return synced

    return publish_callback
