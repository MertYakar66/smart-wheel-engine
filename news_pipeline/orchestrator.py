"""
News Pipeline Orchestrator

Coordinates the multi-model pipeline flow:
Grok (Discovery) → Gemini (Verification) → ChatGPT (Formatting) → Claude (Editorial)

Features:
- Async parallel processing where possible
- Stage-by-stage error handling
- Progress tracking and logging
- Configurable thresholds
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import datetime

from news_pipeline.config import PipelineConfig
from news_pipeline.models import (
    CandidateStory,
    DiscoveryRequest,
    FinalizedStory,
    FormattedStory,
    PipelineResult,
    PipelineStage,
    VerificationResult,
    VerificationStatus,
)
from news_pipeline.providers import (
    ChatGPTProvider,
    ClaudeProvider,
    GeminiProvider,
    GrokProvider,
)

logger = logging.getLogger(__name__)


class NewsPipelineOrchestrator:
    """
    Orchestrates the complete news pipeline.

    Coordinates four AI models through discovery, verification,
    formatting, and editorial stages to produce publication-ready
    financial news.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        publish_callback: Callable | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Pipeline configuration (uses defaults if None)
            publish_callback: Optional callback for publishing stories
        """
        self.config = config or PipelineConfig.from_env()
        self.publish_callback = publish_callback

        # Initialize providers
        self.grok = GrokProvider(self.config.grok)
        self.gemini = GeminiProvider(self.config.gemini)
        self.chatgpt = ChatGPTProvider(self.config.chatgpt)
        self.claude = ClaudeProvider(self.config.claude)

        self._initialized = False
        self._current_stage = PipelineStage.DISCOVERY

    async def initialize(self) -> None:
        """Initialize all providers in parallel."""
        if self._initialized:
            return

        logger.info("Initializing pipeline providers...")

        # Validate configuration
        errors = self.config.validate()
        if errors:
            for error in errors:
                logger.warning(f"Config warning: {error}")

        # Initialize enabled providers in parallel
        init_tasks = []
        if self.config.grok.enabled and self.config.grok.api_key:
            init_tasks.append(self.grok.initialize())
        if self.config.gemini.enabled and self.config.gemini.api_key:
            init_tasks.append(self.gemini.initialize())
        if self.config.chatgpt.enabled and self.config.chatgpt.api_key:
            init_tasks.append(self.chatgpt.initialize())
        if self.config.claude.enabled and self.config.claude.api_key:
            init_tasks.append(self.claude.initialize())

        if init_tasks:
            await asyncio.gather(*init_tasks)

        self._initialized = True
        logger.info(f"Pipeline initialized with providers: {self.config.get_enabled_providers()}")

    async def health_check(self) -> dict[str, bool]:
        """Check health of all enabled providers."""
        results = {}

        checks = []
        names = []

        if self.config.grok.enabled:
            checks.append(self.grok.health_check())
            names.append("grok")
        if self.config.gemini.enabled:
            checks.append(self.gemini.health_check())
            names.append("gemini")
        if self.config.chatgpt.enabled:
            checks.append(self.chatgpt.health_check())
            names.append("chatgpt")
        if self.config.claude.enabled:
            checks.append(self.claude.health_check())
            names.append("claude")

        if checks:
            statuses = await asyncio.gather(*checks, return_exceptions=True)
            for name, status in zip(names, statuses, strict=True):
                results[name] = status is True

        return results

    async def run(self, request: DiscoveryRequest) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            request: Discovery parameters

        Returns:
            PipelineResult with all processed stories
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow()

        result = PipelineResult(
            run_id=run_id,
            started_at=started_at,
            request=request,
        )

        logger.info(f"[Pipeline {run_id}] Starting pipeline run")
        logger.info(f"[Pipeline {run_id}] Categories: {request.categories}")
        logger.info(f"[Pipeline {run_id}] Time window: {request.time_window}")
        if request.tickers:
            logger.info(f"[Pipeline {run_id}] Tickers: {request.tickers}")

        try:
            if not self._initialized:
                await self.initialize()

            # Stage 1: Discovery
            self._current_stage = PipelineStage.DISCOVERY
            logger.info(f"[Pipeline {run_id}] Stage 1: Discovery (Grok)")
            candidates = await self._discover(request)
            result.discovered_count = len(candidates)
            logger.info(f"[Pipeline {run_id}] Discovered {len(candidates)} candidates")

            if not candidates:
                logger.warning(f"[Pipeline {run_id}] No candidates found")
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                return result

            # Stage 2: Verification
            self._current_stage = PipelineStage.VERIFICATION
            logger.info(f"[Pipeline {run_id}] Stage 2: Verification (Gemini)")
            verified = await self._verify_batch(candidates)
            publishable = [
                v for v in verified if v.confidence >= self.config.min_verification_confidence
            ]
            result.verified_count = len(publishable)
            logger.info(
                f"[Pipeline {run_id}] Verified {len(verified)}, "
                f"{len(publishable)} passed threshold ({self.config.min_verification_confidence}+)"
            )

            if not publishable:
                logger.warning(f"[Pipeline {run_id}] No stories passed verification")
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                return result

            # Stage 3: Formatting
            self._current_stage = PipelineStage.FORMATTING
            logger.info(f"[Pipeline {run_id}] Stage 3: Formatting (ChatGPT)")
            formatted = await self._format_batch(publishable)
            result.formatted_count = len(formatted)
            logger.info(f"[Pipeline {run_id}] Formatted {len(formatted)} stories")

            # Stage 4: Editorial
            self._current_stage = PipelineStage.EDITORIAL
            logger.info(f"[Pipeline {run_id}] Stage 4: Editorial (Claude)")
            finalized = await self._finalize_batch(formatted)
            result.finalized_count = len(finalized)
            result.stories = finalized
            logger.info(f"[Pipeline {run_id}] Finalized {len(finalized)} stories")

            # Stage 5: Publishing
            if self.publish_callback and finalized:
                self._current_stage = PipelineStage.PUBLISHING
                logger.info(f"[Pipeline {run_id}] Stage 5: Publishing")
                published = await self._publish_batch(finalized)
                result.published_count = len(published)
                result.published_items = published
                logger.info(f"[Pipeline {run_id}] Published {len(published)} stories")

            result.status = "completed"

        except Exception as e:
            logger.error(f"[Pipeline {run_id}] Pipeline failed at {self._current_stage.value}: {e}")
            result.status = "failed"
            result.errors.append(str(e))
            result.stage_errors[self._current_stage.value] = str(e)

        result.completed_at = datetime.utcnow()
        duration = result.duration_seconds

        logger.info(
            f"[Pipeline {run_id}] Pipeline {result.status} in {duration:.1f}s | "
            f"Discovered: {result.discovered_count} | "
            f"Verified: {result.verified_count} | "
            f"Published: {result.published_count}"
        )

        return result

    async def _discover(self, request: DiscoveryRequest) -> list[CandidateStory]:
        """Stage 1: Discover news with Grok."""
        raw_stories = await self.grok.discover_news(
            tickers=list(request.tickers),
            categories=list(request.categories),
            time_window=request.time_window,
            max_results=request.max_results,
        )

        candidates = []
        for story in raw_stories:
            candidates.append(
                CandidateStory(
                    story_id=str(uuid.uuid4())[:8],
                    headline=story.get("headline", ""),
                    source_name=story.get("source_name", "Unknown"),
                    source_url=story.get("source_url", ""),
                    snippet=story.get("snippet", ""),
                    published_at=story.get("published_at"),
                    tickers=story.get("tickers", []),
                    category=story.get("category", "other"),
                    relevance_score=story.get("relevance_score", 0.5),
                    source_type=story.get("source_type", "unknown"),
                    discovered_at=datetime.utcnow(),
                )
            )

        return candidates

    async def _verify_batch(self, candidates: list[CandidateStory]) -> list[VerificationResult]:
        """Stage 2: Verify stories with Gemini."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_verifications)

        async def verify_one(candidate: CandidateStory) -> VerificationResult | None:
            async with semaphore:
                try:
                    result = await self.gemini.verify_story(
                        headline=candidate.headline,
                        source_url=candidate.source_url,
                        tickers=candidate.tickers,
                        category=candidate.category,
                    )

                    status_map = {
                        "verified": VerificationStatus.VERIFIED,
                        "partial": VerificationStatus.PARTIAL,
                        "unverified": VerificationStatus.UNVERIFIED,
                        "contradicted": VerificationStatus.CONTRADICTED,
                        "error": VerificationStatus.ERROR,
                    }

                    return VerificationResult(
                        story_id=candidate.story_id,
                        candidate=candidate,
                        status=status_map.get(
                            result.get("status", "unverified"),
                            VerificationStatus.UNVERIFIED,
                        ),
                        confidence=result.get("confidence", 0),
                        verified_facts=result.get("verified_facts", []),
                        what_happened=result.get("what_happened", ""),
                        contradictions=result.get("contradictions", []),
                        sources_checked=result.get("sources_checked", 0),
                        verification_notes=result.get("verification_notes", ""),
                        verified_at=datetime.utcnow(),
                    )

                except Exception as e:
                    logger.error(f"Verification failed for {candidate.story_id}: {e}")
                    return None

        results = await asyncio.gather(
            *[verify_one(c) for c in candidates],
            return_exceptions=True,
        )

        return [r for r in results if isinstance(r, VerificationResult)]

    async def _format_batch(self, verified: list[VerificationResult]) -> list[FormattedStory]:
        """Stage 3: Format stories with ChatGPT."""

        async def format_one(v: VerificationResult) -> FormattedStory | None:
            try:
                result = await self.chatgpt.format_story(
                    story_id=v.story_id,
                    verified_facts=v.verified_facts,
                    what_happened=v.what_happened,
                    affected_assets=v.candidate.tickers,
                    category=v.candidate.category,
                )

                return FormattedStory(
                    story_id=v.story_id,
                    title=result.get("title", ""),
                    what_happened=result.get("what_happened", v.what_happened),
                    bullet_points=result.get("bullet_points", v.verified_facts),
                    affected_assets=result.get("affected_assets", v.candidate.tickers),
                    related_tickers=result.get("related_tickers", []),
                    sector_impact=result.get("sector_impact"),
                    time_sensitivity=result.get("time_sensitivity", "normal"),
                    category=v.candidate.category,
                    verification_confidence=v.confidence,
                    formatted_at=datetime.utcnow(),
                )

            except Exception as e:
                logger.error(f"Formatting failed for {v.story_id}: {e}")
                return None

        results = await asyncio.gather(*[format_one(v) for v in verified])
        return [r for r in results if r is not None]

    async def _finalize_batch(self, formatted: list[FormattedStory]) -> list[FinalizedStory]:
        """Stage 4: Finalize stories with Claude."""

        async def finalize_one(f: FormattedStory) -> FinalizedStory | None:
            try:
                result = await self.claude.finalize_story(
                    story_id=f.story_id,
                    title=f.title,
                    what_happened=f.what_happened,
                    bullet_points=f.bullet_points,
                    affected_assets=f.affected_assets,
                    category=f.category,
                    confidence=f.verification_confidence,
                )

                return FinalizedStory(
                    story_id=f.story_id,
                    title=result.get("title", f.title),
                    what_happened=result.get("what_happened", f.what_happened),
                    why_it_matters=result.get("why_it_matters", ""),
                    bullet_points=result.get("bullet_points", f.bullet_points),
                    affected_assets=result.get("affected_assets", f.affected_assets),
                    category=f.category,
                    verification_confidence=f.verification_confidence,
                    market_implications=result.get("market_implications"),
                    trading_considerations=result.get("trading_considerations"),
                    is_breaking=result.get("is_breaking", False),
                    priority=result.get("priority", 5),
                    tags=result.get("tags", []),
                    finalized_at=datetime.utcnow(),
                )

            except Exception as e:
                logger.error(f"Finalization failed for {f.story_id}: {e}")
                return None

        results = await asyncio.gather(*[finalize_one(f) for f in formatted])
        return [r for r in results if r is not None]

    async def _publish_batch(self, finalized: list[FinalizedStory]) -> list:
        """Stage 5: Publish stories via callback."""
        from news_pipeline.models import PublishedFeedItem

        published = []

        for story in finalized:
            try:
                feed_item = PublishedFeedItem(
                    story_id=story.story_id,
                    title=story.title,
                    what_happened=story.what_happened,
                    why_it_matters=story.why_it_matters,
                    bullet_points=story.bullet_points,
                    affected_assets=story.affected_assets,
                    category=story.category,
                    verification_confidence=story.verification_confidence,
                    published_at=datetime.utcnow(),
                    feed_url="",
                )

                if asyncio.iscoroutinefunction(self.publish_callback):
                    result = await self.publish_callback(feed_item)
                else:
                    result = self.publish_callback(feed_item)

                if result:
                    published.append(feed_item)

            except Exception as e:
                logger.error(f"Publishing failed for {story.story_id}: {e}")

        return published
