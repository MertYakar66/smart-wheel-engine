"""
News Pipeline Orchestrator

Chains multiple AI providers to process financial news:
Grok (Discovery) → Gemini (Verification) → ChatGPT (Formatting) → Claude (Editorial)
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from .models.schema import (
    CandidateStory,
    DiscoveryRequest,
    FinalizedStory,
    FormattedStory,
    PipelineRun,
    PipelineStage,
    PublishedFeedItem,
    VerificationResult,
    VerificationStatus,
)
from .providers import ChatGPTProvider, ClaudeProvider, GeminiProvider, GrokProvider

logger = logging.getLogger(__name__)


class NewsPipelineOrchestrator:
    """
    Orchestrates the multi-model news pipeline.

    Coordinates the flow of news stories through discovery, verification,
    formatting, and editorial stages using specialized AI models.
    """

    def __init__(
        self,
        min_confidence: int = 6,
        max_concurrent_verifications: int = 5,
        publish_callback: Any = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            min_confidence: Minimum verification confidence to proceed (0-10)
            max_concurrent_verifications: Max parallel verification tasks
            publish_callback: Optional callback for publishing finalized stories
        """
        self.min_confidence = min_confidence
        self.max_concurrent = max_concurrent_verifications
        self.publish_callback = publish_callback

        # Initialize providers
        self.grok = GrokProvider()
        self.gemini = GeminiProvider()
        self.chatgpt = ChatGPTProvider()
        self.claude = ClaudeProvider()

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all providers."""
        logger.info("Initializing pipeline providers...")

        await asyncio.gather(
            self.grok.initialize(),
            self.gemini.initialize(),
            self.chatgpt.initialize(),
            self.claude.initialize(),
        )

        self._initialized = True
        logger.info("All providers initialized successfully")

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        results = await asyncio.gather(
            self.grok.health_check(),
            self.gemini.health_check(),
            self.chatgpt.health_check(),
            self.claude.health_check(),
        )

        return {
            "grok": results[0],
            "gemini": results[1],
            "chatgpt": results[2],
            "claude": results[3],
        }

    async def run_pipeline(
        self,
        request: DiscoveryRequest,
    ) -> PipelineRun:
        """
        Run the complete news pipeline.

        Args:
            request: Discovery request with tickers, categories, time window

        Returns:
            PipelineRun with all processed stories
        """
        run_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow()

        logger.info(f"[Pipeline {run_id}] Starting pipeline run")
        logger.info(f"[Pipeline {run_id}] Tickers: {request.tickers}")
        logger.info(f"[Pipeline {run_id}] Categories: {request.categories}")
        logger.info(f"[Pipeline {run_id}] Time window: {request.time_window}")

        if not self._initialized:
            await self.initialize()

        # Stage 1: Discovery with Grok
        logger.info(f"[Pipeline {run_id}] Stage 1: Discovery (Grok)")
        candidates = await self._discover(request)
        logger.info(f"[Pipeline {run_id}] Discovered {len(candidates)} candidates")

        # Stage 2: Verification with Gemini
        logger.info(f"[Pipeline {run_id}] Stage 2: Verification (Gemini)")
        verified = await self._verify_batch(candidates)
        passed = [v for v in verified if v.confidence >= self.min_confidence]
        logger.info(
            f"[Pipeline {run_id}] Verified {len(verified)}, "
            f"{len(passed)} passed threshold ({self.min_confidence}+)"
        )

        # Stage 3: Formatting with ChatGPT
        logger.info(f"[Pipeline {run_id}] Stage 3: Formatting (ChatGPT)")
        formatted = await self._format_batch(passed)
        logger.info(f"[Pipeline {run_id}] Formatted {len(formatted)} stories")

        # Stage 4: Editorial with Claude
        logger.info(f"[Pipeline {run_id}] Stage 4: Editorial (Claude)")
        finalized = await self._finalize_batch(formatted)
        logger.info(f"[Pipeline {run_id}] Finalized {len(finalized)} stories")

        # Stage 5: Publishing
        published = []
        if self.publish_callback and finalized:
            logger.info(f"[Pipeline {run_id}] Stage 5: Publishing")
            published = await self._publish_batch(finalized)
            logger.info(f"[Pipeline {run_id}] Published {len(published)} stories")

        completed_at = datetime.utcnow()
        duration = (completed_at - started_at).total_seconds()

        logger.info(f"[Pipeline {run_id}] Pipeline complete in {duration:.1f}s")

        return PipelineRun(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            request=request,
            discovered_count=len(candidates),
            verified_count=len(passed),
            published_count=len(published),
            stories=finalized,
        )

    async def _discover(self, request: DiscoveryRequest) -> list[CandidateStory]:
        """Stage 1: Discover news with Grok."""
        raw_stories = await self.grok.discover_news(
            tickers=request.tickers,
            categories=request.categories,
            time_window=request.time_window,
            max_results=request.max_results,
        )

        # Convert to CandidateStory objects
        candidates = []
        for story in raw_stories:
            candidates.append(
                CandidateStory(
                    story_id=str(uuid.uuid4())[:8],
                    headline=story.get("headline", ""),
                    source_name=story.get("source_name", "Unknown"),
                    source_url=story.get("source_url", ""),
                    published_at=story.get("published_at"),
                    tickers=story.get("tickers", []),
                    category=story.get("category", ""),
                    snippet=story.get("snippet", ""),
                    discovered_at=datetime.utcnow(),
                )
            )

        return candidates

    async def _verify_batch(
        self, candidates: list[CandidateStory]
    ) -> list[VerificationResult]:
        """Stage 2: Verify stories with Gemini."""
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def verify_one(candidate: CandidateStory) -> VerificationResult:
            async with semaphore:
                result = await self.gemini.verify_story(
                    headline=candidate.headline,
                    source_url=candidate.source_url,
                    tickers=candidate.tickers,
                    category=candidate.category,
                )

                # Map status string to enum
                status_map = {
                    "verified": VerificationStatus.VERIFIED,
                    "partial": VerificationStatus.PARTIAL,
                    "unverified": VerificationStatus.UNVERIFIED,
                    "contradicted": VerificationStatus.CONTRADICTED,
                    "error": VerificationStatus.UNVERIFIED,
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
                    contradictions=result.get("contradictions", []),
                    evidence=[],  # Could parse evidence objects here
                    verification_notes=result.get("verification_notes", ""),
                    verified_at=datetime.utcnow(),
                )

        results = await asyncio.gather(
            *[verify_one(c) for c in candidates],
            return_exceptions=True,
        )

        # Filter out exceptions
        verified = []
        for r in results:
            if isinstance(r, VerificationResult):
                verified.append(r)
            else:
                logger.error(f"Verification error: {r}")

        return verified

    async def _format_batch(
        self, verified: list[VerificationResult]
    ) -> list[FormattedStory]:
        """Stage 3: Format stories with ChatGPT."""

        async def format_one(v: VerificationResult) -> FormattedStory | None:
            try:
                result = await self.chatgpt.format_story(
                    story_id=v.story_id,
                    verified_facts=v.verified_facts,
                    verification_confidence=v.confidence,
                    affected_assets=v.candidate.tickers,
                    category=v.candidate.category,
                )

                return FormattedStory(
                    story_id=v.story_id,
                    title=result.get("title", ""),
                    what_happened=result.get("what_happened", ""),
                    bullet_points=result.get("bullet_points", []),
                    affected_assets=result.get("affected_assets", []),
                    category=v.candidate.category,
                    verification_confidence=v.confidence,
                    formatted_at=datetime.utcnow(),
                )
            except Exception as e:
                logger.error(f"Formatting error for {v.story_id}: {e}")
                return None

        results = await asyncio.gather(*[format_one(v) for v in verified])
        return [r for r in results if r is not None]

    async def _finalize_batch(
        self, formatted: list[FormattedStory]
    ) -> list[FinalizedStory]:
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
                    finalized_at=datetime.utcnow(),
                )
            except Exception as e:
                logger.error(f"Finalization error for {f.story_id}: {e}")
                return None

        results = await asyncio.gather(*[finalize_one(f) for f in formatted])
        return [r for r in results if r is not None]

    async def _publish_batch(
        self, finalized: list[FinalizedStory]
    ) -> list[PublishedFeedItem]:
        """Stage 5: Publish stories via callback."""
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
                    feed_url="",  # Set by publisher
                )

                # Call the publish callback
                if asyncio.iscoroutinefunction(self.publish_callback):
                    result = await self.publish_callback(feed_item)
                else:
                    result = self.publish_callback(feed_item)

                if result:
                    published.append(feed_item)

            except Exception as e:
                logger.error(f"Publishing error for {story.story_id}: {e}")

        return published
