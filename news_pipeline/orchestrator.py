"""
News Pipeline Orchestrator (Browser-Based) v2.1

Zero API cost architecture using browser automation.

Pipeline:
1. SCRAPE: RSS feeds + optional browser scraping
2. PREPROCESS: Local LLM or rules (filter, categorize)
3. VERIFY: Claude browser session (with web search)
4. FORMAT: ChatGPT browser session (structure content)
5. EDITORIAL: Claude browser session (why it matters)
6. PUBLISH: Save to database, export to dashboard

Security Features:
- Data sensitivity classification (Tier A/B/C)
- Prompt sanitization before external routing
- Sensitivity-based provider routing

Recovery Features:
- Pre-run health checks
- Stage-level checkpointing
- Resume from last successful stage
- Graceful degradation on failures

Fallback chain: Claude → ChatGPT → Gemini → Local
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from news_pipeline.browser_agents import (
    BrowserModelSession,
    ChatGPTBrowserAgent,
    ClaudeBrowserAgent,
    GeminiBrowserAgent,
    ModelType,
)
from news_pipeline.browser_agents.base import SessionManager
from news_pipeline.local_llm import LocalPreprocessor
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
from news_pipeline.recovery import (
    CheckpointManager,
    FallbackHandler,
    ProviderHealthMonitor,
)
from news_pipeline.recovery.checkpoints import PipelineStage as CheckpointStage
from news_pipeline.scrapers import NewsAggregator, NewsItem
from news_pipeline.security import (
    RoutingPolicy,
    Sanitizer,
    SensitivityClassifier,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the pipeline orchestrator."""

    # Verification
    min_confidence: int = 6
    max_stories_per_run: int = 20

    # Model preferences
    verification_models: list[ModelType] = field(
        default_factory=lambda: [ModelType.CLAUDE, ModelType.GEMINI]
    )
    formatting_models: list[ModelType] = field(
        default_factory=lambda: [ModelType.CHATGPT, ModelType.CLAUDE]
    )
    editorial_models: list[ModelType] = field(
        default_factory=lambda: [ModelType.CLAUDE, ModelType.CHATGPT]
    )

    # Browser settings
    headless: bool = True
    browser_data_dir: str = str(Path.home() / ".news_pipeline" / "browser_data")

    # Persistence
    state_dir: str = str(Path.home() / ".news_pipeline" / "state")
    output_dir: str = str(Path.home() / ".news_pipeline" / "output")
    checkpoint_dir: str = str(Path.home() / ".news_pipeline" / "checkpoints")

    # Local LLM
    use_local_llm: bool = True
    ollama_model: str = "qwen2.5:7b"

    # Security settings
    enable_sensitivity_check: bool = True
    strict_security_mode: bool = True
    auto_sanitize: bool = True

    # Recovery settings
    enable_checkpoints: bool = True
    enable_health_checks: bool = True
    auto_resume: bool = True
    max_retries_per_stage: int = 3
    allow_degraded_mode: bool = True


class NewsPipelineOrchestrator:
    """
    Main orchestrator for the browser-based news pipeline.

    Uses paid subscriptions via browser - zero API cost.

    Security:
    - Classifies data sensitivity before routing
    - Sanitizes prompts for external providers
    - Routes based on sensitivity tier

    Recovery:
    - Checkpoints after each stage
    - Resumes from last successful stage
    - Graceful degradation on failures
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        """Initialize orchestrator."""
        self.config = config or OrchestratorConfig()

        # Create directories
        Path(self.config.state_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Core components
        self.aggregator = NewsAggregator(use_rss=True, use_browser=False)
        self.preprocessor = LocalPreprocessor(
            model=self.config.ollama_model,
            use_llm=self.config.use_local_llm,
        )
        self.session_manager = SessionManager(self.config.state_dir)

        # Security components
        self.classifier = SensitivityClassifier(strict_mode=self.config.strict_security_mode)
        self.sanitizer = Sanitizer(classifier=self.classifier)
        self.routing_policy = RoutingPolicy(
            classifier=self.classifier,
            sanitizer=self.sanitizer,
            strict_mode=self.config.strict_security_mode,
            auto_sanitize=self.config.auto_sanitize,
        )

        # Recovery components
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.checkpoint_dir,
        )
        self.health_monitor = ProviderHealthMonitor()
        self.fallback_handler = FallbackHandler(
            health_monitor=self.health_monitor,
            auto_degrade=self.config.allow_degraded_mode,
        )

        # State
        self._current_stage = PipelineStage.DISCOVERY
        self._checkpoint_stage = CheckpointStage.INIT
        self._run_id: str = ""
        self._degraded_mode = False

    async def run(
        self,
        request: DiscoveryRequest,
        resume_run_id: str | None = None,
    ) -> PipelineResult:
        """
        Execute the complete pipeline.

        Args:
            request: Discovery parameters
            resume_run_id: Optional run ID to resume from

        Returns:
            PipelineResult with all processed stories
        """
        # Initialize or resume run
        if resume_run_id and self.config.auto_resume:
            self._run_id = self.checkpoint_manager.start_run(resume_run_id)
            resume_stage = self.checkpoint_manager.resume_stage
            logger.info(f"[Pipeline {self._run_id}] Resuming from {resume_stage.value}")
        else:
            self._run_id = str(uuid.uuid4())[:8]
            self.checkpoint_manager.start_run(self._run_id)
            resume_stage = CheckpointStage.INIT

        started_at = datetime.utcnow()

        result = PipelineResult(
            run_id=self._run_id,
            started_at=started_at,
            request=request,
        )

        logger.info(f"[Pipeline {self._run_id}] Starting")
        logger.info(f"[Pipeline {self._run_id}] Categories: {request.categories}")
        logger.info(f"[Pipeline {self._run_id}] Time window: {request.time_window}")

        # Pre-run health checks
        if self.config.enable_health_checks:
            await self._run_health_checks()

        try:
            # Stage 1: Scrape news (or load from checkpoint)
            self._current_stage = PipelineStage.DISCOVERY
            self._checkpoint_stage = CheckpointStage.SCRAPE

            if resume_stage.order <= CheckpointStage.SCRAPE.order:
                logger.info(f"[Pipeline {self._run_id}] Stage 1: Scraping news")
                news_items = await self._scrape_news(request)
                result.discovered_count = len(news_items)

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        CheckpointStage.SCRAPE,
                        {"items": [item.to_dict() for item in news_items]},
                        len(news_items),
                    )
            else:
                news_items = self._load_news_from_checkpoint()
                result.discovered_count = len(news_items)

            logger.info(f"[Pipeline {self._run_id}] Scraped {len(news_items)} items")

            if not news_items:
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                self.checkpoint_manager.complete_run()
                return result

            # Stage 2: Preprocess (or load from checkpoint)
            self._checkpoint_stage = CheckpointStage.PREPROCESS

            if resume_stage.order <= CheckpointStage.PREPROCESS.order:
                logger.info(f"[Pipeline {self._run_id}] Stage 2: Preprocessing")
                candidates = await self._preprocess(news_items)

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        CheckpointStage.PREPROCESS,
                        {"candidates": [c.to_dict() for c in candidates]},
                        len(candidates),
                    )
            else:
                candidates = self._load_candidates_from_checkpoint()

            logger.info(f"[Pipeline {self._run_id}] Kept {len(candidates)} candidates")

            if not candidates:
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                self.checkpoint_manager.complete_run()
                return result

            # Stage 3: Verify with Claude (browser)
            self._current_stage = PipelineStage.VERIFICATION
            self._checkpoint_stage = CheckpointStage.VERIFY

            if self.fallback_handler.should_skip_stage(CheckpointStage.VERIFY):
                logger.warning(f"[Pipeline {self._run_id}] Skipping verification (degraded mode)")
                verified = self._create_unverified_results(candidates)
                passed = verified
            elif resume_stage.order <= CheckpointStage.VERIFY.order:
                logger.info(f"[Pipeline {self._run_id}] Stage 3: Verification (browser)")
                verified = await self._verify_batch_with_fallback(candidates)
                passed = [v for v in verified if v.confidence >= self.config.min_confidence]

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        CheckpointStage.VERIFY,
                        {"verified": [v.to_dict() for v in passed]},
                        len(passed),
                    )
            else:
                passed = self._load_verified_from_checkpoint()
                verified = passed

            result.verified_count = len(passed)
            logger.info(
                f"[Pipeline {self._run_id}] Verified {len(verified)}, "
                f"{len(passed)} passed threshold"
            )

            if not passed:
                result.status = "completed"
                result.completed_at = datetime.utcnow()
                self.checkpoint_manager.complete_run()
                return result

            # Stage 4: Format with ChatGPT (browser)
            self._current_stage = PipelineStage.FORMATTING
            self._checkpoint_stage = CheckpointStage.FORMAT

            if self.fallback_handler.should_skip_stage(CheckpointStage.FORMAT):
                logger.warning(f"[Pipeline {self._run_id}] Skipping formatting (degraded mode)")
                formatted = self._create_basic_formatted(passed)
            elif resume_stage.order <= CheckpointStage.FORMAT.order:
                logger.info(f"[Pipeline {self._run_id}] Stage 4: Formatting (browser)")
                formatted = await self._format_batch_with_fallback(passed)

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        CheckpointStage.FORMAT,
                        {"formatted": [f.to_dict() for f in formatted]},
                        len(formatted),
                    )
            else:
                formatted = self._load_formatted_from_checkpoint()

            result.formatted_count = len(formatted)
            logger.info(f"[Pipeline {self._run_id}] Formatted {len(formatted)} stories")

            # Stage 5: Editorial with Claude (browser)
            self._current_stage = PipelineStage.EDITORIAL
            self._checkpoint_stage = CheckpointStage.EDITORIAL

            if self.fallback_handler.should_skip_stage(CheckpointStage.EDITORIAL):
                logger.warning(f"[Pipeline {self._run_id}] Skipping editorial (degraded mode)")
                finalized = self._create_basic_finalized(formatted)
            elif resume_stage.order <= CheckpointStage.EDITORIAL.order:
                logger.info(f"[Pipeline {self._run_id}] Stage 5: Editorial (browser)")
                finalized = await self._finalize_batch_with_fallback(formatted)

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        CheckpointStage.EDITORIAL,
                        {"finalized": [f.to_dict() for f in finalized]},
                        len(finalized),
                    )
            else:
                finalized = self._load_finalized_from_checkpoint()

            result.finalized_count = len(finalized)
            result.stories = finalized
            logger.info(f"[Pipeline {self._run_id}] Finalized {len(finalized)} stories")

            # Stage 6: Publish
            self._current_stage = PipelineStage.PUBLISHING
            self._checkpoint_stage = CheckpointStage.PUBLISH

            if resume_stage.order <= CheckpointStage.PUBLISH.order:
                logger.info(f"[Pipeline {self._run_id}] Stage 6: Publishing")
                published = await self._publish(finalized)
                result.published_count = len(published)

                if self.config.enable_checkpoints:
                    self._save_checkpoint(
                        CheckpointStage.PUBLISH,
                        {"published_ids": [s.story_id for s in published]},
                        len(published),
                    )
            else:
                published = finalized  # Already published
                result.published_count = len(published)

            logger.info(f"[Pipeline {self._run_id}] Published {len(published)} stories")

            result.status = "completed"
            self.checkpoint_manager.complete_run()

        except Exception as e:
            logger.error(f"[Pipeline {self._run_id}] Failed at {self._current_stage.value}: {e}")
            result.status = "failed"
            result.errors.append(str(e))

            # Record error in checkpoint
            if self.config.enable_checkpoints:
                self.checkpoint_manager.add_error(self._checkpoint_stage, str(e))

            # Check if we can partially complete
            if self.config.allow_degraded_mode:
                partial_plan = self.fallback_handler.get_partial_completion_plan(
                    self._checkpoint_stage
                )
                logger.info(
                    f"[Pipeline {self._run_id}] Partial completion options: "
                    f"{partial_plan['recommendations']}"
                )

        finally:
            result.completed_at = datetime.utcnow()
            self.session_manager.save_state()

        duration = result.duration_seconds
        mode_info = " [DEGRADED]" if self._degraded_mode else ""
        logger.info(
            f"[Pipeline {self._run_id}] {result.status.upper()}{mode_info} in {duration:.1f}s | "
            f"Scraped: {result.discovered_count} | "
            f"Verified: {result.verified_count} | "
            f"Published: {result.published_count}"
        )

        return result

    async def _run_health_checks(self) -> None:
        """Run pre-flight health checks on all providers."""
        logger.info(f"[Pipeline {self._run_id}] Running health checks...")

        await self.health_monitor.run_health_checks()
        available = self.health_monitor.get_available_providers()

        if not available:
            logger.warning("[Pipeline] No providers available - entering degraded mode")
            self._degraded_mode = True
            self.fallback_handler.configure_degraded_mode(
                skip_verification=True,
                skip_editorial=True,
            )
        elif ModelType.LOCAL in available and len(available) == 1:
            logger.warning("[Pipeline] Only local LLM available - limited capabilities")
            self._degraded_mode = True
            self.fallback_handler.evaluate_mode()
        else:
            logger.info(f"[Pipeline] {len(available)} providers available")

    def _save_checkpoint(
        self,
        stage: CheckpointStage,
        data: dict,
        item_count: int,
    ) -> None:
        """Save checkpoint for a stage."""
        self.checkpoint_manager.save_checkpoint(
            stage=stage,
            data=data,
            items_processed=item_count,
            items_total=item_count,
            metadata={"run_id": self._run_id},
        )

    def _load_news_from_checkpoint(self) -> list[NewsItem]:
        """Load news items from checkpoint."""
        cp = self.checkpoint_manager.get_checkpoint(CheckpointStage.SCRAPE)
        if not cp:
            return []

        from news_pipeline.scrapers.base import NewsCategory, SourceType

        items = []
        for item_data in cp.data.get("items", []):
            items.append(
                NewsItem(
                    headline=item_data["headline"],
                    source_name=item_data["source_name"],
                    source_url=item_data["source_url"],
                    source_type=SourceType(item_data.get("source_type", "unknown")),
                    snippet=item_data.get("snippet", ""),
                    tickers=item_data.get("tickers", []),
                    category=NewsCategory(item_data.get("category", "other")),
                )
            )
        return items

    def _load_candidates_from_checkpoint(self) -> list[CandidateStory]:
        """Load candidates from checkpoint."""
        cp = self.checkpoint_manager.get_checkpoint(CheckpointStage.PREPROCESS)
        if not cp:
            return []

        return [
            CandidateStory.from_dict(c)
            for c in cp.data.get("candidates", [])
        ]

    def _load_verified_from_checkpoint(self) -> list[VerificationResult]:
        """Load verified results from checkpoint."""
        cp = self.checkpoint_manager.get_checkpoint(CheckpointStage.VERIFY)
        if not cp:
            return []

        return [
            VerificationResult.from_dict(v)
            for v in cp.data.get("verified", [])
        ]

    def _load_formatted_from_checkpoint(self) -> list[FormattedStory]:
        """Load formatted stories from checkpoint."""
        cp = self.checkpoint_manager.get_checkpoint(CheckpointStage.FORMAT)
        if not cp:
            return []

        return [
            FormattedStory.from_dict(f)
            for f in cp.data.get("formatted", [])
        ]

    def _load_finalized_from_checkpoint(self) -> list[FinalizedStory]:
        """Load finalized stories from checkpoint."""
        cp = self.checkpoint_manager.get_checkpoint(CheckpointStage.EDITORIAL)
        if not cp:
            return []

        return [
            FinalizedStory.from_dict(f)
            for f in cp.data.get("finalized", [])
        ]

    def _create_unverified_results(
        self,
        candidates: list[CandidateStory],
    ) -> list[VerificationResult]:
        """Create unverified results for degraded mode."""
        return [
            VerificationResult(
                story_id=c.story_id,
                candidate=c,
                status=VerificationStatus.UNVERIFIED,
                confidence=5,  # Medium confidence
                verified_facts=[],
                what_happened=c.snippet or c.headline,
                verification_notes="Skipped in degraded mode",
            )
            for c in candidates
        ]

    def _create_basic_formatted(
        self,
        verified: list[VerificationResult],
    ) -> list[FormattedStory]:
        """Create basic formatted stories for degraded mode."""
        return [
            FormattedStory(
                story_id=v.story_id,
                title=v.candidate.headline,
                what_happened=v.what_happened,
                bullet_points=v.verified_facts or [v.what_happened],
                affected_assets=v.candidate.tickers,
                category=v.candidate.category,
                verification_confidence=v.confidence,
            )
            for v in verified
        ]

    def _create_basic_finalized(
        self,
        formatted: list[FormattedStory],
    ) -> list[FinalizedStory]:
        """Create basic finalized stories for degraded mode."""
        return [
            FinalizedStory(
                story_id=f.story_id,
                title=f.title,
                what_happened=f.what_happened,
                why_it_matters="[Editorial skipped in degraded mode]",
                bullet_points=f.bullet_points,
                affected_assets=f.affected_assets,
                category=f.category,
                verification_confidence=f.verification_confidence,
                priority=5,
                is_breaking=False,
            )
            for f in formatted
        ]

    async def _verify_batch_with_fallback(
        self,
        candidates: list[CandidateStory],
    ) -> list[VerificationResult]:
        """Verify batch with automatic fallback."""
        results = []

        for candidate in candidates:
            # Check sensitivity before routing
            if self.config.enable_sensitivity_check:
                routing = self.routing_policy.evaluate(
                    f"{candidate.headline}\n{candidate.snippet}",
                    task="verification",
                )

                if not routing.can_use_external:
                    logger.warning(
                        f"[Pipeline] Story {candidate.story_id} contains sensitive data - "
                        "using local verification"
                    )
                    result = await self._verify_locally(candidate)
                    results.append(result)
                    continue

            # Use fallback handler for external verification
            fallback_result = await self.fallback_handler.execute_with_fallback(
                task="verification",
                execute_fn=lambda provider, c=candidate: self._verify_with_provider(c, provider),
            )

            if fallback_result.success and fallback_result.result:
                results.append(fallback_result.result)
            else:
                logger.warning(f"[Pipeline] All verification providers failed for {candidate.story_id}")
                # Create unverified result
                results.append(
                    VerificationResult(
                        story_id=candidate.story_id,
                        candidate=candidate,
                        status=VerificationStatus.UNVERIFIED,
                        confidence=0,
                        what_happened=candidate.snippet or candidate.headline,
                        verification_notes=f"Verification failed: {fallback_result.error}",
                    )
                )

        return results

    async def _verify_with_provider(
        self,
        candidate: CandidateStory,
        provider: ModelType,
    ) -> VerificationResult:
        """Verify a candidate with a specific provider."""
        session = await self.session_manager.get_session(provider)

        if not session:
            raise Exception(f"No session available for {provider.value}")

        # Sanitize content if needed
        headline = candidate.headline
        snippet = candidate.snippet
        if self.config.auto_sanitize:
            headline, snippet = self.sanitizer.sanitize_news_item(headline, snippet)

        if isinstance(session, ClaudeBrowserAgent):
            result_data = await session.verify_news_story(
                headline=headline,
                source=candidate.source_url,
                tickers=candidate.tickers,
                category=candidate.category,
            )
        elif isinstance(session, GeminiBrowserAgent):
            result_data = await session.verify_with_search(
                headline=headline,
                source=candidate.source_url,
                tickers=candidate.tickers,
                category=candidate.category,
            )
        else:
            result_data = await self._verify_generic(session, candidate)

        if "error" in result_data:
            raise Exception(result_data["error"])

        status_map = {
            "verified": VerificationStatus.VERIFIED,
            "partial": VerificationStatus.PARTIAL,
            "unverified": VerificationStatus.UNVERIFIED,
            "contradicted": VerificationStatus.CONTRADICTED,
        }

        return VerificationResult(
            story_id=candidate.story_id,
            candidate=candidate,
            status=status_map.get(
                result_data.get("status", "unverified"),
                VerificationStatus.UNVERIFIED,
            ),
            confidence=result_data.get("confidence", 0),
            verified_facts=result_data.get("verified_facts", []),
            what_happened=result_data.get("what_happened", ""),
            contradictions=result_data.get("contradictions", []),
            verification_notes=result_data.get("verification_notes", ""),
        )

    async def _verify_locally(self, candidate: CandidateStory) -> VerificationResult:
        """Verify using local LLM only (for sensitive data)."""
        # Use local preprocessor for basic verification
        return VerificationResult(
            story_id=candidate.story_id,
            candidate=candidate,
            status=VerificationStatus.PARTIAL,
            confidence=5,
            verified_facts=[candidate.headline],
            what_happened=candidate.snippet or candidate.headline,
            verification_notes="Verified locally (sensitive data)",
        )

    async def _format_batch_with_fallback(
        self,
        verified: list[VerificationResult],
    ) -> list[FormattedStory]:
        """Format batch with automatic fallback."""
        results = []

        for v in verified:
            fallback_result = await self.fallback_handler.execute_with_fallback(
                task="formatting",
                execute_fn=lambda provider, item=v: self._format_with_provider(item, provider),
            )

            if fallback_result.success and fallback_result.result:
                results.append(fallback_result.result)
            else:
                # Use basic formatting
                results.append(
                    FormattedStory(
                        story_id=v.story_id,
                        title=v.candidate.headline,
                        what_happened=v.what_happened,
                        bullet_points=v.verified_facts or [v.what_happened],
                        affected_assets=v.candidate.tickers,
                        category=v.candidate.category,
                        verification_confidence=v.confidence,
                    )
                )

        return results

    async def _format_with_provider(
        self,
        v: VerificationResult,
        provider: ModelType,
    ) -> FormattedStory:
        """Format with a specific provider."""
        session = await self.session_manager.get_session(provider)

        if not session:
            raise Exception(f"No session available for {provider.value}")

        if isinstance(session, ChatGPTBrowserAgent):
            result_data = await session.format_story(
                verified_facts=v.verified_facts,
                what_happened=v.what_happened,
                affected_assets=v.candidate.tickers,
                category=v.candidate.category,
            )
        else:
            result_data = await self._format_generic(session, v)

        if "error" in result_data:
            raise Exception(result_data["error"])

        return FormattedStory(
            story_id=v.story_id,
            title=result_data.get("title", ""),
            what_happened=result_data.get("what_happened", v.what_happened),
            bullet_points=result_data.get("bullet_points", v.verified_facts),
            affected_assets=result_data.get("affected_assets", v.candidate.tickers),
            category=v.candidate.category,
            verification_confidence=v.confidence,
        )

    async def _finalize_batch_with_fallback(
        self,
        formatted: list[FormattedStory],
    ) -> list[FinalizedStory]:
        """Finalize batch with automatic fallback."""
        results = []

        for f in formatted:
            fallback_result = await self.fallback_handler.execute_with_fallback(
                task="editorial",
                execute_fn=lambda provider, item=f: self._finalize_with_provider(item, provider),
            )

            if fallback_result.success and fallback_result.result:
                results.append(fallback_result.result)
            else:
                # Use basic finalization
                results.append(
                    FinalizedStory(
                        story_id=f.story_id,
                        title=f.title,
                        what_happened=f.what_happened,
                        why_it_matters="[Analysis pending]",
                        bullet_points=f.bullet_points,
                        affected_assets=f.affected_assets,
                        category=f.category,
                        verification_confidence=f.verification_confidence,
                        priority=5,
                    )
                )

        return results

    async def _finalize_with_provider(
        self,
        f: FormattedStory,
        provider: ModelType,
    ) -> FinalizedStory:
        """Finalize with a specific provider."""
        session = await self.session_manager.get_session(provider)

        if not session:
            raise Exception(f"No session available for {provider.value}")

        if isinstance(session, ClaudeBrowserAgent):
            result_data = await session.add_editorial(
                title=f.title,
                what_happened=f.what_happened,
                bullet_points=f.bullet_points,
                affected_assets=f.affected_assets,
                category=f.category,
            )
        else:
            result_data = await self._editorial_generic(session, f)

        if "error" in result_data:
            raise Exception(result_data["error"])

        return FinalizedStory(
            story_id=f.story_id,
            title=result_data.get("title", f.title),
            what_happened=result_data.get("what_happened", f.what_happened),
            why_it_matters=result_data.get("why_it_matters", ""),
            bullet_points=result_data.get("bullet_points", f.bullet_points),
            affected_assets=result_data.get("affected_assets", f.affected_assets),
            category=f.category,
            verification_confidence=f.verification_confidence,
            market_implications=result_data.get("market_implications"),
            trading_considerations=result_data.get("trading_considerations"),
            is_breaking=result_data.get("is_breaking", False),
            priority=result_data.get("priority", 5),
            tags=result_data.get("tags", []),
        )

    async def _scrape_news(self, request: DiscoveryRequest) -> list[NewsItem]:
        """Stage 1: Scrape news from sources."""
        hours_map = {
            "overnight": 12,
            "last_1h": 1,
            "last_3h": 3,
            "last_6h": 6,
            "today": 24,
            "yesterday": 48,
        }

        return await self.aggregator.fetch_news(
            categories=list(request.categories),
            tickers=list(request.tickers) if request.tickers else None,
            max_items=self.config.max_stories_per_run * 3,  # Extra for filtering
            hours_back=hours_map.get(request.time_window, 12),
        )

    async def _preprocess(self, items: list[NewsItem]) -> list[CandidateStory]:
        """Stage 2: Preprocess with local LLM or rules."""
        filtered = await self.preprocessor.preprocess_batch(
            items,
            filter_threshold=0.3,
        )

        # Convert to CandidateStory
        candidates = []
        for item in filtered:
            candidates.append(
                CandidateStory(
                    story_id=str(uuid.uuid4())[:8],
                    headline=item.headline,
                    source_name=item.source_name,
                    source_url=item.source_url,
                    snippet=item.snippet,
                    tickers=item.tickers,
                    category=item.category.value,
                    source_type=item.source_type.value,
                    discovered_at=datetime.utcnow(),
                )
            )

        return candidates[: self.config.max_stories_per_run]

    async def _publish(self, stories: list[FinalizedStory]) -> list[FinalizedStory]:
        """Stage 6: Save stories to output."""
        # Save to JSON file
        output_file = Path(self.config.output_dir) / f"run_{self._run_id}.json"

        output_data = {
            "run_id": self._run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "stories": [s.to_dict() for s in stories],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"[Pipeline] Saved to {output_file}")

        return stories

    async def _get_session(
        self,
        preferred_models: list[ModelType],
    ) -> BrowserModelSession | None:
        """Get an available browser session with fallback."""
        for model_type in preferred_models:
            session = await self.session_manager.get_session(model_type)
            if session and session.is_ready:
                logger.info(f"[Pipeline] Using {model_type.value} session")
                return session
            elif session and not session.is_authenticated:
                logger.warning(
                    f"[Pipeline] {model_type.value} needs authentication. "
                    f"Run: python morning_run.py --setup-auth"
                )

        return None

    async def _verify_generic(
        self,
        session: BrowserModelSession,
        candidate: CandidateStory,
    ) -> dict:
        """Generic verification prompt."""
        prompt = f"""Verify this financial news:

HEADLINE: "{candidate.headline}"
SOURCE: {candidate.source_url}

Return JSON:
{{"status": "verified/partial/unverified", "confidence": 0-10, "verified_facts": [], "what_happened": ""}}"""

        response = await session.send_prompt_with_retry(prompt)
        if response.success:
            return self._parse_json(response.content)
        return {"error": response.error}

    async def _format_generic(
        self,
        session: BrowserModelSession,
        v: VerificationResult,
    ) -> dict:
        """Generic formatting prompt."""
        prompt = f"""Format this news:

FACTS: {v.verified_facts}
SUMMARY: {v.what_happened}

Return JSON:
{{"title": "", "what_happened": "", "bullet_points": []}}"""

        response = await session.send_prompt_with_retry(prompt)
        if response.success:
            return self._parse_json(response.content)
        return {"error": response.error}

    async def _editorial_generic(
        self,
        session: BrowserModelSession,
        f: FormattedStory,
    ) -> dict:
        """Generic editorial prompt."""
        prompt = f"""Add editorial analysis:

TITLE: {f.title}
WHAT HAPPENED: {f.what_happened}

Return JSON:
{{"why_it_matters": "", "priority": 1-10, "is_breaking": false}}"""

        response = await session.send_prompt_with_retry(prompt)
        if response.success:
            return self._parse_json(response.content)
        return {"error": response.error}

    def _parse_json(self, content: str) -> dict:
        """Parse JSON from response."""
        try:
            content = content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "JSON parse failed"}

    async def close(self) -> None:
        """Close all sessions."""
        await self.session_manager.close_all()
        await self.aggregator.close()
