"""
Fallback Handler

Manages graceful degradation when providers fail.

Features:
- Automatic fallback to alternative providers
- Degraded mode pipeline completion
- Manual takeover mode
- Partial completion handling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from news_pipeline.browser_agents.base import ModelType
from news_pipeline.recovery.checkpoints import PipelineStage
from news_pipeline.recovery.health import HealthStatus, ProviderHealthMonitor

logger = logging.getLogger(__name__)


class DegradedMode(Enum):
    """Degraded operation modes."""

    NORMAL = "normal"  # All providers available
    PARTIAL = "partial"  # Some providers unavailable
    LOCAL_ONLY = "local_only"  # Only local LLM available
    MANUAL = "manual"  # Requires manual intervention
    OFFLINE = "offline"  # No providers available


@dataclass
class DegradedModeConfig:
    """Configuration for degraded mode operation."""

    mode: DegradedMode = DegradedMode.NORMAL

    # Fallback chains per task
    fallback_chains: dict[str, list[ModelType]] = field(default_factory=dict)

    # What to skip when degraded
    skip_verification: bool = False  # Skip external verification
    skip_editorial: bool = False  # Skip editorial enhancement
    skip_formatting: bool = False  # Skip AI formatting

    # Local processing options
    use_local_verification: bool = True
    use_local_formatting: bool = True

    # Manual mode
    await_manual_input: bool = False
    manual_review_stages: list[PipelineStage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "skip_verification": self.skip_verification,
            "skip_editorial": self.skip_editorial,
            "skip_formatting": self.skip_formatting,
            "use_local_verification": self.use_local_verification,
            "use_local_formatting": self.use_local_formatting,
            "await_manual_input": self.await_manual_input,
            "manual_review_stages": [s.value for s in self.manual_review_stages],
        }


@dataclass
class FallbackResult:
    """Result of a fallback attempt."""

    success: bool
    provider_used: ModelType | None
    providers_tried: list[ModelType] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    degraded_mode: DegradedMode = DegradedMode.NORMAL


class FallbackHandler:
    """
    Handles fallback logic and degraded operation.

    Provides:
    - Automatic provider fallback chains
    - Degraded mode configuration
    - Manual takeover support
    - Partial completion tracking
    """

    # Default fallback chains per task
    DEFAULT_FALLBACK_CHAINS = {
        "verification": [ModelType.CLAUDE, ModelType.CHATGPT, ModelType.GEMINI, ModelType.LOCAL],
        "formatting": [ModelType.CHATGPT, ModelType.CLAUDE, ModelType.GEMINI, ModelType.LOCAL],
        "editorial": [ModelType.CLAUDE, ModelType.CHATGPT, ModelType.LOCAL],
        "preprocessing": [ModelType.LOCAL],
        "analysis": [ModelType.CLAUDE, ModelType.CHATGPT, ModelType.GEMINI, ModelType.LOCAL],
    }

    def __init__(
        self,
        health_monitor: ProviderHealthMonitor | None = None,
        auto_degrade: bool = True,
    ):
        """
        Initialize fallback handler.

        Args:
            health_monitor: Provider health monitor
            auto_degrade: Automatically enter degraded mode on failures
        """
        self._health_monitor = health_monitor
        self.auto_degrade = auto_degrade
        self._config = DegradedModeConfig()
        self._manual_queue: list[dict] = []

    @property
    def health_monitor(self) -> ProviderHealthMonitor:
        """Get health monitor, initializing if needed."""
        if self._health_monitor is None:
            from news_pipeline.recovery.health import get_health_monitor

            self._health_monitor = get_health_monitor()
        return self._health_monitor

    @property
    def current_mode(self) -> DegradedMode:
        """Get current degraded mode."""
        return self._config.mode

    @property
    def config(self) -> DegradedModeConfig:
        """Get current config."""
        return self._config

    def evaluate_mode(self) -> DegradedMode:
        """
        Evaluate current degraded mode based on provider health.

        Returns:
            Current DegradedMode
        """
        available = self.health_monitor.get_available_providers()

        if not available:
            self._config.mode = DegradedMode.OFFLINE
        elif ModelType.LOCAL in available and len(available) == 1:
            self._config.mode = DegradedMode.LOCAL_ONLY
        elif len(available) < len(ModelType):
            self._config.mode = DegradedMode.PARTIAL
        else:
            self._config.mode = DegradedMode.NORMAL

        return self._config.mode

    def configure_degraded_mode(
        self,
        mode: DegradedMode | None = None,
        skip_verification: bool | None = None,
        skip_editorial: bool | None = None,
        skip_formatting: bool | None = None,
        manual_stages: list[PipelineStage] | None = None,
    ) -> DegradedModeConfig:
        """
        Configure degraded mode operation.

        Args:
            mode: Force specific mode
            skip_verification: Skip verification stage
            skip_editorial: Skip editorial stage
            skip_formatting: Skip formatting stage
            manual_stages: Stages requiring manual review

        Returns:
            Updated configuration
        """
        if mode is not None:
            self._config.mode = mode

        if skip_verification is not None:
            self._config.skip_verification = skip_verification

        if skip_editorial is not None:
            self._config.skip_editorial = skip_editorial

        if skip_formatting is not None:
            self._config.skip_formatting = skip_formatting

        if manual_stages is not None:
            self._config.manual_review_stages = manual_stages
            if manual_stages:
                self._config.await_manual_input = True

        # Auto-configure based on mode
        if self._config.mode == DegradedMode.LOCAL_ONLY:
            self._config.use_local_verification = True
            self._config.use_local_formatting = True

        elif self._config.mode == DegradedMode.OFFLINE:
            self._config.skip_verification = True
            self._config.skip_editorial = True

        elif self._config.mode == DegradedMode.MANUAL:
            self._config.await_manual_input = True

        logger.info(f"[Fallback] Configured mode: {self._config.mode.value}")

        return self._config

    def get_fallback_chain(self, task: str) -> list[ModelType]:
        """
        Get fallback chain for a task.

        Args:
            task: Task name

        Returns:
            List of providers to try in order
        """
        # Use custom chain if configured
        if task in self._config.fallback_chains:
            return self._config.fallback_chains[task]

        # Use default chain
        return self.DEFAULT_FALLBACK_CHAINS.get(
            task,
            [ModelType.CLAUDE, ModelType.CHATGPT, ModelType.LOCAL],
        )

    def get_next_provider(
        self,
        task: str,
        tried: list[ModelType] | None = None,
    ) -> ModelType | None:
        """
        Get next provider to try for a task.

        Args:
            task: Task name
            tried: Providers already tried

        Returns:
            Next provider to try or None
        """
        tried_set = set(tried or [])
        chain = self.get_fallback_chain(task)
        available = set(self.health_monitor.get_available_providers())

        for provider in chain:
            if provider not in tried_set and provider in available:
                return provider

        return None

    async def execute_with_fallback(
        self,
        task: str,
        execute_fn,
        max_retries: int = 3,
    ) -> FallbackResult:
        """
        Execute a task with automatic fallback.

        Args:
            task: Task name
            execute_fn: Async function(provider) -> result
            max_retries: Max retries per provider

        Returns:
            FallbackResult with outcome
        """
        tried = []
        last_error = None

        while True:
            provider = self.get_next_provider(task, tried)

            if provider is None:
                # No more providers to try
                break

            tried.append(provider)
            logger.info(f"[Fallback] Trying {provider.value} for task '{task}'")

            for attempt in range(max_retries):
                try:
                    result = await execute_fn(provider)

                    # Record success
                    self.health_monitor.record_request(provider, success=True)

                    return FallbackResult(
                        success=True,
                        provider_used=provider,
                        providers_tried=tried,
                        result=result,
                        degraded_mode=self._config.mode,
                    )

                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        f"[Fallback] {provider.value} failed (attempt {attempt + 1}): {e}"
                    )

                    # Check for rate limiting
                    if "rate limit" in last_error.lower():
                        self.health_monitor.mark_rate_limited(provider)
                        break  # Move to next provider

                    # Record failure
                    self.health_monitor.record_request(provider, success=False)

            # All retries failed for this provider
            self.health_monitor.mark_unhealthy(provider, last_error or "Unknown error")

        # All providers exhausted
        return FallbackResult(
            success=False,
            provider_used=None,
            providers_tried=tried,
            error=last_error or "All providers exhausted",
            degraded_mode=self.evaluate_mode(),
        )

    def should_skip_stage(self, stage: PipelineStage) -> bool:
        """
        Check if a stage should be skipped in current mode.

        Args:
            stage: Pipeline stage

        Returns:
            True if stage should be skipped
        """
        if self._config.mode == DegradedMode.OFFLINE:
            # In offline mode, skip all AI-dependent stages
            return stage in (
                PipelineStage.VERIFY,
                PipelineStage.FORMAT,
                PipelineStage.EDITORIAL,
            )

        if self._config.skip_verification and stage == PipelineStage.VERIFY:
            return True

        if self._config.skip_formatting and stage == PipelineStage.FORMAT:
            return True

        if self._config.skip_editorial and stage == PipelineStage.EDITORIAL:
            return True

        return False

    def requires_manual_review(self, stage: PipelineStage) -> bool:
        """Check if stage requires manual review."""
        return stage in self._config.manual_review_stages

    def queue_for_manual_review(
        self,
        stage: PipelineStage,
        item: dict,
        reason: str,
    ) -> None:
        """
        Queue an item for manual review.

        Args:
            stage: Current stage
            item: Item data
            reason: Why manual review needed
        """
        self._manual_queue.append(
            {
                "stage": stage.value,
                "item": item,
                "reason": reason,
                "queued_at": datetime.utcnow().isoformat(),
            }
        )
        logger.info(f"[Fallback] Queued item for manual review: {reason}")

    def get_manual_queue(self) -> list[dict]:
        """Get items queued for manual review."""
        return list(self._manual_queue)

    def clear_manual_queue(self) -> int:
        """Clear manual review queue."""
        count = len(self._manual_queue)
        self._manual_queue.clear()
        return count

    def enter_manual_mode(self, reason: str = "User requested") -> None:
        """Enter manual takeover mode."""
        self._config.mode = DegradedMode.MANUAL
        self._config.await_manual_input = True
        logger.warning(f"[Fallback] Entering manual mode: {reason}")

    def exit_manual_mode(self) -> None:
        """Exit manual mode."""
        self._config.await_manual_input = False
        self.evaluate_mode()  # Re-evaluate based on health
        logger.info(f"[Fallback] Exiting manual mode, now: {self._config.mode.value}")

    def get_partial_completion_plan(
        self,
        failed_stage: PipelineStage,
    ) -> dict:
        """
        Get plan for partial completion after failure.

        Args:
            failed_stage: Stage that failed

        Returns:
            Plan dict with recommendations
        """
        # Determine what can still be completed
        can_publish_raw = failed_stage.order > PipelineStage.PREPROCESS.order
        can_publish_unverified = failed_stage.order > PipelineStage.VERIFY.order

        recommendations = []

        if can_publish_raw:
            recommendations.append("Publish raw scraped items without AI enhancement")

        if can_publish_unverified:
            recommendations.append("Publish preprocessed items without verification")

        if self.health_monitor.get_health(ModelType.LOCAL).is_available:
            recommendations.append("Use local LLM for basic formatting")

        return {
            "failed_stage": failed_stage.value,
            "can_publish_raw": can_publish_raw,
            "can_publish_unverified": can_publish_unverified,
            "recommendations": recommendations,
            "manual_queue_size": len(self._manual_queue),
        }


# Module-level instance
_fallback_handler: FallbackHandler | None = None


def get_fallback_handler() -> FallbackHandler:
    """Get the default fallback handler instance."""
    global _fallback_handler
    if _fallback_handler is None:
        _fallback_handler = FallbackHandler()
    return _fallback_handler
