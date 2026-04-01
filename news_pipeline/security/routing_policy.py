"""
Routing Policy

Determines where content can be processed based on sensitivity.

Routing Decisions:
- LOCAL_ONLY: Process with local LLM only (Ollama)
- SANITIZE_EXTERNAL: Sanitize then send to browser providers
- EXTERNAL_OK: Safe to send to any provider
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from news_pipeline.browser_agents.base import ModelType
from news_pipeline.security.classifier import (
    ClassificationResult,
    DataSensitivity,
    SensitivityClassifier,
)
from news_pipeline.security.sanitizer import SanitizationResult, Sanitizer

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Where content can be processed."""

    LOCAL_ONLY = "local_only"  # Tier C - only local LLM
    SANITIZE_EXTERNAL = "sanitize_external"  # Tier B - sanitize first
    EXTERNAL_OK = "external_ok"  # Tier A - any provider


@dataclass
class RoutingResult:
    """Result of routing policy evaluation."""

    decision: RoutingDecision
    allowed_providers: list[ModelType]
    requires_sanitization: bool
    original_tier: DataSensitivity
    sanitized_content: str | None = None
    sanitization_result: SanitizationResult | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def can_use_external(self) -> bool:
        """Check if external providers can be used."""
        return self.decision != RoutingDecision.LOCAL_ONLY

    @property
    def ready_content(self) -> str | None:
        """Get content ready for routing (sanitized if needed)."""
        if self.sanitization_result:
            return self.sanitization_result.sanitized
        return self.sanitized_content


class RoutingPolicy:
    """
    Evaluates and enforces routing policies based on data sensitivity.

    Ensures sensitive data stays local while maximizing use of
    capable external providers for non-sensitive content.
    """

    # Provider capabilities by tier
    PROVIDER_TIERS = {
        DataSensitivity.TIER_A: [
            ModelType.CLAUDE,
            ModelType.CHATGPT,
            ModelType.GEMINI,
            ModelType.LOCAL,
        ],
        DataSensitivity.TIER_B: [
            ModelType.CLAUDE,  # After sanitization
            ModelType.CHATGPT,  # After sanitization
            ModelType.GEMINI,  # After sanitization
            ModelType.LOCAL,
        ],
        DataSensitivity.TIER_C: [
            ModelType.LOCAL,  # Only local
        ],
    }

    # Task-specific routing preferences
    TASK_ROUTING = {
        "verification": {
            "preferred": [ModelType.CLAUDE],  # Claude has web search
            "fallback": [ModelType.CHATGPT, ModelType.GEMINI],
            "min_tier": DataSensitivity.TIER_A,  # Verification needs external search
        },
        "formatting": {
            "preferred": [ModelType.CHATGPT],
            "fallback": [ModelType.CLAUDE, ModelType.LOCAL],
            "min_tier": DataSensitivity.TIER_B,  # Can sanitize and format
        },
        "editorial": {
            "preferred": [ModelType.CLAUDE],
            "fallback": [ModelType.CHATGPT],
            "min_tier": DataSensitivity.TIER_A,  # Editorial should use full context
        },
        "preprocessing": {
            "preferred": [ModelType.LOCAL],  # Cheap and fast
            "fallback": [],
            "min_tier": DataSensitivity.TIER_C,  # Can handle anything
        },
        "analysis": {
            "preferred": [ModelType.CLAUDE, ModelType.CHATGPT],
            "fallback": [ModelType.GEMINI, ModelType.LOCAL],
            "min_tier": DataSensitivity.TIER_B,
        },
    }

    def __init__(
        self,
        classifier: SensitivityClassifier | None = None,
        sanitizer: Sanitizer | None = None,
        strict_mode: bool = True,
        auto_sanitize: bool = True,
    ):
        """
        Initialize routing policy.

        Args:
            classifier: Sensitivity classifier
            sanitizer: Content sanitizer
            strict_mode: Err on side of caution
            auto_sanitize: Automatically sanitize Tier B content
        """
        self.classifier = classifier or SensitivityClassifier(strict_mode=strict_mode)
        self.sanitizer = sanitizer or Sanitizer(classifier=self.classifier)
        self.strict_mode = strict_mode
        self.auto_sanitize = auto_sanitize

    def evaluate(
        self,
        content: str,
        task: str = "analysis",
        force_local: bool = False,
    ) -> RoutingResult:
        """
        Evaluate routing for content.

        Args:
            content: Content to route
            task: Task type (verification, formatting, etc.)
            force_local: Force local-only processing

        Returns:
            RoutingResult with decision and allowed providers
        """
        # Classify content
        classification = self.classifier.classify(content)
        original_tier = classification.tier

        warnings = []

        # Force local if requested
        if force_local:
            return RoutingResult(
                decision=RoutingDecision.LOCAL_ONLY,
                allowed_providers=[ModelType.LOCAL],
                requires_sanitization=False,
                original_tier=original_tier,
                sanitized_content=content,
                warnings=["Forced local processing"],
            )

        # Get task routing info
        task_info = self.TASK_ROUTING.get(
            task,
            {
                "preferred": [ModelType.CLAUDE, ModelType.CHATGPT],
                "fallback": [ModelType.LOCAL],
                "min_tier": DataSensitivity.TIER_B,
            },
        )

        # Tier C - local only
        if classification.tier == DataSensitivity.TIER_C:
            warnings.append("Tier C content detected - local processing only")
            if task_info["min_tier"] == DataSensitivity.TIER_A:
                warnings.append(f"Task '{task}' may not produce optimal results locally")

            return RoutingResult(
                decision=RoutingDecision.LOCAL_ONLY,
                allowed_providers=[ModelType.LOCAL],
                requires_sanitization=False,
                original_tier=original_tier,
                sanitized_content=content,
                warnings=warnings,
            )

        # Tier B - sanitize for external use
        if classification.tier == DataSensitivity.TIER_B:
            sanitization_result = None

            if self.auto_sanitize:
                sanitization_result = self.sanitizer.sanitize(content)

                if sanitization_result.sensitivity_tier == DataSensitivity.TIER_A:
                    # Successfully sanitized to Tier A
                    return RoutingResult(
                        decision=RoutingDecision.SANITIZE_EXTERNAL,
                        allowed_providers=task_info["preferred"] + task_info["fallback"],
                        requires_sanitization=True,
                        original_tier=original_tier,
                        sanitization_result=sanitization_result,
                        warnings=[f"Sanitized {sanitization_result.redaction_count} items"],
                    )
                else:
                    # Could not fully sanitize
                    warnings.append("Could not fully sanitize to Tier A")

            # Cannot or did not sanitize - check task requirements
            if task_info["min_tier"] in (DataSensitivity.TIER_B, DataSensitivity.TIER_C):
                # Task accepts Tier B
                allowed = [p for p in task_info["preferred"] + task_info["fallback"]]
                return RoutingResult(
                    decision=RoutingDecision.SANITIZE_EXTERNAL,
                    allowed_providers=allowed,
                    requires_sanitization=True,
                    original_tier=original_tier,
                    sanitization_result=sanitization_result,
                    warnings=warnings,
                )
            else:
                # Task requires Tier A, fall back to local
                warnings.append(f"Task '{task}' requires Tier A, using local")
                return RoutingResult(
                    decision=RoutingDecision.LOCAL_ONLY,
                    allowed_providers=[ModelType.LOCAL],
                    requires_sanitization=False,
                    original_tier=original_tier,
                    sanitized_content=content,
                    warnings=warnings,
                )

        # Tier A - external OK
        return RoutingResult(
            decision=RoutingDecision.EXTERNAL_OK,
            allowed_providers=task_info["preferred"] + task_info["fallback"],
            requires_sanitization=False,
            original_tier=original_tier,
            sanitized_content=content,
            warnings=[],
        )

    def get_provider_for_task(
        self,
        content: str,
        task: str,
        available_providers: list[ModelType],
    ) -> tuple[ModelType | None, str | None]:
        """
        Get best available provider for a task.

        Args:
            content: Content to process
            task: Task type
            available_providers: List of available providers

        Returns:
            Tuple of (provider, content_to_send)
            Returns (None, None) if no suitable provider
        """
        routing = self.evaluate(content, task)

        # Find intersection of allowed and available
        candidates = [p for p in routing.allowed_providers if p in available_providers]

        if not candidates:
            logger.warning(
                f"[RoutingPolicy] No suitable provider for task '{task}' "
                f"(allowed: {routing.allowed_providers}, available: {available_providers})"
            )
            return None, None

        # Get task preferences
        task_info = self.TASK_ROUTING.get(task, {})
        preferred = task_info.get("preferred", [])

        # Return first preferred that's available, or first candidate
        for provider in preferred:
            if provider in candidates:
                return provider, routing.ready_content

        return candidates[0], routing.ready_content

    def can_process_externally(self, content: str, task: str = "analysis") -> bool:
        """Quick check if content can be processed externally."""
        routing = self.evaluate(content, task)
        return routing.can_use_external


# Module-level instance
_routing_policy: RoutingPolicy | None = None


def get_routing_policy() -> RoutingPolicy:
    """Get the default routing policy instance."""
    global _routing_policy
    if _routing_policy is None:
        _routing_policy = RoutingPolicy()
    return _routing_policy
