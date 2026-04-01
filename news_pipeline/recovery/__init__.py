"""
Recovery Module

Checkpoint management, health monitoring, and failure recovery
for the news pipeline orchestrator.

Features:
- Stage-level checkpointing
- Resume from last successful stage
- Provider health monitoring
- Graceful degradation
- Idempotent reruns

Note: FallbackHandler requires playwright for browser agents.
"""

# Core recovery components (no playwright dependency)
from news_pipeline.recovery.checkpoints import (
    Checkpoint,
    CheckpointManager,
    PipelineStage,
    get_checkpoint_manager,
)
from news_pipeline.recovery.health import (
    HealthStatus,
    ProviderHealth,
    ProviderHealthMonitor,
    get_health_monitor,
)

__all__ = [
    "PipelineStage",
    "Checkpoint",
    "CheckpointManager",
    "get_checkpoint_manager",
    "ProviderHealth",
    "HealthStatus",
    "ProviderHealthMonitor",
    "get_health_monitor",
    "DegradedModeConfig",
    "FallbackHandler",
    "get_fallback_handler",
]


def __getattr__(name: str):
    """Lazy import fallback components that require playwright."""
    fallback_exports = {"DegradedModeConfig", "FallbackHandler", "get_fallback_handler"}
    if name in fallback_exports:
        from news_pipeline.recovery import fallbacks
        return getattr(fallbacks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
