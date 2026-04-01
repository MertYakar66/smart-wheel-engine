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
"""

from news_pipeline.recovery.checkpoints import (
    Checkpoint,
    CheckpointManager,
    PipelineStage,
    get_checkpoint_manager,
)
from news_pipeline.recovery.fallbacks import (
    DegradedModeConfig,
    FallbackHandler,
    get_fallback_handler,
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
