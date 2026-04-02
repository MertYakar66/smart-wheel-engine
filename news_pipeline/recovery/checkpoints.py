"""
Checkpoint Manager

Saves pipeline state after each stage completion.
Enables resume from last successful stage on failure.

Features:
- JSON-serializable checkpoints
- Atomic writes (prevent corruption)
- Automatic cleanup of old checkpoints
- Stage-level granularity
"""

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages for checkpointing."""

    INIT = "init"
    SCRAPE = "scrape"
    PREPROCESS = "preprocess"
    VERIFY = "verify"
    FORMAT = "format"
    EDITORIAL = "editorial"
    PUBLISH = "publish"
    COMPLETE = "complete"

    @property
    def order(self) -> int:
        """Get stage order for comparison."""
        order_map = {
            "init": 0,
            "scrape": 1,
            "preprocess": 2,
            "verify": 3,
            "format": 4,
            "editorial": 5,
            "publish": 6,
            "complete": 7,
        }
        return order_map.get(self.value, -1)

    def next_stage(self) -> "PipelineStage":
        """Get the next stage in sequence."""
        stages = list(PipelineStage)
        idx = stages.index(self)
        if idx < len(stages) - 1:
            return stages[idx + 1]
        return self


@dataclass
class Checkpoint:
    """
    A checkpoint representing pipeline state at a specific stage.
    """

    stage: PipelineStage
    timestamp: datetime
    run_id: str
    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # Tracking
    items_processed: int = 0
    items_total: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def progress(self) -> float:
        """Get completion progress (0-1)."""
        if self.items_total == 0:
            return 0.0
        return self.items_processed / self.items_total

    @property
    def is_complete(self) -> bool:
        """Check if stage is complete."""
        return self.items_processed >= self.items_total and self.items_total > 0

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "stage": self.stage.value,
            "timestamp": self.timestamp.isoformat(),
            "run_id": self.run_id,
            "data": self.data,
            "metadata": self.metadata,
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        """Deserialize from dictionary."""
        return cls(
            stage=PipelineStage(d["stage"]),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            run_id=d["run_id"],
            data=d.get("data", {}),
            metadata=d.get("metadata", {}),
            items_processed=d.get("items_processed", 0),
            items_total=d.get("items_total", 0),
            errors=d.get("errors", []),
        )


class CheckpointManager:
    """
    Manages pipeline checkpoints for recovery.

    Provides:
    - Save checkpoint after each stage
    - Resume from last successful stage
    - Idempotent reruns (skip completed items)
    - Cleanup of old checkpoints
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        max_checkpoints: int = 10,
        checkpoint_ttl_hours: int = 24,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
            max_checkpoints: Maximum checkpoints to keep
            checkpoint_ttl_hours: Hours before checkpoint expires
        """
        self.checkpoint_dir = Path(checkpoint_dir or Path.home() / ".news_pipeline" / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.checkpoint_ttl = timedelta(hours=checkpoint_ttl_hours)

        self._current_run_id: str | None = None
        self._checkpoints: dict[PipelineStage, Checkpoint] = {}

    def start_run(self, run_id: str | None = None) -> str:
        """
        Start a new pipeline run.

        Args:
            run_id: Optional run ID (generated if not provided)

        Returns:
            Run ID
        """
        if run_id is None:
            run_id = self._generate_run_id()

        self._current_run_id = run_id
        self._checkpoints.clear()

        # Check for existing run to resume
        existing = self._load_run(run_id)
        if existing:
            self._checkpoints = existing
            logger.info(f"[Checkpoint] Resuming run {run_id} from {self.last_completed_stage}")
        else:
            logger.info(f"[Checkpoint] Starting new run {run_id}")

        return run_id

    def save_checkpoint(
        self,
        stage: PipelineStage,
        data: dict,
        items_processed: int = 0,
        items_total: int = 0,
        metadata: dict | None = None,
    ) -> Checkpoint:
        """
        Save a checkpoint for a stage.

        Args:
            stage: Pipeline stage
            data: Stage output data
            items_processed: Number of items processed
            items_total: Total items to process
            metadata: Additional metadata

        Returns:
            Created checkpoint
        """
        if self._current_run_id is None:
            self._current_run_id = self._generate_run_id()

        checkpoint = Checkpoint(
            stage=stage,
            timestamp=datetime.utcnow(),
            run_id=self._current_run_id,
            data=data,
            metadata=metadata or {},
            items_processed=items_processed,
            items_total=items_total,
        )

        self._checkpoints[stage] = checkpoint
        self._persist_checkpoints()

        logger.info(f"[Checkpoint] Saved {stage.value}: {items_processed}/{items_total} items")

        return checkpoint

    def get_checkpoint(self, stage: PipelineStage) -> Checkpoint | None:
        """Get checkpoint for a stage."""
        return self._checkpoints.get(stage)

    @property
    def last_completed_stage(self) -> PipelineStage | None:
        """Get the last completed stage."""
        completed = [
            cp
            for cp in self._checkpoints.values()
            if cp.is_complete or cp.stage == PipelineStage.COMPLETE
        ]
        if not completed:
            return None
        return max(completed, key=lambda cp: cp.stage.order).stage

    @property
    def resume_stage(self) -> PipelineStage:
        """Get the stage to resume from."""
        last = self.last_completed_stage
        if last is None:
            return PipelineStage.INIT
        if last == PipelineStage.COMPLETE:
            return PipelineStage.COMPLETE
        return last.next_stage()

    def get_stage_data(self, stage: PipelineStage) -> dict:
        """Get data from a completed stage."""
        cp = self._checkpoints.get(stage)
        return cp.data if cp else {}

    def get_processed_ids(self, stage: PipelineStage) -> set[str]:
        """
        Get IDs of items already processed in a stage.

        Useful for idempotent reruns.
        """
        cp = self._checkpoints.get(stage)
        if not cp:
            return set()
        return set(cp.data.get("processed_ids", []))

    def mark_item_processed(self, stage: PipelineStage, item_id: str) -> None:
        """
        Mark an item as processed for idempotent reruns.

        Args:
            stage: Current stage
            item_id: Item identifier
        """
        cp = self._checkpoints.get(stage)
        if cp:
            if "processed_ids" not in cp.data:
                cp.data["processed_ids"] = []
            if item_id not in cp.data["processed_ids"]:
                cp.data["processed_ids"].append(item_id)
                cp.items_processed = len(cp.data["processed_ids"])

    def add_error(self, stage: PipelineStage, error: str) -> None:
        """Record an error for a stage."""
        cp = self._checkpoints.get(stage)
        if cp:
            cp.errors.append(error)
            self._persist_checkpoints()

    def complete_run(self) -> None:
        """Mark the current run as complete."""
        self.save_checkpoint(
            stage=PipelineStage.COMPLETE,
            data={"completed_at": datetime.utcnow().isoformat()},
            items_processed=1,
            items_total=1,
        )
        logger.info(f"[Checkpoint] Run {self._current_run_id} completed")

    def cleanup_old_checkpoints(self) -> int:
        """
        Remove expired checkpoints.

        Returns:
            Number of checkpoints removed
        """
        cutoff = datetime.utcnow() - self.checkpoint_ttl
        removed = 0

        for checkpoint_file in self.checkpoint_dir.glob("run_*.json"):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)
                latest_ts = max(
                    datetime.fromisoformat(cp["timestamp"])
                    for cp in data.get("checkpoints", {}).values()
                )
                if latest_ts < cutoff:
                    checkpoint_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupted file, remove it
                checkpoint_file.unlink()
                removed += 1

        if removed:
            logger.info(f"[Checkpoint] Cleaned up {removed} old checkpoints")

        return removed

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.sha256(f"{timestamp}_{id(self)}".encode()).hexdigest()[:8]
        return f"{timestamp}_{hash_suffix}"

    def _get_run_file(self, run_id: str) -> Path:
        """Get checkpoint file path for a run."""
        return self.checkpoint_dir / f"run_{run_id}.json"

    def _persist_checkpoints(self) -> None:
        """Save checkpoints to disk atomically."""
        if not self._current_run_id:
            return

        run_file = self._get_run_file(self._current_run_id)
        temp_file = run_file.with_suffix(".tmp")

        data = {
            "run_id": self._current_run_id,
            "checkpoints": {stage.value: cp.to_dict() for stage, cp in self._checkpoints.items()},
            "updated_at": datetime.utcnow().isoformat(),
        }

        try:
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            shutil.move(str(temp_file), str(run_file))
        except Exception as e:
            logger.error(f"[Checkpoint] Failed to save: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _load_run(self, run_id: str) -> dict[PipelineStage, Checkpoint] | None:
        """Load checkpoints for a run."""
        run_file = self._get_run_file(run_id)

        if not run_file.exists():
            return None

        try:
            with open(run_file) as f:
                data = json.load(f)

            checkpoints = {}
            for stage_name, cp_data in data.get("checkpoints", {}).items():
                stage = PipelineStage(stage_name)
                checkpoints[stage] = Checkpoint.from_dict(cp_data)

            return checkpoints

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"[Checkpoint] Failed to load {run_id}: {e}")
            return None

    def list_runs(self, include_completed: bool = False) -> list[dict]:
        """
        List available runs.

        Args:
            include_completed: Include completed runs

        Returns:
            List of run info dicts
        """
        runs = []

        for checkpoint_file in sorted(
            self.checkpoint_dir.glob("run_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                with open(checkpoint_file) as f:
                    data = json.load(f)

                run_info = {
                    "run_id": data["run_id"],
                    "updated_at": data.get("updated_at"),
                    "stages_completed": list(data.get("checkpoints", {}).keys()),
                    "is_complete": PipelineStage.COMPLETE.value in data.get("checkpoints", {}),
                }

                if include_completed or not run_info["is_complete"]:
                    runs.append(run_info)

            except (json.JSONDecodeError, KeyError):
                continue

        return runs[: self.max_checkpoints]


# Module-level instance
_checkpoint_manager: CheckpointManager | None = None


def get_checkpoint_manager() -> CheckpointManager:
    """Get the default checkpoint manager instance."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager
