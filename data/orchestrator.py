"""
Pipeline Orchestrator - DAG-based execution with error handling and retry logic.

Production-grade orchestration with:
- Dependency-aware task execution (DAG)
- Parallel execution with worker pools
- Retry logic with exponential backoff
- Checkpoint/resume capability
- Observability (metrics, logging, tracing)
- Health monitoring and alerting

Usage:
    from data.orchestrator import PipelineOrchestrator

    # Define and run pipeline
    orchestrator = PipelineOrchestrator()
    orchestrator.run_full_pipeline(["AAPL", "MSFT", "GOOGL"])

    # Or run specific stage
    orchestrator.run_stage("features", ["AAPL"])

    # Check status
    orchestrator.status()
"""

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import json

import pandas as pd

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Status of a pipeline task."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class StageType(str, Enum):
    """Pipeline stages."""
    LOAD = "load"           # Load raw data
    VALIDATE = "validate"   # Validate data quality
    FEATURES = "features"   # Compute features
    STORE = "store"         # Persist to feature store
    EXPORT = "export"       # Export for downstream


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    stage: StageType
    ticker: str
    status: TaskStatus
    started_at: str
    completed_at: str
    duration_ms: int
    retries: int = 0
    error: Optional[str] = None
    output: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "stage": self.stage.value,
            "ticker": self.ticker,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
            "error": self.error,
        }


@dataclass
class PipelineRun:
    """A complete pipeline run."""
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    tickers: List[str] = field(default_factory=list)
    stages: List[StageType] = field(default_factory=list)
    tasks: List[TaskResult] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.SUCCESS)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)

    @property
    def duration_ms(self) -> int:
        if self.completed_at:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.completed_at)
            return int((end - start).total_seconds() * 1000)
        return 0

    def summary(self) -> str:
        lines = [
            f"Pipeline Run: {self.run_id}",
            f"Status: {self.status.value}",
            f"Duration: {self.duration_ms}ms",
            f"Tickers: {len(self.tickers)}",
            f"Tasks: {self.success_count} succeeded, {self.failed_count} failed",
            "=" * 50,
        ]

        # Group by stage
        by_stage = {}
        for task in self.tasks:
            stage = task.stage.value
            if stage not in by_stage:
                by_stage[stage] = {"success": 0, "failed": 0, "skipped": 0}
            if task.status == TaskStatus.SUCCESS:
                by_stage[stage]["success"] += 1
            elif task.status == TaskStatus.FAILED:
                by_stage[stage]["failed"] += 1
            elif task.status == TaskStatus.SKIPPED:
                by_stage[stage]["skipped"] += 1

        for stage, counts in by_stage.items():
            lines.append(f"  {stage}: {counts['success']}✓ {counts['failed']}✗ {counts['skipped']}⊘")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "tickers": self.tickers,
            "stages": [s.value for s in self.stages],
            "tasks": [t.to_dict() for t in self.tasks],
            "metrics": self.metrics,
            "summary": {
                "success_count": self.success_count,
                "failed_count": self.failed_count,
                "duration_ms": self.duration_ms,
            }
        }


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0
    retryable_exceptions: Tuple[type, ...] = (Exception,)


def with_retry(config: RetryConfig):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay_ms = config.initial_delay_ms

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay_ms}ms...")
                        time.sleep(delay_ms / 1000)
                        delay_ms = min(
                            delay_ms * config.exponential_base,
                            config.max_delay_ms
                        )
                    else:
                        logger.error(f"All {config.max_retries + 1} attempts failed")

            raise last_exception

        return wrapper
    return decorator


class PipelineOrchestrator:
    """
    DAG-based pipeline orchestrator.

    Manages the full data pipeline:
        LOAD → VALIDATE → FEATURES → STORE

    Features:
    - Dependency-aware execution
    - Parallel processing with worker pools
    - Retry logic with exponential backoff
    - Checkpoint/resume for long-running pipelines
    - Comprehensive observability
    """

    # Default DAG: dependencies for each stage
    DEFAULT_DAG = {
        StageType.LOAD: [],
        StageType.VALIDATE: [StageType.LOAD],
        StageType.FEATURES: [StageType.VALIDATE],
        StageType.STORE: [StageType.FEATURES],
    }

    def __init__(
        self,
        max_workers: int = 4,
        retry_config: Optional[RetryConfig] = None,
        checkpoint_dir: Optional[Path] = None,
        enable_metrics: bool = True,
    ):
        self.max_workers = max_workers
        self.retry_config = retry_config or RetryConfig()
        self.checkpoint_dir = checkpoint_dir or Path("data/.checkpoints")
        self.enable_metrics = enable_metrics

        # Lazy imports to avoid circular dependencies
        self._data_pipeline = None
        self._feature_pipeline = None
        self._quality_framework = None
        self._feature_store = None

        # Run history
        self._runs: List[PipelineRun] = []
        self._current_run: Optional[PipelineRun] = None

        # Metrics
        self._metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_duration_ms": 0,
        }

        # Initialize checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PipelineOrchestrator initialized with {max_workers} workers")

    @property
    def data_pipeline(self):
        if self._data_pipeline is None:
            from data.pipeline import DataPipeline
            self._data_pipeline = DataPipeline()
        return self._data_pipeline

    @property
    def feature_pipeline(self):
        if self._feature_pipeline is None:
            from data.feature_pipeline import FeaturePipeline
            self._feature_pipeline = FeaturePipeline(
                data_pipeline=self.data_pipeline,
                feature_store=self.feature_store,
            )
        return self._feature_pipeline

    @property
    def quality_framework(self):
        if self._quality_framework is None:
            from data.quality import DataQualityFramework
            self._quality_framework = DataQualityFramework()
        return self._quality_framework

    @property
    def feature_store(self):
        if self._feature_store is None:
            from data.feature_store import FeatureStore
            self._feature_store = FeatureStore()
        return self._feature_store

    def run_full_pipeline(
        self,
        tickers: List[str],
        stages: Optional[List[StageType]] = None,
        parallel: bool = True,
        resume_from: Optional[str] = None,
    ) -> PipelineRun:
        """
        Run the full pipeline for given tickers.

        Args:
            tickers: List of ticker symbols
            stages: Stages to run (default: all)
            parallel: Whether to run tickers in parallel
            resume_from: Run ID to resume from (for checkpoint recovery)

        Returns:
            PipelineRun with results
        """
        stages = stages or list(self.DEFAULT_DAG.keys())

        # Create or resume run
        if resume_from:
            run = self._load_checkpoint(resume_from)
            if run is None:
                raise ValueError(f"Checkpoint not found: {resume_from}")
            # Filter to only pending tickers
            completed_tickers = {t.ticker for t in run.tasks if t.status == TaskStatus.SUCCESS}
            tickers = [t for t in tickers if t not in completed_tickers]
        else:
            run = PipelineRun(
                run_id=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                started_at=datetime.now().isoformat(),
                tickers=tickers,
                stages=stages,
            )

        self._current_run = run
        run.status = TaskStatus.RUNNING

        logger.info(f"Starting pipeline run {run.run_id} for {len(tickers)} tickers")

        try:
            # Execute stages in order
            for stage in stages:
                self._run_stage(run, stage, tickers, parallel)

                # Save checkpoint after each stage
                self._save_checkpoint(run)

                # Check if we should abort
                failed_rate = run.failed_count / max(len(run.tasks), 1)
                if failed_rate > 0.5:
                    logger.error(f"Aborting: failure rate {failed_rate:.1%} exceeds threshold")
                    break

            # Finalize
            run.completed_at = datetime.now().isoformat()
            run.status = TaskStatus.SUCCESS if run.failed_count == 0 else TaskStatus.FAILED

            # Update metrics
            self._update_metrics(run)

        except Exception as e:
            logger.error(f"Pipeline run failed: {e}")
            run.status = TaskStatus.FAILED
            run.completed_at = datetime.now().isoformat()
            traceback.print_exc()

        finally:
            self._runs.append(run)
            self._current_run = None
            self._save_checkpoint(run)

        logger.info(f"Pipeline run completed:\n{run.summary()}")
        return run

    def _run_stage(
        self,
        run: PipelineRun,
        stage: StageType,
        tickers: List[str],
        parallel: bool,
    ) -> None:
        """Run a single stage for all tickers."""
        logger.info(f"Running stage: {stage.value} for {len(tickers)} tickers")

        # Get the executor for this stage
        executor = self._get_stage_executor(stage)

        if parallel and len(tickers) > 1:
            self._run_parallel(run, stage, tickers, executor)
        else:
            for ticker in tickers:
                result = self._execute_task(stage, ticker, executor)
                run.tasks.append(result)

    def _run_parallel(
        self,
        run: PipelineRun,
        stage: StageType,
        tickers: List[str],
        executor: Callable,
    ) -> None:
        """Run tasks in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures: Dict[Future, str] = {}

            for ticker in tickers:
                future = pool.submit(self._execute_task, stage, ticker, executor)
                futures[future] = ticker

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    run.tasks.append(result)
                except Exception as e:
                    logger.error(f"Task failed for {ticker}: {e}")
                    run.tasks.append(TaskResult(
                        task_id=f"{stage.value}_{ticker}",
                        stage=stage,
                        ticker=ticker,
                        status=TaskStatus.FAILED,
                        started_at=datetime.now().isoformat(),
                        completed_at=datetime.now().isoformat(),
                        duration_ms=0,
                        error=str(e),
                    ))

    def _execute_task(
        self,
        stage: StageType,
        ticker: str,
        executor: Callable,
    ) -> TaskResult:
        """Execute a single task with retry logic."""
        task_id = f"{stage.value}_{ticker}"
        started_at = datetime.now()
        retries = 0

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                output = executor(ticker)

                return TaskResult(
                    task_id=task_id,
                    stage=stage,
                    ticker=ticker,
                    status=TaskStatus.SUCCESS,
                    started_at=started_at.isoformat(),
                    completed_at=datetime.now().isoformat(),
                    duration_ms=int((datetime.now() - started_at).total_seconds() * 1000),
                    retries=retries,
                    output=output,
                )

            except Exception as e:
                retries += 1
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.initial_delay_ms * (
                        self.retry_config.exponential_base ** attempt
                    )
                    delay = min(delay, self.retry_config.max_delay_ms)
                    logger.warning(f"Task {task_id} failed (attempt {attempt + 1}): {e}. Retrying in {delay}ms...")
                    time.sleep(delay / 1000)
                else:
                    logger.error(f"Task {task_id} failed after {retries} retries: {e}")
                    return TaskResult(
                        task_id=task_id,
                        stage=stage,
                        ticker=ticker,
                        status=TaskStatus.FAILED,
                        started_at=started_at.isoformat(),
                        completed_at=datetime.now().isoformat(),
                        duration_ms=int((datetime.now() - started_at).total_seconds() * 1000),
                        retries=retries,
                        error=str(e),
                    )

    def _get_stage_executor(self, stage: StageType) -> Callable:
        """Get the executor function for a stage."""
        executors = {
            StageType.LOAD: self._execute_load,
            StageType.VALIDATE: self._execute_validate,
            StageType.FEATURES: self._execute_features,
            StageType.STORE: self._execute_store,
        }
        return executors[stage]

    def _execute_load(self, ticker: str) -> dict:
        """Execute LOAD stage for a ticker."""
        # Data is loaded lazily by DataPipeline, just trigger the load
        ohlcv = self.data_pipeline.get_ohlcv(ticker)
        options = self.data_pipeline.get_options(ticker)

        return {
            "ohlcv_rows": len(ohlcv) if ohlcv is not None else 0,
            "options_rows": len(options) if options is not None else 0,
        }

    def _execute_validate(self, ticker: str) -> dict:
        """Execute VALIDATE stage for a ticker."""
        results = {}

        # Validate OHLCV
        ohlcv = self.data_pipeline.get_ohlcv(ticker)
        if ohlcv is not None:
            result = self.quality_framework.validate(ohlcv, "ohlcv", ticker)
            results["ohlcv"] = {"valid": result.valid, "issues": len(result.issues)}

        # Validate options
        options = self.data_pipeline.get_options(ticker)
        if options is not None:
            result = self.quality_framework.validate(options, "options_flow", ticker)
            results["options"] = {"valid": result.valid, "issues": len(result.issues)}

        return results

    def _execute_features(self, ticker: str) -> dict:
        """Execute FEATURES stage for a ticker."""
        result = self.feature_pipeline.compute_all(ticker)

        return {
            "success": result.success,
            "categories": len(result.results),
            "total_time_ms": result.total_time_ms,
        }

    def _execute_store(self, ticker: str) -> dict:
        """Execute STORE stage for a ticker."""
        # Features are already stored by feature_pipeline
        # This stage can be used for additional exports
        metadata = self.feature_store.get_metadata("vol_edge", ticker)

        return {
            "stored": metadata is not None,
            "version": metadata.version if metadata else 0,
        }

    def _save_checkpoint(self, run: PipelineRun) -> None:
        """Save checkpoint for resume capability."""
        checkpoint_path = self.checkpoint_dir / f"{run.run_id}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(run.to_dict(), f, indent=2)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

    def _load_checkpoint(self, run_id: str) -> Optional[PipelineRun]:
        """Load checkpoint for resume."""
        checkpoint_path = self.checkpoint_dir / f"{run_id}.json"
        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path) as f:
            data = json.load(f)

        run = PipelineRun(
            run_id=data["run_id"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=TaskStatus(data["status"]),
            tickers=data["tickers"],
            stages=[StageType(s) for s in data["stages"]],
        )

        for task_data in data["tasks"]:
            run.tasks.append(TaskResult(
                task_id=task_data["task_id"],
                stage=StageType(task_data["stage"]),
                ticker=task_data["ticker"],
                status=TaskStatus(task_data["status"]),
                started_at=task_data["started_at"],
                completed_at=task_data["completed_at"],
                duration_ms=task_data["duration_ms"],
                retries=task_data.get("retries", 0),
                error=task_data.get("error"),
            ))

        return run

    def _update_metrics(self, run: PipelineRun) -> None:
        """Update global metrics after a run."""
        self._metrics["total_runs"] += 1
        if run.status == TaskStatus.SUCCESS:
            self._metrics["successful_runs"] += 1
        else:
            self._metrics["failed_runs"] += 1

        self._metrics["total_tasks"] += len(run.tasks)
        self._metrics["successful_tasks"] += run.success_count
        self._metrics["failed_tasks"] += run.failed_count
        self._metrics["total_duration_ms"] += run.duration_ms

    def run_stage(
        self,
        stage: Union[str, StageType],
        tickers: List[str],
        parallel: bool = True,
    ) -> PipelineRun:
        """Run a single stage for given tickers."""
        if isinstance(stage, str):
            stage = StageType(stage)

        return self.run_full_pipeline(tickers, stages=[stage], parallel=parallel)

    def get_run_history(
        self,
        limit: int = 10,
        status: Optional[TaskStatus] = None,
    ) -> List[PipelineRun]:
        """Get recent pipeline runs."""
        runs = self._runs
        if status:
            runs = [r for r in runs if r.status == status]
        return runs[-limit:]

    def status(self) -> dict:
        """Get orchestrator status."""
        return {
            "running": self._current_run is not None,
            "current_run": self._current_run.run_id if self._current_run else None,
            "total_runs": len(self._runs),
            "metrics": self._metrics,
            "checkpoints": len(list(self.checkpoint_dir.glob("*.json"))),
            "workers": self.max_workers,
        }

    def health_check(self) -> dict:
        """Run health checks on the orchestrator."""
        issues = []

        # Check data pipeline
        try:
            _ = self.data_pipeline.status()
        except Exception as e:
            issues.append(f"Data pipeline unhealthy: {e}")

        # Check feature store
        try:
            store_health = self.feature_store.health_check()
            if not store_health["healthy"]:
                issues.extend(store_health["issues"])
        except Exception as e:
            issues.append(f"Feature store unhealthy: {e}")

        # Check quality framework
        try:
            quality_health = self.quality_framework.health_check()
            if not quality_health["healthy"]:
                issues.extend(quality_health["issues"])
        except Exception as e:
            issues.append(f"Quality framework unhealthy: {e}")

        # Check recent run failures
        recent_runs = self.get_run_history(limit=5)
        failure_rate = sum(1 for r in recent_runs if r.status == TaskStatus.FAILED) / max(len(recent_runs), 1)
        if failure_rate > 0.5:
            issues.append(f"High recent failure rate: {failure_rate:.1%}")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "status": self.status(),
        }

    def cleanup_checkpoints(self, max_age_days: int = 7) -> int:
        """Clean up old checkpoints."""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed = 0

        for checkpoint in self.checkpoint_dir.glob("*.json"):
            if datetime.fromtimestamp(checkpoint.stat().st_mtime) < cutoff:
                checkpoint.unlink()
                removed += 1

        logger.info(f"Cleaned up {removed} old checkpoints")
        return removed


# Convenience function for quick pipeline runs
def run_pipeline(
    tickers: List[str],
    stages: Optional[List[str]] = None,
    parallel: bool = True,
) -> PipelineRun:
    """Run the data pipeline for given tickers."""
    orchestrator = PipelineOrchestrator()
    stage_types = [StageType(s) for s in stages] if stages else None
    return orchestrator.run_full_pipeline(tickers, stage_types, parallel)
