"""Tests for news_pipeline/recovery/checkpoints.py."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from news_pipeline.recovery.checkpoints import (
    Checkpoint,
    CheckpointManager,
    PipelineStage,
    get_checkpoint_manager,
)


class TestPipelineStage:
    def test_order_progression(self):
        assert PipelineStage.INIT.order < PipelineStage.SCRAPE.order
        assert PipelineStage.SCRAPE.order < PipelineStage.PREPROCESS.order
        assert PipelineStage.PUBLISH.order < PipelineStage.COMPLETE.order

    def test_next_stage(self):
        assert PipelineStage.INIT.next_stage() == PipelineStage.SCRAPE
        assert PipelineStage.SCRAPE.next_stage() == PipelineStage.PREPROCESS
        # COMPLETE returns itself (no next stage)
        assert PipelineStage.COMPLETE.next_stage() == PipelineStage.COMPLETE


class TestCheckpoint:
    def test_progress_with_zero_total(self):
        cp = Checkpoint(
            stage=PipelineStage.SCRAPE, timestamp=datetime.utcnow(),
            run_id="r1", items_processed=0, items_total=0,
        )
        assert cp.progress == 0.0

    def test_progress_partial(self):
        cp = Checkpoint(
            stage=PipelineStage.SCRAPE, timestamp=datetime.utcnow(),
            run_id="r1", items_processed=3, items_total=10,
        )
        assert cp.progress == 0.3

    def test_is_complete_when_processed_geq_total(self):
        cp = Checkpoint(
            stage=PipelineStage.SCRAPE, timestamp=datetime.utcnow(),
            run_id="r1", items_processed=10, items_total=10,
        )
        assert cp.is_complete is True

    def test_is_complete_false_when_total_zero(self):
        cp = Checkpoint(
            stage=PipelineStage.SCRAPE, timestamp=datetime.utcnow(),
            run_id="r1", items_processed=0, items_total=0,
        )
        assert cp.is_complete is False

    def test_to_from_dict_roundtrip(self):
        ts = datetime.utcnow()
        cp = Checkpoint(
            stage=PipelineStage.PREPROCESS, timestamp=ts,
            run_id="r1", data={"foo": "bar"}, metadata={"m": 1},
            items_processed=5, items_total=10, errors=["e1"],
        )
        d = cp.to_dict()
        cp2 = Checkpoint.from_dict(d)
        assert cp2.stage == cp.stage
        assert cp2.run_id == cp.run_id
        assert cp2.data == cp.data
        assert cp2.errors == cp.errors

    def test_from_dict_with_minimal_fields(self):
        d = {
            "stage": "init",
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": "r1",
        }
        cp = Checkpoint.from_dict(d)
        assert cp.data == {}
        assert cp.metadata == {}
        assert cp.errors == []


class TestCheckpointManager:
    @pytest.fixture
    def mgr(self, tmp_path: Path) -> CheckpointManager:
        return CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    def test_creates_dir_on_init(self, tmp_path: Path):
        d = tmp_path / "newdir"
        CheckpointManager(checkpoint_dir=d)
        assert d.exists()

    def test_start_run_generates_id_when_none(self, mgr: CheckpointManager):
        run_id = mgr.start_run()
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert mgr._current_run_id == run_id

    def test_start_run_with_explicit_id(self, mgr: CheckpointManager):
        run_id = mgr.start_run(run_id="my_run")
        assert run_id == "my_run"

    def test_start_run_resumes_existing(self, mgr: CheckpointManager):
        # Save a checkpoint, then start the same run again
        mgr.start_run("resume_test")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {"x": 1}, items_processed=5, items_total=5)

        # New manager, same dir, same run_id → should resume
        mgr2 = CheckpointManager(checkpoint_dir=mgr.checkpoint_dir)
        mgr2.start_run("resume_test")
        cp = mgr2.get_checkpoint(PipelineStage.SCRAPE)
        assert cp is not None
        assert cp.data == {"x": 1}

    def test_save_checkpoint_persists(self, mgr: CheckpointManager):
        mgr.start_run("test_save")
        cp = mgr.save_checkpoint(
            PipelineStage.SCRAPE, {"foo": "bar"},
            items_processed=3, items_total=10, metadata={"src": "rss"},
        )
        assert cp.stage == PipelineStage.SCRAPE
        assert cp.run_id == "test_save"
        # File on disk
        run_file = mgr._get_run_file("test_save")
        assert run_file.exists()
        data = json.loads(run_file.read_text())
        assert "scrape" in data["checkpoints"]

    def test_save_without_explicit_run_generates(self, mgr: CheckpointManager):
        # No prior start_run; save still works (auto-generates run_id)
        cp = mgr.save_checkpoint(PipelineStage.INIT, {})
        assert cp is not None
        assert mgr._current_run_id is not None

    def test_get_checkpoint_returns_none_for_missing(self, mgr: CheckpointManager):
        mgr.start_run("test_get")
        assert mgr.get_checkpoint(PipelineStage.PUBLISH) is None

    def test_last_completed_stage_none_when_empty(self, mgr: CheckpointManager):
        mgr.start_run("test_last")
        assert mgr.last_completed_stage is None

    def test_last_completed_stage_returns_highest(self, mgr: CheckpointManager):
        mgr.start_run("test_last2")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {}, items_processed=10, items_total=10)
        mgr.save_checkpoint(PipelineStage.PREPROCESS, {}, items_processed=10, items_total=10)
        assert mgr.last_completed_stage == PipelineStage.PREPROCESS

    def test_resume_stage_from_init_when_empty(self, mgr: CheckpointManager):
        mgr.start_run("test_resume")
        assert mgr.resume_stage == PipelineStage.INIT

    def test_resume_stage_after_complete(self, mgr: CheckpointManager):
        mgr.start_run("test_resume2")
        mgr.save_checkpoint(PipelineStage.COMPLETE, {}, items_processed=1, items_total=1)
        assert mgr.resume_stage == PipelineStage.COMPLETE

    def test_resume_stage_after_partial(self, mgr: CheckpointManager):
        mgr.start_run("test_resume3")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {}, items_processed=10, items_total=10)
        # Resume = next stage after last completed
        assert mgr.resume_stage == PipelineStage.PREPROCESS

    def test_get_stage_data(self, mgr: CheckpointManager):
        mgr.start_run("test_data")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {"items": [1, 2, 3]})
        assert mgr.get_stage_data(PipelineStage.SCRAPE) == {"items": [1, 2, 3]}
        # Missing stage returns empty dict
        assert mgr.get_stage_data(PipelineStage.PUBLISH) == {}

    def test_get_processed_ids_empty(self, mgr: CheckpointManager):
        mgr.start_run("test_pids")
        assert mgr.get_processed_ids(PipelineStage.SCRAPE) == set()

    def test_mark_item_processed(self, mgr: CheckpointManager):
        mgr.start_run("test_mark")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {})
        mgr.mark_item_processed(PipelineStage.SCRAPE, "item_1")
        mgr.mark_item_processed(PipelineStage.SCRAPE, "item_2")
        # Idempotent: re-mark doesn't double-count
        mgr.mark_item_processed(PipelineStage.SCRAPE, "item_1")
        ids = mgr.get_processed_ids(PipelineStage.SCRAPE)
        assert ids == {"item_1", "item_2"}

    def test_add_error(self, mgr: CheckpointManager):
        mgr.start_run("test_err")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {})
        mgr.add_error(PipelineStage.SCRAPE, "boom")
        cp = mgr.get_checkpoint(PipelineStage.SCRAPE)
        assert "boom" in cp.errors

    def test_complete_run(self, mgr: CheckpointManager):
        mgr.start_run("test_complete")
        mgr.complete_run()
        assert mgr.get_checkpoint(PipelineStage.COMPLETE) is not None

    def test_cleanup_old_checkpoints(self, mgr: CheckpointManager):
        # Create an old checkpoint by manually writing a file with old timestamp
        old_run = mgr._get_run_file("old_run")
        old_ts = (datetime.utcnow() - timedelta(days=30)).isoformat()
        old_run.write_text(json.dumps({
            "run_id": "old_run",
            "checkpoints": {
                "scrape": {
                    "stage": "scrape", "timestamp": old_ts,
                    "run_id": "old_run", "data": {},
                    "items_processed": 0, "items_total": 0, "errors": [],
                }
            },
            "updated_at": old_ts,
        }))
        removed = mgr.cleanup_old_checkpoints()
        assert removed >= 1
        assert not old_run.exists()

    def test_cleanup_removes_corrupt_files(self, mgr: CheckpointManager):
        bad_run = mgr.checkpoint_dir / "run_corrupt.json"
        bad_run.write_text("{not valid json")
        removed = mgr.cleanup_old_checkpoints()
        assert removed >= 1

    def test_list_runs_excludes_completed_by_default(self, mgr: CheckpointManager):
        mgr.start_run("active_run")
        mgr.save_checkpoint(PipelineStage.SCRAPE, {})
        mgr.start_run("done_run")
        mgr.complete_run()

        runs = mgr.list_runs()
        run_ids = {r["run_id"] for r in runs}
        assert "active_run" in run_ids
        assert "done_run" not in run_ids

    def test_list_runs_includes_completed_when_requested(self, mgr: CheckpointManager):
        mgr.start_run("done_run2")
        mgr.complete_run()
        runs = mgr.list_runs(include_completed=True)
        assert any(r["run_id"] == "done_run2" for r in runs)

    def test_load_run_returns_none_for_missing(self, mgr: CheckpointManager):
        assert mgr._load_run("nonexistent") is None

    def test_load_run_handles_corrupt_file(self, mgr: CheckpointManager):
        bad = mgr._get_run_file("bad")
        bad.write_text("{not json")
        assert mgr._load_run("bad") is None


class TestGetCheckpointManager:
    def test_returns_singleton(self):
        m1 = get_checkpoint_manager()
        m2 = get_checkpoint_manager()
        assert m1 is m2
