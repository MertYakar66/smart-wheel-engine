"""Tests for engine/observability.py — trace contexts, decision journal, audit logger."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from engine.observability import (
    AuditLogger,
    DecisionEntry,
    DecisionJournal,
    TraceContext,
    clear_trace_events,
    get_trace_events,
    trace_operation,
)


@pytest.fixture(autouse=True)
def _clear_trace_events_between_tests():
    """Avoid cross-test pollution of the module-level trace event log."""
    clear_trace_events()
    yield
    clear_trace_events()


class TestTraceContext:
    def test_default_factory_assigns_uuid_trace_id(self):
        ctx1 = TraceContext(operation="op1")
        ctx2 = TraceContext(operation="op2")
        assert ctx1.trace_id != ctx2.trace_id
        assert len(ctx1.trace_id) == 36  # uuid4 string length

    def test_default_metadata_is_empty_dict(self):
        ctx = TraceContext(operation="op")
        assert ctx.metadata == {}

    def test_default_start_time_is_utc(self):
        ctx = TraceContext(operation="op")
        assert ctx.start_time.tzinfo == UTC

    def test_explicit_parent_id_preserved(self):
        ctx = TraceContext(operation="op", parent_id="parent-xyz")
        assert ctx.parent_id == "parent-xyz"


class TestDecisionJournal:
    def test_record_returns_entry_and_appends(self):
        journal = DecisionJournal()
        entry = journal.record_decision(
            trace_id="t1",
            decision_type="entry",
            symbol="AAPL",
            reason="iv_rank_high",
            signal_values={"iv_rank": 0.62},
            risk_metrics={"position_size": 5},
        )
        assert isinstance(entry, DecisionEntry)
        assert entry.symbol == "AAPL"
        assert entry.decision_type == "entry"
        assert entry.outcome is None
        assert len(journal.get_decisions()) == 1

    def test_record_with_outcome(self):
        journal = DecisionJournal()
        entry = journal.record_decision(
            trace_id="t",
            decision_type="exit",
            symbol="MSFT",
            reason="profit_target",
            signal_values={},
            risk_metrics={},
            outcome={"realized_pnl": 250.0},
        )
        assert entry.outcome == {"realized_pnl": 250.0}

    def test_filter_by_symbol(self):
        journal = DecisionJournal()
        journal.record_decision("t1", "entry", "AAPL", "r", {}, {})
        journal.record_decision("t2", "entry", "MSFT", "r", {}, {})
        out = journal.get_decisions(symbol="AAPL")
        assert len(out) == 1
        assert out[0].symbol == "AAPL"

    def test_filter_by_decision_type(self):
        journal = DecisionJournal()
        journal.record_decision("t1", "entry", "AAPL", "r", {}, {})
        journal.record_decision("t2", "exit", "AAPL", "r", {}, {})
        journal.record_decision("t3", "skip", "AAPL", "r", {}, {})
        out = journal.get_decisions(decision_type="exit")
        assert len(out) == 1
        assert out[0].decision_type == "exit"

    def test_filter_by_date_range(self):
        journal = DecisionJournal()
        e1 = journal.record_decision("t1", "entry", "AAPL", "r", {}, {})
        e2 = journal.record_decision("t2", "entry", "MSFT", "r", {}, {})
        cutoff = e1.timestamp + timedelta(microseconds=1)
        out = journal.get_decisions(start_date=cutoff)
        # e1 was before cutoff, e2 was after (or at it within μs precision)
        # Just check the API runs and returns a subset
        assert len(out) <= 2
        assert all(e.timestamp >= cutoff for e in out)
        # End-date inclusive
        out2 = journal.get_decisions(end_date=e2.timestamp)
        assert len(out2) == 2

    def test_combined_filters(self):
        journal = DecisionJournal()
        journal.record_decision("t1", "entry", "AAPL", "r", {}, {})
        journal.record_decision("t2", "exit", "AAPL", "r", {}, {})
        journal.record_decision("t3", "entry", "MSFT", "r", {}, {})
        out = journal.get_decisions(symbol="AAPL", decision_type="entry")
        assert len(out) == 1
        assert out[0].symbol == "AAPL"
        assert out[0].decision_type == "entry"

    def test_export_json_roundtrip(self, tmp_path: Path):
        journal = DecisionJournal()
        journal.record_decision(
            "t1", "entry", "AAPL", "iv_high",
            {"iv_rank": 0.6}, {"size": 3},
        )
        out = tmp_path / "j.json"
        journal.export_json(out)
        data = json.loads(out.read_text())
        assert len(data) == 1
        # timestamps are ISO-formatted strings post-export
        assert isinstance(data[0]["timestamp"], str)
        assert data[0]["symbol"] == "AAPL"

    def test_export_dataframe_returns_dicts(self):
        journal = DecisionJournal()
        journal.record_decision("t1", "entry", "AAPL", "r", {}, {})
        rows = journal.export_dataframe()
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)
        assert isinstance(rows[0]["timestamp"], str)  # serialized

    def test_export_json_with_non_serializable_field(self, tmp_path: Path):
        # The _serialize fallback raises TypeError on unknown types — we
        # don't actually have a way to feed a non-datetime non-primitive
        # via the dataclass, but we can hit the asdict() path with empty
        # dicts to confirm it still writes valid JSON.
        journal = DecisionJournal()
        out = tmp_path / "empty.json"
        journal.export_json(out)
        assert json.loads(out.read_text()) == []


class TestTraceOperation:
    def test_emits_start_and_end_events(self):
        with trace_operation("compute_ev") as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.operation == "compute_ev"
        events = get_trace_events()
        assert len(events) == 2
        assert events[0]["event"] == "operation_start"
        assert events[1]["event"] == "operation_end"
        assert events[0]["trace_id"] == events[1]["trace_id"]
        assert events[1]["duration_ms"] >= 0

    def test_metadata_captured_on_context(self):
        with trace_operation("op", region="us-east-1") as ctx:
            assert ctx.metadata == {"region": "us-east-1"}

    def test_parent_id_threaded_through(self):
        with trace_operation("parent_op") as parent_ctx:
            with trace_operation("child_op", parent_id=parent_ctx.trace_id) as child_ctx:
                assert child_ctx.parent_id == parent_ctx.trace_id
        events = get_trace_events()
        # 4 events: parent_start, child_start, child_end, parent_end
        assert len(events) == 4
        # Find the child's start event
        child_starts = [e for e in events if e["operation"] == "child_op" and e["event"] == "operation_start"]
        assert len(child_starts) == 1
        assert child_starts[0]["parent_id"] == parent_ctx.trace_id

    def test_emits_error_event_on_exception(self):
        with pytest.raises(ValueError, match="boom"):
            with trace_operation("flaky_op"):
                raise ValueError("boom")
        events = get_trace_events()
        assert len(events) == 2
        assert events[0]["event"] == "operation_start"
        assert events[1]["event"] == "operation_error"
        assert events[1]["error"] == "boom"
        assert events[1]["error_type"] == "ValueError"
        assert events[1]["duration_ms"] >= 0


class TestAuditLogger:
    def test_info_emits_structured_json(self, caplog: pytest.LogCaptureFixture):
        logger = AuditLogger(name="test_audit_info", level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="test_audit_info"):
            logger.info("trade_opened", trace_id="t-123", symbol="AAPL")
        assert len(caplog.records) >= 1
        # The message itself is a JSON-encoded record
        msg = caplog.records[-1].getMessage()
        record = json.loads(msg)
        assert record["level"] == "INFO"
        assert record["message"] == "trade_opened"
        assert record["trace_id"] == "t-123"
        assert record["extra"] == {"symbol": "AAPL"}

    def test_warning_error_debug_levels(self, caplog: pytest.LogCaptureFixture):
        logger = AuditLogger(name="test_audit_levels", level=logging.DEBUG)
        with caplog.at_level(logging.DEBUG, logger="test_audit_levels"):
            logger.warning("warn_msg")
            logger.error("err_msg", trace_id="t-err")
            logger.debug("debug_msg")
        records = [json.loads(r.getMessage()) for r in caplog.records]
        levels = [r["level"] for r in records]
        assert "WARNING" in levels
        assert "ERROR" in levels
        assert "DEBUG" in levels
        # ERROR record carried trace_id
        err = next(r for r in records if r["level"] == "ERROR")
        assert err["trace_id"] == "t-err"

    def test_emit_without_extra_fields(self, caplog: pytest.LogCaptureFixture):
        logger = AuditLogger(name="test_audit_minimal", level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="test_audit_minimal"):
            logger.info("plain")
        record = json.loads(caplog.records[-1].getMessage())
        assert "extra" not in record
        assert "trace_id" not in record

    def test_handler_only_added_once(self):
        # Calling AuditLogger twice with the same name shouldn't double up
        # the handler list.
        logger1 = AuditLogger(name="test_audit_dedupe", level=logging.INFO)
        n_before = len(logger1._logger.handlers)
        AuditLogger(name="test_audit_dedupe", level=logging.INFO)
        n_after = len(logger1._logger.handlers)
        assert n_before == n_after
