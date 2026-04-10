"""Structured observability for institutional audit trails.

Provides trace contexts, decision journaling, and structured JSON logging
for full auditability of trading decisions and operations.
"""

import json
import logging
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class TraceContext:
    """Distributed-style trace context for correlating operations."""

    operation: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)


@dataclass
class DecisionEntry:
    """A single recorded trading decision."""

    timestamp: datetime
    trace_id: str
    decision_type: str  # entry / exit / roll / skip
    symbol: str
    reason: str
    signal_values: dict
    risk_metrics: dict
    outcome: dict | None = None


class DecisionJournal:
    """Records trading decisions with full context for audit review."""

    def __init__(self) -> None:
        self._entries: list[DecisionEntry] = []

    def record_decision(
        self,
        trace_id: str,
        decision_type: str,
        symbol: str,
        reason: str,
        signal_values: dict,
        risk_metrics: dict,
        outcome: dict | None = None,
    ) -> DecisionEntry:
        entry = DecisionEntry(
            timestamp=datetime.now(UTC),
            trace_id=trace_id,
            decision_type=decision_type,
            symbol=symbol,
            reason=reason,
            signal_values=signal_values,
            risk_metrics=risk_metrics,
            outcome=outcome,
        )
        self._entries.append(entry)
        return entry

    def get_decisions(
        self,
        symbol: str | None = None,
        decision_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[DecisionEntry]:
        results = self._entries
        if symbol is not None:
            results = [e for e in results if e.symbol == symbol]
        if decision_type is not None:
            results = [e for e in results if e.decision_type == decision_type]
        if start_date is not None:
            results = [e for e in results if e.timestamp >= start_date]
        if end_date is not None:
            results = [e for e in results if e.timestamp <= end_date]
        return results

    def export_json(self, path: str | Path) -> None:
        """Write all entries to a JSON file."""

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Not serializable: {type(obj)}")

        data = [asdict(e) for e in self._entries]
        Path(path).write_text(json.dumps(data, default=_serialize, indent=2))

    def export_dataframe(self) -> list[dict]:
        """Return entries as a list of dicts (lightweight DataFrame stand-in).

        Avoids a pandas dependency; callers can pass the result straight to
        ``pandas.DataFrame(journal.export_dataframe())`` if pandas is available.
        """

        def _to_row(entry: DecisionEntry) -> dict:
            row = asdict(entry)
            row["timestamp"] = row["timestamp"].isoformat()
            return row

        return [_to_row(e) for e in self._entries]


# Module-level event log for trace lifecycle events
_trace_events: list[dict] = []


def get_trace_events() -> list[dict]:
    """Return all trace lifecycle events (for testing/audit)."""
    return list(_trace_events)


def clear_trace_events() -> None:
    """Clear accumulated trace events."""
    _trace_events.clear()


@contextmanager
def trace_operation(
    operation: str,
    parent_id: str | None = None,
    **metadata: Any,
) -> Generator[TraceContext, None, None]:
    """Context manager that creates a TraceContext and emits lifecycle events.

    Automatically records:
    - ``operation_start`` on entry
    - ``operation_end`` on successful exit (with duration_ms)
    - ``operation_error`` on exception (with error message and duration_ms)
    """
    ctx = TraceContext(
        operation=operation,
        parent_id=parent_id,
        metadata=metadata,
    )
    _trace_events.append({
        "event": "operation_start",
        "trace_id": ctx.trace_id,
        "operation": operation,
        "timestamp": ctx.start_time.isoformat(),
        "parent_id": parent_id,
    })
    try:
        yield ctx
    except Exception as exc:
        end_time = datetime.now(UTC)
        duration_ms = (end_time - ctx.start_time).total_seconds() * 1000
        _trace_events.append({
            "event": "operation_error",
            "trace_id": ctx.trace_id,
            "operation": operation,
            "timestamp": end_time.isoformat(),
            "duration_ms": duration_ms,
            "error": str(exc),
            "error_type": type(exc).__name__,
        })
        raise
    else:
        end_time = datetime.now(UTC)
        duration_ms = (end_time - ctx.start_time).total_seconds() * 1000
        _trace_events.append({
            "event": "operation_end",
            "trace_id": ctx.trace_id,
            "operation": operation,
            "timestamp": end_time.isoformat(),
            "duration_ms": duration_ms,
        })


class AuditLogger:
    """Wraps standard logging with structured JSON output including trace IDs."""

    def __init__(self, name: str = "audit", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)
        self._logger.setLevel(level)

    def _emit(
        self,
        level: str,
        message: str,
        trace_id: str | None = None,
        **extra: Any,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
        }
        if trace_id is not None:
            record["trace_id"] = trace_id
        if extra:
            record["extra"] = extra
        self._logger.log(
            getattr(logging, level.upper(), logging.INFO),
            json.dumps(record),
        )

    def info(self, message: str, trace_id: str | None = None, **kw: Any) -> None:
        self._emit("INFO", message, trace_id, **kw)

    def warning(self, message: str, trace_id: str | None = None, **kw: Any) -> None:
        self._emit("WARNING", message, trace_id, **kw)

    def error(self, message: str, trace_id: str | None = None, **kw: Any) -> None:
        self._emit("ERROR", message, trace_id, **kw)

    def debug(self, message: str, trace_id: str | None = None, **kw: Any) -> None:
        self._emit("DEBUG", message, trace_id, **kw)
