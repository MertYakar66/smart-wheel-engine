"""Regression tests for scripts/pull_theta_indices_history.py.

The puller previously sent multi-year ``--years`` windows in a single
request, tripping Theta's 365-day cap on ``/v3/index/history/*``. The
HTTP 400 / 403 / 472 diagnostic bodies were dropped by the shared
``ThetaConnector._fetch`` (engine/theta_connector.py:155) and the puller
fell through three endpoints with all symbols marked FAIL in 11.4s.

These tests lock in:
  1. Multi-year requests are split into ≤365-day chunks before any
     HTTP call.
  2. Per-endpoint chunk responses are concatenated in date order.
  3. Tier-gate / window-cap / coverage-miss text is surfaced to
     stdout exactly as Theta returned it, not silently dropped.
  4. Older chunks that tier-gate do not block newer chunks from
     being returned.
  5. ``--incremental`` runs that only have today left to fetch report
     "up-to-date" + rc=0 instead of FAIL — Theta doesn't publish
     today's EOD until ~17:15 ET, so pre-close cron jobs would
     otherwise always exit non-zero (PR #57).
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

from scripts.pull_theta_indices_history import _cap_end_date, _pull


# ---------------------------------------------------------------------------
# Mock plumbing — emulate ``requests.Session.get`` against the running
# Terminal. We mock at the session layer (not connector._fetch) because the
# real puller needs to see HTTP statuses for tier-gate detection — same
# reason it bypasses _fetch in production.
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeSession:
    def __init__(self, handler: Callable[[str, dict], _FakeResp]) -> None:
        self._handler = handler
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None, timeout: int = 30) -> _FakeResp:
        params = params or {}
        # Keep enough state for assertions on chunking / param flow
        self.calls.append({"url": url, **params})
        return self._handler(url, params)


class _FakeConn:
    def __init__(self, handler: Callable[[str, dict], _FakeResp]) -> None:
        self._session = _FakeSession(handler)

    @property
    def calls(self) -> list[dict]:
        return self._session.calls


def _csv_response(rows: list[dict]) -> _FakeResp:
    """Build a 200 OK response with a CSV body."""
    df = pd.DataFrame(rows)
    return _FakeResp(200, df.to_csv(index=False))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_pull_chunks_multi_year_request_under_365_days() -> None:
    """A 5-year request must produce only chunks ≤ 365 days each."""

    def handler(url: str, params: dict) -> _FakeResp:
        return _FakeResp(200, "")  # empty bodies; we only care about chunk sizes

    conn = _FakeConn(handler)
    _pull(conn, "VIX", date(2021, 5, 1), date(2026, 4, 30))

    # Three endpoints × ⌈1825/350⌉ = 6 chunks each = 18 calls minimum
    assert len(conn.calls) >= 18, (
        f"expected >= 18 HTTP calls (6 chunks × 3 endpoints), got {len(conn.calls)}"
    )

    for c in conn.calls:
        sd = datetime.strptime(c["start_date"], "%Y%m%d").date()
        ed = datetime.strptime(c["end_date"], "%Y%m%d").date()
        days = (ed - sd).days
        assert days <= 365, (
            f"chunk {c['start_date']}..{c['end_date']} is {days} days — exceeds 365"
        )
        assert c["symbol"] == "VIX"
        assert c["format"] == "csv"


def test_pull_concatenates_chunks_in_date_order() -> None:
    """Successive chunk responses are stitched together for a single endpoint."""
    rows_by_call = [
        [
            {"created": "2024-01-15", "open": 13.0, "high": 13.5, "low": 12.8, "close": 13.2},
            {"created": "2024-06-15", "open": 14.0, "high": 14.5, "low": 13.8, "close": 14.2},
        ],
        [
            {"created": "2025-01-15", "open": 15.0, "high": 15.5, "low": 14.8, "close": 15.2},
            {"created": "2025-06-15", "open": 16.0, "high": 16.5, "low": 15.8, "close": 16.2},
        ],
        [
            {"created": "2026-01-15", "open": 17.0, "high": 17.5, "low": 16.8, "close": 17.2},
            {"created": "2026-04-15", "open": 18.0, "high": 18.5, "low": 17.8, "close": 18.2},
        ],
    ]

    state = {"idx": 0}

    def handler(url: str, params: dict) -> _FakeResp:
        if "/eod" not in url:
            return _FakeResp(200, "")  # empty for ohlc + price — try eod first
        i = min(state["idx"], len(rows_by_call) - 1)
        state["idx"] += 1
        return _csv_response(rows_by_call[i])

    df = _pull(_FakeConn(handler), "VIX", date(2024, 1, 1), date(2026, 4, 30))

    assert not df.empty
    assert len(df) == 6, f"expected 6 rows total (3 chunks × 2 days), got {len(df)}"
    assert df["close"].tolist() == [13.2, 14.2, 15.2, 16.2, 17.2, 18.2]
    assert df["date"].is_monotonic_increasing
    assert df["symbol"].eq("VIX").all()
    assert df["source"].eq("theta").all()


def test_pull_surfaces_tier_gate_to_stdout(capsys) -> None:
    """403 tier-gate response text must be surfaced to stdout, not dropped."""
    gate_text = (
        "Requesting index history requiring a STANDARD subscription, "
        "but you only have a FREE subscription"
    )

    def handler(url: str, params: dict) -> _FakeResp:
        if "/eod" not in url:
            return _FakeResp(200, "")
        return _FakeResp(403, gate_text)

    _pull(_FakeConn(handler), "VIX", date(2022, 1, 1), date(2022, 12, 31))
    captured = capsys.readouterr()

    assert "STANDARD subscription" in captured.out, (
        f"tier-gate text must surface to stdout; got: {captured.out!r}"
    )
    assert "VIX" in captured.out, "symbol prefix must be in the surfaced line"
    assert "GATE" in captured.out, "GATE marker must be in the surfaced line"


def test_pull_collects_newer_chunks_when_older_chunks_tier_gate(capsys) -> None:
    """Older chunks tier-gating must not block newer chunks from being kept."""
    gate_text = (
        "Requesting index history requiring a STANDARD subscription, "
        "but you only have a FREE subscription"
    )

    state = {"idx": 0}

    def handler(url: str, params: dict) -> _FakeResp:
        if "/eod" not in url:
            return _FakeResp(200, "")
        state["idx"] += 1
        # First two chunks (oldest) tier-gate; the rest return data.
        if state["idx"] <= 2:
            return _FakeResp(403, gate_text)
        sd = params["start_date"]
        rows = [{
            "created": f"{sd[:4]}-{sd[4:6]}-{sd[6:8]}",
            "open": 13.0, "high": 13.5, "low": 12.5, "close": 13.2,
        }]
        return _csv_response(rows)

    df = _pull(_FakeConn(handler), "VIX", date(2021, 1, 1), date(2026, 1, 1))
    captured = capsys.readouterr()

    assert not df.empty, (
        "puller must keep newer-chunk data when older chunks tier-gate"
    )
    assert (df["close"] == 13.2).all()
    assert df["symbol"].eq("VIX").all()
    # Both older chunks must be surfaced, not just the first
    assert captured.out.count("GATE") >= 2, (
        f"expected ≥2 GATE lines (one per gated chunk), got: {captured.out!r}"
    )


# ---------------------------------------------------------------------------
# PR #57: --incremental + only-today-left should report up-to-date, not FAIL.
# ---------------------------------------------------------------------------
def test_cap_end_date_passes_through_past_dates() -> None:
    """End dates strictly before today are returned unchanged."""
    today = date(2026, 5, 1)
    assert _cap_end_date(date(2026, 4, 25), today) == date(2026, 4, 25)
    assert _cap_end_date(date(2026, 4, 30), today) == date(2026, 4, 30)


def test_cap_end_date_caps_today_to_yesterday() -> None:
    """End == today is capped at today - 1 (today's EOD not yet published)."""
    today = date(2026, 5, 1)
    assert _cap_end_date(today, today) == date(2026, 4, 30)


def test_cap_end_date_caps_future_to_yesterday() -> None:
    """End in the future is also capped at today - 1."""
    today = date(2026, 5, 1)
    assert _cap_end_date(date(2026, 6, 15), today) == date(2026, 4, 30)


def test_main_incremental_skips_when_only_today_remains(capsys, monkeypatch) -> None:
    """--incremental must report up-to-date + rc=0 when nothing new is left.

    Setup: ``_last_theta_date`` returns yesterday for every symbol, so the
    incremental computation produces ``s_start = today``. After the
    end-cap fix, ``end_d = today - 1 < s_start``, the existing
    ``s_start > end_d`` branch fires per symbol, and ``_pull`` is never
    called.

    Without the fix, the script would call _pull(symbol, today, today),
    Theta would return 472 NO_DATA on every request, every symbol would
    FAIL, and the script would exit rc=1 — pre-close cron jobs always
    failing despite nothing being wrong.
    """
    import scripts.pull_theta_indices_history as M

    fake_today = date(2026, 5, 1)

    class _FakeDate(date):
        @classmethod
        def today(cls):
            return fake_today

    # The puller resolves ``date.today()`` and ``date.fromisoformat`` from
    # the module-level ``date`` symbol; replace it wholesale.
    monkeypatch.setattr(M, "date", _FakeDate)

    monkeypatch.setattr(M, "_last_theta_date", lambda sym: date(2026, 4, 30))
    monkeypatch.setattr(M, "_theta_up", lambda *a, **k: True)

    class _StubConn:
        def __init__(self) -> None:
            self._session = None

    monkeypatch.setattr(M, "ThetaConnector", _StubConn)

    pull_calls: list[tuple] = []
    monkeypatch.setattr(
        M, "_pull",
        lambda *args, **kwargs: pull_calls.append(args) or pd.DataFrame(),
    )

    monkeypatch.setattr(
        sys, "argv",
        ["pull_theta_indices_history.py", "--symbols", "VIX", "VIX9D",
         "--incremental"],
    )

    rc = M.main()
    out = capsys.readouterr().out

    assert rc == 0, f"expected rc=0 when only today is left; got {rc}\nstdout:\n{out}"
    assert pull_calls == [], (
        f"_pull must not be called when only today remains; got {pull_calls}"
    )
    assert "up-to-date" in out, (
        f"expected per-symbol 'up-to-date' message; got:\n{out}"
    )
    assert "FAIL" not in out, (
        f"no symbol should be marked FAIL when only today remains; got:\n{out}"
    )
