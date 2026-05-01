"""Regression tests for scripts/pull_theta_iv_surface_history.py.

Two generations of fixes are locked here.

PR #55 — historical endpoint, not snapshot. The puller previously called
``/v3/option/snapshot/greeks/first_order`` (no date parameter), so every
"historical" partition contained the same current-state IV with different
``as_of`` labels. Tests 1–3 below ensure ``_surface_for_date`` calls a
historical Greeks endpoint and that ``as_of`` actually flows to the API.

PR #58 — shared connector, expirations cache, loud strict mode. The
Profile-D run produced partitions that were silently incomplete (3 of 6
chosen expirations on average) because each worker thread instantiated
its own ``ThetaConnector`` (each with its own ``_MAX_CONCURRENT=4``
semaphore), so 4 workers × 4 = 16 in-flight requests overwhelmed the
STANDARD-tier 4-concurrent ceiling. Theta returned 472 NO_DATA under
contention, the puller dropped failing expirations, and degraded
surfaces were written as if they were complete. Tests 4–7 below ensure:

  - per-ticker expirations cache is honoured (no per-date re-fetch)
  - strict mode rejects partial surfaces with ``rejected_partial=True``
  - ``--allow-partial`` (``strict=False``) writes the surface but tags
    its status as ``partial`` so callers can audit
  - status dict records the specific failed expirations
"""

from __future__ import annotations

import threading
from datetime import date, timedelta

import pandas as pd
import pytest

from scripts.pull_theta_iv_surface_history import (
    _get_cached_expirations,
    _surface_for_date,
)


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------
class _FakeConn:
    """Stand-in for ``ThetaConnector`` whose ``_fetch`` is data-driven."""

    def __init__(
        self,
        iv_by_date: dict[str, float] | None = None,
        *,
        skip_expirations: tuple[str, ...] = (),
    ) -> None:
        self.iv_by_date = iv_by_date or {}
        self.skip_expirations = skip_expirations
        self.calls: list[tuple[str, dict]] = []

    def _fetch(self, path: str, params: dict) -> pd.DataFrame:
        self.calls.append((path, dict(params)))
        if path == "/v3/option/list/expirations":
            # Spaced so each TARGET_DTE bucket (7, 14, 30, 60, 90, 180)
            # picks a unique expiration when as_of ∈ {2026-04-27, 2026-04-28}
            # — without this, buckets 7 and 14 both snap to the same expiry
            # and the test's "skip 3 of 6" scenarios become "skip 3 of 5".
            return pd.DataFrame(
                {
                    "symbol": ["AAPL"] * 6,
                    "expiration": [
                        "20260504",  # ~ DTE 7
                        "20260511",  # ~ DTE 14
                        "20260527",  # ~ DTE 30
                        "20260626",  # ~ DTE 60
                        "20260726",  # ~ DTE 90
                        "20261024",  # ~ DTE 180
                    ],
                }
            )
        if path.startswith("/v3/option/history/greeks/"):
            if params.get("expiration") in self.skip_expirations:
                # Simulate Theta returning empty / 472 for this expiration.
                return pd.DataFrame()
            start_date = params.get("start_date")
            iv = self.iv_by_date.get(start_date)
            if iv is None:
                return pd.DataFrame()
            ts = f"2026-{start_date[4:6]}-{start_date[6:8]}T15:30:00.000"
            return pd.DataFrame(
                {
                    "symbol": ["AAPL"] * 4,
                    "expiration": [params["expiration"]] * 4,
                    "strike": [275.0, 275.0, 280.0, 280.0],
                    "right": ["CALL", "PUT", "CALL", "PUT"],
                    "timestamp": [ts] * 4,
                    "implied_vol": [iv] * 4,
                    "midpoint": [10.0, 10.5, 8.0, 12.0],
                    "iv_error": [0.0001] * 4,
                }
            )
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# PR #55 lock-ins
# ---------------------------------------------------------------------------
def test_surface_for_date_returns_different_iv_per_as_of() -> None:
    """Two different as_of dates must yield two different IV values.

    If the puller regresses to a snapshot endpoint (no date parameter),
    both as_of values would receive the same response and this test
    would fail. That is the core regression we are guarding.
    """
    conn = _FakeConn(iv_by_date={"20260427": 0.20, "20260428": 0.22})

    df_27, status_27 = _surface_for_date(conn, "AAPL", date(2026, 4, 27))
    df_28, status_28 = _surface_for_date(conn, "AAPL", date(2026, 4, 28))

    assert not df_27.empty, "expected non-empty surface for 2026-04-27"
    assert not df_28.empty, "expected non-empty surface for 2026-04-28"
    assert not status_27["partial"]
    assert not status_28["partial"]

    iv_27 = sorted(df_27["iv"].dropna().unique().tolist())
    iv_28 = sorted(df_28["iv"].dropna().unique().tolist())

    assert iv_27 == [pytest.approx(0.20)], f"want IV=0.20 on 2026-04-27, got {iv_27}"
    assert iv_28 == [pytest.approx(0.22)], f"want IV=0.22 on 2026-04-28, got {iv_28}"
    assert iv_27 != iv_28, (
        "IV values must differ across as_of dates — equal values mean "
        "the puller regressed to a snapshot endpoint"
    )

    # date column must reflect the as_of, not "today"
    assert df_27["date"].dt.date.eq(date(2026, 4, 27)).all()
    assert df_28["date"].dt.date.eq(date(2026, 4, 28)).all()


def test_surface_for_date_uses_historical_endpoint() -> None:
    """No call may target a snapshot endpoint; date params must be passed."""
    conn = _FakeConn(iv_by_date={"20260427": 0.20})
    _surface_for_date(conn, "AAPL", date(2026, 4, 27))

    paths = [c[0] for c in conn.calls]
    assert paths, "expected at least one _fetch call"
    assert not any("/snapshot/greeks/" in p for p in paths), (
        f"_surface_for_date must not call snapshot/greeks/* — saw {paths}"
    )
    assert any("/history/greeks/" in p for p in paths), (
        f"_surface_for_date must call history/greeks/* — saw {paths}"
    )

    history_calls = [(p, params) for p, params in conn.calls if "/history/greeks/" in p]
    assert history_calls, "no history/greeks call recorded"
    for path, params in history_calls:
        assert params.get("start_date") == "20260427", (
            f"history call missing/wrong start_date: {path} → {params}"
        )
        assert params.get("end_date") == "20260427", (
            f"history call missing/wrong end_date: {path} → {params}"
        )


def test_surface_for_date_collapses_hourly_to_one_row_per_strike_right() -> None:
    """Hourly bars must reduce to one row per (strike, right, expiration)."""
    as_of = date(2026, 4, 27)
    exp_strs = [(as_of + timedelta(days=d)).strftime("%Y%m%d") for d in (7, 14, 30, 60, 90, 180)]

    class _MultiBarConn:
        def _fetch(self, path: str, params: dict) -> pd.DataFrame:
            if path == "/v3/option/list/expirations":
                return pd.DataFrame({"expiration": exp_strs})
            if path.startswith("/v3/option/history/greeks/"):
                rows = []
                for hour, iv in (("10:30", 0.10), ("12:30", 0.20), ("15:30", 0.30)):
                    for strike in (275.0, 280.0):
                        for r in ("CALL", "PUT"):
                            rows.append(
                                {
                                    "symbol": "AAPL",
                                    "expiration": params["expiration"],
                                    "strike": strike,
                                    "right": r,
                                    "timestamp": f"2026-04-27T{hour}:00.000",
                                    "implied_vol": iv,
                                    "midpoint": 10.0,
                                }
                            )
                return pd.DataFrame(rows)
            return pd.DataFrame()

    df, status = _surface_for_date(_MultiBarConn(), "AAPL", as_of)
    assert not df.empty
    assert status["unique_expirations"] == 6
    assert status["succeeded"] == 6
    assert status["partial"] is False

    # 6 distinct expirations × 4 collapsed (strike, right) rows = 24
    assert len(df) == 24, (
        f"expected 24 rows (6 expirations × 4 (strike,right)), got {len(df)}"
    )

    grouped = df.groupby(["strike", "right", "expiration"]).size()
    assert (grouped == 1).all(), (
        f"expected one row per (strike, right, expiration) after hourly "
        f"collapse, got group sizes:\n{grouped}"
    )
    # Must keep the LAST bar (IV=0.30), not an earlier one.
    assert df["iv"].dropna().eq(0.30).all(), (
        f"expected IV=0.30 (last bar of session), got {df['iv'].unique().tolist()}"
    )


# ---------------------------------------------------------------------------
# PR #58 lock-ins — shared cache + strict mode
# ---------------------------------------------------------------------------
def test_expirations_cache_avoids_redundant_fetches() -> None:
    """``option/list/expirations`` is static per ticker; the cache must
    hit on the second + Nth call and avoid re-fetching."""
    conn = _FakeConn(iv_by_date={"20260427": 0.20, "20260428": 0.22, "20260429": 0.25})
    cache: dict[str, pd.Series] = {}
    lock = threading.Lock()

    for d in (date(2026, 4, 27), date(2026, 4, 28), date(2026, 4, 29)):
        df, _ = _surface_for_date(
            conn, "AAPL", d,
            expirations_cache=cache, cache_lock=lock,
        )
        assert not df.empty

    listing_calls = [c for c in conn.calls if c[0] == "/v3/option/list/expirations"]
    assert len(listing_calls) == 1, (
        f"expirations endpoint should be hit exactly once across 3 dates, "
        f"got {len(listing_calls)} listing calls (cache miss)"
    )
    assert "AAPL" in cache, "cache should hold the AAPL listing after first call"


def test_get_cached_expirations_handles_concurrent_first_touch() -> None:
    """Multiple threads calling on a cold cache simultaneously must serialise
    on the lock and produce a single fetch (not one per thread)."""
    conn = _FakeConn(iv_by_date={})  # IV calls aren't reached in this test
    cache: dict[str, pd.Series] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()  # release all 8 threads at the same instant
        results.append(_get_cached_expirations(conn, "AAPL", cache, lock))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is not None for r in results), "all threads should get the listing"
    listing_calls = [c for c in conn.calls if c[0] == "/v3/option/list/expirations"]
    assert len(listing_calls) == 1, (
        f"under double-checked locking, only one thread should fetch; "
        f"got {len(listing_calls)} fetches across 8 threads"
    )


def test_strict_mode_rejects_partial_surface() -> None:
    """When some chosen expirations fail to return data, strict (default)
    must return an empty DataFrame with ``rejected_partial=True``.

    This is the load-bearing PR #58 fix: silently writing 3-of-6 surfaces
    is what poisoned Profile D.
    """
    # Skip 3 of the 6 expirations
    conn = _FakeConn(
        iv_by_date={"20260427": 0.20},
        skip_expirations=("20260527", "20260626", "20261024"),
    )

    df, status = _surface_for_date(conn, "AAPL", date(2026, 4, 27), strict=True)

    assert df.empty, "strict mode must drop a partial surface"
    assert status["rejected_partial"] is True
    assert status["partial"] is True
    assert status["succeeded"] == 3
    assert status["unique_expirations"] == 6
    failed_iso = sorted(d.isoformat() for d in status["failed_expirations"])
    assert failed_iso == ["2026-05-27", "2026-06-26", "2026-10-24"], (
        f"failed_expirations should record the specific failures, got {failed_iso}"
    )


def test_allow_partial_writes_partial_surface() -> None:
    """``strict=False`` (the ``--allow-partial`` CLI flag) must write the
    surface even when some expirations failed, and tag the status."""
    conn = _FakeConn(
        iv_by_date={"20260427": 0.20},
        skip_expirations=("20260527", "20260626", "20261024"),
    )

    df, status = _surface_for_date(conn, "AAPL", date(2026, 4, 27), strict=False)

    assert not df.empty, "allow-partial must keep the partial surface"
    assert status["rejected_partial"] is False
    assert status["partial"] is True, "status should still flag partial"
    assert status["succeeded"] == 3
    assert status["unique_expirations"] == 6
    # 3 expirations × 4 (strike, right) rows = 12
    assert len(df) == 12, f"expected 12 rows from 3-of-6 partial, got {len(df)}"


def test_status_zero_when_all_expirations_fail() -> None:
    """When every expiration returns empty, status reports succeeded=0
    and ``rejected_partial`` is False (the surface is empty, not partial)."""
    conn = _FakeConn(
        iv_by_date={"20260427": 0.20},
        skip_expirations=(
            "20260504", "20260511", "20260527",
            "20260626", "20260726", "20261024",
        ),
    )

    df, status = _surface_for_date(conn, "AAPL", date(2026, 4, 27), strict=True)

    assert df.empty
    assert status["succeeded"] == 0
    assert status["partial"] is False  # 0 < succeeded < unique is the partial check
    assert status["rejected_partial"] is False
    assert len(status["failed_expirations"]) == 6
