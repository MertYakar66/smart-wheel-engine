"""Regression tests for scripts/pull_theta_iv_surface_history.py.

The puller previously called ``/v3/option/snapshot/greeks/first_order`` —
a snapshot endpoint with no date parameter. Every "historical" partition
ended up containing the same current-state IV with different ``as_of``
labels and different ``dte`` values, producing actively misleading data.

These tests lock in the fix:
  1. Different ``as_of`` dates must yield different IV values when the
     underlying API returns different IV for those dates.
  2. The puller must never call a snapshot endpoint; it must call a
     historical Greeks endpoint and pass ``as_of`` as both ``start_date``
     and ``end_date`` so the date parameter actually flows to the API.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from scripts.pull_theta_iv_surface_history import _surface_for_date


class _FakeConn:
    """Stand-in for ``ThetaConnector`` whose ``_fetch`` is data-driven.

    The ``iv_by_date`` map keys are ``YYYYMMDD`` strings and values are
    IV decimals. ``calls`` records every (path, params) tuple so tests
    can assert on the call pattern.
    """

    def __init__(self, iv_by_date: dict[str, float] | None = None) -> None:
        self.iv_by_date = iv_by_date or {}
        self.calls: list[tuple[str, dict]] = []

    def _fetch(self, path: str, params: dict) -> pd.DataFrame:
        self.calls.append((path, dict(params)))
        if path == "/v3/option/list/expirations":
            return pd.DataFrame(
                {
                    "symbol": ["AAPL"] * 6,
                    "expiration": [
                        "20260508",
                        "20260515",
                        "20260605",
                        "20260619",
                        "20260717",
                        "20260918",
                    ],
                }
            )
        if path.startswith("/v3/option/history/greeks/"):
            start_date = params.get("start_date")
            iv = self.iv_by_date.get(start_date)
            if iv is None:
                return pd.DataFrame()
            ts = (
                f"2026-{start_date[4:6]}-{start_date[6:8]}T15:30:00.000"
            )
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


def test_surface_for_date_returns_different_iv_per_as_of() -> None:
    """Two different as_of dates must yield two different IV values.

    If the puller regresses to a snapshot endpoint (no date parameter),
    both as_of values would receive the same response and this test
    would fail. That is the core regression we are guarding.
    """
    conn = _FakeConn(iv_by_date={"20260427": 0.20, "20260428": 0.22})

    df_27 = _surface_for_date(conn, "AAPL", date(2026, 4, 27))
    df_28 = _surface_for_date(conn, "AAPL", date(2026, 4, 28))

    assert not df_27.empty, "expected non-empty surface for 2026-04-27"
    assert not df_28.empty, "expected non-empty surface for 2026-04-28"

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
    """Hourly bars must reduce to one row per (strike, right, expiration).

    Three hourly bars × 2 strikes × 2 rights = 12 input rows per call;
    the collapse must keep only the latest bar of the session, leaving
    4 rows per call (one (strike, right) pair × one expiration).
    """
    # Six distinct expirations spanning the TARGET_DTES buckets so each
    # bucket picks a unique one. Without this, all six TARGET_DTES buckets
    # would resolve to the same expiration and the per-(strike,right,exp)
    # grouping would falsely show six rows per group from concatenation.
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

    df = _surface_for_date(_MultiBarConn(), "AAPL", as_of)
    assert not df.empty, "expected non-empty surface"

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
