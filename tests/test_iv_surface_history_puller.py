"""Regression tests for scripts/pull_theta_iv_surface_history.py.

Three generations of fixes are locked here.

PR #55 — historical endpoint, not snapshot. The puller previously called
``/v3/option/snapshot/greeks/first_order`` (no date parameter), so every
"historical" partition contained the same current-state IV with different
``as_of`` labels. Tests 1–3 below ensure ``_surface_for_date`` calls a
historical Greeks endpoint and that ``as_of`` actually flows to the API.

PR #58 — shared connector + expirations cache + loud strict mode. The
Profile-D run produced partitions that were silently incomplete (3 of 6
chosen expirations on average) because each worker thread instantiated
its own ``ThetaConnector`` (each with its own ``_MAX_CONCURRENT=4``
semaphore), so 4 workers × 4 = 16 in-flight requests overwhelmed the
STANDARD-tier 4-concurrent ceiling. Theta returned 472 NO_DATA under
contention, the puller dropped failing expirations, and degraded
surfaces were written as if they were complete.

PR #59 — per-bucket fallback. Even with proper concurrency, Theta's
``option/list/expirations`` returns weeklies + monthlies + LEAPS but its
IV history is sparser — many weekly expirations on liquid names like
AAPL come back ``"No data found"`` (HTTP 472). The old "snap to nearest,
take it or leave it" approach hit those gaps and strict mode then
rejected the whole partition. The fix iterates up to ``fallback_k``
nearest candidates per TARGET_DTES bucket; first one with data wins.
Tests 8–11 ensure the fallback iterates correctly, succeeds when an
early candidate fails but a later one has data, fails the bucket only
when all candidates fail, and never reuses an expiration across buckets.
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
    """Stand-in for ``ThetaConnector`` whose ``_fetch`` is data-driven.

    ``iv_by_date`` maps ``YYYYMMDD`` → IV decimal.

    ``skip_expirations`` is a tuple of expiration ``YYYYMMDD`` strings
    that should return empty (simulating Theta's 472 NO_DATA gaps).
    """

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
            # picks a unique expiration when as_of ∈ {2026-04-27, 2026-04-28}.
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
                # Simulate Theta's 472 NO_DATA: a listed expiration with no
                # IV history available.
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
    assert status_27["succeeded_buckets"] == 6
    assert status_28["succeeded_buckets"] == 6

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
    assert status["succeeded_buckets"] == 6
    assert status["partial"] is False

    # 6 distinct expirations × 4 collapsed (strike, right) rows = 24
    assert len(df) == 24, f"expected 24 rows (6 expirations × 4 (strike,right)), got {len(df)}"

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
            conn,
            "AAPL",
            d,
            expirations_cache=cache,
            cache_lock=lock,
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
    conn = _FakeConn(iv_by_date={})
    cache: dict[str, pd.Series] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
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


def test_strict_mode_rejects_when_a_bucket_has_no_data() -> None:
    """When a bucket can't find ANY of its candidates with data, strict
    must return an empty DataFrame with ``rejected_partial=True``.

    With fallback_k=1 (no fallback) and 3 of 6 listed expirations skipped,
    we get exactly 3 of 6 buckets succeeding — the partial-strict
    rejection scenario.
    """
    conn = _FakeConn(
        iv_by_date={"20260427": 0.20},
        skip_expirations=("20260527", "20260626", "20261024"),
    )

    df, status = _surface_for_date(
        conn,
        "AAPL",
        date(2026, 4, 27),
        strict=True,
        fallback_k=1,
    )

    assert df.empty, "strict mode must drop a partial surface"
    assert status["rejected_partial"] is True
    assert status["partial"] is True
    assert status["succeeded_buckets"] == 3
    assert status["target_dtes"] == 6
    # 3 buckets failed; their DTEs are surfaced for diagnosis.
    assert len(status["failed_buckets"]) == 3


def test_allow_partial_writes_partial_surface() -> None:
    """``strict=False`` (the ``--allow-partial`` CLI flag) must write the
    surface even when some buckets failed, and tag the status."""
    conn = _FakeConn(
        iv_by_date={"20260427": 0.20},
        skip_expirations=("20260527", "20260626", "20261024"),
    )

    df, status = _surface_for_date(
        conn,
        "AAPL",
        date(2026, 4, 27),
        strict=False,
        fallback_k=1,
    )

    assert not df.empty, "allow-partial must keep the partial surface"
    assert status["rejected_partial"] is False
    assert status["partial"] is True, "status should still flag partial"
    assert status["succeeded_buckets"] == 3
    # 3 expirations × 4 (strike, right) rows = 12
    assert len(df) == 12, f"expected 12 rows from 3-of-6 partial, got {len(df)}"


# ---------------------------------------------------------------------------
# PR #59 lock-ins — per-bucket fallback to next-nearest with data
# ---------------------------------------------------------------------------
def test_fallback_iterates_to_next_nearest_when_first_candidate_empty() -> None:
    """When the closest expiration to a bucket target returns empty,
    the puller must try the next-nearest, until one returns data."""

    # Skip the closest expiration to each TARGET_DTE bucket. For as_of
    # 2026-04-27 the listing's 6 expirations are each the closest match
    # for one bucket; if we skip e.g. ``20260504`` (DTE 7's match), the
    # bucket should fall back to ``20260511`` (originally DTE 14's). With
    # only 6 listed expirations and 6 buckets, you can only skip if
    # there's a candidate the bucket can move to.
    #
    # Construct a richer listing: extra "decoy" expirations interleaved
    # with the matched ones, where the decoys are slightly closer to the
    # target but return empty.
    class _DecoyConn:
        def __init__(self):
            self.calls = []

        def _fetch(self, path, params):
            self.calls.append((path, dict(params)))
            if path == "/v3/option/list/expirations":
                # For DTE 7 (target=2026-05-04): decoy 20260505 (1 day off,
                # empty) closer than 20260504 (0 off, has data). With
                # fallback_k=2, the puller tries 20260505 first, gets
                # empty, then tries 20260504, gets data.
                return pd.DataFrame(
                    {
                        "expiration": [
                            "20260505",  # decoy (empty)
                            "20260504",  # has data (target match)
                            "20260512",  # decoy (empty)
                            "20260511",  # has data
                            "20260528",  # decoy
                            "20260527",  # has data
                            "20260627",  # decoy
                            "20260626",  # has data
                            "20260727",  # decoy
                            "20260726",  # has data
                            "20261025",  # decoy
                            "20261024",  # has data
                        ],
                    }
                )
            if path.startswith("/v3/option/history/greeks/"):
                # Only the second-of-each-pair has data
                if params["expiration"] in {
                    "20260504",
                    "20260511",
                    "20260527",
                    "20260626",
                    "20260726",
                    "20261024",
                }:
                    return pd.DataFrame(
                        {
                            "symbol": ["AAPL"],
                            "expiration": [params["expiration"]],
                            "strike": [275.0],
                            "right": ["CALL"],
                            "timestamp": ["2026-04-27T15:30:00"],
                            "implied_vol": [0.20],
                            "midpoint": [10.0],
                        }
                    )
                return pd.DataFrame()
            return pd.DataFrame()

    conn = _DecoyConn()
    df, status = _surface_for_date(
        conn,
        "AAPL",
        date(2026, 4, 27),
        strict=True,
        fallback_k=2,
    )

    assert not df.empty
    assert status["succeeded_buckets"] == 6
    assert status["partial"] is False
    assert status["rejected_partial"] is False
    # The chosen expirations should be the data-having ones
    chosen_iso = sorted(d.isoformat() for d in status["chosen_expirations"])
    assert chosen_iso == [
        "2026-05-04",
        "2026-05-11",
        "2026-05-27",
        "2026-06-26",
        "2026-07-26",
        "2026-10-24",
    ]


def test_fallback_fails_bucket_only_when_all_k_candidates_empty() -> None:
    """When all ``fallback_k`` nearest candidates for a bucket return
    empty, only that bucket fails — other buckets should still find
    their own data."""

    class _OneBadBucketConn:
        def _fetch(self, path, params):
            if path == "/v3/option/list/expirations":
                return pd.DataFrame(
                    {
                        "expiration": [
                            # 5 expirations have data; one DTE bucket (~30d
                            # away) has none nearby.
                            "20260504",
                            "20260511",
                            "20260626",
                            "20260726",
                            "20261024",
                        ],
                    }
                )
            if path.startswith("/v3/option/history/greeks/"):
                # Always return data — but 30-DTE bucket has no
                # near-30-DTE expiration in the listing, so it falls
                # back to the nearest available (which is 20260504 at 7
                # DTE, but that's already claimed by bucket 7).
                return pd.DataFrame(
                    {
                        "symbol": ["AAPL"],
                        "expiration": [params["expiration"]],
                        "strike": [275.0],
                        "right": ["CALL"],
                        "timestamp": ["2026-04-27T15:30:00"],
                        "implied_vol": [0.20],
                        "midpoint": [10.0],
                    }
                )
            return pd.DataFrame()

    conn = _OneBadBucketConn()
    df, status = _surface_for_date(
        conn,
        "AAPL",
        date(2026, 4, 27),
        strict=False,
        fallback_k=10,
    )
    # 5 listed expirations, 6 buckets — at least one bucket can't get
    # a unique expiration. Expect succeeded_buckets ≤ 5.
    assert status["succeeded_buckets"] == 5, (
        f"with 5 listed expirations and 6 buckets, expect 5 to succeed, "
        f"got {status['succeeded_buckets']}"
    )
    assert len(status["failed_buckets"]) == 1


def test_fallback_does_not_reuse_expirations_across_buckets() -> None:
    """Each TARGET_DTES bucket must end up with a distinct expiration —
    the ``used_exps`` set prevents bucket N from claiming an expiration
    bucket M already took."""
    conn = _FakeConn(iv_by_date={"20260427": 0.20})
    df, status = _surface_for_date(
        conn,
        "AAPL",
        date(2026, 4, 27),
        fallback_k=10,
    )

    chosen = status["chosen_expirations"]
    assert len(chosen) == len(set(chosen)), (
        f"chosen expirations must be unique across buckets, got {chosen}"
    )


def test_status_zero_when_all_buckets_fail() -> None:
    """When every bucket exhausts its fallback budget without finding
    data, status reports ``succeeded_buckets=0`` and ``rejected_partial``
    is False (the surface is empty, not partial)."""
    conn = _FakeConn(
        iv_by_date={"20260427": 0.20},
        skip_expirations=(
            "20260504",
            "20260511",
            "20260527",
            "20260626",
            "20260726",
            "20261024",
        ),
    )

    df, status = _surface_for_date(
        conn,
        "AAPL",
        date(2026, 4, 27),
        strict=True,
        fallback_k=10,
    )

    assert df.empty
    assert status["succeeded_buckets"] == 0
    assert status["partial"] is False  # 0 < succeeded < target_n is the partial check
    assert status["rejected_partial"] is False
    assert status["failed_buckets"] == list((7, 14, 30, 60, 90, 180))
