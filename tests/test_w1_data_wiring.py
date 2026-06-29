"""W1 — data-wiring accuracy pins (heavy-verify 2026-06-27, #436, Mac terminal).

Validation-only. These tests pin what the audit *verified* (green) and pin the
*confirmed* OHLCV split-scale defect to its behaviour via ``xfail(strict)`` —
so the day the committed OHLCV is regenerated cleanly, the strict-xfail flips to
XPASS and CI flags the pin for removal (it can never silently false-green).

See ``docs/HEAVY_VERIFY_2026-06-27_DATA_WIRING_RELIABILITY.md`` §W1 and
``scripts/audit_data_wiring.py``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector

# Liquid, full-history sample — keeps the test fast (no 2M-cell sweep).
_SAMPLE = ["AAPL", "MSFT", "JPM", "XOM", "UNH", "KO", "PG"]


@pytest.fixture(scope="module")
def conn() -> MarketDataConnector:
    return MarketDataConnector()


# ---------------------------------------------------------------------------
# Verified properties (must stay green)
# ---------------------------------------------------------------------------
def test_served_iv_band_is_clean(conn: MarketDataConnector) -> None:
    """Every served IV cell is in the clean PERCENT band (3.0, 10000].

    Pins the ``_clean_vol_iv_inplace`` gate: raw sub-3 / >10000 sentinel cells
    must never reach a consumer, so the downstream ``if iv > 3.0: iv/100``
    conversion is always correct.
    """
    checked = 0
    for t in _SAMPLE:
        iv = conn.get_iv_history(t)
        if iv.empty:
            continue
        for col in ("hist_put_imp_vol", "hist_call_imp_vol"):
            if col not in iv.columns:
                continue
            v = pd.to_numeric(iv[col], errors="coerce").dropna()
            checked += len(v)
            assert v.empty or (v.gt(3.0) & v.le(10_000.0)).all(), (
                f"{t}.{col} has IV outside (3.0, 10000]: "
                f"{v[~(v.gt(3.0) & v.le(10_000.0))].head(3).tolist()}"
            )
    assert checked > 0, "no served IV cells checked — sample/data path broken"


def test_no_deep_iv_sentinel_leaks(conn: MarketDataConnector) -> None:
    """The 134217.7 deep-IV NULL-by-magnitude sentinel never reaches a consumer."""
    for t in _SAMPLE:
        iv = conn.get_iv_history(t)
        if iv.empty:
            continue
        for col in ("hist_put_imp_vol", "hist_call_imp_vol"):
            if col not in iv.columns:
                continue
            v = pd.to_numeric(iv[col], errors="coerce").dropna()
            assert not v.between(134_000, 134_500).any(), f"{t}.{col} leaks the deep-IV sentinel"


def test_ohlcv_invariant_holds(conn: MarketDataConnector) -> None:
    """Post-rename OHLC invariant ``high>=max(o,c,l)`` & ``low<=min(o,c,h)``."""
    for t in _SAMPLE:
        df = conn.get_ohlcv(t).dropna(subset=["open", "high", "low", "close"])
        assert not df.empty
        assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all(), f"{t} high<max"
        assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all(), f"{t} low>min"


def test_ohlcv_dates_monotonic_and_positive(conn: MarketDataConnector) -> None:
    for t in _SAMPLE:
        df = conn.get_ohlcv(t)
        assert df.index.is_monotonic_increasing, f"{t} OHLCV dates not monotonic"
        assert not df.index.has_duplicates, f"{t} OHLCV has duplicate dates"
        assert (df["close"].dropna() > 0).all(), f"{t} has non-positive close"


def test_treasury_covers_feasible_window(conn: MarketDataConnector) -> None:
    """rate_3m must cover the full feasible OHLCV window (starts 2018).

    Pins the W4 finding: treasury coverage now spans 1994→2026, so the
    historical ``get_current_risk_free_rate`` spurious-5% path (which only fires
    *before* coverage begins) is unreachable for any feasible ``as_of``.
    """
    raw = pd.read_csv("data/bloomberg/treasury_yields.csv")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    cov = raw.dropna(subset=["rate_3m"])
    assert cov["date"].min() <= pd.Timestamp("2018-01-01"), "treasury starts after OHLCV"
    # decimal sanity across regimes (÷100, D20)
    for asof, lo, hi in [("2018-06-01", 0.005, 0.04), ("2024-01-02", 0.03, 0.07)]:
        r = conn.get_risk_free_rate(asof, "rate_3m")
        assert lo <= r <= hi, f"rate_3m at {asof} = {r} outside plausible [{lo},{hi}]"


# ---------------------------------------------------------------------------
# Verified property — the 2026-03-23 split-scale splice is repaired (#439)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ticker", ["BKNG", "CVNA"])
def test_ohlcv_has_no_split_scale_discontinuity(conn: MarketDataConnector, ticker: str) -> None:
    """No >2x single-day close move around the 2026-03-23 splice.

    Behaviour-pinned (not signature): computes the actual served close ratio and
    asserts continuity. D-W1-1 (#439) — the BKNG 25:1 / CVNA 5:1 split-seam
    misalignment — was repaired by back-adjusting the full pre-splice history
    onto the split-adjusted scale, so the seam ratio is now the genuine weekend
    move (~1.03 / ~1.06), in band. Was strict-xfail until the fix landed.
    """
    s = conn.get_ohlcv(ticker, "2026-03-10", "2026-04-10")["close"].dropna()
    ratios = (s / s.shift(1)).dropna()
    bad = ratios[(ratios < 0.5) | (ratios > 2.0)]
    assert bad.empty, f"{ticker} split-scale discontinuity: " + "; ".join(
        f"{d.date()} ratio={r:.3f}" for d, r in bad.items()
    )
