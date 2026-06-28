"""W4 — risk-free-rate + PIT correctness pins (heavy-verify 2026-06-27, #436).

Validation-only. Pins (a) that treasury coverage spans the feasible window so
the spurious-5% RFR fallback is unreachable and ``get_current_risk_free_rate``
returns the real PIT decimal rate, and (b) that the ranker's IV is point-in-time
(no lookahead; the as-of value moves with ``as_of`` rather than being a fixed
present-day snapshot). Full quantification in
``docs/HEAVY_VERIFY_2026-06-27_DATA_WIRING_RELIABILITY.md`` §W4.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector
from engine.data_integration import get_current_risk_free_rate
from engine.wheel_runner import _resolve_pit_atm_iv


@pytest.fixture(scope="module")
def conn() -> MarketDataConnector:
    return MarketDataConnector()


def test_rfr_returns_real_pit_rate_not_spurious_5pct(conn) -> None:
    """At ZIRP-era as_of the served rate is ~0, never the 0.05 fallback."""
    # 2021-05-01: 3m T-bill was ~ZIRP; must be well under 1%, not 5%.
    r = get_current_risk_free_rate("2021-05-01", data_dir="data/bloomberg", fallback=0.05)
    assert r == r and r < 0.01, f"expected ZIRP rate, got {r}"
    # connector agrees
    assert conn.get_risk_free_rate("2021-05-01", "rate_3m") < 0.01
    # 2024 should be a real ~5% rate (not the fallback coincidentally)
    assert 0.03 < get_current_risk_free_rate("2024-01-02", data_dir="data/bloomberg") < 0.07


def test_fallback_only_fires_before_coverage(conn) -> None:
    """The 0.05 fallback is reachable only before treasury coverage (pre-1994)."""
    raw = pd.read_csv("data/bloomberg/treasury_yields.csv")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    cov_start = raw.dropna(subset=["rate_3m"])["date"].min()
    assert cov_start <= pd.Timestamp("2018-01-01"), "coverage must precede OHLCV start"
    # before coverage → fallback fires; after → real value
    assert (
        get_current_risk_free_rate("1990-01-01", data_dir="data/bloomberg", fallback=0.05) == 0.05
    )


def test_pit_iv_has_no_lookahead(conn) -> None:
    for t in ["AAPL", "MSFT", "JPM"]:
        for asof in ["2021-06-15", "2024-01-16"]:
            h = conn.get_iv_history(t, end_date=asof)
            if h.empty:
                continue
            assert h.index.max() <= pd.Timestamp(asof), f"{t} IV lookahead past {asof}"


def test_pit_iv_moves_with_asof(conn) -> None:
    """Resolved ATM IV differs across as_of (not a fixed present-day snapshot)."""
    moved = 0
    for t in ["AAPL", "MSFT", "NVDA"]:
        a = _resolve_pit_atm_iv(conn, t, "2021-06-15")
        b = _resolve_pit_atm_iv(conn, t, "2024-01-16")
        if a is not None and b is not None and abs(a - b) > 1e-6:
            moved += 1
    assert moved > 0, "resolved IV did not move with as_of — possible snapshot regression"
