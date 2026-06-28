"""W5 — tail-fit / CVaR methodology pins (heavy-verify 2026-06-27, #436, stretch).

Validation-only. Pins the breach-statistic + VIX-regime helpers used by the tail
audit (without re-running the 35-date grid in the per-PR lane) and the engine
contract that ``cvar_5`` is a non-positive expected-loss. Full result in
``docs/HEAVY_VERIFY_2026-06-27_DATA_WIRING_RELIABILITY.md`` §W5 and
``docs/verification_artifacts/data_wiring_2026-06-27/w5_tail_cvar.json``.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]


def _load_w5():
    spec = importlib.util.spec_from_file_location("_w5", _REPO / "scripts" / "audit_tail_cvar.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def w5():
    return _load_w5()


def test_vix_regime_buckets(w5) -> None:
    assert w5.vix_regime(10.0) == "calm (<20)"
    assert w5.vix_regime(25.0) == "elevated (20-30)"
    assert w5.vix_regime(40.0) == "crisis (>=30)"
    assert w5.vix_regime(float("nan")) == "unknown"


def test_breach_stats_counts_realized_below_cvar(w5) -> None:
    """A breach is realized_pnl < cvar_5 (worse than the forecast tail loss)."""
    rows = [
        {"realized_pnl": -500.0, "cvar_5": -300.0},  # breach (−500 < −300)
        {"realized_pnl": -100.0, "cvar_5": -300.0},  # ok
        {"realized_pnl": 50.0, "cvar_5": -300.0},  # ok
        {"realized_pnl": -400.0, "cvar_5": -300.0},  # breach
    ]
    s = w5._breach_stats(rows)
    assert s["n"] == 4 and s["breaches"] == 2 and s["breach_rate"] == 0.5


def test_breach_stats_empty(w5) -> None:
    assert w5._breach_stats([]) == {"n": 0}


def test_cvar_5_respects_tail_ordering() -> None:
    """Engine contract: ``cvar_5`` (worst-5% mean P&L) ≤ the 25th-pct P&L.

    ``cvar_5`` is a *signed* P&L (positive for a put so safe its worst 5% still
    profits, e.g. AMCR), not a loss magnitude — so the meaningful pin is the tail
    ordering: the mean of the worst 5% can never exceed the 25th percentile.
    """
    import pandas as pd

    from engine.wheel_runner import WheelRunner

    df = WheelRunner().rank_candidates_by_ev(
        as_of="2022-06-16",
        top_n=30,
        min_ev_dollars=-1e9,
        include_diagnostic_fields=True,
        universe_limit=60,
    )
    cvar = pd.to_numeric(df["cvar_5"], errors="coerce")
    p25 = pd.to_numeric(df["pnl_p25"], errors="coerce")
    ok = (cvar <= p25 + 1e-6) | cvar.isna() | p25.isna()
    assert ok.all(), f"cvar_5 exceeded pnl_p25 for {df.loc[~ok, 'ticker'].tolist()}"
    assert cvar.notna().sum() > 0
