"""W6 — top-bin net-cost measurement pins (heavy-verify 2026-06-28, #446).

Validation/measurement-only. Pins the entry-VIX bands, the friction-overlay
realized-P&L (which must reuse the canonical
``backtests/regression/_common._forward_replay_realized_pnl`` the brief names),
the cluster-bootstrap CI, and the net-cost verdict logic — WITHOUT re-running the
72-date forward-replay grid in the per-PR lane. One cheap live-integration pin
confirms the ranker→replay wiring. Full result in
``docs/HEAVY_VERIFY_2026-06-28_TOPBIN_NETCOST.md`` and
``docs/verification_artifacts/topbin_netcost_2026-06-28/w6_topbin_netcost.json``.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]


def _load_w6():
    spec = importlib.util.spec_from_file_location(
        "_w6", _REPO / "scripts" / "audit_topbin_netcost.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def w6():
    return _load_w6()


# --------------------------------------------------------------------------- #
# Entry-VIX bands (#446: calm <=15, elevated 15-25, crisis >25)
# --------------------------------------------------------------------------- #
def test_entry_band_buckets(w6) -> None:
    assert w6.entry_band(10.0) == "calm (<=15)"
    assert w6.entry_band(15.0) == "calm (<=15)"  # inclusive upper
    assert w6.entry_band(15.01) == "elevated (15-25)"
    assert w6.entry_band(25.0) == "elevated (15-25)"  # inclusive upper
    assert w6.entry_band(25.01) == "crisis (>25)"
    assert w6.entry_band(40.0) == "crisis (>25)"
    assert w6.entry_band(float("nan")) == "unknown"


# --------------------------------------------------------------------------- #
# Friction overlay must reuse the canonical helper and only ever reduce P&L
# --------------------------------------------------------------------------- #
def test_realized_uses_canonical_helper(w6) -> None:
    """The brief mandates the canonical ``_forward_replay_realized_pnl``; with no
    friction the overlay must equal it exactly (same object, same arithmetic)."""
    from backtests.regression._common import _forward_replay_realized_pnl

    assert w6._forward_replay_realized_pnl is _forward_replay_realized_pnl
    # OTM put (spot above strike): keep full premium * 100, no assignment.
    assert w6.realized_at("none", 100.0, 2.50, 110.0) == pytest.approx(250.0)
    assert w6.realized_at("none", 100.0, 2.50, 110.0) == pytest.approx(
        _forward_replay_realized_pnl(100.0, 2.50, 110.0)
    )


def test_friction_monotonic_non_increasing(w6) -> None:
    """none >= bid_ask >= full for the same trade — friction never *adds* P&L."""
    for strike, prem, spot in [
        (100.0, 2.50, 110.0),  # OTM winner
        (100.0, 3.00, 96.0),  # ITM, still net positive
        (100.0, 1.00, 80.0),  # deep ITM loss
    ]:
        none = w6.realized_at("none", strike, prem, spot)
        bid = w6.realized_at("bid_ask", strike, prem, spot)
        full = w6.realized_at("full", strike, prem, spot)
        assert none >= bid >= full


def test_assignment_cost_only_when_itm(w6) -> None:
    """At full friction the assignment slip applies only to ITM (assigned) puts."""
    # OTM: difference vs bid_ask is exactly the open commission (0.65).
    otm_full = w6.realized_at("full", 100.0, 2.50, 110.0)
    otm_bid = w6.realized_at("bid_ask", 100.0, 2.50, 110.0)
    assert otm_bid - otm_full == pytest.approx(0.65, abs=1e-9)
    # ITM: difference vs bid_ask is open commission + assignment slip (>0.65).
    itm_full = w6.realized_at("full", 100.0, 3.00, 96.0)
    itm_bid = w6.realized_at("bid_ask", 100.0, 3.00, 96.0)
    assert itm_bid - itm_full > 0.65


# --------------------------------------------------------------------------- #
# Cluster bootstrap
# --------------------------------------------------------------------------- #
def test_boot_ci_brackets_mean_and_is_deterministic(w6) -> None:
    vals = [100.0, -50.0, 200.0, 0.0, 75.0, -25.0]
    dates = ["d1", "d1", "d2", "d2", "d3", "d3"]
    lo, hi = w6._boot_mean_ci(vals, dates)
    mean = sum(vals) / len(vals)
    assert lo <= mean <= hi
    # deterministic (fixed SEED)
    lo2, hi2 = w6._boot_mean_ci(vals, dates)
    assert (lo, hi) == (lo2, hi2)


def test_boot_ci_empty(w6) -> None:
    lo, hi = w6._boot_mean_ci([], [])
    assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------- #
# Verdict logic (synthetic by_window inputs)
# --------------------------------------------------------------------------- #
def _win(top_mean: float, top_ci: list[float], base: float = -38.0, n: int = 120) -> dict:
    """One window's calm_elevated block with all cuts make_verdict reads."""
    tb = {"n": n, "mean_pnl_full": top_mean, "mean_pnl_full_ci95": top_ci}
    return {
        "calm_elevated (VIX<=25)": {
            "top_bin": tb,
            "all": {
                "n": n * 4,
                "mean_pnl_full": base,
                "mean_pnl_full_ci95": [base - 50, base + 50],
            },
            "non_top_bin": {"n": n * 3, "mean_pnl_full": base - 8, "mean_pnl_full_ci95": [0, 0]},
            "top_bin_would_trade": {
                "n": n // 2,
                "mean_pnl_full": top_mean + 20,
                "mean_pnl_full_ci95": top_ci,
            },
        },
        "calm (<=15)": {"top_bin": dict(tb)},
        "elevated (15-25)": {"top_bin": dict(tb)},
        "crisis (>25)": {"top_bin": dict(tb)},
    }


def _byw(default: dict, **overrides) -> dict:
    keys = [
        "2020-2024",
        "2021-2025",
        "disjoint_2020-2022",
        "disjoint_2023-2025",
        "pooled_2020-2025",
    ]
    return {k: overrides.get(k, default) for k in keys}


def test_verdict_insufficient_below_30(w6) -> None:
    v = w6.make_verdict(_byw(_win(50.0, [40.0, 60.0], n=12)))
    assert "INSUFFICIENT" in v["label"]


def test_verdict_not_net_costly_when_all_cuts_positive(w6) -> None:
    """Every cut's point estimate > 0 → NOT NET-COSTLY (no rule), even if a sub-CI straddles 0."""
    v = w6.make_verdict(_byw(_win(132.0, [15.0, 233.0])))
    assert "NOT NET-COSTLY" in v["label"]
    assert v["all_cuts_point_estimate_positive"] is True
    assert v["disjoint_halves_both_positive"] is True
    # the pooled top bin sits above BOTH baselines
    assert v["pooled_calm_elevated_top_bin"]["delta_vs_all_baseline"] > 0
    assert v["pooled_calm_elevated_top_bin"]["delta_vs_non_top_bin"] > 0


def test_verdict_not_net_costly_even_if_a_subcut_ci_straddles_zero(w6) -> None:
    """Positive point estimate but a wide CI straddling 0 is still NOT NET-COSTLY."""
    v = w6.make_verdict(_byw(_win(120.0, [-9.0, 233.0])))
    assert "NOT NET-COSTLY" in v["label"]
    assert v["ci_status_by_cut"]["pooled_2020-2025"] == "straddles_0"


def test_verdict_net_costly_when_a_cut_is_ci_negative(w6) -> None:
    """A cut whose CI excludes 0 BELOW, with not-all-positive cuts → flags net-costly."""
    v = w6.make_verdict(_byw(_win(-40.0, [-70.0, -10.0])))
    assert "NET-COSTLY" in v["label"]


def test_verdict_inconclusive_mixed_signs(w6) -> None:
    """One disjoint half negative (point est) but no CI-conclusive cost → INCONCLUSIVE."""
    pos = _win(60.0, [10.0, 110.0])
    by_window = {
        "2020-2024": pos,
        "2021-2025": pos,
        "disjoint_2020-2022": _win(-30.0, [-80.0, 20.0]),  # negative point est, CI straddles 0
        "disjoint_2023-2025": pos,
        "pooled_2020-2025": _win(40.0, [5.0, 90.0]),
    }
    v = w6.make_verdict(by_window)
    assert "INCONCLUSIVE" in v["label"]
    assert v["all_cuts_point_estimate_positive"] is False


# --------------------------------------------------------------------------- #
# Live-integration pin — ranker -> forward-replay wiring is finite + consistent
# --------------------------------------------------------------------------- #
def test_live_rank_and_replay_smoke(w6) -> None:
    import pandas as pd

    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()
    recs, diag = w6.collect(runner, ["2024-03-01"], limit=30)
    assert len(recs) > 0
    for r in recs:
        assert math.isfinite(r["pnl_full"]) and math.isfinite(r["pnl_none"])
        assert r["pnl_none"] >= r["pnl_full"]  # friction monotonic on real data
        assert r["top_bin"] == bool(r["prob_profit"] > w6.TOP_BIN_PROB)
        assert r["band"] in {"calm (<=15)", "elevated (15-25)", "crisis (>25)", "unknown"}
    # a calm-entry date must classify as calm/elevated, not crisis
    assert {r["band"] for r in recs} <= {"calm (<=15)", "elevated (15-25)"}
    assert pd.notna(recs[0]["vix"])
    # the survivorship-skip diagnostic is present and well-formed
    assert diag["n_ranked"] >= len(recs)
    # skips (no fwd spot) are a subset of all dropped rows (NaN/parse guards drop too)
    assert 0 <= diag["n_no_fwd_spot"] <= diag["n_ranked"] - len(recs)
