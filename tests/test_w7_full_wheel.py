"""W7 — full-wheel realized-P&L pins (heavy-verify 2026-06-28, #450, capstone).

Validation/measurement-only. Pins the entry-VIX bands, the canonical put-leg
realized P&L (reuses ``backtests/regression/_common._forward_replay_realized_pnl``),
the rank-correlation helper, and the end-to-end verdict logic — WITHOUT re-running
the full cycle grid in the per-PR lane. Two live-integration pins confirm the
ranker→assignment→covered-call→recovery accounting. Full result in
``docs/HEAVY_VERIFY_2026-06-28_FULL_WHEEL_REALISM.md`` and
``docs/verification_artifacts/full_wheel_2026-06-28/w7_full_wheel.json``.
"""

from __future__ import annotations

import importlib.util
import math
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

_REPO = Path(__file__).resolve().parents[1]


def _load_w7():
    spec = importlib.util.spec_from_file_location("_w7", _REPO / "scripts" / "audit_full_wheel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def w7():
    return _load_w7()


# --------------------------------------------------------------------------- #
# Entry-VIX bands + canonical put-leg reuse
# --------------------------------------------------------------------------- #
def test_entry_band_buckets(w7) -> None:
    assert w7.entry_band(10.0) == "calm (<=15)"
    assert w7.entry_band(15.0) == "calm (<=15)"
    assert w7.entry_band(25.0) == "elevated (15-25)"
    assert w7.entry_band(25.01) == "crisis (>25)"
    assert w7.entry_band(float("nan")) == "unknown"


def test_put_leg_reuses_canonical_helper(w7) -> None:
    from backtests.regression._common import _forward_replay_realized_pnl

    assert w7._forward_replay_realized_pnl is _forward_replay_realized_pnl
    # OTM put at full friction = premium*100 - open commission (no assignment slip).
    otm = w7.realized_put_leg(100.0, 2.50, 110.0)
    assert otm == pytest.approx(2.50 * 100 - 0.08 * 2.50 * 100 - 0.65, abs=1e-6)
    # ITM put loses intrinsic and pays assignment slip.
    itm = w7.realized_put_leg(100.0, 3.00, 90.0)
    assert itm < otm


# --------------------------------------------------------------------------- #
# Rank correlation helper
# --------------------------------------------------------------------------- #
def test_spearman_perfect_and_small(w7) -> None:
    s = w7._spearman([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    assert s["rho"] == pytest.approx(1.0, abs=1e-9)
    assert s["p"] < 0.05
    assert w7._spearman([1, 2], [1, 2])["rho"] is None  # n<5 guarded


# --------------------------------------------------------------------------- #
# Verdict logic (synthetic by_window inputs)
# --------------------------------------------------------------------------- #
def _all(mean: float, sp_rho: float, sp_p: float, n: int = 200) -> dict:
    return {
        "all": {
            "n": n,
            "mean_full_cycle": mean,
            "mean_full_cycle_ci95": [mean - 30, mean + 30],
            "median_full_cycle": mean + 40,
            "mean_full_cycle_resolved_only": mean + 80,
            "mean_when_win": mean * 3,
            "mean_when_loss": -1500.0,
            "win_rate_full_cycle": 0.89,
            "win_rate_ci95": [0.88, 0.90],
            "mean_put_leg_only": mean - 100,
            "mean_buy_and_hold": mean + 500,
            "mean_ev_dollars_exante": mean - 50,
            "full_minus_putleg": 100.0,
            "full_minus_ev": 50.0,
            "assignment_rate": 0.15,
            "mean_cc_cycles_if_assigned": 2.5,
            "open_mark_rate": 0.05,
            "spearman_ev_vs_full": {"n": n, "rho": sp_rho, "p": sp_p},
        }
    }


def _byw(pooled_mean, sp_rho, sp_p, **cut_means) -> dict:
    bw = {
        "pooled_2020-2025": _all(pooled_mean, sp_rho, sp_p),
        "disjoint_2020-2022": _all(cut_means.get("d2020", pooled_mean), sp_rho, sp_p),
        "disjoint_2023-2025": _all(cut_means.get("d2023", pooled_mean), sp_rho, sp_p),
        "named_2020-2024": _all(cut_means.get("n2024", pooled_mean), sp_rho, sp_p),
        "named_2021-2025": _all(cut_means.get("n2025", pooled_mean), sp_rho, sp_p),
    }
    return bw


def test_verdict_realistic_through_full_wheel(w7) -> None:
    v = w7.make_verdict(_byw(317.0, 0.47, 0.0005))
    assert "STAYS REALISTIC THROUGH THE FULL WHEEL" in v["label"]
    assert v["all_cuts_mean_positive"] is True
    assert v["ranking_authority_holds_moderate"] is True
    assert v["win_rate_majority_robust"] is True
    # ranking strength is reported as moderate with variance-explained, not "authoritative"
    assert v["spearman_ev_vs_full_cycle"]["rho_squared_variance_explained"] is not None
    assert "caveat" in v["spearman_ev_vs_full_cycle"]


def test_verdict_foregrounds_tail_and_bh_and_putev(w7) -> None:
    """The reframed verdict must surface the short-vol tail, B&H underperformance,
    and the put-EV-vs-put-realized (NOT 'conservative')."""
    bw = _byw(317.0, 0.47, 0.0005)
    bw["pooled_2020-2025"]["all"]["mean_full_cycle_ci95"] = [-184.0, 559.0]
    v = w7.make_verdict(bw)
    assert v["mean_ci_straddles_zero"] is True
    assert "short-volatility" in v["label"]
    assert v["buy_and_hold_outperforms_wheel"] is True
    assert "UNDERPERFORMS a capital-matched buy-and-hold" in v["label"]
    # put-EV vs put-realized is surfaced (engine over-estimated the put leg here: 217 vs 217-100)
    assert v["pooled_full_cycle"]["put_ev_minus_put_realized"] is not None
    assert "conservative" not in v["label"]


def test_verdict_positive_but_weak_ranking(w7) -> None:
    v = w7.make_verdict(_byw(317.0, 0.05, 0.40))  # positive $ but ranking not significant
    assert "WEAK" in v["label"]
    assert v["all_cuts_mean_positive"] is True
    assert v["ranking_authority_holds_moderate"] is False


def test_verdict_materially_worse_when_a_cut_negative(w7) -> None:
    v = w7.make_verdict(_byw(317.0, 0.47, 0.0005, d2020=-120.0))
    assert "MATERIALLY WORSE" in v["label"]
    assert v["all_cuts_mean_positive"] is False
    assert v["min_cut_mean_full_cycle"] == -120.0


def test_verdict_insufficient_below_30(w7) -> None:
    v2 = w7.make_verdict(
        {**_byw(317.0, 0.47, 0.0005), "pooled_2020-2025": _all(317.0, 0.47, 0.0005, n=12)}
    )
    assert "INSUFFICIENT" in v2["label"]


# --------------------------------------------------------------------------- #
# VALUE-ASSERT — assigned→called-away cycle pinned to hand-computed ground truth.
# This is the pin that would have caught the assignment-leg double-count (#451 review):
# the stock leg must continue from spot_at_put_expiry (where the canonical put leg's
# mark left off), NOT from the strike.
# --------------------------------------------------------------------------- #
class _StubConn:
    """Returns a controlled close for a queried target date (mirrors _spot_on_or_after)."""

    def __init__(self, prices):
        self._p = prices  # {start_date_iso: close}

    def get_ohlcv(self, ticker, start_date=None, end_date=None):
        import pandas as pd

        if start_date in self._p:
            return pd.DataFrame({"close": [self._p[start_date]]})
        return pd.DataFrame({"close": []})


class _StubRunner:
    """Returns one engine covered call (strike/premium/dte)."""

    def __init__(self, conn, cc):
        self.connector = conn
        self._cc = cc

    def rank_covered_calls_by_ev(self, **kwargs):
        import pandas as pd

        return pd.DataFrame([self._cc])


def test_value_assert_assigned_called_away_ground_truth(w7) -> None:
    from backtests.regression._common import (
        friction_adjusted_premium,
        friction_assignment_cost,
        friction_open_cost,
    )

    # Put: strike 100, premium 3.00, 35 DTE entered 2022-01-03 -> expiry 2022-02-07.
    # ITM at 90 (assigned). CC: strike 105, premium 2.00, 35 DTE -> expiry 2022-03-14.
    # Called away at 110 (> 105).
    conn = _StubConn({"2022-02-07": 90.0, "2022-03-14": 110.0})
    runner = _StubRunner(conn, {"strike": 105.0, "premium": 2.00, "dte": 35})
    cand = {
        "ticker": "TEST",
        "strike": 100.0,
        "premium": 3.00,
        "dte": 35,
        "spot": 95.0,
        "ev_dollars": 50.0,
        "as_of": "2022-01-03",
        "vix": 20.0,
        "band": "elevated (15-25)",
    }
    rec = w7.simulate_cycle(runner, conn, cand)

    # Ground truth, built from the canonical helpers (no double-count of the intrinsic):
    put_leg = w7.realized_put_leg(100.0, 3.00, 90.0)  # premium − intrinsic − frictions
    cc_leg = friction_adjusted_premium(2.00, "full") * 100 - friction_open_cost(1, "full")
    stock_leg = (105.0 - 90.0) * 100 - friction_assignment_cost(105.0, 1, "full")  # basis=spot@exp
    expected = put_leg + cc_leg + stock_leg

    assert rec["resolution"] == "called_away"
    assert rec["assigned"] is True
    assert rec["full_cycle_realized"] == pytest.approx(round(expected, 2), abs=0.01)
    # the bug reported full=0 on the canonical strike=100/spot=90/.. example; guard the sign too
    assert rec["full_cycle_realized"] > 0


# --------------------------------------------------------------------------- #
# Live-integration pins — the cycle accountant on real engine output
# --------------------------------------------------------------------------- #
def test_live_cycle_accounting(w7) -> None:
    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()
    # a crisis date guarantees some assignments + recovery legs to exercise
    recs, diag = w7.collect(runner, ["2022-06-01"], limit=40, top_n=20)
    assert len(recs) > 0
    valid_res = {"put_otm", "called_away", "open_mark"}
    saw_assigned = False
    for r in recs:
        assert r["resolution"] in valid_res
        assert math.isfinite(r["full_cycle_realized"])
        assert math.isfinite(r["put_leg_realized"])
        if r["resolution"] == "put_otm":
            # OTM: no assignment, full-cycle is exactly the put leg
            assert r["assigned"] is False
            assert r["full_cycle_realized"] == r["put_leg_realized"]
        else:
            # assigned -> recovery legs added, so full-cycle differs from put leg
            assert r["assigned"] is True
            saw_assigned = True
            assert r["n_cc_cycles"] >= 0
        assert r["band"] in {"calm (<=15)", "elevated (15-25)", "crisis (>25)", "unknown"}
    assert saw_assigned, "crisis date should produce at least one assignment cycle"
    assert diag["n_cycles"] == len(recs)


def test_live_resolutions_are_forward_and_consistent(w7) -> None:
    """Every resolution_date is strictly after entry (no lookahead); assigned cycles
    that get called away realize at/after the put expiry (recovery takes time)."""
    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()
    recs, _ = w7.collect(runner, ["2022-06-01"], limit=40, top_n=20)
    for r in recs:
        assert r["resolution_date"] is not None
        assert r["resolution_date"] > r["as_of"]  # forward in time
        if r["assigned"]:
            # a called-away cycle must have sold at least one covered call
            if r["resolution"] == "called_away":
                assert r["n_cc_cycles"] >= 1
