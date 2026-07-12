"""Tests for the tail-risk exceedance validation harness
(``backtests/tail_exceedance.py``).

Fast lane, engine-free: every statistic is exercised on hand-computed or
synthetic data.  The end-to-end pair is the load-bearing contract — a
generator whose realized outcomes come from the SAME distribution as the
modeled quantiles must PASS, and a generator whose true tail is fatter than
modeled must FAIL on the lower-tail tests.  If those two ever stop holding,
the harness (not the engine) is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests import tail_exceedance as tex

# ---------------------------------------------------------------------------
# kupiec_pof
# ---------------------------------------------------------------------------


class TestKupiecPOF:
    def test_exact_nominal_rate_gives_zero_lr(self):
        out = tex.kupiec_pof(400, 100, 0.25)
        assert out["lr"] == pytest.approx(0.0, abs=1e-12)
        assert out["p_value"] == pytest.approx(1.0)

    def test_hand_computed_lr(self):
        # n=250, x=25, coverage 5%: LR = -2[ ll(p) - ll(pi_hat) ], pi_hat=0.10.
        n, x, p = 250, 25, 0.05
        pi = x / n
        ll_p = x * np.log(p) + (n - x) * np.log(1 - p)
        ll_pi = x * np.log(pi) + (n - x) * np.log(1 - pi)
        expected = -2.0 * (ll_p - ll_pi)
        out = tex.kupiec_pof(n, x, p)
        assert out["lr"] == pytest.approx(expected, rel=1e-12)
        # 10% observed vs 5% nominal on n=250 must reject at any sane level.
        assert out["p_value"] < 0.01

    def test_zero_violations_edge(self):
        out = tex.kupiec_pof(100, 0, 0.05)
        assert np.isfinite(out["lr"]) and out["lr"] >= 0.0
        assert 0.0 <= out["p_value"] <= 1.0

    def test_all_violations_edge(self):
        out = tex.kupiec_pof(50, 50, 0.05)
        assert np.isfinite(out["lr"])
        assert out["p_value"] < 1e-10

    def test_degenerate_inputs_nan(self):
        assert np.isnan(tex.kupiec_pof(0, 0, 0.05)["lr"])
        assert np.isnan(tex.kupiec_pof(10, 1, 0.0)["lr"])
        assert np.isnan(tex.kupiec_pof(10, 1, 1.0)["lr"])

    def test_more_violations_monotone_lr(self):
        lrs = [tex.kupiec_pof(200, x, 0.05)["lr"] for x in (10, 20, 30, 40)]
        assert lrs == sorted(lrs)


# ---------------------------------------------------------------------------
# date_clustered_rate_ci
# ---------------------------------------------------------------------------


class TestClusteredCI:
    def test_ci_brackets_rate_and_is_deterministic(self):
        rng = np.random.default_rng(7)
        dates = np.repeat([f"2024-01-{d:02d}" for d in range(1, 21)], 10)
        viol = rng.random(200) < 0.25
        a = tex.date_clustered_rate_ci(dates, viol, n_boot=500, seed=1)
        b = tex.date_clustered_rate_ci(dates, viol, n_boot=500, seed=1)
        assert a == b  # seeded determinism
        assert a["ci_low"] <= a["rate"] <= a["ci_high"]
        assert a["n_dates"] == 20.0

    def test_clustered_ci_wider_than_iid_binomial(self):
        # All violations concentrated on a few dates -> date resampling must
        # produce a much wider CI than an iid binomial would suggest.
        dates = np.repeat([f"2024-02-{d:02d}" for d in range(1, 11)], 20)
        viol = np.zeros(200, dtype=bool)
        viol[:40] = True  # dates 01+02 are 100% violation, others 0%
        out = tex.date_clustered_rate_ci(dates, viol, n_boot=1000, seed=3)
        iid_half_width = 1.96 * np.sqrt(0.2 * 0.8 / 200)
        assert (out["ci_high"] - out["ci_low"]) / 2 > 2 * iid_half_width

    def test_empty_inputs(self):
        out = tex.date_clustered_rate_ci(np.array([]), np.array([]))
        assert np.isnan(out["rate"])


# ---------------------------------------------------------------------------
# date_rate_autocorr_test
# ---------------------------------------------------------------------------


class TestClusteringTest:
    def test_bursty_violations_detected(self):
        # 40 dates, violations only in one contiguous 10-date burst.
        dates = np.repeat(
            [f"2023-03-{d:02d}" for d in range(1, 31)] + [f"2023-04-{d:02d}" for d in range(1, 11)],
            5,
        )
        rate = np.zeros(40)
        rate[15:25] = 1.0
        viol = np.repeat(rate, 5).astype(bool)
        out = tex.date_rate_autocorr_test(dates, viol, n_perm=500, seed=2)
        assert out["autocorr"] > 0.5
        assert out["p_value"] < 0.05

    def test_iid_violations_not_flagged(self):
        # Median p over several independent draws: a single draw can land in
        # the 5% Type-I region by construction; the median across draws
        # cannot unless the test statistic itself is biased.
        dates = np.repeat([f"2023-05-{d:02d}" for d in range(1, 31)], 8)
        pvals = []
        for data_seed in (11, 12, 13, 14, 15):
            rng = np.random.default_rng(data_seed)
            viol = rng.random(240) < 0.25
            pvals.append(tex.date_rate_autocorr_test(dates, viol, n_perm=500, seed=2)["p_value"])
        assert float(np.median(pvals)) > 0.05

    def test_constant_series_neutral(self):
        dates = np.repeat([f"2023-06-{d:02d}" for d in range(1, 11)], 3)
        out = tex.date_rate_autocorr_test(dates, np.zeros(30, dtype=bool))
        assert out["autocorr"] == 0.0 and out["p_value"] == 1.0

    def test_too_few_dates_nan(self):
        out = tex.date_rate_autocorr_test(np.array(["2023-01-01"] * 5), np.ones(5, dtype=bool))
        assert np.isnan(out["p_value"])


# ---------------------------------------------------------------------------
# heterogeneous_coverage_z
# ---------------------------------------------------------------------------


class TestHeterogeneousCoverage:
    def test_calibrated_probs_small_z(self):
        rng = np.random.default_rng(5)
        prob = rng.uniform(0.5, 0.95, size=5000)
        win = (rng.random(5000) < prob).astype(float)
        out = tex.heterogeneous_coverage_z(prob, win)
        assert abs(out["z"]) < 3.0

    def test_overconfident_probs_large_negative_z(self):
        rng = np.random.default_rng(5)
        prob = np.full(2000, 0.95)
        win = (rng.random(2000) < 0.80).astype(float)  # true rate 0.80 vs claimed 0.95
        out = tex.heterogeneous_coverage_z(prob, win)
        assert out["z"] < -5.0
        assert out["p_value"] < 1e-6

    def test_empty_nan(self):
        out = tex.heterogeneous_coverage_z(np.array([]), np.array([]))
        assert np.isnan(out["z"])


# ---------------------------------------------------------------------------
# vix_band
# ---------------------------------------------------------------------------


class TestVixBand:
    @pytest.mark.parametrize(
        ("vix", "band"),
        [
            (12.0, "calm"),
            (15.0, "calm"),
            (15.01, "elevated"),
            (25.0, "elevated"),
            (25.01, "crisis"),
            (80.0, "crisis"),
            (float("nan"), "unknown"),
            (None, "unknown"),
            (0.0, "unknown"),
        ],
    )
    def test_bands(self, vix, band):
        assert tex.vix_band(vix) == band


# ---------------------------------------------------------------------------
# End-to-end: synthetic calibrated vs understated-tail tables
# ---------------------------------------------------------------------------


def _synthetic_table(
    *,
    n_dates: int = 60,
    per_date: int = 12,
    tail_scale: float = 1.0,
    win_rate: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic tail table whose modeled quantiles/cvar_5 come from a
    known P&L distribution, with realized P&L drawn from the same
    distribution scaled by ``tail_scale`` on the loss side.

    ``tail_scale=1.0`` -> perfectly specified model (harness must PASS);
    ``tail_scale>1``   -> true losses run ``tail_scale``x the modeled ones
    (harness must FAIL the lower-tail tests).

    Default ``win_rate=0.20`` keeps all three quartiles in the loss region
    (``prob_profit < 1 - nominal`` for p25/p50/p75) so every coverage test
    is informative; ``win_rate=0.80`` reproduces the realistic short-put
    point mass where p25/p50/p75 all sit on the win value and the harness
    must EXCLUDE them (see ``test_point_mass_rows_excluded``).
    """
    rng = np.random.default_rng(seed)
    rows = []
    # Modeled per-path P&L population: short-put-shaped (capped win, long
    # left tail): win = +premium, loss = premium - lognormal drawdown.
    premium = 200.0
    model_pop = np.where(
        rng.random(200_000) < win_rate,
        premium,
        premium - rng.lognormal(mean=6.0, sigma=0.8, size=200_000),
    )
    p25, p50, p75 = np.percentile(model_pop, [25, 50, 75])
    var5 = np.percentile(model_pop, 5)
    cvar5 = model_pop[model_pop <= var5].mean()
    for d in range(n_dates):
        day = f"2024-{1 + d // 28:02d}-{1 + d % 28:02d}"
        for k in range(per_date):
            r = float(
                premium
                if rng.random() < win_rate
                else premium - tail_scale * rng.lognormal(mean=6.0, sigma=0.8)
            )
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{k:02d}",
                    "ev_raw": 10.0,
                    "ev_dollars": 10.0,
                    "prob_profit": win_rate,
                    "n_scenarios": 35,
                    "distribution_source": "empirical_non_overlapping",
                    "hmm_regime": "normal",
                    "iv": 0.25,
                    "premium": premium / 100.0,
                    "strike": 100.0,
                    "spot": 105.0,
                    "pnl_p25": float(p25),
                    "pnl_p50": float(p50),
                    "pnl_p75": float(p75),
                    "cvar_5": float(cvar5),
                    "cvar_99_evt": float("nan"),
                    "tail_widening_factor": 1.0,
                    "vix_entry": 18.0,
                    "expiration_date": "2024-12-20",
                    "spot_at_expiry": 100.0,
                    "realized_pnl": r,
                }
            )
    return pd.DataFrame(rows, columns=list(tex.TAIL_TABLE_COLUMNS))


class TestEndToEnd:
    def test_calibrated_model_passes(self):
        table = _synthetic_table(tail_scale=1.0)
        report = tex.full_report(table, meta={"case": "calibrated"})
        assert report["overall_verdict"] in ("PASS", "WARN")  # never FAIL
        assert report["cvar_5"]["verdict"] != "FAIL"
        # cvar breach rate must respect the ES < VaR bound.
        assert report["cvar_5"]["rate"] < 0.05 + 0.02

    def test_understated_tail_fails(self):
        table = _synthetic_table(tail_scale=3.0)
        report = tex.full_report(table, meta={"case": "understated"})
        assert report["overall_verdict"] == "FAIL"
        assert report["cvar_5"]["verdict"] == "FAIL"
        # Severity block exists and shows losses beyond modeled ES.
        assert report["cvar_5"]["severity"]["mean_excess_dollars"] < 0.0
        assert report["cvar_5"]["severity"]["median_realized_over_cvar"] > 1.0

    def test_insufficient_rows(self):
        table = _synthetic_table(n_dates=2, per_date=2)
        report = tex.full_report(table)
        for k in ("p25", "p50", "p75"):
            assert report["quantiles"][k]["verdict"] == "INSUFFICIENT"
        assert report["cvar_5"]["verdict"] == "INSUFFICIENT"

    def test_unresolved_rows_excluded(self):
        table = _synthetic_table(n_dates=40, per_date=8)
        table.loc[table.index[: len(table) // 2], "realized_pnl"] = float("nan")
        report = tex.full_report(table)
        assert report["rows_resolved"] == len(table) - len(table) // 2

    def test_conservative_model_never_fails(self):
        # Model claims a FATTER tail than reality (tail_scale < 1): safe
        # direction for short vol -> must not FAIL.
        table = _synthetic_table(tail_scale=0.4)
        report = tex.full_report(table, meta={"case": "conservative"})
        assert report["overall_verdict"] in ("PASS", "WARN")

    def test_point_mass_rows_excluded(self):
        # Realistic short-put shape (80% wins): all three quartiles sit ON
        # the max-profit point mass, so continuous-coverage nominals do not
        # apply.  The V1-a 24t run exposed this as byte-identical p50/p75
        # violation rates; the harness must EXCLUDE such rows loudly rather
        # than emit a vacuous PASS.
        table = _synthetic_table(tail_scale=1.0, win_rate=0.80)
        report = tex.full_report(table, meta={"case": "point_mass"})
        for k in ("p25", "p50", "p75"):
            q = report["quantiles"][k]
            assert q["n"] == 0
            assert q["n_excluded_point_mass"] == len(table)
            assert q["verdict"] == "INSUFFICIENT"
        # cvar_5 sits deep in the loss region — unaffected by the point mass.
        assert report["cvar_5"]["n"] == len(table)
        assert report["cvar_5"]["verdict"] != "INSUFFICIENT"

    def test_mixed_point_mass_partial_exclusion(self):
        # Half the rows have prob_profit 0.20 (all quartiles informative),
        # half 0.80 (none informative): the p25 test must run on exactly the
        # informative half.
        low = _synthetic_table(n_dates=40, per_date=6, win_rate=0.20, seed=1)
        high = _synthetic_table(n_dates=40, per_date=6, win_rate=0.80, seed=2)
        table = pd.concat([low, high], ignore_index=True)
        q = tex.quantile_coverage_report(table, quantile_col="pnl_p25", nominal=0.25)
        assert q["n"] == len(low)
        assert q["n_excluded_point_mass"] == len(high)
