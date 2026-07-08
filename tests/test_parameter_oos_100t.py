"""Locks for the 100-name parameter-OOS replication (extends #484).

Reuses the shared ``backtests.parameter_oos`` library — same leakage/re-fit code
as the 24-name gate, so the two runs are directly comparable. Adds coverage for
the daily-sampling significance upgrade (per-date cross-sectional rho +
date-clustered bootstrap) and the E3 drop-name breadth check.

Layers mirror #484: fixture-independent unit tests (always run) + fixture↔snapshot
recompute lock (fast) + a cheap engine spot-check regeneration (`backtest_regression`
marker — the full daily regen is ~3h, so the lock re-ranks only a few dates).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtests import parameter_oos as poos

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "param_oos" / "rank_table_100t.csv"
_SNAPSHOT = _REPO_ROOT / "backtests" / "regression" / "snapshots" / "param_oos_regime_100t.json"


# ---------------------------------------------------------------------------
# Fixture-independent unit tests (always run)
# ---------------------------------------------------------------------------


def _daily_table(dates, tickers, ev, realized):
    rows = []
    for dt, tk, e, r in zip(dates, tickers, ev, realized, strict=True):
        rows.append(
            {
                "date": dt,
                "ticker": tk,
                "ev_raw": e,
                "ev_dollars": e,
                "hmm_regime": "normal",
                "hmm_multiplier": 1.0,
                "dealer_multiplier": 1.0,
                "regime_multiplier": 1.0,
                "prob_profit": 0.8,
                "iv": 0.3,
                "premium": 1.0,
                "strike": 100.0,
                "expiration_date": dt,
                "spot_at_expiry": 100.0,
                "realized_pnl": r,
            }
        )
    return pd.DataFrame(rows)


def test_per_date_cross_sectional_rho_perfect_and_inverted():
    # day A: ev perfectly orders realized (+rho); day B: perfectly inverted (-rho)
    dates = ["2022-01-03"] * 4 + ["2022-01-04"] * 4
    tk = [f"T{i}" for i in range(4)] * 2
    ev = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    realized = [10, 20, 30, 40, 40, 30, 20, 10]
    t = _daily_table(dates, tk, ev, realized)
    xs = poos.per_date_cross_sectional_rho(t, "ev_dollars", min_rows=3)
    assert xs["n_dates"] == 2
    assert xs["mean_rho"] == pytest.approx(0.0, abs=1e-9)  # +1 and -1 average to 0
    assert xs["frac_positive"] == pytest.approx(0.5)


def test_cluster_bootstrap_is_deterministic_and_brackets_point():
    rng = np.random.default_rng(3)
    n_dates = 40
    dates, tk, ev, realized = [], [], [], []
    for d in range(n_dates):
        day = f"2022-{1 + d % 12:02d}-{1 + d % 27:02d}"
        for i in range(6):
            dates.append(day)
            tk.append(f"T{i}")
            e = rng.uniform(0, 100)
            ev.append(e)
            realized.append(e + rng.normal(0, 30))
    t = _daily_table(dates, tk, ev, realized)
    ci1 = poos.cluster_bootstrap_ci(t, stat="pooled", n_boot=300, seed=7)
    ci2 = poos.cluster_bootstrap_ci(t, stat="pooled", n_boot=300, seed=7)
    assert ci1 == ci2  # deterministic under fixed seed
    assert ci1["ci95"][0] <= ci1["point"] <= ci1["ci95"][1]
    # a different seed gives a (slightly) different interval but same point
    ci3 = poos.cluster_bootstrap_ci(t, stat="pooled", n_boot=300, seed=99)
    assert ci3["point"] == pytest.approx(ci1["point"])


def test_moving_block_bootstrap_effective_n():
    """The moving-block bootstrap reports the reduced effective independent-unit
    count: n_eff_blocks = ceil(n_dates / block_len). (The interval WIDENING it
    produces on serially-correlated real data is shown in the snapshot's
    naive-vs-block contrast, not asserted here — it is empirical, not a hard
    inequality on arbitrary synthetic data.)"""
    dates, tk, ev, realized = [], [], [], []
    for d in range(60):
        day = f"2022-{1 + d % 12:02d}-{1 + d % 27:02d}"
        for i in range(6):
            dates.append(day)
            tk.append(f"T{i}")
            ev.append(float(i))
            realized.append(float(i) + (d % 3))
    t = _daily_table(dates, tk, ev, realized)
    ci1 = poos.cluster_bootstrap_ci(t, stat="pooled", n_boot=200, seed=5, block_len=1)
    ci10 = poos.cluster_bootstrap_ci(t, stat="pooled", n_boot=200, seed=5, block_len=10)
    assert ci1["n_eff_blocks"] == 60 and ci1["block_len"] == 1
    assert ci10["n_eff_blocks"] == 6 and ci10["block_len"] == 10  # ceil(60/10)
    assert ci1["ci95"][0] <= ci1["point"] <= ci1["ci95"][1]
    assert ci10["ci95"][0] <= ci10["point"] <= ci10["ci95"][1]


def test_dominant_name_robustness_flags_concentration():
    # A "breadth-less" edge: rho is driven entirely by one name's rows.
    dates, tk, ev, realized = [], [], [], []
    for d in range(30):
        day = f"2022-{1 + d % 12:02d}-{1 + d % 27:02d}"
        # noise names: ev uncorrelated with realized
        for i in range(4):
            dates.append(day)
            tk.append(f"N{i}")
            ev.append(float(i))
            realized.append(float((i * 7) % 5 - 2))
        # dominant name DRIVER: ev strongly tracks realized, large magnitude
        dates.append(day)
        tk.append("BKNG")
        ev.append(1000.0 + d)
        realized.append(5000.0 + 10 * d)
    t = _daily_table(dates, tk, ev, realized)
    part = poos.make_partition(t, train_end="2022-06-30", holdout_start="2022-08-20")
    # early-2022 dates all resolve same-day here; use a trivially-clean split
    e3 = poos.dominant_name_robustness(t, t, dominant="BKNG")
    assert e3["dominant"] == "BKNG"
    assert e3["empirical_dominant"] == "BKNG"
    assert abs(e3["dominant_pnl_share_of_abs"]) > 0.5  # BKNG dominates the P&L
    _ = part


# ---------------------------------------------------------------------------
# Fixture + snapshot recompute lock (fast lane)
# ---------------------------------------------------------------------------

_TOL = 1e-6


def _load_snapshot():
    with open(_SNAPSHOT, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="100t rank-table fixture not committed yet")
def test_fixture_shape_and_sampling():
    t = pd.read_csv(_FIXTURE)
    assert set(poos.RANK_TABLE_COLUMNS).issubset(t.columns)
    assert t["date"].nunique() > 200  # fine (near-daily) cadence
    assert (t["dealer_multiplier"].dropna() == 1.0).all()  # inert on bloomberg


@pytest.mark.skipif(
    not (_FIXTURE.exists() and _SNAPSHOT.exists()), reason="fixture/snapshot not committed yet"
)
def test_snapshot_deterministic_stats_match_fixture():
    """Recompute the DETERMINISTIC statistics from the committed fixture and
    assert they match the snapshot. Bootstrap CIs (RNG) are checked structurally
    (point == deterministic rho; interval finite and brackets the point)."""
    from scripts.run_parameter_oos_100t import CONFIG

    snap = _load_snapshot()
    t = pd.read_csv(_FIXTURE)

    # walk-forward folds
    folds = poos.rolling_folds(t, n_folds=CONFIG["n_walk_forward_folds"])
    for got, exp in zip(folds, snap["walk_forward_folds"], strict=True):
        assert got["n"] == exp["n"]
        for k in ("rho", "hit_rate"):
            if exp[k] is not None and not (isinstance(exp[k], float) and math.isnan(exp[k])):
                assert got[k] == pytest.approx(exp[k], abs=_TOL)

    # parameter hold-out (deterministic rho variants)
    part = poos.make_partition(
        t, train_end=CONFIG["train_end"], holdout_start=CONFIG["holdout_start"]
    )
    rep = poos.parameter_holdout_report(part)
    for variant, ev in snap["parameter_holdout"]["variants"].items():
        for k in ("train_rho", "holdout_rho"):
            assert rep["variants"][variant][k] == pytest.approx(ev[k], abs=_TOL)

    # independence-corrected: cross-sectional aggregates + pooled point
    xs = poos.per_date_cross_sectional_rho(t, "ev_dollars")
    ic = snap["independence_corrected"]
    assert xs["mean_rho"] == pytest.approx(ic["cross_sectional_mean_rho"], abs=_TOL)
    assert xs["n_dates"] == ic["cross_sectional_n_dates"]
    assert poos._pooled_rho(t, "ev_dollars") == pytest.approx(ic["pooled_rho"], abs=_TOL)
    for block in ("pooled_rho_ci", "holdout_pooled_ci"):
        ci = ic[block]
        assert ci["ci95"][0] <= ci["point"] <= ci["ci95"][1]
        assert math.isfinite(ci["se"])

    # E3 deterministic point estimates
    e3 = poos.dominant_name_robustness(
        t, part.holdout, dominant=CONFIG["e3_dominant_name"]
    )
    for k in ("full_rho", "drop_dominant_rho", "loo_min_rho", "loo_max_rho"):
        assert e3[k] == pytest.approx(snap["e3_robustness"][k], abs=_TOL)


@pytest.mark.skipif(
    not (_FIXTURE.exists() and _SNAPSHOT.exists()), reason="fixture/snapshot not committed yet"
)
def test_top_n_tiers_deterministic_and_holdout_edge_survives():
    """Lock the DECISIVE result: the tradeable top-tier rank edge survives
    out-of-window. Recompute the deterministic per-tier ρ from the fixture
    (exact) and assert the committed holdout top-5 / top-15 block-bootstrap CIs
    EXCLUDE zero, while the all-candidate CI includes zero (the reconciliation
    with the #484 24-name null). The top-15 edge is breadth (leave-one-out stays
    positive)."""
    from datetime import date

    from scripts.run_parameter_oos_100t import CONFIG

    snap = _load_snapshot()
    t = pd.read_csv(_FIXTURE)
    d = pd.to_datetime(t["date"]).dt.date
    holdout = t[d >= date.fromisoformat(CONFIG["holdout_start"])]
    exp = snap["top_n_tiers"]["holdout"]
    for n, key in [(5, "5"), (15, "15"), (None, "all")]:
        top = poos.restrict_top_n_per_date(holdout, n)
        assert poos._pooled_rho(top, "ev_dollars") == pytest.approx(exp[key]["pooled_rho"], abs=_TOL)
    # surviving-edge: committed holdout top-5 / top-15 block CIs exclude zero
    assert exp["5"]["block_ci95"][0] > 0, "holdout top-5 block-CI must exclude zero"
    assert exp["15"]["block_ci95"][0] > 0, "holdout top-15 block-CI must exclude zero"
    # reconciliation with #484: the all-candidate holdout CI includes zero
    assert exp["all"]["block_ci95"][0] < 0 < exp["all"]["block_ci95"][1] or exp["all"]["pooled_rho"] <= 0
    # breadth: the holdout top-15 edge is not one name
    et = snap["e3_holdout_top15"]
    assert et["loo_min_rho"] > 0.2, "holdout top-15 edge collapses when one name dropped"
    assert abs(et["full_rho"] - et["drop_dominant_rho"]) < 0.02


@pytest.mark.skipif(not _SNAPSHOT.exists(), reason="snapshot not committed yet")
def test_snapshot_leakage_certificate_is_clean():
    snap = _load_snapshot()
    cert = snap["parameter_holdout"]["leakage_certificate"]
    assert cert["leakage_free"] is True
    assert cert["max_train_expiration_date"] < cert["min_holdout_ranking_date"]
    for split in snap["split_robustness"]:
        assert split["leakage_free"] is True


@pytest.mark.skipif(not _SNAPSHOT.exists(), reason="snapshot not committed yet")
def test_snapshot_fingerprint_and_sampling_labelled():
    snap = _load_snapshot()
    fp = snap["fingerprint"]
    for k in ("universe", "every_n_bdays", "actual_sample_dates", "actual_first_date",
              "actual_last_date", "connector_data_sha256"):
        assert k in fp, f"fingerprint missing {k}"
    assert fp["universe"] == "UNIVERSE_100"


# ---------------------------------------------------------------------------
# Engine spot-check regeneration (slow marker; cheap — re-ranks a few dates)
# ---------------------------------------------------------------------------


@pytest.mark.backtest_regression
@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture not committed yet")
def test_engine_spotcheck_matches_fixture():
    """Re-rank a handful of the fixture's own as_of dates from the live engine
    and assert the ev_dollars match — a cheap engine-drift guard (the full daily
    regen would be ~3h)."""
    from datetime import date

    from scripts.run_parameter_oos_100t import CONFIG, _universe

    committed = pd.read_csv(_FIXTURE)
    sample_dates = sorted(committed["date"].unique())
    probe = [date.fromisoformat(sample_dates[i]) for i in (0, len(sample_dates) // 2, -1)]
    fresh = poos.build_rank_table(
        tickers=_universe(),
        sample_dates=probe,
        dte_target=CONFIG["dte_target"],
        delta_target=CONFIG["delta_target"],
        top_n=CONFIG["top_n"],
        progress=False,
    )
    for d in probe:
        ds = d.isoformat()
        c = committed[committed["date"] == ds].set_index("ticker")["ev_dollars"].sort_index()
        f = fresh[fresh["date"] == ds].set_index("ticker")["ev_dollars"].sort_index()
        common = c.index.intersection(f.index)
        assert len(common) > 0, f"no common tickers on {ds}"
        assert np.allclose(c.loc[common], f.loc[common], atol=0.01), f"ev_dollars drift on {ds}"
