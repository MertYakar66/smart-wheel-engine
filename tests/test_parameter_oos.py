"""Locks for the parameter-OOS validation gate.

Three layers, mirroring the S27/S34 regression pattern:

* **Fixture-independent unit tests** (always run) — the no-leakage certificate,
  offline re-weighting identities, and re-fit determinism.  These protect the
  leakage proof itself in a clean checkout.
* **Fixture + snapshot regression lock** (fast lane) — recompute the
  walk-forward + parameter-holdout report from the committed rank-table fixture
  and assert it matches the committed snapshot.  A change to the analysis (grid,
  split, metric) diverges from the snapshot and forces an explicit re-baseline.
* **Engine-regeneration lock** (``backtest_regression`` marker, slow) —
  rebuild the rank table from the live engine over the pinned config and assert
  it matches the committed fixture.  Catches engine/data drift end-to-end.

Nothing here trades, ranks, or mutates an engine default — the whole gate reads
``ev_raw`` and re-weights it offline (CLAUDE.md §2 / task invariant 3).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtests import parameter_oos as poos

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "param_oos" / "rank_table_24t.csv"
_SNAPSHOT = _REPO_ROOT / "backtests" / "regression" / "snapshots" / "param_oos_regime_24t.json"


# ---------------------------------------------------------------------------
# Fixture-independent unit tests (always run)
# ---------------------------------------------------------------------------


def _synthetic_table(*, dates, exps, labels, ev_raw, ev_dollars, realized, prob=None):
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [f"T{i}" for i in range(n)],
            "ev_raw": ev_raw,
            "ev_dollars": ev_dollars,
            "hmm_regime": labels,
            "hmm_multiplier": [1.0] * n,
            "dealer_multiplier": [1.0] * n,
            "regime_multiplier": [1.0] * n,
            "prob_profit": prob if prob is not None else [0.8] * n,
            "iv": [0.3] * n,
            "premium": [1.0] * n,
            "strike": [100.0] * n,
            "expiration_date": exps,
            "spot_at_expiry": [100.0] * n,
            "realized_pnl": realized,
        }
    )


def test_assert_no_leakage_certifies_clean_split():
    train = _synthetic_table(
        dates=["2022-01-03", "2022-02-01"],
        exps=["2022-02-07", "2022-03-08"],
        labels=["normal", "bear"],
        ev_raw=[100.0, 120.0],
        ev_dollars=[100.0, 60.0],
        realized=[50.0, -30.0],
    )
    holdout = _synthetic_table(
        dates=["2022-06-01", "2022-07-01"],
        exps=["2022-07-06", "2022-08-05"],
        labels=["normal", "bull_quiet"],
        ev_raw=[110.0, 130.0],
        ev_dollars=[110.0, 162.5],
        realized=[40.0, 70.0],
    )
    cert = poos.assert_no_leakage(train, holdout)
    assert cert["leakage_free"] is True
    assert cert["gap_days_expiry_to_holdout"] > 0


def test_assert_no_leakage_rejects_overlap():
    """A train option that expires ON/AFTER the first holdout ranking date must
    be rejected — that is the airtight per-row leakage invariant."""
    train = _synthetic_table(
        dates=["2022-01-03"],
        exps=["2022-06-15"],  # expires AFTER the holdout ranking date below
        labels=["normal"],
        ev_raw=[100.0],
        ev_dollars=[100.0],
        realized=[50.0],
    )
    holdout = _synthetic_table(
        dates=["2022-06-01"],  # ranks BEFORE the train option resolves -> leak
        exps=["2022-07-06"],
        labels=["normal"],
        ev_raw=[110.0],
        ev_dollars=[110.0],
        realized=[40.0],
    )
    with pytest.raises(AssertionError, match="LEAKAGE"):
        poos.assert_no_leakage(train, holdout)


def test_gamma_zero_is_ev_raw_and_gamma_one_is_shipped():
    t = _synthetic_table(
        dates=["2022-01-03", "2022-02-01", "2022-03-01"],
        exps=["2022-02-07", "2022-03-08", "2022-04-05"],
        labels=["crisis", "bear", "bull_quiet"],
        ev_raw=[100.0, 120.0, 80.0],
        ev_dollars=[20.0, 60.0, 100.0],
        realized=[10.0, -5.0, 30.0],
    )
    assert np.allclose(poos.apply_tilt_exponent(t, 0.0), t["ev_raw"].to_numpy())
    assert np.allclose(poos.apply_tilt_exponent(t, 1.0), t["ev_dollars"].to_numpy())


def test_apply_regime_scalars_defaults_missing_label_to_one():
    t = _synthetic_table(
        dates=["2022-01-03", "2022-02-01"],
        exps=["2022-02-07", "2022-03-08"],
        labels=["crisis", "mystery"],
        ev_raw=[100.0, 200.0],
        ev_dollars=[20.0, 200.0],
        realized=[10.0, 5.0],
    )
    out = poos.apply_regime_scalars(t, {"crisis": 0.5})
    assert out[0] == pytest.approx(50.0)  # 100 * 0.5
    assert out[1] == pytest.approx(200.0)  # unknown label -> 1.0


def test_refit_regime_scalars_is_deterministic():
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.choice(poos.REGIME_LABELS, size=n)
    ev_raw = rng.uniform(10, 300, size=n)
    # realized correlated with ev_raw so a re-fit has signal to find
    realized = ev_raw + rng.normal(0, 50, size=n)
    dates = pd.bdate_range("2021-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    exps = pd.bdate_range("2021-02-15", periods=n).strftime("%Y-%m-%d").tolist()
    t = _synthetic_table(
        dates=dates,
        exps=exps,
        labels=list(labels),
        ev_raw=ev_raw,
        ev_dollars=ev_raw,
        realized=realized,
    )
    w1, r1 = poos.refit_regime_scalars(t)
    w2, r2 = poos.refit_regime_scalars(t)
    assert w1 == w2 and r1 == r2
    assert w1["normal"] == 1.0  # gauge pinned


def test_ece_flags_overconfidence():
    # forecast 0.95 but only 50% actually win -> ECE ~0.45 in that bin
    t = _synthetic_table(
        dates=["2022-01-03"] * 4,
        exps=["2022-02-07"] * 4,
        labels=["normal"] * 4,
        ev_raw=[100.0] * 4,
        ev_dollars=[100.0] * 4,
        realized=[10.0, 10.0, -10.0, -10.0],
        prob=[0.95, 0.95, 0.95, 0.95],
    )
    ece = poos.expected_calibration_error(
        t["prob_profit"].to_numpy(), (t["realized_pnl"].to_numpy() > 0).astype(float)
    )
    assert ece == pytest.approx(0.45, abs=0.01)


# ---------------------------------------------------------------------------
# Fixture + snapshot regression lock (fast lane)
# ---------------------------------------------------------------------------

_RHO_TOL = 1e-6  # deterministic pure-numpy recompute -> near-exact


def _load_snapshot():
    import json

    with open(_SNAPSHOT, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.skipif(not _FIXTURE.exists(), reason="rank-table fixture not committed yet")
def test_fixture_has_expected_shape():
    t = pd.read_csv(_FIXTURE)
    assert set(poos.RANK_TABLE_COLUMNS).issubset(t.columns)
    assert len(t) > 100
    # dealer multiplier is inert on the bloomberg provider (documented finding)
    assert (t["dealer_multiplier"].dropna() == 1.0).all()


@pytest.mark.skipif(
    not (_FIXTURE.exists() and _SNAPSHOT.exists()),
    reason="fixture/snapshot not committed yet",
)
def test_snapshot_matches_fixture_recompute():
    """Recompute the report from the committed fixture; assert it matches the
    committed snapshot.  Diverges (and fails) if the analysis code changes —
    forcing an explicit re-baseline."""
    from scripts.run_parameter_oos import CONFIG

    snap = _load_snapshot()
    table = pd.read_csv(_FIXTURE)

    folds = poos.rolling_folds(table, n_folds=CONFIG["n_walk_forward_folds"])
    assert len(folds) == len(snap["walk_forward_folds"])
    for got, exp in zip(folds, snap["walk_forward_folds"], strict=True):
        assert got["n"] == exp["n"]
        for k in ("rho", "hit_rate", "brier", "ece"):
            if exp[k] is None or (isinstance(exp[k], float) and math.isnan(exp[k])):
                continue
            assert got[k] == pytest.approx(exp[k], abs=_RHO_TOL), f"fold {got['fold']} {k}"

    part = poos.make_partition(
        table, train_end=CONFIG["train_end"], holdout_start=CONFIG["holdout_start"]
    )
    rep = poos.parameter_holdout_report(part)
    exp_rep = snap["parameter_holdout"]
    for variant, ev in exp_rep["variants"].items():
        for k in ("train_rho", "holdout_rho"):
            assert rep["variants"][variant][k] == pytest.approx(ev[k], abs=_RHO_TOL), (
                f"{variant}.{k}"
            )
    assert rep["optimism_gap_regime_scalars"] == pytest.approx(
        exp_rep["optimism_gap_regime_scalars"], abs=_RHO_TOL
    )

    # split-robustness recompute + every split leakage-certified
    robo = poos.split_robustness_report(table, [tuple(s) for s in CONFIG["robustness_splits"]])
    exp_robo = snap["split_robustness"]
    assert len(robo) == len(exp_robo)
    for got, exp in zip(robo, exp_robo, strict=True):
        assert got["leakage_free"] is True
        assert got["optimism_gap"] == pytest.approx(exp["optimism_gap"], abs=_RHO_TOL)


@pytest.mark.skipif(not _SNAPSHOT.exists(), reason="snapshot not committed yet")
def test_snapshot_leakage_certificate_is_clean():
    snap = _load_snapshot()
    cert = snap["parameter_holdout"]["leakage_certificate"]
    assert cert["leakage_free"] is True
    assert cert["gap_days_expiry_to_holdout"] > 0
    # train options must all resolve before the holdout starts ranking
    assert cert["max_train_expiration_date"] < cert["min_holdout_ranking_date"]


@pytest.mark.skipif(not _SNAPSHOT.exists(), reason="snapshot not committed yet")
def test_snapshot_fingerprint_has_required_keys():
    snap = _load_snapshot()
    fp = snap["fingerprint"]
    for k in (
        "universe",
        "sample_start",
        "sample_end",
        "train_end",
        "holdout_start",
        "data_csv_sha256",
        "connector_data_sha256",
    ):
        assert k in fp, f"fingerprint missing {k}"


# ---------------------------------------------------------------------------
# Engine-regeneration lock (slow, behind marker)
# ---------------------------------------------------------------------------


@pytest.mark.backtest_regression
@pytest.mark.slow
@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture not committed yet")
def test_rank_table_regenerates_from_engine():
    """Rebuild the rank table from the live engine over the pinned config and
    assert it matches the committed fixture (engine/data drift guard)."""
    from datetime import date

    from scripts.run_parameter_oos import CONFIG, _universe

    committed = pd.read_csv(_FIXTURE)
    dates = poos.sample_business_days(
        CONFIG["sample_start"], CONFIG["sample_end"], CONFIG["every_n_bdays"]
    )
    fresh = poos.build_rank_table(
        tickers=_universe(),
        sample_dates=dates,
        dte_target=CONFIG["dte_target"],
        delta_target=CONFIG["delta_target"],
        top_n=CONFIG["top_n"],
        progress=False,
    )
    assert len(fresh) == len(committed), "row count drifted from fixture"
    # Compare the fixed-parameter rho on all resolved rows.
    got = poos.scorecard(fresh, signal_col="ev_dollars")
    exp = poos.scorecard(committed, signal_col="ev_dollars")
    assert got["rho"] == pytest.approx(exp["rho"], abs=0.005)
    assert got["n"] == exp["n"]
    _ = date  # silence unused if trimmed
