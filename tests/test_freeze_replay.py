"""Tests for the V2 parameter freeze-replay harness (backtests/freeze_replay.py).

Fast lane: pure-logic tests of the truncation machinery, the exact ranker
comparison, the snapshot differ, the freeze context manager (on synthetic
frames — no data files), and the offline frozen-HMM recombination.  The
engine-refit reproducibility lock (C1's definition of done) needs the
committed Bloomberg CSVs and runs in the slow lane
(``backtest_regression`` marker), mirroring test_parameter_oos.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtests import freeze_replay as fz

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "data" / "bloomberg"
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "freeze_replay" / "freeze_snapshot_24t.json"


# ---------------------------------------------------------------------------
# V2-a — truncation
# ---------------------------------------------------------------------------


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def test_truncate_csv_keeps_rows_at_or_before_cutoff(tmp_path):
    src = tmp_path / "series.csv"
    _write_csv(
        src,
        """
date,ticker,close
2023-06-29,AAPL,1.2300
2023-06-30,AAPL,0.1000000000000001
2023-07-03,AAPL,9.9
""",
    )
    dst = tmp_path / "out.csv"
    counts = fz.truncate_csv(src, dst, cutoff="2023-06-30", date_col="date")
    assert counts == {"rows_total": 3, "rows_kept": 2, "rows_dropped_noniso": 0}
    out = dst.read_text()
    assert "2023-07-03" not in out
    # Byte preservation — surviving cells must not be re-formatted.
    assert "1.2300" in out
    assert "0.1000000000000001" in out


def test_truncate_csv_drops_non_iso_dates_and_counts_them(tmp_path):
    src = tmp_path / "series.csv"
    _write_csv(src, "date,x\nnot-a-date,1\n2023-01-05,2")
    dst = tmp_path / "out.csv"
    counts = fz.truncate_csv(src, dst, cutoff="2023-12-31", date_col="date")
    assert counts["rows_kept"] == 1
    assert counts["rows_dropped_noniso"] == 1


def test_truncate_csv_missing_date_column_raises(tmp_path):
    src = tmp_path / "series.csv"
    _write_csv(src, "ticker,x\nAAPL,1")
    with pytest.raises(ValueError, match="date"):
        fz.truncate_csv(src, tmp_path / "out.csv", cutoff="2023-12-31", date_col="date")


def test_build_truncated_data_dir_tier1_vs_intact(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write_csv(src / "sp500_ohlcv.csv", "date,close\n2023-01-03,1\n2024-01-03,2")
    _write_csv(src / "sp500_earnings.csv", "announcement_date,x\n2099-01-01,future")
    sub = src / "broad_pull"
    sub.mkdir()
    _write_csv(sub / "panel.csv", "a\n1")
    dst = tmp_path / "dst"
    manifest = fz.build_truncated_data_dir(src, dst, cutoff="2023-06-30")
    assert manifest["truncated"]["sp500_ohlcv.csv"]["rows_kept"] == 1
    assert "sp500_earnings.csv" in manifest["copied_intact"]
    assert "broad_pull/" in manifest["copied_intact"]
    # Schedule-type file copied verbatim, future rows intact.
    assert "2099-01-01" in (dst / "sp500_earnings.csv").read_text()
    # Re-running with a later cutoff rewrites tier-1 only.
    manifest2 = fz.build_truncated_data_dir(src, dst, cutoff="2024-06-30")
    assert manifest2["truncated"]["sp500_ohlcv.csv"]["rows_kept"] == 2
    assert "2024-01-03" in (dst / "sp500_ohlcv.csv").read_text()


# ---------------------------------------------------------------------------
# V2-a — exact ranker comparison
# ---------------------------------------------------------------------------


def _frame(**overrides) -> pd.DataFrame:
    base = {
        "ticker": ["AAPL", "MSFT", "XOM"],
        "ev_dollars": [10.0, 20.0, np.nan],
        "prob_profit": [0.8, 0.9, 0.7],
        "distribution_source": ["empirical", "empirical", "har_rv"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_compare_identical_frames_pass():
    a, b = _frame(), _frame()
    out = fz.compare_rank_outputs(a, [], b, [])
    assert out["identical"] is True
    assert out["diff_examples"] == []


def test_compare_nan_equals_nan():
    out = fz.compare_rank_outputs(_frame(), [], _frame(), [])
    assert out["identical"] is True  # ev_dollars has NaN on both sides


def test_compare_detects_single_float_perturbation():
    b = _frame(ev_dollars=[10.0, 20.0000001, np.nan])
    out = fz.compare_rank_outputs(_frame(), [], b, [])
    assert out["identical"] is False
    assert out["diff_columns"] == {"ev_dollars": 1}
    ex = out["diff_examples"][0]
    assert ex["ticker"] == "MSFT" and ex["column"] == "ev_dollars"


def test_compare_detects_string_column_diff():
    b = _frame(distribution_source=["empirical", "block_bootstrap", "har_rv"])
    out = fz.compare_rank_outputs(_frame(), [], b, [])
    assert out["identical"] is False
    assert "distribution_source" in out["diff_columns"]


def test_compare_detects_ticker_set_diff():
    b = _frame(ticker=["AAPL", "MSFT", "JPM"])
    out = fz.compare_rank_outputs(_frame(), [], b, [])
    assert out["identical"] is False
    assert out["tickers_only_a"] == ["XOM"]
    assert out["tickers_only_b"] == ["JPM"]


def test_compare_row_order_does_not_matter():
    b = _frame().iloc[[2, 0, 1]].reset_index(drop=True)
    out = fz.compare_rank_outputs(_frame(), [], b, [])
    assert out["identical"] is True


def test_compare_drops_diff_flagged_and_order_invariant():
    d1 = [{"ticker": "JPM", "gate": "event", "reason": "earnings"}]
    d2 = [
        {"ticker": "JPM", "gate": "event", "reason": "earnings"},
        {"ticker": "UNH", "gate": "data", "reason": "no OHLCV"},
    ]
    out = fz.compare_rank_outputs(_frame(), d1, _frame(), d2)
    assert out["identical"] is False
    assert out["drops_only_b"] == [("UNH", "data", "no OHLCV")]
    out2 = fz.compare_rank_outputs(_frame(), list(reversed(d2)), _frame(), d2)
    assert out2["identical"] is True


def test_compare_empty_frames_identical():
    out = fz.compare_rank_outputs(pd.DataFrame(), [], pd.DataFrame(), [])
    assert out["identical"] is True


# ---------------------------------------------------------------------------
# V2-b — snapshot differ
# ---------------------------------------------------------------------------


def _snap(mult=1.1, xi=0.21, labels=("crisis", "bear", "normal", "bull_quiet")) -> dict:
    return {
        "cutoff": "2023-06-30",
        "tickers": {
            "AAPL": {
                "hmm": {
                    "means": [[0.001], [0.0], [-0.001], [0.002]],
                    "stds": [[0.01], [0.02], [0.03], [0.005]],
                    "trans_mat": (np.eye(4) * 0.7 + 0.075).tolist(),
                    "state_labels": list(labels),
                    "converged": True,
                    "multiplier": mult,
                    "log_likelihood": float("nan"),
                },
                "gpd": {"shape_xi": xi, "scale_beta": 0.01, "converged": True},
            }
        },
    }


def test_diff_snapshots_identical():
    assert fz.diff_snapshots(_snap(), _snap())["identical"] is True


def test_diff_snapshots_nan_equals_nan():
    assert fz.diff_snapshots(_snap(), _snap())["identical"] is True  # log_likelihood NaN both sides


def test_diff_snapshots_flags_numeric_drift_beyond_rtol():
    out = fz.diff_snapshots(_snap(xi=0.21), _snap(xi=0.22))
    assert out["identical"] is False
    assert any("gpd.shape_xi" in p for p in out["mismatched"]["AAPL"])


def test_diff_snapshots_tolerates_within_rtol():
    out = fz.diff_snapshots(_snap(xi=0.21), _snap(xi=0.21 * (1 + 1e-9)))
    assert out["identical"] is True


def test_diff_snapshots_flags_label_change():
    out = fz.diff_snapshots(_snap(), _snap(labels=("crisis", "bear", "normal", "bull")))
    assert out["identical"] is False
    assert any("state_labels" in p for p in out["mismatched"]["AAPL"])


def test_diff_snapshots_flags_missing_ticker():
    a = _snap()
    b = _snap()
    b["tickers"]["MSFT"] = {"hmm": {"multiplier": 1.0}}
    out = fz.diff_snapshots(a, b)
    assert out["identical"] is False
    assert "MSFT" in out["mismatched"]


def test_snapshot_json_roundtrip_reproduces(tmp_path):
    # The fixture path: python json emits NaN tokens (allow_nan default) and
    # parses them back to float nan; the differ treats NaN == NaN as equal.
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(_snap(), indent=2))
    loaded = json.loads(p.read_text())
    assert fz.diff_snapshots(_snap(), loaded)["identical"] is True


# ---------------------------------------------------------------------------
# V2-c — freeze lever + offline recombination
# ---------------------------------------------------------------------------


def _synthetic_ohlcv(n_days: int = 700, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=n_days)))
    return pd.DataFrame({"close": close}, index=idx)


def test_forward_distribution_frozen_substitutes_cutoff_and_restores():
    import engine.forward_distribution as fdist

    ohlcv = _synthetic_ohlcv()
    cutoff = "2022-12-30"
    live_asof = str(ohlcv.index[-1].date())
    orig = fdist.best_available_forward_distribution
    ref_frozen, method_frozen = orig(ohlcv, horizon_days=35, as_of=cutoff)
    ref_live, _ = orig(ohlcv, horizon_days=35, as_of=live_asof)
    with fz.forward_distribution_frozen(cutoff):
        got, method_got = fdist.best_available_forward_distribution(
            ohlcv, horizon_days=35, as_of=live_asof
        )
    assert fdist.best_available_forward_distribution is orig  # restored
    assert method_got == method_frozen
    assert np.array_equal(np.asarray(got), np.asarray(ref_frozen))
    # And the freeze genuinely changed something vs live.
    assert not (
        len(ref_live) == len(ref_frozen)
        and np.array_equal(np.asarray(ref_live), np.asarray(ref_frozen))
    )


def test_forward_distribution_frozen_restores_on_exception():
    import engine.forward_distribution as fdist

    orig = fdist.best_available_forward_distribution
    with pytest.raises(RuntimeError), fz.forward_distribution_frozen("2022-12-30"):
        raise RuntimeError("boom")
    assert fdist.best_available_forward_distribution is orig


def test_build_frozen_tail_table_rejects_dates_at_or_before_cutoff():
    from datetime import date

    with pytest.raises(ValueError, match="freeze cutoff"):
        fz.build_frozen_tail_table(
            tickers=["AAPL"],
            sample_dates=[date(2023, 6, 30)],
            cutoff="2023-06-30",
        )


def test_apply_frozen_hmm_clamps_and_defaults():
    table = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-05", "2024-01-12"],
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "ev_raw": [100.0, 100.0, 100.0],
        }
    )
    mults = pd.DataFrame(
        {
            "date": ["2024-01-05", "2024-01-05"],
            "ticker": ["AAPL", "MSFT"],
            "frozen_hmm_multiplier": [0.5, 1.5],  # 1.5 must clamp to 1.25
            "frozen_hmm_regime": ["bear", "bull_quiet"],
        }
    )
    ev = fz.apply_frozen_hmm(table, mults)
    assert ev.tolist() == [50.0, 125.0, 100.0]  # missing pair -> neutral 1.0


def test_risk_rates_counts_breaches_and_informative_p25():
    table = pd.DataFrame(
        {
            "date": ["2024-01-05"] * 4,
            "ticker": list("ABCD"),
            "prob_profit": [0.5, 0.5, 0.9, 0.5],  # 0.9 excluded from p25 (>= 0.75)
            "pnl_p25": [-100.0, -100.0, -100.0, -100.0],
            "cvar_5": [-500.0, -500.0, -500.0, -500.0],
            "realized_pnl": [-600.0, 50.0, -650.0, -150.0],
        }
    )
    out = fz._risk_rates(table)
    assert out["cvar5_n"] == 4
    assert out["cvar5_breach_rate"] == pytest.approx(0.5)  # -600 and -650 breach
    assert out["p25_informative_n"] == 3
    assert out["p25_violation_rate"] == pytest.approx(2 / 3)  # -600, -150 violate


def _mini_tail_table(
    seed: int, dates: list[str], tickers=("A", "B", "C", "D", "E")
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "ev_raw": float(rng.normal(50, 25)),
                    "ev_dollars": float(rng.normal(40, 25)),
                    "prob_profit": float(rng.uniform(0.4, 0.7)),
                    "n_scenarios": 35.0,
                    "distribution_source": "empirical_non_overlapping",
                    "hmm_regime": "normal",
                    "iv": 0.3,
                    "premium": 2.5,
                    "strike": 100.0,
                    "spot": 105.0,
                    "pnl_p25": -50.0,
                    "pnl_p50": 50.0,
                    "pnl_p75": 150.0,
                    "cvar_5": float(-rng.uniform(500, 1500)),
                    "cvar_99_evt": float("nan"),
                    "tail_widening_factor": 1.0,
                    "vix_entry": 18.0,
                    "expiration_date": "2099-01-01",
                    "spot_at_expiry": 105.0,
                    "realized_pnl": float(rng.normal(100, 300)),
                }
            )
    return pd.DataFrame(rows)


def test_compare_frozen_vs_production_report_shape():
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-09-04", periods=8, freq="5B")]
    prod = _mini_tail_table(1, dates)
    froz = _mini_tail_table(2, dates)
    mults = pd.DataFrame(
        [
            {"date": d, "ticker": t, "frozen_hmm_multiplier": 1.1, "frozen_hmm_regime": "normal"}
            for d in dates
            for t in "ABCDE"
        ]
    )
    report = fz.compare_frozen_vs_production(prod, froz, mults, cutoff="2023-06-30", n_boot=50)
    assert report["n_common_dates"] == 8
    assert report["rows_joined"] == len(prod) == len(froz)
    assert set(report["rank"]) == {
        "production_ev_dollars",
        "production_ev_raw",
        "frozen_ev_dollars_live_hmm",
        "frozen_ev_raw",
        "frozen_ev_frozen_hmm",
    }
    for entry in report["rank"].values():
        assert set(entry) == {"all", "top5"}
        assert "mean_rho" in entry["all"]["xsec"]
        assert "ci95" in entry["all"]["ci_block7"]
    assert set(report["risk"]) == {"production", "frozen"}
    assert report["risk"]["frozen"]["cvar_5"]["verdict"] in {"PASS", "WARN", "FAIL", "INSUFFICIENT"}
    assert "paired" in report
    assert report["drift_by_months_since_cutoff"]  # at least one bucket populated
    assert set(report["distribution_source_mix"]) == {"production", "frozen"}


def test_months_since_cutoff_is_monotone():
    s = pd.Series(["2023-07-30", "2023-12-30", "2024-06-30", "2025-07-01"])
    m = fz._months_since(s, "2023-06-30")
    assert np.all(np.diff(m) > 0)
    assert m[0] == pytest.approx(1.0, abs=0.1)
    assert m[3] == pytest.approx(24.0, abs=0.5)


# ---------------------------------------------------------------------------
# V2-b — the C1 reproducibility lock (slow lane: needs the committed CSVs)
# ---------------------------------------------------------------------------

_LOCK_SUBSET = ("AAPL", "MSFT", "XOM")


@pytest.mark.backtest_regression
@pytest.mark.slow
@pytest.mark.skipif(not _FIXTURE.exists(), reason="freeze snapshot fixture not committed yet")
@pytest.mark.skipif(
    not (_DATA_DIR / "sp500_ohlcv.csv").exists(), reason="bloomberg CSVs not present"
)
def test_c1_lock_refit_reproduces_committed_snapshot():
    """C1's definition of done: the frozen parameters must reproduce the
    snapshot's classifier output when refit from the committed data."""
    snapshot = json.loads(_FIXTURE.read_text())
    out = fz.verify_freeze_snapshot(snapshot, _DATA_DIR, tickers=list(_LOCK_SUBSET))
    assert out["identical"] is True, f"freeze snapshot drift: {out['mismatched']}"
