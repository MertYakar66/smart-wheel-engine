"""Locks for the V6 lockbox driver's H-metric computations (scripts/run_v6_lockbox.py).

The lockbox run happens exactly once (plan doc section 9), so the verdict
math is pinned here on synthetic frames BEFORE the spend — an H-metric bug
discovered after the read cannot be fixed by re-running the read.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_v6_lockbox", Path(__file__).resolve().parent.parent / "scripts" / "run_v6_lockbox.py"
)
v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6)


def _rank_log(baseline_pos=0.4, grind_pos=0.1):
    rows = []
    for mth, rate in (
        ("2007-03", baseline_pos),
        ("2007-09", baseline_pos),
        ("2008-10", grind_pos),
        ("2008-11", grind_pos),
    ):
        for i in range(100):
            rows.append(
                {
                    "date": f"{mth}-15",
                    "ticker": f"T{i}",
                    "ev_dollars": 1.0 if i < rate * 100 else -1.0,
                    "premium": 2.0,
                    "expiration_date": f"{mth}-28",
                    "realized_pnl": 200.0 if i % 2 else -500.0,  # ITM iff < 200
                }
            )
    return pd.DataFrame(rows)


def test_h1_pass_at_quarter_ratio_and_fail_when_flat():
    h1 = v6.h1_refusal_grind(_rank_log())
    assert h1["verdict"] == "PASS"
    assert h1["ratio"] == 0.25
    flat = v6.h1_refusal_grind(_rank_log(grind_pos=0.4))
    assert flat["verdict"] == "FAIL" and flat["ratio"] == 1.0


def test_h2_blind_vs_good_news():
    aug = [f"2008-08-{d:02d}" for d in range(1, 21)]
    busy_cliff = [
        "2008-09-02",
        "2008-09-03",
        "2008-09-04",
        "2008-09-05",
        "2008-09-08",
        "2008-09-09",
        "2008-09-10",
        "2008-09-11",
        "2008-09-12",
    ]
    assert v6.h2_cliff_lag(pd.Series(aug + busy_cliff))["verdict"] == "CONFIRMED_BLIND"
    assert v6.h2_cliff_lag(pd.Series(aug + ["2008-09-02"]))["verdict"] == "FALSIFIED_GOOD_NEWS"


def test_h3_itm_from_forward_replay_identity():
    # realized 200.0 == premium x 100 -> OTM (strict <); -500 -> ITM. Half ITM.
    h3 = v6.h3_assignment_wave(_rank_log(grind_pos=1.0))
    assert h3["verdict"] == "PASS"
    assert h3["peak_monthly_itm_rate"] == 0.5
    # All-OTM wave months -> FAIL.
    df = _rank_log(grind_pos=1.0)
    df.loc[df["expiration_date"].str.startswith("2008"), "realized_pnl"] = 200.0
    assert v6.h3_assignment_wave(df)["verdict"] == "FAIL"


def test_h_metrics_insufficient_paths():
    empty = pd.DataFrame(
        columns=["date", "ticker", "ev_dollars", "premium", "expiration_date", "realized_pnl"]
    )
    assert v6.h3_assignment_wave(empty)["verdict"] == "INSUFFICIENT"
    only_2007 = _rank_log().query("date < '2008'")
    assert v6.h1_refusal_grind(only_2007)["verdict"] == "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Post-processing regressions — the 2026-07-13 attempt-1/2 crash class.
# The engine pass completed and the driver died AFTER it, in shape handling;
# these pin the reporting path against the vehicle's EXACT return shape.
# ---------------------------------------------------------------------------


def test_collect_entry_dates_tolerates_every_vehicle_shape():
    closed = [
        {"ticker": "AAPL", "entry_date": "2008-08-05", "realized_pnl": 1.0},
        {"ticker": "XOM", "entry_date": "2008-09-10"},
    ]
    open_records = [{"ticker": "JPM", "state": "short_put", "entry_date": "2008-09-11"}]
    legacy_state_map = {"JPM": "short_put", "GE": "stock_owned"}  # the crash shape

    got = v6.collect_entry_dates(closed, open_records)
    assert list(got) == ["2008-08-05", "2008-09-10", "2008-09-11"]
    # The legacy {ticker: state_string} map must be skipped, never die.
    assert list(v6.collect_entry_dates(closed, legacy_state_map)) == [
        "2008-08-05",
        "2008-09-10",
    ]
    assert list(v6.collect_entry_dates(None, pd.DataFrame(closed))) == [
        "2008-08-05",
        "2008-09-10",
    ]
    assert len(v6.collect_entry_dates(None, [])) == 0


def test_safe_records_error_verdict_instead_of_dying():
    def boom(_):
        raise KeyError("was_assigned")

    out = v6._safe(boom, pd.DataFrame())
    assert out["verdict"] == "ERROR"
    assert "was_assigned" in out["traceback"]


def test_build_report_on_the_vehicle_exact_return_shape():
    """The lock that would have caught the spend crash: a result dict shaped
    byte-for-byte like run_survivorship_backtest's return (legacy state map
    AND records) must post-process into a complete report."""
    from datetime import date

    result = {
        "metrics": {"final_nav": 1_000_000.0},
        "rank_log": _rank_log(),
        "open_positions": {"JPM": "short_put", "GE": "stock_owned"},
        "open_position_records": [
            {"ticker": "JPM", "state": "short_put", "entry_date": "2008-09-11"},
            {"ticker": "GE", "state": "stock_owned", "entry_date": "2008-10-02"},
        ],
        "closed_positions": [
            {"ticker": "AAPL", "entry_date": date(2008, 8, 5), "exit_date": date(2008, 9, 20)}
        ],
    }
    report = v6.build_report(result, [], elapsed_seconds=61.0)
    for h in ("H1_refusal_grind", "H2_cliff_lag", "H3_assignment_wave", "H4_rank_rho"):
        assert report[h].get("verdict") != "ERROR", report[h]
    assert report["n_opens_counted"] == 3  # closed + BOTH still-open records
    assert report["rows_ranked"] == len(result["rank_log"])
    assert report["H2_cliff_lag"]["verdict"] != "INSUFFICIENT"
