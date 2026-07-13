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
