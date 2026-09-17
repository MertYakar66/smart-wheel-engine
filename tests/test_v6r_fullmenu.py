"""Locks for the V6-r1 full-menu driver (scripts/run_v6r_fullmenu.py, plan §10.1).

Same discipline as test_v6_lockbox.py: the deep re-read runs once on the
operator terminal, so the FM-verdict math is pinned on synthetic frames
BEFORE the run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "run_v6r_fullmenu", Path(__file__).resolve().parent.parent / "scripts" / "run_v6r_fullmenu.py"
)
v6r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v6r)


def _rank_log(baseline_pos=0.6, grind_pos=0.6, n_per_date=100):
    rows = []
    for mth, rate in (
        ("2007-03", baseline_pos),
        ("2007-09", baseline_pos),
        ("2008-10", grind_pos),
        ("2008-11", grind_pos),
    ):
        for i in range(n_per_date):
            rows.append(
                {
                    "date": f"{mth}-15",
                    "ticker": f"T{i}",
                    "ev_dollars": 1.0 if i < rate * n_per_date else -1.0,
                }
            )
    return pd.DataFrame(rows)


def test_fm1_caveat_retired_when_grind_stays_high():
    out = v6r.fm1_full_menu_rate(_rank_log())
    assert out["verdict"] == "CAVEAT_RETIRED"
    assert out["ratio"] == 1.0


def test_fm1_censoring_load_bearing_at_the_half_cut():
    # ratio exactly 0.5 -> the caveat WAS load-bearing (<= cut, matching §10.1).
    out = v6r.fm1_full_menu_rate(_rank_log(baseline_pos=0.6, grind_pos=0.3))
    assert out["ratio"] == 0.5
    assert out["verdict"] == "CENSORING_LOAD_BEARING"
    assert v6r.fm1_full_menu_rate(_rank_log().query("date < '2008'"))["verdict"] == "INSUFFICIENT"


def test_fm2_depth_profile_counts_appetite_lines():
    # One grind date with only 2 EV-positive names (below both lines).
    df = _rank_log()
    thin = pd.DataFrame(
        [
            {"date": "2008-12-15", "ticker": f"T{i}", "ev_dollars": 1.0 if i < 2 else -1.0}
            for i in range(100)
        ]
    )
    out = v6r.fm2_depth_profile(pd.concat([df, thin], ignore_index=True))
    assert out["n_dates"] == 5
    assert out["n_dates_below_saturation_guard"] == 1
    assert out["n_dates_below_opens_appetite"] == 1
    assert out["min_ev_pos_per_date"] == 2
    assert out["monthly_min_median"]["2008-12"]["min"] == 2
