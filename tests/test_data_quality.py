"""Tests for data/quality.py — specifically the options-consistency gate.

The gate is exercised live in wheel_runner.py's chain quality check.
Pin its behaviour here so a future column rename or column addition in
the chain pipeline doesn't silently re-trigger the substring-match bug.
"""

from __future__ import annotations

import pandas as pd

from data.quality import DataQualityFramework


def test_quality_gate_does_not_match_iv_error():
    """Regression pin for the substring-match bug at data/quality.py:549.

    Theta's v3 `option/snapshot/greeks/first_order` returns an `iv_error`
    column (the IV solver's diagnostic — `100.0` is the "couldn't compute"
    sentinel for deep-OTM near-worthless contracts). The original gate
    matched any column containing the substring 'iv', so `iv_error` looked
    like an IV column and 100.0 tripped the > 5.0 ERROR branch — blocking
    every Theta-sourced ticker.

    After the fix, the gate only matches exact-name IV columns
    ({"iv", "implied_vol", "implied_volatility", "impvol"}).
    """
    df = pd.DataFrame({
        "strike": [100.0, 200.0],
        "right": ["put", "call"],
        "iv": [0.25, 0.30],           # clean decimal IV
        "iv_error": [0.0, 100.0],     # Theta diagnostic — 100 is "no solution"
        "bid": [1.00, 0.50],
        "ask": [1.05, 0.55],
        "expiration": pd.to_datetime(["2026-12-31", "2026-12-31"]),
        "date": pd.to_datetime(["2026-05-15", "2026-05-15"]),
    })

    issues = DataQualityFramework()._check_options_consistency(df)
    iv_range_issues = [i for i in issues if "IV values outside valid range" in i.message]

    assert iv_range_issues == [], (
        "Quality gate flagged a spurious IV-range issue. "
        f"Offenders: {[(i.column, i.message) for i in iv_range_issues]}"
    )


def test_quality_gate_still_catches_real_invalid_iv():
    """Counterpart to the above — the gate must still catch genuine
    invalid IV (true `iv` column with > 5.0 or negative values).
    """
    df = pd.DataFrame({
        "strike": [100.0, 200.0],
        "right": ["put", "call"],
        "iv": [0.25, 7.5],            # second row is corrupted
        "bid": [1.00, 0.50],
        "ask": [1.05, 0.55],
    })

    issues = DataQualityFramework()._check_options_consistency(df)
    iv_range_issues = [i for i in issues if "IV values outside valid range" in i.message]

    assert len(iv_range_issues) == 1
    assert iv_range_issues[0].column == "iv"
    assert iv_range_issues[0].affected_rows == 1


def test_quality_gate_matches_implied_vol_alias():
    """The fixed allowlist must cover `implied_vol` and `implied_volatility`
    — common alternative IV column names. Otherwise a chain that uses
    those names directly would bypass the gate.
    """
    df = pd.DataFrame({
        "strike": [100.0],
        "right": ["put"],
        "implied_vol": [12.0],         # corrupted, should be flagged
        "bid": [1.00],
        "ask": [1.05],
    })

    issues = DataQualityFramework()._check_options_consistency(df)
    iv_range_issues = [i for i in issues if "IV values outside valid range" in i.message]

    assert len(iv_range_issues) == 1
    assert iv_range_issues[0].column == "implied_vol"
