"""Small-sample honesty for the Ollama trade memo (2026-06-01).

The audit found the trade memo presented ``prob_profit`` with no uncertainty
language. These tests pin the ADDITIVE fix: the memo renders ``prob_profit``
WITH its Wilson 95% interval + ``N`` (n_scenarios) and a one-line small-sample
caveat, sourced from the EV ranker row / ``EVResult`` fields
(``prob_profit``, ``prob_profit_ci_low``, ``prob_profit_ci_high``,
``n_scenarios``).

Scope (CLAUDE.md §2): these assert PRESENTATION only — no EV / verdict /
multiplier text is touched. The formatter is pure (no Ollama, no ranker call),
so the tests are fast and network-free.
"""

import math
from types import SimpleNamespace

from engine.trade_memo import MemoGenerator, _format_prob_profit_line


def _row(**overrides):
    base = {
        "prob_profit": 0.857,
        "prob_profit_ci_low": 0.711,
        "prob_profit_ci_high": 0.943,
        "n_scenarios": 35,
    }
    base.update(overrides)
    return base


def test_format_includes_interval_and_n():
    line = _format_prob_profit_line(_row())
    assert line is not None
    # point estimate, the bracketed interval, and N are all present
    assert "0.86" in line  # prob_profit rounded to 2dp
    assert "[0.71, 0.94]" in line
    assert "N=35" in line
    # small-sample caveat is present
    assert "Wilson" in line
    assert "small-sample" in line.lower()


def test_format_returns_none_when_no_prob_profit():
    assert _format_prob_profit_line(None) is None
    assert _format_prob_profit_line({}) is None
    assert _format_prob_profit_line({"prob_profit": None}) is None
    assert _format_prob_profit_line({"prob_profit": float("nan")}) is None


def test_format_handles_missing_ci_gracefully():
    # prob_profit present but CI/N not evaluated -> point estimate, no false caveat
    line = _format_prob_profit_line(
        {"prob_profit": 0.70, "prob_profit_ci_low": None, "prob_profit_ci_high": None}
    )
    assert line is not None
    assert "0.70" in line
    assert "[" not in line  # no bracket when no interval
    assert "Wilson" not in line  # no caveat without an interval/N to qualify


def test_format_handles_nan_ci():
    line = _format_prob_profit_line(
        {
            "prob_profit": 0.70,
            "prob_profit_ci_low": float("nan"),
            "prob_profit_ci_high": float("nan"),
            "n_scenarios": None,
        }
    )
    assert line is not None
    assert "0.70" in line
    assert "[" not in line


def _stub_analysis():
    return SimpleNamespace(
        ticker="AAPL",
        spot_price=200.0,
        sector="Technology",
        market_cap=3.0e12,
        pe_ratio=30.0,
        beta=1.2,
        credit_rating="AA+",
        iv_30d=25.0,
        rv_30d=20.0,
        iv_rank=40.0,
        vol_risk_premium=5.0,
        days_to_earnings=20,
        next_div_date=None,
        vix_level=18.0,
        risk_free_rate=0.045,
        wheel_score=72.0,
        wheel_recommendation="MODERATE",
    )


def test_data_package_surfaces_interval_when_row_present():
    gen = MemoGenerator()
    pkg = gen._build_data_package(_stub_analysis(), committee={}, strangle={}, ev_row=_row())
    assert "PROBABILITY OF PROFIT" in pkg
    assert "[0.71, 0.94]" in pkg
    assert "N=35" in pkg
    assert "Wilson" in pkg


def test_data_package_omits_block_when_no_row():
    gen = MemoGenerator()
    pkg_none = gen._build_data_package(_stub_analysis(), committee={}, strangle={}, ev_row=None)
    pkg_empty = gen._build_data_package(_stub_analysis(), committee={}, strangle={}, ev_row={})
    assert "PROBABILITY OF PROFIT" not in pkg_none
    assert "PROBABILITY OF PROFIT" not in pkg_empty


def test_data_package_does_not_alter_ev_or_verdict_text():
    # Additive-only guard: the prob-profit block must not introduce EV/verdict
    # wording — it only annotates probability precision.
    gen = MemoGenerator()
    pkg = gen._build_data_package(_stub_analysis(), committee={}, strangle={}, ev_row=_row())
    prob_block = pkg.split("--- PROBABILITY OF PROFIT ---", 1)[1]
    lowered = prob_block.lower()
    assert "ev_dollars" not in lowered
    assert "verdict" not in lowered
    assert "multiplier" not in lowered


def test_interval_math_is_just_passed_through_not_recomputed():
    # The memo must surface the row's CI verbatim (rounded), never recompute or
    # widen/narrow it. Feed an asymmetric interval and check it round-trips.
    line = _format_prob_profit_line(
        _row(prob_profit=0.50, prob_profit_ci_low=0.333, prob_profit_ci_high=0.667, n_scenarios=9)
    )
    assert "[0.33, 0.67]" in line
    assert "N=9" in line
    # prob_profit itself unchanged (0.50 -> "0.50")
    assert "0.50" in line
    assert not math.isnan(0.50)
