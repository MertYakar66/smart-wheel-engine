"""Deposit-aware Portfolio Value history: TWR returns + NAV-curve safety.

Guards the fix for the equity curve being deposit-blind: a live IBKR account
takes deposits/withdrawals, so raw NAV deltas overstate performance (on the real
account ~+$37k net deposits turn a true +3.4% TWR into a fake +44.6% NAV
return). When the history carries a ``twr_returns`` block (from PortfolioAnalyst)
the adapter must report those time-weighted numbers verbatim, null the windows
it lacks, null the raw-NAV Sharpe/Sortino/MaxDD (deposit-distorted), and never
crash on a null ``spy`` benchmark. Legacy (no-TWR) histories keep the old
NAV-delta path — asserted here so the change is backward compatible.
"""

from engine.ibkr_portfolio_adapter import equity_view, returns_view

_TWR_HISTORY = {
    "schema_version": 1,
    "inception_capital": 90005.55,
    "portfolio_measure": "TWR",
    "twr_returns": {
        "YTD": {"pct": 0.08111392, "start_nav": 144280.47},
        "1Y": {"pct": 0.03429991, "start_nav": 90005.55},
        "All": {"pct": 0.03429991, "start_nav": 90005.55},
        # 1M/1W/1D/3M intentionally absent → must come back null
    },
    "points": [
        {"label": "Jul '25", "date": "2025-07-16", "port": 90005.55, "spy": None, "premium": None},
        {"label": "Mar '26", "date": "2026-03-31", "port": 203380.50, "spy": None, "premium": None},
        {"label": "Jul '26", "date": "2026-07-17", "port": 129885.37, "spy": None, "premium": None},
    ],
}

_LEGACY_HISTORY = {
    "schema_version": 1,
    "inception_capital": 100000.0,
    "points": [
        {
            "label": "Jan",
            "date": "2026-01-31",
            "port": 100000.0,
            "spy": 100000.0,
            "premium": 1000.0,
        },
        {
            "label": "Jul",
            "date": "2026-07-17",
            "port": 110000.0,
            "spy": 105000.0,
            "premium": 1200.0,
        },
    ],
}


def test_returns_use_twr_not_nav_delta():
    r = returns_view(_TWR_HISTORY)["returns"]
    # TWR verbatim, NOT (last_nav/inception - 1) which would be +44%
    assert abs(r["YTD"]["pct"] - 0.08111392) < 1e-9
    assert r["YTD"]["usd"] == round(0.08111392 * 144280.47)  # pct x window opening capital
    assert abs(r["1Y"]["pct"] - 0.03429991) < 1e-9
    assert r["All"]["usd"] == round(0.03429991 * 90005.55)
    # windows the source doesn't provide are null (UI renders "—"), never a
    # deposit-inflated NAV delta
    assert r["1M"] == {"pct": None, "usd": None}
    assert r["3M"] == {"pct": None, "usd": None}


def test_twr_history_nulls_deposit_distorted_stats_and_survives_null_spy():
    ev = equity_view(_TWR_HISTORY)
    # raw-NAV Sharpe/Sortino/MaxDD are invalid with cash flows → null (UI hides)
    assert ev["stats"] is None
    # null spy must not crash; it renders as a gap, never a fabricated line
    assert [p["spy"] for p in ev["equity"]] == [None, None, None]
    assert [p["port"] for p in ev["equity"]] == [90006, 203380, 129885]


def test_legacy_history_still_uses_nav_delta_and_computes_stats():
    r = returns_view(_LEGACY_HISTORY)["returns"]
    # no twr_returns → All return is the NAV delta (110k/100k - 1 = +10%)
    assert abs(r["All"]["pct"] - 0.10) < 1e-9
    assert r["All"]["usd"] == 10000
    ev = equity_view(_LEGACY_HISTORY)
    assert ev["stats"] is not None  # stats still computed for a cash-flow-free history
    assert ev["equity"][0]["spy"] == 100000
