"""Tests for the IBKR PortfolioAnalyst PDF importer (scripts/ibkr_import.py)
and the null-safety the importer relies on in engine.ibkr_portfolio_adapter.

The real PDF is gitignored (operator account data), so these tests exercise
the pure parsers/helpers on synthetic page text + pin the adapter's
present-but-null robustness (the bug a real/PDF import exposes).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("ibkr_import", _REPO / "scripts" / "ibkr_import.py")
imp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(imp)


# ----------------------------------------------------------- OCC parsing
def test_parse_occ_put():
    assert imp.parse_occ("MU    260612P00970000") == ("MU", "2026-06-12", "P", 970.0)


def test_parse_occ_call_fractional_strike():
    assert imp.parse_occ("AAPL  250606C00207500") == ("AAPL", "2025-06-06", "C", 207.5)


def test_parse_occ_rejects_plain_ticker():
    assert imp.parse_occ("NVDA") is None
    assert imp.parse_occ("USD.CAD") is None


def test_f_strips_commas():
    assert imp.f("-12,122.25") == pytest.approx(-12122.25)


# ------------------------------------------------ open-position parser (p6)
def test_parse_open_positions_synthetic():
    # Minimal p6 with one USD stock, one USD short put, one CAD stock + the
    # short-cash line — exercises FX derivation, cash capture, dedup.
    page6 = [
        "SHORT POSITIONS AS OF 06/05/2026",
        "USD",
        "-165,014.92",
        "Stocks (USD)",
        "CLS",
        "CELESTICA INC",
        "Technology",
        "500",
        "371.71",
        "185,855.00",
        "209,825.66",
        "-23,970.66",
        "185,855.00",
        "Options (USD)",
        "MU    260612P00970000",
        "MU 12JUN26 970 P",
        "Technology",
        "-1",
        "121.22",
        "-12,122.25",
        "-3,568.87",
        "-8,553.38",
        "-12,122.25",
        "Stocks (CAD)",
        "CNQ",
        "CANADIAN NATURAL RESOURCES",
        "Energy",
        "100",
        "63.76",
        "6,376.00",
        "6,526.00",
        "-150.00",
        "4,573.70",
    ]
    rows, fx_cad, cash = imp.parse_open_positions({6: page6})
    assert cash == pytest.approx(-165014.92)
    assert fx_cad == pytest.approx(0.71733, abs=1e-4)  # 4573.70 / 6376.00
    by_sym = {r["symbol"]: r for r in rows}
    assert by_sym["CLS"]["qty"] == 500.0
    assert by_sym["CLS"]["unrl_nat"] == pytest.approx(-23970.66)
    assert by_sym["MU    260612P00970000"]["ccy"] == "USD"
    assert by_sym["CNQ"]["ccy"] == "CAD"


# ------------------------------------------------ dividends parser (p87)
def test_parse_dividends_dedups_and_nets():
    # Same two rows listed twice (the hidden duplicate layer) + a negative PIL.
    one = ["06/02/25", "05/15/25", "ENB", "Dividend Payment", "5", "0.69", "3.43"]
    pil = ["12/23/25", "12/15/25", "META", "Payment In Lieu", "-200", "0.53", "-105.00"]
    page = one + one + pil + pil  # duplicated
    rows = imp.parse_dividends({87: page})
    assert len(rows) == 2  # deduped
    assert sum(r["amount"] for r in rows) == pytest.approx(3.43 - 105.00)


# ----------------------------------- adapter null-safety (the import-exposed bug)
def _null_field_snapshot():
    return {
        "schema_version": 1,
        "as_of": "2026-06-05T21:36:00Z",
        "base_currency": "USD",
        "fx_rates": {"USD": 1.0, "CAD": 0.7173},
        "source": "ibkr_import",
        "account": {
            "net_liquidation": 143115.0,
            "total_cash": -165014.92,
            # the fields a performance-report import cannot derive -> null
            "available_funds": None,
            "excess_liquidity": None,
            "maintenance_margin": None,
            "unrealized_pnl": -41597.0,
            "realized_pnl_ytd": 88234.0,
            "day_change_usd": None,
            "day_change_pct": None,
            "week_change_usd": None,
            "week_change_pct": None,
        },
        "positions": [
            {
                "symbol": "CLS",
                "name": "Celestica",
                "sec_type": "STK",
                "qty": 500,
                "mark": 371.71,
                "avg_price": 419.65,
                "unrealized_pnl": -23970.66,
                "currency": "USD",
                "sector": "Technology",
                "in_universe": False,
            },
            {
                "symbol": "MU",
                "name": "Micron",
                "sec_type": "OPT",
                "right": "P",
                "strike": 970.0,
                "expiry": "2026-06-12",
                "qty": -1,
                "mark": 121.22,
                "avg_price": 35.69,
                "unrealized_pnl": -8553.38,
                "currency": "USD",
                "sector": "Technology",
                "in_universe": True,
            },
        ],
    }


def _history():
    return {
        "schema_version": 1,
        "inception_capital": 100000.0,
        "points": [
            {
                "label": "Apr '25",
                "date": "2025-04-30",
                "port": 100000.0,
                "spy": 100000.0,
                "premium": 0.0,
            },
            {
                "label": "May",
                "date": "2025-05-30",
                "port": 121000.0,
                "spy": 106000.0,
                "premium": 5000.0,
            },
        ],
    }


def test_adapter_returns_view_null_safe():
    """returns_view must not crash on present-but-null day/week deltas."""
    from engine import ibkr_portfolio_adapter as ad

    out = ad.returns_view(_history(), _null_field_snapshot())
    assert out["returns"]["1D"] == {"pct": 0.0, "usd": 0}
    assert out["returns"]["1W"] == {"pct": 0.0, "usd": 0}


def test_adapter_risk_view_null_safe_margin():
    """risk_view must not crash on null margin fields and reports them as 0."""
    from engine import ibkr_portfolio_adapter as ad

    out = ad.risk_view(_null_field_snapshot())
    assert out["margin"]["maintMargin"] == 0
    assert out["margin"]["cushionPct"] == 0.0
    assert out["margin"]["stressed"] is False
    # CLS at 500*371.71 = 185,855 on 143,115 NAV -> ~130% single-name breach.
    cls = next(s for s in out["singleName"] if s["sym"] == "CLS")
    assert cls["pct"] > 100
