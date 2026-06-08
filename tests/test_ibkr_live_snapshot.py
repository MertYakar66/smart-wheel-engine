"""Tests for the live-connector snapshot builder (scripts/ibkr_live_snapshot.py).

Real account data is gitignored, so these exercise the pure transform on
synthetic IBKR endpoint payloads: contract_description parsing (incl. the
interposed-venue-token case), FX normalization, the closed-position exclusion,
day-change derivation, in-universe / sector joins, and the schema_version: 1
contract the adapter consumes. Also guards the §2/§3 scope: the module imports
nothing from the decision trio.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ibkr_live_snapshot", _REPO / "scripts" / "ibkr_live_snapshot.py"
)
live = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(live)

# Minimal reference: AMD in-universe; CLS/ENB out-of-universe (use overrides).
_REF = {"AMD": {"name": "Advanced Micro Devices", "sector": "Information Technology"}}


# --------------------------------------------------------- parsing
def test_parse_option_standard():
    assert live.parse_option_description("MU Jun12'26 970 PUT @AMEX") == (
        "MU",
        "P",
        970.0,
        "2026-06-12",
    )


def test_parse_option_with_venue_token():
    # Underlying followed by an interposed venue token before the date.
    assert live.parse_option_description("ENB TSE Oct16'26 86 CALL @CDE") == (
        "ENB",
        "C",
        86.0,
        "2026-10-16",
    )


def test_parse_option_fractional_strike():
    sym, right, strike, expiry = live.parse_option_description("AMD Jun12'26 532.5 CALL @AMEX")
    assert (sym, right, strike, expiry) == ("AMD", "C", 532.5, "2026-06-12")


def test_parse_stock_symbol():
    assert live.parse_stock_symbol("CNQ @TSE") == "CNQ"
    assert live.parse_stock_symbol("AMD") == "AMD"


def test_option_label_matches_imported_style():
    assert live._option_label("MU", "2026-06-12", 970.0, "P") == "MU 12JUN26 970 P"


# --------------------------------------------------------- fx + helpers
def test_fx_rates_skips_base_and_keeps_currencies():
    balances = {
        "balances": [
            {"currency": "BASE", "exchange_rate": 1},
            {"currency": "CAD", "exchange_rate": 0.7166773},
            {"currency": "USD", "exchange_rate": 1},
        ]
    }
    fx = live._fx_rates(balances)
    assert fx["USD"] == 1.0
    assert abs(fx["CAD"] - 0.7166773) < 1e-9
    assert "BASE" not in fx


def test_base_unrealized_reads_base_row():
    balances = {
        "balances": [
            {"currency": "BASE", "unrealized_pnl": -22151.74},
            {"currency": "USD", "unrealized_pnl": -22049.52},
        ]
    }
    assert live._base_unrealized(balances) == -22151.74


# --------------------------------------------------------- build_snapshot
def _inputs():
    summary = {
        "net_liquidation": 100000.0,
        "total_cash": None,  # absent on this fixture
        "total_cash_value": -5000.0,
        "available_funds": 322.24,
        "excess_liquidity": 15232.33,
        "maintenance_margin": 158233.67,
        "buying_power": 1074.13,
    }
    balances = {
        "balances": [
            {"currency": "BASE", "unrealized_pnl": -2200.0, "exchange_rate": 1},
            {"currency": "CAD", "exchange_rate": 0.5},
        ]
    }
    positions = {
        "positions": [
            {  # in-universe stock
                "contract_description": "AMD",
                "asset_class": "STK",
                "position": 100,
                "market_price": 488.1,
                "average_price": 496.9,
                "unrealized_pnl": -880.0,
                "daily_pnl": 2000.0,
                "currency": "USD",
            },
            {  # out-of-universe option in CAD (override sector + FX day-pnl)
                "contract_description": "ENB TSE Oct16'26 86 CALL @CDE",
                "asset_class": "OPT",
                "position": -1,
                "market_price": 0.74,
                "average_price": 0.835,
                "unrealized_pnl": 9.85,
                "daily_pnl": 100.0,  # CAD → *0.5 = 50 USD
                "currency": "CAD",
            },
            {  # closed today — must be excluded from positions, kept in day-pnl
                "contract_description": "TSM",
                "asset_class": "STK",
                "position": 0,
                "market_price": 426.0,
                "average_price": 0.0,
                "unrealized_pnl": 0.0,
                "daily_pnl": 500.0,
                "currency": "USD",
            },
        ]
    }
    return summary, balances, positions


def test_build_snapshot_schema_and_account():
    summary, balances, positions = _inputs()
    snap = live.build_snapshot(
        summary, balances, positions, reference=_REF, as_of="2026-06-08T00:00:00Z"
    )
    assert snap["schema_version"] == 1
    assert snap["base_currency"] == "USD"
    # source must NOT read as a fixture/demo marker → live
    assert snap["source"] not in ("fixture", "demo", "mock")

    acct = snap["account"]
    assert acct["net_liquidation"] == 100000.0
    assert acct["total_cash"] == -5000.0
    assert acct["available_funds"] == 322.24
    assert acct["excess_liquidity"] == 15232.33
    assert acct["maintenance_margin"] == 158233.67
    assert acct["unrealized_pnl"] == -2200.0  # from BASE row
    # ledger/accumulation fields stay null on a pure snapshot
    assert acct["realized_pnl_ytd"] is None
    assert acct["week_change_usd"] is None
    assert acct["week_change_pct"] is None


def test_build_snapshot_day_change_fx_normalized_includes_closed():
    summary, balances, positions = _inputs()
    snap = live.build_snapshot(summary, balances, positions, reference=_REF)
    # 2000 (AMD USD) + 100*0.5 (ENB CAD) + 500 (TSM closed) = 2550
    assert snap["account"]["day_change_usd"] == 2550.0
    prior = 100000.0 - 2550.0
    assert snap["account"]["day_change_pct"] == round(2550.0 / prior, 4)


def test_build_snapshot_excludes_zero_qty_positions():
    summary, balances, positions = _inputs()
    snap = live.build_snapshot(summary, balances, positions, reference=_REF)
    syms = [p["symbol"] for p in snap["positions"]]
    assert "TSM" not in syms  # closed today
    assert syms == ["AMD", "ENB"]


def test_build_snapshot_option_vs_stock_fields_and_joins():
    summary, balances, positions = _inputs()
    snap = live.build_snapshot(summary, balances, positions, reference=_REF)
    by = {p["symbol"]: p for p in snap["positions"]}

    amd = by["AMD"]
    assert amd["sec_type"] == "STK"
    assert "strike" not in amd and "right" not in amd
    assert amd["in_universe"] is True
    assert amd["sector"] == "Information Technology"
    assert amd["name"] == "Advanced Micro Devices"

    enb = by["ENB"]
    assert enb["sec_type"] == "OPT"
    assert enb["right"] == "C" and enb["strike"] == 86.0 and enb["expiry"] == "2026-10-16"
    assert enb["in_universe"] is False
    assert enb["sector"] == "Energy"  # SECTOR_OVERRIDE fallback
    assert enb["name"] == "ENB 16OCT26 86 C"


def test_fx_rates_present_on_snapshot():
    summary, balances, positions = _inputs()
    snap = live.build_snapshot(summary, balances, positions, reference=_REF)
    assert snap["fx_rates"]["USD"] == 1.0
    assert snap["fx_rates"]["CAD"] == 0.5


# --------------------------------------------------------- §2/§3 scope guard
def test_module_imports_nothing_from_the_trio():
    # Scan import statements only — the module docstring legitimately *names*
    # the trio when stating the scope contract.
    src = (_REPO / "scripts" / "ibkr_live_snapshot.py").read_text(encoding="utf-8")
    import_lines = "\n".join(
        ln for ln in src.splitlines() if ln.strip().startswith(("import ", "from "))
    )
    for forbidden in ("ev_engine", "wheel_runner", "candidate_dossier"):
        assert forbidden not in import_lines, f"live snapshot must not import {forbidden} (sec 2)"
