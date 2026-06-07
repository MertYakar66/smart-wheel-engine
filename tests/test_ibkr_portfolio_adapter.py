"""Tests for the D24/D26 read-only IBKR performance-viewer adapter.

Covers snapshot→engine-type fidelity, the universe filter, FX normalization
for the USD+CAD book, that the D17 R9/R10 gates fire on the adapter-built
context, that the viewer payloads reproduce the approved numbers, and — the
load-bearing §2/§3 guards — that the adapter is observational only: it
imports nothing from the decision trio and never emits a tradeable verdict /
EV-authority token.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from engine import ibkr_portfolio_adapter as adapter
from engine.portfolio_risk_gates import check_sector_cap, check_single_name_cap

FIXTURES = Path(__file__).parent / "fixtures" / "ibkr"


@pytest.fixture
def snapshot():
    return adapter.load_snapshot(FIXTURES)


@pytest.fixture
def history():
    return adapter.load_history(FIXTURES)


@pytest.fixture
def ledger():
    return adapter.load_ledger(FIXTURES)


# ----------------------------------------------------------------------
# Snapshot → PortfolioContext fidelity
# ----------------------------------------------------------------------
def test_portfolio_context_nav_and_shapes(snapshot):
    ctx = adapter.build_portfolio_context(snapshot)
    assert ctx.nav == pytest.approx(144507.08)

    # 4 short puts (CSP) + 4 short calls (the covered-call legs) = 8 option legs.
    assert len(ctx.held_option_positions) == 8
    required = {"symbol", "option_type", "strike", "dte", "iv", "contracts", "is_short"}
    for pos in ctx.held_option_positions:
        assert required <= set(pos)
        assert pos["is_short"] is True  # every held option in the book is short

    cls = next(p for p in ctx.held_option_positions if p["symbol"] == "CLS")
    assert cls["option_type"] == "put"
    assert cls["contracts"] == 5  # |qty| = 5, not the raw -5
    assert cls["strike"] == 425.0

    # 5 stock holdings (TSM/NVDA/WMT covered-call stock + ENB/CNQ).
    syms = {s for s, _ in ctx.stock_holdings}
    assert syms == {"TSM", "NVDA", "WMT", "ENB", "CNQ"}
    assert all(shares == 100 for _, shares in ctx.stock_holdings)


def test_context_uses_gate_shape_not_ticker_keyed(snapshot):
    # Regression vs engine_api._build_portfolio_context_from_params which keys
    # 'ticker' and omits 'is_short' — that shape zeroes the single-name
    # aggregation. The adapter must emit 'symbol' + 'is_short'.
    ctx = adapter.build_portfolio_context(snapshot)
    for pos in ctx.held_option_positions:
        assert "ticker" not in pos
        assert "symbol" in pos


# ----------------------------------------------------------------------
# Universe filter — out-of-universe = exposure-only, never rankable
# ----------------------------------------------------------------------
def test_universe_filter_excludes_tsx_names(snapshot):
    rankable = adapter.rankable_symbols(snapshot)
    assert "CNQ" not in rankable
    assert "ENB" not in rankable
    assert "CLS" in rankable and "MU" in rankable


def test_out_of_universe_still_counts_as_exposure(snapshot):
    # CNQ/ENB are exposure-only but MUST still consume NAV / appear in the
    # held book (real risk) — design-doc §2.3.
    ctx = adapter.build_portfolio_context(snapshot)
    stock_syms = {s for s, _ in ctx.stock_holdings}
    assert {"ENB", "CNQ"} <= stock_syms
    holdings = {h["sym"] for h in adapter.build_holdings_view(snapshot)}
    assert {"ENB", "CNQ"} <= holdings


# ----------------------------------------------------------------------
# FX normalization (USD + CAD)
# ----------------------------------------------------------------------
def test_cad_positions_normalized_to_usd(snapshot):
    holdings = {h["sym"]: h for h in adapter.build_holdings_view(snapshot)}
    enb, cnq = holdings["ENB"], holdings["CNQ"]
    assert enb["currency"] == "CAD" and cnq["currency"] == "CAD"
    # native CAD mark 77.18 * 0.73 ≈ 56.34 USD; 100 shares ≈ $5,634 USD.
    assert enb["mark"] == pytest.approx(56.34, abs=0.05)
    assert enb["mktValue"] == pytest.approx(5634, abs=2)
    assert cnq["mktValue"] == pytest.approx(4580, abs=2)
    # uPnl FX-normalized too: native -27.4 CAD -> ≈ -20 USD.
    assert enb["uPnl"] == pytest.approx(-20, abs=1)


def test_usd_positions_unscaled(snapshot):
    cls = next(h for h in adapter.build_holdings_view(snapshot) if h["sym"] == "CLS")
    assert cls["mark"] == pytest.approx(53.0)
    assert cls["mktValue"] == pytest.approx(-26500, abs=1)


# ----------------------------------------------------------------------
# Holdings view fidelity vs the approved mock
# ----------------------------------------------------------------------
def test_holdings_states_and_concentration(snapshot):
    h = {row["sym"]: row for row in adapter.build_holdings_view(snapshot)}
    assert h["CLS"]["state"] == "csp" and h["CLS"]["pctNav"] == 147 and h["CLS"]["breach"]
    assert h["MU"]["state"] == "csp" and h["MU"]["pctNav"] == 67 and h["MU"]["breach"]
    assert h["TSM"]["state"] == "cc" and h["TSM"]["breach"] is False
    assert h["CNQ"]["state"] == "assigned"
    # Covered-call stock is never flagged a single-name (CSP) breach even
    # above 10% NAV.
    assert h["TSM"]["pctNav"] == 29 and not h["TSM"]["breach"]


# ----------------------------------------------------------------------
# The D24 payoff: R9/R10 actually fire on the adapter-built context
# ----------------------------------------------------------------------
def test_single_name_gate_fires(snapshot):
    ctx = adapter.build_portfolio_context(snapshot)
    res = check_single_name_cap("CLS", 0.0, ctx.held_option_positions, ctx.nav)
    assert res.passed is False
    assert res.reason == "single_name_breach"
    assert res.details["post_open_name_pct"] == pytest.approx(1.47, abs=0.01)


def test_sector_gate_fires(snapshot):
    ctx = adapter.build_portfolio_context(snapshot)
    # The dominant semis book is grossly over the 25% sector cap.
    res = check_sector_cap("MU", 0.0, ctx.held_option_positions, ctx.nav)
    assert res.passed is False
    assert res.reason == "sector_cap_breach"


# ----------------------------------------------------------------------
# Reused-analytics payloads reproduce the approved numbers
# ----------------------------------------------------------------------
def test_income_reuses_wheel_tracker(snapshot, ledger):
    income = adapter.income_view(ledger, snapshot=snapshot)
    assert income["winRate"] == pytest.approx(0.68)
    assert income["realizedYtd"] == 8900
    assert income["premium30d"] == 12400


def test_period_returns(history, snapshot):
    r = adapter.returns_view(history, snapshot)["returns"]
    assert r["YTD"]["pct"] == pytest.approx(-0.056, abs=0.001)
    assert r["YTD"]["usd"] == -8593
    assert r["1Y"]["pct"] == pytest.approx(0.129, abs=0.001)
    assert r["All"]["pct"] == pytest.approx(0.445, abs=0.001)


def test_summary_account_fields(snapshot, ledger):
    s = adapter.account_summary(snapshot, ledger)
    assert s["netLiq"] == 144507
    assert s["unrealizedPnl"] == -37014
    assert s["availableFunds"] == -9257
    assert s["winRate"] == pytest.approx(0.68)


def test_risk_view_meters_and_overlay(snapshot):
    risk = adapter.risk_view(snapshot)
    assert risk["singleName"][0]["sym"] == "CLS"
    semis = next(s for s in risk["sectorExposure"] if s["name"] == "Semiconductors")
    assert semis["pct"] == 312
    assert risk["margin"]["stressed"] is True
    # Every CSP name trips R10 in the live overlay.
    assert all(g["passed"] is False for g in risk["gates"]["singleName"])
    # VaR honestly skips with no correlation/returns matrix (D11).
    assert risk["gates"]["var"]["reason"] == "missing_data"


def test_history_reuses_performance_metrics(history):
    h = adapter.equity_view(history)
    assert len(h["equity"]) == 13
    # Feb peak 156,200 -> Jun 144,507 trough ≈ 7.5% drawdown.
    assert h["stats"]["maxDrawdown"] == pytest.approx(0.0749, abs=0.002)


# ----------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------
def test_bad_schema_version_rejected(tmp_path):
    bad = tmp_path / "portfolio_snapshot.json"
    bad.write_text(json.dumps({"schema_version": 99, "account": {}, "positions": []}))
    with pytest.raises(adapter.SnapshotSchemaError):
        adapter.load_snapshot(tmp_path)


def test_missing_artifact_rejected(tmp_path):
    with pytest.raises(adapter.SnapshotSchemaError):
        adapter.load_snapshot(tmp_path)


# ----------------------------------------------------------------------
# Live-book robustness — a real MCP pull (assigned stock, null deltas,
# snapshot-only) must not crash and must surface real risk honestly.
# ----------------------------------------------------------------------
def _write_live_like(tmp_path) -> dict:
    """A snapshot mirroring the 2026-06-06 live pull: CLS/AMD short puts
    assigned to stock (CLS 500 sh ≈ 131% NAV on margin), MU put still open,
    a CAD covered call, negative cash, and JSON null on the MCP-non-derivable
    delta / realized-YTD fields."""
    snap = {
        "schema_version": 1,
        "as_of": "2026-06-06T23:17:00Z",
        "base_currency": "USD",
        "fx_rates": {"USD": 1.0, "CAD": 0.73},
        "account": {
            "net_liquidation": 142134.0,
            "total_cash": -157377.0,
            "available_funds": -11842.0,
            "excess_liquidity": 3113.0,
            "maintenance_margin": 150000.0,
            "buying_power": 0.0,
            "unrealized_pnl": -42580.0,
            "realized_pnl_ytd": None,
            "day_change_usd": None,
            "day_change_pct": None,
        },
        "positions": [
            {
                "symbol": "CLS",
                "name": "Celestica",
                "sec_type": "STK",
                "qty": 500,
                "mark": 371.88,
                "avg_price": 419.65,
                "unrealized_pnl": -23885.0,
                "currency": "USD",
                "sector": "Semiconductors",
                "in_universe": True,
            },
            {
                "symbol": "AMD",
                "name": "AMD",
                "sec_type": "STK",
                "qty": 100,
                "mark": 458.64,
                "avg_price": 496.89,
                "unrealized_pnl": -3825.0,
                "currency": "USD",
                "sector": "Semiconductors",
                "in_universe": True,
            },
            {
                "symbol": "MU",
                "name": "Micron",
                "sec_type": "OPT",
                "right": "P",
                "strike": 970.0,
                "expiry": "2026-06-12",
                "qty": -1,
                "mark": 120.0,
                "avg_price": 35.69,
                "unrealized_pnl": -9125.0,
                "currency": "USD",
                "sector": "Semiconductors",
                "in_universe": True,
            },
        ],
    }
    (tmp_path / "portfolio_snapshot.json").write_text(json.dumps(snap))
    return adapter.load_snapshot(tmp_path)


def test_assigned_stock_breaches_and_surfaces(tmp_path):
    snap = _write_live_like(tmp_path)
    holdings = {h["sym"]: h for h in adapter.build_holdings_view(snap)}
    # CLS is now assigned stock at ~131% NAV — the book's biggest single-name
    # risk MUST flag (CSP-only breach logic would have missed it).
    assert holdings["CLS"]["state"] == "assigned"
    assert holdings["CLS"]["pctNav"] == 131
    assert holdings["CLS"]["breach"] is True
    # ...and it appears in the single-name concentration meter.
    syms = {row["sym"] for row in adapter.risk_view(snap)["singleName"]}
    assert "CLS" in syms and "MU" in syms


def test_null_kpis_do_not_crash_and_surface_as_null(tmp_path):
    snap = _write_live_like(tmp_path)
    # Snapshot-only (ledger=None) + null deltas must NOT raise.
    s = adapter.account_summary(snap, None)
    assert s["netLiq"] == 142134 and s["cash"] == -157377
    assert s["unrealizedPnl"] == -42580
    # MCP-non-derivable fields come back null (UI shows "—"), not 0.
    assert s["dayChangeUsd"] is None and s["dayChangePct"] is None
    assert s["realizedYtd"] is None and s["premium30d"] is None and s["winRate"] is None


# ----------------------------------------------------------------------
# §2/§3 GUARDS — observational only
# ----------------------------------------------------------------------
_TRIO = ("ev_engine", "wheel_runner", "candidate_dossier")


def test_adapter_imports_nothing_from_the_trio():
    """The adapter must not directly import any decision-trio module
    (CLAUDE.md §2, design-doc §2.1)."""
    src = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = [m for m in imported for trio in _TRIO if trio in m]
    assert not offenders, f"adapter directly imports trio module(s): {offenders}"


def test_adapter_never_issues_ev_authority():
    """No EVEngine usage, ``.evaluate`` call, or token issuance in the adapter
    *code* (AST, so the docstring documenting these prohibitions is ignored)."""
    src = Path(adapter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "EVEngine" not in names
    assert "issue_ev_authority_token" not in (names | attrs)
    assert "evaluate" not in attrs


def test_payloads_carry_no_tradeable_verdict():
    """Every viewer payload is observational — no verdict / EV-authority /
    ev_dollars field anywhere in the JSON."""
    blob = json.dumps(adapter.build_all(FIXTURES)).lower()
    for forbidden in ("verdict", "ev_authority", "tradeable", "ev_dollars"):
        assert forbidden not in blob, f"viewer payload leaked '{forbidden}'"
