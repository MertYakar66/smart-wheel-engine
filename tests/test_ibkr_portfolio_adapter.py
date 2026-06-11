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
    # Date-anchored windows: each period anchors at the last curve point
    # STRICTLY BEFORE the window start (YTD vs the Dec-31 prior-year-end NAV
    # of 148,900 — the old first-point-INSIDE-the-window anchor at Jan-30
    # dropped January and misstated YTD). pct and usd share the anchor.
    r = adapter.returns_view(history, snapshot)["returns"]
    assert r["YTD"]["pct"] == pytest.approx((144507 - 148900) / 148900, abs=1e-6)
    assert r["YTD"]["usd"] == -4393
    assert r["1M"]["pct"] == pytest.approx((144507 - 149000) / 149000, abs=1e-6)
    assert r["1M"]["usd"] == -4493
    # Curve starts inside the 1Y window → truncates to the first point.
    assert r["1Y"]["pct"] == pytest.approx(0.129, abs=0.001)
    assert r["1Y"]["usd"] == 16507
    assert r["All"]["pct"] == pytest.approx(0.445, abs=0.001)


def _mixed_grain_history():
    """Monthly points + daily June appends — the live shape after
    scripts/dashboard_refresh.py started appending daily points (the June
    points replicate the month's premium aggregate)."""

    def mk(label, d, port, premium):
        return {
            "label": label,
            "date": d,
            "port": float(port),
            "spy": 100.0,
            "premium": float(premium),
        }

    return {
        "schema_version": 1,
        "inception_capital": 100000.0,
        "points": [
            mk("Nov", "2025-11-28", 118000, 900),
            mk("Dec", "2025-12-31", 120000, 1000),
            mk("Jan '26", "2026-01-30", 132000, 1100),
            mk("Feb", "2026-02-27", 126000, 1200),
            mk("Mar", "2026-03-31", 130000, 1300),
            mk("Apr", "2026-04-30", 124000, 1400),
            mk("May", "2026-05-29", 128000, 1500),
            mk("Jun", "2026-06-05", 121000, 2000),
            mk("Jun 9", "2026-06-09", 119000, 2000),
            mk("Jun 10", "2026-06-10", 125000, 2000),
        ],
    }


def test_ytd_anchor_includes_january_on_mixed_grain():
    """The corrected anchor semantics on a monthly-then-daily curve: YTD
    anchors at the prior year-end point (Dec-31), NOT the first point inside
    the year (Jan-30) — the old anchoring flipped this book's YTD sign."""
    r = adapter.returns_view(_mixed_grain_history())["returns"]
    assert r["YTD"]["pct"] == pytest.approx((125000 - 120000) / 120000, abs=1e-9)
    assert r["YTD"]["usd"] == 5000
    # 1M anchors by DATE (last point <= as_of-30d = Apr-30), not by point
    # count (_back(1) would have grabbed yesterday's daily append).
    assert r["1M"]["pct"] == pytest.approx((125000 - 124000) / 124000, abs=1e-9)
    assert r["1M"]["usd"] == 1000
    # pct and usd always describe the same window: same sign, same anchor.
    for period in ("1M", "3M", "YTD", "1Y", "All"):
        pct, usd = r[period]["pct"], r[period]["usd"]
        assert (pct >= 0) == (usd >= 0), f"{period}: pct {pct} vs usd {usd} contradict"


def test_equity_view_dates_and_premium_month_dedup():
    """Points carry their ISO date (client windows by calendar date) and the
    replicated monthly premium lands only on the LAST point of each month —
    no triple-counted June bars."""
    eq = adapter.equity_view(_mixed_grain_history())["equity"]
    assert [p["date"] for p in eq][:2] == ["2025-11-28", "2025-12-31"]
    june = [p for p in eq if (p["date"] or "").startswith("2026-06")]
    assert [p["premium"] for p in june] == [None, None, 2000]
    # Monthly points keep their own aggregate untouched.
    assert next(p for p in eq if p["m"] == "May")["premium"] == 1500


def test_equity_view_dedup_no_cross_year_collision():
    """Two 'Feb' points from different years must NOT collapse to one dedup key;
    both months must retain their own premium bar (idx 6 fix).

    Prior to the fix, _mkey fell back to label[:7] which gave 'Feb' for both
    2025-02-28 and 2026-02-27 → the later index won and the earlier premium
    was silently dropped."""
    history = {
        "schema_version": 1,
        "inception_capital": 100000.0,
        "points": [
            {
                "label": "Feb '25",
                "date": "2025-02-28",
                "port": 110000.0,
                "spy": 100.0,
                "premium": 800.0,
            },
            {
                "label": "Mar '25",
                "date": "2025-03-31",
                "port": 112000.0,
                "spy": 101.0,
                "premium": 900.0,
            },
            {
                "label": "Feb '26",
                "date": "2026-02-27",
                "port": 120000.0,
                "spy": 105.0,
                "premium": 1200.0,
            },
        ],
    }
    eq = adapter.equity_view(history)["equity"]
    premiums = {p["date"]: p["premium"] for p in eq}
    assert premiums["2025-02-28"] == 800, "Feb 2025 premium must not be dropped"
    assert premiums["2025-03-31"] == 900
    assert premiums["2026-02-27"] == 1200, "Feb 2026 premium must not be dropped"


def test_equity_view_null_premium_stays_null():
    """An emitting point whose premium field is null must surface None, not a
    fabricated $0 bar (idx 6 fix — _num(None)=0.0 was the wrong path)."""
    history = {
        "schema_version": 1,
        "inception_capital": 100000.0,
        "points": [
            {"label": "Jan", "date": "2026-01-31", "port": 110000.0, "spy": 100.0, "premium": None},
            {
                "label": "Feb",
                "date": "2026-02-28",
                "port": 112000.0,
                "spy": 101.0,
                "premium": 500.0,
            },
        ],
    }
    eq = adapter.equity_view(history)["equity"]
    jan = next(p for p in eq if p["date"] == "2026-01-31")
    assert jan["premium"] is None, "null premium must stay None, not become 0"


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


def test_risk_view_concentration_covers_every_underlying(snapshot):
    """`concentration` (all-exposure) lists EVERY underlying — including the
    covered-call stock the R10 `singleName` meter excludes by design — so a
    stock position at >100% NAV can never be invisible on the risk radar."""
    risk = adapter.risk_view(snapshot)
    conc = {row["sym"]: row for row in risk["concentration"]}
    assert set(conc) == {h["sym"] for h in adapter.build_holdings_view(snapshot)}
    # Covered-call stock (absent from singleName) is present with its state.
    assert conc["TSM"]["state"] == "cc" and conc["TSM"]["pct"] == 29
    # Sorted descending by exposure.
    pcts = [row["pct"] for row in risk["concentration"]]
    assert pcts == sorted(pcts, reverse=True)


def test_flat_legs_carry_option_fields(snapshot):
    """Option legs emit strike/expiry/dte/moneyness; stock legs emit null.
    Moneyness uses the same-symbol stock leg as spot and is null (never
    fabricated) when the book holds no stock for that name."""
    legs = {(r["sym"], r["state"]): r for r in adapter.build_positions_flat(snapshot)}
    cls_put = legs[("CLS", "short_put")]
    assert cls_put["strike"] == 425.0
    assert cls_put["expiry"] == "2026-06-12"
    assert cls_put["dte"] == 7  # as_of 2026-06-05
    assert cls_put["moneyness"] is None  # no CLS stock leg → no spot
    tsm_call = legs[("TSM", "short_call")]
    assert tsm_call["moneyness"] == pytest.approx((412.5 - 430.0) / 430.0, abs=1e-3)
    stock = legs[("TSM", "shares")]
    assert stock["strike"] is None and stock["dte"] is None and stock["moneyness"] is None


def test_moneyness_is_fx_invariant(snapshot):
    """Moneyness for a CAD-listed covered call must compare local-currency
    spot vs local-currency strike (both in CAD) — NOT FX-normalised spot vs
    local strike.  With ENB: stock mark C$77.18, call strike C$80.0 →
    moneyness = (77.18 − 80.0) / 80.0 ≈ −0.0353 (slightly OTM).

    Before the fix, spots was stored as mark * fx_rate (CAD→USD: 77.18 * 0.73
    = 56.34 USD) while strike stayed at raw C$80.0, giving a wrong −29.6%
    (deep OTM) instead of −3.5%.  The fix: compare pre-FX mark vs pre-FX
    strike so moneyness is dimensionless and FX-invariant (idx 22)."""
    legs = {(r["sym"], r["state"]): r for r in adapter.build_positions_flat(snapshot)}
    enb_call = legs[("ENB", "short_call")]
    # Correct: local-currency comparison (C$77.18 stock vs C$80 strike)
    expected = (77.18 - 80.0) / 80.0
    assert enb_call["moneyness"] == pytest.approx(expected, abs=1e-3)
    # Sanity: the wrong FX-mixed value would be ≈ −0.296; assert we are NOT there
    wrong = (77.18 * 0.73 - 80.0) / 80.0  # pre-fix value ≈ −0.296
    assert abs(enb_call["moneyness"] - wrong) > 0.1, "moneyness still uses FX-mixed spot"


def test_csp_row_named_after_put_leg():
    """A strangle whose short CALL appears first in snapshot order must not
    lend its contract name to the csp row priced off the short PUT."""
    snap = {
        "schema_version": 1,
        "as_of": "2026-06-10T14:04:00Z",
        "base_currency": "USD",
        "fx_rates": {"USD": 1.0},
        "account": {"net_liquidation": 152381.0},
        "positions": [
            {
                "symbol": "MU",
                "name": "MU 12JUN26 1070 C",
                "sec_type": "OPT",
                "right": "C",
                "strike": 1070.0,
                "expiry": "2026-06-12",
                "qty": -1,
                "mark": 2.92,
                "unrealized_pnl": 827.0,
                "currency": "USD",
            },
            {
                "symbol": "MU",
                "name": "MU 12JUN26 970 P",
                "sec_type": "OPT",
                "right": "P",
                "strike": 970.0,
                "expiry": "2026-06-12",
                "qty": -1,
                "mark": 70.35,
                "unrealized_pnl": -3466.0,
                "currency": "USD",
            },
        ],
    }
    row = next(h for h in adapter.build_holdings_view(snap) if h["sym"] == "MU")
    assert row["state"] == "csp"
    assert row["name"] == "MU 12JUN26 970 P"  # the leg the mark describes
    assert row["mark"] == pytest.approx(70.35)


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
