"""IBKR snapshot → engine types for the read-only performance viewer (D24/D26).

Single responsibility: turn the point-in-time IBKR artifacts on disk
(``data_processed/ibkr/portfolio_snapshot.json`` + the accumulated
``portfolio_history.json`` + the ``wheel_ledger.json`` trade export)
into the shapes the dashboard and the existing engine analytics expect.

**Scope contract (CLAUDE.md §2/§3, design-doc IBKR_LIVE_BOOK_INTEGRATION
§2.1/§6.7).** This module is *outside* the CI-gated decision trio and
imports **nothing** from ``ev_engine`` / ``wheel_runner`` /
``candidate_dossier``. It is purely **observational**: it reads the book
and reports. It never ranks a candidate, never calls ``EVEngine.evaluate``,
never issues an EV-authority token, and never converts data into a
tradeable verdict. ``ev_dollars`` is not produced or forecast here —
realized P&L (what happened) is reported distinctly from any forward
score (finding I1).

It *does* reuse the non-trio analytics libraries the viewer is built to
surface — ``portfolio_tracker`` (period returns / TWR), ``wheel_tracker``
(win-rate, realized P&L), ``performance_metrics`` (Sharpe / Sortino /
drawdown), and ``portfolio_risk_gates`` (the D17 R7–R11 concentration /
VaR / stress gates) — to power a live risk overlay against the real book.

**Universe discipline.** In-S&P-500 names get full treatment. Out-of-
universe names (e.g. CNQ, ENB on the TSX) are **exposure-only**: they
count toward NAV, sector, and currency denominators (real risk), but are
flagged ``in_universe=False`` and are never placed in the rankable set
(:func:`rankable_symbols`). The viewer never claims authority over them.

**FX.** ``base_currency`` (USD) is the reporting currency. Per-position
``mark`` / ``avg_price`` / ``unrealized_pnl`` are stored in each row's
native ``currency`` and normalized to base via the snapshot's
``fx_rates`` map before any NAV / concentration math.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Non-trio analytics the observational viewer reuses. (ev_engine /
# wheel_runner / candidate_dossier are deliberately NOT imported — see
# the module docstring + the guard test in
# tests/test_ibkr_portfolio_adapter.py.)
from engine.performance_metrics import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from engine.portfolio_risk_gates import (
    PortfolioContext,
    check_sector_cap,
    check_single_name_cap,
    check_stress_scenario,
    check_var,
)
from engine.portfolio_tracker import PortfolioSnapshot, PortfolioTracker
from engine.wheel_tracker import WheelTracker

# ----------------------------------------------------------------------
# Locations + constants
# ----------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data_processed" / "ibkr"


def _default_dir() -> Path:
    """Resolve the IBKR-artifact directory.

    ``SWE_IBKR_DATA_DIR`` (if set) wins — this is how a deployment points the
    viewer at a live snapshot dir, and how the test suite / a fresh-clone demo
    points it at the frozen ``tests/fixtures/ibkr`` artifacts. Otherwise the
    gitignored runtime location ``data_processed/ibkr`` is used (same
    point-in-time, on-disk discipline as the Bloomberg CSVs).
    """
    env = os.environ.get("SWE_IBKR_DATA_DIR")
    return Path(env) if env else _DEFAULT_DATA_DIR


SNAPSHOT_FILE = "portfolio_snapshot.json"
HISTORY_FILE = "portfolio_history.json"
LEDGER_FILE = "wheel_ledger.json"

# Cap thresholds surfaced to the viewer. These mirror the D17 locked
# defaults in portfolio_risk_gates (_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10,
# _DEFAULT_MAX_SECTOR_PCT = 0.25); the gate functions remain the source
# of truth for the actual pass/fail decisions.
SINGLE_NAME_CAP_PCT = 0.10
SECTOR_CAP_PCT = 0.25

# Implied-vol assumption for the VaR/stress overlay when the snapshot
# carries no live option greeks (Track C — live IV freshness — is a
# separate, later concern). Documented, not silent: the risk overlay
# reports this assumption. R9/R10 (sector + single-name) do not use IV.
_DEFAULT_IV = 0.50

_PERIODS = ["1D", "1W", "1M", "3M", "YTD", "1Y", "All"]


class SnapshotSchemaError(ValueError):
    """Raised when an artifact is missing or carries an unsupported
    ``schema_version``."""


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SnapshotSchemaError(f"artifact not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_snapshot(data_dir: str | Path | None = None) -> dict:
    """Load + validate the point-in-time IBKR snapshot (design-doc §2.2)."""
    base = Path(data_dir) if data_dir else _default_dir()
    snap = _load_json(base / SNAPSHOT_FILE)
    if snap.get("schema_version") != 1:
        raise SnapshotSchemaError(
            f"unsupported snapshot schema_version: {snap.get('schema_version')!r}"
        )
    if "account" not in snap or "positions" not in snap:
        raise SnapshotSchemaError("snapshot missing 'account' or 'positions'")
    return snap


def load_history(data_dir: str | Path | None = None) -> dict:
    """Load the accumulated monthly equity series (design-doc §6.4/§6.5)."""
    base = Path(data_dir) if data_dir else _default_dir()
    hist = _load_json(base / HISTORY_FILE)
    if hist.get("schema_version") != 1:
        raise SnapshotSchemaError(
            f"unsupported history schema_version: {hist.get('schema_version')!r}"
        )
    return hist


def load_ledger(data_dir: str | Path | None = None) -> dict:
    """Load the closed-wheel-trade ledger (the Flex/trades export, §6.4)."""
    base = Path(data_dir) if data_dir else _default_dir()
    led = _load_json(base / LEDGER_FILE)
    if led.get("schema_version") != 1:
        raise SnapshotSchemaError(
            f"unsupported ledger schema_version: {led.get('schema_version')!r}"
        )
    return led


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _as_of_date(snapshot: dict) -> date:
    raw = snapshot.get("as_of", "")
    # Tolerate trailing 'Z' (UTC) which datetime.fromisoformat rejects on
    # older interpreters; we only need the calendar date.
    raw = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return date.today()


def _fx_rate(snapshot: dict, currency: str) -> float:
    rates = snapshot.get("fx_rates") or {}
    return _num(rates.get(currency, 1.0), 1.0)


def _num(value, default: float = 0.0) -> float:
    """``float()`` that treats ``None`` / unparseable / non-finite as the
    default. Real IBKR snapshots carry JSON ``null`` for fields the MCP can't
    derive (day/week deltas, realized-YTD) — a naive ``float(None)`` would
    crash the summary endpoint on a live book."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _opt_num(value) -> float | None:
    """Like :func:`_num` but returns ``None`` (not a default) when the value
    is absent / null / unparseable — so a non-derivable KPI surfaces as JSON
    ``null`` (the dashboard renders it as "—") instead of a misleading 0."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_option(pos: dict) -> bool:
    return str(pos.get("sec_type", "")).upper() == "OPT"


def _is_stock(pos: dict) -> bool:
    return str(pos.get("sec_type", "")).upper() == "STK"


def _option_type(pos: dict) -> str:
    return "put" if str(pos.get("right", "")).upper() == "P" else "call"


# ----------------------------------------------------------------------
# Holdings view — per-symbol, wheel-aware rows matching the dashboard
# Holding shape (sym, name, state, qty, mark, mktValue, uPnl, pctNav,
# breach, sector, currency, in_universe).
# ----------------------------------------------------------------------
def build_holdings_view(snapshot: dict) -> list[dict]:
    """Aggregate the per-instrument snapshot into per-symbol wheel rows.

    A short put (OPT/P, qty<0) → ``csp``. Stock + a short call (OPT/C) →
    ``cc``. Stock alone → ``assigned``. Values are FX-normalized to the
    base currency. ``pctNav`` is the short-put notional (strike×100×|qty|)
    for CSPs and the stock market value for shares; ``breach`` flags a CSP
    **or an assigned name** over the single-name cap (covered-call stock is
    a defined leg and never breaches) — the R10-style concentration meter.
    """
    nav = _num(snapshot["account"]["net_liquidation"])
    # Preserve first-appearance order so the table matches the snapshot.
    order: list[str] = []
    by_symbol: dict[str, list[dict]] = {}
    for pos in snapshot["positions"]:
        sym = str(pos["symbol"]).upper()
        if sym not in by_symbol:
            by_symbol[sym] = []
            order.append(sym)
        by_symbol[sym].append(pos)

    rows: list[dict] = []
    for sym in order:
        legs = by_symbol[sym]
        fx = _fx_rate(snapshot, legs[0].get("currency", "USD"))
        name = legs[0].get("name", sym)
        sector = legs[0].get("sector", "Unknown")
        currency = legs[0].get("currency", "USD")
        in_universe = bool(legs[0].get("in_universe", True))

        short_puts = [
            p for p in legs if _is_option(p) and _option_type(p) == "put" and p["qty"] < 0
        ]
        short_calls = [
            p for p in legs if _is_option(p) and _option_type(p) == "call" and p["qty"] < 0
        ]
        stocks = [p for p in legs if _is_stock(p)]

        # Unrealized P&L is the sum across every leg, FX-normalized.
        upnl = sum(
            _num(p.get("unrealized_pnl")) * _fx_rate(snapshot, p.get("currency", "USD"))
            for p in legs
        )

        if short_puts and not stocks:
            state = "csp"
            put = short_puts[0]
            qty = int(_num(put.get("qty")))
            mark = _num(put.get("mark")) * fx
            mkt_value = -(mark * abs(qty) * 100.0)
            notional = _num(put.get("strike")) * 100.0 * abs(qty) * fx
            pct_nav = (notional / nav * 100.0) if nav else 0.0
            # CSP single-name (R10) concentration: short-put notional vs cap.
            breach = pct_nav > SINGLE_NAME_CAP_PCT * 100.0
        elif stocks:
            shares = sum(int(_num(s.get("qty"))) for s in stocks)
            mark = _num(stocks[0].get("mark")) * fx
            mkt_value = shares * mark
            pct_nav = (mkt_value / nav * 100.0) if nav else 0.0
            qty = shares
            state = "cc" if short_calls else "assigned"
            # Covered-call stock is a DEFINED/intended leg → never a breach.
            # ASSIGNED stock (a CSP that went ITM) is a *realized* single-name
            # concentration — flag it on the same 10%-NAV floor as a CSP, so
            # the live book's biggest risk (e.g. an assigned name at >100% NAV)
            # surfaces. Demo is unaffected (its only assigned name is <cap).
            breach = state == "assigned" and pct_nav > SINGLE_NAME_CAP_PCT * 100.0
        else:
            # Defensive: an option-only row that isn't a short put
            # (e.g. a naked short call) — surface it without crashing.
            opt = legs[0]
            qty = int(_num(opt.get("qty")))
            mark = _num(opt.get("mark")) * fx
            mkt_value = -(mark * abs(qty) * 100.0)
            pct_nav = (
                (_num(opt.get("strike")) * 100.0 * abs(qty) * fx / nav * 100.0) if nav else 0.0
            )
            state = "cc" if _option_type(opt) == "call" else "csp"
            breach = state == "csp" and pct_nav > SINGLE_NAME_CAP_PCT * 100.0

        rows.append(
            {
                "sym": sym,
                "name": name,
                "state": state,
                "qty": qty,
                "mark": round(mark, 2),
                "mktValue": round(mkt_value),
                "uPnl": round(upnl),
                "pctNav": round(pct_nav),
                "pctNavExact": pct_nav,
                "breach": bool(breach),
                "sector": sector,
                "currency": currency,
                "inUniverse": in_universe,
            }
        )
    return rows


# ----------------------------------------------------------------------
# Engine type: PortfolioContext + held_option_positions + nav
# ----------------------------------------------------------------------
def build_portfolio_context(
    snapshot: dict,
    *,
    default_iv: float = _DEFAULT_IV,
) -> PortfolioContext:
    """Build a :class:`engine.portfolio_risk_gates.PortfolioContext`.

    ``held_option_positions`` are emitted in the exact shape the gate
    functions read — ``symbol`` / ``option_type`` / ``strike`` / ``dte``
    / ``iv`` / ``contracts`` / ``is_short`` — NOT the ``ticker``-keyed
    shape ``engine_api._build_portfolio_context_from_params`` produces
    (which omits ``is_short`` and would zero-out the single-name
    aggregation). Both in- and out-of-universe positions are included:
    out-of-universe names are exposure-only but are *real* risk and must
    count toward the concentration denominators (design-doc §2.3).
    """
    as_of = _as_of_date(snapshot)
    nav = _num(snapshot["account"]["net_liquidation"])

    held_option_positions: list[dict] = []
    stock_holdings: list[tuple[str, int]] = []
    spot_prices: dict[str, float] = {}

    for pos in snapshot["positions"]:
        sym = str(pos["symbol"]).upper()
        fx = _fx_rate(snapshot, pos.get("currency", "USD"))
        if _is_option(pos):
            try:
                expiry = date.fromisoformat(str(pos.get("expiry", "")))
                dte = max(0, (expiry - as_of).days)
            except ValueError:
                dte = 35
            qty = int(_num(pos.get("qty")))
            held_option_positions.append(
                {
                    "symbol": sym,
                    "option_type": _option_type(pos),
                    "strike": _num(pos.get("strike")),
                    "dte": dte,
                    "iv": _num(pos.get("iv"), default_iv),
                    "contracts": abs(qty),
                    "is_short": qty < 0,
                }
            )
        elif _is_stock(pos):
            stock_holdings.append((sym, int(_num(pos.get("qty")))))
            spot_prices[sym] = _num(pos.get("mark")) * fx

    return PortfolioContext(
        held_option_positions=held_option_positions,
        stock_holdings=stock_holdings,
        nav=nav,
        spot_prices=spot_prices,
    )


def rankable_symbols(snapshot: dict) -> list[str]:
    """In-universe symbols only — the set the engine could ever rank.

    Out-of-universe names (CNQ/ENB) are intentionally excluded: the
    viewer counts them as exposure but never ranks or claims authority
    over them (universe discipline, design-doc §2.3 / §6.7).
    """
    seen: list[str] = []
    for pos in snapshot["positions"]:
        if bool(pos.get("in_universe", True)):
            sym = str(pos["symbol"]).upper()
            if sym not in seen:
                seen.append(sym)
    return seen


# ----------------------------------------------------------------------
# /summary — account-level KPIs (ACCOUNT shape)
# ----------------------------------------------------------------------
def account_summary(snapshot: dict, ledger: dict | None = None) -> dict:
    """ACCOUNT shape for the KPI header + cards.

    Balance-sheet fields (net-liq, cash, available funds, excess liquidity,
    maintenance margin, unrealized P&L) are always live from the snapshot.
    The KPIs the IBKR MCP cannot derive on a live pull — ``dayChangeUsd`` /
    ``dayChangePct`` (the snapshot tools expose no deltas) and, when no
    ``ledger`` is attached, ``realizedYtd`` / ``premium30d`` / ``winRate`` —
    surface as JSON ``null`` (the dashboard renders "—") rather than a
    misleading 0. ``unrealizedPnl`` is the IBKR account-level BASE figure
    (authoritative; differs from a naive sum of MCP marks by FX + timing).
    Realized history is shown distinctly from any forward score (finding I1).
    """
    acct = snapshot["account"]
    income = income_view(ledger, snapshot=snapshot) if ledger else {}

    def _r(value):
        n = _opt_num(value)
        return round(n) if n is not None else None

    return {
        "asOf": snapshot.get("as_of"),
        "netLiq": round(_num(acct.get("net_liquidation"))),
        "dayChangeUsd": _r(acct.get("day_change_usd")),
        "dayChangePct": _opt_num(acct.get("day_change_pct")),
        "cash": round(_num(acct.get("total_cash"))),
        "unrealizedPnl": _r(acct.get("unrealized_pnl")),
        "realizedYtd": income.get("realizedYtd") if income else _r(acct.get("realized_pnl_ytd")),
        "premium30d": income.get("premium30d") if income else None,
        "winRate": income.get("winRate") if income else None,
        "availableFunds": round(_num(acct.get("available_funds"))),
        "excessLiquidity": round(_num(acct.get("excess_liquidity"))),
        "maintMargin": round(_num(acct.get("maintenance_margin"))),
        "baseCurrency": snapshot.get("base_currency", "USD"),
    }


# ----------------------------------------------------------------------
# /returns — period returns (RETURNS shape: {period: {pct, usd}})
# ----------------------------------------------------------------------
def _history_tracker(history: dict) -> PortfolioTracker:
    """Replay the monthly equity series into a PortfolioTracker so the
    library's TWR ``get_returns`` powers the period percentages (genuine
    reuse; validated to reproduce the approved 1M/3M/YTD/1Y figures)."""
    inception = float(history.get("inception_capital", history["points"][0]["port"]))
    tracker = PortfolioTracker(initial_cash=inception)
    tracker.snapshots = []
    for p in history["points"]:
        d = date.fromisoformat(p["date"])
        tracker.snapshots.append(
            PortfolioSnapshot(
                date=d,
                total_value=float(p["port"]),
                cash=0.0,
                invested_value=float(p["port"]),
                unrealized_pnl=0.0,
                realized_pnl_cumulative=0.0,
                dividends_cumulative=0.0,
                deposits_cumulative=inception,
                withdrawals_cumulative=0.0,
                holdings_count=0,
            )
        )
    tracker.benchmark_snapshots = [
        (date.fromisoformat(p["date"]), float(p["spy"])) for p in history["points"]
    ]
    return tracker


def returns_view(history: dict, snapshot: dict | None = None) -> dict:
    """RETURNS shape. 1M/3M/YTD/1Y percentages come from
    ``PortfolioTracker.get_returns`` (TWR); the dollar deltas come from
    the snapshot series; 1D/1W come from the live account deltas; All-time
    anchors to ``inception_capital`` (the TWR 'all_time' window only spans
    the curve, so it is computed explicitly — §6.4 snapshot-delta method).
    """
    points = history["points"]
    inception = float(history.get("inception_capital", points[0]["port"]))
    last = float(points[-1]["port"])
    tracker = _history_tracker(history)
    m = tracker.get_returns(as_of=date.fromisoformat(points[-1]["date"]))

    # Dollar baselines straight off the monthly series (matches the
    # approved view exactly): 1M = prev month, 3M = 3 months back,
    # YTD = first row of the current year, 1Y = ~12 months back / first.
    def _back(n: int) -> float:
        return float(points[-1 - n]["port"]) if len(points) > n else float(points[0]["port"])

    last_year = date.fromisoformat(points[-1]["date"]).year
    ytd_base = next(
        (float(p["port"]) for p in points if date.fromisoformat(p["date"]).year == last_year),
        float(points[0]["port"]),
    )

    acct = (snapshot or {}).get("account", {})
    day_pct = float(acct.get("day_change_pct", 0.0))
    day_usd = float(acct.get("day_change_usd", 0.0))
    week_pct = float(acct.get("week_change_pct", 0.0))
    week_usd = float(acct.get("week_change_usd", 0.0))

    returns = {
        "1D": {"pct": day_pct, "usd": round(day_usd)},
        "1W": {"pct": week_pct, "usd": round(week_usd)},
        "1M": {"pct": float(m.return_1m), "usd": round(last - _back(1))},
        "3M": {"pct": float(m.return_3m), "usd": round(last - _back(3))},
        "YTD": {"pct": float(m.return_ytd), "usd": round(last - ytd_base)},
        "1Y": {"pct": float(m.return_1y), "usd": round(last - _back(12))},
        "All": {
            "pct": (last - inception) / inception if inception else 0.0,
            "usd": round(last - inception),
        },
    }
    return {"returns": returns}


# ----------------------------------------------------------------------
# /history — equity curve (EQUITY shape) + reused risk stats
# ----------------------------------------------------------------------
def equity_view(history: dict) -> dict:
    """EQUITY shape ([{m, port, spy, premium}]) plus reused Sharpe /
    Sortino / max-drawdown over the monthly ``port`` series (monthly
    periodicity → periods_per_year=12)."""
    points = history["points"]
    equity = [
        {
            "m": p["label"],
            "port": round(float(p["port"])),
            "spy": round(float(p["spy"])),
            "premium": round(float(p["premium"])),
        }
        for p in points
    ]

    eq_df = pd.DataFrame({"portfolio_value": [float(p["port"]) for p in points]})
    rets = eq_df["portfolio_value"].pct_change().dropna()
    max_dd, dd_days = calculate_max_drawdown(eq_df)
    stats = {
        "sharpe": _finite(calculate_sharpe_ratio(rets, periods_per_year=12)),
        "sortino": _finite(calculate_sortino_ratio(rets, periods_per_year=12)),
        "maxDrawdown": _finite(max_dd),
        "maxDrawdownPeriods": int(dd_days),
    }
    return {"equity": equity, "stats": stats}


def _finite(v: float) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ----------------------------------------------------------------------
# /income — realized P&L / premium / win-rate (reuses wheel_tracker)
# ----------------------------------------------------------------------
def income_view(ledger: dict, *, snapshot: dict | None = None) -> dict:
    """Realized P&L, premium income, and win-rate from the closed-trade
    ledger. Win-rate is computed by ``WheelTracker.get_performance_summary``
    (genuine reuse). Realized P&L is reported distinctly from any forward
    EV score (finding I1 — the viewer never presents ev_dollars as a
    realized-P&L forecast)."""
    closed = list(ledger.get("closed_positions", []))
    as_of = _as_of_date(snapshot) if snapshot else date.today()

    tracker = WheelTracker()
    tracker.closed_positions = closed
    summary = tracker.get_performance_summary()
    win_rate = float(summary.iloc[0]["win_rate"]) if not summary.empty else 0.0
    total_realized = float(summary.iloc[0]["total_pnl"]) if not summary.empty else 0.0

    if not closed:
        return {
            "realizedYtd": 0,
            "premium30d": 0,
            "winRate": 0.0,
            "totalRealized": 0,
            "byName": [],
            "byMonth": [],
        }

    df = pd.DataFrame(closed)
    df["exit"] = pd.to_datetime(df["exit_date"])
    df["premium"] = df.get("put_premium", 0.0).fillna(0.0) + df.get("call_premium", 0.0).fillna(0.0)

    as_of_ts = pd.Timestamp(as_of)
    realized_ytd = float(df.loc[df["exit"].dt.year == as_of.year, "net_pnl"].sum())
    premium_30d = float(df.loc[df["exit"] >= as_of_ts - pd.Timedelta(days=30), "premium"].sum())

    by_name = [
        {"sym": str(sym), "pnl": round(float(v))}
        for sym, v in df.groupby("ticker")["net_pnl"].sum().sort_values(ascending=False).items()
    ]
    by_month_series = df.set_index("exit")["net_pnl"].groupby(pd.Grouper(freq="MS")).sum().dropna()
    by_month = [
        {"m": ts.strftime("%b '%y").replace("'20", "'"), "pnl": round(float(v))}
        for ts, v in by_month_series.items()
    ]

    return {
        "realizedYtd": round(realized_ytd),
        "premium30d": round(premium_30d),
        "winRate": round(win_rate, 4),
        "totalRealized": round(total_realized),
        "byName": by_name,
        "byMonth": by_month,
    }


# ----------------------------------------------------------------------
# /risk — concentration meters + live R7–R11 overlay (reuses risk gates)
# ----------------------------------------------------------------------
def risk_view(snapshot: dict, *, default_iv: float = _DEFAULT_IV) -> dict:
    """Concentration meters (single-name + sector), allocation donut,
    currency split, margin health, and the live R9/R10/R7/R8 gate
    overlay run against the real book — the D24 'arm the dormant gates'
    payoff. R9 (sector) and R10 (single-name) are the headline meters;
    R7 (VaR) skips with ``missing_data`` absent a correlation/returns
    matrix (honest, per D11); R8 (stress) runs under the documented IV
    assumption."""
    acct = snapshot["account"]
    nav = float(acct["net_liquidation"])
    holdings = build_holdings_view(snapshot)

    # Single-name meter: the CSP (short-put) names — what R10 bounds.
    # Single-name concentration meter: open CSPs (assignment risk to monitor)
    # plus any ASSIGNED name that breaches the cap (a CSP that went ITM is now
    # realized single-name concentration). Covered-call stock is excluded
    # (defined/intended). Demo is unchanged (its only assigned name is <cap);
    # on a live book this is what surfaces an assigned name at >100% NAV.
    single_name = sorted(
        (
            {"sym": h["sym"], "pct": h["pctNav"]}
            for h in holdings
            if h["state"] == "csp" or (h["state"] == "assigned" and h["breach"])
        ),
        key=lambda d: d["pct"],
        reverse=True,
    )

    # Sector + currency from FX-normalized gross market value / notional.
    sector_notional: dict[str, float] = {}
    sector_gross: dict[str, float] = {}
    currency_gross: dict[str, float] = {}
    for h in holdings:
        # Sum the *exact* (unrounded) FX-normalized notional share so the
        # sector total matches the approved meter (rounding per-name first
        # would drift the sum by ~1pp).
        sector_notional[h["sector"]] = sector_notional.get(h["sector"], 0.0) + abs(h["pctNavExact"])
        sector_gross[h["sector"]] = sector_gross.get(h["sector"], 0.0) + abs(h["mktValue"])
        currency_gross[h["currency"]] = currency_gross.get(h["currency"], 0.0) + abs(h["mktValue"])

    sector_exposure = sorted(
        ({"name": k, "pct": round(v)} for k, v in sector_notional.items()),
        key=lambda d: d["pct"],
        reverse=True,
    )
    total_gross = sum(sector_gross.values()) or 1.0
    sectors = sorted(
        ({"name": k, "val": round(v / total_gross * 100.0, 1)} for k, v in sector_gross.items()),
        key=lambda d: d["val"],
        reverse=True,
    )
    total_ccy = sum(currency_gross.values()) or 1.0
    currency = sorted(
        ({"name": k, "val": round(v / total_ccy * 100.0, 1)} for k, v in currency_gross.items()),
        key=lambda d: d["val"],
        reverse=True,
    )

    # Live gate overlay against the real book.
    ctx = build_portfolio_context(snapshot, default_iv=default_iv)
    gates = _run_gates(ctx, holdings, nav)

    excess = float(acct.get("excess_liquidity", 0.0))
    maint = float(acct.get("maintenance_margin", 0.0))
    margin = {
        "availableFunds": round(float(acct.get("available_funds", 0.0))),
        "excessLiquidity": round(excess),
        "maintMargin": round(maint),
        # Cushion ratio in [0,1]: how much excess liquidity covers the
        # maintenance requirement. A stressed book reads low.
        "cushionPct": round(max(0.0, excess) / maint, 4) if maint else 0.0,
        "stressed": float(acct.get("available_funds", 0.0)) < 0,
    }

    return {
        "singleNameCap": round(SINGLE_NAME_CAP_PCT * 100),
        "sectorCap": round(SECTOR_CAP_PCT * 100),
        "singleName": single_name,
        "sectorExposure": sector_exposure,
        "sectors": sectors,
        "currency": currency,
        "margin": margin,
        "gates": gates,
        "ivAssumption": default_iv,
    }


def _run_gates(ctx: PortfolioContext, holdings: list[dict], nav: float) -> dict:
    """Run R9/R10/R7/R8 against the held book (no candidate — exposure
    of the *current* portfolio). Returns a JSON-safe summary per gate."""
    held = ctx.held_option_positions

    # R10 single-name: one check per CSP name (proposed_notional=0 → pure
    # current exposure of the held book).
    single_name_gates = []
    for h in holdings:
        if h["state"] != "csp":
            continue
        res = check_single_name_cap(h["sym"], 0.0, held, nav)
        single_name_gates.append(
            {
                "sym": h["sym"],
                "passed": res.passed,
                "reason": res.reason,
                "pctNav": round(res.details.get("post_open_name_pct", 0.0) * 100, 1),
            }
        )

    # R9 sector: run check_sector_cap per option-bearing name and dedupe
    # on the GICS sector the GATE itself derives (DEFAULT_SECTOR_MAP) —
    # NOT the display sector — so two display sectors that the gate maps
    # to the same GICS bucket don't double-report the same exposure.
    sector_gates = []
    seen_gics: set[str] = set()
    for h in holdings:
        if h["state"] not in ("csp", "cc"):
            continue
        res = check_sector_cap(h["sym"], 0.0, held, nav)
        gics = str(res.details.get("sector", "Unknown"))
        if gics in seen_gics:
            continue
        seen_gics.add(gics)
        sector_gates.append(
            {
                "sector": gics,
                "passed": res.passed,
                "reason": res.reason,
                "pctNav": round(res.details.get("post_open_sector_pct", 0.0) * 100, 1),
            }
        )

    # R7 VaR — no correlation/returns matrix in the snapshot → honest skip.
    var_res = check_var(held, ctx.spot_prices, {}, nav)
    # R8 stress — C4 vol-spike under the documented IV assumption.
    stress_res = check_stress_scenario(held, ctx.spot_prices, {}, nav)

    return {
        "singleName": single_name_gates,
        "sector": sector_gates,
        "var": {
            "passed": var_res.passed,
            "reason": var_res.reason,
            "varPct": _finite(var_res.details.get("var_pct")),
        },
        "stress": {
            "passed": stress_res.passed,
            "reason": stress_res.reason,
            "drawdownPct": _finite(stress_res.details.get("drawdown_pct")),
            "scenario": stress_res.details.get("scenario_name"),
        },
    }


# ----------------------------------------------------------------------
# Convenience: load everything + emit every viewer shape in one call.
# ----------------------------------------------------------------------
def build_all(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load all three artifacts and return every endpoint payload. Used
    by the engine_api handlers and the adapter tests."""
    snapshot = load_snapshot(data_dir)
    history = load_history(data_dir)
    ledger = load_ledger(data_dir)
    return {
        "summary": account_summary(snapshot, ledger),
        "positions": {"holdings": build_holdings_view(snapshot)},
        "returns": returns_view(history, snapshot),
        "income": income_view(ledger, snapshot=snapshot),
        "risk": risk_view(snapshot),
        "history": equity_view(history),
    }
