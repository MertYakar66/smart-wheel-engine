"""Live IBKR connector → ``portfolio_snapshot.json`` (read-only, observational).

Builds the *exact* ``schema_version: 1`` snapshot that
:mod:`engine.ibkr_portfolio_adapter` consumes, straight from the three
read-only IBKR account endpoints — **no PortfolioAnalyst PDF, no Flex CSV**:

* ``get_account_summary``   → balance-sheet account fields (NAV, cash, margins…)
* ``get_account_balances``  → base-currency unrealized P&L + per-currency FX rates
* ``get_account_positions`` → the position book (stock + option legs)

It is the shared core of the morning-refresh: an agent (Mode 1) or a headless
``ib_insync`` / Client-Portal puller (Mode 2) hands the raw endpoint JSON to
:func:`build_snapshot`, which writes the snapshot the dashboard re-reads per
request (engine_api re-reads the file every call — no restart needed).

**Scope contract (CLAUDE.md §2/§3).** Strictly read-only and observational.
It imports nothing from the decision trio (``ev_engine`` / ``wheel_runner`` /
``candidate_dossier``), never ranks a candidate, never issues an EV-authority
token, and uses **only read endpoints** — never an order tool. Output lands in
the gitignored ``data_processed/ibkr`` runtime dir; real account data is never
committed.

**What this builds vs. what accumulates.** The snapshot (current NAV, cash,
margin, positions, unrealized P&L, FX, day-change) is fully live from one pull.
The equity *curve* (``portfolio_history.json``) and the closed-trade *ledger*
(``wheel_ledger.json``) are seed-once-then-append artifacts — the live trades
endpoint is ~1-year-deep and basic option fills are symbol-level — so they keep
their existing seed and are extended incrementally elsewhere. This module owns
the snapshot only.

Reference joins (sector / company name / in-universe membership) come from the
repo's own S&P-500 constituents file — the live positions feed carries neither
GICS sector nor membership.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONSTITUENTS = _REPO_ROOT / "data_raw" / "sp500_constituents_current.csv"

# Display-sector continuity for the operator's recurring *out-of-universe*
# holdings (not in the S&P-500 constituents file, so no GICS join). Used only
# for the dashboard donut/label — the R9 sector GATE derives its own GICS
# bucket internally (DEFAULT_SECTOR_MAP) and is unaffected by this map.
SECTOR_OVERRIDE = {
    "CLS": "Information Technology",
    "TSM": "Information Technology",
    "MRVL": "Information Technology",
    "ENB": "Energy",
    "CNQ": "Energy",
}

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
_MON3 = {v: k.upper() for k, v in _MONTHS.items()}

# IBKR option contract_description date token, e.g. "Jun12'26".
_DATE_TOK = re.compile(r"^([A-Z][a-z]{2})(\d{1,2})'(\d{2})$")


# ----------------------------------------------------------------------
# contract_description parsing
# ----------------------------------------------------------------------
def parse_option_description(desc: str) -> tuple[str, str, float, str]:
    """``"MU Jun12'26 970 PUT @AMEX"`` → ``("MU", "P", 970.0, "2026-06-12")``.

    Robust to an interposed venue token (``"ENB TSE Oct16'26 86 CALL @CDE"``):
    the underlying is the first token, and the expiry/strike/right are anchored
    on the date token rather than a fixed position.
    """
    toks = desc.split()
    sym = toks[0].upper()
    date_i = next(i for i, t in enumerate(toks) if _DATE_TOK.match(t))
    m = _DATE_TOK.match(toks[date_i])
    assert m is not None
    mon = _MONTHS[m.group(1)]
    day = int(m.group(2))
    year = 2000 + int(m.group(3))
    expiry = f"{year:04d}-{mon:02d}-{day:02d}"
    strike = float(toks[date_i + 1])
    right = "C" if toks[date_i + 2].upper().startswith("C") else "P"
    return sym, right, strike, expiry


def parse_stock_symbol(desc: str) -> str:
    """``"CNQ @TSE"`` / ``"AMD"`` → underlying ticker (first token)."""
    return desc.split()[0].upper()


def _option_label(sym: str, expiry: str, strike: float, right: str) -> str:
    """Compact holdings-table name, matching the imported style:
    ``"MU 12JUN26 970 P"``."""
    y, m, d = (int(x) for x in expiry.split("-"))
    return f"{sym} {d:02d}{_MON3[m]}{y % 100:02d} {strike:g} {right}"


# ----------------------------------------------------------------------
# Reference join (sector / name / in-universe)
# ----------------------------------------------------------------------
def load_reference(constituents_csv: str | Path | None = None) -> dict[str, dict[str, str]]:
    """``{ticker: {"name": Security, "sector": GICS Sector}}`` from the repo's
    S&P-500 constituents file. Presence in this dict *is* in-universe."""
    path = Path(constituents_csv) if constituents_csv else _CONSTITUENTS
    ref: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            tkr = (row.get("ticker") or "").strip().upper()
            if tkr:
                ref[tkr] = {
                    "name": (row.get("Security") or tkr).strip(),
                    "sector": (row.get("GICS Sector") or "Unknown").strip(),
                }
    return ref


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _f(value: Any) -> float | None:
    """``float`` or ``None`` (preserves the snapshot's null-for-unknown
    contract the adapter is built to tolerate)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fx_rates(balances: dict) -> dict[str, float]:
    fx: dict[str, float] = {"USD": 1.0}
    for b in balances.get("balances", []):
        cur = (b.get("currency") or "").upper()
        if cur and cur != "BASE":
            rate = _f(b.get("exchange_rate"))
            if rate is not None:
                fx[cur] = rate
    return fx


def _base_unrealized(balances: dict) -> float | None:
    for b in balances.get("balances", []):
        if (b.get("currency") or "").upper() == "BASE":
            return _f(b.get("unrealized_pnl"))
    return None


# ----------------------------------------------------------------------
# the builder
# ----------------------------------------------------------------------
def build_snapshot(
    summary: dict,
    balances: dict,
    positions: dict,
    *,
    reference: dict[str, dict[str, str]] | None = None,
    as_of: str | None = None,
    source: str = "ibkr_live_connector",
    include_day_change: bool = True,
) -> dict:
    """Assemble the ``schema_version: 1`` snapshot from the three raw IBKR
    read-only endpoint payloads.

    ``include_day_change=False`` nulls ``day_change_usd`` / ``day_change_pct``
    even when ``daily_pnl`` is present — the conservative default until the
    derived day-change has been reconciled against IBKR's own account Day P&L
    (the connector exposes no authoritative account-level day figure to verify
    against, so an unverified headline is withheld rather than shipped)."""
    ref = reference if reference is not None else load_reference()
    fx = _fx_rates(balances)
    as_of = as_of or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    nav = _f(summary.get("net_liquidation")) or 0.0

    # Day-change derived from IBKR's per-position daily P&L (FX-normalized),
    # summed across the whole book INCLUDING names closed today (their realized
    # move still belongs to the day). pct is vs the implied prior NAV; on a
    # levered book this reads larger than the unlevered names' moves suggest.
    day_usd = 0.0
    have_daily = False
    for p in positions.get("positions", []):
        dp = _f(p.get("daily_pnl"))
        if dp is None:
            continue
        have_daily = True
        day_usd += dp * fx.get((p.get("currency") or "USD").upper(), 1.0)
    emit_day = have_daily and include_day_change
    prior_nav = nav - day_usd
    day_pct = (day_usd / prior_nav) if (emit_day and prior_nav) else None

    account = {
        "net_liquidation": nav,
        "total_cash": _f(summary.get("total_cash_value")),
        "available_funds": _f(summary.get("available_funds")),
        "excess_liquidity": _f(summary.get("excess_liquidity")),
        "maintenance_margin": _f(summary.get("maintenance_margin")),
        "buying_power": _f(summary.get("buying_power")),
        "unrealized_pnl": _base_unrealized(balances),
        # Realized-YTD is a ledger-accumulation figure, not on the live summary
        # (its realized_pnl is a daily number) — left null, filled by the ledger.
        "realized_pnl_ytd": None,
        "day_change_usd": round(day_usd, 2) if emit_day else None,
        "day_change_pct": round(day_pct, 4) if day_pct is not None else None,
        # Week-change needs ~5 accumulated daily snapshots — null until then.
        "week_change_usd": None,
        "week_change_pct": None,
    }

    out_positions: list[dict] = []
    for p in positions.get("positions", []):
        qty = int(_f(p.get("position")) or 0)
        if qty == 0:
            continue  # closed today → not part of the current book
        asset = (p.get("asset_class") or "").upper()
        desc = p.get("contract_description") or ""
        currency = (p.get("currency") or "USD").upper()

        if asset == "OPT":
            sym, right, strike, expiry = parse_option_description(desc)
            name = _option_label(sym, expiry, strike, right)
        else:
            sym = parse_stock_symbol(desc)
            right = strike = expiry = None
            name = (ref.get(sym) or {}).get("name") or p.get("company_name") or sym

        info = ref.get(sym, {})
        in_universe = sym in ref
        sector = info.get("sector") or SECTOR_OVERRIDE.get(sym, "Unknown")

        row: dict[str, Any] = {
            "symbol": sym,
            "name": name,
            "sec_type": "OPT" if asset == "OPT" else "STK",
        }
        if asset == "OPT":
            row["right"] = right
            row["strike"] = strike
            row["expiry"] = expiry
        row.update(
            {
                "qty": qty,
                "mark": round(_f(p.get("market_price")) or 0.0, 4),
                "avg_price": round(_f(p.get("average_price")) or 0.0, 4),
                "unrealized_pnl": round(_f(p.get("unrealized_pnl")) or 0.0, 2),
                "currency": currency,
                "sector": sector,
                "in_universe": in_universe,
            }
        )
        out_positions.append(row)

    return {
        "schema_version": 1,
        "as_of": as_of,
        "base_currency": "USD",
        "fx_rates": fx,
        "source": source,
        "note": (
            "Built from the live IBKR read-only connector "
            "(get_account_summary + get_account_balances + get_account_positions) "
            "by scripts/ibkr_live_snapshot.py. No PDF/Flex. Observational "
            "(CLAUDE.md §2/§3). day_change_* derived from per-position daily_pnl; "
            "realized_pnl_ytd / week_change_* are ledger/accumulation fields "
            "(null here)."
        ),
        "account": account,
        "positions": out_positions,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, help="get_account_summary JSON")
    ap.add_argument("--balances", required=True, help="get_account_balances JSON")
    ap.add_argument("--positions", required=True, help="get_account_positions JSON")
    ap.add_argument("--out", required=True, help="destination portfolio_snapshot.json")
    ap.add_argument("--as-of", default=None, help="ISO-8601 timestamp (default: now UTC)")
    ap.add_argument("--constituents", default=None, help="override constituents CSV path")
    ap.add_argument(
        "--no-day-change",
        action="store_true",
        help="null day_change_* (until reconciled against IBKR's own account Day P&L)",
    )
    args = ap.parse_args(argv)

    snap = build_snapshot(
        _read_json(args.summary),
        _read_json(args.balances),
        _read_json(args.positions),
        reference=load_reference(args.constituents),
        as_of=args.as_of,
        include_day_change=not args.no_day_change,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    acct = snap["account"]
    print(
        f"wrote {out}  netLiq={acct['net_liquidation']:.0f}  "
        f"positions={len(snap['positions'])}  "
        f"dayChange={acct['day_change_usd']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
