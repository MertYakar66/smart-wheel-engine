"""Headless read-only IB Gateway puller → ``portfolio_snapshot.json``.

The unattended (Mode 2) sibling of :mod:`scripts.ibkr_live_snapshot`: instead of
the agent-time cloud connector, it talks to a **local IB Gateway** over the
socket API and feeds the *same* :func:`build_snapshot` transform, so the snapshot
the dashboard reads is identical regardless of source. Intended to run from a
Windows Task Scheduler job before the open.

**Scope contract (CLAUDE.md §2/§3) — strictly read-only.** It connects with
``readonly=True`` (the API session itself refuses to transmit orders) and calls
**only** read methods (``accountSummary`` / ``portfolio`` / a Forex price
snapshot). It NEVER calls ``placeOrder`` or any order/modify/cancel-order method.
Belt-and-suspenders: enable "Read-Only API" in IB Gateway's API settings too.
Imports nothing from the decision trio; output lands in the gitignored
``data_processed/ibkr`` runtime dir.

**Field sourcing (chosen to avoid the flaky account-update subscription, which
blocks on this Gateway):**
- account scalars  → ``ib.accountSummary()``  (NetLiquidation, TotalCashValue,
  AvailableFunds, ExcessLiquidity, MaintMarginReq, BuyingPower)
- positions        → ``ib.portfolio()``  (structured contract fields — no string
  parsing; option ``averageCost`` is per-contract, divided by the multiplier)
- unrealized P&L   → summed from the portfolio (FX-normalized) — the account-level
  ``UnrealizedPnL`` tag requires the update stream, so it is derived instead
- FX (e.g. CAD→USD)→ a Forex price snapshot (``1 / USD<ccy>``), best-effort with
  a configurable fallback so a missing quote never crashes the morning job
- day-change       → withheld (null) by default — same gate as the cloud path,
  until reconciled against IBKR's own account Day P&L (``--enable-day-change``
  is intentionally absent; day-change is not derivable here without the PnL
  subscription, and is owned by the cloud path's reconciliation)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# Load the shared transform by path (sibling script; no package install needed).
_SPEC = importlib.util.spec_from_file_location(
    "ibkr_live_snapshot", _HERE / "ibkr_live_snapshot.py"
)
_live = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_live)
build_snapshot = _live.build_snapshot
load_reference = _live.load_reference

_MON_TITLE = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

# Tag → snapshot account field (the scalars accountSummary returns directly).
_SUMMARY_TAGS = {
    "NetLiquidation": "net_liquidation",
    "TotalCashValue": "total_cash_value",
    "AvailableFunds": "available_funds",
    "ExcessLiquidity": "excess_liquidity",
    "MaintMarginReq": "maintenance_margin",
    "BuyingPower": "buying_power",
}


def _syn_description(
    symbol: str, sec_type: str, right: str, strike: float, expiry: str, exchange: str
) -> str:
    """Synthesize the MCP-style ``contract_description`` the transform parses,
    from IB Gateway's structured contract fields. ``expiry`` is ``YYYYMMDD``."""
    if sec_type != "OPT":
        return symbol
    y, m, d = int(expiry[0:4]), int(expiry[4:6]), int(expiry[6:8])
    cp = "CALL" if str(right).upper().startswith("C") else "PUT"
    return f"{symbol} {_MON_TITLE[m]}{d}'{y % 100:02d} {strike:g} {cp} @{exchange or 'SMART'}"


def _fx_rates(ib, currencies: set[str], default_cad: float) -> dict[str, float]:
    """CAD/other → USD via a Forex snapshot (``1 / USD<ccy>``). Best-effort: a
    missing quote falls back to the configured default (or 1.0) and never raises."""
    from ib_insync import Forex

    fx = {"USD": 1.0}
    for ccy in sorted(c for c in currencies if c and c != "USD"):
        rate = None
        try:
            contract = Forex(f"USD{ccy}")
            ib.qualifyContracts(contract)
            tkr = ib.reqMktData(contract, "", False, False)
            ib.sleep(3.0)
            px = tkr.marketPrice()
            if not (px and px == px and px > 0):  # nan/0 guard
                px = tkr.close
            if px and px == px and px > 0:
                rate = 1.0 / px
            ib.cancelMktData(contract)
        except Exception:
            rate = None
        if rate is None:
            rate = default_cad if ccy == "CAD" else 1.0
            print(f"  [fx] {ccy}: no live quote -> fallback {rate:.6f}")
        fx[ccy] = rate
    return fx


def pull(host: str, port: int, client_id: int, *, default_cad: float) -> tuple[dict, dict, dict]:
    """Connect read-only and return the three MCP-shaped payloads
    (summary, balances, positions) that :func:`build_snapshot` consumes."""
    from ib_insync import IB

    ib = IB()
    # readonly=True: the session refuses order transmission at the API layer.
    ib.connect(host, port, clientId=client_id, readonly=True, timeout=30)
    try:
        acct = (ib.managedAccounts() or [""])[0]

        summary: dict = {}
        for av in ib.accountSummary(acct):
            field = _SUMMARY_TAGS.get(av.tag)
            if field and av.currency in ("", "USD", "BASE"):
                try:
                    summary[field] = float(av.value)
                except ValueError:
                    pass

        items = ib.portfolio(acct)
        currencies = {it.contract.currency for it in items}
        fx = _fx_rates(ib, currencies, default_cad)

        positions = []
        base_unrealized = 0.0
        for it in items:
            c = it.contract
            mult = int(c.multiplier) if getattr(c, "multiplier", "") else 1
            is_opt = c.secType == "OPT"
            avg = it.averageCost / mult if (is_opt and mult) else it.averageCost
            ccy = c.currency or "USD"
            base_unrealized += float(it.unrealizedPNL or 0.0) * fx.get(ccy, 1.0)
            positions.append(
                {
                    "contract_description": _syn_description(
                        c.symbol,
                        c.secType,
                        getattr(c, "right", ""),
                        float(getattr(c, "strike", 0) or 0),
                        str(getattr(c, "lastTradeDateOrContractMonth", "") or ""),
                        getattr(c, "exchange", "") or getattr(c, "primaryExchange", ""),
                    ),
                    "asset_class": c.secType,
                    "position": it.position,
                    "market_price": it.marketPrice,
                    "average_price": avg,
                    "unrealized_pnl": it.unrealizedPNL,
                    "currency": ccy,
                }
            )

        balances = {
            "balances": (
                [{"currency": c, "exchange_rate": r} for c, r in fx.items()]
                + [{"currency": "BASE", "unrealized_pnl": round(base_unrealized, 2)}]
            )
        }
        return summary, balances, {"positions": positions}
    finally:
        ib.disconnect()


def _default_out() -> Path:
    base = os.environ.get("SWE_IBKR_DATA_DIR")
    root = Path(base) if base else _REPO_ROOT / "data_processed" / "ibkr"
    return root / "portfolio_snapshot.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only IB Gateway → portfolio_snapshot.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--port", type=int, default=4001, help="IB Gateway socket (4001 live / 4002 paper)"
    )
    ap.add_argument("--client-id", type=int, default=17)
    ap.add_argument(
        "--out", default=None, help="snapshot path (default: $SWE_IBKR_DATA_DIR or repo data dir)"
    )
    ap.add_argument("--constituents", default=None)
    ap.add_argument(
        "--fx-cad", type=float, default=0.7167, help="CAD->USD fallback if no live forex quote"
    )
    args = ap.parse_args(argv)

    import json

    summary, balances, positions = pull(
        args.host, args.port, args.client_id, default_cad=args.fx_cad
    )
    snap = build_snapshot(
        summary,
        balances,
        positions,
        reference=load_reference(args.constituents),
        source="ibkr_gateway",
        include_day_change=False,  # gated null until reconciled (see module docstring)
    )
    out = Path(args.out) if args.out else _default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2)
    acct = snap["account"]
    print(
        f"wrote {out}  netLiq={acct['net_liquidation']:.0f}  "
        f"unrealized={acct['unrealized_pnl']:.0f}  positions={len(snap['positions'])}  "
        f"fx={snap['fx_rates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
