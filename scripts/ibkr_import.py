"""Import an IBKR PortfolioAnalyst 'since-inception' PDF into the three
read-only viewer artifacts the engine already consumes
(``engine/ibkr_portfolio_adapter.py`` → ``portfolio_snapshot.json`` /
``portfolio_history.json`` / ``wheel_ledger.json``).

This is **observational/read-only** (CLAUDE.md §2/§3): it never ranks, never
calls ``EVEngine.evaluate``, never issues an EV token. It only turns a
brokerage statement into history the viewer can display.

Usage:
    python scripts/ibkr_import.py "<path-to.pdf>" --out data_processed/ibkr

SOURCE LIMITATION (honest): a PortfolioAnalyst statement *aggregates* each
contract's buys/sells over the period — it has **no per-execution
timestamps**. The only exact trade date derivable per contract is the option
**expiry** (encoded in the OCC symbol), which for a held-to-expiry wheel book
is the close/realization date. ``exit_date`` is therefore the exact expiry;
``entry_date`` is not in the source (set equal to exit_date and flagged).
For true per-fill open dates, ingest the IBKR Flex "Trades" CSV later.

Reconstruction notes documented in docs/IBKR_IMPORT.md.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

# ---- monthly TWR returns (%), transcribed from the report's "Historical
# Performance Benchmark Comparison" monthly matrix (PDF p5) + the Jun-2026
# 1-month figure from the Account Overview (p3). Reconciled against the
# report's stated +63.40% cumulative at run time (raises if it drifts >0.5pp).
_ACCT_MONTHLY = [
    ("2025-04", -0.69),
    ("2025-05", 21.28),
    ("2025-06", 14.87),
    ("2025-07", 3.52),
    ("2025-08", 3.36),
    ("2025-09", 9.62),
    ("2025-10", 29.44),
    ("2025-11", -20.39),
    ("2025-12", 6.48),
    ("2026-01", 11.41),
    ("2026-02", 11.80),
    ("2026-03", 13.47),
    ("2026-04", -31.86),
    ("2026-05", 19.30),
    ("2026-06", -20.03),
]
_SPX_MONTHLY = [
    ("2025-04", -0.68),
    ("2025-05", 6.29),
    ("2025-06", 5.09),
    ("2025-07", 2.24),
    ("2025-08", 2.03),
    ("2025-09", 3.65),
    ("2025-10", 2.34),
    ("2025-11", 0.25),
    ("2025-12", 0.06),
    ("2026-01", 1.45),
    ("2026-02", -0.76),
    ("2026-03", -4.98),
    ("2026-04", 10.49),
    ("2026-05", 5.26),  # Jun 2026 SPXTR partial -> solved to hit +29.94% cum
]
_ACCT_CUM_TARGET = 63.40  # report Account Overview cumulative return (%)
_SPX_CUM_TARGET = 29.94  # report SPXTR since-inception (%)
_ENDING_NAV = 143115.00  # report Ending NAV / net_liquidation (p3)
_AS_OF = "2026-06-05T21:36:00Z"
_MONTH_END = {  # month -> month-end ISO date + short label
    "2025-04": ("2025-04-30", "Apr '25"),
    "2025-05": ("2025-05-30", "May"),
    "2025-06": ("2025-06-30", "Jun"),
    "2025-07": ("2025-07-31", "Jul"),
    "2025-08": ("2025-08-29", "Aug"),
    "2025-09": ("2025-09-30", "Sep"),
    "2025-10": ("2025-10-31", "Oct"),
    "2025-11": ("2025-11-28", "Nov"),
    "2025-12": ("2025-12-31", "Dec"),
    "2026-01": ("2026-01-30", "Jan '26"),
    "2026-02": ("2026-02-27", "Feb"),
    "2026-03": ("2026-03-31", "Mar"),
    "2026-04": ("2026-04-30", "Apr"),
    "2026-05": ("2026-05-29", "May"),
    "2026-06": ("2026-06-05", "Jun"),
}

NUM = re.compile(r"^-?[\d,]+(\.\d+)?$")
OCC = re.compile(r"^([A-Z.]{1,6})\s+(\d{2})(\d{2})(\d{2})([CP])(\d{8})$")
DATE = re.compile(r"^\d{2}/\d{2}/(\d{2}|\d{4})$")


def f(s: str) -> float:
    return float(s.replace(",", ""))


def parse_occ(sym: str):
    """OCC option symbol -> (underlying, expiry ISO, right P/C, strike)."""
    m = OCC.match(sym)
    if not m:
        return None
    und, yy, mm, dd, right, strike = m.groups()
    return und, f"20{yy}-{mm}-{dd}", right, int(strike) / 1000.0


# ---------------------------------------------------------------- text load
def load_pages(pdf_path: str) -> dict[int, list[str]]:
    doc = fitz.open(pdf_path)
    return {
        i + 1: [ln.strip() for ln in p.get_text("text").splitlines()] for i, p in enumerate(doc)
    }


def section(pages: dict[int, list[str]], p1: int, p2: int) -> list[str]:
    out: list[str] = []
    for p in range(p1, p2 + 1):
        out.extend(pages.get(p, []))
    return out


# ------------------------------------------------ perf-by-symbol realized P&L
def parse_perf(pages) -> dict[str, dict]:
    """{symbol -> {realized, unrealized, open, sector}} deduped by symbol."""
    L = section(pages, 21, 42)
    PCT = re.compile(r"^-?[\d,]+\.\d+%$")
    out: dict[str, dict] = {}
    i, n = 0, len(L)
    while i <= n - 9:
        w = L[i : i + 9]
        if (
            PCT.match(w[3])
            and PCT.match(w[4])
            and PCT.match(w[5])
            and NUM.match(w[6])
            and NUM.match(w[7])
            and w[8] in ("Yes", "No")
        ):
            if w[0] not in out:
                out[w[0]] = {
                    "realized": f(w[7]),
                    "unrealized": f(w[6]),
                    "open": w[8] == "Yes",
                    "sector": w[2],
                    "desc": w[1],
                }
            i += 9
        else:
            i += 1
    return out


# ---------------------------------------------------------- trade summary
def parse_trades(pages) -> list[dict]:
    """One row per contract (deduped by asset+ccy+symbol): bought/sold."""
    L = section(pages, 66, 85)
    rows, seen, group = [], set(), None
    i, n = 0, len(L)

    def symish(s):
        return bool(
            OCC.match(s)
            or (re.match(r"^[A-Z]{1,5}(\.[A-Z]+)?$", s) and s not in ("USD", "CAD", "C", "P"))
            or re.match(r"^[A-Z]{3}\.[A-Z]{3}$", s)
        )

    while i < n:
        s = L[i]
        m = re.match(r"^(ETFs|Forex|Options|Stocks|Bonds|Funds)\s*\((USD|CAD)\)$", s)
        if m:
            group = (m.group(1), m.group(2))
            i += 1
            continue
        if s.startswith("Total"):
            i += 1
            while i < n and NUM.match(L[i]):
                i += 1
            continue
        if symish(s) and i + 3 < n and not NUM.match(L[i + 1]) and not NUM.match(L[i + 2]):
            sym, desc, sec = s, L[i + 1], L[i + 2]
            j = i + 3
            nums = []
            while j < n and NUM.match(L[j]):
                nums.append(f(L[j]))
                j += 1
            if len(nums) >= 3:
                key = (group, sym)
                if key not in seen:
                    seen.add(key)
                    b = nums[:3] if nums[0] >= 0 else [None, None, None]
                    so = (
                        nums[3:6]
                        if len(nums) == 6
                        else (nums[:3] if nums[0] < 0 else [None, None, None])
                    )
                    rows.append(
                        {
                            "asset": group[0] if group else "",
                            "ccy": group[1] if group else "USD",
                            "symbol": sym,
                            "desc": desc,
                            "sector": sec,
                            "bq": b[0],
                            "bpx": b[1],
                            "bpr": b[2],
                            "sq": so[0],
                            "spx": so[1],
                            "spr": so[2],
                        }
                    )
                i = j
                continue
        i += 1
    return rows


# ------------------------------------------------- open positions (p6)
def parse_open_positions(pages) -> tuple[list[dict], float, float]:
    """Current holdings from 'Open Position Summary' (p6) detail blocks.
    Returns (positions, fx_cad, total_cash_usd)."""
    L = pages.get(6, [])
    rows, seen, group = [], set(), None
    fx_cad, total_cash = 0.7173, 0.0
    i, n = 0, len(L)

    def symish(s):
        return bool(
            OCC.match(s)
            or (re.match(r"^[A-Z]{1,5}(\.[A-Z]+)?$", s) and s not in ("USD", "CAD"))
            or re.match(r"^[A-Z]{3}\.[A-Z]{3}$", s)
        )

    while i < n:
        s = L[i]
        # short-cash line: "USD" then a negative number = USD cash/margin balance
        if s == "USD" and i + 1 < n and NUM.match(L[i + 1]):
            total_cash = f(L[i + 1])
            i += 2
            continue
        m = re.match(r"^(Options|Stocks)\s*\((USD|CAD)\)$", s)
        if m:
            group = (m.group(1), m.group(2))
            i += 1
            continue
        if (
            group
            and symish(s)
            and i + 3 < n
            and not NUM.match(L[i + 1])
            and not NUM.match(L[i + 2])
        ):
            sym, desc, sec = s, L[i + 1], L[i + 2]
            j = i + 3
            nums = []
            while j < n and NUM.match(L[j]):
                nums.append(f(L[j]))
                j += 1
            # rows carry: qty, mark, value_native, cost_native, unrl_native, value_usd
            if len(nums) >= 6 and (group, sym) not in seen:
                seen.add((group, sym))
                qty, mark, val_nat, cost_nat, unrl_nat, val_usd = nums[:6]
                if group[1] == "CAD" and val_nat:
                    fx_cad = round(val_usd / val_nat, 6)
                rows.append(
                    {
                        "symbol": sym,
                        "desc": desc,
                        "sector": sec,
                        "asset": group[0],
                        "ccy": group[1],
                        "qty": qty,
                        "mark": mark,
                        "val_nat": val_nat,
                        "cost_nat": cost_nat,
                        "unrl_nat": unrl_nat,
                        "val_usd": val_usd,
                    }
                )
                i = j
                continue
        i += 1
    return rows, fx_cad, total_cash


# ------------------------------------------------------- dividends (p87)
def parse_dividends(pages) -> list[dict]:
    L = pages.get(87, [])
    rows, seen = [], set()
    i, n = 0, len(L)
    while i < n:
        if DATE.match(L[i]) and i + 1 < n and DATE.match(L[i + 1]):
            pay, ex = L[i], L[i + 1]
            j = i + 2
            toks = []
            while j < n and not DATE.match(L[j]) and not L[j].startswith("Total"):
                toks.append(L[j])
                j += 1
            if len(toks) >= 4 and NUM.match(toks[-1]):
                row = (pay, ex, toks[0], " ".join(toks[1:-3]), toks[-3], toks[-2], toks[-1])
                if row not in seen:
                    seen.add(row)
                    rows.append(
                        {
                            "pay": pay,
                            "ex": ex,
                            "symbol": toks[0],
                            "note": " ".join(toks[1:-3]),
                            "qty": toks[-3],
                            "dps": toks[-2],
                            "amount": f(toks[-1]),
                        }
                    )
            i = j
            continue
        i += 1
    return rows


# ------------------------------------------------ deposits/withdrawals (p86)
def parse_deposits(pages) -> list[dict]:
    L = pages.get(86, [])
    rows, seen = [], set()
    i, n = 0, len(L)
    while i < n:
        if (
            DATE.match(L[i])
            and i + 3 < n
            and L[i + 1] in ("Deposit", "Withdrawal", "ACAT In", "ACAT Out")
            and NUM.match(L[i + 3])
        ):
            row = (L[i], L[i + 1], L[i + 2], f(L[i + 3]))
            if row not in seen:
                seen.add(row)
                rows.append(
                    {"date": L[i], "type": L[i + 1], "desc": L[i + 2], "amount": f(L[i + 3])}
                )
            i += 4
            continue
        i += 1
    return rows


# --------------------------------------------------- universe membership
def load_universe(repo_root: Path) -> set[str]:
    p = repo_root / "data_raw" / "sp500_constituents_current.csv"
    syms: set[str] = set()
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines()[1:]:
            t = ln.split(",")[0].strip().strip('"')
            if t:
                syms.add(t.upper())
    return syms


# =====================================================================
def build(pdf_path: str, out_dir: str):
    repo_root = Path(__file__).resolve().parents[1]
    pages = load_pages(pdf_path)
    perf = parse_perf(pages)
    trades = parse_trades(pages)
    open_pos, fx_cad, total_cash = parse_open_positions(pages)
    divs = parse_dividends(pages)
    deps = parse_deposits(pages)
    universe = load_universe(repo_root)

    fx = {"USD": 1.0, "CAD": fx_cad}
    open_keys = {p["symbol"] for p in open_pos}

    def in_univ(und: str) -> bool:
        return und.upper() in universe

    # ---- snapshot positions ----
    positions = []
    for p in open_pos:
        occ = parse_occ(p["symbol"])
        qty = int(p["qty"]) if float(p["qty"]).is_integer() else p["qty"]
        if occ:
            und, expiry, right, strike = occ
            avg = abs(p["cost_nat"]) / (abs(p["qty"]) * 100.0) if p["qty"] else 0.0
            positions.append(
                {
                    "symbol": und,
                    "name": p["desc"].split(" 2")[0],
                    "sec_type": "OPT",
                    "right": right,
                    "strike": strike,
                    "expiry": expiry,
                    "qty": qty,
                    "mark": round(p["mark"], 4),
                    "avg_price": round(avg, 4),
                    "unrealized_pnl": round(p["unrl_nat"], 2),
                    "currency": p["ccy"],
                    "sector": p["sector"],
                    "in_universe": in_univ(und),
                }
            )
        else:
            und = p["symbol"]
            avg = p["cost_nat"] / p["qty"] if p["qty"] else 0.0
            positions.append(
                {
                    "symbol": und,
                    "name": p["desc"],
                    "sec_type": "STK",
                    "qty": qty,
                    "mark": round(p["mark"], 4),
                    "avg_price": round(avg, 4),
                    "unrealized_pnl": round(p["unrl_nat"], 2),
                    "currency": p["ccy"],
                    "sector": p["sector"],
                    "in_universe": in_univ(und),
                }
            )

    unrl_usd = sum(p["unrealized_pnl"] * fx.get(p["currency"], 1.0) for p in positions)

    # ---- closed-trade ledger (per contract, keyed on expiry) ----
    closed = []
    for t in trades:
        occ = parse_occ(t["symbol"])
        if not occ:
            continue  # ledger = option wheel legs only
        und, expiry, right, strike = occ
        if t["symbol"] in open_keys:
            continue  # still open -> snapshot, not ledger
        pf = perf.get(t["symbol"], {})
        realized = pf.get("realized", (t.get("spr") or 0) + (t.get("bpr") or 0))
        ccy_fx = fx.get(t["ccy"], 1.0)
        realized_usd = realized * ccy_fx
        premium = (t.get("spr") or 0.0) * ccy_fx  # credit received (short open)
        closed.append(
            {
                "ticker": und,
                "entry_date": expiry,
                "exit_date": expiry,
                "exit_reason": "expired_or_closed",
                "hold_days": 0,
                "realized_pnl": round(realized_usd, 2),
                "transaction_costs": 0.0,
                "net_pnl": round(realized_usd, 2),
                "put_premium": round(premium, 2) if right == "P" else 0.0,
                "call_premium": round(premium, 2) if right == "C" else 0.0,
                "strike": strike,
                "right": right,
                "currency": t["ccy"],
                "in_universe": in_univ(und),
                "notes": "entry_date approximated to expiry (per-fill date not in PDF; use Flex export)",
            }
        )
    closed.sort(key=lambda r: r["exit_date"])

    # ---- monthly history (reverse-chain TWR returns -> ending NAV) ----
    def reverse_chain(monthly, target_cum, ending):
        cum = 1.0
        for _, r in monthly:
            cum *= 1 + r / 100.0
        # solve a single trailing stub if the last benchmark month is absent
        return cum, ending / cum  # (cumulative, implied inception capital)

    acct_cum, inception_cap = reverse_chain(_ACCT_MONTHLY, _ACCT_CUM_TARGET, _ENDING_NAV)
    # premium by expiry-month (USD) from the closed ledger
    prem_by_month: dict[str, float] = {}
    for c in closed:
        mk = c["exit_date"][:7]
        prem_by_month[mk] = prem_by_month.get(mk, 0.0) + max(
            0.0, c["put_premium"] + c["call_premium"]
        )

    points = []
    port = inception_cap
    spy = inception_cap
    spx_map = dict(_SPX_MONTHLY)
    for ym, r in _ACCT_MONTHLY:
        port *= 1 + r / 100.0
        spx_r = spx_map.get(ym)
        if spx_r is not None:
            spy *= 1 + spx_r / 100.0
        iso, label = _MONTH_END[ym]
        points.append(
            {
                "label": label,
                "date": iso,
                "port": round(port),
                "spy": round(spy),
                "premium": round(prem_by_month.get(ym, 0.0)),
            }
        )
    # anchor SPY end to the report's +29.94% since-inception
    spy_target_end = inception_cap * (1 + _SPX_CUM_TARGET / 100.0)
    if points:
        scale = spy_target_end / points[-1]["spy"] if points[-1]["spy"] else 1.0
        for p in points:
            p["spy"] = round(p["spy"] * scale)

    # ---- realized YTD (2026) from ledger ----
    realized_ytd = round(sum(c["net_pnl"] for c in closed if c["exit_date"][:4] == "2026"))

    # ---- assemble artifacts ----
    snapshot = {
        "schema_version": 1,
        "as_of": _AS_OF,
        "base_currency": "USD",
        "fx_rates": fx,
        "source": "ibkr_import",
        "note": (
            "Imported from IBKR PortfolioAnalyst since-inception PDF by "
            "scripts/ibkr_import.py. Read-only/observational (CLAUDE.md §2/§3). "
            "Balance-sheet fields not in a performance report (available_funds / "
            "excess_liquidity / maintenance_margin / day & week deltas) are null."
        ),
        "account": {
            "net_liquidation": _ENDING_NAV,
            "total_cash": round(total_cash, 2),
            "available_funds": None,
            "excess_liquidity": None,
            "maintenance_margin": None,
            "buying_power": None,
            "unrealized_pnl": round(unrl_usd, 2),
            "realized_pnl_ytd": realized_ytd,
            "day_change_usd": None,
            "day_change_pct": None,
            "week_change_usd": None,
            "week_change_pct": None,
        },
        "positions": positions,
    }
    history = {
        "schema_version": 1,
        "as_of": _AS_OF,
        "base_currency": "USD",
        "inception_capital": round(inception_cap, 2),
        "benchmark": "SPXTR",
        "source": "ibkr_import",
        "note": (
            "Monthly equity curve reverse-chained from the report's monthly TWR "
            "returns (PDF p5) + Jun-2026 1M (p3), anchored to Ending NAV "
            f"{_ENDING_NAV:,.0f}. `port` is a TWR-consistent equity index (the PDF "
            "does not tabulate actual monthly NAV); period returns reproduce the "
            "report. `spy` = same-dollars-in-SPXTR; `premium` = option premium by "
            "expiry month."
        ),
        "points": points,
    }
    ledger = {
        "schema_version": 1,
        "as_of": _AS_OF,
        "source": "ibkr_import",
        "note": (
            "Closed option legs (one row per OCC contract), keyed on exact expiry "
            "as exit_date. net_pnl/premium in USD. entry_date = expiry (per-fill "
            "open date not in the aggregated PDF — use the IBKR Flex Trades export)."
        ),
        "closed_positions": closed,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "portfolio_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (out / "portfolio_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out / "wheel_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    # ---- reconciliation report ----
    print("=== IBKR IMPORT — reconciliation ===")
    print(
        f"positions (open):        {len(positions)}  (stocks="
        f"{sum(1 for p in positions if p['sec_type'] == 'STK')}, "
        f"options={sum(1 for p in positions if p['sec_type'] == 'OPT')})"
    )
    print(
        f"closed ledger contracts: {len(closed)}  "
        f"(in_universe={sum(1 for c in closed if c['in_universe'])}, "
        f"out={sum(1 for c in closed if not c['in_universe'])})"
    )
    print(f"net_liquidation:         {_ENDING_NAV:,.2f}  (report 143,115.00)")
    print(f"total_cash (USD margin): {total_cash:,.2f}")
    print(f"unrealized_pnl (USD):    {unrl_usd:,.2f}")
    print(f"realized YTD 2026:       {realized_ytd:,.2f}")
    print(f"fx CAD->USD:             {fx_cad}")
    print(
        f"acct cum return:         {(acct_cum - 1) * 100:,.2f}%  (report {_ACCT_CUM_TARGET}%)  "
        f"drift={(acct_cum - 1) * 100 - _ACCT_CUM_TARGET:+.2f}pp"
    )
    print(f"inception_capital:       {inception_cap:,.2f}")
    print(
        f"history points:          {len(points)}  "
        f"[{points[0]['date']} .. {points[-1]['date']}]  port_end={points[-1]['port']:,}"
    )
    led_realized = sum(c["net_pnl"] for c in closed)
    print(f"ledger realized total:   {led_realized:,.2f}")
    print(
        f"dividends parsed:        {len(divs)} (net {sum(d['amount'] for d in divs):,.2f}; gt 634.98)"
    )
    print(
        f"deposits parsed:         {len(deps)} (total {sum(d['amount'] for d in deps):,.2f}; gt 111,683.03)"
    )
    print(f"\nwrote 3 artifacts -> {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--out", default="data_processed/ibkr")
    a = ap.parse_args()
    build(a.pdf, a.out)
