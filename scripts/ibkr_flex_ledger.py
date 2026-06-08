"""Phase 4 — re-key the wheel ledger to EXACT per-fill data from the IBKR Flex
'Trades' export (two contiguous Activity-Flex CSVs), reconstruct wheel legs
with real open/close dates, and refresh the viewer artifacts'
`wheel_ledger.json` (+ the `premium` series in `portfolio_history.json`).

Read-only / observational (CLAUDE.md §2/§3): never ranks, never calls
EVEngine.evaluate, never issues an EV-authority token. The decision-layer trio
(ev_engine / wheel_runner / candidate_dossier) is untouched. Real account data
stays gitignored (data_processed/ibkr); committed fixtures keep source:"fixture".

Pipeline (docs/IBKR_IMPORT.md → "Phase 4"):
  1. Ingest both CSVs: keep all of file A, append file B fills strictly after A's
     last timestamp (drops the boundary-day overlap; OrigTradeID is blank in this
     export, so the boundary is the dedup mechanism).
  2. FX: CAD option/stock fills → USD at the trade-date USD.CAD rate carried by
     the forex (CASH) fills (nearest-prior). Forex conversions are not trading
     P&L and are excluded from realized.
  3. Replay per underlying in real chronological order. Stock uses the IBKR
     **Open/Close** indicator to drive separate LONG and SHORT average-cost books
     (the account trades both sides) — order-robust and reproduces the p6 book.
     The long book is seeded with the ACAT-in transfer basis (from the PDF).
     Options realize per contract by net cash (a $0 close = expiry/assignment,
     premium kept); assignment vs expiry-OTM is detected from a stock fill at
     ~strike near expiry.
  4. Emit closed_positions (per option contract + per stock close) with exact
     dates; refresh history `premium`; reconcile and print.

Usage:
    python scripts/ibkr_flex_ledger.py "<A.csv>" "<B.csv>" --out data_processed/ibkr
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

AS_OF = "20260605"
_PDF_EXTRACT = Path(r"C:\Users\merty\AppData\Local\Temp\ibkr_inception_extract.txt")
_UNREALIZED_END = -41597.26  # p6 marks (FX-normalized) — from the snapshot artifact
_MTM_TARGET = 32637.69


def fnum(s):
    s = (s or "").strip().replace(",", "")
    return float(s) if s else 0.0


def iso8(dt):
    s = dt.split(";")[0]
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def days_between(a, b):
    try:
        return max(0, (date.fromisoformat(b) - date.fromisoformat(a)).days)
    except ValueError:
        return 0


def underlying(sym):
    return sym.split()[0] if " " in sym else sym


def load_fills(path_a, path_b):
    def load(p):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            return [r for r in csv.DictReader(fh) if r.get("DateTime", "").strip()]

    a, b = load(path_a), load(path_b)
    a_max = max(r["DateTime"] for r in a)
    b_kept = [r for r in b if r["DateTime"] > a_max]
    return a + b_kept, len(a), len(b), len(b) - len(b_kept), a_max


def build_fx(fills):
    pts = sorted(
        (r["DateTime"][:8], abs(fnum(r["TradePrice"])))
        for r in fills
        if r["AssetClass"] == "CASH" and r["Symbol"] == "USD.CAD" and fnum(r["TradePrice"])
    )

    def usdcad(date8):
        prior = [rate for d, rate in pts if d <= date8]
        return prior[-1] if prior else (pts[0][1] if pts else 1.394)

    return usdcad


def load_acat():
    raw = _PDF_EXTRACT.read_text(encoding="utf-8").splitlines()
    marks = [
        (i, int(re.search(r"PAGE (\d+)", ln).group(1)))
        for i, ln in enumerate(raw)
        if ln.startswith("===== PAGE ")
    ]
    pb = {
        p: (i + 1, (marks[k + 1][0] if k + 1 < len(marks) else len(raw)))
        for k, (i, p) in enumerate(marks)
    }
    p86 = [raw[i].strip() for i in range(pb[86][0], pb[86][1])]
    DATE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    TICK = re.compile(r"\(([A-Z]{1,5})\)\s*Quantity:\s*(\d+)")
    lots = defaultdict(lambda: [0.0, 0.0, None])  # ticker -> [qty, total_basis_usd, first_date]
    seen = set()
    i = 0
    while i < len(p86):
        if DATE.match(p86[i]) and i + 3 < len(p86) and p86[i + 1] == "ACAT In":
            m = TICK.search(p86[i + 2])
            key = (p86[i], p86[i + 2], p86[i + 3])
            if m and key not in seen:
                seen.add(key)
                di = p86[i]
                t = m.group(1)
                lots[t][0] += int(m.group(2))
                lots[t][1] += fnum(p86[i + 3])
                lots[t][2] = lots[t][2] or f"{di[6:10]}-{di[0:2]}-{di[3:5]}"
            i += 4
            continue
        i += 1
    return lots


class StockBook:
    """Separate long & short average-cost books for one underlying, driven by the
    IBKR Open/Close indicator. Emits a (entry, exit, realized, comm) tuple on each
    position-reducing close."""

    def __init__(self, seed_qty=0.0, seed_basis=0.0, seed_date=None):
        self.lq, self.lc, self.l_open = float(seed_qty), float(seed_basis), seed_date
        self.sq, self.sp, self.s_open = 0.0, 0.0, None
        self.closes = []  # (entry_iso, exit_iso, realized_usd, comm_usd)

    def event(self, q, price, comm, oc, iso):
        qty = abs(q)
        cpp = comm / qty if qty else 0.0  # commission per share
        if q > 0:  # BUY: cover short first (close), then open long
            close = min(qty, self.sq) if (not oc or oc.startswith("C")) else 0.0
            if close > 0:
                savg = self.sp / self.sq if self.sq > 1e-9 else price
                realized = close * (savg - price) + cpp * close
                self.sq -= close
                self.sp -= savg * close
                self.closes.append((self.s_open or iso, iso, realized, cpp * close))
            openq = qty - close
            if openq > 0:
                if self.lq <= 1e-9:
                    self.l_open = iso
                self.lq += openq
                self.lc += openq * price - cpp * openq  # comm increases basis (cpp<0)
        else:  # SELL: close long first, then open short
            close = min(qty, self.lq) if (not oc or oc.startswith("C")) else 0.0
            if close > 0:
                lavg = self.lc / self.lq if self.lq > 1e-9 else price
                realized = close * (price - lavg) + cpp * close
                self.lq -= close
                self.lc -= lavg * close
                self.closes.append((self.l_open or iso, iso, realized, cpp * close))
            openq = qty - close
            if openq > 0:
                if self.sq <= 1e-9:
                    self.s_open = iso
                self.sq += openq
                self.sp += openq * price

    @property
    def net_qty(self):
        return self.lq - self.sq

    @property
    def held_basis(self):
        return self.lc  # ends long for every name in p6


def build(path_a, path_b, out_dir):
    fills, na, nb, dropped, a_max = load_fills(path_a, path_b)
    usdcad = build_fx(fills)
    acat = load_acat()

    def usd(amount, ccy, date8):
        return amount if ccy == "USD" else amount / usdcad(date8)

    # assignment set: (und, expiry, strike_str, right) with a stock fill at ~strike
    # near expiry in the matching direction (put->buy, call->sell).
    stk_fills = [r for r in fills if r["AssetClass"] == "STK"]
    assigned = set()
    for r in (r for r in fills if r["AssetClass"] == "OPT"):
        und, strike, exp = underlying(r["Symbol"]), fnum(r["Strike"]), r["Expiry"]
        try:
            exp_iso = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
        except Exception:
            continue
        for s in stk_fills:
            if s["Symbol"] != und:
                continue
            if 0 <= days_between(exp_iso, iso8(s["DateTime"])) <= 3 and abs(
                fnum(s["TradePrice"]) - strike
            ) <= max(0.5, 0.01 * strike):
                if (r["Put/Call"] == "P" and fnum(s["Quantity"]) > 0) or (
                    r["Put/Call"] == "C" and fnum(s["Quantity"]) < 0
                ):
                    assigned.add((und, exp, r["Strike"], r["Put/Call"]))
                    break

    by_und = defaultdict(list)
    for r in fills:
        if r["AssetClass"] in ("OPT", "STK"):
            by_und[underlying(r["Symbol"])].append(r)

    closed = []
    cycles = defaultdict(lambda: defaultdict(int))
    recon_realized = 0.0
    held_basis_total = 0.0
    end_stock = {}

    for und, rows in by_und.items():
        rows.sort(key=lambda r: r["DateTime"])
        seed = acat.get(und, [0.0, 0.0, None])
        book = StockBook(seed[0], seed[1], seed[2])
        opt = {}

        for r in rows:
            dt8 = r["DateTime"][:8]
            iso = iso8(r["DateTime"])
            q = fnum(r["Quantity"])
            proc = usd(fnum(r["Proceeds"]), r["CurrencyPrimary"], dt8)
            comm = usd(fnum(r["IBCommission"]), r["CurrencyPrimary"], dt8)
            price = usd(fnum(r["TradePrice"]), r["CurrencyPrimary"], dt8)
            if r["AssetClass"] == "STK":
                book.event(q, price, comm, r["Open/CloseIndicator"], iso)
                continue
            right = r["Put/Call"]
            key = (right, r["Strike"], r["Expiry"])
            rec = opt.get(key)
            if rec is None:
                rec = opt[key] = {
                    "qty": 0.0,
                    "premium": 0.0,
                    "comm": 0.0,
                    "open_date": iso,
                    "close_date": iso,
                    "close_px": None,
                    "expiry": r["Expiry"],
                    "right": right,
                    "strike": fnum(r["Strike"]),
                    "strike_s": r["Strike"],
                }
            rec["qty"] += q
            rec["premium"] += proc
            rec["comm"] += comm
            rec["close_date"] = iso
            rec["close_px"] = fnum(r["TradePrice"])
            if abs(rec["qty"]) <= 1e-9:
                realized = rec["premium"] + rec["comm"]
                _emit_option(closed, cycles, und, rec, realized, assigned)
                recon_realized += realized
                del opt[key]

        # flush stock closes
        for entry, exit_, realized, comm in book.closes:
            closed.append(_rec(und, entry, exit_, "stock_sold", realized, comm, 0.0, 0.0, "stock"))
            recon_realized += realized
            cycles[und]["stock_sold"] += 1

        # leftover options: expired/assigned shorts (premium kept) or still-open (skip)
        for rec in list(opt.values()):
            if rec["expiry"] >= AS_OF and abs(rec["qty"]) > 1e-9:
                continue
            realized = rec["premium"] + rec["comm"]
            _emit_option(closed, cycles, und, rec, realized, assigned)
            recon_realized += realized

        if book.net_qty > 1e-9:
            end_stock[und] = (int(round(book.net_qty)), round(book.held_basis, 2))
            held_basis_total += book.held_basis

    closed.sort(key=lambda r: r["exit_date"])

    # ---- write artifacts ----
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wheel_ledger.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "as_of": "2026-06-05T21:36:00Z",
                "source": "ibkr_flex_import",
                "note": (
                    "Closed wheel legs re-keyed to EXACT IBKR Flex fills (per option contract "
                    "+ per stock close) with real open/close dates. net_pnl/premium in USD (CAD "
                    "fills FX-converted at trade-date USD.CAD; forex conversions excluded). Stock "
                    "uses Open/Close-driven long & short average-cost, long book seeded with ACAT "
                    "transfer basis. Decision trio untouched; real data gitignored."
                ),
                "closed_positions": closed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    hist_path = out / "portfolio_history.json"
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding="utf-8"))
        prem = defaultdict(float)
        for c in closed:
            prem[c["exit_date"][:7]] += c["put_premium"] + c["call_premium"]
        for p in hist["points"]:
            p["premium"] = round(prem.get(p["date"][:7], 0.0))
        if "[premium refreshed" not in hist.get("note", ""):
            hist["note"] = hist.get("note", "") + " [premium refreshed from exact Flex ledger]"
        hist_path.write_text(json.dumps(hist, indent=2), encoding="utf-8")

    # ---- reconciliation ----
    opt_recs = [c for c in closed if c["kind"] == "option"]
    stk_recs = [c for c in closed if c["kind"] == "stock"]
    opt_real = sum(c["net_pnl"] for c in opt_recs)
    stk_real = sum(c["net_pnl"] for c in stk_recs)
    wins = sum(1 for c in closed if c["net_pnl"] > 0)
    by_month = defaultdict(float)
    for c in closed:
        by_month[c["exit_date"][:7]] += c["net_pnl"]
    expected_stock = {
        "CLS": 500,
        "AMD": 100,
        "NVDA": 100,
        "TSM": 100,
        "WMT": 100,
        "CNQ": 100,
        "ENB": 100,
    }
    recon_mtm = recon_realized + _UNREALIZED_END

    print("=== PHASE 4 — exact Flex ledger reconciliation ===")
    print(
        f"fills: A={na} B={nb} (boundary dropped={dropped}, A_max={a_max}) -> combined={len(fills)}"
    )
    print(
        f"closed_positions: {len(closed)}  (option contracts={len(opt_recs)}, stock closes={len(stk_recs)})"
    )
    print(f"win-rate: {wins / len(closed):.4f} ({wins}/{len(closed)})")
    print(f"realized: option {opt_real:,.2f} + stock {stk_real:,.2f} = {recon_realized:,.2f}")
    print(
        f"realized YTD 2026: {sum(c['net_pnl'] for c in closed if c['exit_date'][:4] == '2026'):,.2f}"
    )
    print("\n-- ending stock (Open/Close long-short replay) vs p6 --")
    ok = True
    for k in sorted(set(list(end_stock) + list(expected_stock))):
        got = end_stock.get(k, (0, 0))
        exp = expected_stock.get(k)
        flag = "OK" if exp == got[0] else f"!! p6={exp}"
        if exp != got[0]:
            ok = False
        print(f"  {k:5} qty={got[0]:>5} basis={got[1]:>12,.2f}   {flag}")
    print(f"ending-book matches p6: {ok}")
    print(
        f"held stock basis total: {held_basis_total:,.2f}  (p6 cost of held stock = long mkt 310,524 region)"
    )
    print(
        f"\nreconstructed MTM = realized {recon_realized:,.0f} + unrealized {_UNREALIZED_END:,.0f} "
        f"= {recon_mtm:,.2f}   (statement MTM +{_MTM_TARGET:,.2f}, residual {recon_mtm - _MTM_TARGET:+,.2f})"
    )
    print("\nrealized P&L by month:")
    for ym in sorted(by_month):
        print(f"  {ym}: {by_month[ym]:>12,.2f}")
    print("\nexit_reason mix:", dict_counter(closed))
    return closed


def dict_counter(closed):
    c = defaultdict(int)
    for r in closed:
        c[r["exit_reason"]] += 1
    return dict(sorted(c.items()))


def _rec(ticker, entry, exit_, reason, realized, comm, putp, callp, kind):
    return {
        "ticker": ticker,
        "entry_date": entry,
        "exit_date": exit_,
        "exit_reason": reason,
        "hold_days": days_between(entry, exit_),
        "realized_pnl": round(realized, 2),
        "transaction_costs": round(-comm, 2),
        "net_pnl": round(realized, 2),
        "put_premium": round(putp, 2),
        "call_premium": round(callp, 2),
        "kind": kind,
    }


def _emit_option(closed, cycles, und, rec, realized, assigned):
    is_assigned = (und, rec["expiry"], rec["strike_s"], rec["right"]) in assigned
    px = rec["close_px"]
    expired = px is not None and abs(px) < 1e-9
    if rec["right"] == "P":
        reason = (
            "csp_assigned"
            if is_assigned
            else ("csp_expired_otm" if expired else "csp_bought_to_close")
        )
    else:
        reason = (
            "cc_called_away"
            if is_assigned
            else ("cc_expired_otm" if expired else "cc_bought_to_close")
        )
    putp = rec["premium"] if rec["right"] == "P" and rec["premium"] > 0 else 0.0
    callp = rec["premium"] if rec["right"] == "C" and rec["premium"] > 0 else 0.0
    exit_date = iso8(rec["expiry"]) if (expired or is_assigned) else rec["close_date"]
    closed.append(
        {
            "ticker": und,
            "entry_date": rec["open_date"],
            "exit_date": exit_date,
            "exit_reason": reason,
            "hold_days": days_between(rec["open_date"], exit_date),
            "realized_pnl": round(realized, 2),
            "transaction_costs": round(-rec["comm"], 2),
            "net_pnl": round(realized, 2),
            "put_premium": round(putp, 2),
            "call_premium": round(callp, 2),
            "strike": rec["strike"],
            "right": rec["right"],
            "kind": "option",
        }
    )
    cycles[und][reason] += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--out", default="data_processed/ibkr")
    a = ap.parse_args()
    build(a.csv_a, a.csv_b, a.out)
