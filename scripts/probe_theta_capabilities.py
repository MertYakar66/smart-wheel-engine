#!/usr/bin/env python3
"""
Probe a running Theta Terminal and report exactly which endpoints work on
your current subscription tier.

Why you want this
-----------------
Theta's tier structure is not visible from the outside — the same URL can
return 200 (subscribed), 403 (not subscribed), or 404 (not implemented).
This probe hits every endpoint the wheel engine could consume with cheap
test params and classifies each one.

Run it whenever you upgrade your Theta plan to see what opened up.

Output
------
Console: one line per endpoint, grouped by category, with status + row/col.
File:    ``data_processed/theta_capabilities.json`` (machine-readable).

Classifications
---------------
  ✓ OK        HTTP 200 with non-empty CSV body  → you can pull this data
  - EMPTY     HTTP 200 with empty body          → endpoint works but no test data
  × BLOCKED   HTTP 403                          → not on your subscription tier
  ? MISSING   HTTP 404                          → endpoint path wrong or removed
  ! ERROR     HTTP 4xx/5xx or timeout           → investigate

Usage
-----
    python scripts/probe_theta_capabilities.py                # default SPY
    python scripts/probe_theta_capabilities.py --symbol AAPL  # different probe
    python scripts/probe_theta_capabilities.py --verbose      # full URLs
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = _ROOT / "data_processed" / "theta_capabilities.json"

try:
    import requests
except ImportError:
    print("requests is required: pip install requests")
    sys.exit(2)


HOST = "127.0.0.1"
PORT = 25503
BASE = f"http://{HOST}:{PORT}"


@dataclass
class Probe:
    category: str
    name: str
    path: str
    params: dict
    notes: str = ""


@dataclass
class ProbeResult:
    category: str
    name: str
    status: str          # "OK", "EMPTY", "BLOCKED", "MISSING", "ERROR"
    http: int = 0
    rows: int = 0
    cols: list = field(default_factory=list)
    preview: str = ""
    notes: str = ""


def _check_socket() -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((HOST, PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _probes(symbol: str, exp: str) -> list[Probe]:
    """Build the probe matrix. Keep each probe cheap (small window, one
    strike) so a full sweep finishes in under a minute."""
    today = date.today()
    d7 = (today - timedelta(days=7)).strftime("%Y%m%d")
    d28 = (today - timedelta(days=28)).strftime("%Y%m%d")
    d365 = (today - timedelta(days=365)).strftime("%Y%m%d")
    now_str = today.strftime("%Y%m%d")

    return [
        # ---------------- Stock tier (free) ----------------
        Probe("stock", "list_roots",                "/v3/stock/list/roots", {}),
        Probe("stock", "history_eod",               "/v3/stock/history/eod",
              {"symbol": symbol, "start_date": d28, "end_date": now_str}),
        Probe("stock", "history_intraday_1h",       "/v3/stock/history/intraday",
              {"symbol": symbol, "start_date": d7, "end_date": now_str, "interval": "1h"}),
        Probe("stock", "snapshot_ohlc",             "/v3/stock/snapshot/ohlc", {"symbol": symbol}),
        Probe("stock", "snapshot_quote",            "/v3/stock/snapshot/quote", {"symbol": symbol}),
        Probe("stock", "history_dividend",          "/v3/stock/history/dividend",
              {"symbol": symbol, "start_date": d365, "end_date": now_str}),
        Probe("stock", "history_split",             "/v3/stock/history/split",
              {"symbol": symbol, "start_date": d365, "end_date": now_str}),

        # ---------------- Options tier ----------------
        Probe("option", "list_expirations",         "/v3/option/list/expirations", {"symbol": symbol}),
        Probe("option", "list_strikes",             "/v3/option/list/strikes",
              {"symbol": symbol, "expiration": exp}),
        Probe("option", "snapshot_greeks_first",    "/v3/option/snapshot/greeks/first_order",
              {"symbol": symbol, "expiration": exp}),
        Probe("option", "snapshot_quote",           "/v3/option/snapshot/quote",
              {"symbol": symbol, "expiration": exp}),
        Probe("option", "snapshot_open_interest",   "/v3/option/snapshot/open_interest",
              {"symbol": symbol, "expiration": exp}),
        Probe("option", "history_eod",              "/v3/option/history/eod",
              {"symbol": symbol, "expiration": exp,
               "start_date": d28, "end_date": now_str}),
        Probe("option", "history_quote",            "/v3/option/history/quote",
              {"symbol": symbol, "expiration": exp, "start_date": d7,
               "end_date": now_str, "interval": "1h"}),
        Probe("option", "history_trade",            "/v3/option/history/trade",
              {"symbol": symbol, "expiration": exp, "start_date": d7,
               "end_date": now_str, "interval": "1h"},
              "Intraday trade tape — enables dealer flow classification"),
        Probe("option", "history_volume",           "/v3/option/history/volume",
              {"symbol": symbol, "expiration": exp, "start_date": d28, "end_date": now_str}),
        Probe("option", "history_open_interest",    "/v3/option/history/open_interest",
              {"symbol": symbol, "expiration": exp, "start_date": d28, "end_date": now_str}),
        Probe("option", "history_greeks_iv",        "/v3/option/history/greeks/implied_volatility",
              {"symbol": symbol, "expiration": exp, "start_date": d28,
               "end_date": now_str, "interval": "1h"}),
        Probe("option", "history_greeks_first",     "/v3/option/history/greeks/first_order",
              {"symbol": symbol, "expiration": exp, "start_date": d28,
               "end_date": now_str, "interval": "1h"}),
        Probe("option", "bulk_snapshot",            "/v3/option/bulk_snapshot/greeks/first_order",
              {"symbol": symbol, "expiration": exp},
              "Single call for all strikes — huge speedup if available"),

        # ---------------- Index tier ----------------
        Probe("index", "list_roots",                "/v3/index/list/roots", {}),
        Probe("index", "snapshot_price_vix",        "/v3/index/snapshot/price", {"symbol": "VIX"}),
        Probe("index", "snapshot_price_skew",       "/v3/index/snapshot/price", {"symbol": "SKEW"}),
        Probe("index", "history_eod_vix",           "/v3/index/history/eod",
              {"symbol": "VIX", "start_date": d28, "end_date": now_str}),
        Probe("index", "history_eod_skew",          "/v3/index/history/eod",
              {"symbol": "SKEW", "start_date": d28, "end_date": now_str}),
        Probe("index", "history_ohlc_vix",          "/v3/index/history/ohlc",
              {"symbol": "VIX", "start_date": d28, "end_date": now_str,
               "interval": "1d"}),
        Probe("index", "history_price_vix9d",       "/v3/index/history/price",
              {"symbol": "VIX9D", "start_date": d28, "end_date": now_str}),
        Probe("index", "history_price_vvix",        "/v3/index/history/price",
              {"symbol": "VVIX", "start_date": d28, "end_date": now_str}),

        # ---------------- Futures tier ----------------
        Probe("future", "list_roots",               "/v3/future/list/roots", {}),
        Probe("future", "list_expirations_vx",      "/v3/future/list/expirations", {"symbol": "VX"}),
        Probe("future", "snapshot_ohlc_vx",         "/v3/future/snapshot/ohlc", {"symbol": "VX"},
              "VIX futures front month — unlocks UX1"),
        Probe("future", "history_eod_vx",           "/v3/future/history/eod",
              {"symbol": "VX", "start_date": d28, "end_date": now_str},
              "VIX futures EOD history"),
        Probe("future", "history_ohlc_vx",          "/v3/future/history/ohlc",
              {"symbol": "VX", "start_date": d28, "end_date": now_str, "interval": "1d"}),
    ]


def run_probe(p: Probe, timeout: float = 10.0) -> ProbeResult:
    params = {**p.params, "format": "csv"}
    url = f"{BASE}{p.path}"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return ProbeResult(p.category, p.name, "ERROR", 0, preview=str(e)[:120], notes=p.notes)

    if resp.status_code == 403:
        return ProbeResult(p.category, p.name, "BLOCKED", 403,
                           preview="not on subscription tier", notes=p.notes)
    if resp.status_code == 404:
        return ProbeResult(p.category, p.name, "MISSING", 404,
                           preview="endpoint does not exist", notes=p.notes)
    if not (200 <= resp.status_code < 300):
        body = resp.text.strip().splitlines()[:1]
        return ProbeResult(p.category, p.name, "ERROR", resp.status_code,
                           preview=(body[0] if body else "")[:120], notes=p.notes)

    text = resp.text.strip()
    if not text:
        return ProbeResult(p.category, p.name, "EMPTY", 200, notes=p.notes)
    if text.startswith(("<", "We have upgraded")):
        return ProbeResult(p.category, p.name, "ERROR", 200, preview=text[:120], notes=p.notes)

    try:
        import pandas as pd
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        return ProbeResult(p.category, p.name, "ERROR", resp.status_code,
                           preview=f"parse: {e}"[:120], notes=p.notes)

    if df.empty:
        return ProbeResult(p.category, p.name, "EMPTY", 200,
                           cols=list(df.columns), notes=p.notes)
    return ProbeResult(p.category, p.name, "OK", 200,
                       rows=len(df), cols=list(df.columns)[:10], notes=p.notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY", help="Probe symbol (default SPY)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    if not _check_socket():
        print(f"Theta Terminal not reachable at {BASE}.")
        print("Start the Terminal (the tray app) and re-run.")
        return 2

    # Find a real expiration close to 35 DTE to make option probes realistic.
    import pandas as pd
    exp = None
    try:
        r = requests.get(f"{BASE}/v3/option/list/expirations",
                         params={"symbol": args.symbol, "format": "csv"}, timeout=10)
        if r.ok:
            df = pd.read_csv(io.StringIO(r.text))
            col = next((c for c in ("expiration", "date") if c in df.columns), None)
            if col is not None and not df.empty:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                target = pd.Timestamp.now().normalize() + pd.Timedelta(days=35)
                best = dates.iloc[(dates - target).abs().argsort().iloc[0]]
                exp = best.strftime("%Y%m%d")
    except Exception:
        pass
    if not exp:
        # Fallback: third Friday of next month
        nxt = date.today().replace(day=1) + timedelta(days=32)
        nxt = nxt.replace(day=1)
        # first Friday
        first_friday = nxt + timedelta(days=(4 - nxt.weekday()) % 7)
        exp = (first_friday + timedelta(days=14)).strftime("%Y%m%d")
        print(f"Could not enumerate expirations — falling back to synthetic {exp}")

    print()
    print("=" * 90)
    print(f" Theta capability probe   base={BASE}   symbol={args.symbol}   expiry={exp}")
    print("=" * 90)

    probes = _probes(args.symbol, exp)
    results: list[ProbeResult] = []
    t0 = time.perf_counter()

    current_cat = ""
    for p in probes:
        if p.category != current_cat:
            current_cat = p.category
            print(f"\n--- {p.category.upper()} ---")
        r = run_probe(p)
        results.append(r)
        tag = {"OK": "✓", "EMPTY": "-", "BLOCKED": "×", "MISSING": "?", "ERROR": "!"}[r.status]
        detail = f"{r.status:<7} {r.http:>3}"
        if r.status == "OK":
            detail += f"  rows={r.rows:<4}  cols={','.join(r.cols[:5])}"
        else:
            detail += f"  {r.preview[:50]}"
        line = f"  {tag} {p.name:<30} {detail}"
        if p.notes and r.status in ("OK", "EMPTY"):
            line += f"   # {p.notes[:50]}"
        if args.verbose:
            line += f"\n      {p.path}  {p.params}"
        print(line, flush=True)

    elapsed = time.perf_counter() - t0

    # Summary
    by_status = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    print()
    print("=" * 90)
    print(f" Summary  ({elapsed:.1f}s)")
    for s in ("OK", "EMPTY", "BLOCKED", "MISSING", "ERROR"):
        if s in by_status:
            print(f"   {s:<8} {by_status[s]}")
    print("=" * 90)

    # What the caller can actually use
    usable_by_cat: dict[str, list[str]] = {}
    for r in results:
        if r.status == "OK":
            usable_by_cat.setdefault(r.category, []).append(r.name)
    if usable_by_cat:
        print("\nYou can pull these data types RIGHT NOW:")
        for cat, names in usable_by_cat.items():
            print(f"  {cat}: {', '.join(names)}")

    blocked = [r for r in results if r.status == "BLOCKED"]
    if blocked:
        print("\nBlocked by tier (would need an upgrade):")
        for r in blocked:
            print(f"  {r.category}/{r.name}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"probed_at": datetime.utcnow().isoformat(),
         "base": BASE, "symbol": args.symbol, "expiry": exp,
         "results": [asdict(r) for r in results]},
        indent=2,
    ))
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
