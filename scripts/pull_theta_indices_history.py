#!/usr/bin/env python3
"""
Pull daily OHLC history for the VIX family directly from Theta Terminal.

``pull_vol_indices.py`` already covers these via yfinance fallback, but
Theta is the authoritative CBOE feed — cleaner data, end-of-day timestamps
match exchange session close, no delisting gaps. Run this after
``probe_theta_capabilities.py`` confirms ``index/history_*`` returns 200.

Symbols tried (each can individually fail — blocked tier, new symbol, etc):

    VIX, VIX9D, VIX3M, VIX6M, SKEW, VVIX, VXN, MOVE, OVX, GVZ

Endpoints tried, in order, per symbol:
    /v3/index/history/eod
    /v3/index/history/ohlc
    /v3/index/history/price

First one that returns rows wins.

Output
------
Appends to ``data_processed/vol_indices.parquet`` with ``source="theta"``.
Downstream readers (regime detector, smoke test) already understand this
file — replacing yahoo rows with theta rows is transparent.

Usage
-----
    # Full 5y refresh from Theta
    python scripts/pull_theta_indices_history.py --years 5

    # Only what's missing since last Theta write
    python scripts/pull_theta_indices_history.py --incremental

    # A specific symbol
    python scripts/pull_theta_indices_history.py --symbols SKEW VVIX --years 2
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import socket
import sys
import time
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from engine.theta_connector import ThetaConnector  # noqa: E402

logger = logging.getLogger(__name__)
OUT_LONG = _ROOT / "data_processed" / "vol_indices.parquet"
OUT_WIDE = _ROOT / "data_processed" / "vol_indices_wide.parquet"

# MOVE: ICE rates index, not in Theta v3 coverage. yfinance fallback only —
# see pull_vol_indices.py.
DEFAULT_SYMBOLS = (
    "VIX",
    "VIX9D",
    "VIX3M",
    "VIX6M",
    "SKEW",
    "VVIX",
    "VXN",
    "OVX",
    "GVZ",
)

ENDPOINTS = (
    "/v3/index/history/eod",
    "/v3/index/history/ohlc",
    "/v3/index/history/price",
)

# Theta's /v3/index/history/eod caps single requests at 365 days. Use a
# 10-day buffer so off-by-one in date math can't trip the cap.
CHUNK_DAYS = 350

# Theta returns diagnostic plain-text bodies on these status codes:
#   400 — window-cap ("Too many days between start and end date")
#   403 — tier-gate ("Requesting index history requiring a STANDARD …")
#   472 — coverage miss ("No data found for your request")
# The shared ThetaConnector swallows 400/403/4xx silently (engine/
# theta_connector.py:155); we issue the requests directly through the
# connector's pooled session so the body text reaches the operator.
_DIAGNOSTIC_STATUSES = (400, 403, 472)
_BASE_URL = "http://127.0.0.1:25503"


def _theta_up(host: str = "127.0.0.1", port: int = 25503) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _pull(conn: ThetaConnector, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Pull ``symbol`` history across [``start``, ``end``].

    Splits the range into ``CHUNK_DAYS`` windows so the request never trips
    Theta's 365-day-per-call cap on ``/v3/index/history/*``. Tries the
    endpoints in ``ENDPOINTS`` order and returns the first endpoint that
    yields any data; older windows that tier-gate are surfaced to stdout
    but do not stop the puller from collecting newer windows that are
    inside our subscription's lookback budget.

    HTTP requests go through ``conn._session`` directly rather than
    ``conn._fetch`` so 400/403/472 diagnostic bodies are visible — the
    shared connector returns an empty DataFrame on those statuses without
    surfacing the text.
    """
    session = conn._session
    for ep in ENDPOINTS:
        ep_frames: list[pd.DataFrame] = []
        gate_messages: list[str] = []
        cur = start
        while cur <= end:
            nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
            params = {
                "symbol": symbol,
                "start_date": cur.strftime("%Y%m%d"),
                "end_date": nxt.strftime("%Y%m%d"),
                "format": "csv",
            }
            if "ohlc" in ep:
                params["interval"] = "1d"

            try:
                resp = session.get(f"{_BASE_URL}{ep}", params=params, timeout=30)
            except Exception as e:
                logger.debug("%s %s [%s..%s]: %s", symbol, ep, cur, nxt, e)
                cur = nxt + timedelta(days=1)
                continue

            body = (resp.text or "").strip()
            if resp.status_code in _DIAGNOSTIC_STATUSES and body:
                gate_messages.append(f"{cur}..{nxt}: {body}")
                cur = nxt + timedelta(days=1)
                continue
            if resp.status_code != 200 or not body:
                cur = nxt + timedelta(days=1)
                continue

            try:
                df = pd.read_csv(io.StringIO(body))
            except Exception as e:
                logger.debug("%s %s [%s..%s] parse: %s", symbol, ep, cur, nxt, e)
                cur = nxt + timedelta(days=1)
                continue

            if not df.empty:
                ep_frames.append(df)
            cur = nxt + timedelta(days=1)

        # Surface gate / no-data messages exactly as Theta returned them so
        # the operator can decide whether to upgrade their tier.
        for msg in gate_messages:
            print(f"  {symbol:<7} GATE  {msg[:140]}", flush=True)

        if not ep_frames:
            continue

        df = pd.concat(ep_frames, ignore_index=True)
        df.columns = [c.lower() for c in df.columns]
        ts_col = next((c for c in ("date", "timestamp", "created") if c in df.columns), None)
        if ts_col is None:
            continue
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col]).rename(columns={ts_col: "date"})

        # Harmonise any aliases
        for alias, target in (
            ("last", "close"),
            ("price", "close"),
            ("value", "close"),
            ("px_close", "close"),
        ):
            if alias in df.columns and target not in df.columns:
                df[target] = df[alias]
        if "close" not in df.columns:
            continue
        keep = [c for c in ("date", "open", "high", "low", "close") if c in df.columns]
        out = (
            df[keep]
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
            .copy()
        )
        out["symbol"] = symbol
        out["source"] = "theta"
        return out

    return pd.DataFrame()


def _cap_end_date(end_d: date, today: date) -> date:
    """Cap ``end_d`` at the most recently settled trading day.

    Theta's ``/v3/index/history`` endpoints don't publish today's EOD
    until ~17:15 ET. A pre-close cron run that asks for today gets 472
    NO_DATA on every symbol, which previously surfaced as FAIL in the
    orchestrator even when there was nothing actually wrong — yesterday's
    rows were already on disk and today's hadn't settled yet.

    We cap ``end_d`` at ``today - 1 day`` so the existing
    ``s_start > end_d`` skip in :func:`main` fires correctly and the
    symbol is reported as "up-to-date" with rc=0. Same idempotent
    same-day semantic the iv_surface_history puller adopted in PR #55.
    """
    if end_d >= today:
        return today - timedelta(days=1)
    return end_d


def _last_theta_date(symbol: str) -> date | None:
    if not OUT_LONG.exists():
        return None
    try:
        df = pd.read_parquet(OUT_LONG, columns=["date", "symbol", "source"])
    except Exception:
        return None
    rows = df[(df["symbol"] == symbol) & (df["source"] == "theta")]
    if rows.empty:
        return None
    return pd.to_datetime(rows["date"]).max().date()


def main() -> int:
    # CLI utf-8 fix for Windows: the console / redirected-output default
    # code page (cp1252) cannot encode the em-dash and arrow characters
    # this script prints. reconfigure() changes the encoding *in place* —
    # unlike reassigning sys.stdout to a fresh TextIOWrapper, it keeps the
    # same stream object, so pytest's capsys still captures main()'s
    # output when the suite runs on Windows. (The old reassignment did
    # not: it silently broke test_main_incremental_skips_when_only_today_
    # remains on Windows while still passing on the Linux CI runner.)
    if sys.platform == "win32":
        for _stream in (sys.stdout, sys.stderr):
            if isinstance(_stream, io.TextIOWrapper):
                _stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Only fetch data newer than last Theta write per symbol",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not _theta_up():
        print("Theta Terminal not reachable on 127.0.0.1:25503 — start it and retry.")
        return 2

    conn = ThetaConnector()
    try:
        end_d = date.fromisoformat(args.end) if args.end else date.today()
        capped = _cap_end_date(end_d, date.today())
        if capped != end_d:
            print(f"Capping end_d {end_d} → {capped} (today's EOD not yet published)")
            end_d = capped
        default_start = end_d - timedelta(days=int(args.years * 365))
        if args.start:
            default_start = date.fromisoformat(args.start)

        print(f"Theta indices pull  symbols={len(args.symbols)}  end={end_d}")
        t0 = time.perf_counter()
        frames: list[pd.DataFrame] = []
        n_ok = n_fail = n_skipped = 0
        for sym in args.symbols:
            s_start = default_start
            if args.incremental:
                last = _last_theta_date(sym)
                if last is not None:
                    s_start = max(default_start, last + timedelta(days=1))
                    if s_start > end_d:
                        print(f"  {sym:<7} up-to-date (last theta={last})")
                        n_skipped += 1
                        continue
            df = _pull(conn, sym, s_start, end_d)
            if df.empty:
                n_fail += 1
                print(f"  {sym:<7} FAIL  no data from any endpoint  range={s_start}..{end_d}")
                continue
            n_ok += 1
            print(
                f"  {sym:<7} OK    rows={len(df):<5}  "
                f"{df['date'].min().date()} to {df['date'].max().date()}"
            )
            frames.append(df)

        if not frames:
            if n_skipped == len(args.symbols):
                # Every symbol is up-to-date. Nothing to fetch, nothing failed.
                elapsed = time.perf_counter() - t0
                print()
                print(f"Done in {elapsed:.1f}s  |  all {n_skipped} symbols up-to-date")
                return 0
            print(
                "Nothing fetched. Your tier may not include indices history — "
                "run probe_theta_capabilities.py to confirm."
            )
            return 1

        new = pd.concat(frames, ignore_index=True)
        new["date"] = pd.to_datetime(new["date"]).dt.normalize()

        # Merge with existing: Theta rows win over yahoo on the same (date, symbol).
        if OUT_LONG.exists():
            old = pd.read_parquet(OUT_LONG)
            old["date"] = pd.to_datetime(old["date"]).dt.normalize()
            # Rank sources so Theta beats Yahoo during dedup
            src_rank = {"theta": 2, "yahoo": 1}
            combined = pd.concat([old, new], ignore_index=True)
            combined["_rank"] = combined["source"].map(src_rank).fillna(0)
            combined = (
                combined.sort_values(["symbol", "date", "_rank"])
                .drop_duplicates(subset=["symbol", "date"], keep="last")
                .drop(columns="_rank")
                .sort_values(["symbol", "date"])
                .reset_index(drop=True)
            )
        else:
            combined = new

        OUT_LONG.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(OUT_LONG, index=False)

        # Rebuild wide view
        wide = combined.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        wide.columns = [f"{c.lower()}_close" for c in wide.columns]
        wide = wide.reset_index().sort_values("date")
        wide.to_parquet(OUT_WIDE, index=False)

        elapsed = time.perf_counter() - t0
        print()
        print(f"Wrote {len(combined)} total rows → {OUT_LONG}")
        print(f"Wide view → {OUT_WIDE}")
        print(f"Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
        return 0 if n_fail == 0 else 1
    finally:
        # Per-endpoint failure manifest sidecar (issue #71, DECISIONS.md D11).
        # Drains conn.get_failures() and writes a JSON sidecar so a downstream
        # observer can distinguish "this ticker had no data" from "this ticker
        # hit a per-endpoint timeout while the Terminal was healthy". Runs in
        # finally so half-run pulls (KeyboardInterrupt etc.) still emit the
        # manifest — half-run is exactly when this signal matters most.
        failures = conn.get_failures()
        if failures:
            manifest_dir = Path("data_processed/theta")
            manifest_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            step_name = Path(__file__).stem
            manifest_path = manifest_dir / f"_manifest_failures_{step_name}_{ts}.json"
            payload = [asdict(r) for r in failures]
            manifest_path.write_text(json.dumps(payload, indent=2))
            print(
                f"[{step_name}] wrote {len(failures)} per-endpoint failure(s) → "
                f"{manifest_path.relative_to(Path.cwd())}",
                flush=True,
            )


if __name__ == "__main__":
    sys.exit(main())
