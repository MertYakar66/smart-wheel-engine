"""
Shared Bloomberg per-name panel-pull engine.

Backs the three core pullers (pull_ohlcv.py, pull_liquidity.py,
pull_vol_iv.py). Keeping the window math + durable-write logic in ONE place
means the only thing that differs per file is its config (fields, field map,
column order, floor, ticker-suffix handling).

CONTIGUOUS BACKFILL (the whole point):
  Given an existing CSV, this fills the FORWARD gap
  [existing_max+1 -> END_DATE] first, then walks the BACKWARD gap
  [existing_min-1 -> FLOOR] NEWEST-FIRST in ~chunk-month windows. Because
  windows are processed newest-first and the on-disk CSV is rewritten after
  EVERY window, coverage always stays hole-free outward from the recent edge:
  if the metered API caps mid-run, everything up to the last completed window
  is already saved, and the next session resumes further back.

Per-window: pull all members (in `chunk_size`-ticker bdh calls) -> reshape
long->wide -> merge into the running frame -> dedupe (date,ticker) keep-last
-> sort (ticker,date) -> write. An optional `validate` callback runs on the
merged frame after each write and may raise to halt (e.g. the OHLCV rotation
gate).

Env knobs (all optional; defaults preserve prior behaviour):
  SWE_PULL_LIMIT            pull only first N members (smoke). 0/unset = all.
  SWE_PULL_NO_WRITE         pull+reshape+print, skip ALL merge/write.
  SWE_PULL_END              override END_DATE (default = config.end_date).
  SWE_PULL_FLOOR            override backfill floor (default = config.floor).
  SWE_PULL_MODE             forward | backfill | both     (default both).
  SWE_BACKFILL_CHUNK_MONTHS months per backward window     (default 30).
  SWE_BACKFILL_MAX_WINDOWS  cap # BACKWARD windows this run (0/unset = all).
  SWE_OUT_PATH              write to this path instead of data/bloomberg/<out_name>
                            (used to grow a deep-history scratch off the frozen
                            connector monolith).
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass, field

import pandas as pd
from xbbg import blp

# UTF-8-safe stdout/stderr: xbbg 1.2 returns narwhals frames whose repr uses
# box-drawing chars that crash the Windows cp1252 console. PYTHONUTF8=1 also
# covers this; this is belt-and-suspenders for subprocess runs.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

MEMBERS_TICKER_DEFAULT = "Member Ticker and Exchange Code"


@dataclass
class PanelConfig:
    out_name: str                       # csv filename under data/bloomberg/
    fields: list                        # Bloomberg fields to request
    field_map: dict                     # bbg field -> output column name
    out_cols: list                      # final column order (must incl. ticker)
    start_date_full: str                # used ONLY when no existing CSV
    end_date: str                       # default forward end
    floor: str = "1994-01-01"           # backfill floor (per-name panels)
    strip_equity_suffix: bool = False   # "AAPL UW Equity" -> "AAPL UW"
    bdh_kwargs: dict = field(default_factory=dict)  # e.g. {"Fill": "P"}
    chunk_size: int = 30                # tickers per bdh call
    chunk_months: int = 30              # months per backward window
    validate: object = None             # optional callable(combined_df) -> None


def to_native(obj):
    return obj.to_native() if hasattr(obj, "to_native") else obj


def _env(name, default=None):
    v = os.environ.get(name)
    return v if v else default


def plan_windows(existing, start_date_full, end_date, floor, mode, chunk_months, max_windows):
    """Ordered (start, end) date-string windows to pull.

    Forward gap first (oldest-to-edge is irrelevant; it's a single window),
    then backward windows NEWEST-FIRST so partial runs stay contiguous from
    the recent edge. No overlap between windows; boundaries are inclusive and
    handed to bdh which returns trading days only.
    """
    end_ts = pd.Timestamp(end_date)
    floor_ts = pd.Timestamp(floor)
    windows: list[tuple[str, str]] = []

    if existing is None or len(existing) == 0:
        start = max(pd.Timestamp(start_date_full), floor_ts)
        if start <= end_ts:
            windows.append((start.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))
        return windows

    emin = pd.Timestamp(existing["date"].min())
    emax = pd.Timestamp(existing["date"].max())

    # --- forward gap: existing_max+1 -> end ---
    if mode in ("forward", "both"):
        fstart = emax + pd.Timedelta(days=1)
        if fstart <= end_ts:
            windows.append((fstart.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")))

    # --- backward gap: existing_min-1 -> floor, newest-first ---
    if mode in ("backfill", "both"):
        cur_end = emin - pd.Timedelta(days=1)
        n = 0
        while cur_end >= floor_ts:
            if max_windows and n >= max_windows:
                break
            cur_start = cur_end - pd.DateOffset(months=chunk_months) + pd.Timedelta(days=1)
            if cur_start < floor_ts:
                cur_start = floor_ts
            windows.append((cur_start.strftime("%Y-%m-%d"), cur_end.strftime("%Y-%m-%d")))
            cur_end = cur_start - pd.Timedelta(days=1)
            n += 1

    return windows


def _pull_window(cfg: PanelConfig, tickers, ws, we):
    """Pull every ticker for [ws, we] -> tidy wide frame in cfg.out_cols, or None."""
    chunks = []
    for i in range(0, len(tickers), cfg.chunk_size):
        chunk = tickers[i : i + cfg.chunk_size]
        print(f"    tickers {i + 1}-{min(i + cfg.chunk_size, len(tickers))}/{len(tickers)}", flush=True)
        try:
            raw = to_native(
                blp.bdh(tickers=chunk, flds=cfg.fields, start_date=ws, end_date=we, **cfg.bdh_kwargs)
            )
            if raw is None or len(raw) == 0:
                continue
            wide = raw.pivot_table(
                index=["date", "ticker"], columns="field", values="value", aggfunc="first"
            ).reset_index()
            wide.columns.name = None
            wide = wide.rename(columns=cfg.field_map)
            for c in cfg.field_map.values():
                if c not in wide.columns:
                    wide[c] = pd.NA
            if cfg.strip_equity_suffix:
                wide["ticker"] = wide["ticker"].str.replace(" Equity", "", regex=False)
            wide["date"] = pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")
            chunks.append(wide[cfg.out_cols])
        except Exception as e:
            print(f"    ERROR chunk {i}: {e}", flush=True)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True)


def run(cfg: PanelConfig):
    limit = int(_env("SWE_PULL_LIMIT", "0") or "0")
    no_write = bool(_env("SWE_PULL_NO_WRITE"))
    end_date = _env("SWE_PULL_END", cfg.end_date)
    floor = _env("SWE_PULL_FLOOR", cfg.floor)
    mode = _env("SWE_PULL_MODE", "both")
    chunk_months = int(_env("SWE_BACKFILL_CHUNK_MONTHS", str(cfg.chunk_months)))
    max_windows = int(_env("SWE_BACKFILL_MAX_WINDOWS", "0") or "0")

    # SWE_OUT_PATH lets the deep-history backfill grow a scratch file off the
    # connector monolith (which stays frozen <100 MB on the refresh branch).
    out_path = _env("SWE_OUT_PATH") or os.path.join(
        os.path.dirname(__file__), "..", "data", "bloomberg", cfg.out_name
    )

    existing = None
    if os.path.exists(out_path) and os.path.getsize(out_path) > 100:
        existing = pd.read_csv(out_path, dtype={"date": str})
        print(f"Existing {cfg.out_name}: {len(existing):,} rows, "
              f"{existing['date'].min()} -> {existing['date'].max()}")
    else:
        print(f"No existing {cfg.out_name}; fresh pull from {cfg.start_date_full}.")

    windows = plan_windows(existing, cfg.start_date_full, end_date, floor, mode, chunk_months, max_windows)
    print(f"END={end_date} FLOOR={floor} MODE={mode} CHUNK_MONTHS={chunk_months} "
          f"MAX_WINDOWS={max_windows or 'all'}")
    if not windows:
        print("Nothing to pull (already contiguous floor->end for this mode).")
        return
    print("Planned windows (execution order, newest-first after forward):")
    for ws, we in windows:
        print(f"  {ws} -> {we}")

    print("Fetching SPX members (INDX_MWEIGHT)...")
    members = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
    cand = [c for c in members.columns if "member" in c.lower() and "ticker" in c.lower()]
    mcol = cand[0] if cand else MEMBERS_TICKER_DEFAULT
    tickers = [t + " Equity" for t in members[mcol].tolist()]
    if limit:
        tickers = tickers[:limit]
        print(f"SWE_PULL_LIMIT active -> {len(tickers)} tickers")
    print(f"{len(tickers)} tickers (current SPX members; historical windows use today's constituents)")

    combined = existing[cfg.out_cols].copy() if existing is not None else None
    total_added = 0

    for wi, (ws, we) in enumerate(windows, 1):
        print(f"\n[window {wi}/{len(windows)}] {ws} -> {we}", flush=True)
        delta = _pull_window(cfg, tickers, ws, we)
        if delta is None or len(delta) == 0:
            print("  (no data in window)")
            continue
        print(f"  pulled {len(delta):,} rows ({delta['date'].min()} -> {delta['date'].max()}, "
              f"{delta['ticker'].nunique()} tickers)")
        if no_write:
            print("  SWE_PULL_NO_WRITE set -> not merging. sample:")
            print(delta.head(6).to_string())
            continue
        combined = pd.concat([combined, delta], ignore_index=True) if combined is not None else delta
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        combined.to_csv(out_path, index=False)
        total_added += len(delta)
        print(f"  WROTE {cfg.out_name}: {len(combined):,} rows, "
              f"{combined['date'].min()} -> {combined['date'].max()}", flush=True)
        if cfg.validate is not None:
            cfg.validate(combined)

    if no_write:
        print("\nNO_WRITE run complete (nothing written).")
    elif combined is not None:
        print(f"\nDONE {cfg.out_name}: {len(combined):,} rows x {len(combined.columns)} cols, "
              f"{combined['ticker'].nunique()} tickers, "
              f"{combined['date'].min()} -> {combined['date'].max()} (+{total_added:,} rows this run)")
