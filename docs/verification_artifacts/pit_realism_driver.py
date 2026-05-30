"""PIT realism driver — engine forecast vs realized 35-day outcome.

HT-B (heavy-verify cycle, 2026-05-30): does the engine's `prob_profit`
match what actually happens in the market on past as-of dates? Tests
both an in-sample window (2022-2024) and two out-of-sample windows
(2020 crisis + freshest 2024-09 → 2026-02-13). Specifically probes
the documented top-bin defect (calibration doc 2026-05-28 reports
−15 to −17pp over-confidence on prob_profit > 0.95).

Method
------
1. For each as-of date in a sample, call
   ``WheelRunner.rank_candidates_by_ev(tickers=UNIVERSE_100,
   as_of=..., top_n=100, min_ev_dollars=-1e9)`` — gets up to 100
   short-put candidates with engine ``prob_profit`` per row.
2. For each candidate, look up the underlying's close at
   ``as_of + dte_target`` calendar days (next trading day if
   weekend) from the Bloomberg OHLCV file. CRITICAL: the file
   has the column-rename quirk — CSV ``high`` is the true close
   (see docs/REAL_DATA_VERIFICATION_2026-05-28.md §Bloomberg
   CSV column-rename quirk). The connector handles the rename;
   this driver mimics it locally rather than calling the
   connector per-ticker per-as-of (much faster).
3. Mark ``realized_otm = 1`` if close ≥ strike at expiry
   (engine's prob_profit ≈ P(short-put expires OTM)).
4. Aggregate per (sub-window, prob_profit bin) and apply the
   pre-declared standard from PROB_PROFIT_CALIBRATION_2026-05-28:
   |Δ| ≤ 5pp → ✅ calibrated; 5-10pp → ⚠ warn; >10pp → ❌ MISCAL.
5. Print per-window calibration tables + top-bin focus line.

PIT discipline
--------------
- ``as_of`` is passed straight through to the engine. PIT
  guarantees are the engine's: OHLCV PIT-filter, IV history
  pulled at_of date, fundamentals, etc. The driver itself reads
  no future data — realized close comes only from
  ``as_of + dte_target``-or-later rows that exist *now* but were
  the realized outcome at the time the engine couldn't know.
- Realized close is the underlying's price 35 calendar days after
  as_of, NOT an exact option settlement price; the engine's
  prob_profit is "probability of net-positive P&L on a 35-day
  trade", and the standard calibration convention (used by
  PROB_PROFIT_CALIBRATION_2026-05-28.md) approximates this with
  the realized-OTM frequency. Close ≥ strike at +35d is the same
  question for a synthetic 35d short put.

Checkpointing
-------------
Writes one row to ``pit_realism_raw_<run-date>.csv`` after each
as_of completes. A crash mid-run can resume by re-running — the
driver skips dates already in the CSV.

Re-runnable from any worktree by editing the ``WORKTREE`` constant.
"""

from __future__ import annotations

import io
import sys
import time
from datetime import date
from pathlib import Path

# UTF-8-safe stdout/stderr — Windows console defaults to cp1252 which
# can't encode the arrow / box-drawing chars used in progress lines.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

WORKTREE = Path(r"C:\Users\merty\Desktop\swe-terminal-b").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import numpy as np  # noqa: E402  (sys.path bootstrap above)
import pandas as pd  # noqa: E402

from backtests.regression.universes import UNIVERSE_100  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DTE_TARGET = 35
TOP_N = 100
MIN_EV = -1e9  # capture every candidate, including negative-EV

RUN_DATE = date.today().isoformat()
ARTIFACT_DIR = WORKTREE / "docs" / "verification_artifacts"
RAW_CSV = ARTIFACT_DIR / f"pit_realism_raw_{RUN_DATE}.csv"
OHLCV_FILE = WORKTREE / "data" / "bloomberg" / "sp500_ohlcv.csv"

# Pre-declared calibration standard (mirrors PROB_PROFIT_CALIBRATION_2026-05-28.md)
BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001]
BIN_LABELS = [
    "(0.0, 0.5]",
    "(0.5, 0.6]",
    "(0.6, 0.7]",
    "(0.7, 0.8]",
    "(0.8, 0.85]",
    "(0.85, 0.9]",
    "(0.9, 0.95]",
    "(0.95, 1.0]",
]


def verdict(delta_pp: float) -> str:
    a = abs(delta_pp)
    if a <= 5.0:
        return "OK"
    if a <= 10.0:
        return "WARN"
    return "MISCAL"


# ---------------------------------------------------------------------
# As-of date design
# ---------------------------------------------------------------------
def _bday(d: pd.Timestamp) -> pd.Timestamp:
    """Snap to the next business day (Mon-Fri)."""
    while d.weekday() >= 5:
        d = d + pd.Timedelta(days=1)
    return d


def design_dates() -> list[tuple[str, str]]:
    """Returns (window, as_of_iso) tuples covering 3 sub-windows.

    - in_sample_2022_2024: monthly cadence (36 dates)
    - oos_2020_crisis: bi-weekly Jan-Dec 2020 (~26 dates, COVID covered)
    - oos_fresh_2024_2026: bi-weekly Sep 2024 → 2026-02-13 (~38 dates)
    """
    out: list[tuple[str, str]] = []

    # in-sample: 1st business day of each month, 2022-01 to 2024-12
    for year in range(2022, 2025):
        for month in range(1, 13):
            d = _bday(pd.Timestamp(year=year, month=month, day=1))
            out.append(("in_sample_2022_2024", d.date().isoformat()))

    # OOS 2020 crisis: bi-weekly. Use Monday-of-week stride.
    d = pd.Timestamp("2020-01-06")  # first Monday of 2020
    while d <= pd.Timestamp("2020-12-21"):
        out.append(("oos_2020_crisis", _bday(d).date().isoformat()))
        d = d + pd.Timedelta(days=14)

    # OOS fresh: bi-weekly from 2024-09 through 2026-02-13 (last as_of
    # leaving 35d of forward room before data_end = 2026-03-20).
    d = pd.Timestamp("2024-09-02")  # first Monday of Sep 2024
    last = pd.Timestamp("2026-02-13")
    while d <= last:
        out.append(("oos_fresh_2024_2026", _bday(d).date().isoformat()))
        d = d + pd.Timedelta(days=14)

    return out


# ---------------------------------------------------------------------
# OHLCV cache (with the Bloomberg CSV column-rename quirk)
# ---------------------------------------------------------------------
def load_ohlcv_close() -> dict[str, pd.Series]:
    """Returns {ticker: pd.Series(close, index=date)} for the full
    universe. CRITICAL: in the Bloomberg CSV the column labelled
    ``high`` is the TRUE close (column-rename quirk documented in
    docs/REAL_DATA_VERIFICATION_2026-05-28.md). The connector
    handles this transparently; we mimic it here so the driver
    doesn't have to call the connector per (ticker, as_of) pair.
    """
    df = pd.read_csv(
        OHLCV_FILE,
        usecols=["date", "ticker", "high"],  # csv "high" = real close
        parse_dates=["date"],
    )
    df["ticker_short"] = df["ticker"].str.split(" ").str[0]
    df = df.rename(columns={"high": "close"})
    df = df.set_index(["ticker_short", "date"]).sort_index()["close"]
    out: dict[str, pd.Series] = {}
    for t in df.index.get_level_values(0).unique():
        out[t] = df.loc[t].sort_index()
    return out


def realized_close_at(close_series: pd.Series, as_of: pd.Timestamp, dte: int) -> float | None:
    """Close on the first trading day ≥ as_of + dte calendar days.
    Returns None if no such row exists in the series (data ends
    before the realized horizon)."""
    target = as_of + pd.Timedelta(days=dte)
    after = close_series[close_series.index >= target]
    if after.empty:
        return None
    return float(after.iloc[0])


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def already_done() -> set[str]:
    """Return the set of as_of_iso values already in RAW_CSV — for
    safe resume after a mid-run crash."""
    if not RAW_CSV.exists():
        return set()
    try:
        prev = pd.read_csv(RAW_CSV, usecols=["as_of"])
        return set(prev["as_of"].astype(str).unique())
    except Exception:
        return set()


def main() -> int:
    print("=" * 78)
    print(f"HT-B PIT REALISM DRIVER — {RUN_DATE}")
    print("=" * 78)
    print(f"Worktree: {WORKTREE}")
    print(f"Raw output: {RAW_CSV}")
    print()

    print("Loading OHLCV close series (with CSV-rename fix)...")
    t0 = time.perf_counter()
    closes = load_ohlcv_close()
    print(f"  loaded {len(closes)} tickers in {time.perf_counter() - t0:.1f}s")
    print(f"  e.g. AAPL last: {closes['AAPL'].index.max().date()} = {closes['AAPL'].iloc[-1]:.2f}")
    print()

    dates = design_dates()
    print(f"As-of dates: {len(dates)} total")
    for w in ("in_sample_2022_2024", "oos_2020_crisis", "oos_fresh_2024_2026"):
        n = sum(1 for win, _ in dates if win == w)
        print(f"  {w:24s} -> {n} dates")
    print()

    done = already_done()
    if done:
        print(f"RESUME mode: {len(done)} as_of already in RAW_CSV; skipping.")
        print()

    runner = WheelRunner()
    print(f"Connector: {type(runner.connector).__name__}")
    print(f"Universe size: {len(UNIVERSE_100)}")
    print()

    write_header = not RAW_CSV.exists()
    cum_rows = 0
    cum_t0 = time.perf_counter()

    for i, (window, as_of) in enumerate(dates, start=1):
        if as_of in done:
            continue
        t_as = time.perf_counter()
        try:
            df = runner.rank_candidates_by_ev(
                tickers=list(UNIVERSE_100),
                dte_target=DTE_TARGET,
                top_n=TOP_N,
                min_ev_dollars=MIN_EV,
                as_of=as_of,
                include_diagnostic_fields=False,
                max_as_of_staleness_days=10_000,  # historical as_of
            )
        except Exception as exc:
            print(
                f"[{i:3d}/{len(dates)}] {window:24s} {as_of} → ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

        elapsed = time.perf_counter() - t_as
        if df is None or df.empty:
            print(
                f"[{i:3d}/{len(dates)}] {window:24s} {as_of} → 0 candidates ({elapsed:.1f}s)",
                flush=True,
            )
            continue

        # Resolve realized close per row.
        as_of_ts = pd.Timestamp(as_of)
        records = []
        for _, r in df.iterrows():
            t = r["ticker"]
            strike = float(r["strike"])
            close = closes.get(t)
            if close is None:
                realized = None
            else:
                realized = realized_close_at(close, as_of_ts, DTE_TARGET)
            records.append(
                {
                    "window": window,
                    "as_of": as_of,
                    "ticker": t,
                    "strike": strike,
                    "premium": float(r.get("premium", np.nan)),
                    "spot": float(r.get("spot", np.nan)),
                    "iv": float(r.get("iv", np.nan)),
                    "dte": int(r.get("dte", DTE_TARGET)),
                    "ev_dollars": float(r.get("ev_dollars", np.nan)),
                    "prob_profit": float(r.get("prob_profit", np.nan)),
                    "prob_assignment": float(r.get("prob_assignment", np.nan)),
                    "distribution_source": str(r.get("distribution_source", "")),
                    "realized_close": realized,
                    "realized_otm": (None if realized is None else int(realized >= strike)),
                }
            )
        out_df = pd.DataFrame.from_records(records)
        out_df.to_csv(RAW_CSV, mode="a", header=write_header, index=False)
        write_header = False
        cum_rows += len(out_df)
        n_with_realized = out_df["realized_otm"].notna().sum()
        n_otm = int(out_df["realized_otm"].fillna(0).sum())
        rate = cum_rows / max(time.perf_counter() - cum_t0, 1e-9)
        print(
            f"[{i:3d}/{len(dates)}] {window:24s} {as_of} → "
            f"{len(out_df):3d} rows, realized {n_with_realized}, OTM {n_otm}  "
            f"({elapsed:.1f}s, cum {cum_rows} rows @ {rate:.1f} rows/s)",
            flush=True,
        )

    print()
    print(f"DRIVER DONE — {cum_rows} new rows in {time.perf_counter() - cum_t0:.1f}s")
    print(f"Raw output: {RAW_CSV}")
    print()
    aggregate()
    return 0


# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------
def aggregate() -> None:
    """Read the raw CSV and emit per-window calibration tables to stdout.

    Two OTM-style metrics are computed:
      * ``realized_otm``      — close >= strike (matches the convention
        used by PROB_PROFIT_CALIBRATION_2026-05-28.md and prior
        regression-harness rank_logs). Strict; under-states the engine's
        actual prob_profit because pnl > 0 also covers shallow ITM
        assignments where premium > intrinsic.
      * ``realized_pnl_positive`` — close >= strike - premium (engine's
        EXACT prob_profit definition: pnl = premium - max(0, strike -
        spot); pnl > 0 ⇔ spot > strike - premium). Derived on the fly
        from already-captured columns.
    """
    if not RAW_CSV.exists():
        print("AGGREGATE: no raw CSV yet, nothing to summarize.")
        return
    raw = pd.read_csv(RAW_CSV)
    # Keep only rows with both a prob_profit and a realized outcome.
    raw = raw.dropna(subset=["prob_profit", "realized_otm"]).copy()
    # Engine-exact P(net P&L > 0) realised flag (close >= breakeven).
    raw["breakeven"] = raw["strike"] - raw["premium"]
    raw["realized_pnl_positive"] = (raw["realized_close"] >= raw["breakeven"]).astype(int)
    raw["bin"] = pd.cut(
        raw["prob_profit"],
        bins=BINS,
        labels=BIN_LABELS,
        include_lowest=True,
        right=True,
    )

    print("=" * 78)
    print("CALIBRATION SUMMARY (per sub-window)")
    print("=" * 78)
    print(f"Total usable rows (engine + realized): {len(raw):,}")
    print()

    summary_rows = []
    for window, sub in raw.groupby("window"):
        n = len(sub)
        otm_rate = sub["realized_otm"].mean() if n else float("nan")
        print(f"--- {window} (n={n:,}, overall OTM = {otm_rate:.4f}) ---")
        print(
            f"  {'Bin':<14}{'n':>7}{'Engine mean':>14}{'Actual OTM':>14}{'Δ pp':>10}{'Verdict':>10}"
        )
        weighted_mad_num = 0.0
        weighted_mad_den = 0
        ok = warn = miscal = 0
        top_bin_row = None
        for label in BIN_LABELS:
            bin_sub = sub[sub["bin"] == label]
            if bin_sub.empty:
                continue
            engine_mean = float(bin_sub["prob_profit"].mean())
            actual_otm = float(bin_sub["realized_otm"].mean())
            delta_pp = (actual_otm - engine_mean) * 100
            v = verdict(delta_pp)
            weighted_mad_num += abs(delta_pp) * len(bin_sub)
            weighted_mad_den += len(bin_sub)
            if v == "OK":
                ok += 1
            elif v == "WARN":
                warn += 1
            else:
                miscal += 1
            row = {
                "window": window,
                "bin": label,
                "n": int(len(bin_sub)),
                "engine_mean": round(engine_mean, 4),
                "actual_otm": round(actual_otm, 4),
                "delta_pp": round(delta_pp, 2),
                "verdict": v,
            }
            summary_rows.append(row)
            print(
                f"  {label:<14}{len(bin_sub):>7,}{engine_mean:>14.4f}{actual_otm:>14.4f}"
                f"{delta_pp:>10.2f}{v:>10}"
            )
            if label == "(0.95, 1.0]":
                top_bin_row = row
        wmad = weighted_mad_num / max(weighted_mad_den, 1)
        print(f"  Weighted MAD: {wmad:.2f}pp | OK {ok} / WARN {warn} / MISCAL {miscal}")
        if top_bin_row:
            print(
                f"  TOP-BIN focus (0.95, 1.0]: n={top_bin_row['n']} "
                f"engine={top_bin_row['engine_mean']:.4f} "
                f"actual={top_bin_row['actual_otm']:.4f} "
                f"Δ={top_bin_row['delta_pp']:+.2f}pp → {top_bin_row['verdict']}"
            )
        else:
            print(
                "  TOP-BIN focus (0.95, 1.0]: NO ROWS (no candidates with engine prob>0.95 here)."
            )
        print()

    # Section-2 invariant check on this run.
    n_neg_ev = int((raw["ev_dollars"] <= 0).sum())
    n_nonfinite = int((~np.isfinite(raw["ev_dollars"])).sum())
    print("--- Section 2 invariant on this driver's emitted rows ---")
    print(f"  rows with ev_dollars <= 0     : {n_neg_ev:,} (informational - we asked for all)")
    print(f"  rows with non-finite ev_dollars: {n_nonfinite}")
    print()

    # ----------------------------------------------------------------
    # Bonus #1: Within-2020 phase breakdown (pre-COVID / COVID-spike /
    # recovery). The aggregate OOS-2020 row pools all three; this
    # surfaces whether the calibration finding is dominated by the
    # COVID-spike weeks or holds across phases.
    # ----------------------------------------------------------------
    raw_2020 = raw[raw["window"] == "oos_2020_crisis"].copy()
    if len(raw_2020):

        def _phase_2020(d: str) -> str:
            d_ts = pd.Timestamp(d)
            if d_ts < pd.Timestamp("2020-02-24"):
                return "2020a_pre_covid"
            if d_ts <= pd.Timestamp("2020-05-15"):
                return "2020b_covid_spike"
            return "2020c_recovery"

        raw_2020["phase"] = raw_2020["as_of"].apply(_phase_2020)
        print("=" * 78)
        print("BONUS 1 -- Within OOS-2020 phase breakdown")
        print("=" * 78)
        for phase, sub in raw_2020.groupby("phase"):
            n = len(sub)
            otm_rate = sub["realized_otm"].mean()
            print(f"--- {phase} (n={n:,}, overall OTM = {otm_rate:.4f}) ---")
            for label in BIN_LABELS:
                bin_sub = sub[sub["bin"] == label]
                if bin_sub.empty:
                    continue
                engine_mean = float(bin_sub["prob_profit"].mean())
                actual_otm = float(bin_sub["realized_otm"].mean())
                delta_pp = (actual_otm - engine_mean) * 100
                v = verdict(delta_pp)
                print(
                    f"  {label:<14}{len(bin_sub):>7,}{engine_mean:>14.4f}{actual_otm:>14.4f}"
                    f"{delta_pp:>10.2f}{v:>10}"
                )
            print()

    # ----------------------------------------------------------------
    # Bonus #2: By distribution_source (does HAR-RV / block-bootstrap
    # behave differently from empirical_non_overlapping?)
    # ----------------------------------------------------------------
    print("=" * 78)
    print("BONUS 2 -- By distribution_source (overall, all windows)")
    print("=" * 78)
    for src, sub in raw.groupby("distribution_source"):
        n = len(sub)
        if n < 50:
            print(f"--- source={src} (n={n} too small for meaningful binning) ---")
            print()
            continue
        otm_rate = sub["realized_otm"].mean()
        print(f"--- source={src} (n={n:,}, overall OTM = {otm_rate:.4f}) ---")
        for label in BIN_LABELS:
            bin_sub = sub[sub["bin"] == label]
            if bin_sub.empty:
                continue
            engine_mean = float(bin_sub["prob_profit"].mean())
            actual_otm = float(bin_sub["realized_otm"].mean())
            delta_pp = (actual_otm - engine_mean) * 100
            v = verdict(delta_pp)
            print(
                f"  {label:<14}{len(bin_sub):>7,}{engine_mean:>14.4f}{actual_otm:>14.4f}"
                f"{delta_pp:>10.2f}{v:>10}"
            )
        print()

    # ----------------------------------------------------------------
    # Bonus #3: In-sample vs OOS top-bin contrast (OTM convention)
    # ----------------------------------------------------------------
    print("=" * 78)
    print("BONUS 3 -- In-sample vs OOS top-bin contrast (OTM convention)")
    print("=" * 78)
    print(f"  {'Window':<25}{'top-bin n':>12}{'engine':>10}{'actual':>10}{'Delta pp':>12}")
    for window, sub in raw.groupby("window"):
        top = sub[sub["bin"] == "(0.95, 1.0]"]
        if top.empty:
            print(f"  {window:<25}{'0':>12}{'-':>10}{'-':>10}{'(no data)':>12}")
            continue
        em = float(top["prob_profit"].mean())
        am = float(top["realized_otm"].mean())
        dp = (am - em) * 100
        print(f"  {window:<25}{len(top):>12,}{em:>10.4f}{am:>10.4f}{dp:>12.2f}")
    print()

    # ----------------------------------------------------------------
    # Bonus #3b: Same contrast but using EXACT engine prob_profit
    # definition (P[net pnl > 0] = P[spot >= strike - premium]). This
    # is what the engine actually claims; the OTM convention above
    # under-states actual prob_profit by the band [strike - premium,
    # strike] where the put is ITM but still profitable.
    # ----------------------------------------------------------------
    print("=" * 78)
    print("BONUS 3b -- Top-bin contrast using EXACT engine prob_profit definition")
    print("=" * 78)
    print(f"  {'Window':<25}{'top-bin n':>12}{'engine':>10}{'actual':>10}{'Delta pp':>12}")
    for window, sub in raw.groupby("window"):
        top = sub[sub["bin"] == "(0.95, 1.0]"]
        if top.empty:
            print(f"  {window:<25}{'0':>12}{'-':>10}{'-':>10}{'(no data)':>12}")
            continue
        em = float(top["prob_profit"].mean())
        am = float(top["realized_pnl_positive"].mean())
        dp = (am - em) * 100
        print(f"  {window:<25}{len(top):>12,}{em:>10.4f}{am:>10.4f}{dp:>12.2f}")
    print()

    # ----------------------------------------------------------------
    # Bonus #3c: Full per-window calibration using EXACT engine
    # prob_profit definition, ALL bins. Side-by-side with the OTM
    # convention helps quantify how much of the headline "over-
    # confidence" is methodology and how much is real.
    # ----------------------------------------------------------------
    print("=" * 78)
    print("BONUS 3c -- Full per-window calibration, EXACT engine prob_profit definition")
    print("=" * 78)
    for window, sub in raw.groupby("window"):
        n = len(sub)
        pnl_rate = sub["realized_pnl_positive"].mean()
        otm_rate = sub["realized_otm"].mean()
        print(f"--- {window} (n={n:,}, overall pnl>0 = {pnl_rate:.4f}, OTM = {otm_rate:.4f}) ---")
        wmad_num, wmad_den = 0.0, 0
        ok = warn = miscal = 0
        for label in BIN_LABELS:
            bin_sub = sub[sub["bin"] == label]
            if bin_sub.empty:
                continue
            engine_mean = float(bin_sub["prob_profit"].mean())
            actual_pnl = float(bin_sub["realized_pnl_positive"].mean())
            delta_pp = (actual_pnl - engine_mean) * 100
            v = verdict(delta_pp)
            wmad_num += abs(delta_pp) * len(bin_sub)
            wmad_den += len(bin_sub)
            if v == "OK":
                ok += 1
            elif v == "WARN":
                warn += 1
            else:
                miscal += 1
            print(
                f"  {label:<14}{len(bin_sub):>7,}{engine_mean:>14.4f}{actual_pnl:>14.4f}"
                f"{delta_pp:>10.2f}{v:>10}"
            )
        wmad = wmad_num / max(wmad_den, 1)
        print(f"  Weighted MAD: {wmad:.2f}pp | OK {ok} / WARN {warn} / MISCAL {miscal}")
        print()

    # ----------------------------------------------------------------
    # Bonus #4: Per-year breakdown within the 2022-2024 in-sample window
    # ----------------------------------------------------------------
    raw_in = raw[raw["window"] == "in_sample_2022_2024"].copy()
    if len(raw_in):
        raw_in["year"] = raw_in["as_of"].str[:4]
        print("=" * 78)
        print("BONUS 4 -- Per-year within in-sample 2022-2024 (top-bin only)")
        print("=" * 78)
        print(f"  {'Year':<6}{'top-bin n':>12}{'engine':>10}{'actual':>10}{'Delta pp':>12}")
        for year, sub in raw_in.groupby("year"):
            top = sub[sub["bin"] == "(0.95, 1.0]"]
            if top.empty:
                print(f"  {year:<6}{'0':>12}{'-':>10}{'-':>10}{'(no data)':>12}")
                continue
            em = float(top["prob_profit"].mean())
            am = float(top["realized_otm"].mean())
            dp = (am - em) * 100
            print(f"  {year:<6}{len(top):>12,}{em:>10.4f}{am:>10.4f}{dp:>12.2f}")
        print()


if __name__ == "__main__":
    sys.exit(main())
