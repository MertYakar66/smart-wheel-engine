"""HT-C driver — news-mult distribution + prob_profit calibration re-verify.

Card HT-C (2026-05-30 heavy-verify cycle, GO comment #issuecomment-4581580...). The card's
acceptance is MEASURE-FIRST: before re-running anything heavy, measure the actual
``news_multiplier`` distribution that the engine applied in a representative pre-#249
ranking. The campaign claim that "all prior backtests obsolete after the D18 severance"
(``docs/NEWS_REDESIGN_CAMPAIGN.md`` §3) is unverified on this Bloomberg-only environment:
the news store is empty (sentiment 0.0 → multiplier 1.0) and there is no skew, so the
historical multiplier stack may have already been ~1.0, making D18 a no-op for those
backtests. This driver makes that measurement, then re-verifies prob_profit calibration.

Three sections, run in order:

  §A. EMPIRICAL news_multiplier measurement.
      Applies the pre-D18 ``sentiment_multiplier`` ladder (extracted verbatim from
      ``9e8edcd~1:engine/news_sentiment.py``) to ``NewsSentimentReader`` over
      ``UNIVERSE_100`` × 13 representative as_of dates spanning 2022-01 → 2024-12.
      Reports the (sentiment, n_articles, multiplier) distribution.

  §B. CALIBRATION on existing S27 rank_log (pre-#249 engine output).
      The published doc ``PROB_PROFIT_CALIBRATION_2026-05-28.md`` reports S27 top-bin
      Δ = -4.88pp. Re-derives the 7-bin calibration table from the raw rank_log on
      disk to confirm the published number reproduces.

  §C. RE-RUN S27 on the post-#249 engine, compute calibration, and contrast against §B
      row-by-row on ``prob_profit``. Because ``prob_profit`` is computed from
      ``np.mean(pnls > 0)`` BEFORE the regime multiplier is applied (``ev_engine.py``
      line 385 vs line ~469 comment "Regime multiplier is applied *last*"), and
      because the regime multiplier is the only channel where ``news_mult`` enters
      the EV path (``wheel_runner.py:1461`` combines it into
      ``combined_regime_mult`` which is passed as ``trade.regime_multiplier``),
      ``prob_profit`` is invariant to ``news_mult`` by construction. §C is the
      empirical confirmation of that analytical claim.

The §A measurement is the load-bearing one: if news_mult was ~1.0 historically on
Bloomberg, D18 is a no-op for every backtest run in this environment and the
"re-baseline mandatory" claim is wrong for D18 (separate from R9/EDGAR
re-baseline rationale, which is still future-tense in the campaign).

Run::

    py -3.12 docs/verification_artifacts/news_calibration_driver.py \
        --skip-rerun                       # §A + §B only (~3 min)
    py -3.12 docs/verification_artifacts/news_calibration_driver.py \
        --output-dir <tempdir>             # full §A + §B + §C (~25 min)

Outputs stream to stdout; the consumer redirects to
``docs/verification_artifacts/heavy_news_calibration_<date>_raw_output.txt``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# UTF-8-safe stdout/stderr (Windows console / redirected output).
# reconfigure() mutates in place so it stays safe under pytest's capsys
# and the cp1252 console default would otherwise crash on Greek capital
# delta + arrow + ellipsis characters used in tables and synthesis.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# §A — pre-D18 sentiment_multiplier ladder (verbatim from
# 9e8edcd~1:engine/news_sentiment.py:sentiment_multiplier)
# ---------------------------------------------------------------------------
def pre_d18_sentiment_multiplier(sentiment: float, n_articles: int) -> float:
    """Reproduce the pre-D18 multiplier ladder exactly.

    Source: ``9e8edcd~1:engine/news_sentiment.py`` lines 188-218
    (the ``NewsSentimentReader.sentiment_multiplier`` method body).

    Bands:
      - n_articles < 5                          -> 1.00
      - sentiment <= -0.3 and n >= 5            -> 0.88
      - sentiment in (-0.3, -0.1]               -> 0.95
      - sentiment >= 0.3 and n >= 5             -> 1.05
      - otherwise (neutral)                     -> 1.00
    """
    if n_articles < 5:
        return 1.00
    if sentiment <= -0.3:
        return 0.88
    if sentiment <= -0.1:
        return 0.95
    if sentiment >= 0.3:
        return 1.05
    return 1.00


# ---------------------------------------------------------------------------
# §A measurement
# ---------------------------------------------------------------------------
def measure_news_mult_distribution() -> dict:
    """Probe ``NewsSentimentReader.get_ticker_sentiment`` over a representative
    universe × dates grid and apply the pre-D18 multiplier ladder.

    Returns a summary dict with the multiplier distribution + a sample of
    non-neutral cells (if any).
    """
    from backtests.regression.universes import UNIVERSE_100
    from engine.news_sentiment import NewsSentimentReader

    # Quarterly samples spanning the S38 window (2020-2024) — covers the
    # configurations PROB_PROFIT_CALIBRATION_2026-05-28.md analysed.
    sample_dates = [
        "2020-03-31",
        "2020-09-30",
        "2021-03-31",
        "2021-09-30",
        "2022-03-31",
        "2022-09-30",
        "2023-03-31",
        "2023-09-30",
        "2024-03-31",
        "2024-09-30",
        "2024-12-31",
        "2025-06-30",  # mid-S40-W3 window
        "2026-03-20",  # freshest data, sentinel for "right now"
    ]

    reader = NewsSentimentReader(base_dir=str(_ROOT))

    # Force-load the store once so we observe its shape, not the 5-min
    # cache TTL.
    df_store = reader._load()
    store_state = {
        "shape": tuple(df_store.shape),
        "columns": list(df_store.columns),
        "is_empty": bool(df_store.empty),
    }

    rows: list[dict] = []
    for ticker in UNIVERSE_100:
        for as_of in sample_dates:
            try:
                payload = reader.get_ticker_sentiment(ticker, lookback_hours=72, as_of=as_of)
            except Exception as exc:
                payload = {
                    "sentiment": 0.0,
                    "n_articles": 0,
                    "_error": repr(exc),
                }
            mult = pre_d18_sentiment_multiplier(
                sentiment=float(payload.get("sentiment", 0.0) or 0.0),
                n_articles=int(payload.get("n_articles", 0) or 0),
            )
            rows.append(
                {
                    "ticker": ticker,
                    "as_of": as_of,
                    "sentiment": float(payload.get("sentiment", 0.0) or 0.0),
                    "n_articles": int(payload.get("n_articles", 0) or 0),
                    "pre_d18_mult": mult,
                    "error": payload.get("_error", ""),
                }
            )

    df = pd.DataFrame(rows)
    mult_counts = df["pre_d18_mult"].value_counts().sort_index().to_dict()
    non_neutral = df[df["pre_d18_mult"] != 1.0]

    return {
        "store_state": store_state,
        "n_probes": len(df),
        "n_universe": len(set(df["ticker"])),
        "n_dates": len(set(df["as_of"])),
        "mult_value_counts": {float(k): int(v) for k, v in mult_counts.items()},
        "n_non_neutral": int((df["pre_d18_mult"] != 1.0).sum()),
        "non_neutral_sample": (
            non_neutral.head(20).to_dict(orient="records") if len(non_neutral) else []
        ),
        "any_nonzero_sentiment": bool((df["sentiment"] != 0).any()),
        "any_nonzero_n_articles": bool((df["n_articles"] != 0).any()),
        "max_n_articles": int(df["n_articles"].max()),
    }


# ---------------------------------------------------------------------------
# §B — calibration on existing rank_log
# ---------------------------------------------------------------------------
# The calibration "edges + label" scheme that PROB_PROFIT_CALIBRATION_2026-05-28.md
# uses. Fine in the high-prob range (0.85-1.0) where the structural defect lives.
CALIBRATION_BINS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
CALIBRATION_LABELS = [
    "(0.5, 0.6]",
    "(0.6, 0.7]",
    "(0.7, 0.8]",
    "(0.8, 0.85]",
    "(0.85, 0.9]",
    "(0.9, 0.95]",
    "(0.95, 1.0]",
]


def compute_calibration(rank_log_path: Path) -> dict:
    """Compute the 7-bin prob_profit calibration table from a rank_log.csv.

    Hit-rate definition (matches PROB_PROFIT_CALIBRATION_2026-05-28.md
    "Method appendix"): ``actual_otm = (exit_reason == 'otm_expire')``.
    Assigned puts that nevertheless netted positive realized P&L via large
    premium are NOT counted as "OTM" in this analysis — consistent with the
    PR #197 / S38 doc convention.

    Verdict thresholds (pre-declared, same as the doc):
      delta <= 5pp           -> "calibrated"
      5pp < delta <= 10pp    -> "warn"
      delta > 10pp           -> "MISCAL"
    """
    df = pd.read_csv(rank_log_path)
    # Subset to PUT rows that actually had an executed/expired outcome.
    if "option_type" in df.columns:
        df = df[df["option_type"].str.lower() == "put"]
    df = df[df["exit_reason"].notna()]
    df = df[df["prob_profit"].notna()]
    # Bin by the engine's claimed prob_profit
    df = df.copy()
    df["bin"] = pd.cut(
        df["prob_profit"],
        bins=CALIBRATION_BINS,
        labels=CALIBRATION_LABELS,
        include_lowest=False,
        right=True,
    )
    df["actual_otm"] = (df["exit_reason"].astype(str).str.lower() == "otm_expire").astype(int)

    rows: list[dict] = []
    weighted_mad_num = 0.0
    weighted_mad_den = 0
    ok_bins = warn_bins = miscal_bins = 0
    top_bin_delta: float | None = None
    for label in CALIBRATION_LABELS:
        sub = df[df["bin"] == label]
        n = int(len(sub))
        if n == 0:
            rows.append(
                {
                    "bin": label,
                    "n": 0,
                    "engine_mean": None,
                    "actual_otm": None,
                    "delta_pp": None,
                    "verdict": "n/a",
                }
            )
            continue
        engine_mean = float(sub["prob_profit"].mean())
        actual_otm = float(sub["actual_otm"].mean())
        delta_pp = (actual_otm - engine_mean) * 100.0
        abs_delta = abs(delta_pp)
        if abs_delta <= 5.0:
            verdict = "calibrated"
            ok_bins += 1
        elif abs_delta <= 10.0:
            verdict = "warn"
            warn_bins += 1
        else:
            verdict = "MISCAL"
            miscal_bins += 1
        weighted_mad_num += abs_delta * n
        weighted_mad_den += n
        rows.append(
            {
                "bin": label,
                "n": n,
                "engine_mean": engine_mean,
                "actual_otm": actual_otm,
                "delta_pp": delta_pp,
                "verdict": verdict,
            }
        )
        if label == "(0.95, 1.0]":
            top_bin_delta = delta_pp
    weighted_mad = (weighted_mad_num / weighted_mad_den) if weighted_mad_den else None
    return {
        "rank_log": str(rank_log_path),
        "n_total_rows": int(len(df)),
        "table": rows,
        "weighted_mad_pp": weighted_mad,
        "ok_bins": ok_bins,
        "warn_bins": warn_bins,
        "miscal_bins": miscal_bins,
        "top_bin_delta_pp": top_bin_delta,
    }


# ---------------------------------------------------------------------------
# §C — re-run S27 on post-#249 engine
# ---------------------------------------------------------------------------
def rerun_s27(output_dir: Path) -> tuple[Path, dict]:
    """Run the canonical S27 reproducer with the current engine.

    Returns the new rank_log path + the metrics dict. ~20 min on this dev box.
    """
    from backtests.regression.s27_ivpit_24t_100k import run as run_s27

    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_s27(output_dir=output_dir)
    return output_dir / "rank_log.csv", dict(result.metrics)


def discover_existing_rank_logs() -> dict[str, Path]:
    """Find existing per-config ``rank_log.csv`` files under ``%TEMP%``.

    Returns ``{config_label: path}`` for every backtest directory whose
    layout PROB_PROFIT_CALIBRATION_2026-05-28.md describes (Sn × scale).
    Skips configs not present on this box.
    """
    temp_root = Path(os.environ.get("TEMP", tempfile.gettempdir()))
    candidates = {
        "S22 (24t/$100k/2022-2024 pre-PIT)": temp_root / "s22_backtest" / "rank_log.csv",
        "S27 (24t/$100k/2022-2024 post-PIT)": temp_root / "s27_backtest" / "rank_log.csv",
        "S32 (24t/$1M/2022-2024)": temp_root / "s32_backtest" / "rank_log.csv",
        "S34 (100t/$1M/2022-2024)": temp_root / "s34_backtest" / "rank_log.csv",
        "S35 (24t/$100k/2018-2020 OOS)": temp_root / "s35_backtest" / "rank_log.csv",
        "S38 (100t/$1M/2020-2024 pre-F4)": temp_root / "s38_backtest" / "rank_log.csv",
        "S38-postF4 (100t/$1M/2020-2024 post-F4)": temp_root
        / "s38_postf4_backtest"
        / "rank_log.csv",
        "S40-W1 (100t/$1M/2021-2026)": temp_root / "s40_backtest_2021" / "rank_log.csv",
        "S40-W2 (100t/$1M/2022-2026)": temp_root / "s40_backtest_2022" / "rank_log.csv",
        "S40-W3 (100t/$1M/2023-2026)": temp_root / "s40_backtest_2023" / "rank_log.csv",
        "S41 (24t/$100k F4 validation)": temp_root / "s41_backtest" / "rank_log.csv",
        "S43 (rolling, post-#260)": temp_root / "s43_backtest" / "rank_log.csv",
    }
    return {label: p for label, p in candidates.items() if p.exists()}


def compute_multi_config_calibration(rank_log_map: dict[str, Path]) -> list[dict]:
    """Compute calibration for every available config + return summary rows."""
    summary: list[dict] = []
    for label, path in rank_log_map.items():
        try:
            r = compute_calibration(path)
            summary.append(
                {
                    "config": label,
                    "rank_log": str(path),
                    "n_rows": r["n_total_rows"],
                    "weighted_mad_pp": r["weighted_mad_pp"],
                    "ok_bins": r["ok_bins"],
                    "warn_bins": r["warn_bins"],
                    "miscal_bins": r["miscal_bins"],
                    "top_bin_delta_pp": r["top_bin_delta_pp"],
                    "full_table": r,
                }
            )
        except Exception as exc:
            summary.append(
                {
                    "config": label,
                    "rank_log": str(path),
                    "error": repr(exc),
                }
            )
    return summary


def compare_rank_logs(old_path: Path, new_path: Path) -> dict:
    """Row-aligned comparison of two rank_logs on ``prob_profit``.

    The two rank_logs are produced by different engine SHAs (pre- vs post-#249)
    over identical inputs. If ``prob_profit`` is INVARIANT to ``news_mult`` (the
    structural claim) AND ``news_mult`` was 1.0 in both runs (the §A claim),
    the two columns are byte-identical row-for-row.
    """
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    # Align on the natural join key: (entry_date, rank_position, ticker, option_type, strike).
    join_cols = ["entry_date", "rank_position", "ticker", "option_type", "strike"]
    join_cols = [c for c in join_cols if c in old.columns and c in new.columns]
    merged = old.merge(new, on=join_cols, suffixes=("_old", "_new"), how="inner")
    if "prob_profit_old" not in merged or "prob_profit_new" not in merged:
        return {
            "ok": False,
            "reason": "missing prob_profit on one side",
            "old_cols": list(old.columns),
            "new_cols": list(new.columns),
        }
    diff = (merged["prob_profit_old"] - merged["prob_profit_new"]).abs()
    return {
        "ok": True,
        "n_old": int(len(old)),
        "n_new": int(len(new)),
        "n_matched": int(len(merged)),
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
        "mean_abs_diff": float(diff.mean()) if len(diff) else 0.0,
        "n_exact": int((diff == 0).sum()),
        "n_within_1e6": int((diff <= 1e-6).sum()),
        "n_within_1e3": int((diff <= 1e-3).sum()),
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def banner(text: str) -> str:
    bar = "=" * 78
    return f"\n{bar}\n  {text}\n{bar}"


def print_calibration_table(result: dict) -> None:
    print(f"  rank_log    : {result['rank_log']}")
    print(f"  n_rows      : {result['n_total_rows']}")
    print(
        f"  weighted MAD: {result['weighted_mad_pp']:.2f}pp"
        if result["weighted_mad_pp"] is not None
        else "  weighted MAD: n/a"
    )
    print(
        f"  ok / warn / MISCAL bins: {result['ok_bins']} / {result['warn_bins']} / {result['miscal_bins']}"
    )
    if result["top_bin_delta_pp"] is not None:
        print(f"  (0.95, 1.0] delta: {result['top_bin_delta_pp']:+.2f}pp")
    print()
    print(
        f"  {'bin':<13}  {'n':>6}  {'engine_mean':>11}  {'actual_otm':>10}  {'delta_pp':>9}  verdict"
    )
    print(f"  {'-' * 13}  {'-' * 6}  {'-' * 11}  {'-' * 10}  {'-' * 9}  {'-' * 11}")
    for row in result["table"]:
        if row["n"] == 0:
            print(f"  {row['bin']:<13}  {row['n']:>6}  {'-':>11}  {'-':>10}  {'-':>9}  -")
            continue
        print(
            f"  {row['bin']:<13}  {row['n']:>6}  "
            f"{row['engine_mean']:>11.4f}  {row['actual_otm']:>10.4f}  "
            f"{row['delta_pp']:>+9.2f}  {row['verdict']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-rerun",
        action="store_true",
        help="Skip §C (the S27 re-run). §A + §B still run (~3 min total).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write the §C re-run's rank_log.csv + metrics.json here. Defaults to a tempdir.",
    )
    ap.add_argument(
        "--reference-rank-log",
        type=Path,
        default=Path(os.environ.get("TEMP", tempfile.gettempdir()))
        / "s27_backtest"
        / "rank_log.csv",
        help="Existing pre-#249 S27 rank_log used in §B and as the §C comparison baseline.",
    )
    args = ap.parse_args(argv)

    # --- §A ---------------------------------------------------------------
    print(banner("§A. EMPIRICAL news_multiplier distribution (UNIVERSE_100 × 13 dates)"))
    section_a = measure_news_mult_distribution()
    print(f"  store state    : {section_a['store_state']}")
    print(
        f"  total probes   : {section_a['n_probes']} "
        f"({section_a['n_universe']} tickers × {section_a['n_dates']} dates)"
    )
    print(f"  max n_articles : {section_a['max_n_articles']}")
    print(f"  any non-zero sentiment: {section_a['any_nonzero_sentiment']}")
    print(f"  any non-zero n_articles: {section_a['any_nonzero_n_articles']}")
    print(f"  non-neutral cells: {section_a['n_non_neutral']}")
    print(f"  multiplier value counts: {section_a['mult_value_counts']}")
    if section_a["non_neutral_sample"]:
        print("  sample non-neutral cells:")
        for row in section_a["non_neutral_sample"]:
            print(f"    {row}")

    section_a_verdict_no_op = section_a["n_non_neutral"] == 0
    print(
        f"\n  >>> §A VERDICT: pre-D18 news_multiplier was "
        f"{'CONSTANT 1.0 (D18 = no-op for news_mult)' if section_a_verdict_no_op else 'NON-1.0 in some cells — see sample'}."
    )

    # --- §B ---------------------------------------------------------------
    print(banner("§B. Multi-config calibration on existing rank_logs (pre-#249 engine output)"))
    discovered = discover_existing_rank_logs()
    print(f"  configs discovered: {len(discovered)}")
    section_b_multi = compute_multi_config_calibration(discovered)
    print()
    print(f"  {'config':<48}  {'n_rows':>7}  {'MAD':>6}  {'top':>7}  {'OK':>3}/{'W':>2}/{'MIS':>3}")
    print(f"  {'-' * 48}  {'-' * 7}  {'-' * 6}  {'-' * 7}  {'-' * 3}/{'-' * 2}/{'-' * 3}")
    for row in section_b_multi:
        if "error" in row:
            print(f"  {row['config']:<48}  ERROR: {row['error']}")
            continue
        mad = f"{row['weighted_mad_pp']:.2f}" if row["weighted_mad_pp"] is not None else "n/a"
        top = f"{row['top_bin_delta_pp']:+.2f}" if row["top_bin_delta_pp"] is not None else "n/a"
        print(
            f"  {row['config']:<48}  {row['n_rows']:>7}  {mad:>6}  "
            f"{top:>7}  {row['ok_bins']:>3}/{row['warn_bins']:>2}/{row['miscal_bins']:>3}"
        )

    section_b: dict | None = None
    if args.reference_rank_log.exists():
        print()
        print(banner("§B.1. Reference S27 calibration table (detail — comparison baseline for §C)"))
        section_b = compute_calibration(args.reference_rank_log)
        print_calibration_table(section_b)
    else:
        print(f"  WARN: --reference-rank-log not found at {args.reference_rank_log}")

    # --- §C ---------------------------------------------------------------
    section_c_metrics: dict | None = None
    section_c_calibration: dict | None = None
    section_c_compare: dict | None = None
    if args.skip_rerun:
        print(banner("§C. Re-run S27 — SKIPPED (--skip-rerun)"))
    else:
        out_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="htc_s27_rerun_"))
        print(banner(f"§C. Re-run S27 on post-#249 engine → {out_dir} (~20 min)"))
        sys.stdout.flush()
        new_rank_log, section_c_metrics = rerun_s27(out_dir)
        print(f"  metrics: {json.dumps(section_c_metrics, sort_keys=True, default=str)}")
        section_c_calibration = compute_calibration(new_rank_log)
        print()
        print_calibration_table(section_c_calibration)
        if args.reference_rank_log.exists():
            print()
            print("  Row-aligned prob_profit comparison vs existing rank_log:")
            section_c_compare = compare_rank_logs(args.reference_rank_log, new_rank_log)
            for k, v in section_c_compare.items():
                print(f"    {k:<18}: {v}")

    # --- Final synthesis --------------------------------------------------
    print(banner("FINAL SYNTHESIS"))
    print(
        f"  §A: news_multiplier on Bloomberg historically = "
        f"{'1.0 always (D18 no-op)' if section_a_verdict_no_op else 'NON-1.0 — see sample'}."
    )
    if section_b is not None:
        print(
            f"  §B: S27 published calibration reproduces "
            f"(top-bin Δ {section_b.get('top_bin_delta_pp', float('nan')):+.2f}pp, "
            f"MAD {section_b.get('weighted_mad_pp', float('nan')):.2f}pp)."
        )
    else:
        print("  §B: reference rank_log unavailable — skipped.")
    if section_c_calibration is not None:
        print(
            f"  §C: post-#249 S27 calibration "
            f"(top-bin Δ {section_c_calibration.get('top_bin_delta_pp', float('nan')):+.2f}pp, "
            f"MAD {section_c_calibration.get('weighted_mad_pp', float('nan')):.2f}pp)."
        )
    if section_c_compare is not None and section_c_compare.get("ok"):
        max_diff = section_c_compare["max_abs_diff"]
        if max_diff <= 1e-6:
            print(
                f"  §C: prob_profit byte-identical row-for-row "
                f"(max abs diff {max_diff:.2e}) — D18 leaves prob_profit untouched as predicted."
            )
        else:
            print(f"  §C: prob_profit max abs diff = {max_diff:.4f}; investigate.")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
