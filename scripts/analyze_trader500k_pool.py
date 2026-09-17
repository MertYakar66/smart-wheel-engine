"""Recompute the TRADER-500K report's §3 headline rank-quality figures from
the committed per-row bundles — the traceability closure for PR #520.

The campaign report (``docs/TRADER_500K_RELIABILITY_2026-07.md`` §3) headlines a
cross-window pooled Spearman(ev_dollars, realized), its date-clustered CI, and a
regime-conditional split. Those numbers were originally derived in a sandbox
from the committed ``calibration_rows.csv.gz`` files, but the deriving script
was never committed — leaving the headline unreproducible. This script closes
that gap: it reads ONLY committed bundle files and deterministically recomputes

* the per-window pooled Spearman(ev_dollars, realized_pnl_synth), cross-checked
  against each committed ``summary.json`` (pipeline-fidelity proof);
* the cross-window pooled Spearman over the 11 core Phase-A/B windows, with a
  date-clustered bootstrap CI (resamples whole ``as_of`` dates, never rows);
* the same for ``prob_profit``;
* the regime-conditional pooled Spearman per ``hmm_regime`` label, both signals;
* the row-count accounting (raw concat / finite pool pairs / reliability-valid);
* the event-gate share from the committed ``drop_gate_tallies`` fingerprints.

Output: ``docs/verification_artifacts/trader500k/CROSS_WINDOW_SUMMARY.json``
(committed). Byte-stable: fixed seed, no wall-clock, input SHA-256s recorded.

Method notes (mirrors the report's construction — do not "fix" here):

* The pool is the PLAIN CONCATENATION of the 11 core windows. Windows overlap
  at 6-month steps, so a mid-campaign calendar week appears in up to 3 windows
  and is counted that many times. The date-clustered bootstrap puts all rows
  sharing an ``as_of`` — across windows — in ONE cluster, so the duplication is
  carried, not hidden.
* Rails (W05_rail / W09_rail) are different premium provenance and are NEVER
  pooled with the core windows; they get a separate per-window block.
* This is a measurement script: it feeds no trade, no ranking, no ``ev_dollars``
  (CLAUDE.md §2). It never imports the engine.

Usage::

    python scripts/analyze_trader500k_pool.py            # write + print
    python scripts/analyze_trader500k_pool.py --no-write # print only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_ROOT = REPO_ROOT / "docs" / "verification_artifacts" / "trader500k"
OUT_PATH = BUNDLE_ROOT / "CROSS_WINDOW_SUMMARY.json"

CORE_WINDOWS = ("W01", "W02", "W03", "W04", "W05", "W06", "W07", "W08", "W09", "W10", "W11")
RAIL_WINDOWS = ("W05_rail", "W09_rail")
REGIME_LABELS = ("crisis", "bear", "normal", "bull_quiet")

N_BOOT = 2000
SEED = 12345

#: Columns the recomputation needs; their absence is a hard error, never proxied.
REQUIRED_COLUMNS = (
    "as_of",
    "ev_dollars",
    "realized_pnl_synth",
    "prob_profit",
    "engine_exact",
    "hmm_regime",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pooled_rho(sig: np.ndarray, pnl: np.ndarray) -> float:
    """Spearman over jointly-finite pairs; NaN when undefined (n<3/constant)."""
    m = np.isfinite(sig) & np.isfinite(pnl)
    if m.sum() < 3:
        return float("nan")
    s, p = sig[m], pnl[m]
    if np.all(s == s[0]) or np.all(p == p[0]):
        return float("nan")
    return float(spearmanr(s, p).correlation)


def date_clustered_bootstrap_ci(
    df: pd.DataFrame, signal_col: str, *, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Pooled Spearman(signal, realized_pnl_synth) with a date-clustered CI.

    Resamples whole ``as_of`` dates with replacement (never individual rows),
    so same-day cross-sectional dependence — and the cross-WINDOW duplication of
    overlapping weeks — is carried by construction. This is the ``block_len=1``
    date-cluster treatment the report's limitation 5 describes; it does NOT
    block contiguous dates, so serial dependence across nearby weeks is not
    corrected (see ``backtests/parameter_oos.py`` §7.0 for the block upgrade).
    """
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for _, g in df.groupby("as_of", sort=True):
        sig = g[signal_col].to_numpy(dtype=float)
        pnl = g["realized_pnl_synth"].to_numpy(dtype=float)
        m = np.isfinite(sig) & np.isfinite(pnl)
        if m.sum() >= 3:
            groups.append((sig[m], pnl[m]))
    n_dates = len(groups)
    point = _pooled_rho(
        np.concatenate([g[0] for g in groups]), np.concatenate([g[1] for g in groups])
    )
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_dates, size=n_dates)
        boots[b] = _pooled_rho(
            np.concatenate([groups[i][0] for i in idx]),
            np.concatenate([groups[i][1] for i in idx]),
        )
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "point": point,
        "ci95": [float(lo), float(hi)],
        "se": float(np.std(boots, ddof=1)),
        "n_boot": int(boots.size),
        "n_dates_clusters": n_dates,
    }


def load_window(window: str) -> tuple[pd.DataFrame, dict]:
    """One window's committed calibration rows + its committed summary.json."""
    rows_path = BUNDLE_ROOT / window / "calibration_rows.csv.gz"
    summary_path = BUNDLE_ROOT / window / "summary.json"
    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{rows_path}: missing required column(s) {missing} — the §3 figures "
            "cannot be reproduced from this bundle; do NOT substitute a proxy."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return df, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-write", action="store_true", help="print only; do not write the JSON")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="output path override")
    args = ap.parse_args()

    inputs: dict[str, str] = {}
    per_window: dict[str, dict] = {}
    core_frames: list[pd.DataFrame] = []
    event_drops = 0
    universe_weeks = 0

    for w in CORE_WINDOWS + RAIL_WINDOWS:
        df, summary = load_window(w)
        for name in ("calibration_rows.csv.gz", "summary.json"):
            rel = f"docs/verification_artifacts/trader500k/{w}/{name}"
            inputs[rel] = _sha256(REPO_ROOT / rel)
        recomputed = _pooled_rho(
            df["ev_dollars"].to_numpy(dtype=float),
            df["realized_pnl_synth"].to_numpy(dtype=float),
        )
        committed = summary["calibration"]["spearman_ev_realized_pool"]["rho"]
        per_window[w] = {
            "n_rows": int(len(df)),
            "pool_rho_recomputed": recomputed,
            "pool_rho_committed": committed,
            "abs_delta": abs(recomputed - committed),
        }
        if w in CORE_WINDOWS:
            core_frames.append(df)
            event_drops += int(summary["drop_gate_tallies"].get("event", 0))
            universe_weeks += int(
                summary["fingerprint"]["universe_size"] * summary["fingerprint"]["n_rank_days_run"]
            )

    pool = pd.concat(core_frames, ignore_index=True)
    ev = pool["ev_dollars"].to_numpy(dtype=float)
    pnl = pool["realized_pnl_synth"].to_numpy(dtype=float)
    prob = pool["prob_profit"].to_numpy(dtype=float)
    exact = pool["engine_exact"].to_numpy(dtype=float)

    n_pool_pairs = int((np.isfinite(ev) & np.isfinite(pnl)).sum())
    reliability_mask = np.isfinite(prob) & np.isfinite(exact)
    n_reliability = int(reliability_mask.sum())
    n_prob_below_0p5 = int((prob[reliability_mask] < 0.5).sum())

    ev_ci = date_clustered_bootstrap_ci(pool, "ev_dollars")
    prob_ci = date_clustered_bootstrap_ci(pool, "prob_profit")

    by_regime_ev: dict[str, dict] = {}
    by_regime_prob: dict[str, dict] = {}
    for lbl in REGIME_LABELS:
        sub = pool[pool["hmm_regime"] == lbl]
        s_ev = sub["ev_dollars"].to_numpy(dtype=float)
        s_prob = sub["prob_profit"].to_numpy(dtype=float)
        s_pnl = sub["realized_pnl_synth"].to_numpy(dtype=float)
        by_regime_ev[lbl] = {"n": int(len(sub)), "rho": _pooled_rho(s_ev, s_pnl)}
        by_regime_prob[lbl] = {"n": int(len(sub)), "rho": _pooled_rho(s_prob, s_pnl)}
    n_regime_unknown = int((~pool["hmm_regime"].isin(REGIME_LABELS)).sum())

    result = {
        "script": "scripts/analyze_trader500k_pool.py",
        "method": {
            "pool": "plain concatenation of the 11 core Phase-A/B windows; overlapping "
            "calendar weeks are counted once per containing window (matches the "
            "report's construction)",
            "signal_outcome": "spearman(<signal>, realized_pnl_synth) over jointly-finite rows",
            "ci": "date-clustered bootstrap: resample whole as_of dates with replacement "
            "(block_len=1 — no contiguous-date blocking), percentile 95% CI",
            "n_boot": N_BOOT,
            "seed": SEED,
            "regime_key": "per-row hmm_regime label",
            "rails_excluded_from_pool": list(RAIL_WINDOWS),
        },
        "inputs_sha256": inputs,
        "per_window": per_window,
        "cross_window": {
            "n_rows_concat": int(len(pool)),
            "n_pool_pairs_finite": n_pool_pairs,
            "n_reliability_rows": n_reliability,
            "n_reliability_prob_below_0p5": n_prob_below_0p5,
            "n_regime_unknown": n_regime_unknown,
            "pooled_spearman_ev": ev_ci,
            "pooled_spearman_prob": prob_ci,
            "by_regime_ev": by_regime_ev,
            "by_regime_prob": by_regime_prob,
        },
        "event_gate": {
            "event_drops": event_drops,
            "universe_weeks": universe_weeks,
            "share": event_drops / universe_weeks,
        },
    }

    if not args.no_write:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"windows pooled: {len(CORE_WINDOWS)} core (rails separate: {list(RAIL_WINDOWS)})")
    print(
        f"rows: concat={len(pool)}  finite ev/pnl pairs={n_pool_pairs}  "
        f"reliability-valid={n_reliability}  (of which prob<0.5: {n_prob_below_0p5})"
    )
    e = ev_ci
    print(
        f"pooled Spearman(ev_dollars, realized): {e['point']:+.4f}  "
        f"CI95 [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]  "
        f"(se {e['se']:.4f}, {e['n_dates_clusters']} date clusters, n_boot {e['n_boot']})"
    )
    p = prob_ci
    print(
        f"pooled Spearman(prob_profit, realized): {p['point']:+.4f}  "
        f"CI95 [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
    )
    print("by regime (ev_dollars | prob_profit):")
    for lbl in REGIME_LABELS:
        r_ev, r_pp = by_regime_ev[lbl], by_regime_prob[lbl]
        print(f"  {lbl:10s} n={r_ev['n']:6d}  ev {r_ev['rho']:+.4f}  prob {r_pp['rho']:+.4f}")
    print(f"  (unknown label: {n_regime_unknown} rows, excluded from the split)")
    print(
        f"event gate: {event_drops} / {universe_weeks} name-weeks = "
        f"{100 * event_drops / universe_weeks:.2f}%"
    )
    print(
        "per-window recompute vs committed summary.json (max |delta|): "
        f"{max(v['abs_delta'] for v in per_window.values()):.3e}"
    )
    if not args.no_write:
        print(f"wrote {args.out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
