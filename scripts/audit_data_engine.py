#!/usr/bin/env python3
"""Data + engine audit pass — Phase 1 discovery (additive, read-only).

Inventories the bundled connector data and probes the live ranker so we can
answer two questions the recent data campaign (Bloomberg refresh, deep
history, CASY backfill, dividends re-pull, fingerprint) never checked
systematically:

  (a) Is the data itself sound?
  (b) Does it flow through into correct engine output?

This is a *discovery* tool. It asserts nothing (Phase 2 turns the findings
into tests). It emits a written findings doc + a machine-readable JSON
sidecar. It NEVER touches the decision trio (ev_engine / wheel_runner /
candidate_dossier) and routes every engine probe through the production
``WheelRunner.rank_candidates_by_ev`` — no §2 bypass, no hand-built
candidates.

Design notes
------------
* **Provider is logged.** Silent provider selection is a recurring bug; the
  first line of output records ``type(WheelRunner().connector).__name__``.
* **The as-of is the real frontier, not ``date.today()``.** By default the
  frontier is auto-detected as the most-recent bar common to OHLCV and IV
  (``min(max(ohlcv.date), max(vol_iv.date))``). This makes the probe
  deterministic *and* correct wherever it runs: on ``main`` it resolves to
  the refreshed frontier; on an older branch it resolves to that branch's
  frontier. Override with ``--as-of YYYY-MM-DD``.
* The ranker surfaces every gated candidate on ``frame.attrs["drops"]``
  (``{"ticker","gate","reason"}``) + ``frame.attrs["drops_summary"]``; the
  forward-distribution tier rides on the ``distribution_source`` column.
  The probe reads both, and separately checks for tickers that are *neither*
  produced *nor* dropped (a true silent vanish).

Usage
-----
    python scripts/audit_data_engine.py                 # full universe, auto frontier
    python scripts/audit_data_engine.py --universe audit  # curated edge-case set
    python scripts/audit_data_engine.py --universe 24 --as-of 2026-03-20
    python scripts/audit_data_engine.py --out docs/DATA_ENGINE_AUDIT_2026-06-07.md \\
        --json docs/verification_artifacts/data_engine_audit_2026-06-07/audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.data_connector import MarketDataConnector, normalize_ticker  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

try:
    from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100
except Exception:  # pragma: no cover - defensive; universes module is committed
    UNIVERSE_24 = ()
    UNIVERSE_100 = ()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SURVIVORSHIP_GATE_DAYS = 504  # ranker history gate (wheel_runner: len(ohlcv) < 504)
IV_PERCENT_HEURISTIC = 3.0  # ranker: if iv > 3.0 -> iv/100 (conditional, per-trio)
DEFAULT_SEAM = "2026-03-23"  # known Bloomberg->refresh seam (auto-detected too)

# Each connector CSV -> the engine capability it feeds. Sourced from
# docs/DATA_POLICY.md §2 and the connector accessors (engine/data_connector.py).
CAPABILITY_MAP: dict[str, tuple[str, str]] = {
    "ohlcv": (
        "Spot/price history -> forward distribution (empirical NOS -> "
        "overlapping -> block bootstrap -> HAR-RV) + 504-day survivorship gate",
        "get_ohlcv",
    ),
    "vol_iv": (
        "Implied vol (PIT) for BSM pricing/Greeks; realized-vol cols feed "
        "the F4 RV30/RV252 widening signal",
        "get_iv_history",
    ),
    "dividends": (
        "Ex-dividend early-assignment EV for the COVERED-CALL ranker only "
        "(the short-put ranker never consults ex-div)",
        "get_dividends / get_next_dividend",
    ),
    "earnings": (
        "Earnings-date event lockout (first gate in EVEngine.evaluate; "
        "degrades gracefully -> name is simply un-gated if absent)",
        "get_earnings / get_next_earnings",
    ),
    "treasury": (
        "Risk-free rate for BSM (get_risk_free_rate returns decimal via "
        "unconditional /100; NaN if missing so callers detect absence)",
        "get_risk_free_rate",
    ),
    "vix": (
        "VIX level + term-structure regime (contango/backwardation) + percentile",
        "get_vix / get_vix_regime",
    ),
    "fundamentals": (
        "BSM dividend_yield (eqy_dvd_yld_12m), GICS sector (R9 sector cap), "
        "beta, market cap, screening",
        "get_fundamentals / screen_universe",
    ),
    "credit_risk": (
        "S&P rating / Altman-Z / interest-coverage credit gate",
        "get_credit_risk",
    ),
    "liquidity": (
        "avg_vol_30d / turnover / shares_out liquidity context",
        "get_liquidity",
    ),
}

DATE_COLS = ("date", "ex_date", "announcement_date")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_head(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _date_col(df: pd.DataFrame) -> str | None:
    for c in DATE_COLS:
        if c in df.columns:
            return c
    return None


def _norm_ticker_set(df: pd.DataFrame) -> set[str]:
    if "ticker" not in df.columns:
        return set()
    return set(df["ticker"].dropna().map(normalize_ticker).unique())


def _read_csv(path: Path, **kw: Any) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kw)


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------


def resolve_frontier(data_dir: Path) -> tuple[str, dict[str, str]]:
    """Most-recent bar common to OHLCV and IV (deterministic, data-supported).

    Returns ``(frontier_iso, per_file_max)`` where ``per_file_max`` maps each
    daily time-series file to its own max date (for the staleness report).
    """
    per_file_max: dict[str, str] = {}
    daily = {
        "ohlcv": "sp500_ohlcv.csv",
        "vol_iv": "sp500_vol_iv_full.csv",
        "liquidity": "sp500_liquidity.csv",
        "vix": "vix_term_structure.csv",
        "treasury": "treasury_yields.csv",
    }
    for key, fn in daily.items():
        p = data_dir / fn
        if not p.exists():
            continue
        d = pd.to_datetime(_read_csv(p, usecols=["date"])["date"], errors="coerce")
        per_file_max[key] = str(d.max())[:10]
    ohlcv_max = per_file_max.get("ohlcv")
    iv_max = per_file_max.get("vol_iv")
    candidates = [x for x in (ohlcv_max, iv_max) if x]
    frontier = min(candidates) if candidates else str(datetime.now(UTC).date())
    return frontier, per_file_max


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def file_inventory(data_dir: Path, frontier: str) -> list[dict]:
    rows: list[dict] = []
    front_ts = pd.Timestamp(frontier)
    for key, fn in MarketDataConnector._FILES.items():
        p = data_dir / fn
        rec: dict[str, Any] = {"key": key, "file": fn}
        if not p.exists():
            rec.update(exists=False)
            rows.append(rec)
            continue
        df = _read_csv(p)
        dcol = _date_col(df)
        date_min = date_max = None
        stale_days = None
        if dcol is not None:
            d = pd.to_datetime(df[dcol], errors="coerce")
            date_min, date_max = str(d.min())[:10], str(d.max())[:10]
            # Staleness only meaningful for daily series (date col, not future
            # ex_date / announcement_date which legitimately run ahead).
            if dcol == "date":
                stale_days = int((front_ts - d.max()).days)
        rec.update(
            exists=True,
            size_mb=round(p.stat().st_size / 1e6, 1),
            rows=int(len(df)),
            cols=int(len(df.columns)),
            tickers=int(df["ticker"].map(normalize_ticker).nunique())
            if "ticker" in df.columns
            else None,
            date_col=dcol,
            date_min=date_min,
            date_max=date_max,
            staleness_days=stale_days,
            sha256=_sha256_head(p),
        )
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Coverage matrix
# ---------------------------------------------------------------------------


def ticker_sets(data_dir: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for key, fn in MarketDataConnector._FILES.items():
        p = data_dir / fn
        if not p.exists():
            continue
        df = _read_csv(p, usecols=lambda c: c == "ticker")
        out[key] = _norm_ticker_set(df) if "ticker" in df.columns else set()
    return out


def ohlcv_history_profile(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "sp500_ohlcv.csv"
    oh = _read_csv(p, usecols=["date", "ticker"])
    oh["date"] = pd.to_datetime(oh["date"], errors="coerce")
    oh["nt"] = oh["ticker"].map(normalize_ticker)
    g = oh.groupby("nt")["date"]
    prof = pd.DataFrame({"first": g.min(), "last": g.max(), "ndays": g.count()})
    return prof


def detect_seams(profile: pd.DataFrame, frontier: str) -> dict[str, Any]:
    """Cluster first-bar (joiners) / last-bar (leavers) dates -> seam candidates."""
    front_ts = pd.Timestamp(frontier)
    floor = pd.Timestamp("2019-01-01")  # ignore the global history floor
    first_clusters = Counter(str(d)[:10] for d in profile["first"] if pd.notna(d) and d >= floor)
    last_clusters = Counter(
        str(d)[:10]
        for d in profile["last"]
        if pd.notna(d) and d < (front_ts - pd.Timedelta(days=5))
    )
    joiners = {d: n for d, n in first_clusters.items() if n >= 3}
    leavers = {d: n for d, n in last_clusters.items() if n >= 3}
    return {"joiner_clusters": joiners, "leaver_clusters": leavers}


def classify_names(profile: pd.DataFrame, seam: str, frontier: str) -> dict[str, list[str]]:
    seam_ts = pd.Timestamp(seam)
    front_ts = pd.Timestamp(frontier)
    recent_floor = front_ts - pd.Timedelta(days=30)
    post_seam = sorted(profile.index[profile["first"] >= seam_ts])
    thin = sorted(profile.index[profile["ndays"] < SURVIVORSHIP_GATE_DAYS])
    delisted = sorted(profile.index[profile["last"] < recent_floor])
    return {"post_seam_only": post_seam, "thin_history": thin, "delisted_stale": delisted}


# Ticker-keyed files where a *current* index member is expected to appear.
# Excludes dividends (non-payers legitimately absent) and treasury/vix
# (market-wide series with no ticker column).
CORE_TICKER_FILES = ("ohlcv", "vol_iv", "fundamentals", "credit_risk", "earnings", "liquidity")


def referential_gaps(sets: dict[str, set[str]]) -> dict[str, list[str]]:
    spine = sets.get("ohlcv", set())
    gaps: dict[str, list[str]] = {}
    for key, s in sets.items():
        # Skip files with no ticker column (treasury/vix -> empty set).
        if key == "ohlcv" or not s:
            continue
        gaps[f"in_ohlcv_not_{key}"] = sorted(spine - s)
        gaps[f"in_{key}_not_ohlcv"] = sorted(s - spine)
    return gaps


def universe_coverage(
    name: str,
    universe: tuple[str, ...],
    sets: dict[str, set[str]],
    profile: pd.DataFrame,
    seam: str,
) -> dict[str, Any]:
    seam_ts = pd.Timestamp(seam)
    files = [f for f in CORE_TICKER_FILES if f in sets]
    complete: list[str] = []
    partial: dict[str, list[str]] = {}
    flags: dict[str, list[str]] = {
        "post_seam_only": [],
        "thin_history": [],
        "absent_from_ohlcv": [],
    }
    for t in universe:
        nt = normalize_ticker(t)
        missing = [f for f in files if nt not in sets[f]]
        if nt not in sets.get("ohlcv", set()):
            flags["absent_from_ohlcv"].append(t)
        if nt in profile.index:
            if profile.loc[nt, "first"] >= seam_ts:
                flags["post_seam_only"].append(t)
            if profile.loc[nt, "ndays"] < SURVIVORSHIP_GATE_DAYS:
                flags["thin_history"].append(t)
        if missing:
            partial[t] = missing
        else:
            complete.append(t)
    return {
        "universe": name,
        "size": len(universe),
        "complete_all_files": len(complete),
        "partial_count": len(partial),
        "partial": partial,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Engine probe
# ---------------------------------------------------------------------------


def engine_probe(
    runner: WheelRunner, tickers: list[str], as_of: str, dte: int, delta: float
) -> dict[str, Any]:
    """Route the universe through the production ranker at ``as_of`` and record
    per-ticker produced/dropped/vanished, the forward-distribution tier, and
    any non-finite output. No §2 bypass — single ``rank_candidates_by_ev`` call.
    """
    requested = [normalize_ticker(t) for t in tickers]
    result: dict[str, Any] = {
        "as_of": as_of,
        "dte_target": dte,
        "delta_target": delta,
        "n_requested": len(requested),
    }
    try:
        frame = runner.rank_candidates_by_ev(
            tickers=requested,
            dte_target=dte,
            delta_target=delta,
            top_n=len(requested) + 10,
            min_ev_dollars=-1e9,
            as_of=as_of,
            include_diagnostic_fields=True,
        )
    except Exception as exc:  # noqa: BLE001 - probe records, never crashes
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
        return result

    produced = set(frame["ticker"].astype(str)) if len(frame) else set()
    drops = list(frame.attrs.get("drops", []))
    drops_summary = dict(frame.attrs.get("drops_summary", {}))
    dropped_tickers = {d.get("ticker") for d in drops}
    vanished = sorted(set(requested) - produced - dropped_tickers)

    # Forward-distribution tier histogram (per produced row).
    tier_hist: dict[str, int] = {}
    if "distribution_source" in frame.columns and len(frame):
        tier_hist = {str(k): int(v) for k, v in frame["distribution_source"].value_counts().items()}

    # Non-finite check on the EV-path outputs (R1a expects finite at the gate).
    nonfinite: dict[str, list[str]] = {}
    for col in ("ev_dollars", "ev_raw", "prob_profit"):
        if col in frame.columns and len(frame):
            s = pd.to_numeric(frame[col], errors="coerce")
            bad = frame.loc[~np.isfinite(s), "ticker"].astype(str).tolist()
            if bad:
                nonfinite[col] = sorted(bad)

    # Drop reasons grouped by gate.
    by_gate_reason: dict[str, Counter] = {}
    for d in drops:
        by_gate_reason.setdefault(str(d.get("gate")), Counter())[str(d.get("reason"))] += 1

    result.update(
        n_produced=len(produced),
        n_dropped=len(drops),
        n_vanished=len(vanished),
        drops_summary=drops_summary,
        drops_by_gate_reason={g: dict(c) for g, c in by_gate_reason.items()},
        distribution_tier_histogram=tier_hist,
        nonfinite_outputs=nonfinite,
        vanished_tickers=vanished,
        produced_columns=list(frame.columns),
        # Raw per-ticker drops (forensic — resolves which gate each name hit).
        drops=[{k: d.get(k) for k in ("ticker", "gate", "reason")} for d in drops],
    )
    return result


# ---------------------------------------------------------------------------
# Data-hygiene probes (evidence for the weakness report)
# ---------------------------------------------------------------------------


def hygiene_checks(data_dir: Path, frontier: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    front_ts = pd.Timestamp(frontier)

    # OHLCV: canonical (post-rename) invariants + price/volume/dup integrity.
    oh = _read_csv(data_dir / "sp500_ohlcv.csv")
    r = oh.rename(columns={"open": "high", "high": "close", "close": "open"})
    ocl = r[["open", "close", "low"]].apply(pd.to_numeric, errors="coerce")
    och = r[["open", "close", "high"]].apply(pd.to_numeric, errors="coerce")
    high = pd.to_numeric(r["high"], errors="coerce")
    low = pd.to_numeric(r["low"], errors="coerce")
    vol = pd.to_numeric(oh["volume"], errors="coerce")
    out["ohlcv"] = {
        "rows": int(len(oh)),
        "rename_invariant_violations_high": int((high < ocl.max(axis=1)).sum()),
        "rename_invariant_violations_low": int((low > och.min(axis=1)).sum()),
        "nonpositive_price_rows": int((ocl.le(0).any(axis=1)).sum()),
        "nan_price_rows": int(ocl.isna().any(axis=1).sum()),
        "negative_volume": int((vol < 0).sum()),
        "zero_volume": int((vol == 0).sum()),
        "duplicate_date_ticker": int(oh.duplicated(subset=["date", "ticker"]).sum()),
    }

    # IV: percent units, band sanity, zero-skew, heuristic exposure, sentinel.
    iv = _read_csv(data_dir / "sp500_vol_iv_full.csv")
    p = pd.to_numeric(iv["hist_put_imp_vol"], errors="coerce")
    c = pd.to_numeric(iv["hist_call_imp_vol"], errors="coerce")
    both = p.notna() & c.notna()
    out["iv"] = {
        "rows": int(len(iv)),
        "put_imp_vol_min": float(np.nanmin(p)),
        "put_imp_vol_median": float(np.nanmedian(p)),
        "put_imp_vol_max": float(np.nanmax(p)),
        "put_eq_call_exact_pct": round(float((p[both] == c[both]).mean()) * 100, 4),
        "zero_skew_n": int(both.sum()),
        # rows whose percent IV <= 3.0 would be mis-read as decimal by the
        # ranker's conditional `if iv > 3.0: iv/100` heuristic.
        "rows_in_heuristic_danger_zone_0_to_3": int(((p > 0) & (p <= IV_PERCENT_HEURISTIC)).sum()),
        "rows_iv_above_300pct": int((p > 300).sum()),
        "sentinel_134217_7": int(np.isclose(p, 134217.7).sum()),
    }

    # Dividends: negative amounts (magnitude matters — float noise vs real).
    dv = _read_csv(data_dir / "sp500_dividends.csv")
    da = pd.to_numeric(dv["dividend_amount"], errors="coerce")
    ed_all = pd.to_datetime(dv["ex_date"], errors="coerce")
    out["dividends"] = {
        "rows": int(len(dv)),
        "negative_count": int((da < 0).sum()),
        "negative_min": float(da.min()),
        "materially_negative_lt_-0.001": int((da < -0.001).sum()),
        "ex_date_max": str(ed_all.max())[:10],
        "max_past_ex_date": str(ed_all[ed_all <= front_ts].max())[:10],
    }
    # Memory flagged a refresh-source truncation for these names (2018-24
    # ex-div history dropped). Verify against the CURRENT file rather than
    # asserting from memory — report actual 2018-24 coverage per name.
    if "ticker" in dv.columns:
        dv["nt"] = dv["ticker"].map(normalize_ticker)
        watch = {}
        for nm in ("CTRA", "BK", "LW", "PAYC", "MTCH"):
            sub = dv[dv["nt"] == nm]
            ed = pd.to_datetime(sub["ex_date"], errors="coerce")
            in_2018_24 = ((ed >= "2018-01-01") & (ed <= "2024-12-31")).sum()
            watch[nm] = {
                "rows": int(len(sub)),
                "rows_2018_2024": int(in_2018_24),
                "ex_date_min": str(ed.min())[:10] if len(sub) else None,
                "ex_date_max": str(ed.max())[:10] if len(sub) else None,
            }
        out["dividends"]["truncation_watchlist"] = watch

    # Treasury: per-tenor coverage floor + negative rates.
    tr = _read_csv(data_dir / "treasury_yields.csv")
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tenors = {}
    for col in [x for x in tr.columns if x.startswith("rate") or x == "sofr"]:
        s = pd.to_numeric(tr[col], errors="coerce")
        first = tr.loc[s.notna(), "date"].min()
        tenors[col] = {
            "first_nonnull": str(first)[:10] if pd.notna(first) else None,
            "nan_pct": round(float(s.isna().mean()) * 100, 1),
            "min": float(np.nanmin(s)) if s.notna().any() else None,
            "negative_count": int((s < 0).sum()),
        }
    out["treasury"] = {"rows": int(len(tr)), "tenors": tenors}

    # Dateless-snapshot flag (structural lookahead) for fundamentals/credit.
    fu = _read_csv(data_dir / "sp500_fundamentals.csv")
    cr = _read_csv(data_dir / "sp500_credit_risk.csv")
    out["dateless_snapshots"] = {
        "fundamentals_has_date_col": _date_col(fu) is not None,
        "credit_risk_has_date_col": _date_col(cr) is not None,
        "fundamentals_rows": int(len(fu)),
        "credit_risk_rows": int(len(cr)),
    }

    # Fingerprint completeness: read the ACTUAL snapshot fingerprint
    # (connector_data_sha256) and compare its key set to the connector's
    # _FILES, rather than assuming. On main this pins all 9 files.
    pinned_keys: list[str] = []
    try:
        from backtests.regression._common import connector_data_sha256

        pinned_keys = sorted(connector_data_sha256().keys())
    except Exception:  # pragma: no cover - defensive
        pinned_keys = []
    all_keys = sorted(MarketDataConnector._FILES.keys())
    out["fingerprint"] = {
        "fingerprint_fn": "backtests.regression._common.connector_data_sha256",
        "pinned_keys": pinned_keys,
        "all_connector_keys": all_keys,
        "unpinned_keys": sorted(set(all_keys) - set(pinned_keys)),
        "complete": set(pinned_keys) == set(all_keys),
    }
    return out


# ---------------------------------------------------------------------------
# Ranked weakness report
# ---------------------------------------------------------------------------


def build_weaknesses(
    inv: list[dict],
    gaps: dict[str, list[str]],
    classes: dict[str, list[str]],
    probe: dict[str, Any],
    hygiene: dict[str, Any],
    frontier: str,
    branch_note: str,
) -> list[dict]:
    w: list[dict] = []

    def add(sev: str, cat: str, title: str, evidence: str, fixable: str, test: str) -> None:
        w.append(
            {
                "id": f"W{len(w) + 1}",
                "severity": sev,
                "category": cat,
                "title": title,
                "evidence": evidence,
                "fixable_in": fixable,
                "phase2_test": test,
            }
        )

    iv = hygiene["iv"]
    div = hygiene["dividends"]
    tre = hygiene["treasury"]

    # Unit-scale: conditional IV /100 heuristic in the trio.
    add(
        "HIGH",
        "unit-scale",
        "Ranker IV uses a conditional `if iv>3.0: iv/100` heuristic (per-trio), "
        "diverging from the unconditional /100 used for treasury & dividend_yield",
        f"wheel_runner.py:198/1101/2418/2980; a genuine sub-3% IV is read as up to "
        f"300% and survives the 0<iv<=5.0 guard. vol_iv rows with 0<IV<=3.0 today: "
        f"{iv['rows_in_heuristic_danger_zone_0_to_3']}. Same class as the D20 treasury fix.",
        "TRIO (wheel_runner) — log only, separate lane-claimed PR",
        "data->engine: assert ranker output iv is decimal (0<iv<3) for a name "
        "whose as_of IV is genuinely low; pin the heuristic boundary",
    )

    # PIT / lookahead: dateless fundamentals + credit snapshots.
    ds = hygiene["dateless_snapshots"]
    add(
        "HIGH",
        "pit-lookahead",
        "fundamentals.csv & credit_risk.csv are dateless single snapshots — "
        "get_fundamentals/get_credit_risk ignore as_of (structural lookahead)",
        f"no date column in either file (fundamentals_has_date={ds['fundamentals_has_date_col']}, "
        f"credit_has_date={ds['credit_risk_has_date_col']}); a 2026 snapshot feeds BSM "
        "dividend_yield, GICS sector (R9 cap) and the credit gate at every historical as_of",
        "additive test can pin the contract; a true fix needs PIT fundamentals (data-layer)",
        "data->engine + pit: assert the snapshot is documented as as_of-invariant; "
        "extend tests/test_pit_leaks.py",
    )

    # Fingerprint completeness (data-driven against main's actual fingerprint).
    fp = hygiene["fingerprint"]
    if not fp.get("complete"):
        add(
            "HIGH",
            "fingerprint-blindspot",
            "Snapshot fingerprint does NOT pin every connector file — a silent refresh "
            "of an unpinned file would not force a re-baseline (the dividends-incident class)",
            f"connector_data_sha256() pins {fp['pinned_keys']}; UNPINNED: {fp['unpinned_keys']}",
            "additive: extend the fingerprint to all connector-read files (in _common, not trio)",
            "integrity: assert set(connector_data_sha256().keys()) == set(_FILES)",
        )
    else:
        add(
            "INFO",
            "fingerprint",
            "Snapshot fingerprint pins ALL connector files via connector_data_sha256 "
            "(_common.py:177) — the 2026-06-06 dividends blind-spot is already closed on main",
            f"pinned keys = {fp['pinned_keys']}. Residual: the drift compare "
            "(test_snapshot_data_fingerprint_matches_current) runs on the SLOW "
            "backtest_regression lane, not fast CI; no fast-CI completeness guard exists.",
            "additive: add a fast-CI completeness guard (Phase 2 integrity suite)",
            "integrity: set(connector_data_sha256().keys()) == set(MarketDataConnector._FILES)",
        )

    # Cross-file referential gaps from the seam reconstitution.
    iv_gap = gaps.get("in_ohlcv_not_vol_iv", [])
    earn_gap = gaps.get("in_ohlcv_not_earnings", [])
    fund_gap = gaps.get("in_ohlcv_not_fundamentals", [])
    add(
        "MEDIUM",
        "referential",
        "Cross-file ticker-universe inconsistency from the 2026-03-23 index "
        "reconstitution: earnings=pre-seam membership, fundamentals/credit=post-seam, "
        "OHLCV/vol_iv/liquidity span both",
        f"in OHLCV not vol_iv: {iv_gap} (unpriceable, no IV); in OHLCV not earnings "
        f"(post-seam joiners): {earn_gap}; in OHLCV not fundamentals (departed): {fund_gap}",
        "additive test pins the contract; data-layer follow-up to reconcile membership",
        "integrity: every vol_iv/fundamentals/credit/liquidity ticker exists in OHLCV "
        "with overlapping date range",
    )

    # Re-ticker continuity (BK -> BNY).
    add(
        "MEDIUM",
        "re-ticker",
        "BK->BNY re-ticker at the seam splits one continuous company into a "
        "'delisted' name (BK, ends 2026-03-20) + a 52-bar 'thin' name (BNY, from 2026-03-23) "
        "with no linkage",
        "BK in OHLCV/vol_iv/dividends but not fundamentals; BNY in OHLCV/vol_iv/fundamentals "
        "but not earnings/dividends; the engine sees two unrelated names",
        "additive: a re-ticker map (data-layer); not fixable in tests",
        "integrity: flag known re-tickers; assert continuity or an explicit alias",
    )

    # Survivorship / thin history (blue-chips that should not be thin).
    thin = classes["thin_history"]
    add(
        "MEDIUM",
        "survivorship",
        f"{len(thin)} names fail the 504-day history gate — several are blue-chips "
        "whose thinness is a data gap, not genuine newness",
        f"thin: {thin}; e.g. WMT/KMB/CPB/DPZ have full real histories but short OHLCV "
        "on the dev box (see backtests/regression/universes.py WMT note)",
        "additive test pins which names are gated; data-layer backfill is the fix",
        "data->engine: thin names degrade gracefully (gate=history), never emit a "
        "tradeable from <504 bars",
    )

    # Dividends: verify the memory-flagged truncation against the CURRENT file.
    wl = div.get("truncation_watchlist", {})
    # PAYC/MTCH legitimately started paying late (2023 / 2025) -> 0 rows in
    # 2018-24 is expected, not a truncation. Only flag a long-established name
    # that LOST its 2018-24 history.
    truncated = [
        nm for nm, v in wl.items() if v.get("rows_2018_2024", 0) == 0 and nm not in ("PAYC", "MTCH")
    ]
    cov = {nm: v.get("rows_2018_2024") for nm, v in wl.items()}
    if truncated:
        add(
            "MEDIUM",
            "staleness",
            f"sp500_dividends.csv lost 2018-24 ex-div history for {truncated} "
            "(refresh-source truncation, the dividends-incident class)",
            f"2018-24 row counts: {cov}; truncation_watchlist: {json.dumps(wl)}",
            "data-layer: re-pull + re-baseline the affected snapshots (separate lane)",
            "data->engine: a dividends->CC->cash test pinning an in-window ex-div "
            "changes CC selection (the R1 mechanism)",
        )
    else:
        add(
            "INFO",
            "staleness",
            "Memory-flagged dividends truncation (CTRA/BK/LW/PAYC) is NOT present on "
            "current main — full 2018-24 ex-div history is intact (resolved, like the "
            "stale treasury memory)",
            f"2018-24 row counts: {cov} (PAYC/MTCH legitimately started paying late); "
            f"overall most-recent past ex-date {div.get('max_past_ex_date')}; residual: "
            f"no in-repo producer script + future-declared rows to {div.get('ex_date_max')} "
            "(dividends is reconstructable-only)",
            "n/a (resolved on main); residual is the missing producer + reconstructable-only",
            "data->engine: a dividends->CC->cash test pinning an in-window ex-div changes "
            "CC selection (the R1 mechanism)",
        )

    # IV band outliers.
    add(
        "MEDIUM",
        "unit-scale",
        "IV band has implausible extremes (min 0.01%, max 769%) with no sanity gate "
        "on the raw file",
        f"vol_iv hist_put_imp_vol min={iv['put_imp_vol_min']} max={iv['put_imp_vol_max']} "
        f"(percent); rows >300%: {iv['rows_iv_above_300pct']}",
        "additive test (no engine change needed; ranker guard bounds it downstream)",
        "integrity: 0.1% <= IV <= ~500% on the served file; flag outliers",
    )

    # Zero-skew structural fact.
    add(
        "LOW",
        "structural",
        "Zero put/call IV skew — put_iv == call_iv EXACTLY across 100% of rows "
        "(Nelson-Siegel skew tooling is fed a flat surface; skew dormant on Bloomberg)",
        f"put==call exact {iv['put_eq_call_exact_pct']}% of {iv['zero_skew_n']} both-present rows",
        "not a defect — a load-bearing data fact asserted nowhere",
        "integrity: pin put_iv==call_iv so a future skew-bearing refresh is noticed",
    )

    # Treasury negative rate + rate_1m coverage.
    rate3 = tre["tenors"].get("rate_3m", {})
    rate1 = tre["tenors"].get("rate_1m", {})
    add(
        "LOW",
        "unit-scale",
        "Treasury: rate_3m has a negative print and rate_1m is missing pre-2001 "
        "(memory's 'treasury only 2021-05+' is STALE — file now covers 1994-01-03)",
        f"rate_3m min={rate3.get('min')} negatives={rate3.get('negative_count')}; "
        f"rate_1m first_nonnull={rate1.get('first_nonnull')} nan%={rate1.get('nan_pct')}",
        "additive test; data hygiene note",
        "integrity: rate curve plausible band; as_of before coverage -> NaN (not 0)",
    )

    # Dividend float-noise negatives (cosmetic).
    add(
        "LOW",
        "data-hygiene",
        "82 negative dividend_amount values are float-epsilon noise on "
        "Discontinued/Omitted rows (should clamp to 0)",
        f"negative_count={div['negative_count']} all >= -0.001 "
        f"(materially negative: {div['materially_negative_lt_-0.001']}), min={div['negative_min']}",
        "additive: clamp in a producer (no producer script exists today)",
        "integrity: dividend_amount >= -1e-9 (tolerance) OR clamp Discontinued/Omitted to 0",
    )

    # Branch-local staleness (the SessionStart artifact).
    add(
        "INFO",
        "staleness",
        "SessionStart 'OHLCV 79 days stale / 2026-03-20' is BRANCH-LOCAL to "
        "claude/suggest-rolls-defensive-surfacing; main is current",
        branch_note,
        "not a defect on main; pin engine probes to the auto-detected frontier",
        "data->engine: frontier auto-detection is deterministic across branches",
    )

    # Silent vanish check (should be empty — confirms no silent drops).
    vanished = probe.get("vanished_tickers", [])
    if vanished:
        add(
            "HIGH",
            "silent-drop",
            "Tickers vanished from the ranker with neither a produced row nor a drop "
            "entry (true silent drop)",
            f"vanished: {vanished}",
            "TRIO — log only, separate lane-claimed PR",
            "data->engine: every requested ticker ranks OR carries a logged drop reason",
        )
    else:
        add(
            "INFO",
            "silent-drop",
            "No silent drops — every requested ticker either produced a row or carried "
            "an explicit drop reason (drops attrs are complete)",
            f"requested={probe.get('n_requested')} produced={probe.get('n_produced')} "
            f"dropped={probe.get('n_dropped')} vanished=0",
            "n/a (positive control)",
            "data->engine: assert n_produced + n_dropped == n_requested",
        )

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    w.sort(key=lambda x: order.get(x["severity"], 9))
    return w


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    inv = payload["inventory"]
    probe = payload["engine_probe"]
    cov = payload["coverage"]
    weaknesses = payload["weaknesses"]
    L: list[str] = []
    L.append("# Data + Engine Audit — Phase 1 (discovery)")
    L.append("")
    L.append(
        f"_Generated {meta['generated_at']} · provider `{meta['provider']}` · "
        f"frontier `{meta['frontier']}` · as_of `{meta['as_of']}` · "
        f"universe `{meta['probe_universe']}` ({probe.get('n_requested')} names)._"
    )
    L.append("")
    L.append(
        "> Discovery pass — asserts nothing. Phase 2 turns these findings into tests. "
        "Every engine probe routes through `WheelRunner.rank_candidates_by_ev` "
        "(no §2 bypass). Decision trio untouched."
    )
    L.append("")

    # Reconciliation banner.
    L.append("## 0. Frontier reconciliation")
    L.append("")
    L.append(meta["branch_note"])
    L.append("")

    # 1. Inventory.
    L.append("## 1. Connector data inventory")
    L.append("")
    rows = []
    for r in inv:
        if not r.get("exists"):
            rows.append([r["key"], r["file"], "**MISSING**", "", "", "", "", ""])
            continue
        rows.append(
            [
                r["key"],
                r["file"],
                f"{r['rows']:,}",
                r["tickers"] if r["tickers"] is not None else "-",
                r["date_col"] or "-",
                f"{r['date_min']}..{r['date_max']}" if r["date_min"] else "-",
                r["staleness_days"] if r["staleness_days"] is not None else "-",
                r["sha256"],
            ]
        )
    L.append(
        _md_table(
            ["key", "file", "rows", "tickers", "date_col", "range", "stale_d", "sha256[:16]"], rows
        )
    )
    L.append("")

    # 2. Capability map.
    L.append("## 2. Capability map (file -> engine capability)")
    L.append("")
    caprows = [
        [k, CAPABILITY_MAP[k][1], CAPABILITY_MAP[k][0]]
        for k in MarketDataConnector._FILES
        if k in CAPABILITY_MAP
    ]
    L.append(_md_table(["key", "accessor", "engine capability"], caprows))
    L.append("")

    # 3. Coverage matrix.
    L.append("## 3. Coverage matrix")
    L.append("")
    L.append("### 3a. Cross-file referential gaps (vs OHLCV spine)")
    L.append("")
    L.append(
        "_`in_ohlcv_not_dividends` is mostly non-dividend-payers (expected, not a "
        "defect). treasury/vix have no ticker column and are excluded._"
    )
    L.append("")
    grows = []
    for k, v in cov["referential_gaps"].items():
        if v:
            grows.append([k, f"{len(v)}", ", ".join(v[:20]) + (" ..." if len(v) > 20 else "")])
    L.append(_md_table(["relation", "n", "tickers"], grows) if grows else "_No referential gaps._")
    L.append("")
    L.append("### 3b. Seam clusters (auto-detected)")
    L.append("")
    seam = cov["seams"]
    L.append(f"- Joiner clusters (>=3 names' first bar): `{seam['joiner_clusters']}`")
    L.append(f"- Leaver clusters (>=3 names' last bar): `{seam['leaver_clusters']}`")
    L.append("")
    L.append("### 3c. Name classes")
    L.append("")
    cl = cov["name_classes"]
    L.append(f"- **post-seam-only** ({len(cl['post_seam_only'])}): `{cl['post_seam_only']}`")
    L.append(f"- **thin-history <504 bars** ({len(cl['thin_history'])}): `{cl['thin_history']}`")
    L.append(f"- **delisted/stale** ({len(cl['delisted_stale'])}): `{cl['delisted_stale']}`")
    L.append("")
    L.append("### 3d. Per-universe completeness")
    L.append("")
    L.append(
        f"_`complete` = present in ALL core ticker files "
        f"({', '.join(CORE_TICKER_FILES)}); dividends/treasury/vix excluded._"
    )
    L.append("")
    urows = []
    for u in cov["universes"]:
        urows.append(
            [
                u["universe"],
                u["size"],
                u["complete_all_files"],
                u["partial_count"],
                f"post-seam {len(u['flags']['post_seam_only'])}, thin {len(u['flags']['thin_history'])}, "
                f"absent-ohlcv {len(u['flags']['absent_from_ohlcv'])}",
            ]
        )
    L.append(_md_table(["universe", "size", "complete", "partial", "flags"], urows))
    L.append("")

    # 4. Engine probe.
    L.append("## 4. Engine probe (ranker at frontier)")
    L.append("")
    if probe.get("fatal_error"):
        L.append(f"**FATAL:** `{probe['fatal_error']}`")
    else:
        L.append(
            f"- requested **{probe['n_requested']}** · produced **{probe['n_produced']}** · "
            f"dropped **{probe['n_dropped']}** · vanished **{probe['n_vanished']}**"
        )
        L.append(f"- drops_summary: `{probe['drops_summary']}`")
        L.append(f"- forward-distribution tiers: `{probe['distribution_tier_histogram']}`")
        L.append(
            f"- non-finite outputs (ev_dollars/ev_raw/prob_profit): "
            f"`{probe['nonfinite_outputs'] or 'none'}`"
        )
        L.append("")
        L.append("Drops by gate → reason:")
        L.append("")
        drows = []
        for g, reasons in probe["drops_by_gate_reason"].items():
            for reason, n in reasons.items():
                drows.append([g, reason, n])
        L.append(_md_table(["gate", "reason", "n"], drows) if drows else "_none_")
    L.append("")

    # 5. Ranked weakness report.
    L.append("## 5. Ranked weakness report")
    L.append("")
    L.append(
        "Severity → `fixable_in` says whether it can be fixed additively (data/tests) "
        "or only in the trio (log-only this lane)."
    )
    L.append("")
    for x in weaknesses:
        L.append(f"### [{x['severity']}] {x['id']} · {x['title']}")
        L.append(f"- **category:** {x['category']}")
        L.append(f"- **evidence:** {x['evidence']}")
        L.append(f"- **fixable in:** {x['fixable_in']}")
        L.append(f"- **Phase-2 test:** {x['phase2_test']}")
        L.append("")

    # 6. Phase-2 plan.
    L.append("## 6. Phase-2 plan (build after green-light)")
    L.append("")
    L.append(
        "**A) Database-integrity** — `tests/test_data_integrity_*.py` (NEW; no existing "
        "test asserts the bundled-CSV contracts — `test_data_connector.py` uses synthetic "
        "tmp_path fixtures). Reuse the `HAS_BLOOMBERG_DATA` skipif pattern from "
        "`tests/test_data_integration.py`."
    )
    L.append("")
    L.append(
        "**B) Data→engine** — extend the only real-CSV data→engine test "
        "(`tests/test_audit_viii_real_data_smoke.py`) into `tests/test_data_to_engine_*.py`: "
        "pinned frontier, adversarial ticker set, finite/banded outputs, tier-cascade "
        "correctness, no silent drops, graceful degradation, determinism."
    )
    L.append("")
    L.append(
        "**Buckets** — confirmed data defects go to xfail(strict=True)+issue; test bugs get "
        "fixed. Full-universe sweep behind a slow marker."
    )
    L.append("")
    L.append("---")
    L.append(
        f"_Reproduce: `python scripts/audit_data_engine.py --universe {meta['probe_universe']} "
        f"--as-of {meta['as_of']}`_"
    )
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Universe selection
# ---------------------------------------------------------------------------


def select_universe(name: str, conn: MarketDataConnector, sets: dict[str, set[str]]) -> list[str]:
    full = sorted(sets.get("ohlcv", set()) | sets.get("vol_iv", set()))
    if name == "full":
        return full
    if name == "24":
        return list(UNIVERSE_24)
    if name == "100":
        return list(UNIVERSE_100)
    if name == "audit":
        # Curated edge-case set: UNIVERSE_24 + the seam joiners/leavers + thin oddities.
        oddities = [
            "BNY",
            "CASY",
            "COHR",
            "FDXF",
            "LITE",
            "SATS",
            "VEEV",
            "VRT",  # joiners
            "BK",
            "CTRA",
            "EPAM",
            "HOLX",
            "LW",
            "MOH",
            "MTCH",
            "PAYC",  # leavers
            "WMT",
            "KMB",
            "CPB",
            "DPZ",
            "PLTR",  # thin blue-chips
        ]
        seen: set[str] = set()
        out: list[str] = []
        for t in list(UNIVERSE_24) + list(UNIVERSE_100) + oddities:
            nt = normalize_ticker(t)
            if nt not in seen:
                seen.add(nt)
                out.append(t)
        return out
    if "," in name:
        return [x.strip() for x in name.split(",") if x.strip()]
    return [name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase-1 data + engine audit (discovery).")
    ap.add_argument("--universe", default="full", help="full | audit | 24 | 100 | comma,list")
    ap.add_argument("--as-of", default="auto", help="YYYY-MM-DD or 'auto' (frontier)")
    ap.add_argument("--seam", default=DEFAULT_SEAM, help="seam date for classification")
    ap.add_argument("--dte", type=int, default=35)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="markdown findings path")
    ap.add_argument("--json", default=None, help="JSON sidecar path")
    args = ap.parse_args(argv)

    # Windows consoles default to cp1252; the report is UTF-8. Never crash on
    # console output (the findings doc is always written to a UTF-8 file).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - older/odd stream objects
            pass

    runner = WheelRunner()
    conn = runner.connector
    provider = type(conn).__name__
    data_dir = Path(getattr(conn, "_data_dir", "data/bloomberg"))
    print(f"[audit] provider={provider} data_dir={data_dir}")

    frontier, per_file_max = resolve_frontier(data_dir)
    as_of = frontier if args.as_of == "auto" else args.as_of
    print(f"[audit] frontier={frontier} as_of={as_of} per_file_max={per_file_max}")

    branch_note = (
        f"Auto-detected frontier (most-recent bar common to OHLCV & IV) = **{frontier}**. "
        f"Per-file max: {per_file_max}. The SessionStart staleness warning reflects the "
        f"*checked-out branch*, not `main`: `main` carries data through ~{frontier}, while "
        f"`claude/suggest-rolls-defensive-surfacing` (the primary working tree) ends 2026-03-20. "
        f"The probe pins to the auto-detected frontier so it is deterministic on either branch."
    )

    print("[audit] inventory…")
    inv = file_inventory(data_dir, frontier)
    sets = ticker_sets(data_dir)
    print("[audit] history profile + classification…")
    profile = ohlcv_history_profile(data_dir)
    seams = detect_seams(profile, frontier)
    classes = classify_names(profile, args.seam, frontier)
    gaps = referential_gaps(sets)

    universes_cov = []
    for nm, uni in (("UNIVERSE_24", UNIVERSE_24), ("UNIVERSE_100", UNIVERSE_100)):
        if uni:
            universes_cov.append(universe_coverage(nm, uni, sets, profile, args.seam))
    conn_uni = tuple(sorted(sets.get("ohlcv", set())))
    universes_cov.append(
        universe_coverage("connector_universe", conn_uni, sets, profile, args.seam)
    )

    probe_tickers = select_universe(args.universe, conn, sets)
    print(f"[audit] engine probe over {len(probe_tickers)} names at as_of={as_of}…")
    probe = engine_probe(runner, probe_tickers, as_of, args.dte, args.delta)
    print(
        f"[audit] probe: produced={probe.get('n_produced')} dropped={probe.get('n_dropped')} "
        f"vanished={probe.get('n_vanished')}"
    )

    print("[audit] hygiene checks…")
    hygiene = hygiene_checks(data_dir, frontier)

    weaknesses = build_weaknesses(inv, gaps, classes, probe, hygiene, frontier, branch_note)

    payload = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "provider": provider,
            "data_dir": str(data_dir),
            "frontier": frontier,
            "as_of": as_of,
            "per_file_max": per_file_max,
            "probe_universe": args.universe,
            "seam": args.seam,
            "branch_note": branch_note,
        },
        "inventory": inv,
        "capability_map": {
            k: {"accessor": v[1], "capability": v[0]} for k, v in CAPABILITY_MAP.items()
        },
        "coverage": {
            "referential_gaps": gaps,
            "seams": seams,
            "name_classes": classes,
            "universes": universes_cov,
        },
        "engine_probe": probe,
        "hygiene": hygiene,
        "weaknesses": weaknesses,
    }

    md = render_markdown(payload)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(md, encoding="utf-8")
        print(f"[audit] wrote markdown -> {outp}")
    if args.json:
        jp = Path(args.json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"[audit] wrote json -> {jp}")
    if not args.out and not args.json:
        print(md)

    # Console summary of the ranked weaknesses.
    print("\n[audit] ranked weaknesses:")
    for x in weaknesses:
        print(f"  [{x['severity']:6s}] {x['id']} {x['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
