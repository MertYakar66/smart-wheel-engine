"""Parameter freeze-replay validation (V2 / C1) — amnesia test, freeze
snapshot lock, and frozen-knowledge replay.

Why this exists
---------------
`docs/PRODUCTION_READINESS.md` C1 asks for HMM / POT-GPD parameter freeze +
replay infrastructure; `docs/VALIDATION_PHASE_PLAN.md` section 5 is the
pre-registered design this module implements.  The record asserts the
engine's online fits are "PIT-clean / leakage-free *by construction*"
(`docs/PARAMETER_OOS.md` section 1) and the surviving top-tier rank edge
rests on that claim — this module converts the assertion into evidence
(or a finding), in three parts:

* **V2-a — the amnesia test.**  Physically truncate every unambiguous
  market time-series CSV to ``date <= T``, run the production ranker at
  ``as_of = T`` against the full and the truncated data directory, and
  demand byte-identical output.  An A/A determinism control runs first —
  without it, A/B diffs are uninterpretable.  This covers the leak
  vectors per-function unit tests cannot see (swallowed-exception PIT
  slices, components fed pre-sliced frames instead of ``as_of``,
  connector caches serving unsliced data).
* **V2-b — the C1 freeze snapshot + reproducibility lock.**  Snapshot the
  fitted HMM (start_prob / trans_mat / means / stds / labels / posterior /
  multiplier) and a canonical return-space POT-GPD fit per ticker at a
  cutoff date into a committed fixture; a test refits from the committed
  CSVs and asserts reproduction.  Fires on silent data restatement,
  numpy/scipy behavior drift, or a refactor that changes the classifier.
* **V2-c — the frozen-knowledge replay.**  Replay a leakage-certified
  holdout grid with all *fitted* knowledge frozen at the cutoff: the
  scenario distribution via a harness-scoped patch of
  ``best_available_forward_distribution`` (the parameter_oos section-6
  sanctioned pattern), and the HMM overlay offline by recombination.
  Market-state readings (spot, premium, IV, F4 vol-ratio) stay live at
  ``T`` — freeze-replay freezes fitted knowledge, not the market.

Scope / invariants (CLAUDE.md section 2)
----------------------------------------
Measurement-only.  The ranker is called READ-ONLY with the option-premium
rail pinned off; the ``forward_distribution_frozen`` patch is scoped to a
context manager inside this harness and never touches a production
default; nothing here feeds an EV, a verdict, or a gate; the
decision-layer trio is untouched.

Documented tier-1 exclusions (pre-registered, section 5.1):
``sp500_credit_risk.csv`` is a dateless snapshot (inherent PIT limitation
of that dataset — nothing to truncate); the credit-regime de-rank reads
FRED network series, so the amnesia harness disables it on BOTH sides
(``use_credit_regime=False``) and records the exclusion; schedule-type
files (earnings calendar, dividends, corporate actions, fundamentals,
broad_pull) are copied intact — identical in both runs, they cannot
create false diffs.

Tier-2 triage (pre-registered, section 11): ``amnesia_report(tier=2)``
additionally truncates the dated tier-2 files
(``sp500_corporate_actions.csv`` on ``announcement_date``,
``sp500_dividends.csv`` on ``declared_date``) — a black-box confirmation
that no cache/fallback serves post-cutoff tier-2 rows into the rank
(corporate_actions is internally gated ``<= as_of``; dividends is off the
short-put path). The dateless snapshots (fundamentals, credit_risk),
FRED, and the forward-by-design earnings calendar remain un-truncatable
and are recorded as standing PIT limitations, not fixed.
"""

from __future__ import annotations

import logging
import re
import shutil
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror the production recipes exactly (verified against source)
# ---------------------------------------------------------------------------

#: V2 canonical freeze cutoff — the parameter_oos canonical split date.
FREEZE_CUTOFF_DEFAULT = "2023-06-30"

#: Tier-1 truncation set: unambiguous market time series under ``data_dir``
#: with a literal ``date`` column.  Everything else is copied intact.
TIER1_TRUNCATE_FILES: dict[str, str] = {
    "sp500_ohlcv.csv": "date",
    "sp500_vol_iv_full.csv": "date",
    "treasury_yields.csv": "date",
    "vix_term_structure.csv": "date",
    "sp500_liquidity.csv": "date",
}

#: Tier-2 truncation set (plan §11): dated tier-2 CSVs cut on their
#: PIT-*announcement* column, not the effective/ex column — keeping
#: already-announced future-effective rows (legitimately knowable at T)
#: and dropping only post-T announcements. ``corporate_actions`` is the
#: one dated tier-2 source ON the short-put rank path (internally gated
#: ``announcement_date <= as_of``; the truncation is a black-box
#: confirmation). ``dividends`` is OFF the put path (feeds only
#: analyze_ticker + the covered-call ranker) and is truncated for
#: completeness — an expected no-op. Both columns are 100% ISO-dated, so
#: no rows drop on missing dates (no false diffs).
TIER2_TRUNCATE_FILES: dict[str, str] = {
    "sp500_corporate_actions.csv": "announcement_date",
    "sp500_dividends.csv": "declared_date",
}


def _truncate_set(tier: int) -> dict[str, str]:
    """Files to date-truncate at a given tier. ``tier=1`` is the V2-a set
    (byte-identical); ``tier>=2`` adds the tier-2 dated files."""
    if tier <= 1:
        return dict(TIER1_TRUNCATE_FILES)
    return {**TIER1_TRUNCATE_FILES, **TIER2_TRUNCATE_FILES}


#: HMM recipe — must match ``wheel_runner.rank_candidates_by_ev`` (~L1977-2003):
#: log-returns of the PIT-sliced close, >= 200 required, last 504 used,
#: ``GaussianHMM(n_states=4, n_iter=20, random_state=42)``.
HMM_N_STATES = 4
HMM_N_ITER = 20
HMM_SEED = 42
HMM_TAIL_LEN = 504
HMM_MIN_RETURNS = 200

#: The ev_engine regime-multiplier clamp (ev_engine.py ~L552).
REGIME_MULT_CLAMP = (0.0, 1.25)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# V2-a — amnesia test: truncation, rank passes, exact comparison
# ---------------------------------------------------------------------------


def truncate_csv(src: Path, dst: Path, *, cutoff: str, date_col: str) -> dict[str, int]:
    """Write a copy of ``src`` keeping only rows with ``date_col <= cutoff``.

    Cells are read and written as raw strings (``dtype=str``) so surviving
    rows are byte-preserved — re-parsing floats and re-formatting them could
    otherwise perturb the very comparison this harness exists to make.
    ISO dates compare lexicographically, so no datetime parsing is needed;
    rows whose date cell is not ISO-shaped are dropped and counted.
    """
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    if date_col not in df.columns:
        raise ValueError(f"{src.name}: expected a {date_col!r} column, got {list(df.columns)}")
    dates = df[date_col].astype(str)
    iso = dates.str.match(_ISO_DATE_RE)
    keep = iso & (dates.str[:10] <= cutoff[:10])
    out = df[keep]
    out.to_csv(dst, index=False)
    return {
        "rows_total": int(len(df)),
        "rows_kept": int(keep.sum()),
        "rows_dropped_noniso": int((~iso).sum()),
    }


def build_truncated_data_dir(
    src_dir: Path | str, dst_dir: Path | str, *, cutoff: str, tier: int = 1
) -> dict:
    """Materialize a truncated copy of the data directory at ``cutoff``.

    Files in the tier's truncate set are date-truncated (rewritten on every
    call — the cutoff changes per amnesia date); everything else (files and
    subdirectories, e.g. ``broad_pull/``) is copied intact once and reused
    across calls. ``tier=1`` truncates the V2-a tier-1 set only (§5.1);
    ``tier=2`` also truncates the dated tier-2 files (§11). Because the
    tier-2 files change with the cutoff too, a truncated dir built at one
    tier must not be silently reused at another — callers key the work dir
    by tier (``amnesia_report``).
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    truncate_files = _truncate_set(tier)
    manifest: dict[str, Any] = {
        "cutoff": cutoff,
        "tier": tier,
        "truncated": {},
        "copied_intact": [],
    }
    for entry in sorted(src_dir.iterdir()):
        target = dst_dir / entry.name
        if entry.is_dir():
            if not target.exists():
                shutil.copytree(entry, target)
            manifest["copied_intact"].append(entry.name + "/")
        elif entry.name in truncate_files:
            manifest["truncated"][entry.name] = truncate_csv(
                entry, target, cutoff=cutoff, date_col=truncate_files[entry.name]
            )
        else:
            if not target.exists():
                shutil.copy2(entry, target)
            manifest["copied_intact"].append(entry.name)
    return manifest


def rank_pass(
    data_dir: Path | str,
    tickers: Sequence[str],
    as_of: str,
    *,
    dte_target: int = 35,
    delta_target: float = 0.25,
    top_n: int = 100,
) -> tuple[pd.DataFrame, list[dict]]:
    """One production rank at ``as_of`` against ``data_dir``, fresh runner.

    Rail pinned off (regression-lock convention); ``use_credit_regime=False``
    on every amnesia pass — the credit de-rank reads FRED *network* series,
    which file truncation cannot cover and whose availability could differ
    between two runs (documented tier-1 exclusion; its PIT slice is
    code-enforced and unit-tested in ``fred_adapter``).
    """
    from backtests.regression._common import _option_premium_rail_pinned_off
    from engine.wheel_runner import WheelRunner

    with _option_premium_rail_pinned_off():
        runner = WheelRunner(data_dir=str(data_dir))
        _ = runner.connector  # resolve inside the pinned-rail scope
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = runner.rank_candidates_by_ev(
            tickers=list(tickers),
            dte_target=dte_target,
            delta_target=delta_target,
            contracts=1,
            top_n=top_n,
            min_ev_dollars=-1e9,
            as_of=as_of,
            include_diagnostic_fields=True,
            use_credit_regime=False,
        )
    if frame is None:
        frame = pd.DataFrame()
    drops = [dict(d) for d in frame.attrs.get("drops", [])]
    return frame, drops


def _normalize_drops(drops: list[dict]) -> list[tuple]:
    return sorted(
        (
            str(d.get("ticker", d.get("new_dte", ""))),
            str(d.get("gate", "")),
            str(d.get("reason", "")),
        )
        for d in drops
    )


def compare_rank_outputs(
    frame_a: pd.DataFrame,
    drops_a: list[dict],
    frame_b: pd.DataFrame,
    drops_b: list[dict],
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Exact comparison of two ranker outputs (rows + drops).

    Numeric cells must be bit-equal (NaN == NaN allowed); everything else
    compares as strings.  ``attrs['staleness']`` is deliberately NOT
    compared — it embeds the wall-clock data frontier, which differs by
    construction between a full and a truncated directory and carries no
    row-level information.
    """
    out: dict[str, Any] = {
        "identical": True,
        "n_rows_a": int(len(frame_a)),
        "n_rows_b": int(len(frame_b)),
    }
    diffs: list[dict] = []

    da, db = _normalize_drops(drops_a), _normalize_drops(drops_b)
    if da != db:
        out["identical"] = False
        only_a = [d for d in da if d not in db]
        only_b = [d for d in db if d not in da]
        out["drops_only_a"] = only_a[:max_examples]
        out["drops_only_b"] = only_b[:max_examples]

    a = frame_a.copy()
    b = frame_b.copy()
    if len(a) != len(b) or (len(a) and set(a["ticker"]) != set(b["ticker"])):
        out["identical"] = False
        ta = set(a["ticker"]) if len(a) else set()
        tb = set(b["ticker"]) if len(b) else set()
        out["tickers_only_a"] = sorted(ta - tb)
        out["tickers_only_b"] = sorted(tb - ta)
        out["diff_examples"] = diffs
        return out
    if len(a) == 0:
        out["diff_examples"] = diffs
        return out

    a = a.sort_values("ticker").reset_index(drop=True)
    b = b.sort_values("ticker").reset_index(drop=True)
    cols_a, cols_b = set(a.columns), set(b.columns)
    if cols_a != cols_b:
        out["identical"] = False
        out["columns_only_a"] = sorted(cols_a - cols_b)
        out["columns_only_b"] = sorted(cols_b - cols_a)
    diff_columns: dict[str, int] = {}
    for col in sorted(cols_a & cols_b):
        va, vb = a[col], b[col]
        try:
            fa = va.to_numpy(dtype=float)
            fb = vb.to_numpy(dtype=float)
            neq = ~((fa == fb) | (np.isnan(fa) & np.isnan(fb)))
        except (TypeError, ValueError):
            neq = va.astype(str).to_numpy() != vb.astype(str).to_numpy()
        n_neq = int(neq.sum())
        if n_neq:
            out["identical"] = False
            diff_columns[col] = n_neq
            for i in np.flatnonzero(neq)[: max(1, max_examples // 4)]:
                diffs.append(
                    {
                        "ticker": str(a["ticker"].iloc[int(i)]),
                        "column": col,
                        "a": str(va.iloc[int(i)]),
                        "b": str(vb.iloc[int(i)]),
                    }
                )
    if diff_columns:
        out["diff_columns"] = diff_columns
    out["diff_examples"] = diffs[:max_examples]
    return out


#: Exclusions still standing after each tier's truncation (reported in
#: the amnesia payload so the proof's boundary is explicit).
_TIER_EXCLUSIONS: dict[int, dict[str, str]] = {
    1: {
        "sp500_credit_risk.csv": "dateless snapshot — inherent PIT limitation, nothing to truncate",
        "fred_network_series": "credit-regime de-rank disabled on both sides (use_credit_regime=False)",
        "schedule_type_files": "earnings/dividends/fundamentals/corporate_actions/broad_pull copied intact",
    },
    2: {
        "sp500_corporate_actions.csv": "NOW TRUNCATED @announcement_date (was tier-1 exclusion)",
        "sp500_dividends.csv": "NOW TRUNCATED @declared_date (off the put path; confirmatory)",
        "sp500_credit_risk.csv": "dateless + off the put rank path — inherent PIT limitation",
        "sp500_fundamentals.csv": "dateless snapshot — served whole; residual IV-fallback surface (T2-2)",
        "fred_network_series": "credit-regime de-rank disabled on both sides (use_credit_regime=False)",
        "earnings": "forward-by-design (event-lockout needs the announced future date); DATE-only, no outcome field",
        "split_adjust_parquet_rail": "_split_adjust_option_premium reads corp_actions un-as_of'd; dormant without option_premium parquets (T2-2)",
    },
}


def amnesia_report(
    *,
    data_dir: Path | str,
    tickers: Sequence[str],
    dates: Sequence[str],
    work_dir: Path | str,
    dte_target: int = 35,
    delta_target: float = 0.25,
    top_n: int = 100,
    tier: int = 1,
    keep_truncated: bool = False,
) -> dict[str, Any]:
    """Run the full amnesia protocol: per date, A/A control then A/B.

    ``tier=1`` truncates the V2-a tier-1 set (§5.1); ``tier=2`` also
    truncates the dated tier-2 files (§11). Verdicts per date: ``PASS``
    (A/A and A/B identical), ``NONDETERMINISTIC`` (A/A differs — the A/B
    comparison is uninterpretable and is still reported), ``FAIL`` (A/A
    clean, A/B differs — post-cutoff data leakage).
    """
    src = Path(data_dir)
    # Key the work dir by tier so tier-1 and tier-2 truncated copies never
    # collide (a tier-1 run copies the tier-2 files intact; reusing that dir
    # for tier-2 — or vice versa — could serve a stale copy).
    trunc = Path(work_dir) / f"truncated_data_tier{tier}"
    per_date: list[dict] = []
    for as_of in dates:
        fa1, da1 = rank_pass(
            src, tickers, as_of, dte_target=dte_target, delta_target=delta_target, top_n=top_n
        )
        fa2, da2 = rank_pass(
            src, tickers, as_of, dte_target=dte_target, delta_target=delta_target, top_n=top_n
        )
        aa = compare_rank_outputs(fa1, da1, fa2, da2)
        manifest = build_truncated_data_dir(src, trunc, cutoff=as_of, tier=tier)
        fb, db = rank_pass(
            trunc, tickers, as_of, dte_target=dte_target, delta_target=delta_target, top_n=top_n
        )
        ab = compare_rank_outputs(fa1, da1, fb, db)
        if not aa["identical"]:
            verdict = "NONDETERMINISTIC"
        elif not ab["identical"]:
            verdict = "FAIL"
        else:
            verdict = "PASS"
        per_date.append(
            {
                "as_of": as_of,
                "n_ranked_rows": int(len(fa1)),
                "n_drops": len(da1),
                "aa_control": aa,
                "ab_truncated": ab,
                "truncation_rows_kept": {
                    k: v["rows_kept"] for k, v in manifest["truncated"].items()
                },
                "verdict": verdict,
            }
        )
        logger.info("amnesia t%d %s: %s (%d rows)", tier, as_of, verdict, len(fa1))
    if not keep_truncated:
        shutil.rmtree(trunc, ignore_errors=True)
    verdicts = [d["verdict"] for d in per_date]
    return {
        "tier": tier,
        "tickers": list(tickers),
        "dates": list(dates),
        "truncated_files": sorted(_truncate_set(tier)),
        "still_excluded": _TIER_EXCLUSIONS.get(tier, {}),
        "per_date": per_date,
        "overall_verdict": "PASS" if all(v == "PASS" for v in verdicts) else "FAIL",
    }


# ---------------------------------------------------------------------------
# V2-b — C1 freeze snapshot + reproducibility lock
# ---------------------------------------------------------------------------


def _pit_ohlcv(conn: Any, ticker: str, as_of: str) -> pd.DataFrame | None:
    try:
        ohlcv = conn.get_ohlcv(ticker)
    except Exception:  # noqa: BLE001 — a bad ticker degrades, never aborts
        return None
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return None
    ohlcv = ohlcv.loc[ohlcv.index <= pd.Timestamp(as_of)]
    return None if ohlcv.empty else ohlcv


def _hmm_tail(ohlcv: pd.DataFrame) -> np.ndarray | None:
    """The exact observation window wheel_runner feeds the HMM."""
    log_rets = np.diff(np.log(ohlcv["close"].values))
    if len(log_rets) < HMM_MIN_RETURNS:
        return None
    return log_rets[-HMM_TAIL_LEN:]


def hmm_snapshot_at(conn: Any, ticker: str, as_of: str) -> dict[str, Any] | None:
    """Fit the production-recipe HMM at ``as_of``; return a serializable snapshot."""
    from engine.regime_hmm import GaussianHMM

    ohlcv = _pit_ohlcv(conn, ticker, as_of)
    if ohlcv is None:
        return None
    tail = _hmm_tail(ohlcv)
    if tail is None:
        return None
    try:
        hmm = GaussianHMM(n_states=HMM_N_STATES, n_iter=HMM_N_ITER, random_state=HMM_SEED)
        hmm.fit(tail)
        probs = hmm.predict_proba(tail)
    except Exception as exc:  # noqa: BLE001 — e.g. the #386 non-finite guard on halt-day NaN closes
        logger.warning("snapshot HMM fit failed for %s @ %s: %s", ticker, as_of, exc)
        return None
    fr = hmm.fit_result
    return {
        "n_obs": int(len(tail)),
        "start_prob": np.asarray(fr.start_prob, dtype=float).tolist(),
        "trans_mat": np.asarray(fr.trans_mat, dtype=float).tolist(),
        "means": np.asarray(fr.means, dtype=float).tolist(),
        "stds": np.asarray(fr.stds, dtype=float).tolist(),
        "state_labels": list(fr.state_labels),
        "converged": bool(fr.converged),
        "posterior_at_cutoff": np.asarray(probs[-1], dtype=float).tolist(),
        "multiplier": float(hmm.position_multiplier(probs[-1])),
        "regime": str(fr.state_labels[int(np.argmax(probs[-1]))]) if fr.state_labels else "unknown",
    }


def gpd_snapshot_at(
    conn: Any, ticker: str, as_of: str, *, horizon_days: int = 35
) -> dict[str, Any] | None:
    """Canonical return-space POT-GPD fit at ``as_of``.

    The engine's in-situ GPD fits on trade-space scenario *P&L* (strike- and
    premium-dependent); this per-ticker lock fits on the return-space losses
    of the same frozen scenario array, so it is trade-independent while the
    trade-space fit inherits its determinism from the frozen array.  Stated
    here so nobody mistakes the fixture for the engine's in-situ numbers.
    """
    from engine.forward_distribution import best_available_forward_distribution
    from engine.tail_risk import fit_gpd_tail

    ohlcv = _pit_ohlcv(conn, ticker, as_of)
    if ohlcv is None:
        return None
    try:
        fwd_rets, method = best_available_forward_distribution(
            ohlcv, horizon_days=horizon_days, as_of=as_of
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never abort a snapshot build
        logger.warning("snapshot forward distribution failed for %s @ %s: %s", ticker, as_of, exc)
        return None
    arr = np.asarray(fwd_rets, dtype=float)
    if arr.size == 0:
        return None
    fit = fit_gpd_tail(-arr)
    return {
        "method": str(method),
        "n_scenarios": int(arr.size),
        "scenario_first": float(arr[0]),
        "scenario_last": float(arr[-1]),
        "threshold": float(fit.threshold),
        "n_exceedances": int(fit.n_exceedances),
        "shape_xi": float(fit.shape_xi),
        "scale_beta": float(fit.scale_beta),
        "tail_fraction": float(fit.tail_fraction),
        "converged": bool(fit.converged),
        "log_likelihood": float(fit.log_likelihood),
    }


def build_freeze_snapshot(
    data_dir: Path | str,
    tickers: Sequence[str],
    *,
    cutoff: str = FREEZE_CUTOFF_DEFAULT,
    horizon_days: int = 35,
) -> dict[str, Any]:
    """Build the C1 freeze snapshot for every ticker at ``cutoff``."""
    from engine.data_connector import MarketDataConnector

    conn = MarketDataConnector(str(data_dir))
    snap: dict[str, Any] = {
        "cutoff": cutoff,
        "horizon_days": horizon_days,
        "hmm_recipe": {
            "n_states": HMM_N_STATES,
            "n_iter": HMM_N_ITER,
            "random_state": HMM_SEED,
            "tail_len": HMM_TAIL_LEN,
            "min_returns": HMM_MIN_RETURNS,
        },
        "tickers": {},
    }
    for ticker in tickers:
        entry: dict[str, Any] = {}
        hmm = hmm_snapshot_at(conn, ticker, cutoff)
        gpd = gpd_snapshot_at(conn, ticker, cutoff, horizon_days=horizon_days)
        if hmm is not None:
            entry["hmm"] = hmm
        if gpd is not None:
            entry["gpd"] = gpd
        snap["tickers"][ticker] = entry
        logger.info("freeze snapshot %s: hmm=%s gpd=%s", ticker, hmm is not None, gpd is not None)
    return snap


def diff_snapshots(a: dict, b: dict, *, rtol: float = 1e-7, atol: float = 1e-12) -> dict[str, Any]:
    """Pure structural comparison of two freeze snapshots.

    Numeric leaves compare with ``np.allclose`` (NaN == NaN allowed); string
    and bool leaves compare exactly.  Returns per-ticker mismatched field
    paths — empty means reproduction holds.
    """

    def _leaf_diffs(path: str, x: Any, y: Any, acc: list[str]) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    acc.append(f"{path}.{k} (missing on one side)")
                else:
                    _leaf_diffs(f"{path}.{k}", x[k], y[k], acc)
            return
        if isinstance(x, list) and isinstance(y, list):
            try:
                xa = np.asarray(x, dtype=float)
                ya = np.asarray(y, dtype=float)
                if xa.shape != ya.shape or not np.allclose(
                    xa, ya, rtol=rtol, atol=atol, equal_nan=True
                ):
                    acc.append(path)
                return
            except (TypeError, ValueError):
                if x != y:
                    acc.append(path)
                return
        if isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str):
            if x != y:
                acc.append(path)
            return
        try:
            if not np.allclose(float(x), float(y), rtol=rtol, atol=atol, equal_nan=True):
                acc.append(path)
        except (TypeError, ValueError):
            if x != y:
                acc.append(path)

    tick_a = a.get("tickers", {})
    tick_b = b.get("tickers", {})
    per_ticker: dict[str, list[str]] = {}
    for t in sorted(set(tick_a) | set(tick_b)):
        acc: list[str] = []
        _leaf_diffs(t, tick_a.get(t, {}), tick_b.get(t, {}), acc)
        if acc:
            per_ticker[t] = acc
    return {
        "identical": not per_ticker,
        "n_tickers": len(set(tick_a) | set(tick_b)),
        "mismatched": per_ticker,
        "rtol": rtol,
    }


def verify_freeze_snapshot(
    snapshot: dict,
    data_dir: Path | str,
    *,
    rtol: float = 1e-7,
    tickers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Refit from the (committed) CSVs and diff against ``snapshot``."""
    names = list(tickers) if tickers is not None else list(snapshot.get("tickers", {}))
    rebuilt = build_freeze_snapshot(
        data_dir,
        names,
        cutoff=snapshot["cutoff"],
        horizon_days=int(snapshot.get("horizon_days", 35)),
    )
    if tickers is not None:
        pruned = {k: v for k, v in snapshot.get("tickers", {}).items() if k in set(names)}
        snapshot = {**snapshot, "tickers": pruned}
    return diff_snapshots(snapshot, rebuilt, rtol=rtol)


# ---------------------------------------------------------------------------
# V2-c — frozen-knowledge replay
# ---------------------------------------------------------------------------


@contextmanager
def forward_distribution_frozen(cutoff: str):
    """Scope-limited freeze of the scenario distribution at ``cutoff``.

    Patches ``engine.forward_distribution.best_available_forward_distribution``
    so every call resolves with ``as_of = cutoff`` regardless of the caller's
    ``as_of`` — ``wheel_runner`` imports it function-locally, so the module
    attribute is re-resolved on every rank call and the patch takes effect
    without touching the trio.  The F4 widening calls are deliberately NOT
    patched (market-state reading, live at T — plan doc section 5.3).
    """
    import engine.forward_distribution as fdist

    orig = fdist.best_available_forward_distribution

    def _frozen(ohlcv: pd.DataFrame, horizon_days: int = 30, as_of: Any = None, **kw: Any):
        return orig(ohlcv, horizon_days=horizon_days, as_of=cutoff, **kw)

    fdist.best_available_forward_distribution = _frozen
    try:
        yield
    finally:
        fdist.best_available_forward_distribution = orig


def build_frozen_tail_table(
    *,
    tickers: Sequence[str],
    sample_dates: Sequence[date],
    cutoff: str = FREEZE_CUTOFF_DEFAULT,
    dte_target: int = 35,
    delta_target: float = 0.25,
    top_n: int = 100,
    progress: bool = True,
) -> pd.DataFrame:
    """V1's ``build_tail_table`` under the scenario freeze — same schema,
    so ``tail_exceedance.full_report`` runs unchanged on the result."""
    from backtests.tail_exceedance import build_tail_table

    bad = [d for d in sample_dates if d.isoformat() <= cutoff]
    if bad:
        raise ValueError(
            f"{len(bad)} sample dates at/before the freeze cutoff {cutoff} (first: {bad[0]})"
        )
    with forward_distribution_frozen(cutoff):
        return build_tail_table(
            tickers=tickers,
            sample_dates=sample_dates,
            dte_target=dte_target,
            delta_target=delta_target,
            top_n=top_n,
            progress=progress,
        )


def frozen_hmm_multipliers(
    data_dir: Path | str,
    tickers: Sequence[str],
    dates: Sequence[str],
    *,
    cutoff: str = FREEZE_CUTOFF_DEFAULT,
) -> pd.DataFrame:
    """Offline frozen-HMM overlay: parameters fit once at ``cutoff`` per
    ticker, posterior evaluated on each date's own PIT tail.

    Mirrors the wheel_runner recipe exactly (tail construction, seed,
    n_iter); where the fit or a date's tail is unavailable the multiplier
    degrades to 1.0 / "unknown" — the same neutral no-op the engine uses.
    """
    from engine.data_connector import MarketDataConnector
    from engine.regime_hmm import GaussianHMM

    conn = MarketDataConnector(str(data_dir))
    rows: list[dict] = []
    for ticker in tickers:
        # One OHLCV fetch per ticker (35k per-date fetches at 100t would
        # dominate the runtime); per-date slicing is identical to
        # _pit_ohlcv's cut.
        try:
            ohlcv_full = conn.get_ohlcv(ticker)
        except Exception:  # noqa: BLE001 — a bad ticker degrades, never aborts
            ohlcv_full = None
        if ohlcv_full is not None and (ohlcv_full.empty or "close" not in ohlcv_full.columns):
            ohlcv_full = None
        ohlcv_cut = (
            ohlcv_full.loc[ohlcv_full.index <= pd.Timestamp(cutoff)]
            if ohlcv_full is not None
            else None
        )
        if ohlcv_cut is not None and ohlcv_cut.empty:
            ohlcv_cut = None
        tail0 = _hmm_tail(ohlcv_cut) if ohlcv_cut is not None else None
        hmm = None
        if tail0 is not None:
            # Guarded like wheel_runner's own fit site: the engine's #386
            # non-finite guard (regime_hmm) deliberately RAISES on NaN
            # observations (e.g. BIIB's 2023-06-09 halt-day close, inside the
            # 504-return tail at the canonical cutoff) and callers degrade to
            # the neutral multiplier. Found by the V2-c-100t run — the fit was
            # the one unguarded call in this function.
            try:
                hmm = GaussianHMM(n_states=HMM_N_STATES, n_iter=HMM_N_ITER, random_state=HMM_SEED)
                hmm.fit(tail0)
            except Exception as exc:  # noqa: BLE001 — degrade like the engine does
                logger.warning("frozen HMM fit failed for %s @ %s: %s", ticker, cutoff, exc)
                hmm = None
        for d in dates:
            mult, regime = 1.0, "unknown"
            if hmm is not None:
                ohlcv_d = ohlcv_full.loc[ohlcv_full.index <= pd.Timestamp(d)]
                tail_d = _hmm_tail(ohlcv_d) if not ohlcv_d.empty else None
                if tail_d is not None:
                    try:
                        probs = hmm.predict_proba(tail_d)
                        mult = float(hmm.position_multiplier(probs[-1]))
                        labels = hmm.fit_result.state_labels
                        regime = str(labels[int(np.argmax(probs[-1]))]) if labels else "unknown"
                    except Exception:  # noqa: BLE001 — degrade like the engine does
                        mult, regime = 1.0, "unknown"
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "frozen_hmm_multiplier": mult,
                    "frozen_hmm_regime": regime,
                }
            )
    return pd.DataFrame(
        rows, columns=["date", "ticker", "frozen_hmm_multiplier", "frozen_hmm_regime"]
    )


def apply_frozen_hmm(frozen_table: pd.DataFrame, mults: pd.DataFrame) -> pd.Series:
    """Recombine offline: ``frozen_ev_raw x clamp(frozen_hmm_multiplier)``.

    The ev_engine clamp [0.0, 1.25] is applied for fidelity; rows without a
    multiplier fall back to the engine's neutral 1.0.
    """
    lo, hi = REGIME_MULT_CLAMP
    merged = frozen_table[["date", "ticker", "ev_raw"]].merge(
        mults, on=["date", "ticker"], how="left"
    )
    m = merged["frozen_hmm_multiplier"].fillna(1.0).clip(lo, hi)
    out = merged["ev_raw"].to_numpy(dtype=float) * m.to_numpy(dtype=float)
    return pd.Series(out, index=frozen_table.index, name="ev_frozen_hmm")


def _months_since(dates: pd.Series, cutoff: str) -> np.ndarray:
    delta = pd.to_datetime(dates) - pd.Timestamp(cutoff)
    return (delta.dt.days / 30.44).to_numpy(dtype=float)


_MONTH_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 6.0, "0-6m"),
    (6.0, 12.0, "6-12m"),
    (12.0, 24.0, "12-24m"),
    (24.0, np.inf, "24m+"),
)


def _risk_rates(table: pd.DataFrame) -> dict[str, Any]:
    """Per-bucket helper: cvar_5 breach rate + informative-p25 violation rate."""
    from backtests.tail_exceedance import _resolved

    t = _resolved(table)
    out: dict[str, Any] = {"n_resolved": int(len(t))}
    cv = t[np.isfinite(t["cvar_5"].to_numpy(dtype=float))]
    out["cvar5_n"] = int(len(cv))
    out["cvar5_breach_rate"] = (
        float(
            (cv["realized_pnl"].to_numpy(dtype=float) < cv["cvar_5"].to_numpy(dtype=float)).mean()
        )
        if len(cv)
        else float("nan")
    )
    q = t[np.isfinite(t["pnl_p25"].to_numpy(dtype=float))]
    q = q[q["prob_profit"].to_numpy(dtype=float) < 0.75]
    out["p25_informative_n"] = int(len(q))
    out["p25_violation_rate"] = (
        float((q["realized_pnl"].to_numpy(dtype=float) < q["pnl_p25"].to_numpy(dtype=float)).mean())
        if len(q)
        else float("nan")
    )
    return out


def compare_frozen_vs_production(
    production: pd.DataFrame,
    frozen: pd.DataFrame,
    mults: pd.DataFrame,
    *,
    cutoff: str = FREEZE_CUTOFF_DEFAULT,
    n_boot: int = 800,
    seed: int = 12345,
    block_len: int = 7,
) -> dict[str, Any]:
    """The V2-c report: rank, risk, drift, and identity blocks.

    ``production`` is the standard capture restricted to the frozen grid's
    dates (the V1 table subset); ``frozen`` is ``build_frozen_tail_table``'s
    output; ``mults`` is ``frozen_hmm_multipliers``'s output.  ``block_len``
    scales the moving-block CI to the option-horizon overlap at the grid's
    cadence (~25 trading days / cadence: every-5-bday -> 7, every-2 -> 13).
    """
    from backtests import tail_exceedance as tex
    from backtests.parameter_oos import (
        cluster_bootstrap_ci,
        per_date_cross_sectional_rho,
        restrict_top_n_per_date,
        spearman_rho,
    )

    common = sorted(set(production["date"]) & set(frozen["date"]))
    prod = production[production["date"].isin(common)].copy()
    froz = frozen[frozen["date"].isin(common)].copy()
    froz_fh = froz.copy()
    froz_fh["ev_frozen_hmm"] = apply_frozen_hmm(froz, mults)

    report: dict[str, Any] = {
        "cutoff": cutoff,
        "n_common_dates": len(common),
        "rows_production": int(len(prod)),
        "rows_frozen": int(len(froz)),
    }

    # --- rank block: five signals, all + top-5 tiers, moving-block clustered CI
    signals = {
        "production_ev_dollars": (prod, "ev_dollars"),
        "production_ev_raw": (prod, "ev_raw"),
        "frozen_ev_dollars_live_hmm": (froz, "ev_dollars"),
        "frozen_ev_raw": (froz, "ev_raw"),
        "frozen_ev_frozen_hmm": (froz_fh, "ev_frozen_hmm"),
    }
    rank: dict[str, Any] = {}
    for name, (tbl, col) in signals.items():
        entry: dict[str, Any] = {}
        # top15 is the parameter_oos section-7.2 headline tier (the S34-class
        # tradeable menu at 100 names); on 24t it nearly equals "all".
        for tier_name, tier_n in (("all", None), ("top15", 15), ("top5", 5)):
            sub = restrict_top_n_per_date(tbl, tier_n, signal_col=col)
            entry[tier_name] = {
                "xsec": per_date_cross_sectional_rho(sub, col),
                "ci_block": cluster_bootstrap_ci(
                    sub,
                    stat="cross_sectional",
                    signal_col=col,
                    n_boot=n_boot,
                    seed=seed,
                    block_len=block_len,
                ),
            }
        rank[name] = entry
    report["rank"] = rank

    # --- risk block: the full V1 statistics on both tables
    report["risk"] = {
        "production": tex.full_report(prod, meta={"variant": "production"}),
        "frozen": tex.full_report(froz, meta={"variant": "frozen", "cutoff": cutoff}),
    }

    # --- paired deltas + ordering identity on jointly ranked rows
    joined = prod.merge(froz, on=["date", "ticker"], suffixes=("_prod", "_froz"))
    report["rows_joined"] = int(len(joined))
    report["rows_only_production"] = int(len(prod) - len(joined))
    report["rows_only_frozen"] = int(len(froz) - len(joined))
    if len(joined):
        d_cvar = joined["cvar_5_froz"].to_numpy(dtype=float) - joined["cvar_5_prod"].to_numpy(
            dtype=float
        )
        d_prob = joined["prob_profit_froz"].to_numpy(dtype=float) - joined[
            "prob_profit_prod"
        ].to_numpy(dtype=float)
        d_evr = joined["ev_raw_froz"].to_numpy(dtype=float) - joined["ev_raw_prod"].to_numpy(
            dtype=float
        )
        agree = [
            spearman_rho(
                g["ev_raw_prod"].to_numpy(dtype=float), g["ev_raw_froz"].to_numpy(dtype=float)
            )
            for _, g in joined.groupby("date")
            if len(g) >= 3
        ]
        agree_arr = np.asarray([r for r in agree if np.isfinite(r)], dtype=float)
        report["paired"] = {
            "median_d_cvar5_dollars": float(np.nanmedian(d_cvar)),
            "median_d_prob_profit": float(np.nanmedian(d_prob)),
            "median_d_ev_raw": float(np.nanmedian(d_evr)),
            "frac_prob_profit_moved_gt_5pp": float(np.mean(np.abs(d_prob) > 0.05)),
            "ev_raw_ordering_agreement_mean_rho": (
                float(np.mean(agree_arr)) if agree_arr.size else float("nan")
            ),
            "ev_raw_ordering_agreement_n_dates": int(agree_arr.size),
        }

    # --- drift by months-since-cutoff
    drift: dict[str, Any] = {}
    m_prod = _months_since(prod["date"], cutoff)
    m_froz = _months_since(froz["date"], cutoff)
    for lo, hi, label in _MONTH_BUCKETS:
        pb = prod[(m_prod >= lo) & (m_prod < hi)]
        fb = froz[(m_froz >= lo) & (m_froz < hi)]
        if len(pb) == 0 and len(fb) == 0:
            continue
        drift[label] = {"production": _risk_rates(pb), "frozen": _risk_rates(fb)}
    report["drift_by_months_since_cutoff"] = drift

    # --- distribution-source tier mix
    report["distribution_source_mix"] = {
        "production": prod["distribution_source"].value_counts(normalize=True).round(4).to_dict(),
        "frozen": froz["distribution_source"].value_counts(normalize=True).round(4).to_dict(),
    }
    return report
