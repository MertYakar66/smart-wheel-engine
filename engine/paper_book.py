"""Forward paper-trading book — reporting / state library (SIM namespace).

What this is
------------
The **library half** of the forward paper-trading loop: a simulated wheel book
that the *real* engine ranks and manages day-by-day, accumulating a live equity
curve with **zero money at risk**. This module owns the parts that do NOT touch
the ranker — the SIM-namespace persistence, the phase-labelled equity history,
the calibration accumulator, and the Monte-Carlo band assembly. The driver
(:mod:`scripts.run_paper_book`) owns the parts that DO drive the engine
(ranking + the armed :class:`~engine.wheel_tracker.WheelTracker`), then hands
its outputs here for consumption and persistence.

Where it sits relative to the decision layer (CLAUDE.md §2)
-----------------------------------------------------------
**Nowhere on it.** This module is pure post-hoc state + reporting. Mirroring
:mod:`engine.sim_portfolio` (PR #483) and ``backtests.parameter_oos`` (#484/#485),
it **never imports the decision-layer trio** — ``engine.ev_engine`` /
``engine.wheel_runner`` / ``engine.candidate_dossier``. It CONSUMES ranker
output (equity curves, forecast rows) and never mutates ``ev_dollars`` /
``ev_raw`` / ``prob_profit`` / a verdict, and never feeds anything back into
ranking. ``tests/test_paper_book.py`` AST-guards that trio-free property.

SIM namespace (task invariant 4)
--------------------------------
Every artifact this module writes lands under ``$SWE_SIM_DATA_DIR`` (or the
gitignored ``data_processed/sim/`` default) — the same namespace as #483. It
**never** reads or writes ``data_processed/ibkr/`` (owned by the Dashboard
terminal, CLAUDE.md §6). :func:`resolve_book_dir` refuses any path that escapes
the SIM root or names ``ibkr`` — a defense-in-depth guard, not just a
convention.

Honesty guards carried on every report (do not paper over)
----------------------------------------------------------
* **Synthetic fills.** Paper P&L uses synthetic BSM premiums (the engine's
  `premium` column net of the S32 friction model), *not* real option fills —
  so paper edge is **optimistic** vs reality. Real fill/slippage realism is the
  first tiny-live-size test, not this.
* **Backfill vs live-forward.** Only the ``live-forward`` segment is a true
  out-of-sample paper record; the ``backfill`` seed is in-sample-ish. The
  boundary is labelled per point (``phase``) and carried in ``meta``.
* **prob_profit top bin is known over-confident.** The (0.90, 1.0] bin of
  ``prob_profit`` is 10-18pp optimistic at calm/elevated entry (finding W3 /
  ``docs/PROB_PROFIT_CALIBRATION_2026-05-28.md``); the calibration report flags
  it rather than presenting it as truth.
* The MC bands are a **model**; the realized paper curve is the honest track.
  E1 (returns beta-dominated), E3 (single-name P&L), E5 (parameter-in-sample),
  D19 (exit-leg cost omitted), D21 (horizon over-dispersion) all still stand.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

# NOTE: intentionally NO import of engine.ev_engine / engine.wheel_runner /
# engine.candidate_dossier here — AST-guarded by tests/test_paper_book.py.
# engine.sim_portfolio is a REPORTING module (also trio-free) and is imported
# lazily inside build_mc_report so this library degrades gracefully when the
# #483 module is absent (e.g. a checkout that predates it).

SCHEMA_VERSION = 1
CANONICAL_SEED = 42

# Artifact filenames written under the book dir (all inside the SIM namespace).
TRACKER_STATE_FILE = "tracker_state.json"  # WheelTracker.to_dict() continuation state
HISTORY_FILE = "sim_portfolio_history.json"  # panel-facing phase-labelled curve
FORECAST_LEDGER_FILE = "forecast_ledger.json"  # per-open prob_profit forecast + outcome
CLOSED_TRADES_FILE = "sim_closed_trades.json"  # realized wheel-cycle ledger
CALIBRATION_FILE = "calibration.json"  # rolling Brier/ECE + Wilson reliability bins
MC_BANDS_FILE = "mc_equity_bands.json"  # #483 fan (bands only), for the panel
MC_REPORT_FILE = "mc_forward_sim.json"  # full #483 distributional report
META_FILE = "meta.json"  # provenance + config + caveats

#: The prob_profit bin the engine is structurally over-confident in (finding W3).
OVERCONFIDENT_PROB_BIN = 0.90

#: Caveats surfaced on every report (mirrors engine.sim_portfolio.SIM_CAVEATS,
#: plus the two that are specific to a *paper* book: synthetic fills and the
#: backfill/live-forward boundary).
PAPER_CAVEATS: dict[str, str] = {
    "synthetic_fills": "Paper P&L uses SYNTHETIC BSM premiums (engine `premium` "
    "net of the S32 friction model), NOT real option fills — so paper edge is "
    "OPTIMISTIC vs reality. Real fill/slippage realism is the first tiny-live-size "
    "test, not this book.",
    "backfill_vs_forward": "Only the live-forward segment is a true out-of-sample "
    "paper record; the backfill seed is in-sample-ish. The boundary is labelled "
    "per point (phase) and at meta.backfill_end_date.",
    "prob_profit_top_bin": "The (0.90, 1.0] prob_profit bin is known over-confident "
    "by 10-18pp at calm/elevated entry (finding W3 / "
    "docs/PROB_PROFIT_CALIBRATION_2026-05-28.md). The calibration report FLAGS it; "
    "it is not presented as truth.",
    "E1": "~92% of the seed backtest NAV gain was equity-beta on assigned stock, "
    "not put-selection alpha.",
    "E3": "Backtest P&L was single-name-dominated — concentration risk the aggregate curve hides.",
    "E5": "Locked engine claims are parameter-IN-sample (HMM/POT-GPD/dealer clamp "
    "tuned on full history); this book inherits that.",
    "D19": "ev_dollars nets only the entry-leg cost (~$1-4/ct optimistic); the "
    "realized paper P&L shares that bias.",
    "D21": "forward-distribution samplers index trading-day bars against calendar "
    "DTE (~46% horizon over-dispersion in the underlying engine).",
    "mc_is_model": "The MC bands are a MODEL projection; the realized paper curve "
    "is the honest track. The copula tail is reporting/stress-overlay only (§2).",
    "caps_backdated": "The armed R9/R10 caps size against the tracker's live NAV "
    "(_compute_live_nav marks at date.today() + the latest close). A genuine "
    "wall-clock forward run (as_of=today) is point-in-time correct; a BACKDATED "
    "replay (as_of < the data frontier — e.g. a demo) sizes the cap denominator "
    "against the latest data, not as_of. Caps still fire (no bypass), but the "
    "denominator is not PIT on backdated runs.",
}


# ---------------------------------------------------------------------------
# 1. SIM-namespace resolution + guard (task invariant 4)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sim_root() -> Path:
    """Resolve the SIM output namespace (env override or the gitignored default).

    Mirrors :func:`scripts.run_forward_sim._sim_dir` so the paper book shares the
    #483 SIM namespace: ``$SWE_SIM_DATA_DIR`` if set, else
    ``<repo>/data_processed/sim`` (the whole ``data_processed/`` tree is
    gitignored, so SIM artifacts never commit).
    """
    env = os.environ.get("SWE_SIM_DATA_DIR")
    return Path(env) if env else _repo_root() / "data_processed" / "sim"


def resolve_book_dir(book_name: str, *, create: bool = False) -> Path:
    """Resolve (and optionally create) a book directory INSIDE the SIM root.

    Refuses to escape the SIM root and refuses any component that names
    ``ibkr`` — the real IBKR data dir owned by the Dashboard terminal is
    off-limits (task invariant 4 / CLAUDE.md §6). This is defense-in-depth:
    even a maliciously-crafted ``book_name`` cannot make the paper book touch
    ``data_processed/ibkr/``.
    """
    if not book_name or not str(book_name).strip():
        raise ValueError("book_name must be a non-empty string")
    root = sim_root().resolve()
    candidate = (root / str(book_name)).resolve()
    # Containment: candidate must live under the SIM root.
    if root != candidate and root not in candidate.parents:
        raise ValueError(
            f"book dir {candidate} escapes the SIM root {root} — refusing "
            "(paper book writes ONLY to the SIM namespace, task invariant 4)"
        )
    # Refuse an 'ibkr' component only in the operator-controlled part BELOW the
    # SIM root — NOT in an ancestor the operator never chose (e.g. a SIM root
    # that itself lives under a directory named 'ibkr'). Scoping to the relative
    # tail avoids an over-broad false-positive while still blocking any
    # book_name that would resolve into an 'ibkr' subdir.
    if any(part.lower() == "ibkr" for part in _rel_parts(candidate, root)):
        raise ValueError(
            f"book dir {candidate} names 'ibkr' — the real IBKR data dir is owned "
            "by the Dashboard terminal and is off-limits (CLAUDE.md §6)"
        )
    if create:
        candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _rel_parts(candidate: Path, root: Path) -> tuple[str, ...]:
    """The path components of ``candidate`` BELOW ``root`` (empty if equal).

    Only these are operator-controlled via ``book_name`` / the target filename;
    ancestors of the SIM root are not, so the ``ibkr`` guard scopes to these.
    Assumes containment has already been checked by the caller.
    """
    try:
        return candidate.relative_to(root).parts
    except ValueError:
        return candidate.parts  # not contained — caller's containment check rejects


def _write_json(path: Path, payload: Any) -> Path:
    """Write ``payload`` as indented UTF-8 JSON, asserting the SIM guard first."""
    path = Path(path)
    # Guard: the resolved path must be inside the SIM root and not name ibkr
    # in the tail below the root (see _rel_parts / resolve_book_dir).
    root = sim_root().resolve()
    rp = path.resolve()
    if root not in rp.parents:
        raise ValueError(f"refusing to write {rp} — outside the SIM root {root} (task invariant 4)")
    if any(part.lower() == "ibkr" for part in _rel_parts(rp, root)):
        raise ValueError(f"refusing to write {rp} — names 'ibkr' below the SIM root (CLAUDE.md §6)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")
    return path


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Held-to-expiry outcome (the quantity prob_profit models)
# ---------------------------------------------------------------------------


def held_to_expiry_put_pnl(strike: float, premium: float, spot_at_expiry: float) -> float:
    """Held-to-expiry P&L of a short cash-secured put, dollars per contract.

    OTM at expiry → keep the full premium. ITM → assigned at strike, the
    long-stock leg marked at ``spot_at_expiry`` (the immediate-sell convention
    S22 / S27 / ``_common._forward_replay_realized_pnl`` use). Mirrors that
    helper so the paper book's outcome definition is byte-identical to the
    locked regression harness — kept local to preserve this module's trio-free
    property.
    """
    intrinsic = max(0.0, float(strike) - float(spot_at_expiry))
    return (float(premium) - intrinsic) * 100.0


def put_win(strike: float, premium: float, spot_at_expiry: float) -> int:
    """Binary realized outcome for a short put: 1 iff held-to-expiry P&L > 0.

    This matches the engine's ``prob_profit`` definition exactly —
    ``prob_profit = mean(pnls > 0)`` (``ev_engine.py``) — and the outcome
    convention the locked calibration studies use
    (``test_parameter_oos.py``: ``(realized_pnl > 0).astype(float)``).
    """
    return 1 if held_to_expiry_put_pnl(strike, premium, spot_at_expiry) > 0 else 0


# ---------------------------------------------------------------------------
# 3. Calibration — wilson / reliability (mirrors scripts/ibkr_ev_calibration.py)
# ---------------------------------------------------------------------------
#
# scripts/ibkr_ev_calibration.py imports the decision-layer trio at module top
# (EVEngine + wheel_runner._resolve_pit_atm_iv), so this trio-free module cannot
# import its wilson/reliability. They are re-implemented here byte-identically;
# tests/test_paper_book.py pins that this reimplementation matches the script's
# functions on a shared fixture (verified reuse, single behavioural source).


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (mirror of the helper in
    ``scripts/ibkr_ev_calibration.py``)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def reliability(preds: Sequence[float], outcomes: Sequence[float], n_bins: int = 10):
    """Per-decile reliability table + Brier + ECE (mirror of the helper in
    ``scripts/ibkr_ev_calibration.py``). Returns ``(rows, brier, ece)``."""
    preds = np.asarray(preds, float)
    outcomes = np.asarray(outcomes, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    N = len(preds)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin": f"[{lo:.1f},{hi:.1f})",
                    "n": 0,
                    "mean_pred": None,
                    "obs": None,
                    "ci_lo": None,
                    "ci_hi": None,
                }
            )
            continue
        mp = float(preds[mask].mean())
        k = int(outcomes[mask].sum())
        obs = k / n
        lo_ci, hi_ci = wilson(k, n)
        ece += (n / N) * abs(mp - obs)
        rows.append(
            {
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": n,
                "mean_pred": round(mp, 4),
                "obs": round(obs, 4),
                "ci_lo": round(lo_ci, 4),
                "ci_hi": round(hi_ci, 4),
            }
        )
    brier = float(np.mean((preds - outcomes) ** 2)) if N else float("nan")
    return rows, brier, ece


def _calibration_block(entries: list[dict], n_bins: int) -> dict:
    """Reliability + Brier + ECE over a set of SETTLED forecast entries.

    Each entry needs ``prob_profit`` (forecast) and ``outcome`` (0/1). Entries
    with a null/NaN forecast or outcome are dropped. Returns an empty-shaped
    block (``n=0``) when nothing has settled yet.
    """
    preds, outs = [], []
    for e in entries:
        p = e.get("prob_profit")
        o = e.get("outcome")
        if p is None or o is None:
            continue
        try:
            pf = float(p)
            of = float(o)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(pf) and math.isfinite(of)):
            continue
        preds.append(pf)
        outs.append(of)
    n = len(preds)
    if n == 0:
        return {"n": 0, "brier": None, "ece": None, "bins": [], "mean_pred": None, "mean_obs": None}
    rows, brier, ece = reliability(preds, outs, n_bins=n_bins)
    return {
        "n": n,
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "mean_pred": round(float(np.mean(preds)), 4),
        "mean_obs": round(float(np.mean(outs)), 4),
        "bins": rows,
    }


def compute_calibration(
    entries: Iterable[dict], *, n_bins: int = 10, as_of: str | None = None
) -> dict:
    """Rolling calibration accumulator over the forecast ledger.

    Splits the settled forecast entries into ``pooled`` / ``backfill`` /
    ``forward`` blocks (by the entry's ``phase``) and computes a reliability
    table (per-bin n, mean predicted, observed frequency, Wilson 95% CI) +
    Brier + ECE for each. The (0.90, 1.0] top bin is FLAGGED as known
    over-confident (finding W3) rather than trusted.

    Args:
        entries: the forecast-ledger entries (settled + pending; pending are
            skipped inside each block).
        n_bins: reliability decile count (10 canonical).
        as_of: optional stamp for the report header.

    Returns:
        A JSON-safe dict: ``{schema_version, as_of, pooled, backfill, forward,
        n_settled_total, n_pending_total, top_bin_flag, notes}``.
    """
    entries = list(entries)
    settled = [e for e in entries if e.get("outcome") is not None]
    pending = [e for e in entries if e.get("outcome") is None]
    backfill = [e for e in settled if e.get("phase") == "backfill"]
    forward = [e for e in settled if e.get("phase") == "forward"]

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "engine-measured",  # observed calibration, not a model
        "as_of": as_of,
        "n_settled_total": len(settled),
        "n_pending_total": len(pending),
        "pooled": _calibration_block(settled, n_bins),
        "backfill": _calibration_block(backfill, n_bins),
        "forward": _calibration_block(forward, n_bins),
        "top_bin_flag": {
            "bin": f"({OVERCONFIDENT_PROB_BIN:.2f}, 1.0]",
            "known_overconfident": True,
            "note": PAPER_CAVEATS["prob_profit_top_bin"],
        },
        "notes": [
            "outcome = 1 iff held-to-expiry short-put P&L > 0 (matches the "
            "engine's prob_profit = mean(pnls>0) definition).",
            "Only the 'forward' block is a genuine out-of-sample paper record; "
            "'backfill' is in-sample-ish.",
            "Per-bin n is small early — read the Wilson CI, do not label a bin "
            "SUPPORTED until n is adequate.",
        ],
    }


# ---------------------------------------------------------------------------
# 4. Forecast ledger — append at open, settle at expiry
# ---------------------------------------------------------------------------


def new_forecast_entry(
    *,
    entry_date: str,
    ticker: str,
    strike: float,
    premium: float,
    prob_profit: float | None,
    expiration_date: str,
    phase: str,
    ev_dollars: float | None = None,
) -> dict:
    """Build a single forecast-ledger entry (unsettled).

    ``premium`` is the (friction-adjusted, synthetic-BSM) credit actually
    booked; ``prob_profit`` / ``ev_dollars`` are the engine's *shipped*
    forecasts, copied verbatim — never recomputed here (§2: this module only
    consumes ranker output).
    """
    return {
        "entry_date": str(entry_date),
        "ticker": str(ticker),
        "strike": float(strike),
        "premium": float(premium),
        "prob_profit": None if prob_profit is None else float(prob_profit),
        "ev_dollars": None if ev_dollars is None else float(ev_dollars),
        "expiration_date": str(expiration_date),
        "phase": str(phase),
        "settled": False,
        "spot_at_expiry": None,
        "realized_pnl": None,
        "outcome": None,
    }


def settle_due_forecasts(
    entries: list[dict],
    as_of: date,
    spot_at_expiry_lookup: Callable[[str, date], float | None],
) -> int:
    """Settle every unsettled entry whose expiry has passed (<= ``as_of``).

    ``spot_at_expiry_lookup(ticker, expiry_date)`` is injected by the driver
    (it wraps the connector) so this module stays connector-free and trio-free.
    Returns the number of entries newly settled. Idempotent: already-settled
    entries are skipped, and an entry whose spot is unavailable stays pending
    (settled again on a later call once data lands).
    """
    n_settled = 0
    for e in entries:
        if e.get("settled"):
            continue
        try:
            exp = date.fromisoformat(str(e["expiration_date"])[:10])
        except (ValueError, KeyError):
            continue
        if exp > as_of:
            continue
        spot = spot_at_expiry_lookup(str(e["ticker"]), exp)
        if spot is None:
            continue  # data not available yet — stay pending
        pnl = held_to_expiry_put_pnl(e["strike"], e["premium"], spot)
        e["spot_at_expiry"] = float(spot)
        e["realized_pnl"] = float(pnl)
        e["outcome"] = 1 if pnl > 0 else 0
        e["settled"] = True
        n_settled += 1
    return n_settled


# ---------------------------------------------------------------------------
# 5. Phase-labelled equity history (mirrors portfolio_history.json)
# ---------------------------------------------------------------------------


def _phase_for(day: date, backfill_end: date | None) -> str:
    """A point is 'backfill' up to and including backfill_end, else 'forward'."""
    if backfill_end is None:
        return "forward"
    return "backfill" if day <= backfill_end else "forward"


def _label_for(day: date) -> str:
    """Cross-platform 'Mon D' label (matches dashboard_refresh._sync_curve)."""
    return f"{day.strftime('%b')} {day.day}"


def _coerce_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def history_points_from_equity_curve(
    equity_curve: Sequence[dict],
    *,
    backfill_end: date | None,
    spy_index: dict[str, float] | None = None,
) -> list[dict]:
    """Turn a tracker ``equity_curve`` into panel points with a ``phase`` tag.

    Output shape mirrors ``portfolio_history.json`` points
    (``{label, date, port, spy, premium, phase}``) so the dashboard's existing
    ``equity_view`` / ``equity-curve.tsx`` can consume it. ``spy`` comes from
    ``spy_index`` (``{iso_date: indexed_value}``) when supplied, else ``None``
    (the chart renders a benchmark gap — never a fabricated copy, matching
    ``_sync_curve``'s no-Gateway behaviour). ``premium`` is left ``None`` here
    (the panel's premium bars are optional and are driven off the forecast
    ledger, not the equity marks).
    """
    spy_index = spy_index or {}
    points: list[dict] = []
    for rec in equity_curve:
        day = _coerce_date(rec.get("date"))
        if day is None:
            continue
        iso = day.isoformat()
        pv = rec.get("portfolio_value")
        if pv is None:
            continue
        points.append(
            {
                "label": _label_for(day),
                "date": iso,
                "port": round(float(pv), 2),
                "spy": spy_index.get(iso),
                "premium": None,
                "phase": _phase_for(day, backfill_end),
            }
        )
    return points


def indexed_spy_series(
    spy_closes: dict[str, float],
    point_dates: Sequence[str],
    base_port: float,
) -> dict[str, float]:
    """Index a raw ``{iso: spy_close}`` map so ``spy[first] == base_port``.

    Matches the dashboard convention (benchmark indexed to the book's starting
    NAV so the two lines share an origin). A date with no close is omitted
    (its point renders a benchmark gap). Returns ``{iso: indexed_value}`` for
    the requested ``point_dates`` only.
    """
    if not spy_closes or not point_dates:
        return {}
    # Anchor on the first point date that has a close.
    anchor_close = None
    for d in point_dates:
        if d in spy_closes and spy_closes[d] > 0:
            anchor_close = spy_closes[d]
            break
    if anchor_close is None:
        return {}
    out: dict[str, float] = {}
    for d in point_dates:
        c = spy_closes.get(d)
        if c and c > 0:
            out[d] = round(float(base_port) * float(c) / float(anchor_close), 2)
    return out


def upsert_history_point(history: dict, new_point: dict) -> bool:
    """Idempotently append/replace a point (mirrors ``_sync_curve`` same-day).

    If the last point shares ``new_point``'s ``date``, it is REPLACED in place
    and ``False`` is returned (no new day added). Otherwise the point is
    appended and ``True`` is returned. This is the idempotency backbone of the
    daily forward-append: re-running the same ``as_of`` never grows the curve.
    """
    pts = history.setdefault("points", [])
    if pts and pts[-1].get("date") == new_point.get("date"):
        pts[-1] = new_point
        return False
    pts.append(new_point)
    return True


def last_history_date(history: dict) -> date | None:
    pts = history.get("points") or []
    if not pts:
        return None
    return _coerce_date(pts[-1].get("date"))


def new_history(*, label: str, initial_capital: float, backfill_end: date | None) -> dict:
    """A fresh, empty panel-history artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "simulated",
        "kind": "engine-measured",
        "label": label,
        "as_of": None,
        "backfill_end_date": backfill_end.isoformat() if backfill_end else None,
        "initial_capital": float(initial_capital),
        "final_nav": None,
        "points": [],
    }


# ---------------------------------------------------------------------------
# 6. Monte-Carlo bands — reuse engine.sim_portfolio (#483), soft dependency
# ---------------------------------------------------------------------------


def build_mc_report(
    *,
    equity_curve: Sequence[dict],
    initial_capital: float,
    realized_final_nav: float | None = None,
    per_name_returns: dict[str, np.ndarray] | None = None,
    weights: dict[str, float] | None = None,
    seed: int = CANONICAL_SEED,
    block_size: int = 21,
    n_simulations: int = 10_000,
    label: str = "paper_book",
) -> dict:
    """Assemble the #483 distributional report, degrading gracefully.

    Reuses :func:`engine.sim_portfolio.build_sim_report` (block-bootstrap equity
    fan + terminal/drawdown distributions + correlation-to-1 copula tail,
    reconciled against the realized final NAV). Returns a ``status`` field so
    the caller/panel can tell a real report from a stub:

    * ``ok`` — the full #483 report (with an ``mc_bands`` convenience slice).
    * ``insufficient_history`` — too few equity marks for a block (short book).
    * ``sim_portfolio_unavailable`` — the #483 module is not importable in this
      checkout (the paper loop still runs; the fan is simply absent).
    """
    try:
        from engine.sim_portfolio import build_sim_report  # trio-free reporting module
    except Exception as exc:  # pragma: no cover — only when #483 absent
        return {"status": "sim_portfolio_unavailable", "reason": repr(exc), "label": label}

    try:
        report = build_sim_report(
            equity_curve=list(equity_curve),
            initial_capital=float(initial_capital),
            realized_final_nav=realized_final_nav,
            per_name_returns=per_name_returns or None,
            weights=weights or None,
            n_simulations=n_simulations,
            block_size=block_size,
            seed=seed,
            label=label,
        )
    except ValueError as exc:
        # Short/empty book — the block bootstrap cannot form a block.
        n_marks = len(list(equity_curve))
        return {
            "status": "insufficient_history",
            "reason": str(exc),
            "n_equity_marks": n_marks,
            "block_size": block_size,
            "label": label,
        }

    report["status"] = "ok"
    model = report.get("model", {})
    report["mc_bands"] = {
        "kind": "model",
        "band_days": model.get("band_days"),
        "equity_bands": model.get("equity_bands"),
        "quantiles": model.get("quantiles"),
    }
    return report


def mc_bands_only(mc_report: dict) -> dict:
    """Extract the panel's bands-only payload from a full MC report."""
    if mc_report.get("status") != "ok":
        return {
            "kind": "model",
            "status": mc_report.get("status"),
            "reason": mc_report.get("reason"),
        }
    bands = mc_report.get("mc_bands") or {}
    return {
        "kind": "model",
        "status": "ok",
        "band_days": bands.get("band_days"),
        "equity_bands": bands.get("equity_bands"),
        "quantiles": bands.get("quantiles"),
    }


# ---------------------------------------------------------------------------
# 7. PaperBookStore — SIM-namespace persistence of the panel artifacts
# ---------------------------------------------------------------------------


class PaperBookStore:
    """Thin persistence wrapper for a paper book's SIM-namespace artifacts.

    Owns nothing but I/O + the SIM guard. The driver builds the payloads (it has
    the ranker + tracker); this class only reads/writes them under the book dir.
    """

    def __init__(self, book_name: str, *, create: bool = False):
        self.book_name = str(book_name)
        self.dir = resolve_book_dir(book_name, create=create)

    # ---- paths -----------------------------------------------------------
    def _p(self, name: str) -> Path:
        return self.dir / name

    def tracker_state_path(self) -> Path:
        return self._p(TRACKER_STATE_FILE)

    # ---- generic exists/load --------------------------------------------
    def exists(self, name: str) -> bool:
        return self._p(name).exists()

    def load(self, name: str) -> Any | None:
        p = self._p(name)
        return _read_json(p) if p.exists() else None

    def save(self, name: str, payload: Any) -> Path:
        return _write_json(self._p(name), payload)

    # ---- typed convenience ----------------------------------------------
    def load_history(self) -> dict | None:
        return self.load(HISTORY_FILE)

    def save_history(self, history: dict) -> Path:
        return self.save(HISTORY_FILE, history)

    def load_forecast_ledger(self) -> list[dict]:
        led = self.load(FORECAST_LEDGER_FILE)
        if led is None:
            return []
        return led.get("entries", []) if isinstance(led, dict) else list(led)

    def save_forecast_ledger(self, entries: list[dict]) -> Path:
        return self.save(
            FORECAST_LEDGER_FILE,
            {"schema_version": SCHEMA_VERSION, "entries": list(entries)},
        )

    def save_closed_trades(self, closed: list[dict], *, as_of: str | None = None) -> Path:
        return self.save(
            CLOSED_TRADES_FILE,
            {
                "schema_version": SCHEMA_VERSION,
                "source": "simulated",
                "as_of": as_of,
                "trades": list(closed),
            },
        )

    def save_calibration(self, calib: dict) -> Path:
        return self.save(CALIBRATION_FILE, calib)

    def load_calibration(self) -> dict | None:
        return self.load(CALIBRATION_FILE)

    def save_mc(self, mc_report: dict) -> tuple[Path, Path]:
        rp = self.save(MC_REPORT_FILE, mc_report)
        bp = self.save(MC_BANDS_FILE, mc_bands_only(mc_report))
        return rp, bp

    def load_mc_bands(self) -> dict | None:
        return self.load(MC_BANDS_FILE)

    def save_meta(self, meta: dict) -> Path:
        return self.save(META_FILE, meta)

    def load_meta(self) -> dict | None:
        return self.load(META_FILE)


def now_iso() -> str:
    """UTC now, ISO-8601 — a single stamping point (kept out of pure fns)."""
    return datetime.now(UTC).isoformat()


def build_meta(
    *,
    book_name: str,
    provider: str,
    universe: Sequence[str],
    window: dict,
    backfill_end: date | None,
    data_frontier: str | None,
    config: dict,
    caps_armed: bool,
) -> dict:
    """Assemble the provenance/meta artifact carried beside the numbers."""
    return {
        "schema_version": SCHEMA_VERSION,
        "book_name": book_name,
        "source": "simulated",
        "provider": provider,
        "universe": list(universe),
        "universe_size": len(list(universe)),
        "window": dict(window),
        "backfill_end_date": backfill_end.isoformat() if backfill_end else None,
        "data_frontier": data_frontier,
        "caps_armed": bool(caps_armed),
        "caps_detail": "R9 sector (25% NAV) + R10 single-name (10% NAV) via "
        "make_live_book_tracker on the FORWARD path; the backfill seed uses the "
        "caps-off _common.run_backtest tracker (PIT-correct; the armed caps' NAV "
        "uses date.today() marks and would leak future prices in a historical seed).",
        "config": dict(config),
        "caveats": dict(PAPER_CAVEATS),
        "generated_at": now_iso(),
    }
