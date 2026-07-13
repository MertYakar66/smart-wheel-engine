"""Reverse stress (V5) — what is the cheapest path to ruin the gate stack permits?

`docs/VALIDATION_PHASE_PLAN.md` section 8 is the pre-registered design.
Two components, both pure offline computation (no ranker re-runs):

* **V5-a — worst admissible book.**  For each entry date in a V1 tail
  table, a deliberately-hindsight adversary composes the worst-realized
  book the entry-time gate stack PERMITS: engine-tradeable rows only
  (`ev_dollars > 0`), up to the R10 per-name cap in contracts, the R9
  sector cap, and the cash-secured collateral budget.  Greedy
  loss-per-collateral-dollar search (a documented lower bound on the true
  optimum — conservative).  The hindsight is the point and is guarded in
  the plan doc: this measures what the gates permit, not what the
  strategy does.
* **V5-b — margin procyclicality, resolved honestly.**  The tracker's BP
  reserve is FULL collateral (cash-secured by construction), so the
  classical margin spiral is structurally absent under the CSP mandate.
  What remains: the assignment-wave stress (BP-saturated engine-chosen
  book on a crisis eve, daily intrinsic-only marking — a LOWER bound on
  trough damage, conservative toward the engine — with the
  trough-liquidation counterfactual) and the Reg-T LEVERED counterfactual
  (daily re-margin at a stressed multiplier; first margin-call day).

Scope / invariants (CLAUDE.md section 2): measurement-only; nothing here
feeds a trade, a verdict, or a gate; reported "worst books" are damage
bounds, never sizing or selection advice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-registered constants (plan doc section 8)
# ---------------------------------------------------------------------------

NAV = 1_000_000.0
RUIN_PCT = 0.25  # ruin-class: realized book loss >= 25% NAV in one cycle
R10_CAP_PCT = 0.10  # per-name short-option notional cap (fraction of NAV)
R9_CAP_PCT = 0.25  # per-sector cap (fraction of NAV)
STRESS_MULTS: tuple[float, ...] = (1.0, 1.25, 1.5)
CRISIS_EVES: tuple[str, ...] = ("2020-02-19", "2022-01-03", "2024-07-31", "2025-04-01")
HORIZON_BDAYS = 25  # ~35 calendar days of daily marks


# ---------------------------------------------------------------------------
# V5-a — worst admissible book
# ---------------------------------------------------------------------------


def load_sector_map(conn: Any, tickers: Sequence[str]) -> dict[str, str]:
    """GICS sector per ticker from the fundamentals snapshot.

    Current-labels approximation (documented in the plan doc); missing
    sector -> "unknown" (exempt from the R9 cap, counted loudly).
    """
    out: dict[str, str] = {}
    for t in tickers:
        sector = "unknown"
        try:
            f = conn.get_fundamentals(t)
            if f:
                sector = str(f.get("gics_sector_name") or f.get("sector") or "unknown")
        except Exception:  # noqa: BLE001 — missing fundamentals degrade, never abort
            pass
        out[t] = sector or "unknown"
    return out


def worst_admissible_book(
    rows: pd.DataFrame,
    sectors: dict[str, str],
    *,
    nav: float = NAV,
    top_bin_only: bool = False,
) -> dict[str, Any] | None:
    """The adversary's book for one entry date's rows.

    Entry-time-honest admissibility: `ev_dollars > 0` (the trade filter),
    per-name contracts up to the R10 cap, R9 sector budget, total
    collateral <= nav.  Hindsight selection: only realized LOSERS enter
    (adding a winner never increases loss), greedy by loss per collateral
    dollar.  Returns None when no admissible loser exists.
    """
    if rows is None or rows.empty:
        return None
    t = rows.copy()
    t = t[np.isfinite(pd.to_numeric(t["realized_pnl"], errors="coerce"))]
    t = t[pd.to_numeric(t["ev_dollars"], errors="coerce") > 0.0]
    if top_bin_only:
        t = t[pd.to_numeric(t["prob_profit"], errors="coerce") > 0.90]
    if t.empty:
        return None
    # One row per ticker (best loss density if duplicated strikes/dtes).
    t = t.sort_values("realized_pnl").drop_duplicates(subset="ticker", keep="first")

    picks: list[dict[str, Any]] = []
    budget = nav
    sector_used: dict[str, float] = {}
    cand = []
    for r in t.itertuples(index=False):
        strike = float(r.strike)
        pnl = float(r.realized_pnl)
        if strike <= 0 or pnl >= 0:
            continue  # winners never help the adversary
        per_contract_collateral = strike * 100.0
        max_c = int((R10_CAP_PCT * nav) // per_contract_collateral)
        if max_c < 1:
            continue  # R10 refuses even one contract
        cand.append(
            {
                "ticker": str(r.ticker),
                "strike": strike,
                "pnl_per_contract": pnl,
                "cvar_per_contract": float(r.cvar_5)
                if np.isfinite(pd.to_numeric(pd.Series([r.cvar_5]), errors="coerce").iloc[0])
                else float("nan"),
                "collateral_per_contract": per_contract_collateral,
                "max_contracts": max_c,
                "loss_density": -pnl / per_contract_collateral,
                "sector": sectors.get(str(r.ticker), "unknown"),
            }
        )
    if not cand:
        return None
    cand.sort(key=lambda c: c["loss_density"], reverse=True)
    for c in cand:
        if budget <= 0:
            break
        sector_room = (
            float("inf")
            if c["sector"] == "unknown"
            else R9_CAP_PCT * nav - sector_used.get(c["sector"], 0.0)
        )
        room = min(budget, sector_room)
        n_c = min(c["max_contracts"], int(room // c["collateral_per_contract"]))
        if n_c < 1:
            continue
        collateral = n_c * c["collateral_per_contract"]
        budget -= collateral
        if c["sector"] != "unknown":
            sector_used[c["sector"]] = sector_used.get(c["sector"], 0.0) + collateral
        picks.append({**c, "n_contracts": n_c, "collateral": collateral})
    if not picks:
        return None

    loss = -sum(p["n_contracts"] * p["pnl_per_contract"] for p in picks)
    cvars = [
        p["n_contracts"] * p["cvar_per_contract"]
        for p in picks
        if np.isfinite(p["cvar_per_contract"])
    ]
    modeled_cvar = float(sum(cvars)) if cvars else float("nan")
    vix = pd.to_numeric(rows["vix_entry"], errors="coerce").dropna()
    return {
        "loss_dollars": float(loss),
        "loss_pct_nav": float(loss / nav),
        "modeled_book_cvar_dollars": modeled_cvar,
        "realized_over_modeled_cvar": (
            float(loss / -modeled_cvar)
            if np.isfinite(modeled_cvar) and modeled_cvar < 0
            else float("nan")
        ),
        "n_names": len(picks),
        "n_unknown_sector": sum(1 for p in picks if p["sector"] == "unknown"),
        "collateral_used": float(nav - budget),
        "vix_entry": float(vix.iloc[0]) if len(vix) else float("nan"),
        "names": [
            {k: p[k] for k in ("ticker", "n_contracts", "strike", "pnl_per_contract", "sector")}
            for p in picks[:10]
        ],
    }


def reverse_stress_search(
    table: pd.DataFrame,
    sectors: dict[str, str],
    *,
    nav: float = NAV,
    top_bin_only: bool = False,
) -> dict[str, Any]:
    """V5-a over every entry date; distribution + ruin list + VIX strata."""
    from backtests.tail_exceedance import vix_band

    per_date: list[dict[str, Any]] = []
    for d, rows in table.groupby("date"):
        book = worst_admissible_book(rows, sectors, nav=nav, top_bin_only=top_bin_only)
        if book is not None:
            per_date.append({"date": str(d), **book})
    losses = np.array([b["loss_pct_nav"] for b in per_date], dtype=float)
    ruin = [b for b in per_date if b["loss_pct_nav"] >= RUIN_PCT]
    strata: dict[str, Any] = {}
    bands = np.array([vix_band(b["vix_entry"]) for b in per_date])
    for band in ("calm", "elevated", "crisis", "unknown"):
        m = bands == band
        if m.sum():
            strata[band] = {
                "n_dates": int(m.sum()),
                "worst_loss_pct": float(losses[m].max()),
                "mean_loss_pct": float(losses[m].mean()),
                "n_ruin": int((losses[m] >= RUIN_PCT).sum()),
            }
    return {
        "variant": "top_bin_only" if top_bin_only else "tradeable",
        "nav": nav,
        "ruin_pct": RUIN_PCT,
        "n_dates": len(per_date),
        "loss_pct_quantiles": {
            q: float(np.quantile(losses, float(q))) if len(losses) else float("nan")
            for q in ("0.5", "0.9", "0.99")
        },
        "worst": sorted(per_date, key=lambda b: -b["loss_pct_nav"])[:10],
        "n_ruin_dates": len(ruin),
        "ruin_dates": [b["date"] for b in sorted(ruin, key=lambda b: -b["loss_pct_nav"])],
        "by_vix_band": strata,
    }


# ---------------------------------------------------------------------------
# V5-b — assignment wave + levered counterfactual
# ---------------------------------------------------------------------------


def build_saturated_book(rows: pd.DataFrame, *, nav: float = NAV) -> list[dict[str, Any]]:
    """The ENGINE-chosen (not adversarial) BP-saturated book on a crisis eve:
    rows ranked by `ev_dollars` desc, 1 contract per name, until the
    cash-secured collateral budget is exhausted (the V4 saturation fact)."""
    t = rows.copy()
    t = t[pd.to_numeric(t["ev_dollars"], errors="coerce") > 0.0]
    t = t.sort_values("ev_dollars", ascending=False).drop_duplicates(subset="ticker", keep="first")
    book: list[dict[str, Any]] = []
    budget = nav
    for r in t.itertuples(index=False):
        strike = float(r.strike)
        if strike <= 0:
            continue
        collateral = strike * 100.0
        if collateral > budget:
            continue
        budget -= collateral
        book.append(
            {
                "ticker": str(r.ticker),
                "strike": strike,
                "premium": float(r.premium),
                "collateral": collateral,
            }
        )
    return book


def _closes(conn: Any, ticker: str, start: str, n_bdays: int) -> pd.Series | None:
    try:
        df = conn.get_ohlcv(ticker, start_date=start)
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty or "close" not in df.columns:
        return None
    return df["close"].dropna().iloc[: n_bdays + 1]


def replay_assignment_wave(
    book: Sequence[dict[str, Any]],
    conn: Any,
    *,
    eve: str,
    nav: float = NAV,
    horizon_bdays: int = HORIZON_BDAYS,
    stress_mults: Sequence[float] = STRESS_MULTS,
) -> dict[str, Any]:
    """Daily intrinsic-only replay of a saturated book over one crisis window.

    Cash-secured path: book mark_t = sum(premium - max(0, K - S_t)) x 100 —
    intrinsic-only (no time value), so the trough is a LOWER bound on damage
    (conservative toward the engine).  Levered counterfactual: capital =
    entry Reg-T margin; daily maintenance = Reg-T recomputed at S_t x a
    stressed multiplier; first day equity < maintenance = the margin call
    the CSP mandate structurally cannot have.
    """
    from engine.transaction_costs import calculate_reg_t_margin_short_put

    paths: dict[str, pd.Series] = {}
    for pos in book:
        s = _closes(conn, pos["ticker"], eve, horizon_bdays)
        if s is not None and len(s) >= 2:
            paths[pos["ticker"]] = s
    live = [p for p in book if p["ticker"] in paths]
    if not live:
        return {"eve": eve, "error": "no price paths"}
    n_days = min(len(paths[p["ticker"]]) for p in live)

    pnl_path = np.zeros(n_days)
    for p in live:
        s = paths[p["ticker"]].to_numpy(dtype=float)[:n_days]
        pnl_path += (p["premium"] - np.maximum(0.0, p["strike"] - s)) * 100.0
    terminal_spots = {p["ticker"]: float(paths[p["ticker"]].iloc[n_days - 1]) for p in live}
    assigned = [p for p in live if terminal_spots[p["ticker"]] < p["strike"]]

    entry_margin = sum(
        calculate_reg_t_margin_short_put(
            strike=p["strike"],
            underlying_price=float(paths[p["ticker"]].iloc[0]),
            premium=p["premium"],
        )
        for p in live
    )
    levered: dict[str, Any] = {}
    for mult in stress_mults:
        call_day = None
        shortfall = 0.0
        for d in range(1, n_days):
            maint = mult * sum(
                calculate_reg_t_margin_short_put(
                    strike=p["strike"],
                    underlying_price=float(paths[p["ticker"]].iloc[d]),
                    premium=p["premium"],
                )
                for p in live
            )
            equity = entry_margin + pnl_path[d]
            if equity < maint:
                call_day = d
                shortfall = float(maint - equity)
                break
        levered[f"{mult:g}"] = {"first_call_bday": call_day, "shortfall_dollars": shortfall}

    trough = int(np.argmin(pnl_path))
    return {
        "eve": eve,
        "n_positions": len(live),
        "collateral_used": float(sum(p["collateral"] for p in live)),
        "n_days_replayed": n_days,
        "trough_bday": trough,
        "trough_pnl_dollars": float(pnl_path[trough]),
        "trough_liquidation_pct_nav": float(-pnl_path[trough] / nav),
        "terminal_pnl_dollars": float(pnl_path[n_days - 1]),
        "terminal_pct_nav": float(-pnl_path[n_days - 1] / nav),
        "assignment_fraction": float(len(assigned) / len(live)),
        "entry_reg_t_margin": float(entry_margin),
        "levered_counterfactual": levered,
    }
