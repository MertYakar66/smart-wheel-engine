"""Phase 3 — EV calibration: does the engine's PIT prediction match what actually
happened on the operator's real wheel options — short puts (CSPs) AND covered
calls (short calls)?

For every short option the operator SOLD-TO-OPEN (from the IBKR Flex fills) —
both the put leg and the covered-call (short-call) leg in isolation — this:
  1. Reconstructs the engine's **point-in-time** prediction AT ENTRY by reusing
     the exact ranker machinery — PIT spot (OHLCV <= entry), PIT ATM IV
     (`_resolve_pit_atm_iv`), as-of risk-free, the empirical forward distribution
     (`best_available_forward_distribution` + `realized_vol_widened_log_returns`)
     — then calls `EVEngine.evaluate` at the operator's **exact strike / DTE /
     actual opening credit**. (`prob_profit`/`prob_assignment`/`mean_pnl` are
     computed from the forward distribution *before* any regime scaling, so
     `regime_multiplier=1.0` does not affect them.)
  2. Computes the realized **hold-to-expiry** outcome from the underlying's actual
     close on the option's expiry date (Bloomberg OHLCV): short-put P&L =
     premium - max(0, strike - S_expiry); win = P&L > 0; assigned = S_expiry < K.
  3. Aggregates reliability diagrams (predicted prob_profit / prob_assignment vs
     observed), Brier score, expected calibration error (ECE) with per-bin n and
     Wilson 95% CIs, plus mean_pnl (ev_raw) vs realized.

READ-ONLY / observational (CLAUDE.md §2/§3): this is analysis. It *uses*
`EVEngine.evaluate` (the authoritative evaluator) — it never bypasses it, never
issues an EV-authority token, never converts anything into a tradeable verdict,
and does not modify the decision-layer trio. Real account data stays gitignored.

LIMITATIONS (documented in docs/IBKR_EV_CALIBRATION.md):
  - Universe-restricted to S&P-500 names with Bloomberg data (CLS, TSM ADR, CCO,
    CNQ, ENB, etc. are out of the engine's mandate and excluded).
  - dividend_yield comes from the dateless fundamentals snapshot (slow-moving;
    matches engine-as-deployed; minor lookahead).
  - Calibrates against the *hold-to-expiry* counterfactual even for puts the
    operator closed early — that is the quantity the engine's probabilities model.

Usage:
    python scripts/ibkr_ev_calibration.py "<A.csv>" "<B.csv>" --out data_processed/ibkr
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]

# Reuse the Phase-4 fill ingestion (load_fills / build_fx / iso8 / underlying / fnum).
_SPEC = importlib.util.spec_from_file_location(
    "ibkr_flex_ledger", _REPO / "scripts" / "ibkr_flex_ledger.py"
)
flex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(flex)

from engine.data_connector import MarketDataConnector  # noqa: E402
from engine.data_integration import get_current_risk_free_rate  # noqa: E402
from engine.ev_engine import EVEngine, ShortOptionTrade  # noqa: E402
from engine.forward_distribution import (  # noqa: E402
    best_available_forward_distribution,
    realized_vol_widened_log_returns,
)
from engine.wheel_runner import _resolve_pit_atm_iv  # noqa: E402


def load_universe() -> set[str]:
    p = _REPO / "data_raw" / "sp500_constituents_current.csv"
    out = set()
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines()[1:]:
            t = ln.split(",")[0].strip().strip('"').upper()
            if t:
                out.add(t)
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def reliability(preds, outcomes, n_bins=10):
    """Per-decile reliability table + Brier + ECE."""
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
    brier = float(np.mean((preds - outcomes) ** 2))
    return rows, brier, ece


def main(csv_a, csv_b, out_dir):
    fills, na, nb, dropped, a_max = flex.load_fills(csv_a, csv_b)
    usdcad = flex.build_fx(fills)
    universe = load_universe()
    conn = MarketDataConnector()
    ev = EVEngine()

    # short-option OPENS (SELL, Open) — each short PUT is a CSP entry, each short
    # CALL is a covered-call entry. Both legs are calibrated (the short-call leg in
    # isolation, which is exactly what the engine's call-leg prob_profit models).
    opens = [
        (r["Put/Call"], r)
        for r in fills
        if r["AssetClass"] == "OPT"
        and r["Put/Call"] in ("P", "C")
        and r["Buy/Sell"] == "SELL"
        and (r["Open/CloseIndicator"] or "").startswith("O")
    ]

    ohlcv_cache: dict[str, pd.DataFrame | None] = {}

    def get_ohlcv(t):
        if t not in ohlcv_cache:
            try:
                df = conn.get_ohlcv(t)
                if df is not None and len(df):
                    df = df.copy()
                    df.columns = [c.lower() for c in df.columns]
                    if "date" not in df.columns:
                        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
                        df.columns = [c.lower() for c in df.columns]
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                ohlcv_cache[t] = df
            except Exception:
                ohlcv_cache[t] = None
        return ohlcv_cache[t]

    funnel = defaultdict(int)
    records = []
    for right, r in opens:
        funnel["opens"] += 1
        und = flex.underlying(r["Symbol"])
        if und not in universe:
            funnel["drop_out_of_universe"] += 1
            continue
        entry = flex.iso8(r["DateTime"])
        exp = r["Expiry"]
        try:
            expiry = f"{exp[:4]}-{exp[4:6]}-{exp[6:8]}"
        except Exception:
            funnel["drop_bad_expiry"] += 1
            continue
        dte = flex.days_between(entry, expiry)
        if dte <= 0:
            funnel["drop_nonpos_dte"] += 1
            continue
        qty = abs(flex.fnum(r["Quantity"])) or 1
        dt8 = r["DateTime"][:8]
        ccy = r["CurrencyPrimary"]

        def usd(a, _ccy=ccy, _dt8=dt8):
            return a if _ccy == "USD" else a / usdcad(_dt8)

        premium_ps = usd(flex.fnum(r["Proceeds"])) / (100.0 * qty)
        strike = flex.fnum(r["Strike"])
        if premium_ps <= 0 or strike <= 0:
            funnel["drop_bad_premium"] += 1
            continue

        ohlcv = get_ohlcv(und)
        if ohlcv is None or "close" not in ohlcv.columns:
            funnel["drop_no_ohlcv"] += 1
            continue
        pit = ohlcv[ohlcv["date"] <= entry]
        if pit.empty:
            funnel["drop_no_pit_spot"] += 1
            continue
        spot = float(pit["close"].iloc[-1])
        # Scale-sanity gate: a wheel CSP is sold near-the-money (99% of this book
        # is strike/spot in [0.88, 1.04]). A wild ratio means the Bloomberg price
        # scale != the IBKR strike scale (e.g. Bloomberg's NFLX series is mis-
        # scaled ~10x: strike 1075 vs "spot" 110) — drop it rather than feed the
        # engine a garbage moneyness. Without this, the single corrupted trade
        # dominates the EV correlation and the EV-sign split (verified).
        if not (0.5 <= strike / spot <= 1.5):
            funnel["drop_moneyness_scale"] += 1
            continue
        # outcome: underlying close on expiry (nearest trading day <= expiry, within 4d)
        at_exp = ohlcv[ohlcv["date"] <= expiry]
        if at_exp.empty:
            funnel["drop_no_outcome"] += 1
            continue
        last_outcome_date = at_exp["date"].iloc[-1]
        if flex.days_between(expiry, last_outcome_date) < -4 and last_outcome_date < expiry:
            # expiry is beyond our data (e.g. > 2026-06-04) by >4 trading days
            funnel["drop_outcome_beyond_data"] += 1
            continue
        if last_outcome_date < expiry and flex.days_between(last_outcome_date, expiry) > 4:
            funnel["drop_outcome_beyond_data"] += 1
            continue
        s_expiry = float(at_exp["close"].iloc[-1])

        iv = _resolve_pit_atm_iv(conn, und, entry)
        if iv is None or not (0 < float(iv) <= 5.0):
            funnel["drop_no_iv"] += 1
            continue
        iv = float(iv)
        if iv > 3.0:  # stored as percent
            iv = iv / 100.0
        try:
            rf = float(get_current_risk_free_rate(entry, data_dir=str(_REPO / "data")))
        except Exception:
            rf = 0.05
        try:
            f = conn.get_fundamentals(und) or {}
            divy = float(f.get("dividend_yield", 0) or 0)
            if divy > 1:
                divy /= 100.0
        except Exception:
            divy = 0.0

        try:
            fwd, _m = best_available_forward_distribution(ohlcv, horizon_days=dte, as_of=entry)
            fwd = realized_vol_widened_log_returns(fwd, ohlcv, as_of=entry)
        except Exception:
            funnel["drop_no_fwd_dist"] += 1
            continue
        if fwd is None or len(fwd) < 30:
            funnel["drop_thin_fwd_dist"] += 1
            continue

        trade = ShortOptionTrade(
            option_type=("call" if right == "C" else "put"),
            underlying=und,
            spot=spot,
            strike=strike,
            premium=premium_ps,
            dte=dte,
            iv=iv,
            risk_free_rate=rf,
            dividend_yield=divy,
            contracts=1,
            regime_multiplier=1.0,
        )
        try:
            res = ev.evaluate(
                trade,
                forward_log_returns=fwd,
                trade_start=date.fromisoformat(entry),
                trade_end=date.fromisoformat(expiry),
            )
        except Exception:
            funnel["drop_evaluate_error"] += 1
            continue
        if res.event_lockout_reason:
            funnel["drop_event_lockout"] += 1
            continue

        # hold-to-expiry outcome — put: intrinsic when S<K, called; call: when S>K.
        if right == "P":
            intrinsic = max(0.0, strike - s_expiry)
            assigned = int(s_expiry < strike)
        else:
            intrinsic = max(0.0, s_expiry - strike)
            assigned = int(s_expiry > strike)
        sp_pnl = (premium_ps - intrinsic) * 100.0  # short-option leg P&L, per-contract $
        funnel["evaluated"] += 1
        funnel["evaluated_put" if right == "P" else "evaluated_call"] += 1
        records.append(
            {
                "leg": "put" if right == "P" else "call",
                "ticker": und,
                "entry": entry,
                "expiry": expiry,
                "dte": dte,
                "strike": strike,
                "spot": round(spot, 2),
                "premium_ps": round(premium_ps, 4),
                "iv": round(iv, 4),
                "s_expiry": round(s_expiry, 2),
                "pred_prob_profit": round(float(res.prob_profit), 4),
                "pred_prob_assignment": round(float(res.prob_assignment), 4),
                "pred_ev_raw": round(float(res.mean_pnl), 2),
                "realized_pnl": round(sp_pnl, 2),
                "win": int(sp_pnl > 0),
                "assigned": assigned,
            }
        )

    # ---- aggregate ----
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"funnel": dict(funnel), "n": len(records)}
    if records:
        from scipy.stats import spearmanr

        df = pd.DataFrame(records)

        def _split(d):
            return {
                "n": int(len(d)),
                "win_rate": round(float(d["win"].mean()), 4) if len(d) else None,
                "mean_realized": round(float(d["realized_pnl"].mean()), 2) if len(d) else None,
                "median_realized": round(float(d["realized_pnl"].median()), 2) if len(d) else None,
                "assignment_rate": round(float(d["assigned"].mean()), 4) if len(d) else None,
            }

        def _metrics(d):
            if not len(d):
                return {"n": 0}
            pp_rows, pp_brier, pp_ece = reliability(d["pred_prob_profit"], d["win"])
            pa_rows, pa_brier, pa_ece = reliability(d["pred_prob_assignment"], d["assigned"])
            ev_pear = (
                float(np.corrcoef(d["pred_ev_raw"], d["realized_pnl"])[0, 1])
                if len(d) > 2
                else float("nan")
            )
            ev_spear = (
                float(spearmanr(d["pred_ev_raw"], d["realized_pnl"]).correlation)
                if len(d) > 2
                else float("nan")
            )
            return {
                "n": int(len(d)),
                "observed_win_rate": round(float(d["win"].mean()), 4),
                "mean_pred_prob_profit": round(float(d["pred_prob_profit"].mean()), 4),
                "observed_assignment_rate": round(float(d["assigned"].mean()), 4),
                "mean_pred_prob_assignment": round(float(d["pred_prob_assignment"].mean()), 4),
                "prob_profit": {
                    "brier": round(pp_brier, 4),
                    "ece": round(pp_ece, 4),
                    "bins": pp_rows,
                },
                "prob_assignment": {
                    "brier": round(pa_brier, 4),
                    "ece": round(pa_ece, 4),
                    "bins": pa_rows,
                },
                "ev_raw_vs_realized": {
                    "pearson_r": round(ev_pear, 4),
                    "spearman_rho": round(ev_spear, 4),
                    "mean_pred_ev_raw": round(float(d["pred_ev_raw"].mean()), 2),
                    "mean_realized": round(float(d["realized_pnl"].mean()), 2),
                    "median_realized": round(float(d["realized_pnl"].median()), 2),
                    "note": "Pearson is outlier-sensitive; Spearman is the robust rank measure.",
                },
                "ev_sign_split": {
                    "pred_ev_positive": _split(d[d["pred_ev_raw"] > 0]),
                    "pred_ev_nonpositive": _split(d[d["pred_ev_raw"] <= 0]),
                },
            }

        result["all"] = _metrics(df)
        result["put"] = _metrics(df[df["leg"] == "put"])
        result["call"] = _metrics(df[df["leg"] == "call"])
        result["by_ticker"] = {t: int(c) for t, c in df["ticker"].value_counts().items()}
        (out / "ev_calibration_detail.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out / "ev_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ---- print ----
    print("=== PHASE 3 — EV calibration (PIT, strike-matched, hold-to-expiry) ===")
    print("funnel:", dict(funnel))
    if records:
        for leg in ("put", "call", "all"):
            m = result.get(leg, {})
            if not m.get("n"):
                continue
            print(
                f"\n[{leg.upper()}] n={m['n']}  obs_win={m['observed_win_rate']} vs pred {m['mean_pred_prob_profit']}"
                f"  obs_assign={m['observed_assignment_rate']} vs pred {m['mean_pred_prob_assignment']}"
                f"  | prob_profit Brier={m['prob_profit']['brier']} ECE={m['prob_profit']['ece']}"
            )
            evr = m["ev_raw_vs_realized"]
            print(
                f"   ev_raw: Pearson={evr['pearson_r']} Spearman={evr['spearman_rho']} "
                f"mean_pred={evr['mean_pred_ev_raw']} mean_real={evr['mean_realized']}"
            )
            sp = m["ev_sign_split"]
            print(
                f"   EV-sign: pred>0 n={sp['pred_ev_positive']['n']} win={sp['pred_ev_positive']['win_rate']} "
                f"mean_real={sp['pred_ev_positive']['mean_realized']} | "
                f"pred<=0 n={sp['pred_ev_nonpositive']['n']} win={sp['pred_ev_nonpositive']['win_rate']} "
                f"mean_real={sp['pred_ev_nonpositive']['mean_realized']}"
            )
            print("   prob_profit reliability (pred bin -> obs [95% CI], n):")
            for b in m["prob_profit"]["bins"]:
                if b["n"]:
                    print(
                        f"     {b['bin']} pred={b['mean_pred']:.3f} obs={b['obs']:.3f} "
                        f"[{b['ci_lo']:.3f},{b['ci_hi']:.3f}] n={b['n']}"
                    )
    print(f"\nwrote -> {out.resolve()}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--out", default="data_processed/ibkr")
    a = ap.parse_args()
    main(a.csv_a, a.csv_b, a.out)
