"""Premium-correction pilot — observe-only.

Question (Refinement 1): if the engine's synthetic ``BSM(iv)`` short-put
premium were swapped for the **real Theta EOD mid**, holding ``fair =
BSM(iv)`` fixed, how large is the resulting ``edge_vs_fair`` — and on which
names? Because ``fair`` is held, this number is **identically the premium
correction** ``real_mid - BSM(iv)``. For OTM puts it is positive largely
because real OTM IV exceeds flat ATM IV — it is **skew-driven, not the
economic VRP**. It is reported here as *"how much the engine under-prices
the premium,"* NOT as "the VRP".

Question (Refinement 2 — the deliverable): do the strikes with the **largest
premium correction** (fattest skew / most market-priced fear) sit in
**benign empirical tails** as the engine's forward distribution sees them? If
yes, repricing the premium alone manufactures free-money ``edge_vs_fair`` on
exactly the scariest strikes — i.e. the honest wiring is *reprice-and-reshape*,
not *reprice-premium*. The cross-plot (correction vs the engine-vs-market
tail-probability gap) is the result.

This module NEVER mutates the decision-layer trio. It calls the authoritative
EV path (`WheelRunner.explore_ticker` → `rank_candidates_by_ev` →
`EVEngine.evaluate`) read-only and joins the real larder mid afterwards.

Run:
    python -m studies.premium_correction.pilot

Outputs land in ``studies/premium_correction/output/``.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from engine.data_integration import get_current_risk_free_rate
from engine.option_pricer import black_scholes_price
from engine.wheel_runner import WheelRunner

from .splits import (
    adjusted_to_raw_strike,
    cumulative_factor_after,
    raw_to_adjusted_price,
)

PILOT_TICKERS = ["TSLA", "NVDA", "AAPL"]
LARDER_ROOT = Path("data_processed/theta/option_history")
OUT_DIR = Path("studies/premium_correction/output")
DATA_DIR = "data/bloomberg"

# Post-split, engine-history-rich, larder-covered band. Monthly as_of grid.
AS_OF_START = date(2024, 10, 1)
AS_OF_END = date(2026, 3, 1)

# Join-quality tolerances. A candidate whose nearest listed contract is
# further than these from the engine's solved (strike, expiry) is recorded
# but flagged ``join_ok=False`` so the headline can be computed on clean
# joins only.
MAX_STRIKE_GAP_PCT = 0.04  # nearest listed strike within 4% of target
MAX_EXP_GAP_DAYS = 5  # nearest listed expiry within 5 days of target DTE
MAX_SPREAD_PCT = 1.50  # discard pathologically wide markets (>150% of mid)
AS_OF_BACKFILL_DAYS = 4  # tolerate weekend/holiday: use latest date <= as_of

# Statistical-rigor guard. Realized assignment frequency is a small-sample
# binomial proportion, so every realized point carries a Wilson score interval
# and any bin below MIN_BIN_N is FLAGGED as not-signal (drawn faded, never read
# as a finding). Mirrors the R11 "don't read a noisy point estimate as signal".
MIN_BIN_N = 30  # bins with fewer resolved contracts are flagged untrustworthy
# Pseudo-replication guard. Many (delta, dte, strike) contracts share a
# (ticker, as_of) and resolve against the SAME terminal price, so they are NOT
# independent. A bin must also clear MIN_CLUSTERS distinct (ticker, as_of)
# events, and CIs are cluster-bootstrapped (not naive Wilson) over those events.
MIN_CLUSTERS = 8  # distinct (ticker, as_of) resolution events required to trust a bin
N_BOOT = 2000  # cluster-bootstrap resamples


# --------------------------------------------------------------------------
# Larder access (raw strike space)
# --------------------------------------------------------------------------
def _larder_expirations(ticker: str) -> list[date]:
    d = LARDER_ROOT / f"ticker={ticker}"
    if not d.exists():
        return []
    out = []
    for p in d.glob("expiration=*"):
        try:
            out.append(pd.Timestamp(p.name.split("=", 1)[1]).date())
        except ValueError:
            continue
    return sorted(out)


def _larder_mid(ticker: str, real_exp: date, as_of: date, raw_target_strike: float) -> dict | None:
    """Nearest-strike PUT EOD mid (raw space) at ``as_of`` for ``real_exp``.

    Tolerates a few non-trading days by taking the latest available date
    ``<= as_of`` within ``AS_OF_BACKFILL_DAYS``.
    """
    part = LARDER_ROOT / f"ticker={ticker}" / f"expiration={real_exp:%Y%m%d}" / "data.parquet"
    if not part.exists():
        return None
    df = pd.read_parquet(
        part, columns=["strike", "right", "created", "bid", "ask", "open_interest"]
    )
    df = df[df["right"] == "PUT"].copy()
    if df.empty:
        return None
    df["d"] = pd.to_datetime(df["created"]).dt.date
    lo = as_of - timedelta(days=AS_OF_BACKFILL_DAYS)
    df = df[(df["d"] <= as_of) & (df["d"] >= lo)]
    if df.empty:
        return None
    use_date = max(df["d"])
    df = df[df["d"] == use_date].drop_duplicates(["strike", "right", "d"])
    df = df.assign(_gap=(df["strike"] - raw_target_strike).abs()).sort_values("_gap")
    row = df.iloc[0]
    bid, ask = float(row["bid"]), float(row["ask"])
    mid = (bid + ask) / 2.0
    return {
        "raw_strike": float(row["strike"]),
        "bid": bid,
        "ask": ask,
        "raw_mid": mid,
        "open_interest": float(row["open_interest"]),
        "spread_pct": (ask - bid) / mid if mid > 0 else np.inf,
        "quote_date": use_date,
    }


# --------------------------------------------------------------------------
# Pricing helpers
# --------------------------------------------------------------------------
def _backout_q(fair: float, S: float, K: float, T: float, r: float, iv: float) -> float:
    """Recover the dividend yield the engine used, so the re-priced fair at
    the listed contract is exactly consistent with the engine's own fair."""
    try:
        return float(
            brentq(
                lambda q: black_scholes_price(S, K, T, r, iv, "put", q) - fair,
                -0.5,
                0.5,
                xtol=1e-8,
            )
        )
    except (ValueError, RuntimeError):
        return 0.0


def _implied_vol_put(price: float, S: float, K: float, T: float, r: float, q: float) -> float:
    """Invert BSM for the put's implied vol (for the market-implied tail prob)."""
    if price <= 0 or T <= 0:
        return np.nan
    try:
        return float(
            brentq(
                lambda s: black_scholes_price(S, K, T, r, s, "put", q) - price,
                1e-3,
                5.0,
                xtol=1e-6,
            )
        )
    except (ValueError, RuntimeError):
        return np.nan


def _prob_itm_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Risk-neutral (Q-measure) P(S_T < K) = N(-d2) under the given vol.

    NOTE: this is risk-neutral. Differencing it against the engine's *physical*
    prob_assignment yields the **risk-premium wedge** (Q − P), which is > 0 for
    OTM equity puts *by construction* — it is NOT a calibration gap and cannot
    answer reprice-vs-reshape. The honest axis is physical-vs-physical:
    engine-predicted assignment prob vs the **realized** assignment frequency
    (`_terminal_close` below). See ``docs/PREMIUM_CORRECTION_PILOT.md`` §3.
    """
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(norm.cdf(-d2))


def _ohlcv_adjusted(wr: WheelRunner, ticker: str) -> pd.DataFrame | None:
    """Split-adjusted daily OHLCV (Bloomberg), date-indexed. Cached per call."""
    try:
        o = wr.connector.get_ohlcv(ticker)
    except Exception:  # noqa: BLE001
        return None
    if o is None or len(o) == 0:
        return None
    o = o.copy()
    o.index = pd.to_datetime(o.index)
    return o


def _terminal_close(ohlcv: pd.DataFrame, exp: date, tol_days: int = 4) -> float:
    """Adjusted close at/just before ``exp`` (nearest trading day <= exp)."""
    if ohlcv is None:
        return np.nan
    lo = pd.Timestamp(exp) - pd.Timedelta(days=tol_days)
    win = ohlcv[(ohlcv.index <= pd.Timestamp(exp)) & (ohlcv.index >= lo)]
    if win.empty:
        return np.nan
    return float(win.iloc[-1]["close"])


# --------------------------------------------------------------------------
# Pilot
# --------------------------------------------------------------------------
def _as_of_grid() -> list[str]:
    out, d = [], AS_OF_START
    while d <= AS_OF_END:
        out.append(d.isoformat())
        # first of next month
        d = (d.replace(day=28) + timedelta(days=7)).replace(day=1)
    return out


def run_pilot() -> pd.DataFrame:
    wr = WheelRunner(data_dir=DATA_DIR)
    larder_exps = {t: _larder_expirations(t) for t in PILOT_TICKERS}
    records: list[dict] = []

    for ticker in PILOT_TICKERS:
        exps = larder_exps[ticker]
        if not exps:
            print(f"[skip] {ticker}: no larder partitions yet")
            continue
        ohlcv = _ohlcv_adjusted(wr, ticker)
        bbg_last = ohlcv.index.max().date() if ohlcv is not None else date(1900, 1, 1)
        for as_of in _as_of_grid():
            aod = date.fromisoformat(as_of)
            # Split factor MUST be 1.0 in the pilot band — assert the join is
            # split-free here (the layer is tested against 2020 separately).
            assert cumulative_factor_after(ticker, aod) == 1.0, f"{ticker} {as_of} not post-split"
            r = get_current_risk_free_rate(as_of, data_dir=DATA_DIR)
            try:
                grid = wr.explore_ticker(ticker, as_of=as_of, include_diagnostic_fields=True)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] explore_ticker({ticker}, {as_of}) failed: {e!r}")
                continue
            if grid is None or len(grid) == 0:
                continue
            for _, c in grid.iterrows():
                spot, strike, iv = float(c["spot"]), float(c["strike"]), float(c["iv"])
                dte = int(c["dte"])
                fair_eng = float(c["fair_value"])
                T_eng = dte / 365.0
                q = _backout_q(fair_eng, spot, strike, T_eng, r, iv)

                target_exp = aod + timedelta(days=dte)
                # nearest listed expiration in the larder
                cand_exps = [e for e in exps if abs((e - target_exp).days) <= MAX_EXP_GAP_DAYS]
                if not cand_exps:
                    continue
                real_exp = min(cand_exps, key=lambda e: abs((e - target_exp).days))
                raw_target = adjusted_to_raw_strike(ticker, aod, strike)
                hit = _larder_mid(ticker, real_exp, aod, raw_target)
                if hit is None:
                    continue

                # Bring the raw listed contract into adjusted space (identity
                # in-band, but go through the layer for correctness).
                adj_listed_strike = hit["raw_strike"] / cumulative_factor_after(ticker, aod)
                adj_mid = raw_to_adjusted_price(ticker, aod, hit["raw_mid"])
                T_real = (real_exp - aod).days / 365.0

                # Recompute fair at the LISTED contract with the engine's own
                # (r, q, iv) — apples-to-apples with the real mid.
                fair_listed = black_scholes_price(
                    S=spot, K=adj_listed_strike, T=T_real, r=r, sigma=iv, option_type="put", q=q
                )
                # The headline quantity. With fair held at BSM(iv) this IS the
                # edge_vs_fair the real-premium wiring would produce.
                correction = adj_mid - fair_listed
                correction_pct = correction / fair_listed if fair_listed > 0 else np.nan

                # Engine's PHYSICAL predicted assignment probability.
                eng_prob_itm = float(c["prob_assignment"])

                # Refinement-2 DELIVERABLE axis (physical-vs-physical): did the
                # contract actually finish ITM? realized assignment vs the
                # engine's predicted prob is the honest calibration check — it
                # separates "engine genuinely under-sees risk" from "normal
                # risk premium". Resolved only when Bloomberg has the terminal
                # price (real_exp <= bbg_last); the last ~3 months are unresolved.
                resolved = real_exp <= bbg_last
                terminal_close = _terminal_close(ohlcv, real_exp) if resolved else np.nan
                realized_itm = (
                    float(terminal_close < adj_listed_strike)
                    if (resolved and np.isfinite(terminal_close))
                    else np.nan
                )

                # Risk-premium WEDGE (Q − P): kept for context, NOT a
                # calibration gap. mkt is risk-neutral (N(-d2) at contract iv),
                # eng is physical — their difference is the risk premium, > 0
                # for OTM puts by construction. Do NOT use as the deliverable.
                contract_iv = _implied_vol_put(adj_mid, spot, adj_listed_strike, T_real, r, q)
                mkt_prob_itm = _prob_itm_put(spot, adj_listed_strike, T_real, r, q, contract_iv)
                risk_premium_wedge = mkt_prob_itm - eng_prob_itm

                strike_gap_pct = abs(adj_listed_strike - strike) / strike
                exp_gap_days = abs((real_exp - target_exp).days)
                join_ok = (
                    strike_gap_pct <= MAX_STRIKE_GAP_PCT
                    and exp_gap_days <= MAX_EXP_GAP_DAYS
                    and hit["spread_pct"] <= MAX_SPREAD_PCT
                    and adj_mid > 0
                )

                records.append(
                    {
                        "ticker": ticker,
                        "as_of": as_of,
                        "delta_target": float(c["delta_target"]),
                        "dte": dte,
                        "real_exp": real_exp.isoformat(),
                        "quote_date": hit["quote_date"].isoformat(),
                        "spot": spot,
                        "iv_atm": iv,
                        "eng_strike": strike,
                        "listed_strike_adj": adj_listed_strike,
                        "raw_listed_strike": hit["raw_strike"],
                        "synthetic_bsm": fair_listed,
                        "real_mid_adj": adj_mid,
                        "correction": correction,
                        "correction_pct": correction_pct,
                        "correction_dollars_per_contract": correction * 100.0,
                        # DELIVERABLE risk axis (physical-vs-physical)
                        "eng_prob_itm": eng_prob_itm,
                        "realized_itm": realized_itm,
                        "resolved": resolved,
                        "terminal_close": terminal_close,
                        # risk-premium wedge (Q − P) — context only, NOT calibration
                        "contract_iv": contract_iv,
                        "mkt_prob_itm": mkt_prob_itm,
                        "risk_premium_wedge": risk_premium_wedge,
                        "cvar_5": float(c["cvar_5"]),
                        "tail_xi": float(c["tail_xi"]) if pd.notna(c["tail_xi"]) else np.nan,
                        "heavy_tail": bool(c["heavy_tail"]),
                        "ev_dollars_synth": float(c["ev_dollars"]),
                        "distribution_source": c.get("distribution_source", ""),
                        # join quality
                        "open_interest": hit["open_interest"],
                        "spread_pct": hit["spread_pct"],
                        "strike_gap_pct": strike_gap_pct,
                        "exp_gap_days": exp_gap_days,
                        "join_ok": join_ok,
                    }
                )
        print(f"[done] {ticker}: {sum(1 for x in records if x['ticker'] == ticker)} candidate rows")

    return pd.DataFrame(records)


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    if s.dropna().empty:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def _wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Robust for small n and p near 0/1 (where the normal approximation fails),
    which is exactly the realized-assignment regime here. Returns (lo, hi);
    (nan, nan) for n == 0.
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def _cluster_id(g: pd.DataFrame) -> pd.Series:
    """Independent-information unit. Many (delta, dte, strike) contracts at the
    same (ticker, as_of) resolve against the SAME terminal price path, so they
    are NOT independent trials — the cluster is (ticker, as_of). When those
    columns are absent (synthetic test data) each row is its own cluster.
    """
    if {"ticker", "as_of"}.issubset(g.columns):
        return g["ticker"].astype(str) + "|" + g["as_of"].astype(str)
    return pd.Series(range(len(g)), index=g.index).astype(str)


def _cluster_bootstrap(gb: pd.DataFrame, b: int = N_BOOT, seed: int = 12345):
    """Cluster (block) bootstrap CI for realized rate and gap: resample whole
    (ticker, as_of) clusters with replacement so within-cluster correlation is
    respected. Returns (realized_lo, realized_hi, gap_lo, gap_hi, n_clusters).
    A naive Wilson CI on the raw contract count would be far too narrow here.
    """
    cid = _cluster_id(gb)
    clusters = cid.unique()
    nc = len(clusters)
    # Per-cluster sufficient statistics (sum realized, count, sum predicted).
    grp = gb.assign(_c=cid.values).groupby("_c")
    cs_real = grp["realized_itm"].sum().to_numpy()
    cs_n = grp["realized_itm"].size().to_numpy().astype(float)
    cs_pred = grp["eng_prob_itm"].sum().to_numpy()
    if nc < 2:
        return (np.nan, np.nan, np.nan, np.nan, nc)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, nc, size=(b, nc))
    n_tot = cs_n[picks].sum(axis=1)
    realized_s = cs_real[picks].sum(axis=1) / n_tot
    gap_s = realized_s - cs_pred[picks].sum(axis=1) / n_tot
    return (
        float(np.percentile(realized_s, 2.5)),
        float(np.percentile(realized_s, 97.5)),
        float(np.percentile(gap_s, 2.5)),
        float(np.percentile(gap_s, 97.5)),
        nc,
    )


def _binned(clean: pd.DataFrame, by: str, q: int = 6) -> pd.DataFrame:
    """Quantile-bin ``clean`` by ``by``; per bin report predicted vs realized
    assignment frequency with BOTH a naive Wilson CI (independence-assuming)
    and a **cluster-robust** bootstrap CI over (ticker, as_of), plus
    ``n_clusters``. A bin is ``trustworthy`` only when it has >= MIN_BIN_N
    contracts AND >= MIN_CLUSTERS independent (ticker, as_of) events — because
    many contracts per event are pseudo-replicates, not independent trials.
    """
    g = clean.dropna(subset=[by, "realized_itm", "eng_prob_itm"]).copy()
    if len(g) < 2 * MIN_BIN_N:
        return pd.DataFrame()  # too few resolved contracts to bin honestly
    q = max(2, min(q, len(g) // MIN_BIN_N))
    g["_bin"] = pd.qcut(g[by], q=q, duplicates="drop")
    rows = []
    for b, gb in g.groupby("_bin", observed=True):
        n = len(gb)
        k = float(gb["realized_itm"].sum())
        realized = k / n
        predicted = gb["eng_prob_itm"].mean()
        wlo, whi = _wilson_ci(k, n)
        rlo, rhi, glo, ghi, nc = _cluster_bootstrap(gb)
        rows.append(
            {
                "bin": str(b),
                "n": n,
                "n_clusters": nc,
                f"{by}_mid": gb[by].mean(),
                "predicted": predicted,
                "realized": realized,
                # cluster-robust CI is the reported (honest) interval
                "realized_lo": rlo,
                "realized_hi": rhi,
                "gap_realized_minus_pred": realized - predicted,
                "gap_lo": glo,
                "gap_hi": ghi,
                # naive Wilson kept as the (overconfident) independence reference
                "realized_lo_wilson": wlo,
                "realized_hi_wilson": whi,
                "trustworthy": (n >= MIN_BIN_N) and (nc >= MIN_CLUSTERS),
            }
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> str:
    clean = df[df["join_ok"]].copy()
    res = clean[clean["resolved"] & clean["realized_itm"].notna()]
    lines = ["# Premium-correction pilot — summary", ""]
    lines.append(
        f"Total rows: {len(df)}  |  clean joins: {len(clean)}  |  "
        f"resolved (terminal price known): {len(res)}"
    )
    lines.append("")
    lines.append("HEADLINE (Refinement 1) = premium under-pricing (real_mid − BSM(iv));")
    lines.append("skew-driven, NOT VRP.")
    lines.append("")
    if clean.empty:
        lines.append("No clean joins — larder coverage in the band is still landing.")
        return "\n".join(lines)
    for ticker, g in clean.groupby("ticker"):
        lines.append(f"## {ticker}  (n={len(g)})")
        lines.append(
            f"  correction $/contract : median {g.correction_dollars_per_contract.median():+.1f}  "
            f"IQR [{g.correction_dollars_per_contract.quantile(0.25):+.1f}, "
            f"{g.correction_dollars_per_contract.quantile(0.75):+.1f}]"
        )
        lines.append(
            f"  correction % of BSM   : median {100 * g.correction_pct.median():+.1f}%  "
            f"IQR [{100 * g.correction_pct.quantile(0.25):+.1f}%, "
            f"{100 * g.correction_pct.quantile(0.75):+.1f}%]"
        )
        lines.append("")

    # Refinement 2 — the deliverable: physical-vs-physical calibration.
    lines.append("## Refinement 2 — engine predicted vs REALIZED assignment (physical-vs-physical)")
    if len(res) < 2 * MIN_BIN_N:
        lo, hi = (
            _wilson_ci(float(res["realized_itm"].sum()), len(res)) if len(res) else (np.nan, np.nan)
        )
        lines.append(
            f"  Only {len(res)} resolved contracts (< {2 * MIN_BIN_N}) — TOO FEW to bin honestly. "
            "Preliminary; do NOT read as a finding."
        )
        if len(res):
            lines.append(
                f"  overall realized {res['realized_itm'].mean():.3f} "
                f"[Wilson95 {lo:.3f}, {hi:.3f}, n={len(res)}] vs predicted "
                f"{res['eng_prob_itm'].mean():.3f}"
            )
        lines.append("")
    else:
        n_all = len(res)
        n_clust_all = res.pipe(_cluster_id).nunique()
        overall_pred = res["eng_prob_itm"].mean()
        overall_real = res["realized_itm"].mean()
        lines.append(
            f"  overall: predicted {overall_pred:.3f} vs realized {overall_real:.3f} "
            f"(gap {overall_real - overall_pred:+.3f}) — but n={n_all} contracts come from only "
            f"n_clusters={n_clust_all} independent (ticker, as_of) events: NOT 155 trials."
        )
        # Per-name decomposition — a single-name directional run masquerades as
        # a cross-sectional finding when names are pooled.
        lines.append("  per-name gap (realized − predicted):")
        for tk, gtk in res.groupby("ticker"):
            lines.append(
                f"    {tk}: pred {gtk.eng_prob_itm.mean():.3f}  real {gtk.realized_itm.mean():.3f}  "
                f"gap {gtk.realized_itm.mean() - gtk.eng_prob_itm.mean():+.3f}  "
                f"(n={len(gtk)}, clusters={gtk.pipe(_cluster_id).nunique()})"
            )
        mc = _binned(res, "correction_pct")
        if not mc.empty:
            lines.append("")
            lines.append("  miscalibration vs premium correction (cluster-robust CI):")
            lines.append(
                "    correction%    n  clus  predicted  realized [cluster95]    gap[cluster95]      flag"
            )
            for _, row in mc.iterrows():
                flag = "" if row["trustworthy"] else "  <not signal: <8 clusters or <30 n>"
                lines.append(
                    f"    {100 * row['correction_pct_mid']:+7.1f}%  {int(row['n']):>4} {int(row['n_clusters']):>4}   "
                    f"{row['predicted']:.3f}     {row['realized']:.3f} "
                    f"[{row['realized_lo']:.3f},{row['realized_hi']:.3f}]  "
                    f"{row['gap_realized_minus_pred']:+.3f}[{row['gap_lo']:+.3f},{row['gap_hi']:+.3f}]{flag}"
                )
            lines.append("")
            trust = mc[mc["trustworthy"]]
            if len(trust) >= 2:
                hi_bin = trust.iloc[trust["correction_pct_mid"].argmax()]
                lo_bin = trust.iloc[trust["correction_pct_mid"].argmin()]
                hi_clears = hi_bin["gap_lo"] > 0
                lines.append(
                    f"    READ (trustworthy bins, n>={MIN_BIN_N} AND clusters>={MIN_CLUSTERS}): "
                    + (
                        f"high-corr gap {hi_bin['gap_realized_minus_pred']:+.3f} "
                        f"[{hi_bin['gap_lo']:+.3f},{hi_bin['gap_hi']:+.3f}] cluster-robust CI clears 0 "
                        "AND exceeds low-corr ⇒ engine under-sees realized risk where premium is fat "
                        "⇒ reprice-AND-reshape."
                        if (
                            hi_clears
                            and hi_bin["gap_realized_minus_pred"]
                            > lo_bin["gap_realized_minus_pred"]
                        )
                        else "high-corr gap cluster-robust CI does NOT clear 0 ⇒ NO calibration failure "
                        "established; the apparent signal is within cluster-robust noise."
                    )
                )
            else:
                lines.append(
                    f"    Fewer than 2 TRUSTWORTHY bins (need n>={MIN_BIN_N} AND clusters>={MIN_CLUSTERS}) "
                    "— with 3 names / ~10 clusters this is NOT a finding. Needs ~15 names (the "
                    "user's own prior) so direction averages out and clusters multiply."
                )
        else:
            lines.append("  Not enough resolved contracts to bin at the trust floor yet.")
        lines.append("")
        lines.append(
            "  NOTE: `risk_premium_wedge` (Q − P) is in the records for context only — it is the "
            "risk premium, NOT a calibration gap, and is not used as a deliverable axis."
        )
        lines.append(
            "  CAVEAT: calm 2024–25 band; the crisis-onset under-seeing case needs a 2020/2022 "
            "stress window (deferred to the full study)."
        )
    return "\n".join(lines)


def make_plots(df: pd.DataFrame, outdir: Path) -> bool:
    """Render the reliability + deliverable cross-plot PNG. Returns False (no
    crash) when matplotlib is absent or there is nothing resolved to plot."""
    clean = df[df["join_ok"]].copy()
    res = clean[clean["resolved"] & clean["realized_itm"].notna()].copy()
    if res.empty:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Cluster-robust CI error bars; bins with too few independent (ticker,
    # as_of) clusters drawn faded/open so a pseudo-replicated bin is never read
    # as signal (the R11 noisy-/correlated-estimate guard).
    def _draw(ax, x, row, y, ylo, yhi, color):
        trust = bool(row["trustworthy"])
        ax.errorbar(
            x,
            y,
            yerr=[[y - ylo], [yhi - y]],
            fmt="o",
            mfc=color if trust else "white",
            mec=color,
            ecolor=color,
            alpha=0.9 if trust else 0.45,
            capsize=3,
            ms=7 if trust else 6,
            mew=1.3,
        )
        ax.annotate(
            f"n={int(row['n'])}/c{int(row['n_clusters'])}" + ("" if trust else " low"),
            (x, y),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
            color="black" if trust else "gray",
        )

    # (1) Reliability curve: engine predicted vs realized assignment frequency.
    rel = _binned(res, "eng_prob_itm")
    if not rel.empty:
        top = max(rel["predicted"].max(), rel["realized_hi"].max()) * 1.1
        ax1.plot([0, top], [0, top], "k--", lw=0.8, label="perfect calibration")
        for _, row in rel.iterrows():
            _draw(
                ax1,
                row["predicted"],
                row,
                row["realized"],
                row["realized_lo"],
                row["realized_hi"],
                "#0077bb",
            )
    ax1.set_xlabel("engine predicted P(assignment)  [physical]")
    ax1.set_ylabel("realized assignment frequency  (cluster-robust 95% CI)")
    ax1.set_title(
        "Reliability: predicted vs realized  (n contracts / c clusters)\n"
        "(above line ⇒ engine under-sees risk; faded = <8 clusters, not signal)"
    )
    ax1.legend()

    # (2) The deliverable: miscalibration (realized − predicted) vs correction.
    # Quantile binning (qcut in _binned) is already robust to the near-zero-fair
    # blow-ups in correction_pct; winsorize only the x display for outliers.
    mc = _binned(res, "correction_pct")
    if not mc.empty:
        mc = mc.assign(_x=100 * _winsorize(mc["correction_pct_mid"]))
        ax2.axhline(0, color="k", lw=0.7, ls="--")
        for _, row in mc.iterrows():
            _draw(
                ax2,
                row["_x"],
                row,
                row["gap_realized_minus_pred"],
                row["gap_lo"],
                row["gap_hi"],
                "#cc3311",
            )
    ax2.set_xlabel("premium under-pricing  (real_mid − BSM(iv), % of BSM)")
    ax2.set_ylabel("realized − predicted assignment freq")
    ax2.set_title(
        "Refinement 2 deliverable: does the engine under-see realized\n"
        "risk where premium is fat?  (rising & >0 ⇒ reprice-and-reshape)"
    )
    fig.tight_layout()
    fig.savefig(outdir / "refinement2_calibration.png", dpi=120)
    plt.close(fig)
    return True


def main() -> int:
    warnings.filterwarnings("ignore")
    # The summary uses typographic minus/arrows; force utf-8 stdout so a
    # cp1252 Windows console doesn't crash on print (the .md is already utf-8).
    try:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_pilot()
    if df.empty:
        print("No candidate rows produced (larder band coverage not ready).")
        (OUT_DIR / "pilot_summary.md").write_text(
            "No candidate rows produced — larder band coverage not ready.\n", encoding="utf-8"
        )
        return 0
    df.to_csv(OUT_DIR / "pilot_records.csv", index=False)
    summary = summarize(df)
    (OUT_DIR / "pilot_summary.md").write_text(summary, encoding="utf-8")

    # Binned cross-plot DATA is written regardless of matplotlib — it is the
    # data of record (with Wilson CIs + n + trustworthy flags); the PNG is just
    # the visual. So the deliverable survives even where matplotlib is absent.
    res = df[df["join_ok"] & df["resolved"] & df["realized_itm"].notna()]
    rel = _binned(res, "eng_prob_itm")
    mc = _binned(res, "correction_pct")
    if not rel.empty:
        rel.to_csv(OUT_DIR / "reliability_bins.csv", index=False)
    if not mc.empty:
        mc.to_csv(OUT_DIR / "crossplot_bins.csv", index=False)
    plotted = make_plots(df, OUT_DIR)

    print("\n" + summary)
    extra = "refinement2_calibration.png, " if plotted else "(matplotlib absent — PNG skipped) "
    print(
        f"\nWrote: {OUT_DIR}/ pilot_records.csv, pilot_summary.md, "
        f"reliability_bins.csv, crossplot_bins.csv, {extra}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
