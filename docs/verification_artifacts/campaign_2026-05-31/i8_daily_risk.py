"""I8 -- DAILY mark-to-market risk for the wheel vs I2's MONTHLY marks.

The confirmed blind spot in I2 (docs/HEAVY_VERIFY_2026-05-31_I2_NET_OF_REALITY_PNL.md):
I2 marks NAV only at the monthly snapshot dates, so its reported max-drawdown
(2022 bear -6.6%, 2020 crash -2.6%) UNDERSTATES the true intra-month drawdown.
A sharp intra-month sell-off (assignment, gap-down) that recovers by month-end is
invisible to a monthly mark.

This driver REUSES I2's exact position bookkeeping (Sim / Position / the
resolve->sell_calls->enter lifecycle) but inserts a DAILY mark-to-market on every
trading day in the regime window. From the daily NAV series it computes the TRUE
intra-month max-drawdown, daily-return annualized Sharpe/Sortino, worst 1-day and
worst 5-day return, and the trough date -- then reports them side-by-side with I2's
monthly figures and a daily-marked passive index proxy over the same window.

OBSERVE-AND-DOCUMENT ONLY. Does not import or modify anything under engine/ beyond
what i2_pnl already imports (WheelRunner, read-only). engine/ is untouched.

Option-MTM approximation
------------------------
The daily book value buys back open options at INTRINSIC value only
  put  liability = max(0, strike - spot) * 100
  call liability = max(0, spot - strike) * 100
ignoring remaining time value. This is identical to I2's monthly mtm() so the
two paths agree exactly at the snapshot dates (apples-to-apples). Direction of
the bias: a short option's true buyback cost = intrinsic + time value > intrinsic,
so intrinsic-only UNDERSTATES the liability => the daily NAV is marked slightly
HIGH and the true daily drawdown is, if anything, marginally DEEPER than reported
here. The drawdown figures below are therefore conservative LOWER bounds on path
risk -- the same direction I2 already flagged, now quantified at daily resolution.

Usage:
    SWE_DATA_PROVIDER=bloomberg python i8_daily_risk.py
    SWE_DATA_PROVIDER=bloomberg python i8_daily_risk.py --regime bear_2022
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import campaign_lib as L  # noqa: E402
import i2_pnl as I2  # noqa: E402  -- REUSE Sim / Position / lifecycle / SNAP_DIR / DTE
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

OUT_DIR = os.path.join(HERE, "raw_output")
os.makedirs(OUT_DIR, exist_ok=True)

# I2's reported MONTHLY figures (from HEAVY_VERIFY_2026-05-31_I2_NET_OF_REALITY_PNL.md,
# slippage=1.0 / modeled-fallback table) -- for the side-by-side comparison.
I2_MONTHLY = {
    "crash_2020": {"max_dd_pct": -2.6, "sharpe_monthly_ann": None, "wheel_return_pct": 9.1},
    "bear_2022": {"max_dd_pct": -6.6, "sharpe_monthly_ann": None, "wheel_return_pct": 4.5},
    "recovery_2023_2024": {
        "max_dd_pct": -0.8,
        "sharpe_monthly_ann": None,
        "wheel_return_pct": 40.2,
    },
}

TRADING_DAYS = 252.0


# --------------------------------------------------------------------------- #
# Pure (non-mutating) daily valuation of the current book.
# --------------------------------------------------------------------------- #
def value_book(sim: I2.Sim, on_date, accrued_cash: float) -> float:
    """Value the open book on `on_date` WITHOUT mutating sim.

    Mirrors I2.Sim.mtm() exactly: nav = cash + reserved collateral + stock MTM
    - option buyback liability (intrinsic-only). `accrued_cash` is the cash
    balance with risk-free interest accrued up to `on_date` (computed by the
    caller so daily NAV reflects the same rf-on-collateral component I2 credits).
    Positions whose underlying has no close on `on_date` are skipped (same as I2).
    """
    on = pd.Timestamp(on_date)
    liab = 0.0
    stockval = 0.0
    reserved = 0.0
    for p in sim.positions:
        spot = L.close_series(p.ticker).asof(on)
        if pd.isna(spot):
            continue
        if p.kind == "put":
            reserved += p.strike * 100
            liab += max(0.0, p.strike - spot) * 100
        elif p.kind == "stock":
            stockval += spot * 100
        elif p.kind == "call":
            liab += max(0.0, spot - p.strike) * 100
    return accrued_cash + reserved + stockval - liab


def rf_accrued_cash(sim: I2.Sim, from_date, to_date) -> float:
    """Return what sim.cash WOULD be if rf accrued from `from_date` to `to_date`
    on the current liquid base (free cash + reserved put collateral), WITHOUT
    mutating sim. Uses the exact I2.accrue_rf formula (liquid * rf * days/365).
    The daily mark uses this so intra-month NAV reflects accruing T-bill interest;
    the real sim still accrues (and persists) rf at the next snapshot mtm()/end."""
    if from_date is None:
        # No snapshot processed yet -> nothing deployed, cash == capital, no accrual.
        return sim.cash
    a = pd.Timestamp(from_date)
    b = pd.Timestamp(to_date)
    dt = (b - a).days
    if dt <= 0:
        return sim.cash
    reserved = sum(p.strike * 100 for p in sim.positions if p.kind == "put")
    liquid = max(0.0, sim.cash) + reserved
    rf = L.risk_free_for(b)
    return sim.cash + liquid * rf * dt / 365.0


# --------------------------------------------------------------------------- #
# Daily-marked regime run -- reuses I2's monthly lifecycle verbatim.
# --------------------------------------------------------------------------- #
def run_regime_daily(regime: str, capital: float, slippage: float, modeled_fallback: bool) -> dict:
    start, end = L.REGIMES[regime]
    snaps = sorted(
        f for f in os.listdir(I2.SNAP_DIR) if f.startswith("put_") and f.endswith(".parquet")
    )
    snap_dates = [f[4:-8] for f in snaps]
    snap_dates = [d for d in snap_dates if start <= d <= end]
    if not snap_dates:
        return {"regime": regime, "error": "no snapshots in window"}

    panel = L.close_panel()
    trading_days = list(
        panel.index[(panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))]
    )

    sim = I2.Sim(capital=capital, slippage=slippage, modeled_fallback=modeled_fallback)

    daily_nav: list[tuple[str, float]] = []
    monthly_nav: list[tuple[str, float]] = []  # I2-equivalent marks (snapshot dates only)
    snap_ts = {pd.Timestamp(d) for d in snap_dates}
    snap_iter = iter(snap_dates)
    next_snap = pd.Timestamp(next(snap_iter))

    for td in trading_days:
        # Process every snapshot date that has arrived at/before this trading day,
        # in I2's exact order (resolve -> sell calls -> enter -> mtm). This keeps
        # the position book identical to I2 at the monthly marks.
        while next_snap is not None and next_snap <= td:
            d = str(next_snap.date())
            sim.resolve_expiries(d)
            sim.sell_calls_on_stock(d)
            snap_path = os.path.join(I2.SNAP_DIR, f"put_{d}.parquet")
            if os.path.exists(snap_path):
                snap = pd.read_parquet(snap_path)
                if "ev_dollars" in snap.columns and len(snap):
                    sim.enter_puts(snap, d)  # empty/pre-gate snapshots have no ev col
            nav_m = sim.mtm(d)  # I2's monthly mark: accrues + persists rf, records path
            monthly_nav.append((d, nav_m))
            try:
                nxt = next(snap_iter)
                next_snap = pd.Timestamp(nxt)
            except StopIteration:
                next_snap = None

        # DAILY mark for this trading day. If td is itself a snapshot date we just
        # recorded the monthly mark via mtm() above (sim.last_date == td, cash has
        # rf persisted) -- reuse that exact value so daily == monthly on snap days.
        if td in snap_ts and monthly_nav and monthly_nav[-1][0] == str(td.date()):
            daily_nav.append((str(td.date()), monthly_nav[-1][1]))
        else:
            cash_today = rf_accrued_cash(sim, sim.last_date, td)
            nav_d = value_book(sim, td, cash_today)
            daily_nav.append((str(td.date()), nav_d))

    # Final liquidation exactly as I2 (does not affect the daily path leading in).
    sim.resolve_expiries(end)
    sim.accrue_rf(end)
    sim.liquidate(end)
    final_nav = sim.cash
    ret = (final_nav - capital) / capital * 100

    # ---- daily risk metrics ---- #
    dnav = pd.Series([v for _, v in daily_nav], index=[d for d, _ in daily_nav])
    ddates = list(dnav.index)
    drisk = _risk_from_nav(dnav, ddates, TRADING_DAYS)

    # ---- monthly risk metrics (recomputed here from THIS run, to confirm we
    # reproduce I2's monthly-mark drawdown with the same machinery) ---- #
    mnav = pd.Series([v for _, v in monthly_nav], index=[d for d, _ in monthly_nav])
    mdates = list(mnav.index)
    mrisk = _risk_from_nav(mnav, mdates, 12.0)

    # ---- daily-marked passive index proxy over the same window ---- #
    idx_daily = _index_daily_nav(start, end, capital, weight="cap")
    idx_risk = _risk_from_nav(idx_daily["nav"], idx_daily["dates"], TRADING_DAYS)
    idx_ew_daily = _index_daily_nav(start, end, capital, weight="ew")
    idx_ew_risk = _risk_from_nav(idx_ew_daily["nav"], idx_ew_daily["dates"], TRADING_DAYS)

    i2_ref = I2_MONTHLY.get(regime, {})
    return {
        "regime": regime,
        "window": f"{start}..{end}",
        "capital": capital,
        "slippage": slippage,
        "modeled_fallback": modeled_fallback,
        "n_trading_days": len(trading_days),
        "n_snapshots": len(snap_dates),
        "final_nav": round(final_nav, 0),
        "wheel_return_pct": round(ret, 2),
        # TRUE daily-marked risk
        "daily_max_dd_pct": drisk["max_dd_pct"],
        "daily_max_dd_trough_date": drisk["trough_date"],
        "daily_max_dd_peak_date": drisk["peak_date"],
        "daily_sharpe_ann": drisk["sharpe_ann"],
        "daily_sortino_ann": drisk["sortino_ann"],
        "daily_worst_1d_pct": drisk["worst_1d_pct"],
        "daily_worst_5d_pct": drisk["worst_5d_pct"],
        "daily_vol_ann_pct": drisk["vol_ann_pct"],
        # monthly-marked risk recomputed from THIS run (reproduces I2)
        "monthly_max_dd_pct_reproduced": mrisk["max_dd_pct"],
        "monthly_max_dd_trough_date": mrisk["trough_date"],
        "monthly_sharpe_ann_reproduced": mrisk["sharpe_ann"],
        # I2's published monthly figures (from the findings doc)
        "i2_doc_monthly_max_dd_pct": i2_ref.get("max_dd_pct"),
        "i2_doc_wheel_return_pct": i2_ref.get("wheel_return_pct"),
        # how much the monthly mark understated the drawdown
        "dd_understatement_pp": (
            round(drisk["max_dd_pct"] - mrisk["max_dd_pct"], 2)
            if drisk["max_dd_pct"] is not None and mrisk["max_dd_pct"] is not None
            else None
        ),
        "dd_understatement_ratio": (
            round(drisk["max_dd_pct"] / mrisk["max_dd_pct"], 2)
            if mrisk["max_dd_pct"] not in (None, 0)
            else None
        ),
        # daily-marked passive index (apples-to-apples drawdown comparison)
        "index_cap_max_dd_pct": idx_risk["max_dd_pct"],
        "index_cap_max_dd_trough_date": idx_risk["trough_date"],
        "index_cap_return_pct": idx_daily["return_pct"],
        "index_cap_sharpe_ann": idx_risk["sharpe_ann"],
        "index_cap_worst_1d_pct": idx_risk["worst_1d_pct"],
        "index_ew_max_dd_pct": idx_ew_risk["max_dd_pct"],
        "index_ew_return_pct": idx_ew_daily["return_pct"],
        # the headline apples-to-apples ratio
        "wheel_dd_as_frac_of_index_dd": (
            round(drisk["max_dd_pct"] / idx_risk["max_dd_pct"], 3)
            if idx_risk["max_dd_pct"] not in (None, 0)
            else None
        ),
    }


def _risk_from_nav(nav: pd.Series, dates: list, periods_per_year: float) -> dict:
    """Drawdown + Sharpe/Sortino from a NAV series. Sharpe/Sortino are EXCESS of
    zero (raw return Sharpe -- matches I2's `rets.mean()/rets.std()` convention,
    which does NOT subtract rf because rf is already inside the wheel NAV)."""
    out = {
        "max_dd_pct": None,
        "trough_date": None,
        "peak_date": None,
        "sharpe_ann": None,
        "sortino_ann": None,
        "worst_1d_pct": None,
        "worst_5d_pct": None,
        "vol_ann_pct": None,
    }
    if nav is None or len(nav) < 2:
        return out
    # Drop any NaN marks (e.g. pre-deployment days with no positions are fine, but
    # a NaN would poison cummax/argmin) keeping date alignment.
    s = pd.Series(list(nav), index=list(dates)).dropna()
    if len(s) < 2:
        return out
    dates = list(s.index)
    vals = s.to_numpy(dtype=float)
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    i_trough = int(np.argmin(dd))
    out["max_dd_pct"] = round(float(dd[i_trough]) * 100.0, 2)
    out["trough_date"] = str(dates[i_trough])
    # peak that preceded the trough
    i_peak = int(np.argmax(vals[: i_trough + 1])) if i_trough > 0 else 0
    out["peak_date"] = str(dates[i_peak])

    rets = pd.Series(vals).pct_change().dropna()
    if len(rets) and rets.std(ddof=1) > 0:
        out["sharpe_ann"] = round(
            float(rets.mean() / rets.std(ddof=1) * np.sqrt(periods_per_year)), 2
        )
        downside = rets[rets < 0]
        dstd = float(np.sqrt((downside**2).mean())) if len(downside) else 0.0
        out["sortino_ann"] = (
            round(float(rets.mean() / dstd * np.sqrt(periods_per_year)), 2) if dstd > 0 else None
        )
        out["vol_ann_pct"] = round(float(rets.std(ddof=1) * np.sqrt(periods_per_year)) * 100, 2)
        out["worst_1d_pct"] = round(float(rets.min()) * 100, 2)
    # worst rolling 5-period return
    if len(vals) >= 6:
        roll = pd.Series(vals).pct_change(5).dropna()
        if len(roll):
            out["worst_5d_pct"] = round(float(roll.min()) * 100, 2)
    return out


def _index_daily_nav(start, end, capital: float, weight: str = "cap") -> dict:
    """Daily NAV of a buy-and-hold of the 503-name basket over (start..end].

    Cap weight = point-in-time (shares_out x price_at_first_day), mirroring
    campaign_lib.index_proxy_return so the endpoint return matches I2's benchmark.
    Equal weight = $capital/N per name at the first day. NAV(t) = sum_i units_i *
    price_i(t). Price-return only (dividends omitted -- same as the drawdown leg of
    I2's benchmark; div yield is a separate additive total-return term in I2 and is
    immaterial to DRAWDOWN shape)."""
    panel = L.close_panel()
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    days = list(panel.index[(panel.index >= s) & (panel.index <= e)])
    if not days:
        return {"nav": pd.Series(dtype=float), "dates": [], "return_pct": None}
    d0 = days[0]
    uni = list(L.universe_503())
    units = {}  # ticker -> share count held
    for t in uni:
        t = L._norm(t)
        if t not in panel.columns:
            continue
        p0 = panel.at[d0, t]
        if pd.isna(p0) or p0 <= 0:
            continue
        if weight == "cap":
            sh = L.shares_out(t)
            dollars = (sh * float(p0)) if sh is not None else None  # PIT cap weight (dollar)
        else:
            dollars = 1.0  # placeholder; normalized below
        if dollars is None or dollars <= 0:
            continue
        units[t] = (dollars, float(p0))
    if not units:
        return {"nav": pd.Series(dtype=float), "dates": [], "return_pct": None}
    # normalize dollar weights to `capital`, convert to share units at d0.
    wsum = sum(d for d, _ in units.values())
    share_units = {t: (capital * d / wsum) / p0 for t, (d, p0) in units.items()}

    navs = []
    dates = []
    held = list(share_units.keys())
    sub = panel[held].ffill()  # carry last close over untraded gaps
    for td in days:
        row = sub.loc[td]
        nav = float(sum(share_units[t] * row[t] for t in held if pd.notna(row[t])))
        navs.append(nav)
        dates.append(str(td.date()))
    nav_series = pd.Series(navs, index=dates)
    ret_pct = (navs[-1] / navs[0] - 1.0) * 100.0 if navs[0] else None
    return {"nav": nav_series, "dates": dates, "return_pct": round(ret_pct, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="risk")  # 'risk' = the two risk-critical + calm contrast
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--no-modeled-fallback", action="store_true")
    a = ap.parse_args()
    if a.regime == "risk":
        regimes = ["bear_2022", "crash_2020", "recovery_2023_2024"]
    elif a.regime == "all":
        regimes = list(L.REGIMES)
    else:
        regimes = [a.regime]

    print("=" * 78)
    print("I8 -- DAILY mark-to-market risk vs I2 MONTHLY marks")
    print(
        "provider:", os.environ.get("SWE_DATA_PROVIDER", "<unset>"), "| theta:", L.THETA_AVAILABLE
    )
    print("option-MTM: INTRINSIC-ONLY (same as I2) -> daily NAV marked HIGH,")
    print("            true daily drawdown is, if anything, marginally DEEPER.")
    print("=" * 78)

    results = []
    for rg in regimes:
        res = run_regime_daily(rg, a.capital, a.slippage, not a.no_modeled_fallback)
        results.append(res)
        print(json.dumps(res, indent=2))
        print("-" * 78)

    # compact side-by-side table
    print("\nSIDE-BY-SIDE  (max-drawdown: TRUE daily vs MONTHLY marks vs daily index)")
    print(
        f"{'regime':<20}{'daily_DD':>10}{'monthly_DD':>12}{'i2_doc_DD':>11}"
        f"{'under_pp':>10}{'idx_DD':>10}{'wheel/idx':>11}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['regime']:<20}  {r['error']}")
            continue
        print(
            f"{r['regime']:<20}"
            f"{_f(r['daily_max_dd_pct']):>10}"
            f"{_f(r['monthly_max_dd_pct_reproduced']):>12}"
            f"{_f(r['i2_doc_monthly_max_dd_pct']):>11}"
            f"{_f(r['dd_understatement_pp']):>10}"
            f"{_f(r['index_cap_max_dd_pct']):>10}"
            f"{_f(r['wheel_dd_as_frac_of_index_dd']):>11}"
        )
    print("\nSHARPE/SORTINO  (annualized; wheel daily vs monthly-reproduced vs index)")
    print(
        f"{'regime':<20}{'d_sharpe':>10}{'d_sortino':>11}{'m_sharpe':>10}"
        f"{'idx_sharpe':>12}{'worst_1d':>10}{'worst_5d':>10}"
    )
    for r in results:
        if "error" in r:
            continue
        print(
            f"{r['regime']:<20}"
            f"{_f(r['daily_sharpe_ann']):>10}"
            f"{_f(r['daily_sortino_ann']):>11}"
            f"{_f(r['monthly_sharpe_ann_reproduced']):>10}"
            f"{_f(r['index_cap_sharpe_ann']):>12}"
            f"{_f(r['daily_worst_1d_pct']):>10}"
            f"{_f(r['daily_worst_5d_pct']):>10}"
        )

    tag = f"slip{a.slippage}_{'nomf' if a.no_modeled_fallback else 'mf'}"
    with open(os.path.join(OUT_DIR, f"i8_daily_risk_{a.regime}_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.2f}" if isinstance(x, float) else str(x)


if __name__ == "__main__":
    raise SystemExit(main())
