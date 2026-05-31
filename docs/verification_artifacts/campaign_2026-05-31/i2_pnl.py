"""I2 -- net-of-reality, capital-constrained, multi-regime wheel P&L vs passive.

The confirmed blind spot: no prior backtest combined REAL Theta bid/ask fills, a
capital-constrained NAV path, and a passive index benchmark. This does.

Design (respects the section-2 invariant):
  * DECISIONS come from the engine. We only enter puts the engine ranked with
    ev_dollars > 0 (from the monthly snapshots). No negative-EV entry.
  * BOOKKEEPING is independent so we can inject REAL fills the engine's internal
    WheelTracker (synthetic-premium) cannot: short puts filled at the real Theta
    quote (mid - slippage*half_spread; slippage=1.0 == sell at the bid), modeled
    10%-spread fallback only where Theta has no coverage (flagged + counted).
  * WHEEL: put assignment -> own 100 shares at strike -> sell a ~25-delta covered
    call each subsequent month (engine-ranked, real Theta call credit) until
    called away or the window ends.
  * Capital constraint: fixed account; one position per name; max_positions cap;
    a put reserves strike*100 collateral (cash-secured). Overlapping monthly
    cohorts build deployment until cash is exhausted -> realistic cash drag.

Outputs: terminal NAV, total return, win rate, # puts/assignments/calls, average
capital utilization, max drawdown, and the passive index proxy over the same
window (naive 100%-index and capital-matched). Friction sensitivity via --slippage.

Usage:
    python i2_pnl.py --regime bull_2021 --capital 1000000 --slippage 1.0
    python i2_pnl.py --regime all --slippage 1.0
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

SNAP_DIR = os.path.join(HERE, "snapshots")
OUT_DIR = os.path.join(HERE, "raw_output")
os.makedirs(OUT_DIR, exist_ok=True)
DTE = 35
COMMISSION = 0.65          # per contract per leg
ASSIGN_FEE = 5.0


@dataclass
class Position:
    ticker: str
    kind: str               # 'put' | 'stock' | 'call'
    strike: float
    entry_date: str
    expiry: pd.Timestamp
    credit: float           # per-share option credit (puts/calls); 0 for stock
    basis: float = 0.0      # stock basis per share
    contracts: int = 1
    real_fill: bool = True
    shares: int = 0         # for stock


@dataclass
class Sim:
    capital: float
    slippage: float
    max_positions: int = 25
    modeled_fallback: bool = True
    cash: float = 0.0
    realized: float = 0.0
    positions: list = field(default_factory=list)
    log: list = field(default_factory=list)
    nav_path: list = field(default_factory=list)
    n_put: int = 0
    n_assign: int = 0
    n_call: int = 0
    n_called_away: int = 0
    n_real_fill: int = 0
    n_modeled_fill: int = 0
    deploy_samples: list = field(default_factory=list)
    last_date: object = None
    rf_earned: float = 0.0
    friction_paid: float = 0.0      # within-path half-spread + commission cost ($)

    def __post_init__(self):
        self.cash = self.capital
        self.runner = WheelRunner()

    def accrue_rf(self, on_date):
        """Credit the risk-free rate on the liquid base (free cash + reserved
        cash-secured-put collateral) since the last accrual. Cash-secured puts
        hold their collateral in T-bills earning rf -- a real, material return
        component in a 4-5% world that premium-only accounting omits."""
        on = pd.Timestamp(on_date)
        if self.last_date is None:
            self.last_date = on
            return
        dt = (on - self.last_date).days
        if dt <= 0:
            return
        reserved = sum(p.strike * 100 for p in self.positions if p.kind == "put")
        liquid = max(0.0, self.cash) + reserved
        rf = L.risk_free_for(on)
        interest = liquid * rf * dt / 365.0
        self.cash += interest
        self.rf_earned += interest
        self.last_date = on

    # ---- fills ---------------------------------------------------------- #
    def put_fill(self, ticker, engine_strike, as_of, synth_prem):
        """Return (strike, credit_per_share, real_fill_bool, gross_mid) for a short put."""
        exp = L.theta_expiration_for(ticker, as_of, DTE)
        if exp is not None:
            q = L.theta_nearest_put(ticker, exp, engine_strike, as_of)
            if q and q.get("mid") is not None and q["mid"] > 0:
                half = max(0.0, q["mid"] - q["bid"]) if q.get("bid") is not None else 0.05 * q["mid"]
                credit = q["mid"] - self.slippage * half
                return float(q["strike"]), max(0.0, credit), True, q["mid"]
        if not self.modeled_fallback:
            return None
        half = 0.05 * synth_prem
        return float(engine_strike), max(0.0, synth_prem - self.slippage * half), False, synth_prem

    def call_fill(self, ticker, engine_strike, as_of, synth_prem):
        exp = L.theta_expiration_for(ticker, as_of, DTE)
        if exp is not None:
            fr = L._theta_expiration_frame(ticker, exp)
            if fr is not None and not fr.empty:
                want = pd.Timestamp(as_of).normalize()
                sub = fr[(fr["right"] == "call") & (fr["snap_date"] <= want)
                         & (fr["snap_date"] >= want - pd.Timedelta(days=7))]
                if not sub.empty:
                    last = sub["snap_date"].max()
                    sub = sub[sub["snap_date"] == last]
                    idx = (sub["strike"] - engine_strike).abs().idxmin()
                    q = L.theta_quote(ticker, exp, float(sub.loc[idx, "strike"]), "call", last)
                    if q and q.get("mid") is not None and q["mid"] > 0:
                        half = max(0.0, q["mid"] - q["bid"]) if q.get("bid") is not None else 0.05 * q["mid"]
                        return float(q["strike"]), max(0.0, q["mid"] - self.slippage * half), True, q["mid"]
        if not self.modeled_fallback:
            return None
        return float(engine_strike), max(0.0, synth_prem - self.slippage * 0.05 * synth_prem), False, synth_prem

    # ---- lifecycle ------------------------------------------------------ #
    def held_names(self):
        return {p.ticker for p in self.positions}

    def resolve_expiries(self, on_date):
        """Resolve any put/call legs whose expiry has passed. Credit was already
        received at entry; here we only release collateral / handle assignment /
        called-away. Stock legs persist until called away or window-end liquidation."""
        on = pd.Timestamp(on_date)
        survivors = []
        remove_stock_ids: set[int] = set()
        for p in self.positions:
            if p.kind in ("put", "call") and p.expiry <= on:
                rc = L.realized_close_at(p.ticker, p.entry_date,
                                         (p.expiry - pd.Timestamp(p.entry_date)).days)
                if rc is None:
                    survivors.append(p)               # data not available yet
                    continue
                if p.kind == "put":
                    if rc >= p.strike:                 # OTM expire -> release collateral
                        self.cash += p.strike * 100
                    else:                              # assigned -> collateral buys 100 shares
                        self.n_assign += 1
                        self.cash -= ASSIGN_FEE
                        survivors.append(Position(p.ticker, "stock", 0.0, p.entry_date, on,
                                                  0.0, basis=p.strike, shares=100,
                                                  real_fill=p.real_fill))
                else:  # call
                    stk = next((s for s in self.positions
                                if s.ticker == p.ticker and s.kind == "stock"), None)
                    if rc >= p.strike and stk is not None:   # called away: sell stock @ strike
                        self.n_called_away += 1
                        self.cash += p.strike * 100
                        remove_stock_ids.add(id(stk))
                    # else: call expires worthless, stock + credit already kept
            else:
                survivors.append(p)
        self.positions = [p for p in survivors if id(p) not in remove_stock_ids]

    def sell_calls_on_stock(self, on_date):
        for stk in [p for p in self.positions if p.kind == "stock"]:
            if any(c.ticker == stk.ticker and c.kind == "call" for c in self.positions):
                continue  # already has an open call
            try:
                df = self.runner.rank_covered_calls_by_ev(
                    stk.ticker, shares_held=100, as_of=on_date, target_dtes=(DTE,),
                    target_deltas=(0.25,), top_n=1, min_ev_dollars=-1e9,
                    use_event_gate=True, max_as_of_staleness_days=10_000)
            except Exception:  # noqa: BLE001
                df = None
            if df is None or not len(df):
                continue
            r = df.iloc[0]
            fill = self.call_fill(stk.ticker, float(r["strike"]), on_date, float(r["premium"]))
            if fill is None:
                continue
            strike, credit, real, gross = fill
            self.cash += credit * 100 - COMMISSION       # receive call premium at entry
            self.friction_paid += (gross - credit) * 100 + COMMISSION
            exp = pd.Timestamp(on_date) + pd.Timedelta(days=DTE)
            self.positions.append(Position(stk.ticker, "call", strike, on_date, exp,
                                            credit, real_fill=real))
            self.n_call += 1
            self.n_real_fill += int(real); self.n_modeled_fill += int(not real)

    def enter_puts(self, snap: pd.DataFrame, on_date):
        cands = snap[snap["ev_dollars"] > 0].sort_values("ev_dollars", ascending=False)
        held = self.held_names()
        for _, r in cands.iterrows():
            if len([p for p in self.positions if p.kind == "put"]) >= self.max_positions:
                break
            tk = r["ticker"]
            if tk in held:
                continue
            collat = float(r["strike"]) * 100
            if self.cash < collat:
                continue
            fill = self.put_fill(tk, float(r["strike"]), on_date, float(r["premium"]))
            if fill is None:
                continue
            strike, credit, real, gross = fill
            collat = strike * 100
            if self.cash < collat:
                continue
            self.cash -= collat                          # reserve cash-secured collateral
            self.cash += credit * 100 - COMMISSION       # receive premium at entry
            self.friction_paid += (gross - credit) * 100 + COMMISSION
            exp = pd.Timestamp(on_date) + pd.Timedelta(days=DTE)
            self.positions.append(Position(tk, "put", strike, on_date, exp, credit, real_fill=real))
            held.add(tk)
            self.n_put += 1
            self.n_real_fill += int(real); self.n_modeled_fill += int(not real)

    def mtm(self, on_date):
        self.accrue_rf(on_date)
        on = pd.Timestamp(on_date)
        liab = 0.0; stockval = 0.0; reserved = 0.0
        for p in self.positions:
            spot = L.close_series(p.ticker).asof(on)
            if pd.isna(spot):
                continue
            if p.kind == "put":
                reserved += p.strike * 100
                liab += max(0.0, p.strike - spot) * 100        # buyback liability (intrinsic)
            elif p.kind == "stock":
                stockval += spot * 100
            elif p.kind == "call":
                liab += max(0.0, spot - p.strike) * 100
        nav = self.cash + reserved + stockval - liab
        self.nav_path.append((str(on.date()), nav))
        self.deploy_samples.append((reserved + stockval) / (nav if nav > 0 else self.capital))
        return nav

    def liquidate(self, on_date):
        on = pd.Timestamp(on_date)
        for p in list(self.positions):
            spot = L.close_series(p.ticker).asof(on)
            if pd.isna(spot):
                spot = p.strike
            if p.kind == "put":
                self.cash += p.strike * 100 - max(0.0, p.strike - spot) * 100
                self.realized += -max(0.0, p.strike - spot) * 100
            elif p.kind == "stock":
                self.cash += spot * 100
                self.realized += (spot - p.basis) * 100
            elif p.kind == "call":
                self.cash -= max(0.0, spot - p.strike) * 100
                self.realized += -max(0.0, spot - p.strike) * 100
        self.positions = []


def run_regime(regime: str, capital: float, slippage: float, modeled_fallback: bool) -> dict:
    start, end = L.REGIMES[regime]
    snaps = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("put_") and f.endswith(".parquet")])
    dates = [f[4:-8] for f in snaps]
    dates = [d for d in dates if start <= d <= end]
    if not dates:
        return {"regime": regime, "error": "no snapshots in window"}
    sim = Sim(capital=capital, slippage=slippage, modeled_fallback=modeled_fallback)
    for d in dates:
        sim.resolve_expiries(d)
        sim.sell_calls_on_stock(d)
        snap = pd.read_parquet(os.path.join(SNAP_DIR, f"put_{d}.parquet"))
        sim.enter_puts(snap, d)
        sim.mtm(d)
    sim.resolve_expiries(end)
    sim.accrue_rf(end)                # final stub of T-bill interest on collateral
    sim.liquidate(end)
    final_nav = sim.cash
    ret = (final_nav - capital) / capital * 100

    navs = pd.Series([v for _, v in sim.nav_path])
    if len(navs) > 1:
        peak = navs.cummax()
        mdd = float(((navs - peak) / peak).min() * 100)
        rets = navs.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * np.sqrt(12)) if rets.std() > 0 else float("nan")
    else:
        mdd, sharpe = float("nan"), float("nan")
    avg_deploy = float(np.mean(sim.deploy_samples) * 100) if sim.deploy_samples else 0.0

    uni = list(L.universe_503())
    idx_cap = L.index_proxy_return(uni, start, end, "cap")
    idx_ew = L.index_proxy_return(uni, start, end, "ew")
    # Passive TOTAL return proxy = cap-weighted price return + dividends over the window.
    years = (pd.Timestamp(idx_cap["d1"]) - pd.Timestamp(idx_cap["d0"])).days / 365.0
    divy = L.cap_weighted_div_yield(tuple(uni))           # decimal
    idx_total = (idx_cap["return_pct"] or 0.0) + divy * 100.0 * years
    # capital-matched: passive TOTAL return scaled by the wheel's avg deployment;
    # the undeployed remainder already earns rf inside the wheel (credited above).
    matched_total = idx_total * avg_deploy / 100.0

    return {
        "regime": regime, "window": f"{start}..{end}", "capital": capital,
        "slippage": slippage, "modeled_fallback": modeled_fallback,
        "final_nav": round(final_nav, 0), "wheel_return_pct": round(ret, 2),
        "rf_earned_pct": round(sim.rf_earned / capital * 100, 2),
        "friction_paid_pct": round(sim.friction_paid / capital * 100, 3),
        "n_put": sim.n_put, "n_assign": sim.n_assign, "n_call": sim.n_call,
        "n_called_away": sim.n_called_away,
        "assign_rate": round(sim.n_assign / sim.n_put, 3) if sim.n_put else None,
        "real_fills": sim.n_real_fill, "modeled_fills": sim.n_modeled_fill,
        "avg_capital_deployed_pct": round(avg_deploy, 1),
        "max_drawdown_pct": round(mdd, 2), "sharpe_monthly_ann": round(sharpe, 2),
        "passive_index_price_pct": round(idx_cap["return_pct"], 2),
        "passive_index_ew_pct": round(idx_ew["return_pct"], 2),
        "passive_div_yield_pct_ann": round(divy * 100, 2),
        "passive_index_total_pct": round(idx_total, 2),
        "passive_capital_matched_total_pct": round(matched_total, 2),
        "wheel_minus_matched_pp": round(ret - matched_total, 2),
        "wheel_minus_index_total_pp": round(ret - idx_total, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="all")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--slippage", type=float, default=1.0)
    ap.add_argument("--no-modeled-fallback", action="store_true")
    a = ap.parse_args()
    regimes = list(L.REGIMES) if a.regime == "all" else [a.regime]
    results = []
    for rg in regimes:
        res = run_regime(rg, a.capital, a.slippage, not a.no_modeled_fallback)
        results.append(res)
        print(json.dumps(res, indent=2))
    tag = f"slip{a.slippage}_{'nomf' if a.no_modeled_fallback else 'mf'}"
    with open(os.path.join(OUT_DIR, f"i2_pnl_{a.regime}_{tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
