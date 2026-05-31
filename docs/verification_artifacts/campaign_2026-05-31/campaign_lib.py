"""Shared library for the 2026-05-31 heavy-verification campaign.

READ-ONLY against engine + data. This module is the single source of truth for
every campaign driver (I1..I5) so methodology is identical across investigations.

Contents
--------
* Path resolution (portable from __file__; Theta lives in the primary clone).
* OHLCV access with the documented rotated-column rename + a fail-loud invariant
  (true close == raw `high`; see engine/data_connector.py:186-240). Cross-checked
  against the connector in the self-test.
* Fundamentals (market cap, sector, dividend yield, beta).
* The 503-name universe.
* Theta real option quotes (bid/ask/close/OI by ticker/expiration/strike/right/date).
* Forward-outcome realizer for a short put (OTM-expire vs engine-EXACT attribution;
  synthetic-premium and real-premium realized P&L).
* Regime window definitions + a cap-weighted / equal-weight passive index proxy.

Run the self-test:
    python docs/verification_artifacts/campaign_2026-05-31/campaign_lib.py
"""
from __future__ import annotations

import os
import sys
import glob
import functools
from datetime import date, datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)  # load-bearing: avoid user-site .pth shadowing engine.*

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA = os.path.join(ROOT, "data", "bloomberg")
OHLCV_CSV = os.path.join(DATA, "sp500_ohlcv.csv")
IV_CSV = os.path.join(DATA, "sp500_vol_iv_full.csv")
FUND_CSV = os.path.join(DATA, "sp500_fundamentals_yf.csv")

_THETA_CANDIDATES = [
    os.path.join(ROOT, "data_processed", "theta", "option_history"),
    r"C:\Users\merty\Desktop\smart-wheel-engine\data_processed\theta\option_history",
]
THETA_ROOT = next((p for p in _THETA_CANDIDATES if os.path.isdir(p)), _THETA_CANDIDATES[0])
THETA_AVAILABLE = os.path.isdir(THETA_ROOT)


# --------------------------------------------------------------------------- #
# OHLCV  (the rotated-column rename is load-bearing)
# --------------------------------------------------------------------------- #
def _norm(t: str) -> str:
    """'AAPL UN Equity' -> 'AAPL'."""
    return str(t).split(" ")[0].strip().upper()


@functools.lru_cache(maxsize=1)
def _ohlcv_renamed() -> pd.DataFrame:
    """Full OHLCV with TRUE column labels and a fail-loud invariant check.

    Raw CSV columns are rotated: raw open=HIGH, raw high=CLOSE, raw close=OPEN.
    We rename to true labels then assert ``high >= max(open,close,low)`` and
    ``low <= min(...)`` on every populated row. If the CSV is ever regenerated
    in canonical order this raises instead of silently inverting prices.
    """
    raw = pd.read_csv(OHLCV_CSV)
    df = raw.rename(columns={"open": "high", "high": "close", "close": "open", "low": "low"})
    df["ticker"] = df["ticker"].map(_norm)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    ok = df.dropna(subset=["open", "high", "low", "close"])
    bad_high = int((ok["high"] < ok[["open", "close", "low"]].max(axis=1)).sum())
    bad_low = int((ok["low"] > ok[["open", "close", "high"]].min(axis=1)).sum())
    if bad_high or bad_low:
        raise RuntimeError(
            f"OHLCV rename invariant FAILED (bad_high={bad_high}, bad_low={bad_low}). "
            "The rotated-column assumption has drifted -- realized prices would be wrong."
        )
    return df


@functools.lru_cache(maxsize=1)
def close_panel() -> pd.DataFrame:
    """Wide close-price panel: index=date, columns=ticker. NaN where untraded."""
    df = _ohlcv_renamed()
    return df.pivot_table(index="date", columns="ticker", values="close")


@functools.lru_cache(maxsize=2048)
def close_series(ticker: str) -> pd.Series:
    df = _ohlcv_renamed()
    s = df[df["ticker"] == _norm(ticker)].set_index("date")["close"].sort_index()
    return s


def realized_close_at(ticker: str, as_of, horizon_days: int) -> float | None:
    """Close on the first trading day at or after as_of + horizon_days (calendar)."""
    s = close_series(ticker)
    if s.empty:
        return None
    target = pd.Timestamp(as_of) + pd.Timedelta(days=horizon_days)
    fwd = s[s.index >= target]
    if fwd.empty:
        return None
    return float(fwd.iloc[0])


def path_between(ticker: str, start, end) -> pd.Series:
    """Close path on (start, end] inclusive -- for mark-to-market / max-drawdown."""
    s = close_series(ticker)
    return s[(s.index > pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# --------------------------------------------------------------------------- #
# Fundamentals + universe
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def fundamentals() -> pd.DataFrame:
    f = pd.read_csv(FUND_CSV)
    f["ticker"] = f["ticker"].map(_norm)
    return f.set_index("ticker")


@functools.lru_cache(maxsize=1)
def universe_503() -> tuple[str, ...]:
    df = _ohlcv_renamed()
    return tuple(sorted(df["ticker"].unique()))


TREASURY_CSV = os.path.join(DATA, "treasury_yields.csv")


@functools.lru_cache(maxsize=1)
def _treasury() -> pd.Series:
    t = pd.read_csv(TREASURY_CSV)
    t["date"] = pd.to_datetime(t["date"])
    return (t.set_index("date")["rate_3m"].sort_index() / 100.0)  # percent -> decimal


def risk_free_for(d) -> float:
    """PIT 3-month T-bill rate (decimal). Falls back to 0.003 before the file
    starts (2021-05; 2020 short rates were near zero -> immaterial credit)."""
    s = _treasury()
    v = s.asof(pd.Timestamp(d))
    return float(v) if pd.notna(v) else 0.003


@functools.lru_cache(maxsize=1)
def cap_weighted_div_yield(tickers: tuple[str, ...]) -> float:
    """Current cap-weighted dividend yield (decimal) of a basket, for the
    passive total-return proxy. eqy_dvd_yld_12m is in percent."""
    f = fundamentals()
    num = den = 0.0
    for t in tickers:
        t = _norm(t)
        if t in f.index:
            dy = f.loc[t, "eqy_dvd_yld_12m"]
            cap = market_cap_of(t)
            if pd.notna(dy) and cap:
                num += float(dy) * cap
                den += cap
    return (num / den / 100.0) if den else 0.0


def sector_of(ticker: str) -> str:
    f = fundamentals()
    t = _norm(ticker)
    if t in f.index and "gics_sector_name" in f.columns:
        v = f.loc[t, "gics_sector_name"]
        return str(v) if pd.notna(v) else "Unknown"
    return "Unknown"


def market_cap_of(ticker: str) -> float | None:
    f = fundamentals()
    t = _norm(ticker)
    if t in f.index and "cur_mkt_cap" in f.columns:
        v = f.loc[t, "cur_mkt_cap"]
        return float(v) if pd.notna(v) else None
    return None


@functools.lru_cache(maxsize=2048)
def shares_out(ticker: str) -> float | None:
    """Approx shares outstanding = current market cap / latest close.

    Used to build POINT-IN-TIME cap weights (shares x price_at_window_start),
    which removes the look-ahead bias of weighting a historical return by today's
    cap (today's cap over-weights names that have since grown, e.g. NVDA). Assumes
    shares roughly constant over the window -- a small second-order error vs the
    large price-growth look-ahead it removes (buyback-heavy names slightly
    under-weighted in the past). Documented caveat, not a hidden assumption.
    """
    cap = market_cap_of(ticker)
    s = close_series(ticker)
    if cap is None or s.empty:
        return None
    last_px = float(s.iloc[-1])
    return cap / last_px if last_px > 0 else None


# --------------------------------------------------------------------------- #
# Theta real option quotes
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def _theta_expiration_frame(ticker: str, expiration: str) -> pd.DataFrame | None:
    """All rows for one (ticker, expiration): every strike x right x snapshot date."""
    p = os.path.join(THETA_ROOT, f"ticker={_norm(ticker)}", f"expiration={expiration}", "data.parquet")
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    # 'created' is the snapshot timestamp; reduce to a date for joins.
    df["snap_date"] = pd.to_datetime(df["created"]).dt.normalize()
    return df


@functools.lru_cache(maxsize=2048)
def theta_expirations(ticker: str) -> tuple[str, ...]:
    d = os.path.join(THETA_ROOT, f"ticker={_norm(ticker)}")
    if not os.path.isdir(d):
        return ()
    exps = [x.split("=", 1)[1] for x in os.listdir(d) if x.startswith("expiration=")]
    return tuple(sorted(exps))


def theta_quote(ticker: str, expiration: str, strike: float, right: str,
                on_date) -> dict | None:
    """Real {bid, ask, close, mid, open_interest, strike} for a contract on a date.

    `right` in {'put','call'}. Picks the snapshot on `on_date` (or the last one
    at/before it within 5 trading sessions). Returns None if no match.
    """
    fr = _theta_expiration_frame(ticker, expiration)
    if fr is None or fr.empty:
        return None
    want = pd.Timestamp(on_date).normalize()
    sub = fr[(fr["right"] == right) & (np.isclose(fr["strike"], strike))]
    if sub.empty:
        return None
    on = sub[sub["snap_date"] == want]
    if on.empty:
        prior = sub[(sub["snap_date"] <= want) & (sub["snap_date"] >= want - pd.Timedelta(days=7))]
        if prior.empty:
            return None
        on = prior.sort_values("snap_date").tail(1)
    r = on.iloc[-1]
    bid = float(r["bid"]) if pd.notna(r["bid"]) else None
    ask = float(r["ask"]) if pd.notna(r["ask"]) else None
    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None and ask >= bid >= 0) else None
    # open_interest only present in the 2023-2026 pull (include_oi=True); guard it.
    oi = None
    if "open_interest" in r.index and pd.notna(r["open_interest"]):
        oi = float(r["open_interest"])
    return {
        "bid": bid, "ask": ask, "mid": mid,
        "close": float(r["close"]) if pd.notna(r["close"]) else None,
        "open_interest": oi,
        "strike": float(r["strike"]), "snap_date": r["snap_date"].date().isoformat(),
    }


def theta_nearest_put(ticker: str, expiration: str, target_strike: float,
                      on_date) -> dict | None:
    """Nearest available put strike to target on a date, with its real quote."""
    fr = _theta_expiration_frame(ticker, expiration)
    if fr is None or fr.empty:
        return None
    want = pd.Timestamp(on_date).normalize()
    sub = fr[(fr["right"] == "put") & (fr["snap_date"] == want)]
    if sub.empty:
        sub = fr[(fr["right"] == "put") & (fr["snap_date"] <= want)
                 & (fr["snap_date"] >= want - pd.Timedelta(days=7))]
        if sub.empty:
            return None
        last_day = sub["snap_date"].max()
        sub = sub[sub["snap_date"] == last_day]
    idx = (sub["strike"] - target_strike).abs().idxmin()
    return theta_quote(ticker, expiration, float(sub.loc[idx, "strike"]), "put",
                       sub.loc[idx, "snap_date"])


def theta_expiration_for(ticker: str, as_of, dte: int, tol_days: int = 12) -> str | None:
    """The Theta expiration closest to as_of+dte (within tol)."""
    exps = theta_expirations(ticker)
    if not exps:
        return None
    target = (pd.Timestamp(as_of) + pd.Timedelta(days=dte)).normalize()
    best, bestdiff = None, None
    for e in exps:
        try:
            ed = pd.Timestamp(datetime.strptime(e, "%Y%m%d"))
        except ValueError:
            continue
        diff = abs((ed - target).days)
        if bestdiff is None or diff < bestdiff:
            best, bestdiff = e, diff
    if bestdiff is not None and bestdiff <= tol_days:
        return best
    return None


# --------------------------------------------------------------------------- #
# Forward-outcome realizer (short put)
# --------------------------------------------------------------------------- #
def realize_short_put(ticker: str, strike: float, as_of, dte: int,
                      premium: float) -> dict | None:
    """Realize a short-put forecast against the actual underlying path.

    Returns the realized close, both attribution flags, and frictionless realized
    P&L per contract using the supplied (engine-synthetic) premium.
      * otm_expire  : close >= strike            (HT-B / calibration-doc convention)
      * engine_exact: close >  strike - premium  (matches prob_profit definition)
    """
    rc = realized_close_at(ticker, as_of, dte)
    if rc is None:
        return None
    intrinsic = max(0.0, float(strike) - rc)
    pnl = (float(premium) - intrinsic) * 100.0
    return {
        "realized_close": rc,
        "otm_expire": bool(rc >= strike),
        "engine_exact": bool(rc > (strike - premium)),
        "intrinsic": intrinsic,
        "assigned": bool(rc < strike),
        "realized_pnl_synth": pnl,
    }


# --------------------------------------------------------------------------- #
# Regimes + passive index proxy
# --------------------------------------------------------------------------- #
# Calendar-defined regimes (entry windows). Realization caps at OHLCV end 2026-03-20.
REGIMES: dict[str, tuple[str, str]] = {
    "crash_2020": ("2020-01-02", "2020-06-30"),     # COVID crash + V-recovery
    "bull_2021": ("2021-01-04", "2021-12-31"),       # low-vol bull
    "bear_2022": ("2022-01-03", "2022-12-30"),       # drawdown / rising rates
    "recovery_2023_2024": ("2023-01-03", "2024-12-31"),
    "recent_2025": ("2025-01-02", "2026-02-10"),     # most recent (realize <= 2026-03-20)
}


def index_proxy_return(tickers, start, end, weight: str = "cap") -> dict:
    """Buy-and-hold close-to-close return of a basket over (start, end].

    weight='cap' uses POINT-IN-TIME cap (shares x price_at_start, S&P-index-like
    proxy, no look-ahead); 'ew' equal-weights. Only tickers with both bookend
    closes are included. Price return only (dividends handled separately in I2).
    """
    panel = close_panel()
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    d0 = panel.index[panel.index >= s]
    d1 = panel.index[panel.index <= e]
    if len(d0) == 0 or len(d1) == 0:
        return {"return_pct": None, "n": 0}
    d0, d1 = d0[0], d1[-1]
    rows = []
    for t in tickers:
        t = _norm(t)
        if t not in panel.columns:
            continue
        p0, p1 = panel.at[d0, t], panel.at[d1, t]
        if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
            continue
        if weight == "cap":
            sh = shares_out(t)
            w = sh * float(p0) if sh is not None else None  # PIT cap = shares x price_at_start
        else:
            w = 1.0
        if w is None or w <= 0:
            continue
        rows.append((t, p0, p1, p1 / p0 - 1.0, w))
    if not rows:
        return {"return_pct": None, "n": 0}
    rdf = pd.DataFrame(rows, columns=["ticker", "p0", "p1", "ret", "w"])
    wsum = rdf["w"].sum()
    ret = float((rdf["ret"] * rdf["w"]).sum() / wsum)
    return {
        "return_pct": ret * 100.0, "n": len(rdf),
        "weight": weight, "d0": d0.date().isoformat(), "d1": d1.date().isoformat(),
        "median_ret_pct": float(rdf["ret"].median() * 100.0),
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import math
    print(f"ROOT={ROOT}")
    print(f"THETA_ROOT={THETA_ROOT} available={THETA_AVAILABLE}")
    fails = []

    # 1. OHLCV invariant + connector cross-check
    panel = close_panel()
    print(f"[1] close_panel {panel.shape} dates {panel.index.min().date()}..{panel.index.max().date()}")
    try:
        from engine.data_connector import MarketDataConnector
        conn = MarketDataConnector()
        for t, d, want in [("AAPL", "2022-06-01", None), ("MSFT", "2021-03-15", None)]:
            o = conn.get_ohlcv(t, start_date="2021-01-01", end_date="2026-03-20")
            cc = float(o.loc[pd.Timestamp(d), "close"])
            mine = float(close_series(t).loc[pd.Timestamp(d)])
            agree = math.isclose(cc, mine, rel_tol=1e-9)
            print(f"    {t} {d}: connector={cc} mine={mine} agree={agree}")
            if not agree:
                fails.append(f"close mismatch {t} {d}")
    except Exception as ex:  # noqa: BLE001
        print(f"    connector cross-check skipped: {type(ex).__name__}: {ex}")

    # 2. fundamentals + universe
    f = fundamentals()
    u = universe_503()
    print(f"[2] fundamentals {f.shape}, universe {len(u)}, AAPL sector={sector_of('AAPL')} cap={market_cap_of('AAPL'):.3e}")
    if len(u) < 400:
        fails.append("universe too small")

    # 3. Theta quote (AAPL exp 20240809 put, mid-2024) + nearest-put
    if THETA_AVAILABLE:
        exps = theta_expirations("AAPL")
        print(f"[3] AAPL theta expirations: {len(exps)} ({exps[0]}..{exps[-1]})")
        e = theta_expiration_for("AAPL", "2024-07-08", 35)
        print(f"    expiration_for(2024-07-08,+35d) -> {e}")
        if e:
            spot = float(close_series("AAPL").asof(pd.Timestamp("2024-07-08")))
            q = theta_nearest_put("AAPL", e, spot * 0.92, "2024-07-08")
            print(f"    spot~{spot:.2f} nearest 8%-OTM put quote: {q}")
            if q is None or q.get("bid") is None:
                fails.append("theta put quote missing")
    else:
        print("[3] THETA unavailable -- real-spread paths will fall back to modeled")

    # 4. realizer
    r = realize_short_put("AAPL", 140.0, "2022-06-01", 35, 2.388)
    print(f"[4] realize AAPL K140 2022-06-01 +35d: {r}")
    if r is None:
        fails.append("realizer returned None")

    # 5. index proxy (bull 2021)
    s0, s1 = REGIMES["bull_2021"]
    ip_cap = index_proxy_return(u, s0, s1, "cap")
    ip_ew = index_proxy_return(u, s0, s1, "ew")
    print(f"[5] 2021 index proxy: cap={ip_cap['return_pct']:.2f}% (n={ip_cap['n']}) ew={ip_ew['return_pct']:.2f}%")

    print("\nSELF-TEST:", "FAIL " + "; ".join(fails) if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
