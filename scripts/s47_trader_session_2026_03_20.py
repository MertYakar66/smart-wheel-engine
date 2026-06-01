#!/usr/bin/env python
"""S47 — Real wheel-trading session at as_of=2026-03-20 (reproducible driver).

A quant sits down to *use* the Smart Wheel Engine, not test it. The Bloomberg
data runs to 2026-03-20; we treat that as "today" and stay point-in-time
correct (the data simply ends there, so look-ahead is impossible). The driver
walks the full workflow a wheel trader actually runs and dumps every engine
answer it gets so the findings doc (``docs/worklog/s47-*.md``) can quote real
numbers only.

Run (from the repo root, Bloomberg provider):

    set SWE_DATA_PROVIDER=bloomberg            # Windows
    python scripts/s47_trader_session_2026_03_20.py

Output: a readable transcript to stdout + a JSON artifact at
``%TEMP%/s47_session_2026-03-20.json`` (regenerable; not committed).

OBSERVE-ONLY: this driver imports the engine and reads data. It does NOT
modify engine/ and does not place orders. The WheelTracker positions it
builds are an in-memory trader's book, discarded at exit.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback
from datetime import date, datetime
from pathlib import Path

# --- import the engine from THIS checkout, not a shadowing primary clone ---
# (per the sys.path-worktree-shadow gotcha: user-site .pth can import engine
# from an older clone if we don't put the worktree first)
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
os.chdir(_HERE)
os.environ.setdefault("SWE_DATA_PROVIDER", "bloomberg")

# Windows consoles default to cp1252 and crash on non-ASCII (e.g. the engine's
# "±5d buffer" drop reasons) when stdout is redirected. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import pandas as pd  # noqa: E402

from engine.option_pricer import black_scholes_delta, black_scholes_price  # noqa: E402
from engine.tradingview_bridge import ChartContext  # noqa: E402
from engine.wheel_runner import WheelRunner, _resolve_pit_atm_iv  # noqa: E402
from engine.wheel_tracker import WheelTracker  # noqa: E402

AS_OF = "2026-03-20"
ACCOUNT = 250_000.0
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 260)

# A realistic, liquid, wheel-appropriate watchlist spread across sectors —
# the kind of book a $250k income trader would actually screen.
WHEEL_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "GOOGL",
    "META",
    "CRM",
    "ORCL",
    "AVGO",
    "ADBE",
    "AMZN",
    "HD",
    "NKE",
    "SBUX",
    "MCD",
    "KO",
    "PG",
    "WMT",
    "COST",
    "DIS",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "WFC",
    "V",
    "MA",
    "AXP",
    "UNH",
    "JNJ",
    "PFE",
    "ABBV",
    "MRK",
    "LLY",
    "XOM",
    "CVX",
    "CAT",
    "BA",
    "GE",
    "HON",
    "T",
    "VZ",
    "NFLX",
    "QCOM",
    "TXN",
    "INTC",
]

OUT: dict = {"as_of": AS_OF, "account": ACCOUNT, "steps": {}}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def spot_at(conn, ticker: str, as_of: str) -> float | None:
    """Last close on or before as_of (PIT). get_ohlcv returns date as the index."""
    try:
        df = conn.get_ohlcv(ticker)
        if df is None or len(df) == 0:
            return None
        df = df.reset_index()  # date index -> column named 'date'
        datecol = "date" if "date" in df.columns else df.columns[0]
        df[datecol] = pd.to_datetime(df[datecol])
        df = df[df[datecol] <= pd.Timestamp(as_of)]
        if len(df) == 0:
            return None
        return float(df.iloc[-1]["close"])
    except Exception:
        return None


def pit_iv(conn, ticker: str, as_of: str) -> float | None:
    try:
        return _resolve_pit_atm_iv(conn, ticker, as_of)
    except Exception:
        return None


def show(df: pd.DataFrame, cols: list[str]) -> None:
    use = [c for c in cols if c in df.columns]
    print(df[use].to_string())


class CleanChartProvider:
    """A chart provider that returns a clean, price-matching chart so the
    dossier reviewer's R2 (chart-missing) and R3 (spot-mismatch) do not fire —
    isolating R11 (elevated-vol top-bin) so we can see it in a VIX>25 tape.
    Mirrors a trader who DOES have a live chart up showing the real price."""

    def __init__(self, conn, as_of: str):
        self._conn = conn
        self._as_of = as_of

    def fetch(self, ticker: str, timeframe="1D", *, as_of=None) -> ChartContext:
        px = spot_at(self._conn, ticker, self._as_of)
        return ChartContext(
            ticker=ticker,
            timeframe="1D",
            captured_at=datetime(2026, 3, 20, 16, 0, 0),
            screenshot_path=Path(tempfile.gettempdir()) / f"s47_{ticker}.png",
            visible_price=px,
            visible_indicators={},
            source="s47-clean-stub",
        )


# --------------------------------------------------------------------------- #
def step0_provenance(runner: WheelRunner) -> None:
    banner("STEP 0 — Provenance / reproducibility header")
    import engine.wheel_runner as wr

    prov = {"engine_module": wr.__file__, "connector": type(runner.connector).__name__}
    try:
        import subprocess

        prov["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_HERE), text=True
        ).strip()
    except Exception as e:  # noqa: BLE001
        prov["git_head"] = f"(unavailable: {e})"
    # data coverage + OHLCV hash (so a re-run on different data is detectable)
    try:
        ohlcv_path = _HERE / "data" / "bloomberg" / "sp500_ohlcv.csv"
        h = hashlib.sha256()
        with open(ohlcv_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        prov["ohlcv_sha256"] = h.hexdigest()[:16]
        prov["ohlcv_bytes"] = ohlcv_path.stat().st_size
    except Exception as e:  # noqa: BLE001
        prov["ohlcv_sha256"] = f"(unavailable: {e})"
    for k, v in prov.items():
        print(f"  {k}: {v}")
    OUT["steps"]["0_provenance"] = prov


def step1_market_context(runner: WheelRunner) -> None:
    banner("STEP 1 — Market context at 2026-03-20 (what tape are we trading?)")
    conn = runner.connector
    ctx: dict = {}
    try:
        ctx["vix_regime"] = conn.get_vix_regime(AS_OF)
        print(f"  VIX regime: {ctx['vix_regime']}")
    except Exception as e:  # noqa: BLE001
        ctx["vix_regime"] = f"err: {e}"
    try:
        from engine.data_integration import get_current_risk_free_rate

        ctx["risk_free_rate"] = get_current_risk_free_rate(AS_OF, data_dir="data/bloomberg")
        print(f"  risk-free rate: {ctx['risk_free_rate']:.4f}")
    except Exception as e:  # noqa: BLE001
        ctx["risk_free_rate"] = f"err: {e}"
    # HMM regime read off the ranker's own diagnostic for a couple of names
    print(f"  account: ${ACCOUNT:,.0f} cash-secured-put wheel book")
    OUT["steps"]["1_market_context"] = ctx


def step2_rank_puts(runner: WheelRunner) -> pd.DataFrame:
    banner("STEP 2 — 'What should I sell puts on today?' (35 DTE, 25-delta, +EV only)")
    df = runner.rank_candidates_by_ev(
        tickers=WHEEL_UNIVERSE,
        dte_target=35,
        delta_target=0.25,
        top_n=25,
        min_ev_dollars=0.0,  # a real desk wants positive EV
        as_of=AS_OF,
        include_diagnostic_fields=True,
        use_event_gate=True,
    )
    print(
        f"\nUniverse screened: {len(WHEEL_UNIVERSE)} names -> {len(df)} +EV candidates returned\n"
    )
    cols = [
        "ticker",
        "spot",
        "strike",
        "dte",
        "iv",
        "premium",
        "prob_profit",
        "prob_assignment",
        "ev_dollars",
        "ev_per_day",
        "roc",
        "cvar_5",
        "days_to_earnings",
        "hmm_regime",
        "distribution_source",
    ]
    show(df, cols)
    drops = df.attrs.get("drops", [])
    print(f"\nDROPPED (not shown above): {len(drops)} names, with recorded reasons:")
    # group drop reasons by gate
    by_gate: dict = {}
    for d in drops:
        by_gate.setdefault(d.get("gate", "?"), []).append(d)
    for gate, items in sorted(by_gate.items()):
        print(f"  [{gate}] x{len(items)}: " + ", ".join(sorted(i["ticker"] for i in items)))
    # show a few full reasons
    for d in drops[:8]:
        print(f"    - {d['ticker']}: {d['reason']}")
    OUT["steps"]["2_rank_puts"] = {
        "n_universe": len(WHEEL_UNIVERSE),
        "n_returned": len(df),
        "candidates": df[[c for c in cols if c in df.columns]].to_dict("records"),
        "drops": drops,
    }
    return df


def step3_interrogate(runner: WheelRunner, df: pd.DataFrame) -> None:
    banner("STEP 3 — Interrogate the top candidates: are strike/premium/prob realistic?")
    rfr = OUT["steps"]["1_market_context"].get("risk_free_rate")
    if not isinstance(rfr, float):
        rfr = 0.0362
    recs = []
    for _, row in df.head(5).iterrows():
        S, K = float(row["spot"]), float(row["strike"])
        iv, dte = float(row["iv"]), int(row["dte"])
        prem = float(row["premium"])
        T = dte / 365.0
        # 1) internal consistency: recompute the BSM put with the SAME inputs
        bsm_put = black_scholes_price(S, K, T, rfr, iv, "put", q=0.0)
        bsm_delta = black_scholes_delta(S, K, T, rfr, iv, "put", q=0.0)
        # 2) realism vs a real chain: large-cap 25-delta puts carry skew
        #    (OTM put IV > ATM IV). The Bloomberg connector has NO skew
        #    (put_iv==call_iv), so the engine prices the OTM put at ATM IV.
        #    Estimate the understatement with a conservative +3 vol-pt skew bump.
        skew_bump = 0.03
        bsm_put_skew = black_scholes_price(S, K, T, rfr, iv + skew_bump, "put", q=0.0)
        # 3) breakeven + does prob_profit ignore the premium cushion?
        breakeven = K - prem
        rec = {
            "ticker": row["ticker"],
            "spot": round(S, 2),
            "strike": K,
            "otm_pct": round((S - K) / S * 100, 2),
            "iv": round(iv, 4),
            "dte": dte,
            "engine_premium": round(prem, 3),
            "bsm_recompute_q0": round(bsm_put, 3),
            "bsm_minus_engine": round(bsm_put - prem, 3),
            "bsm_put_delta": round(bsm_delta, 4),
            "prem_skew_+3vol": round(bsm_put_skew, 3),
            "skew_uplift_pct": round((bsm_put_skew - prem) / prem * 100, 1),
            "engine_prob_profit": round(float(row["prob_profit"]), 4),
            "engine_prob_assignment": round(float(row["prob_assignment"]), 4),
            "pp_plus_pa": round(float(row["prob_profit"]) + float(row["prob_assignment"]), 4),
            "breakeven": round(breakeven, 2),
            "roc_pct": round(float(row["roc"]) * 100, 3),
        }
        recs.append(rec)
        print(
            f"\n  {rec['ticker']}: spot {rec['spot']} strike {rec['strike']} "
            f"({rec['otm_pct']}% OTM), {dte}DTE, IV {rec['iv']}"
        )
        print(
            f"    premium: engine={rec['engine_premium']}  "
            f"BSM-recompute(q=0)={rec['bsm_recompute_q0']}  "
            f"diff={rec['bsm_minus_engine']}  -> internal consistency"
        )
        print(f"    solved-strike delta (BSM, q=0)={rec['bsm_put_delta']}  (target was -0.25)")
        print(
            f"    skew realism: +3vol-pt premium={rec['prem_skew_+3vol']}  "
            f"=> a real chain would pay ~{rec['skew_uplift_pct']}% MORE than the engine quotes"
        )
        # prob_profit = P(S_T > breakeven=strike-premium); prob_assignment =
        # P(S_T < strike). Since breakeven < strike, prob_profit >= 1-prob_assignment,
        # so the two SHOULD sum to >= 1 — confirming prob_profit counts the premium
        # cushion (it is NOT merely 1 - prob_assignment).
        cushion = (
            "counts the premium cushion (>1-prob_assignment)"
            if rec["pp_plus_pa"] >= 1.0
            else "BELOW 1-prob_assignment (unexpected)"
        )
        print(
            f"    prob_profit={rec['engine_prob_profit']}  "
            f"prob_assignment={rec['engine_prob_assignment']}  "
            f"sum={rec['pp_plus_pa']} => {cushion}; breakeven {rec['breakeven']}; "
            f"both are k/35 empirical counts (35-DTE -> 35 non-overlapping windows)"
        )
    OUT["steps"]["3_interrogate"] = recs


def step4_earnings_gate(runner: WheelRunner) -> None:
    banner("STEP 4 — Earnings edge: does the engine refuse to sell over earnings?")
    # NKE earnings 2026-03-31 — inside any 35-DTE put's life from 2026-03-20.
    res = {}
    for name, gate in [("event_gate_ON", True), ("event_gate_OFF", False)]:
        df = runner.rank_candidates_by_ev(
            tickers=["NKE", "KO"],  # NKE has earnings 03-31; KO is a clean control
            dte_target=35,
            delta_target=0.25,
            top_n=10,
            min_ev_dollars=-1e9,
            as_of=AS_OF,
            include_diagnostic_fields=True,
            use_event_gate=gate,
        )
        present = sorted(df["ticker"].tolist())
        drops = {d["ticker"]: d["reason"] for d in df.attrs.get("drops", [])}
        print(f"\n  {name}: returned {present}")
        if drops:
            for t, r in drops.items():
                print(f"    dropped {t}: {r}")
        res[name] = {"returned": present, "drops": drops}
    OUT["steps"]["4_earnings_gate"] = res


def step5_elevated_vol_r11(runner: WheelRunner) -> None:
    banner("STEP 5 — Elevated-vol edge (VIX 28.97 > 25): does R11 size down top-bin picks?")
    conn = runner.connector
    chart = CleanChartProvider(conn, AS_OF)
    # Deep-OTM puts (low delta) push prob_profit > 0.90 -> the R11 'top bin'.
    res = {}
    for label, delta in [("25-delta (normal)", 0.25), ("12-delta (deep OTM, top-bin)", 0.12)]:
        try:
            dossiers = runner.build_candidate_dossiers(
                tickers=["AAPL", "MSFT", "KO", "PG", "WMT", "V", "MA"],
                dte_target=35,
                delta_target=delta,
                top_n=8,
                min_ev_dollars=0.0,
                as_of=AS_OF,
                chart_provider=chart,
            )
            rows = []
            for d in dossiers:
                ev = getattr(d, "ev_row", {}) or {}
                rows.append(
                    {
                        "ticker": getattr(d, "ticker", None),
                        "prob_profit": ev.get("prob_profit"),
                        "vix_level": getattr(d, "vix_level", None),
                        "verdict": getattr(d, "verdict", None),
                        "verdict_reason": getattr(d, "verdict_reason", None),
                    }
                )
            print(f"\n  {label}: {len(rows)} dossiers")
            for r in rows:
                print(
                    f"    {r['ticker']}: prob_profit={r['prob_profit']} "
                    f"vix_level={r['vix_level']} -> {r['verdict']} ({r['verdict_reason']})"
                )
            res[label] = rows
        except Exception as e:  # noqa: BLE001
            print(f"  {label}: ERROR {type(e).__name__}: {e}")
            res[label] = f"err: {e}"
    OUT["steps"]["5_elevated_vol_r11"] = res


def step6_concentration(runner: WheelRunner, df: pd.DataFrame) -> None:
    banner("STEP 6 — Concentrated book: does the engine warn about sector/name clustering?")
    import inspect

    conn = runner.connector
    out: dict = {}
    # (a) Does the PRIMARY 'what should I sell' call know your book at all?
    risk_cols = [
        c
        for c in df.columns
        if any(
            k in c.lower()
            for k in (
                "sector_cap",
                "concentration",
                "position_size",
                "single_name",
                "max_position",
                "kelly",
            )
        )
    ]
    rank_params = [
        p
        for p in inspect.signature(runner.rank_candidates_by_ev).parameters
        if any(k in p.lower() for k in ("portfolio", "book", "existing", "holdings"))
    ]
    print(f"  (a) rank_candidates_by_ev concentration columns: {risk_cols or 'NONE'}")
    print(f"      rank_candidates_by_ev params accepting an existing book: {rank_params or 'NONE'}")
    if "sector" in df.columns:
        bysec = df.head(15).groupby("sector").size().sort_values(ascending=False)
        print("      Top-15 +EV picks by sector (clusters, but engine is silent):")
        for sec, n in bysec.items():
            print(f"        {sec}: {n}")
    out["ranker_concentration_cols"] = risk_cols
    out["ranker_portfolio_params"] = rank_params
    out["top15_by_sector"] = (
        df.head(15).groupby("sector").size().to_dict() if "sector" in df.columns else {}
    )

    # (b) The chart-aware path CAN take a book (portfolio_context), but it is
    #     opt-in and OFF by default. Big-notional names (strike*100 > 20% of
    #     $250k NAV) should trip R10 single-name — IF you pass the context.
    dos_params = [
        p
        for p in inspect.signature(runner.build_candidate_dossiers).parameters
        if "portfolio" in p.lower() or "context" in p.lower()
    ]
    print(
        f"\n  (b) build_candidate_dossiers book param: {dos_params or 'NONE'}  (defaults to None)"
    )
    chart = CleanChartProvider(conn, AS_OF)
    big = ["LLY", "CAT", "MA", "AAPL"]  # LLY/CAT notionals > 20% of $250k
    for K in big:
        s = spot_at(conn, K, AS_OF)
        if s:
            print(
                f"      {K}: spot {s:.0f} -> ~25d put notional ≈ ${s * 0.92 * 100:,.0f} "
                f"({s * 0.92 * 100 / ACCOUNT * 100:.0f}% of NAV)"
            )

    def verdicts(pctx):
        ds = runner.build_candidate_dossiers(
            tickers=big,
            dte_target=35,
            delta_target=0.25,
            top_n=8,
            min_ev_dollars=0.0,
            as_of=AS_OF,
            chart_provider=chart,
            portfolio_context=pctx,
        )
        return [
            {
                "ticker": getattr(d, "ticker", None),
                "verdict": getattr(d, "verdict", None),
                "reason": getattr(d, "verdict_reason", None),
                "notes": [n for n in getattr(d, "review_notes", []) if "R9" in n or "R10" in n],
            }
            for d in ds
        ]

    no_ctx = verdicts(None)
    print("\n      WITHOUT portfolio_context (default trader path):")
    for r in no_ctx:
        print(f"        {r['ticker']}: {r['verdict']} ({r['reason']})")

    booked = WheelTracker(initial_capital=ACCOUNT, connector=conn)
    spots = {t: spot_at(conn, t, AS_OF) for t in big if spot_at(conn, t, AS_OF)}
    try:
        pctx = booked.portfolio_context_snapshot(spot_prices=spots)
        with_ctx = verdicts(pctx)
        nav = float(getattr(pctx, "nav", 0.0) or 0.0)
        print(f"\n      WITH portfolio_context (NAV ${nav:,.0f}, empty book):")
        for r in with_ctx:
            print(f"        {r['ticker']}: {r['verdict']} ({r['reason']})")
            for n in r.get("notes", []):
                print(f"            {n}")
        out["with_context"] = with_ctx
        out["nav"] = nav
    except Exception as e:  # noqa: BLE001
        print(f"      portfolio_context_snapshot ERROR: {type(e).__name__}: {e}")
        out["with_context"] = f"err: {e}"
    out["dossier_portfolio_params"] = dos_params
    out["without_context"] = no_ctx
    print("\n  -> R9 (sector_cap) / R10 (single_name) are OPT-IN: the primary ranker can't")
    print("     see your book; only build_candidate_dossiers(portfolio_context=...) engages them.")
    OUT["steps"]["6_concentration"] = out


def _suggest_rolls_both(conn, pos: dict, rfr: float) -> dict:
    """Open the challenged put in a fresh tracker, then ask the engine for rolls
    two ways: the default (credit-only) and with the credit filter removed (so a
    debit roll can surface). Returns counts + the rows."""
    entry = date(2026, 2, 13)
    expiration = date(2026, 4, 10)  # ~21 DTE left at 2026-03-20
    iv_entry = pit_iv(conn, pos["ticker"], "2026-02-13") or 0.30
    entry_prem = black_scholes_price(
        pos["s_entry"],
        pos["strike"],
        (expiration - entry).days / 365.0,
        rfr,
        iv_entry,
        "put",
        q=0.0,
    )
    iv_now = pit_iv(conn, pos["ticker"], AS_OF) or 0.35
    rcols = [
        "new_strike",
        "new_expiry",
        "new_dte",
        "net_credit_debit",
        "roll_ev",
        "hold_ev",
        "new_ev_dollars",
        "prob_otm",
        "recommend",
    ]
    out = {"entry_premium": round(entry_prem, 2), "iv_now": round(iv_now, 4), "variants": {}}
    for label, min_credit in [("credit_only(default)", 0.0), ("allow_debit", -1e9)]:
        tracker = WheelTracker(initial_capital=ACCOUNT, connector=conn)
        tracker.open_short_put(
            ticker=pos["ticker"],
            strike=pos["strike"],
            premium=entry_prem,
            entry_date=entry,
            expiration_date=expiration,
            iv=iv_entry,
        )
        try:
            rolls = tracker.suggest_rolls(
                ticker=pos["ticker"],
                as_of=date(2026, 3, 20),
                current_spot=pos["s_now"],
                current_iv=iv_now,
                risk_free_rate=rfr,
                min_net_credit=min_credit,
            )
            cols = [c for c in rcols if c in rolls.columns]
            print(f"    suggest_rolls [{label}] -> {len(rolls)} candidates")
            if len(rolls):
                show(rolls.head(6), cols)
            out["variants"][label] = rolls[cols].to_dict("records") if len(rolls) else []
            out["roll_columns"] = list(rolls.columns)
        except Exception as e:  # noqa: BLE001
            print(f"    suggest_rolls [{label}] ERROR: {type(e).__name__}: {e}")
            out["variants"][label] = f"err: {e}"
    return out


def step7_roll(runner: WheelRunner) -> dict:
    banner("STEP 7 — Challenged short puts: what does the engine say about rolling?")
    conn = runner.connector
    rfr = OUT["steps"]["1_market_context"].get("risk_free_rate")
    if not isinstance(rfr, float):
        rfr = 0.0362
    # Scan the whole watchlist for puts sold ~5 weeks ago (ATM at entry) that are
    # now ITM — bucket into 'deeply challenged' and 'moderately challenged'.
    challenged = []
    for tkr in WHEEL_UNIVERSE:
        s_entry = spot_at(conn, tkr, "2026-02-13")
        s_now = spot_at(conn, tkr, AS_OF)
        if not s_entry or not s_now:
            continue
        strike = round(s_entry * 2) / 2.0
        itm = strike - s_now
        if itm > 0:
            challenged.append(
                {
                    "ticker": tkr,
                    "s_entry": s_entry,
                    "s_now": s_now,
                    "strike": strike,
                    "itm": itm,
                    "itm_pct": itm / strike,
                }
            )
    challenged.sort(key=lambda d: d["itm_pct"], reverse=True)
    deep = challenged[0] if challenged else None
    moderate = next((c for c in challenged if 0.02 <= c["itm_pct"] <= 0.09), None)
    print(
        f"  Challenged positions found: {len(challenged)} "
        f"(of {len(WHEEL_UNIVERSE)} watchlist names)"
    )
    results = {}
    for label, pos in [("DEEP", deep), ("MODERATE", moderate)]:
        if not pos:
            print(f"\n  [{label}] none found in band")
            continue
        print(
            f"\n  [{label}] {pos['ticker']}: sold {pos['strike']} put 2026-02-13 "
            f"(spot then {pos['s_entry']:.2f}); spot now {pos['s_now']:.2f} "
            f"=> {pos['itm']:.2f} ITM ({pos['itm_pct'] * 100:.1f}% of strike)"
        )
        results[label] = {"position": pos, **_suggest_rolls_both(conn, pos, rfr)}
    OUT["steps"]["7_roll"] = results
    return {"deep": deep, "rfr": rfr}


def step8_lifecycle(runner: WheelRunner, roll_ctx: dict) -> None:
    banner("STEP 8 — Lifecycle: assignment -> covered-call leg (does it refuse -EV calls?)")
    conn = runner.connector
    deep = roll_ctx["deep"]
    if not deep:
        print("  no challenged position to assign")
        return
    tkr, strike, s_now = deep["ticker"], deep["strike"], deep["s_now"]
    # Take assignment on the deeply-ITM put.
    tracker = WheelTracker(initial_capital=ACCOUNT, connector=conn)
    iv_entry = pit_iv(conn, tkr, "2026-02-13") or 0.30
    entry_prem = black_scholes_price(
        deep["s_entry"],
        strike,
        (date(2026, 4, 10) - date(2026, 2, 13)).days / 365.0,
        roll_ctx["rfr"],
        iv_entry,
        "put",
        q=0.0,
    )
    tracker.open_short_put(
        ticker=tkr,
        strike=strike,
        premium=entry_prem,
        entry_date=date(2026, 2, 13),
        expiration_date=date(2026, 4, 10),
        iv=iv_entry,
    )
    ok = tracker.handle_put_assignment(tkr, date(2026, 3, 20), s_now)
    pos = tracker.positions.get(tkr)
    basis = getattr(pos, "stock_basis", None)
    print(
        f"  handle_put_assignment -> {ok}; state={pos.state if pos else None}; "
        f"stock_basis={basis} (strike was {strike}); spot now {s_now:.2f}"
    )
    print(
        f"  => underwater on the shares by ${(basis - s_now) * 100:,.0f}/100sh "
        f"(before the ${entry_prem:.2f}/sh put premium already booked to realized P&L)"
    )

    # Covered-call leg, two ways: +EV-only (what a trader sees) and EV-floor-off
    # (so we can SEE the calls the engine is refusing and why).
    res = {
        "assignment_ok": ok,
        "stock_basis": basis,
        "spot_now": s_now,
        "entry_premium": round(entry_prem, 2),
    }
    for label, floor in [("+EV only (default)", 0.0), ("EV-floor OFF (show refused)", -1e9)]:
        cc = runner.rank_covered_calls_by_ev(
            ticker=tkr,
            shares_held=100,
            target_dtes=(21, 35, 49),
            target_deltas=(0.30, 0.25, 0.20, 0.15),
            as_of=AS_OF,
            top_n=12,
            min_ev_dollars=floor,
            include_diagnostic_fields=True,
        )
        print(f"\n  rank_covered_calls_by_ev({tkr}) [{label}] -> {len(cc)} candidates")
        ccols = [
            c
            for c in [
                "strike",
                "dte",
                "iv",
                "premium",
                "prob_profit",
                "prob_assignment",
                "ev_dollars",
                "ev_per_day",
                "roc",
            ]
            if c in cc.columns
        ]
        if len(cc):
            show(cc, ccols)
            below = cc[cc["strike"] < basis] if basis else cc.iloc[0:0]
            print(
                f"    CC strikes BELOW basis {basis} (would cap a loss if assigned): "
                f"{sorted(below['strike'].tolist()) if len(below) else 'NONE'}"
            )
            res[label] = {
                "candidates": cc[ccols].to_dict("records"),
                "strikes_below_basis": sorted(below["strike"].tolist()) if len(below) else [],
            }
        else:
            res[label] = {"candidates": [], "drops": cc.attrs.get("drops", [])}
    OUT["steps"]["8_lifecycle"] = res


def main() -> None:
    print(f"S47 — wheel trading session @ {AS_OF} | account ${ACCOUNT:,.0f}")
    runner = WheelRunner()
    step0_provenance(runner)
    step1_market_context(runner)
    df = step2_rank_puts(runner)
    if len(df):
        step3_interrogate(runner, df)
    step4_earnings_gate(runner)
    step5_elevated_vol_r11(runner)
    if len(df):
        step6_concentration(runner, df)
    try:
        roll_ctx = step7_roll(runner)
        step8_lifecycle(runner, roll_ctx)
    except Exception:  # noqa: BLE001
        print("STEP 7/8 ERROR:\n" + traceback.format_exc())
        OUT["steps"].setdefault("7_roll", {})["error"] = traceback.format_exc()

    art = Path(tempfile.gettempdir()) / "s47_session_2026-03-20.json"
    with open(art, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    banner(f"DONE — JSON artifact: {art}")


if __name__ == "__main__":
    main()
