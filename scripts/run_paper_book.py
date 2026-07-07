"""Drive the forward paper-trading book — the "simulation world" loop.

A simulated wheel book that the **real** engine ranks and manages day-by-day,
accumulating a live equity curve with ZERO money at risk. This is the driver
half; the reporting/state half is :mod:`engine.paper_book` (trio-free).

Two motions, honestly labelled (task KEY DESIGN)
------------------------------------------------
* ``seed`` — seed a backfill from a recent point-in-time run by reusing
  ``backtests.regression._common.run_backtest`` to drive a WheelTracker over the
  last N months. Labelled ``backfill`` (in-sample-ish). The seed uses the
  caps-OFF ``run_backtest`` tracker on purpose: it is PIT-correct (never calls
  ``_compute_live_nav``), whereas an armed tracker's caps mark NAV at
  ``date.today()`` and would leak future prices into a *historical* seed.
* ``forward`` — one idempotent daily FORWARD-APPEND: re-rank ``as_of`` through
  the real engine and open up to N EV>0 positions on a **caps-armed** book
  (``make_live_book_tracker`` → R9 sector + R10 single-name), mark-to-market,
  settle due trades, and append exactly ONE ``forward`` (genuinely OOS) point.
  For a genuine forward step ``as_of == today`` and the armed caps' ``date.today()``
  NAV is correct. ``as_of`` is clamped to the data frontier (the engine returns
  no candidates once ``today`` drifts past the latest healthy bar).

§2 (CLAUDE.md): this driver CONSUMES the ranker read-only. It never mutates
``ev_dollars`` / ``ev_raw`` / ``prob_profit`` / a verdict and never feeds back
into ranking. It imports ``wheel_runner`` (allowed — the driver drives the
engine); the trio-free half is ``engine.paper_book`` (AST-guarded).

SIM namespace (task invariant 4): all artifacts go to ``$SWE_SIM_DATA_DIR`` (or
gitignored ``data_processed/sim/``). ``data_processed/ibkr/`` (Dashboard
terminal, §6) is never touched.

Usage
-----
Seed a backfill, then demonstrate one forward-append::

    python scripts/run_paper_book.py seed --book live_forward \\
        --tickers AAPL MSFT NVDA JPM GS XOM UNH JNJ HD PG KO CAT \\
        --start 2025-10-01 --end 2026-05-15 --capital 200000

    python scripts/run_paper_book.py forward --book live_forward --as-of 2026-06-04
    python scripts/run_paper_book.py forward --book live_forward --as-of 2026-06-04   # idempotent no-op

    python scripts/run_paper_book.py report --book live_forward   # recompute calibration + MC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ direct invocation lacks the repo root on sys.path (tests->scripts
# sys.path trap) — add it before importing engine.*.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine import paper_book as pb  # noqa: E402  (trio-free reporting/state lib)

DEFAULT_BOOK = "live_forward"
DEFAULT_DTE = 35
DEFAULT_DELTA = 0.25
DEFAULT_MAX_NEW_PER_DAY = 3
DEFAULT_FRICTION = "full"


# ---------------------------------------------------------------------------
# Engine / connector helpers (driver side — trio imports are allowed here)
# ---------------------------------------------------------------------------


def _runner():
    """Construct the WheelRunner and log the actually-selected provider
    (silent provider selection is a recurring bug — CLAUDE.md §4)."""
    os.environ.setdefault("SWE_DATA_PROVIDER", "bloomberg")
    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()
    provider = type(runner.connector).__name__
    print(f"[paper_book] provider={provider}", flush=True)
    return runner, runner.connector, provider


def _data_frontier(conn, probe_tickers: list[str]) -> date | None:
    """Latest OHLCV date available across a few probe tickers (the frontier)."""
    frontier: date | None = None
    for t in probe_tickers[:5]:
        try:
            df = conn.get_ohlcv(t)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        if "date" in df.columns:
            d = pd.to_datetime(df["date"]).dt.date.iloc[-1]
        else:
            d = pd.to_datetime(df.index[-1]).date()
        frontier = d if frontier is None else max(frontier, d)
    return frontier


def _spot_at_expiry(conn, ticker: str, expiry: date) -> float | None:
    """Close on/after ``expiry`` (settlement spot). Reuses the regression
    harness helper so the paper book settles identically to S27/S34."""
    from backtests.regression._common import _spot_on_or_after

    return _spot_on_or_after(conn, ticker, expiry)


def _spy_closes(conn, isos: list[str]) -> dict[str, float]:
    """Best-effort ``{iso: SPY close}`` over the point dates. SPY may be absent
    from the S&P-constituent CSVs; on any failure returns ``{}`` (the panel
    renders a benchmark gap — never a fabricated copy)."""
    if not isos:
        return {}
    try:
        df = conn.get_ohlcv("SPY", start_date=min(isos), end_date=max(isos))
    except Exception:
        return {}
    if df is None or len(df) == 0 or "close" not in getattr(df, "columns", []):
        return {}
    d = df.copy()
    if "date" in d.columns:
        d["iso"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    else:
        d["iso"] = pd.to_datetime(d.index).strftime("%Y-%m-%d")
    return {r.iso: float(r.close) for r in d.itertuples(index=False) if pd.notna(r.close)}


def _apply_spy(history: dict, conn) -> None:
    """(Re)derive the indexed SPY series for every history point (self-healing,
    mirrors ``dashboard_refresh._sync_curve``). Leaves points untouched when SPY
    data is unavailable rather than nulling a previously-derived value."""
    pts = history.get("points") or []
    if not pts:
        return
    isos = [p["date"] for p in pts if p.get("date")]
    closes = _spy_closes(conn, isos)
    if not closes:
        return
    base_port = float(pts[0]["port"])
    idx = pb.indexed_spy_series(closes, isos, base_port)
    for p in pts:
        if p.get("date") in idx:
            p["spy"] = idx[p["date"]]


def _per_name_returns(conn, tickers: list[str], start: str, end: str) -> dict[str, np.ndarray]:
    """Date-aligned daily close-to-close returns per name over [start, end].

    Inner-joins on the shared trading calendar BEFORE differencing so the copula
    (via engine.sim_portfolio) sees date-matched returns. Mirrors
    ``scripts.run_forward_sim._per_name_returns`` but shares this driver's
    connector instead of building a second one.
    """
    series: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            df = conn.get_ohlcv(t, start_date=start, end_date=end)
        except Exception:
            continue
        if df is None or df.empty or "close" not in df.columns:
            continue
        d = df.copy()
        if "date" in d.columns:
            d = d.set_index(pd.to_datetime(d["date"]))
        series[t] = pd.to_numeric(d["close"], errors="coerce")
    if not series:
        return {}
    frame = pd.DataFrame(series).sort_index()
    rets = frame.pct_change().dropna(how="any")
    out: dict[str, np.ndarray] = {}
    for t in rets.columns:
        col = rets[t].to_numpy(dtype=float)
        if len(col) >= 2:
            out[t] = col
    return out


def _weights_from_ledger(ledger: list[dict]) -> dict[str, float]:
    """Per-name short-put notional weights for the copula overlay, from the
    forecast ledger: ``notional[name] = median(strike) * 100 * count``.
    Short-put books are long the underlying → positive weights."""
    by_name: dict[str, list[float]] = {}
    for e in ledger:
        t = e.get("ticker")
        k = e.get("strike")
        if t and k and float(k) > 0:
            by_name.setdefault(str(t), []).append(float(k))
    return {t: float(np.median(ks)) * 100.0 * len(ks) for t, ks in by_name.items() if ks}


def _load_armed_tracker(store: pb.PaperBookStore, conn):
    """Load the persisted continuation state and ARM the R9/R10 caps.

    ``WheelTracker.to_dict`` does not serialise the ``enforce_*`` flags, so a
    reloaded tracker is caps-OFF; we re-arm it to the ``make_live_book_tracker``
    configuration (R9 sector + R10 single-name) so the forward path reproduces
    the gated live behaviour (task invariant 5). Existing positions carried from
    the backfill are grandfathered — the caps only refuse NEW over-concentrated
    opens, exactly as production would.
    """
    from engine.wheel_tracker import WheelTracker

    state_path = store.tracker_state_path()
    if not state_path.exists():
        return None
    tracker = WheelTracker.load(state_path, connector=conn)
    tracker.enforce_sector_cap = True
    tracker.enforce_single_name_cap = True
    if not (tracker.enforce_sector_cap and tracker.enforce_single_name_cap):
        raise RuntimeError("failed to arm R9/R10 caps on the forward tracker")
    return tracker


# ---------------------------------------------------------------------------
# Backfill forecast ledger (opened positions joined to the rank log)
# ---------------------------------------------------------------------------


def _backfill_forecast_ledger(
    state: dict, rank_log: pd.DataFrame, backfill_end: date
) -> list[dict]:
    """One forecast entry per position the seed backtest actually OPENED.

    Joins each opened short put (closed-with-put + still-open-with-put) back to
    its ``(entry_date, ticker)`` row in the rank log to recover the shipped
    ``prob_profit`` / ``ev_dollars`` and the held-to-expiry ``realized_pnl``
    (already forward-replayed by ``run_backtest``). Outcome = ``realized_pnl > 0``
    — the put-leg held-to-expiry result the engine's ``prob_profit`` models.
    """
    # Index the rank log by (date, ticker) → row dict.
    rl: dict[tuple[str, str], dict] = {}
    if rank_log is not None and not rank_log.empty:
        for r in rank_log.to_dict("records"):
            key = (str(r.get("date"))[:10], str(r.get("ticker")))
            rl.setdefault(key, r)  # first (only) row per (date, ticker)

    opened: list[tuple[str, str]] = []
    for rec in state.get("closed_positions", []):
        if (rec.get("put_premium") or 0) > 0 and rec.get("ticker"):
            opened.append((str(rec.get("entry_date"))[:10], str(rec["ticker"])))
    for t, pos in state.get("positions", {}).items():
        if (pos.get("put_premium") or 0) and pos.get("state") != "no_position":
            ed = pos.get("put_entry_date") or pos.get("entry_date")
            opened.append((str(ed)[:10], str(t)))

    entries: list[dict] = []
    for entry_iso, ticker in opened:
        row = rl.get((entry_iso, ticker))
        if row is None:
            continue  # no matching rank row → cannot recover the forecast; skip
        strike = float(row.get("strike", 0.0) or 0.0)
        premium = float(row.get("premium", 0.0) or 0.0)  # friction-adjusted in the rank log
        pp = row.get("prob_profit")
        pp = None if pp is None or (isinstance(pp, float) and np.isnan(pp)) else float(pp)
        exp = str(row.get("expiration_date"))[:10]
        e = pb.new_forecast_entry(
            entry_date=entry_iso,
            ticker=ticker,
            strike=strike,
            premium=premium,
            prob_profit=pp,
            expiration_date=exp,
            phase="backfill",
            ev_dollars=(float(row["ev_dollars"]) if row.get("ev_dollars") is not None else None),
        )
        realized = row.get("realized_pnl")
        spot_exp = row.get("spot_at_expiry")
        exp_dt = date.fromisoformat(exp) if exp and exp != "None" else None
        if (
            realized is not None
            and not (isinstance(realized, float) and np.isnan(realized))
            and spot_exp is not None
            and not (isinstance(spot_exp, float) and np.isnan(spot_exp))
            and exp_dt is not None
            and exp_dt <= backfill_end
        ):
            e["spot_at_expiry"] = float(spot_exp)
            e["realized_pnl"] = float(realized)
            e["outcome"] = 1 if float(realized) > 0 else 0
            e["settled"] = True
        entries.append(e)
    return entries


# ---------------------------------------------------------------------------
# Shared: (re)build calibration + MC + persist
# ---------------------------------------------------------------------------


def _refresh_reports(
    store: pb.PaperBookStore,
    conn,
    *,
    ledger: list[dict],
    equity_curve: list[dict],
    initial_capital: float,
    final_nav: float,
    window_start: str,
    window_end: str,
    as_of: str,
) -> dict:
    """Recompute calibration + MC bands from current state and persist them."""
    store.save_calibration(pb.compute_calibration(ledger, as_of=as_of))

    weights = _weights_from_ledger(ledger)
    per_name = _per_name_returns(conn, list(weights), window_start, window_end) if weights else {}
    mc = pb.build_mc_report(
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        realized_final_nav=final_nav,
        per_name_returns=per_name or None,
        weights=weights or None,
        label=store.book_name,
    )
    store.save_mc(mc)
    return mc


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------


def cmd_seed(args) -> int:
    from backtests.regression._common import run_backtest

    runner, conn, provider = _runner()
    frontier = _data_frontier(conn, list(args.tickers))
    print(f"[paper_book] data frontier ~{frontier}", flush=True)

    store = pb.PaperBookStore(args.book, create=True)
    backfill_end = date.fromisoformat(args.end)
    raw_dir = store.dir / "_backfill_raw"

    print(
        f"[paper_book] SEED backtest {args.start}->{args.end} "
        f"{len(args.tickers)} names cap=${args.capital:,.0f} friction={args.friction} "
        f"(caps-OFF PIT seed) -> {raw_dir}",
        flush=True,
    )
    run_backtest(
        capital=args.capital,
        tickers=list(args.tickers),
        start=args.start,
        end=args.end,
        seed=args.seed,
        friction_level=args.friction,
        top_n=len(args.tickers),
        max_new_per_day=args.max_new_per_day,
        dte_target=args.dte_target,
        delta_target=args.delta_target,
        output_dir=raw_dir,
    )

    with open(raw_dir / "tracker_state.json", encoding="utf-8") as f:
        state = json.load(f)
    rank_log = (
        pd.read_csv(raw_dir / "rank_log.csv")
        if (raw_dir / "rank_log.csv").exists()
        else pd.DataFrame()
    )

    # The seed backtest's serialised state becomes the book's continuation
    # tracker_state (re-armed on the first forward load).
    pb._write_json(store.tracker_state_path(), state)

    equity_curve = state.get("equity_curve", [])
    if not equity_curve:
        print("[paper_book] WARNING: seed produced no equity marks (empty book).", flush=True)
    final_nav = float(equity_curve[-1]["portfolio_value"]) if equity_curve else float(args.capital)

    # Panel history (phase=backfill) + SPY benchmark.
    history = pb.new_history(
        label=args.book, initial_capital=args.capital, backfill_end=backfill_end
    )
    history["points"] = pb.history_points_from_equity_curve(equity_curve, backfill_end=backfill_end)
    history["as_of"] = args.end
    history["final_nav"] = round(final_nav, 2)
    _apply_spy(history, conn)
    store.save_history(history)

    # Forecast ledger (calibration) + closed-trade ledger.
    ledger = _backfill_forecast_ledger(state, rank_log, backfill_end)
    store.save_forecast_ledger(ledger)
    store.save_closed_trades(state.get("closed_positions", []), as_of=args.end)

    mc = _refresh_reports(
        store,
        conn,
        ledger=ledger,
        equity_curve=equity_curve,
        initial_capital=float(args.capital),
        final_nav=final_nav,
        window_start=args.start,
        window_end=args.end,
        as_of=args.end,
    )

    store.save_meta(
        pb.build_meta(
            book_name=args.book,
            provider=provider,
            universe=list(args.tickers),
            window={"start": args.start, "end": args.end},
            backfill_end=backfill_end,
            data_frontier=frontier.isoformat() if frontier else None,
            config={
                "dte_target": args.dte_target,
                "delta_target": args.delta_target,
                "max_new_per_day": args.max_new_per_day,
                "friction": args.friction,
                "seed": args.seed,
                "capital": args.capital,
            },
            caps_armed=True,
        )
    )

    _print_summary(store, history, ledger, mc, tag="SEED (backfill)")
    return 0


# ---------------------------------------------------------------------------
# forward — one idempotent daily append
# ---------------------------------------------------------------------------


def cmd_forward(args) -> int:
    from backtests.regression._common import (
        _next_business_day,
        _spot_on_or_after,
        _tracker_step,
        friction_adjusted_premium,
        friction_open_cost,
    )
    from engine.wheel_tracker import PositionState

    runner, conn, provider = _runner()

    store = pb.PaperBookStore(args.book)
    if not store.tracker_state_path().exists():
        print(f"[paper_book] no seeded book '{args.book}' — run `seed` first.", flush=True)
        return 2

    history = store.load_history() or {}
    ledger = store.load_forecast_ledger()
    meta = store.load_meta() or {}
    cfg = meta.get("config", {})
    universe = meta.get("universe", [])
    capital = float(meta.get("initial_capital", cfg.get("capital", 100_000.0)))
    dte_target = int(cfg.get("dte_target", DEFAULT_DTE))
    delta_target = float(cfg.get("delta_target", DEFAULT_DELTA))
    max_new = int(cfg.get("max_new_per_day", DEFAULT_MAX_NEW_PER_DAY))
    window_start = (meta.get("window") or {}).get("start", args.as_of)
    backfill_end = pb._coerce_date(meta.get("backfill_end_date"))

    frontier = _data_frontier(conn, list(universe)) or date.today()
    as_of_req = date.fromisoformat(args.as_of) if args.as_of else date.today()
    as_of = min(as_of_req, frontier)
    if as_of != as_of_req:
        print(
            f"[paper_book] as_of {as_of_req} clamped to data frontier {as_of} "
            "(engine returns no candidates past the latest healthy bar)",
            flush=True,
        )

    # Idempotency + monotonicity guards.
    last_dt = pb.last_history_date(history)
    if last_dt is not None and as_of == last_dt:
        print(f"[paper_book] as_of {as_of} already appended — idempotent no-op.", flush=True)
        return 0
    if last_dt is not None and as_of < last_dt:
        print(
            f"[paper_book] as_of {as_of} is BEFORE the last point {last_dt} — "
            "forward-append must move strictly forward. Refusing.",
            flush=True,
        )
        return 2

    tracker = _load_armed_tracker(store, conn)
    exp_default = _next_business_day(as_of + timedelta(days=dte_target))

    # 1) Manage carried book: MTM + settle expirations + wheel into CCs.
    #    (_tracker_step appends one equity mark at as_of if positions are open.)
    _tracker_step(
        tracker=tracker,
        runner=runner,
        conn=conn,
        today=as_of,
        friction_level=args.friction,
        contracts=1,
        dte_target=dte_target,
        delta_target=delta_target,
        expiration_default=exp_default,
        PositionState=PositionState,
    )

    # 2) Settle due FORECAST-ledger entries (put-leg held-to-expiry outcomes).
    n_settled = pb.settle_due_forecasts(ledger, as_of, lambda t, exp: _spot_at_expiry(conn, t, exp))

    # 2.5) Record the day's equity mark BEFORE opening new positions. _tracker_step
    #      (step 1) marks only when it can price carried positions; on a flat book
    #      it doesn't, and marking AFTER the opens would book the just-credited
    #      premium as NAV without the offsetting short-put liability (a ~NAV/pt
    #      spike). Marking pre-open — with PIT spots for whatever IS open (empty
    #      dict on a genuinely flat book → cash NAV, the correct pre-open value) —
    #      matches run_backtest's pre-open convention and avoids the double-count.
    ec = tracker.equity_curve
    if not ec or pb._coerce_date(ec[-1].get("date")) != as_of:
        pre_prices: dict[str, float] = {}
        for tk, pos in tracker.positions.items():
            if pos.state != PositionState.NO_POSITION:
                s = _spot_on_or_after(conn, tk, as_of, max_lookahead_days=1)
                if s is not None:
                    pre_prices[tk] = s
        tracker.mark_to_market(as_of, pre_prices)

    # 3) Rank as_of + open up to N EV>0 (caps ARMED, full friction). Record the
    #    shipped forecast for each opened position.
    opened: list[str] = []
    refused = 0
    try:
        frame = runner.rank_candidates_by_ev(
            tickers=list(universe),
            dte_target=dte_target,
            delta_target=delta_target,
            contracts=1,
            top_n=len(universe),
            min_ev_dollars=-1e9,
            as_of=as_of.isoformat(),
            include_diagnostic_fields=True,
        )
    except Exception as exc:
        print(f"[paper_book] rank failed @ {as_of}: {exc!r}", flush=True)
        frame = None
    if frame is not None and len(frame):
        opens_today = 0
        for _, row in frame.iterrows():
            if opens_today >= max_new:
                break
            ev = float(row.get("ev_dollars", 0.0) or 0.0)
            if ev <= 0:
                continue
            t = str(row.get("ticker", ""))
            if t in tracker.positions and tracker.positions[t].state != PositionState.NO_POSITION:
                continue
            strike = float(row.get("strike", 0.0) or 0.0)
            prem = friction_adjusted_premium(float(row.get("premium", 0.0) or 0.0), args.friction)
            if prem <= 0 or strike <= 0:
                continue
            if tracker.available_buying_power() < strike * 100:
                continue
            n_log_before = len(tracker._ev_authority_log)
            ok = tracker.open_short_put(
                ticker=t,
                strike=strike,
                premium=prem,
                entry_date=as_of,
                expiration_date=exp_default,
                iv=float(row.get("iv", 0.0) or 0.0),
                prob_profit=(
                    float(row["prob_profit"]) if row.get("prob_profit") is not None else None
                ),
            )
            if ok:
                if args.friction == "full":
                    tracker.cash -= friction_open_cost(1, args.friction)
                opens_today += 1
                opened.append(t)
                ledger.append(
                    pb.new_forecast_entry(
                        entry_date=as_of.isoformat(),
                        ticker=t,
                        strike=strike,
                        premium=prem,
                        prob_profit=(
                            float(row["prob_profit"])
                            if row.get("prob_profit") is not None
                            else None
                        ),
                        expiration_date=exp_default.isoformat(),
                        phase=pb._phase_for(as_of, backfill_end),
                        ev_dollars=ev,
                    )
                )
            elif len(tracker._ev_authority_log) > n_log_before:
                # A D17 cap (R9/R10) refused this open — count it.
                refused += 1

    # 4) Build the day's point from the pre-open mark recorded in step 2.5 (the
    #    opens above credit cash but append no mark). Phase is derived — an
    #    as_of on/before backfill_end is still an in-sample point, not OOS.
    port = float(tracker.equity_curve[-1]["portfolio_value"])
    new_point = {
        "label": pb._label_for(as_of),
        "date": as_of.isoformat(),
        "port": round(port, 2),
        "spy": None,
        "premium": None,
        "phase": pb._phase_for(as_of, backfill_end),
    }
    pb.upsert_history_point(history, new_point)
    _apply_spy(history, conn)
    history["as_of"] = as_of.isoformat()
    history["final_nav"] = round(port, 2)
    store.save_history(history)

    # 5) Persist ledger, closed trades, calibration, MC, tracker continuation state.
    store.save_forecast_ledger(ledger)
    store.save_closed_trades(tracker.closed_positions, as_of=as_of.isoformat())
    mc = _refresh_reports(
        store,
        conn,
        ledger=ledger,
        equity_curve=tracker.equity_curve,
        initial_capital=capital,
        final_nav=port,
        window_start=window_start,
        window_end=as_of.isoformat(),
        as_of=as_of.isoformat(),
    )
    tracker.save(store.tracker_state_path())
    if meta:
        meta["last_forward_as_of"] = as_of.isoformat()
        meta["generated_at"] = pb.now_iso()
        store.save_meta(meta)

    print(
        f"[paper_book] FORWARD as_of={as_of} port=${port:,.0f} "
        f"opened={opened or '-'} settled_forecasts={n_settled} caps_refused={refused} "
        f"(phase=forward, backfill_end={backfill_end})",
        flush=True,
    )
    _print_summary(store, history, ledger, mc, tag="FORWARD")
    return 0


# ---------------------------------------------------------------------------
# report — recompute calibration + MC from current state (no engine drive)
# ---------------------------------------------------------------------------


def cmd_report(args) -> int:
    runner, conn, provider = _runner()
    store = pb.PaperBookStore(args.book)
    history = store.load_history()
    if history is None or not store.tracker_state_path().exists():
        print(f"[paper_book] no seeded book '{args.book}'.", flush=True)
        return 2
    with open(store.tracker_state_path(), encoding="utf-8") as f:
        state = json.load(f)
    ledger = store.load_forecast_ledger()
    meta = store.load_meta() or {}
    window = meta.get("window") or {}
    equity_curve = state.get("equity_curve", [])
    final_nav = (
        float(equity_curve[-1]["portfolio_value"])
        if equity_curve
        else float(meta.get("initial_capital", 0.0))
    )
    mc = _refresh_reports(
        store,
        conn,
        ledger=ledger,
        equity_curve=equity_curve,
        initial_capital=float(meta.get("initial_capital", 0.0)),
        final_nav=final_nav,
        window_start=window.get("start", history.get("as_of", "")),
        window_end=history.get("as_of", ""),
        as_of=history.get("as_of", ""),
    )
    _print_summary(store, history, ledger, mc, tag="REPORT")
    return 0


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------


def _print_summary(store, history: dict, ledger: list[dict], mc: dict, *, tag: str) -> None:
    pts = history.get("points") or []
    n_bf = sum(1 for p in pts if p.get("phase") == "backfill")
    n_fw = sum(1 for p in pts if p.get("phase") == "forward")
    settled = [e for e in ledger if e.get("outcome") is not None]
    calib = store.load_calibration() or {}
    pooled = calib.get("pooled", {})
    print(f"\n=== paper book [{store.book_name}] — {tag} ===", flush=True)
    print(f"  dir: {store.dir}", flush=True)
    print(f"  history points: {len(pts)}  (backfill={n_bf}, forward={n_fw})", flush=True)
    if pts:
        print(
            f"  final NAV: ${history.get('final_nav'):,.0f}  as_of={history.get('as_of')}",
            flush=True,
        )
    print(f"  forecast ledger: {len(ledger)} entries ({len(settled)} settled)", flush=True)
    if pooled.get("n"):
        print(
            f"  calibration pooled: n={pooled['n']} Brier={pooled['brier']} ECE={pooled['ece']}",
            flush=True,
        )
    if mc.get("status") == "ok":
        model = mc.get("model", {})
        rec = mc.get("reconciliation", {})
        tr = model.get("terminal_return_quantiles", {})
        print(
            f"  MC median terminal return: {model.get('median_return'):+.2%} "
            f"(p5 {tr.get('p5', float('nan')):+.2%} / p95 {tr.get('p95', float('nan')):+.2%})",
            flush=True,
        )
        print(
            f"  MC reconciliation: reconciled={rec.get('reconciled')} "
            f"gap={rec.get('median_abs_pct_gap', float('nan')):.2f}%",
            flush=True,
        )
    else:
        print(f"  MC bands: {mc.get('status')} ({mc.get('reason', '')})", flush=True)
    print(
        "  HONESTY: synthetic BSM fills (optimistic vs real); only the forward "
        "segment is true OOS; MC is a model. See meta.caveats.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def _common_flags(sp):
        sp.add_argument("--book", default=DEFAULT_BOOK, help="Book name (SIM-namespace subdir)")

    ps = sub.add_parser("seed", help="Seed the backfill via _common.run_backtest")
    _common_flags(ps)
    ps.add_argument("--tickers", nargs="+", required=True)
    ps.add_argument("--start", required=True)
    ps.add_argument("--end", required=True)
    ps.add_argument("--capital", type=float, default=200_000.0)
    ps.add_argument("--friction", default=DEFAULT_FRICTION, choices=["none", "bid_ask", "full"])
    ps.add_argument("--dte-target", type=int, default=DEFAULT_DTE, dest="dte_target")
    ps.add_argument("--delta-target", type=float, default=DEFAULT_DELTA, dest="delta_target")
    ps.add_argument(
        "--max-new-per-day", type=int, default=DEFAULT_MAX_NEW_PER_DAY, dest="max_new_per_day"
    )
    ps.add_argument("--seed", type=int, default=pb.CANONICAL_SEED)
    ps.set_defaults(func=cmd_seed)

    pf = sub.add_parser("forward", help="One idempotent daily forward-append")
    _common_flags(pf)
    pf.add_argument(
        "--as-of",
        default=None,
        dest="as_of",
        help="Append date (default today; clamped to the data frontier)",
    )
    pf.add_argument("--friction", default=DEFAULT_FRICTION, choices=["none", "bid_ask", "full"])
    pf.set_defaults(func=cmd_forward)

    pr = sub.add_parser("report", help="Recompute calibration + MC from current state")
    _common_flags(pr)
    pr.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
