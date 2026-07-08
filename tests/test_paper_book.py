"""Tests for the forward paper-trading book (engine.paper_book + driver).

Coverage map (task Phase 4):
* §2 structural guard — engine.paper_book never imports the decision-layer trio.
* SIM-namespace isolation — every write lands under the SIM root and NEVER
  touches data_processed/ibkr/ (Dashboard terminal, CLAUDE.md §6).
* Calibration accumulator — wilson/reliability match scripts/ibkr_ev_calibration
  byte-for-byte; phase split + top-bin over-confidence flag; ECE hand-check.
* Held-to-expiry outcome — the quantity prob_profit models.
* Monte-Carlo bands — deterministic under seed 42 (reuses #483 sim_portfolio).
* History — phase labelling + idempotent same-day upsert.
* Forecast-ledger settlement — settles due, leaves pending, idempotent.
* Caps-armed refusal — a save/load-round-tripped armed tracker still refuses an
  over-concentrated open (the forward path's contract, task invariant 5).
* Driver pure helpers — backfill forecast-ledger join + copula weights.
* Slow lane — one end-to-end engine-driven seed+forward integration.
"""

from __future__ import annotations

import ast
import importlib.util
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from engine import paper_book as pb

_REPO = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO / "engine" / "paper_book.py"


# ---------------------------------------------------------------------------
# §2 structural guard — mirror engine.sim_portfolio's AST test
# ---------------------------------------------------------------------------


def test_paper_book_does_not_import_ev_trio():
    """engine.paper_book must never import the decision-layer trio
    (ev_engine / wheel_runner / candidate_dossier). It CONSUMES ranker output;
    it never ranks. The driver (scripts.run_paper_book) is the piece that drives
    the engine — that separation is what keeps this module §2-clean."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {
        "engine.ev_engine",
        "engine.wheel_runner",
        "engine.candidate_dossier",
        ".ev_engine",
        ".wheel_runner",
        ".candidate_dossier",
    }
    assert not (imported & forbidden), f"paper_book imports the EV trio: {imported & forbidden}"


def test_driver_reporting_half_is_the_trio_free_module():
    """Belt-and-suspenders: the trio-free half is engine.paper_book; the driver
    is allowed to import the ranker. Confirm paper_book exposes the calibration +
    MC surface without needing the trio at import time (it imported at module
    top for this test file with no trio side effect on THIS module's namespace)."""
    for name in (
        "compute_calibration",
        "build_mc_report",
        "settle_due_forecasts",
        "history_points_from_equity_curve",
        "PaperBookStore",
        "resolve_book_dir",
    ):
        assert hasattr(pb, name), name


# ---------------------------------------------------------------------------
# Calibration — verified reuse of scripts/ibkr_ev_calibration helpers
# ---------------------------------------------------------------------------


def _load_ibkr_ev_calibration_helpers():
    """Import wilson/reliability from the trio-importing calibration script by
    path (it imports EVEngine at module top, which is fine in a test)."""
    spec = importlib.util.spec_from_file_location(
        "ibkr_ev_calibration", _REPO / "scripts" / "ibkr_ev_calibration.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.wilson, mod.reliability


def test_wilson_matches_ibkr_ev_calibration():
    script_wilson, _ = _load_ibkr_ev_calibration_helpers()
    for k, n in [(0, 0), (0, 10), (5, 10), (10, 10), (3, 7), (97, 100)]:
        a = pb.wilson(k, n)
        b = script_wilson(k, n)
        assert (np.isnan(a[0]) and np.isnan(b[0])) or a == pytest.approx(b), (k, n)


def test_reliability_matches_ibkr_ev_calibration():
    _, script_reliability = _load_ibkr_ev_calibration_helpers()
    rng = np.random.default_rng(7)
    preds = rng.uniform(0, 1, size=300)
    outcomes = (rng.uniform(0, 1, size=300) < preds).astype(float)
    rows_a, brier_a, ece_a = pb.reliability(preds, outcomes)
    rows_b, brier_b, ece_b = script_reliability(preds, outcomes)
    assert brier_a == pytest.approx(brier_b)
    assert ece_a == pytest.approx(ece_b)
    assert rows_a == rows_b  # per-bin dicts identical (n, mean_pred, obs, Wilson CI)


def test_compute_calibration_phase_split_and_flag():
    entries = []
    # backfill: perfectly calibrated at p=0.8 (80% win) → low ECE contribution
    for i in range(100):
        entries.append({"prob_profit": 0.8, "outcome": 1 if i < 80 else 0, "phase": "backfill"})
    # forward: over-confident top bin — forecast 0.95 but only 50% win
    for i in range(40):
        entries.append({"prob_profit": 0.95, "outcome": 1 if i < 20 else 0, "phase": "forward"})
    # one pending (unsettled) — must be excluded
    entries.append({"prob_profit": 0.9, "outcome": None, "phase": "forward"})

    cal = pb.compute_calibration(entries, as_of="2026-06-04")
    assert cal["n_settled_total"] == 140
    assert cal["n_pending_total"] == 1
    assert cal["backfill"]["n"] == 100
    assert cal["forward"]["n"] == 40
    assert cal["pooled"]["n"] == 140
    # forward top bin observed ~0.5 vs predicted 0.95 → forward ECE clearly high
    assert cal["forward"]["ece"] > 0.4
    assert cal["top_bin_flag"]["known_overconfident"] is True
    assert cal["top_bin_flag"]["bin"] == "(0.90, 1.0]"
    # Wilson CI present on populated bins.
    populated = [b for b in cal["pooled"]["bins"] if b["n"] > 0]
    assert populated and all(b["ci_lo"] is not None and b["ci_hi"] is not None for b in populated)


def test_compute_calibration_empty_is_safe():
    cal = pb.compute_calibration([], as_of=None)
    assert cal["pooled"]["n"] == 0
    assert cal["pooled"]["brier"] is None
    assert cal["n_settled_total"] == 0


# ---------------------------------------------------------------------------
# Held-to-expiry outcome
# ---------------------------------------------------------------------------


def test_held_to_expiry_put_outcome():
    # OTM at expiry (spot above strike) → keep full premium → win.
    assert pb.held_to_expiry_put_pnl(100, 2.0, 105) == pytest.approx(200.0)
    assert pb.put_win(100, 2.0, 105) == 1
    # Deep ITM → assigned, big loss.
    assert pb.held_to_expiry_put_pnl(100, 2.0, 90) == pytest.approx((2.0 - 10.0) * 100)
    assert pb.put_win(100, 2.0, 90) == 0
    # Small ITM but still net-positive (premium covers intrinsic) → win.
    assert pb.put_win(100, 3.0, 98) == 1  # 3 - 2 = +1 → +100
    # Exactly break-even (premium == intrinsic) → not > 0 → loss.
    assert pb.put_win(100, 2.0, 98) == 0


# ---------------------------------------------------------------------------
# Monte-Carlo bands — deterministic under seed 42
# ---------------------------------------------------------------------------


def _synthetic_curve(n=120, start=100_000.0):
    rng = np.random.default_rng(0)
    vals, v = [], start
    for _ in range(n):
        v *= 1 + rng.normal(0.0004, 0.008)
        vals.append(v)
    return [
        {
            "date": (date(2026, 1, 1) + _days(i)).isoformat(),
            "portfolio_value": val,
            "cash": val * 0.5,
            "num_positions": 3,
        }
        for i, val in enumerate(vals)
    ]


def _days(i):
    from datetime import timedelta

    return timedelta(days=i)


def test_monte_carlo_bands_deterministic():
    curve = _synthetic_curve()
    r1 = pb.build_mc_report(equity_curve=curve, initial_capital=100_000.0, label="t")
    r2 = pb.build_mc_report(equity_curve=curve, initial_capital=100_000.0, label="t")
    assert r1["status"] == "ok"
    # Byte-identical bands across runs (seed 42 is baked in).
    assert r1["model"]["equity_bands"] == r2["model"]["equity_bands"]
    assert r1["model"]["median_return"] == r2["model"]["median_return"]
    # Fan is ordered p5 <= p50 <= p95 at the terminal step.
    b = r1["model"]["equity_bands"]
    assert b["p5"][-1] <= b["p50"][-1] <= b["p95"][-1]
    # Reconciliation block present + a bands-only convenience slice.
    assert "reconciliation" in r1 and r1["mc_bands"]["equity_bands"]


def test_monte_carlo_insufficient_history():
    short = _synthetic_curve(n=5)
    r = pb.build_mc_report(equity_curve=short, initial_capital=100_000.0, block_size=21, label="t")
    assert r["status"] == "insufficient_history"


# ---------------------------------------------------------------------------
# History — phase labelling + idempotent upsert
# ---------------------------------------------------------------------------


def test_history_points_phase_labelling():
    curve = [
        {"date": "2026-01-05", "portfolio_value": 100_000},
        {"date": "2026-02-05", "portfolio_value": 101_000},
        {"date": "2026-03-05", "portfolio_value": 102_000},
    ]
    pts = pb.history_points_from_equity_curve(curve, backfill_end=date(2026, 2, 5))
    phases = [p["phase"] for p in pts]
    assert phases == ["backfill", "backfill", "forward"]
    assert all(set(p) == {"label", "date", "port", "spy", "premium", "phase"} for p in pts)


def test_upsert_history_point_idempotent():
    hist = pb.new_history(label="t", initial_capital=100_000, backfill_end=date(2026, 1, 1))
    p1 = {
        "label": "Jan 5",
        "date": "2026-01-05",
        "port": 100,
        "spy": None,
        "premium": None,
        "phase": "forward",
    }
    assert pb.upsert_history_point(hist, p1) is True  # new day
    assert pb.upsert_history_point(hist, {**p1, "port": 200}) is False  # same day → replace
    assert len(hist["points"]) == 1 and hist["points"][-1]["port"] == 200
    p2 = {**p1, "date": "2026-01-06", "port": 300}
    assert pb.upsert_history_point(hist, p2) is True  # next day → append
    assert len(hist["points"]) == 2
    assert pb.last_history_date(hist) == date(2026, 1, 6)


def test_indexed_spy_series_anchors_to_base():
    closes = {"2026-01-05": 500.0, "2026-01-06": 505.0, "2026-01-07": 495.0}
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    idx = pb.indexed_spy_series(closes, dates, base_port=100_000.0)
    assert idx["2026-01-05"] == pytest.approx(100_000.0)  # anchored to base
    assert idx["2026-01-06"] == pytest.approx(100_000.0 * 505 / 500)
    # Missing close → omitted (benchmark gap), never fabricated.
    assert pb.indexed_spy_series({}, dates, 100_000.0) == {}


# ---------------------------------------------------------------------------
# Forecast-ledger settlement
# ---------------------------------------------------------------------------


def test_settle_due_forecasts():
    led = [
        pb.new_forecast_entry(
            entry_date="2026-01-02",
            ticker="AAPL",
            strike=100,
            premium=2.0,
            prob_profit=0.8,
            expiration_date="2026-02-06",
            phase="forward",
        ),
        pb.new_forecast_entry(
            entry_date="2026-01-02",
            ticker="MSFT",
            strike=200,
            premium=3.0,
            prob_profit=0.7,
            expiration_date="2026-03-06",
            phase="forward",
        ),
    ]
    spots = {("AAPL", date(2026, 2, 6)): 105.0, ("MSFT", date(2026, 3, 6)): 180.0}

    # As of 2026-02-10: only AAPL is due; MSFT expiry is in the future.
    n = pb.settle_due_forecasts(led, date(2026, 2, 10), lambda t, e: spots.get((t, e)))
    assert n == 1
    assert led[0]["settled"] is True and led[0]["outcome"] == 1  # OTM win
    assert led[1]["settled"] is False and led[1]["outcome"] is None

    # As of 2026-03-10: MSFT now due but spot lookup returns None → stays pending.
    n2 = pb.settle_due_forecasts(led, date(2026, 3, 10), lambda t, e: None)
    assert n2 == 0 and led[1]["settled"] is False

    # Spot now available → settles (ITM: 200 strike, spot 180 → loss).
    n3 = pb.settle_due_forecasts(led, date(2026, 3, 10), lambda t, e: spots.get((t, e)))
    assert n3 == 1 and led[1]["settled"] is True and led[1]["outcome"] == 0

    # Idempotent — nothing left to settle.
    assert pb.settle_due_forecasts(led, date(2026, 3, 10), lambda t, e: spots.get((t, e))) == 0


# ---------------------------------------------------------------------------
# SIM-namespace isolation (task invariant 4 / CLAUDE.md §6)
# ---------------------------------------------------------------------------


def test_resolve_book_dir_rejects_ibkr_and_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_SIM_DATA_DIR", str(tmp_path))
    assert pb.sim_root() == tmp_path
    with pytest.raises(ValueError):
        pb.resolve_book_dir("ibkr")
    with pytest.raises(ValueError):
        pb.resolve_book_dir("sub/ibkr")
    with pytest.raises(ValueError):
        pb.resolve_book_dir("../ibkr")
    with pytest.raises(ValueError):
        pb.resolve_book_dir("../../data_processed/ibkr")
    # A normal book name resolves under the SIM root.
    d = pb.resolve_book_dir("live_forward", create=True)
    assert tmp_path in d.parents and d.name == "live_forward"


def test_resolve_book_dir_allows_ibkr_ancestor(tmp_path, monkeypatch):
    """The 'ibkr' guard must scope to the tail BELOW the SIM root, not any
    ancestor — a SIM root that itself lives under an 'ibkr'-named dir (a plausible
    IBKR-integrated checkout) must still accept legit book writes."""
    sim = tmp_path / "ibkr" / "swe" / "data_processed" / "sim"
    monkeypatch.setenv("SWE_SIM_DATA_DIR", str(sim))
    d = pb.resolve_book_dir("live_forward", create=True)  # ancestor 'ibkr' is fine
    assert d.name == "live_forward"
    pb._write_json(d / "meta.json", {"x": 1})  # write under an 'ibkr' ancestor: fine
    assert (d / "meta.json").exists()
    # But an 'ibkr' component BELOW the root is still refused.
    with pytest.raises(ValueError):
        pb.resolve_book_dir("ibkr")


def test_write_json_refuses_outside_sim_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SWE_SIM_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        pb._write_json(tmp_path.parent / "escape.json", {"x": 1})
    ibkr_like = tmp_path / "ibkr" / "portfolio_snapshot.json"
    with pytest.raises(ValueError):
        pb._write_json(ibkr_like, {"x": 1})


def test_sim_isolation_never_touches_real_ibkr_dir(tmp_path, monkeypatch):
    """A full persistence cycle writes ONLY under the SIM root; the real
    data_processed/ibkr/ dir (Dashboard terminal) is provably untouched."""
    real_ibkr = _REPO / "data_processed" / "ibkr"
    before = _dir_fingerprint(real_ibkr)

    monkeypatch.setenv("SWE_SIM_DATA_DIR", str(tmp_path))
    store = pb.PaperBookStore("live_forward", create=True)
    store.save_history(pb.new_history(label="t", initial_capital=100_000, backfill_end=None))
    store.save_forecast_ledger(
        [
            pb.new_forecast_entry(
                entry_date="2026-01-02",
                ticker="AAPL",
                strike=100,
                premium=2.0,
                prob_profit=0.8,
                expiration_date="2026-02-06",
                phase="forward",
            ),
        ]
    )
    store.save_calibration(pb.compute_calibration([]))
    store.save_closed_trades([])
    store.save_meta(
        pb.build_meta(
            book_name="t",
            provider="MarketDataConnector",
            universe=["AAPL"],
            window={"start": "2026-01-01", "end": "2026-02-01"},
            backfill_end=None,
            data_frontier="2026-02-01",
            config={},
            caps_armed=True,
        )
    )
    store.save_mc({"status": "insufficient_history"})

    # Everything landed under the SIM root.
    written = list(tmp_path.rglob("*.json"))
    assert written, "no artifacts written"
    assert all(str(p).startswith(str(tmp_path)) for p in written)
    # The real IBKR dir is byte-for-byte unchanged.
    assert _dir_fingerprint(real_ibkr) == before


def _dir_fingerprint(d: Path):
    if not d.exists():
        return None
    return sorted(
        (str(p.relative_to(d)), p.stat().st_size, int(p.stat().st_mtime))
        for p in d.rglob("*")
        if p.is_file()
    )


# ---------------------------------------------------------------------------
# Pre-open equity mark — the forward point excludes just-credited premium
# ---------------------------------------------------------------------------


def test_pre_open_mark_excludes_new_premium():
    """cmd_forward records the day's equity mark BEFORE opening new positions, so
    the recorded NAV never double-counts a just-credited premium (whose offsetting
    short-put liability the open does not yet reflect). Pins the tracker-level
    invariants the fix relies on: a flat-book mark == cash NAV, and open_short_put
    appends no equity mark."""
    from engine.wheel_tracker import WheelTracker

    t = WheelTracker(initial_capital=100_000.0)
    t.mark_to_market(date(2026, 6, 1), {})  # flat book, pre-open
    pre = t.equity_curve[-1]["portfolio_value"]
    assert pre == pytest.approx(100_000.0, abs=1.0)  # cash NAV, no premium yet

    n_before = len(t.equity_curve)
    assert t.open_short_put("AAPL", 100.0, 3.0, date(2026, 6, 1), date(2026, 7, 6), 0.30) is True
    # The open credits cash but appends NO equity mark → the recorded point stays
    # the pre-open NAV (excludes the $300 premium), not a $100,300 cash spike.
    assert len(t.equity_curve) == n_before
    assert t.equity_curve[-1]["portfolio_value"] == pytest.approx(pre)
    assert t.cash > 100_000.0  # premium WAS credited to cash


# ---------------------------------------------------------------------------
# Caps-armed refusal — the forward path's contract (task invariant 5)
# ---------------------------------------------------------------------------


def test_forward_tracker_arms_and_refuses_over_concentration(tmp_path, monkeypatch):
    """Save a caps-off tracker, reload it via the driver's _load_armed_tracker,
    and confirm R10 (single-name >10% NAV) fires on the reloaded book — proving
    the forward path reproduces the gated live behaviour after a state
    round-trip (the enforce_* flags are NOT serialised, so re-arming matters)."""
    from engine.wheel_tracker import WheelTracker
    from scripts.run_paper_book import _load_armed_tracker

    monkeypatch.setenv("SWE_SIM_DATA_DIR", str(tmp_path))
    store = pb.PaperBookStore("live_forward", create=True)

    # A bare (caps-off) tracker, saved to the book's continuation-state path.
    bare = WheelTracker(initial_capital=100_000.0)
    assert bare.enforce_single_name_cap is False  # library default
    bare.save(store.tracker_state_path())

    armed = _load_armed_tracker(store, conn=None)
    assert armed.enforce_sector_cap is True
    assert armed.enforce_single_name_cap is True

    # $200 strike × 100 = $20k notional = 20% of $100k NAV > 10% → R10 refuses.
    opened = armed.open_short_put("AAPL", 200.0, 2.0, date(2026, 1, 2), date(2026, 2, 6), 0.30)
    assert opened is False
    assert "AAPL" not in armed.positions
    assert armed._ev_authority_log[-1]["reason"] == "single_name_breach"


# ---------------------------------------------------------------------------
# Driver pure helpers (no engine drive) — backfill join + copula weights
# ---------------------------------------------------------------------------


def test_backfill_forecast_ledger_join():
    from scripts.run_paper_book import _backfill_forecast_ledger

    state = {
        "closed_positions": [
            {"ticker": "AAPL", "entry_date": "2026-01-05", "put_premium": 200},
            {"ticker": "XOM", "entry_date": "2026-01-05", "put_premium": 0},  # not a put open
        ],
        "positions": {
            "MSFT": {"put_premium": 3.0, "state": "short_put", "put_entry_date": "2026-01-12"},
            "KO": {"put_premium": 0, "state": "no_position"},
        },
    }
    import pandas as pd

    rank_log = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "ticker": "AAPL",
                "strike": 100.0,
                "premium": 2.0,
                "prob_profit": 0.82,
                "ev_dollars": 55.0,
                "expiration_date": "2026-02-09",
                "spot_at_expiry": 104.0,
                "realized_pnl": 200.0,
            },
            {
                "date": "2026-01-12",
                "ticker": "MSFT",
                "strike": 200.0,
                "premium": 3.5,
                "prob_profit": 0.75,
                "ev_dollars": 40.0,
                "expiration_date": "2026-02-16",
                "spot_at_expiry": 190.0,
                "realized_pnl": (3.5 - 10.0) * 100,
            },
        ]
    )
    led = _backfill_forecast_ledger(state, rank_log, backfill_end=date(2026, 3, 1))
    by = {e["ticker"]: e for e in led}
    assert set(by) == {"AAPL", "MSFT"}  # XOM (premium 0) and KO (no_position) excluded
    assert (
        by["AAPL"]["prob_profit"] == 0.82 and by["AAPL"]["settled"] and by["AAPL"]["outcome"] == 1
    )
    assert by["MSFT"]["settled"] and by["MSFT"]["outcome"] == 0
    assert all(e["phase"] == "backfill" for e in led)


def test_backfill_ledger_leaves_unexpired_pending():
    import pandas as pd

    from scripts.run_paper_book import _backfill_forecast_ledger

    state = {
        "closed_positions": [],
        "positions": {
            "AAPL": {"put_premium": 2.0, "state": "short_put", "put_entry_date": "2026-02-25"}
        },
    }
    rank_log = pd.DataFrame(
        [
            {
                "date": "2026-02-25",
                "ticker": "AAPL",
                "strike": 100.0,
                "premium": 2.0,
                "prob_profit": 0.8,
                "ev_dollars": 30.0,
                "expiration_date": "2026-04-01",
                "spot_at_expiry": float("nan"),
                "realized_pnl": float("nan"),
            },
        ]
    )
    # backfill_end BEFORE expiry → not settled (pending, settles later in forward).
    led = _backfill_forecast_ledger(state, rank_log, backfill_end=date(2026, 3, 1))
    assert led[0]["settled"] is False and led[0]["outcome"] is None


def test_weights_from_ledger():
    from scripts.run_paper_book import _weights_from_ledger

    led = [
        {"ticker": "AAPL", "strike": 100.0},
        {"ticker": "AAPL", "strike": 120.0},
        {"ticker": "XOM", "strike": 90.0},
    ]
    w = _weights_from_ledger(led)
    # AAPL: median(100,120)=110 * 100 * 2 = 22000 ; XOM: 90 * 100 * 1 = 9000
    assert w["AAPL"] == pytest.approx(22000.0)
    assert w["XOM"] == pytest.approx(9000.0)


# ---------------------------------------------------------------------------
# Slow lane — one end-to-end engine-driven integration (seed + forward)
# ---------------------------------------------------------------------------


@pytest.mark.backtest_regression
def test_end_to_end_seed_and_forward_integration(tmp_path, monkeypatch):
    """Engine-driven smoke: seed a tiny backfill, do two forward-appends, and
    assert the honest boundary + idempotency + calibration accrual end to end.
    Marked backtest_regression (slow lane) — CI runs `-m "not backtest_regression"`.
    """
    monkeypatch.setenv("SWE_SIM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SWE_DATA_PROVIDER", "bloomberg")
    from scripts.run_paper_book import main as driver

    assert (
        driver(
            [
                "seed",
                "--book",
                "it",
                "--tickers",
                "AAPL",
                "MSFT",
                "XOM",
                "KO",
                "--start",
                "2026-02-02",
                "--end",
                "2026-04-01",
                "--capital",
                "100000",
            ]
        )
        == 0
    )
    store = pb.PaperBookStore("it")
    hist = store.load_history()
    assert hist and all(p["phase"] == "backfill" for p in hist["points"])
    n_backfill = len(hist["points"])

    # Forward append #1 → exactly one new forward point.
    assert driver(["forward", "--book", "it", "--as-of", "2026-04-15"]) == 0
    hist = store.load_history()
    assert len(hist["points"]) == n_backfill + 1
    assert hist["points"][-1]["phase"] == "forward"

    # Idempotent: same as_of → no growth.
    assert driver(["forward", "--book", "it", "--as-of", "2026-04-15"]) == 0
    assert len(store.load_history()["points"]) == n_backfill + 1

    # Backwards as_of is refused (monotonic forward-append).
    assert driver(["forward", "--book", "it", "--as-of", "2026-04-10"]) == 2

    # Calibration artifact exists and is phase-aware.
    cal = store.load_calibration()
    assert cal is not None and "backfill" in cal and "forward" in cal
