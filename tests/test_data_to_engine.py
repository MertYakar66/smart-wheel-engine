"""Data -> engine-output tests on the real Bloomberg CSVs (Phase 2 B).

Extends ``tests/test_audit_viii_real_data_smoke.py``: routes every probe through
``WheelRunner.rank_candidates_by_ev`` / ``rank_covered_calls_by_ev`` — no §2
bypass, never hand-builds a candidate around ``EVEngine.evaluate`` — pinned to
the data-supported frontier (deterministic, never ``today()``). Asserts the
committed data produces finite, banded, well-formed engine output, that thin /
post-seam / garbage inputs degrade gracefully (skip or block, never a tradeable
from bad data), and that the forward-distribution cascade + drop accounting are
correct.

Policy: a test that fails because the data is genuinely bad is a SUCCESS — we do
not soften. Confirmed defects are ``xfail(strict=True)`` + a linked issue.

Findings map to ``docs/DATA_ENGINE_AUDIT_2026-06-07.md`` (W-numbers).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector, normalize_ticker
from engine.wheel_runner import WheelRunner

try:
    from backtests.regression.universes import UNIVERSE_24
except Exception:  # pragma: no cover
    UNIVERSE_24 = ()

DATA_DIR = Path("data/bloomberg")
HAS_BLOOMBERG_DATA = (DATA_DIR / "sp500_ohlcv.csv").exists()

# Pinned, data-supported frontier (most-recent bar common to OHLCV & IV on main).
FRONTIER = "2026-06-04"
DTE = 35
DELTA = 0.25

# Cascade tier labels emitted by best_available_forward_distribution.
VALID_TIERS = {
    "empirical_non_overlapping",
    "empirical_overlapping",
    "block_bootstrap",
    "har_rv",
    "lognormal_fallback",
    "none",
}

# W6 split (audit §3c + issue #355 / #356). Backfill-defects = long real history
# truncated in the file; legit-recent = genuine IPO/spinoff/merger/re-ticker.
W6_BACKFILL_DEFECT = [
    "WMT",
    "KMB",
    "CPB",
    "DPZ",
    "PLTR",
    "VEEV",
    "COHR",
    "CASY",
    "LITE",
    "SATS",
    "VRT",
]
W6_LEGIT_RECENT = ["BNY", "FDXF", "SNDK", "SW", "PSKY", "Q"]
POST_SEAM_ONLY = ["BNY", "CASY", "COHR", "LITE", "SATS", "VEEV", "VRT"]
ALL_THIN = sorted(set(W6_BACKFILL_DEFECT) | set(W6_LEGIT_RECENT))

pytestmark = pytest.mark.skipif(
    not HAS_BLOOMBERG_DATA, reason="Bloomberg CSVs not available (data/bloomberg)"
)


@pytest.fixture(scope="module")
def runner():
    return WheelRunner()


@pytest.fixture(scope="module")
def frontier(runner):
    """Confirm the bundled data actually covers the pinned frontier; skip on a
    stale branch (e.g. data ending 2026-03-20) so the probe stays deterministic."""
    df = runner.connector.get_ohlcv("AAPL", start_date="2026-05-20", end_date="2026-06-30")
    if df.empty or str(df.index.max())[:10] < FRONTIER:
        pytest.skip(f"bundled data does not cover frontier {FRONTIER}")
    return FRONTIER


def _rank(runner, tickers, **kw):
    kw.setdefault("dte_target", DTE)
    kw.setdefault("delta_target", DELTA)
    kw.setdefault("top_n", len(tickers) + 10)
    kw.setdefault("min_ev_dollars", -1e9)
    kw.setdefault("include_diagnostic_fields", True)
    return runner.rank_candidates_by_ev(tickers=list(tickers), as_of=FRONTIER, **kw)


# ---------------------------------------------------------------------------
# Output sanity on a clean positive control (UNIVERSE_24)
# ---------------------------------------------------------------------------


def test_clean_universe_output_is_well_formed(runner, frontier):
    """Every produced row from the clean control set is finite (R1a), banded,
    and internally coherent."""
    frame = _rank(runner, UNIVERSE_24)
    assert len(frame) > 0, "clean control produced zero rows"

    for _, row in frame.iterrows():
        t = row["ticker"]
        # R1a: ev_dollars / ev_raw finite (no NaN/inf reaches output).
        assert math.isfinite(float(row["ev_dollars"])), f"{t}: non-finite ev_dollars"
        assert math.isfinite(float(row["ev_raw"])), f"{t}: non-finite ev_raw"
        # Premium and IV plausible (IV decimal: the ranker converts percent->decimal).
        assert float(row["premium"]) > 0, f"{t}: premium not > 0"
        iv = float(row["iv"])
        assert 0.0 < iv < 3.0, f"{t}: iv {iv} implausible (expected decimal in (0,3))"
        # prob_profit / prob_assignment are probabilities.
        pp = float(row["prob_profit"])
        assert 0.0 <= pp <= 1.0, f"{t}: prob_profit {pp} out of [0,1]"
        assert 0.0 <= float(row["prob_assignment"]) <= 1.0, f"{t}: prob_assignment out of [0,1]"
        # Short put is OTM: strike below spot (delta is not surfaced as a column,
        # so strike<spot + prob_assignment band stand in for delta in [-1,0]).
        assert float(row["strike"]) > 0 and float(row["strike"]) < float(row["spot"]), (
            f"{t}: short-put strike {row['strike']} not below spot {row['spot']}"
        )
        # Wilson CI coherence: when present, low <= prob_profit <= high and N>0.
        lo, hi, n = row["prob_profit_ci_low"], row["prob_profit_ci_high"], row["n_scenarios"]
        if pd.notna(lo) and pd.notna(hi):
            assert float(lo) <= pp <= float(hi), f"{t}: prob_profit outside its Wilson CI"
            assert pd.notna(n) and int(n) > 0, f"{t}: CI present but n_scenarios not > 0"
        # Distribution tier reported and valid.
        assert row["distribution_source"] in VALID_TIERS, (
            f"{t}: bad tier {row['distribution_source']}"
        )


def test_distribution_cascade_picks_expected_tier(runner, frontier):
    """A full-history liquid name uses the richest empirical tier; every produced
    row reports a valid, non-null cascade tier (distribution_source correct)."""
    frame = _rank(runner, ["AAPL", "MSFT", "JPM", "XOM", "UNH"])
    src = dict(zip(frame["ticker"], frame["distribution_source"], strict=False))
    assert src.get("AAPL") == "empirical_non_overlapping", f"AAPL tier {src.get('AAPL')}"
    assert frame["distribution_source"].notna().all()
    assert set(frame["distribution_source"]).issubset(VALID_TIERS)


# ---------------------------------------------------------------------------
# No silent drops — every requested name ranks OR carries a logged reason
# ---------------------------------------------------------------------------


def test_no_silent_drops_on_control(runner, frontier):
    """Pin the audit's accounting invariant: produced + dropped == requested and
    nothing vanishes (drops attrs are complete with {ticker,gate,reason})."""
    tickers = list(UNIVERSE_24)
    frame = _rank(runner, tickers)
    produced = set(frame["ticker"].astype(str))
    drops = list(frame.attrs.get("drops", []))
    dropped = {d["ticker"] for d in drops}
    for d in drops:
        assert {"ticker", "gate", "reason"} <= set(d), f"drop entry missing keys: {d}"
    requested = {normalize_ticker(t) for t in tickers}
    vanished = requested - produced - dropped
    assert not vanished, f"silently vanished (neither produced nor dropped): {sorted(vanished)}"
    assert len(produced) + len(dropped) >= len(requested)


# ---------------------------------------------------------------------------
# Graceful degradation: thin / post-seam never emit a tradeable
# ---------------------------------------------------------------------------


def test_thin_names_degrade_gracefully(runner, frontier):
    """All 17 thin (<504-bar) names — backfill-defect AND legit-recent — are
    blocked by the history/data gate, never emitting a tradeable from bad data."""
    frame = _rank(runner, ALL_THIN)
    produced = set(frame["ticker"].astype(str)) if len(frame) else set()
    drops = {d["ticker"]: d["gate"] for d in frame.attrs.get("drops", [])}
    leaked = [t for t in ALL_THIN if t in produced]
    assert not leaked, f"thin names produced a tradeable candidate: {leaked}"
    for t in ALL_THIN:
        assert drops.get(t) in {"history", "data"}, (
            f"{t} not gracefully gated (gate={drops.get(t)})"
        )


@pytest.mark.parametrize("ticker", W6_BACKFILL_DEFECT)
@pytest.mark.xfail(
    strict=True,
    reason="W6 backfill defect (#355): long-history name truncated in sp500_ohlcv.csv "
    "(<504 bars) — flips green when its pre-seam OHLCV is backfilled",
)
def test_blue_chip_history_is_complete(runner, ticker):
    """Long-history blue-chips SHOULD clear the 504-day gate. They currently
    don't (truncated file) — xfail(strict) tracks each backfill."""
    df = runner.connector.get_ohlcv(ticker, end_date=FRONTIER)
    assert len(df) >= 504, f"{ticker} has only {len(df)} OHLCV bars (< 504-day gate)"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_as_of_twice(runner, frontier):
    """Same as_of twice -> identical ranked output (deterministic engine)."""
    a = _rank(runner, UNIVERSE_24)[
        ["ticker", "ev_dollars", "strike", "premium", "iv", "prob_profit"]
    ]
    b = _rank(runner, UNIVERSE_24)[
        ["ticker", "ev_dollars", "strike", "premium", "iv", "prob_profit"]
    ]
    pd.testing.assert_frame_equal(
        a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True
    )


# ---------------------------------------------------------------------------
# Dividends -> covered-call selection (R1 mechanism)
# ---------------------------------------------------------------------------


def test_in_window_exdiv_flows_into_cc_selection(runner, frontier):
    """R1 mechanism: an ex-dividend inside the CC window feeds the covered-call
    ex-div early-assignment EV. DIS has a 2026-06-30 ex-div ($0.75), 26 days out
    (< 35 DTE) — the CC ranker's expected_dividend must equal it."""
    nd = runner.connector.get_next_dividend("DIS", as_of=FRONTIER)
    assert nd is not None and nd["dividend_amount"] > 0, "DIS upcoming dividend missing"
    cc = runner.rank_covered_calls_by_ev(
        ticker="DIS",
        shares_held=100,
        as_of=FRONTIER,
        target_dtes=(DTE,),
        target_deltas=(DELTA,),
        top_n=5,
        min_ev_dollars=-1e9,
        include_diagnostic_fields=True,
    )
    assert cc is not None and len(cc) > 0, "DIS produced no covered-call candidate"
    row = cc.iloc[0]
    assert int(row["days_to_ex_div"]) < DTE, "ex-div should fall inside the CC window"
    assert float(row["expected_dividend"]) == pytest.approx(
        float(nd["dividend_amount"]), abs=1e-6
    ), "CC expected_dividend does not match the in-window ex-dividend (R1 mechanism broken)"


# ---------------------------------------------------------------------------
# W2 — dateless fundamentals/credit lookahead (confirmed defect)
# ---------------------------------------------------------------------------


_PIT_TICKER = "PITX"


def _write_pit_fundamentals_fixture(dirpath: Path) -> None:
    """Write synthetic ``fundamentals`` + ``credit_risk`` CSVs carrying TWO dated
    snapshots (2023 + 2024) for one ticker — deliberately ordered so the
    *dateless / no-op* path returns the **2024** row two different ways:

    * the 2024 row is written **first**, so today's ``df[...].iloc[0]`` returns it;
    * the 2024 row also has the **latest date**, so a "take the most recent" stub
      returns it too.

    A PIT-correct accessor asked for an ``as_of`` *between* the two snapshots must
    return the **2023** row. That is the only behaviour a genuine as_of-selection
    produces — a no-op ``as_of`` param (added just to satisfy the signature) cannot.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2024-12-31",
                "ticker": _PIT_TICKER,
                "eqy_dvd_yld_12m": 4.56,
                "gics_sector_name": "Sector2024",
            },
            {
                "date": "2023-12-31",
                "ticker": _PIT_TICKER,
                "eqy_dvd_yld_12m": 1.23,
                "gics_sector_name": "Sector2023",
            },
        ]
    ).to_csv(dirpath / "sp500_fundamentals.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2024-12-31",
                "ticker": _PIT_TICKER,
                "rtg_sp_lt_lc_issuer_credit": "CR2024",
                "altman_z_score": 9.99,
            },
            {
                "date": "2023-12-31",
                "ticker": _PIT_TICKER,
                "rtg_sp_lt_lc_issuer_credit": "CR2023",
                "altman_z_score": 1.11,
            },
        ]
    ).to_csv(dirpath / "sp500_credit_risk.csv", index=False)


@pytest.mark.xfail(
    strict=True,
    reason="W2 (#354): get_fundamentals/get_credit_risk are dateless snapshots with "
    "no as_of SELECTION, so historical backtests read the latest (2026) snapshot "
    "(structural lookahead). Flips red only when a PIT accessor that genuinely "
    "selects by as_of lands — a no-op as_of param cannot satisfy this assertion.",
)
def test_fundamentals_credit_are_point_in_time(tmp_path):
    """PIT *behaviour*, not signature shape.

    With two dated snapshots (2023 + 2024) for one ticker — ordered so the
    dateless/no-op path returns 2024 — a PIT-correct accessor asked for an
    ``as_of`` between them must return the **2023** row.

    Today both accessors take no ``as_of`` (``data_connector.py`` get_fundamentals
    / get_credit_risk), so the call raises ``TypeError`` → xfail. Crucially this
    can no longer be **false-greened** by bolting on a no-op ``as_of`` param: such
    a stub still returns the 2024 row (``iloc[0]`` / latest) and fails the 2023
    assertion. Only a genuine as_of SELECTION flips this green — which is the
    signal that #354 is truly fixed (then remove the xfail). See #354 and the
    memory note "assert behaviour, not shape".
    """
    _write_pit_fundamentals_fixture(tmp_path)
    conn = MarketDataConnector(data_dir=tmp_path)

    # as_of falls strictly between the 2023-12-31 and 2024-12-31 snapshots, so a
    # PIT accessor (date <= as_of, latest such) selects the 2023 row.
    fund = conn.get_fundamentals(_PIT_TICKER, as_of="2024-06-01")
    assert fund is not None and float(fund["dividend_yield"]) == pytest.approx(1.23), (
        "get_fundamentals must SELECT the 2023 snapshot for as_of=2024-06-01 "
        f"(got dividend_yield={None if fund is None else fund.get('dividend_yield')!r}); "
        "a no-op as_of that returns the 2024 row (4.56) does NOT satisfy PIT"
    )

    credit = conn.get_credit_risk(_PIT_TICKER, as_of="2024-06-01")
    assert credit is not None and credit["sp_rating"] == "CR2023", (
        "get_credit_risk must SELECT the 2023 snapshot for as_of=2024-06-01 "
        f"(got sp_rating={None if credit is None else credit.get('sp_rating')!r})"
    )


# ---------------------------------------------------------------------------
# Negative control — corrupt / truncated fixture must be rejected
# ---------------------------------------------------------------------------


def _write_fixture(dirpath: Path, specs: dict[str, dict]) -> None:
    """Write a minimal connector fixture (ohlcv + vol_iv) for synthetic tickers.

    ``specs[ticker] = {"n": rows, "price": float}``. OHLCV is written in the
    Bloomberg RAW column convention the connector expects (it renames
    open->high, high->close, close->open); flat bars (all four equal) keep the
    rename invariant valid, so any rejection is due to the spec, not malformed
    structure.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    oh_rows, iv_rows = [], []
    for ticker, spec in specs.items():
        dates = pd.bdate_range(end=FRONTIER, periods=spec["n"])
        price = spec["price"]
        for d in dates:
            ds = d.strftime("%Y-%m-%d")
            oh_rows.append(
                {
                    "date": ds,
                    "ticker": ticker,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
            iv_rows.append(
                {
                    "date": ds,
                    "ticker": ticker,
                    "hist_put_imp_vol": 25.0,
                    "hist_call_imp_vol": 25.0,
                    "volatility_30d": 25.0,
                    "volatility_60d": 25.0,
                    "volatility_90d": 25.0,
                    "volatility_260d": 25.0,
                }
            )
    pd.DataFrame(oh_rows).to_csv(dirpath / "sp500_ohlcv.csv", index=False)
    pd.DataFrame(iv_rows).to_csv(dirpath / "sp500_vol_iv_full.csv", index=False)


def test_negative_control_corrupt_data_is_rejected(tmp_path):
    """Truncated history and garbage prices must be REJECTED by the real ranker
    (no tradeable emitted), each with an explicit drop gate. Routes through
    rank_candidates_by_ev — no §2 bypass."""
    _write_fixture(
        tmp_path,
        {
            "THIN": {"n": 100, "price": 100.0},  # < 504 bars -> history gate
            "GARB": {"n": 600, "price": -1.0},  # non-positive prices -> data gate
        },
    )
    runner = WheelRunner(data_dir=tmp_path)
    frame = runner.rank_candidates_by_ev(
        tickers=["THIN", "GARB"],
        dte_target=DTE,
        delta_target=DELTA,
        top_n=10,
        min_ev_dollars=-1e9,
        as_of=FRONTIER,
        include_diagnostic_fields=True,
    )
    produced = set(frame["ticker"].astype(str)) if len(frame) else set()
    drops = {d["ticker"]: d["gate"] for d in frame.attrs.get("drops", [])}
    assert "THIN" not in produced and "GARB" not in produced, "corrupt data leaked a candidate"
    assert drops.get("THIN") == "history", f"THIN gate={drops.get('THIN')} (expected history)"
    assert drops.get("GARB") == "data", f"GARB gate={drops.get('GARB')} (expected data)"


# ---------------------------------------------------------------------------
# 5-ticker smoke (CLAUDE.md) stays green
# ---------------------------------------------------------------------------


def test_five_ticker_smoke_stays_green(runner, frontier):
    frame = _rank(runner, ["AAPL", "MSFT", "JPM", "XOM", "UNH"])
    assert len(frame) >= 1
    for col in ("ev_dollars", "iv", "premium"):
        assert frame[col].notna().all(), f"{col} has nulls in the 5-ticker smoke"


# ---------------------------------------------------------------------------
# W18 — the ranker's iv column IS the as-of PIT IV from the REAL connector
# (existing PIT tests use stub connectors with hand-fed IV dicts; this pins the
# wiring end-to-end on the bundled vol_iv file). 2026-06-09 data-test audit.
# ---------------------------------------------------------------------------


def test_ranker_iv_equals_real_pit_iv(runner, frontier):
    """W18: ``rank_candidates_by_ev(as_of=FRONTIER).iv`` for AAPL equals the PIT
    IV computed independently from the bundled file — ``mean(put,call)`` of the
    last ``get_iv_history(end_date=FRONTIER)`` row, ``/100``. Mirrors
    ``_resolve_pit_atm_iv`` exactly, so a connector-side PIT regression (wrong
    ``iloc[-1]`` row, snapshot instead of as-of) fails here. Tied to FRONTIER so
    a data refresh re-baselines the expectation in lockstep."""
    frame = _rank(runner, ["AAPL"])
    assert len(frame) == 1, "AAPL should produce exactly one row at the frontier"
    hist = runner.connector.get_iv_history("AAPL", end_date=FRONTIER)
    assert not hist.empty, "AAPL has no PIT IV history at the frontier"
    last = hist.iloc[-1]
    expected = (float(last["hist_put_imp_vol"]) + float(last["hist_call_imp_vol"])) / 2.0 / 100.0
    assert float(frame.iloc[0]["iv"]) == pytest.approx(expected, rel=1e-3), (
        f"ranker iv {float(frame.iloc[0]['iv'])} is not the as-of PIT IV {expected} "
        "(a snapshot-vs-PIT divergence would be ~10%+)"
    )


# ---------------------------------------------------------------------------
# W27 / #369 (LANDED 2026-06-27) — the fundamentals-FALLBACK IV path is now
# cleaned by the #363 IV band via ``get_fundamentals`` -> ``_clean_served_iv``,
# so the put/CC/strangle rankers' fallback IV is unambiguously PERCENT and the
# inline ``if iv>3.0: iv/=100`` conversion is always correct (a sub-3 garbage
# reading is NULLed, not accepted as a 200% decimal). The connector-side clean
# adds no decision-trio logic (the #363 / PANEL precedent).
# 2026-06-09 data-test audit (docs/DATA_TEST_AUDIT_2026-06-09.md, W27).
# ---------------------------------------------------------------------------


def test_fundamentals_fallback_iv_input_is_percent(runner, frontier):
    """W27a: the fallback IV input — ``get_fundamentals(...)['implied_vol_atm']``
    (Bloomberg ``30day_impvol_100.0%mny_df``) — is in PERCENT on the real file
    (so the ranker's inline ``/100`` is required to make it a decimal)."""
    f = runner.connector.get_fundamentals("AAPL")
    assert f is not None and f["implied_vol_atm"] is not None
    assert float(f["implied_vol_atm"]) > 3.0, (
        f"implied_vol_atm {f['implied_vol_atm']} not in percent units (expected tens)"
    )


def test_369_gate_cleans_fundamentals_iv(tmp_path):
    """W27b (#369, LANDED): the #363 IV band now ALSO cleans the
    fundamentals-fallback IV. ``get_fundamentals`` runs ``_clean_served_iv`` on
    ``implied_vol_atm``, so the same sub-3.0 value the vol_iv gate NULLs on a
    served ``vol_iv`` read is NULLed here too — the fallback IV is no longer
    accepted as a 200 % *decimal* by the rankers' inline heuristic. An in-band
    percent value passes through unchanged. (Was the W27b characterisation of the
    gap; flipped — not deleted — when #369 landed the connector-side clean.)"""
    pd.DataFrame(
        [
            {
                "date": "2025-01-02",
                "ticker": "ZZZ",
                "hist_put_imp_vol": 2.0,
                "hist_call_imp_vol": 2.0,
                "volatility_30d": 2.0,
                "volatility_60d": 2.0,
                "volatility_90d": 2.0,
                "volatility_260d": 2.0,
            }
        ]
    ).to_csv(tmp_path / "sp500_vol_iv_full.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "ZZZ",  # sub-3.0 garbage -> NULLed
                "30day_impvol_100.0%mny_df": 2.0,
                "volatility_30d": 2.0,
                "eqy_dvd_yld_12m": 0.0,
                "gics_sector_name": "Information Technology",
            },
            {
                "ticker": "YYY",  # valid percent -> passes through unchanged
                "30day_impvol_100.0%mny_df": 26.0,
                "volatility_30d": 25.0,
                "eqy_dvd_yld_12m": 0.0,
                "gics_sector_name": "Information Technology",
            },
        ]
    ).to_csv(tmp_path / "sp500_fundamentals.csv", index=False)
    conn = MarketDataConnector(data_dir=tmp_path)

    # The gate NULLs the 2.0 IV on the served vol_iv read...
    served = conn._load("vol_iv")
    assert pd.isna(served.iloc[0]["hist_put_imp_vol"]), "vol_iv gate should NULL the 2.0 IV"
    # ...and #369 now NULLs the SAME 2.0 through the fundamentals path too.
    f = conn.get_fundamentals("ZZZ")
    assert f is not None and pd.isna(f["implied_vol_atm"]), (
        f"#369: get_fundamentals must NULL the sub-3.0 fallback IV (got {f['implied_vol_atm']!r})"
    )
    # An in-band percent reading is preserved unchanged.
    g = conn.get_fundamentals("YYY")
    assert g is not None and float(g["implied_vol_atm"]) == 26.0, (
        f"#369: in-band percent IV must pass through (got {g['implied_vol_atm']!r})"
    )


# ---------------------------------------------------------------------------
# W15 — data -> EVEngine.evaluate -> EVResult SIGN controls.
# A sign inversion in the data->forward-dist->EV transform on real inputs passes
# every other real-data test (which assert finite + banded only). Pin the SIGN of
# a known +EV and a known -EV trade, FRONTIER-tied so the pending ev_mean
# re-baseline moves the magnitudes in lockstep but the sign must hold.
# 2026-06-09 data-test audit (docs/DATA_TEST_AUDIT_2026-06-09.md, W15).
# ---------------------------------------------------------------------------


def test_ev_dollars_sign_controls(runner, frontier):
    """W15: route real names through ``rank_candidates_by_ev`` (no §2 bypass) at
    the fixed FRONTIER and assert the SIGN of ev_dollars — not the magnitude:

    * XOM is a clean +EV cash-secured put (comfortable margin, ~+113 at lock);
    * UNH is structurally -EV at the same as_of (fat left tail, ~-77).

    Pins only the sign, so the pending ev_mean re-baseline moves the numbers but a
    sign INVERSION (the scariest silent failure for a decision engine) fails."""
    frame = _rank(runner, ["XOM", "UNH"])
    ev = dict(zip(frame["ticker"].astype(str), frame["ev_dollars"].astype(float), strict=False))
    assert "XOM" in ev and "UNH" in ev, f"sign controls did not both produce: {sorted(ev)}"
    assert ev["XOM"] > 0, f"XOM expected +EV at the frontier, got {ev['XOM']}"
    assert ev["UNH"] < 0, f"UNH expected -EV at the frontier, got {ev['UNH']}"


# ---------------------------------------------------------------------------
# W16 — real earnings file -> get_next_earnings -> EventGate -> EVEngine.evaluate
# lockout fires on a REAL near-earnings name (existing lockout tests use synthetic
# hand-built dates). Asserts the EXISTING wire — builds none.
# 2026-06-09 data-test audit (W16).
# ---------------------------------------------------------------------------


def test_real_earnings_event_lockout_fires(runner, frontier):
    """W16: JPM has a real earnings date inside the 35-DTE window at the FRONTIER.
    With the hard event gate ON it is dropped with ``gate=='event'``; with the
    gate OFF the same name produces. Confirms the real ``sp500_earnings.csv`` (date
    format + the ``'JPM UN'`` Bloomberg suffix) actually populates the gate, i.e.
    the §2 first-gate lockout is DATA-wired, not only synthetic-date-tested."""
    on = _rank(runner, ["JPM"])  # use_event_gate=True (default)
    produced_on = set(on["ticker"].astype(str)) if len(on) else set()
    drops_on = {d["ticker"]: d["gate"] for d in on.attrs.get("drops", [])}
    assert "JPM" not in produced_on, "JPM should be locked out by the event gate at the frontier"
    assert drops_on.get("JPM") == "event", f"JPM expected gate=='event', got {drops_on.get('JPM')}"

    off = _rank(runner, ["JPM"], use_event_gate=False)
    produced_off = set(off["ticker"].astype(str)) if len(off) else set()
    off_event_drops = [d["ticker"] for d in off.attrs.get("drops", []) if d.get("gate") == "event"]
    assert "JPM" in produced_off, "JPM should produce with the event gate off"
    assert not off_event_drops, f"no event-gate drops expected with the gate off: {off_event_drops}"


# ---------------------------------------------------------------------------
# W20 — real dividend_yield (eqy_dvd_yld_12m, PERCENT) flows into the BSM carry q.
# 2026-06-09 data-test audit (docs/DATA_TEST_AUDIT_2026-06-09.md, W20).
# ---------------------------------------------------------------------------


def test_dividend_yield_reaches_bsm_carry(tmp_path):
    """W20: the real ``eqy_dvd_yld_12m`` (PERCENT) reaches the BSM carry term.

    (a) real-data: ``get_fundamentals`` returns a high-yield name's yield in percent.
    (b) controlled: two synthetic fixtures identical EXCEPT ``dividend_yield`` (0% vs
        8%) produce a carry-correct difference — higher q lowers the forward, so the
        25-delta short-put strike is strictly LOWER. A *sane* ~80.5 strike at 8% also
        proves the unconditional percent->decimal ``/100`` fired (8.0 read as 800%
        would collapse the forward). Routes through ``rank_candidates_by_ev`` — no §2
        bypass; all other inputs (OHLCV, IV, rate fallback) are byte-identical."""
    # (a) real-data — the carry input is in percent
    f = WheelRunner().connector.get_fundamentals("CAG")
    assert f is not None and f["dividend_yield"] is not None and float(f["dividend_yield"]) > 3.0, (
        f"expected CAG dividend_yield in percent (>3), got "
        f"{None if f is None else f.get('dividend_yield')}"
    )

    # (b) controlled synthetic — identical except dividend_yield
    def _build(dirpath: Path, yld: float) -> None:
        n = 600
        dates = pd.bdate_range(end=FRONTIER, periods=n)
        rng = np.random.default_rng(0)
        price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
        pd.DataFrame(
            [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "ticker": "FBT",
                    "open": p,
                    "high": p,
                    "low": p,
                    "close": p,
                    "volume": 1_000_000,
                }
                for d, p in zip(dates, price, strict=False)
            ]
        ).to_csv(dirpath / "sp500_ohlcv.csv", index=False)
        pd.DataFrame(
            [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "ticker": "FBT",
                    "hist_put_imp_vol": 25.0,
                    "hist_call_imp_vol": 25.0,
                    "volatility_30d": 25.0,
                    "volatility_60d": 25.0,
                    "volatility_90d": 25.0,
                    "volatility_260d": 25.0,
                }
                for d in dates
            ]
        ).to_csv(dirpath / "sp500_vol_iv_full.csv", index=False)
        pd.DataFrame(
            [
                {
                    "ticker": "FBT",
                    "30day_impvol_100.0%mny_df": 25.0,
                    "volatility_30d": 25.0,
                    "eqy_dvd_yld_12m": yld,
                    "gics_sector_name": "Information Technology",
                }
            ]
        ).to_csv(dirpath / "sp500_fundamentals.csv", index=False)

    def _strike(yld: float) -> float:
        d = tmp_path / f"y{int(yld)}"
        d.mkdir()
        _build(d, yld)
        frame = WheelRunner(data_dir=d).rank_candidates_by_ev(
            tickers=["FBT"],
            as_of=FRONTIER,
            top_n=5,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        assert len(frame) == 1, f"FBT(yld={yld}) produced no row"
        return float(frame.iloc[0]["strike"])

    strike_zero = _strike(0.0)
    strike_high = _strike(8.0)
    assert strike_high < strike_zero, (
        "higher dividend yield should LOWER the carry-adjusted short-put strike "
        f"(real q reaches BSM): yld=8% strike {strike_high} not < yld=0% strike {strike_zero}"
    )
    # percent->decimal PIN (not just probed): a correct 8% carry leaves a sane
    # strike (~80.5). If the unconditional /100 were dropped, 8.0 would read as
    # 800%, collapsing the forward (F ~ 0.47*S) and crushing the 25-delta strike to
    # ~37 — which is ALSO < strike_zero, so the direction assert alone can't catch
    # it. The absolute floor distinguishes correct-8% from the percent/decimal bug.
    assert strike_high > 70.0, (
        f"8% carry should leave a sane strike (~80); got {strike_high} — the "
        "unconditional percent->decimal /100 may have been dropped (8.0 read as 800%)"
    )


# ---------------------------------------------------------------------------
# W25 — dividend epsilon-negatives stay immaterial through the connector path.
# 2026-06-09 data-test audit (W25); the producer clamp-to-0 is tracked as (D) #357.
# ---------------------------------------------------------------------------


def test_dividend_epsilon_negative_is_immaterial(tmp_path):
    """W25: the real file carries 82 epsilon-negative ``dividend_amount`` values
    (~-2.4e-14 float noise on Discontinued/Omitted rows). ``get_next_dividend`` does
    NOT clamp, so confirm the epsilon stays IMMATERIAL (>= -1e-9) and cannot surface
    as a spurious negative carry. The producer clamp-to-0 is (D) #357; this passes
    whether the value is the epsilon today or a clamped 0 after the fix."""
    pd.DataFrame(
        [
            {
                "ticker": "ZZZ",
                "declared_date": "2026-06-01",
                "ex_date": "2026-07-01",
                "record_date": "2026-07-02",
                "payable_date": "2026-07-15",
                "dividend_amount": -2.4245362661989844e-14,
                "dividend_frequency": "Quarterly",
                "dividend_type": "Discontinued",
            }
        ]
    ).to_csv(tmp_path / "sp500_dividends.csv", index=False)
    nd = MarketDataConnector(data_dir=tmp_path).get_next_dividend("ZZZ", as_of="2026-06-04")
    assert nd is not None, "synthetic upcoming dividend not found"
    assert float(nd["dividend_amount"]) >= -1e-9, (
        f"epsilon-negative dividend leaked as material: {nd['dividend_amount']} (< -1e-9)"
    )


# ---------------------------------------------------------------------------
# Covered-call ranker real-data coverage (the wheel's 2nd leg) — W29/W30/W31/W32
# (2026-06-09 data-test audit round 2; docs/DATA_TEST_AUDIT_2026-06-09.md). The CC
# ranker was real-data-starved vs the put side (its only real-data assertion was the
# single DIS ex-div row); these mirror the put-side well-formed/W15/W16 coverage.
# ---------------------------------------------------------------------------


def _rank_cc(runner, ticker, **kw):
    kw.setdefault("shares_held", 100)
    kw.setdefault("target_dtes", (DTE,))
    kw.setdefault("target_deltas", (DELTA,))
    kw.setdefault("top_n", 10)
    kw.setdefault("min_ev_dollars", -1e9)
    kw.setdefault("include_diagnostic_fields", True)
    return runner.rank_covered_calls_by_ev(ticker=ticker, as_of=FRONTIER, **kw)


def test_cc_clean_universe_output_is_well_formed(runner, frontier):
    """W29: every produced covered-call row on real data is finite (R1a), banded, and
    coherent — the CC analogue of test_clean_universe_output_is_well_formed. The CC
    ranker's only prior real-data assertion was the single DIS ex-div row."""
    frames = []
    for t in UNIVERSE_24:
        cc = _rank_cc(runner, t)
        if len(cc):
            frames.append(cc)
    assert frames, "no covered-call rows produced across the clean control set"
    frame = pd.concat(frames, ignore_index=True)
    for _, row in frame.iterrows():
        t = row["ticker"]
        assert math.isfinite(float(row["ev_dollars"])), f"{t}: non-finite ev_dollars"
        assert math.isfinite(float(row["ev_per_day"])), f"{t}: non-finite ev_per_day"
        assert float(row["premium"]) > 0, f"{t}: premium not > 0"
        iv = float(row["iv"])
        assert 0.0 < iv < 3.0, f"{t}: iv {iv} implausible (expected decimal in (0,3))"
        pp = float(row["prob_profit"])
        assert 0.0 <= pp <= 1.0, f"{t}: prob_profit {pp} out of [0,1]"
        assert 0.0 <= float(row["prob_assignment"]) <= 1.0, f"{t}: prob_assignment out of [0,1]"
        # Covered call: strike ABOVE spot (OTM call) — the mirror of the short put.
        assert float(row["strike"]) > float(row["spot"]) > 0, (
            f"{t}: CC strike {row['strike']} not above spot {row['spot']}"
        )
        lo, hi, n = row["prob_profit_ci_low"], row["prob_profit_ci_high"], row["n_scenarios"]
        if pd.notna(lo) and pd.notna(hi):
            assert float(lo) <= pp <= float(hi), f"{t}: prob_profit outside its Wilson CI"
            assert pd.notna(n) and int(n) > 0, f"{t}: CI present but n_scenarios not > 0"
        assert row["distribution_source"] in VALID_TIERS, (
            f"{t}: bad tier {row['distribution_source']}"
        )


def test_cc_real_earnings_event_lockout_fires(runner, frontier):
    """W30: the CC analogue of W16. JPM has a real earnings date inside the 49/63-DTE
    CC window at FRONTIER; with the event gate ON it is locked out (0 rows, every drop
    gate=='event'), with it OFF it produces. Pins the real sp500_earnings.csv ->
    get_next_earnings -> EventGate -> evaluate wire on the CC leg (CC event-gate tests
    use synthetic dates only)."""
    on = _rank_cc(runner, "JPM", target_dtes=(49, 63), use_event_gate=True)
    produced_on = set(on["ticker"].astype(str)) if len(on) else set()
    gates_on = [d.get("gate") for d in on.attrs.get("drops", [])]
    assert "JPM" not in produced_on, "JPM CC should be event-locked at the frontier"
    assert gates_on and all(g == "event" for g in gates_on), (
        f"JPM CC expected only event-gate drops, got {gates_on}"
    )
    off = _rank_cc(runner, "JPM", target_dtes=(49, 63), use_event_gate=False)
    assert len(off) > 0, "JPM CC should produce with the event gate off"
    off_event = [d for d in off.attrs.get("drops", []) if d.get("gate") == "event"]
    assert not off_event, f"no event-gate drops expected with the gate off: {off_event}"


def test_cc_exdiv_penalty_lowers_ev(runner, frontier):
    """W31: the in-window ex-dividend early-assignment penalty (call-only) LOWERS
    ev_dollars. The DIS real-data test pins only that expected_dividend reaches the
    row, not that it BITES — a regression dropping the penalty subtraction would pass
    it. Toggle the REAL DIS ex-div (from get_next_dividend) on a controlled call
    evaluation over the SAME forward distribution and assert the penalty is
    negative-signed. Calls EVEngine.evaluate directly (the ranker can't suppress the
    ex-div for an A/B) — no §2 bypass; evaluate is the authoritative EV path."""
    from engine.ev_engine import EVEngine, ShortOptionTrade

    nd = runner.connector.get_next_dividend("DIS", as_of=FRONTIER)
    assert nd is not None and float(nd["dividend_amount"]) > 0, "DIS upcoming dividend missing"
    div = float(nd["dividend_amount"])
    # 35-day log-returns with enough spread that some calls finish ITM (penalty bites).
    fwd = np.random.default_rng(0).normal(0.0, 0.10, 500)
    base = {
        "option_type": "call",
        "underlying": "DIS",
        "spot": 100.0,
        "strike": 105.0,
        "premium": 1.20,
        "dte": 35,
        "iv": 0.25,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.0,
    }
    eng = EVEngine()
    ev_no = eng.evaluate(
        ShortOptionTrade(**base, days_to_ex_div=None, expected_dividend=0.0),
        forward_log_returns=fwd,
    ).ev_dollars
    ev_div = eng.evaluate(
        ShortOptionTrade(**base, days_to_ex_div=20, expected_dividend=div),
        forward_log_returns=fwd,
    ).ev_dollars
    assert ev_div < ev_no, (
        "in-window ex-div should LOWER CC ev (early-assignment penalty): "
        f"with-exdiv {ev_div} not < without {ev_no}"
    )


def test_cc_ev_dollars_sign_controls(runner, frontier):
    """W32: the CC analogue of W15 — pin ev_dollars SIGN on real CC data. HD is a
    clean +EV covered call at FRONTIER; UNH and AAPL are -EV. Sign only (FRONTIER-tied)
    so the pending ev_mean re-baseline moves magnitudes in lockstep but a CC sign
    inversion fails. Gate off so earnings lockout doesn't suppress the controls."""
    hd = _rank_cc(runner, "HD", target_deltas=(0.20, 0.25, 0.30), use_event_gate=False)
    assert len(hd) > 0, "HD CC produced no rows"
    assert float(hd["ev_dollars"].max()) > 0, (
        f"HD expected a +EV covered call at the frontier, got max {hd['ev_dollars'].max()}"
    )
    for name in ("UNH", "AAPL"):
        neg = _rank_cc(runner, name, target_deltas=(0.20, 0.25, 0.30), use_event_gate=False)
        assert len(neg) > 0, f"{name} CC produced no rows"
        assert float(neg["ev_dollars"].max()) < 0, (
            f"{name} expected all -EV covered calls, got max {neg['ev_dollars'].max()}"
        )


# ---------------------------------------------------------------------------
# Full-universe sweep (slow) — pins the produced/dropped split
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_universe_no_silent_drops_and_split(runner, frontier):
    """Full connector universe: the accounting invariant holds and the
    produced/dropped split is pinned to the 2026-06-04 frontier (re-baseline on
    a data refresh — the pin going red is the signal, by design)."""
    universe = sorted(set(MarketDataConnector(str(runner.data_dir)).get_universe()))
    frame = _rank(runner, universe)
    produced = set(frame["ticker"].astype(str))
    drops = list(frame.attrs.get("drops", []))
    dropped = {d["ticker"] for d in drops}
    requested = {normalize_ticker(t) for t in universe}
    vanished = requested - produced - dropped
    assert not vanished, f"silent vanish: {sorted(vanished)}"
    gates = {d["gate"] for d in drops}
    assert {"event", "data", "history"} <= gates, f"expected gate set; got {gates}"
    # #3A: the corporate-action event lockout is DATE-SENSITIVE on the live
    # (as_of=None) path — the holding window is wall-clock, so which names have an
    # upcoming split/spinoff/special-cash in-window slides with the calendar.
    # Isolate it: the conserved, deterministic invariant is
    #   produced + corp-action-gated == 480  (the frontier tradeable count), and
    #   non-corp drops == 31  (the data/history/earnings baseline).
    # Pinned at frontier 2026-06-04 (see docs/DATA_ENGINE_AUDIT_2026-06-07.md).
    corp_drops = [d for d in drops if "corp_action" in str(d.get("reason", ""))]
    non_corp_drops = [d for d in drops if "corp_action" not in str(d.get("reason", ""))]
    assert len(produced) + len(corp_drops) == 480, (
        f"produced {len(produced)} + corp-gated {len(corp_drops)} != 480 "
        "(re-baseline if data refreshed)"
    )
    assert len(non_corp_drops) == 31, (
        f"non-corp drops {len(non_corp_drops)} != 31 (re-baseline if data refreshed)"
    )
    assert all(math.isfinite(float(v)) for v in frame["ev_dollars"]), (
        "non-finite ev_dollars at scale"
    )
    assert np.isfinite(pd.to_numeric(frame["ev_raw"], errors="coerce")).all(), "non-finite ev_raw"
    # W33: full BANDING at scale (extends the finite-only pin above) — every produced
    # row satisfies the same bands the UNIVERSE_24 well-formed test checks. Catches an
    # out-of-band tail/post-seam/fallback-tier row the 24-name control never exercises.
    # (2026-06-09 data-test audit round 2; probe at FRONTIER: 480 produced / 0 viol.)
    for _, row in frame.iterrows():
        t = row["ticker"]
        iv = float(row["iv"])
        assert 0.0 < iv < 3.0, f"{t}: iv {iv} out of (0,3) at scale"
        pp = float(row["prob_profit"])
        assert 0.0 <= pp <= 1.0, f"{t}: prob_profit {pp} out of [0,1] at scale"
        assert 0.0 <= float(row["prob_assignment"]) <= 1.0, (
            f"{t}: prob_assignment out of [0,1] at scale"
        )
        assert float(row["premium"]) > 0, f"{t}: premium not > 0 at scale"
        assert 0 < float(row["strike"]) < float(row["spot"]), f"{t}: strike not below spot at scale"
        assert row["distribution_source"] in VALID_TIERS, (
            f"{t}: bad tier {row['distribution_source']} at scale"
        )
