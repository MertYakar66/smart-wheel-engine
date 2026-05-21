"""Tests for ranker transparency: drop-reason diagnostics, the HMM
regime label, and the ev_raw column on
engine.wheel_runner.WheelRunner.rank_candidates_by_ev.

Usage tests S1, S9, S11 and S13 repeatedly logged three observability
gaps in the EV ranker:

  * Silent rejection. A candidate gated out (history, event,
    chain-quality, below-threshold EV, ...) simply vanished from the
    returned DataFrame. A trader could not tell "gated out" from "never
    a candidate". The fix records every drop as a structured
    {"ticker", "gate", "reason"} dict on the returned frame's
    .attrs["drops"].
  * Unlabelled regime. The HMM regime multiplier was emitted as a bare
    number (hmm_multiplier; 0.20 is an 80% EV cut) with no companion
    label, unlike dealer_regime / credit_regime which both carry one.
    The fix adds an hmm_regime column carrying the regime name the HMM
    selected.
  * EV before vs. after the regime cut. The ranker emitted ev_dollars
    (post-overlay EV) but not the pre-overlay EV, so the size of the
    regime adjustment was invisible. The fix adds an ev_raw column,
    sourced from EVResult.mean_pnl -- the mean scenario P&L the engine
    computes as ``ev_raw`` before the regime multiplier.

All are diagnostics-only (CLAUDE.md section 2): no gate, no multiplier
and no EV-authority change. These tests pin that -- survivor rows are
unchanged and the additions make zero extra EVEngine.evaluate calls.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.ev_engine import EVEngine
from engine.regime_hmm import GaussianHMM
from engine.wheel_runner import WheelRunner

# Overlays that need a live chain / network -- off for determinism. The
# HMM regime block is NOT one of these: it always runs inside the loop,
# so hmm_regime / hmm_multiplier stay populated here.
_OFFLINE = {
    "use_dealer_positioning": False,
    "use_news_sentiment": False,
    "use_credit_regime": False,
    "use_skew_dynamics": False,
}

# The drop-reason gate taxonomy. Every entry in .attrs["drops"] must
# carry one of these strings as its "gate".
_VALID_GATES = frozenset(
    {"data", "history", "event", "strike", "premium", "chain_quality", "ev_threshold"}
)

# The four regime names a 4-state HMM can select.
_HMM_REGIMES = frozenset({"crisis", "bear", "normal", "bull_quiet"})

_TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


# ----------------------------------------------------------------------
# Deterministic connector -- no data-on-disk dependency.
# ----------------------------------------------------------------------
class _GBMConn:
    """Deterministic multi-ticker connector for transparency tests.

    Each ticker gets its own seeded geometric-Brownian price path. Knobs
    let a test force a specific drop gate without touching the engine:

      * history={ticker: n_days} -- a shorter OHLCV history (drives the
        history gate).
      * empty={ticker, ...} -- get_ohlcv returns an empty frame (the
        "data" gate).
      * earnings={ticker: date} -- get_next_earnings reports an imminent
        announcement (the "event" gate).
    """

    def __init__(
        self,
        tickers,
        *,
        default_days: int = 1400,
        history: dict | None = None,
        empty=None,
        earnings: dict | None = None,
    ) -> None:
        self._tickers = list(tickers)
        self._empty = set(empty or ())
        self._earnings = dict(earnings or {})
        self._ohlcv: dict[str, pd.DataFrame] = {}
        history = history or {}
        for i, t in enumerate(tickers):
            n = int(history.get(t, default_days))
            idx = pd.date_range("2016-01-01", periods=n, freq="B")
            rng = np.random.default_rng(100 + i)
            base = 80.0 * (1.0 + 0.45 * i)
            close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n)))
            self._ohlcv[t] = pd.DataFrame({"close": close}, index=idx)

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        if ticker in self._empty:
            return pd.DataFrame()
        return self._ohlcv[ticker]

    def get_fundamentals(self, ticker: str) -> dict:
        return {"implied_vol_atm": 0.28, "volatility_30d": 0.25, "dividend_yield": 0.01}

    def get_risk_free_rate(self, as_of=None) -> float:
        return 0.05

    def get_next_earnings(self, ticker: str, as_of=None):
        d = self._earnings.get(ticker)
        return {"announcement_date": d} if d is not None else None

    def get_universe(self) -> list[str]:
        return list(self._tickers)


def _runner(tickers=_TICKERS, **conn_kwargs) -> WheelRunner:
    r = WheelRunner()
    r._connector = _GBMConn(tickers, **conn_kwargs)
    return r


def _rank(runner: WheelRunner, **extra) -> pd.DataFrame:
    """rank_candidates_by_ev with offline overlays and a permissive EV floor.

    Defaults: all five tickers, no top-N truncation, EV floor wide open
    so the ev_threshold gate is silent unless a test opts into it.
    """
    kw = dict(tickers=_TICKERS, top_n=50, min_ev_dollars=-1e9, **_OFFLINE)
    kw.update(extra)
    return runner.rank_candidates_by_ev(**kw)


def _spy_evaluate():
    """A patch context that counts EVEngine.evaluate calls and passes through."""
    from unittest.mock import patch

    original = EVEngine.evaluate

    def _pass_through(self, *a, **k):
        return original(self, *a, **k)

    return patch.object(EVEngine, "evaluate", autospec=True, side_effect=_pass_through)


def _capture_evaluate():
    """A patch context that captures every (ticker, EVResult) the engine
    returns, then passes the result through unchanged.

    Returns ``(patch_context, captured)`` -- ``captured`` is a list that
    fills with ``(ticker, EVResult)`` tuples as evaluate runs, letting a
    test read EVResult fields the ranker does not surface as columns
    (here: mean_pnl and the engine's final regime_multiplier).
    """
    from unittest.mock import patch

    original = EVEngine.evaluate
    captured: list = []

    def _cap(self, trade, *a, **k):
        res = original(self, trade, *a, **k)
        captured.append((trade.underlying, res))
        return res

    ctx = patch.object(EVEngine, "evaluate", autospec=True, side_effect=_cap)
    return ctx, captured


# ======================================================================
# 1. Drop-reasons exposed on .attrs["drops"]
# ======================================================================
class TestDropReasonsAttr:
    def test_drops_attr_is_present_and_a_list(self):
        df = _rank(_runner())
        assert "drops" in df.attrs
        assert isinstance(df.attrs["drops"], list)

    def test_clean_run_records_no_drops(self):
        """Five healthy tickers, wide-open EV floor: every name survives,
        so the drop log is empty."""
        df = _rank(_runner())
        assert len(df) == 5
        assert df.attrs["drops"] == []

    def test_history_gate_drop_is_recorded(self):
        """A known-gated case: an impossibly long history requirement
        gates out every ticker -- each appears in drops as gate=history."""
        df = _rank(_runner(), min_history_days=10_000_000)
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 5
        assert {d["ticker"] for d in drops} == set(_TICKERS)
        for d in drops:
            assert d["gate"] == "history"
            assert "history" in d["reason"] and "10000000" in d["reason"]

    def test_ev_threshold_drop_is_recorded(self):
        """A known-gated case: an impossibly high EV floor gates out
        every ticker after its EV evaluation -- gate=ev_threshold."""
        df = _rank(_runner(), min_ev_dollars=1e9)
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 5
        for d in drops:
            assert d["gate"] == "ev_threshold"
            assert "ev_dollars" in d["reason"] and "min_ev_dollars" in d["reason"]

    def test_data_gate_drop_is_recorded(self):
        """A ticker whose OHLCV pull comes back empty is dropped at the
        data gate; the other four still rank."""
        df = _rank(_runner(empty={"CCC"}))
        assert "CCC" not in set(df["ticker"])
        assert len(df) == 4
        ccc = [d for d in df.attrs["drops"] if d["ticker"] == "CCC"]
        assert len(ccc) == 1
        assert ccc[0]["gate"] == "data"

    def test_event_gate_drop_is_recorded(self):
        """A ticker with earnings two days out is locked out by the event
        gate -- recorded as gate=event, dropped from the output."""
        soon = pd.Timestamp(date.today() + timedelta(days=2))
        df = _rank(_runner(earnings={"BBB": soon}))
        assert "BBB" not in set(df["ticker"])
        bbb = [d for d in df.attrs["drops"] if d["ticker"] == "BBB"]
        assert len(bbb) == 1
        assert bbb[0]["gate"] == "event"
        assert isinstance(bbb[0]["reason"], str) and bbb[0]["reason"]

    def test_every_drop_entry_is_well_formed(self):
        """Across several forced-drop runs, every drop dict carries
        exactly {ticker, gate, reason}, with a known gate and non-empty
        string fields."""
        frames = [
            _rank(_runner(), min_history_days=10_000_000),
            _rank(_runner(), min_ev_dollars=1e9),
            _rank(_runner(empty={"AAA", "DDD"})),
        ]
        seen = 0
        for df in frames:
            for d in df.attrs["drops"]:
                assert set(d.keys()) == {"ticker", "gate", "reason"}
                assert d["gate"] in _VALID_GATES
                assert isinstance(d["ticker"], str) and d["ticker"]
                assert isinstance(d["reason"], str) and d["reason"]
                seen += 1
        assert seen >= 10

    def test_survivors_and_drops_are_disjoint(self):
        """No ticker is ever both a returned row and a logged drop."""
        for df in (_rank(_runner()), _rank(_runner(empty={"CCC"}))):
            survivors = set(df["ticker"]) if not df.empty else set()
            dropped = {d["ticker"] for d in df.attrs["drops"]}
            assert survivors.isdisjoint(dropped)

    def test_drops_attr_present_even_when_result_is_empty(self):
        """An all-gated run returns an empty frame that still carries the
        drop log -- the diagnostic survives the empty-DataFrame path."""
        df = _rank(_runner(), min_history_days=10_000_000)
        assert df.empty
        assert "drops" in df.attrs
        assert len(df.attrs["drops"]) == 5


# ======================================================================
# 2. CLAUDE.md section 2 -- survivor rows are byte-for-byte unchanged
# ======================================================================
class TestSurvivorsUnchanged:
    def test_survivor_rows_are_deterministic(self):
        """Two independent runs over identical inputs produce identical
        survivor rows -- drop-capture introduces no run-to-run drift."""
        a = _rank(_runner()).reset_index(drop=True)
        b = _rank(_runner()).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    def test_raising_min_ev_only_moves_names_to_drops(self):
        """Raising min_ev_dollars must only remove names -- recording
        them as ev_threshold drops -- never alter the rows that remain.

        For any ticker surviving both a wide-open and a raised EV floor,
        its row is byte-for-byte identical: min_ev_dollars is a filter,
        never an input to row construction. This is the section-2
        survivor-invariance check.
        """
        runner = _runner()
        df_all = _rank(runner, min_ev_dollars=-1e9)
        assert len(df_all) >= 2

        evs = sorted(df_all["ev_dollars"])
        thr = (evs[0] + evs[-1]) / 2.0  # strictly between min and max EV
        df_hi = _rank(runner, min_ev_dollars=thr)

        all_t = set(df_all["ticker"])
        hi_t = set(df_hi["ticker"])
        # The raised floor strictly removes names (top survives, bottom drops).
        assert hi_t < all_t
        removed = all_t - hi_t
        assert removed

        # Every removed name is logged as an ev_threshold drop.
        for t in removed:
            hits = [d for d in df_hi.attrs["drops"] if d["ticker"] == t]
            assert hits and all(d["gate"] == "ev_threshold" for d in hits)

        # Names surviving BOTH runs keep byte-identical rows.
        common = sorted(all_t & hi_t)
        assert common
        left = df_all[df_all["ticker"].isin(common)].sort_values("ticker").reset_index(drop=True)
        right = df_hi[df_hi["ticker"].isin(common)].sort_values("ticker").reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right)


# ======================================================================
# 3. hmm_regime label column
# ======================================================================
class TestHmmRegimeLabel:
    def test_hmm_regime_column_present_alongside_multiplier(self):
        df = _rank(_runner())
        assert not df.empty
        assert "hmm_regime" in df.columns
        assert "hmm_multiplier" in df.columns

    def test_hmm_regime_values_are_valid_regime_names(self):
        """With a full history the HMM runs; every label is one of the
        four regime names."""
        df = _rank(_runner())
        assert set(df["hmm_regime"]) <= _HMM_REGIMES

    def test_hmm_regime_absent_without_diagnostic_fields(self):
        """hmm_regime is a diagnostic column (it sits beside
        hmm_multiplier) -- it must not appear when diagnostics are off."""
        df = _rank(_runner(), include_diagnostic_fields=False)
        assert "hmm_regime" not in df.columns
        assert "hmm_multiplier" not in df.columns

    def test_hmm_regime_is_unknown_when_hmm_unavailable(self, monkeypatch):
        """When the HMM fit fails the ranker falls back to a neutral
        multiplier -- the label must report 'unknown', never a fabricated
        regime, and the pair must stay consistent (1.0 with unknown)."""

        class _BoomHMM:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("HMM unavailable")

        monkeypatch.setattr("engine.regime_hmm.GaussianHMM", _BoomHMM)
        df = _rank(_runner())
        assert not df.empty
        assert (df["hmm_regime"] == "unknown").all()
        assert df["hmm_multiplier"].eq(1.0).all()

    def test_position_multiplier_is_consistent_with_argmax_label(self):
        """The ranker derives the label as state_labels[argmax(probs)]
        and the multiplier as position_multiplier(probs) -- from the same
        posterior. A one-hot posterior pins the correspondence the prompt
        calls out: ~0.20 with crisis, 1.25 with bull_quiet.
        """
        hmm = GaussianHMM(n_states=4, random_state=42)
        rng = np.random.default_rng(0)
        hmm.fit(rng.normal(0.0, 0.01, 600))
        labels = hmm.fit_result.state_labels
        assert labels == ["crisis", "bear", "normal", "bull_quiet"]

        for idx, (name, mult) in enumerate(
            [("crisis", 0.2), ("bear", 0.5), ("normal", 1.0), ("bull_quiet", 1.25)]
        ):
            onehot = np.zeros(4)
            onehot[idx] = 1.0
            assert labels[int(np.argmax(onehot))] == name
            assert hmm.position_multiplier(onehot) == pytest.approx(mult)


# ======================================================================
# 4. CLAUDE.md section 2 -- drop-capture makes zero extra evaluate calls
# ======================================================================
class TestSection2ZeroExtraEvaluate:
    def test_clean_run_evaluates_once_per_survivor(self):
        """The baseline path: one EVEngine.evaluate call per surviving
        candidate, nothing more."""
        with _spy_evaluate() as spy:
            df = _rank(_runner())
        assert len(df) == 5
        assert spy.call_count == 5

    def test_pre_evaluate_drop_triggers_no_evaluate(self):
        """A drop recorded at a pre-evaluation gate (history) costs zero
        EVEngine.evaluate calls -- the drop is logged without evaluating.
        """
        with _spy_evaluate() as spy:
            df = _rank(_runner(), min_history_days=10_000_000)
        assert df.empty
        assert len(df.attrs["drops"]) == 5
        assert spy.call_count == 0

    def test_post_evaluate_drop_reuses_the_single_evaluate(self):
        """A drop recorded at a post-evaluation gate (ev_threshold) reuses
        the EVResult already produced -- five tickers means five evaluate
        calls, not ten. Recording the reason adds no second evaluation.
        """
        with _spy_evaluate() as spy:
            df = _rank(_runner(), min_ev_dollars=1e9)
        assert df.empty
        assert len(df.attrs["drops"]) == 5
        assert spy.call_count == 5


# ======================================================================
# 5. ev_raw column -- EV before the regime overlays
# ======================================================================
class TestEvRawColumn:
    def test_ev_raw_column_present_with_diagnostics(self):
        """ev_raw appears alongside the core ev_dollars whenever
        include_diagnostic_fields is on (the default), and every value
        is a finite number."""
        df = _rank(_runner())
        assert not df.empty
        assert "ev_raw" in df.columns
        assert "ev_dollars" in df.columns
        assert np.isfinite(df["ev_raw"].to_numpy()).all()

    def test_ev_raw_absent_without_diagnostic_fields(self):
        """ev_raw is diagnostic-gated exactly like hmm_multiplier -- it
        must not appear when diagnostics are off. ev_dollars, a core
        (non-diagnostic) field, still does: the core column set is
        unchanged by this fix."""
        df = _rank(_runner(), include_diagnostic_fields=False)
        assert "ev_raw" not in df.columns
        assert "ev_dollars" in df.columns

    def test_ev_raw_is_the_pre_regime_evresult_mean_pnl(self):
        """The ranker sources ev_raw from EVResult.mean_pnl -- the mean
        scenario P&L the engine computes as the local ``ev_raw`` before
        the regime multiplier. Captured EVResults pin the column to that
        field, per surviving ticker, rounded to 2dp."""
        ctx, captured = _capture_evaluate()
        with ctx:
            df = _rank(_runner())
        assert not df.empty
        res_by_ticker = dict(captured)
        assert set(res_by_ticker) == set(df["ticker"])
        for _, row in df.iterrows():
            res = res_by_ticker[row["ticker"]]
            assert row["ev_raw"] == round(res.mean_pnl, 2)

    def test_ev_dollars_is_ev_raw_times_the_final_regime_multiplier(self):
        """The relationship ev_raw exists to expose. Inside the engine
        ev_dollars == mean_pnl * regime_multiplier, where
        regime_multiplier is the engine's *final* multiplier (the
        clamped overlay product, x heavy-tail penalty, x dealer mult) --
        not the runner's raw combined_regime_mult. The identity holds
        exactly on the captured EVResults and survives, within rounding,
        into the ev_raw / ev_dollars output columns a caller sees."""
        ctx, captured = _capture_evaluate()
        with ctx:
            df = _rank(_runner())
        assert not df.empty
        assert len(captured) >= 2
        for ticker, res in captured:
            assert res.ev_dollars == pytest.approx(res.mean_pnl * res.regime_multiplier)
            r = df[df["ticker"] == ticker].iloc[0]
            assert r["ev_dollars"] == pytest.approx(r["ev_raw"] * res.regime_multiplier, abs=0.05)
