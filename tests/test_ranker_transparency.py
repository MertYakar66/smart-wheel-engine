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
# 1b. .attrs["drops_summary"] roll-up (S31 F1 discoverability closer)
# ======================================================================
class TestDropsSummaryAttr:
    """The S31 F1 fix: alongside .attrs['drops'] (which already exists),
    a trader-facing roll-up .attrs['drops_summary'] = {total_dropped,
    by_gate} is attached so a caller scanning the output sees at a
    glance 'N dropped — by_gate={...}' without iterating the full
    drops list. Mirrors the same shape on
    WheelTracker.suggest_rolls."""

    def test_summary_present_on_clean_run(self):
        """No drops → total_dropped=0, by_gate is empty."""
        df = _rank(_runner())
        assert "drops_summary" in df.attrs
        s = df.attrs["drops_summary"]
        assert s == {"total_dropped": 0, "by_gate": {}}

    def test_summary_present_on_all_gated_run(self):
        """An impossibly long history requirement gates out all 5; the
        summary correctly reports total_dropped=5 with by_gate={'history':5}."""
        df = _rank(_runner(), min_history_days=10_000_000)
        s = df.attrs["drops_summary"]
        assert s["total_dropped"] == 5
        assert s["by_gate"] == {"history": 5}

    def test_summary_present_on_ev_threshold_run(self):
        """An impossibly high EV floor gates out all 5 at the ev_threshold
        gate; the summary correctly classifies them."""
        df = _rank(_runner(), min_ev_dollars=1e9)
        s = df.attrs["drops_summary"]
        assert s["total_dropped"] == 5
        assert s["by_gate"] == {"ev_threshold": 5}

    def test_summary_counts_match_drops_list(self):
        """Across any run, sum of by_gate counts equals total_dropped
        and matches len(drops). No orphans, no double-counting."""
        for kwargs in [
            {},
            {"min_history_days": 10_000_000},
            {"min_ev_dollars": 1e9},
            {"empty": {"CCC"}},
        ]:
            df = _rank(_runner(empty=kwargs.pop("empty", None) or set()), **kwargs)
            s = df.attrs["drops_summary"]
            assert s["total_dropped"] == len(df.attrs["drops"]), (
                f"total_dropped mismatch with drops list at kwargs={kwargs}"
            )
            assert sum(s["by_gate"].values()) == s["total_dropped"], (
                f"by_gate sum mismatch with total_dropped at kwargs={kwargs}"
            )


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

    def test_hmm_realized_vol_and_return_columns_present(self):
        """S33 F4 regression: alongside `hmm_regime` / `hmm_multiplier`,
        every surviving row carries `hmm_realized_vol_252d_ann` and
        `hmm_realized_return_252d_ann` — the realized statistics of
        the same 252-day window the HMM fitted to. These disambiguate
        the "crisis" label (which means "high-vol regardless of
        direction") from the trader's "crashing" mental model. NaN /
        None is permitted when the HMM did not run (history too short
        or fit failed)."""
        df = _rank(_runner())
        assert "hmm_realized_vol_252d_ann" in df.columns
        assert "hmm_realized_return_252d_ann" in df.columns
        # On the _GBMConn fixture the HMM runs (default 1400 business
        # days >> the 200-required-day floor), so all rows have
        # non-None / non-NaN values.
        assert df["hmm_realized_vol_252d_ann"].notna().all()
        assert df["hmm_realized_return_252d_ann"].notna().all()

    def test_hmm_realized_vol_matches_manual_computation(self):
        """The columns must equal np.std(tail_252) * sqrt(252) and
        np.mean(tail_252) * 252 where tail_252 is the last 252 log
        returns of the OHLCV the HMM fitted to. This pins the math
        against an external reference, not engine-vs-itself."""
        runner = _runner()
        df = _rank(runner)
        # Pull AAA's synthetic OHLCV from the connector and recompute
        # the realized stats by hand.
        ohlcv = runner._connector.get_ohlcv("AAA")
        log_rets = np.diff(np.log(ohlcv["close"].values))
        tail_252 = log_rets[-252:]
        expected_vol = float(np.std(tail_252) * np.sqrt(252))
        expected_mean = float(np.mean(tail_252) * 252)
        row = df[df["ticker"] == "AAA"].iloc[0]
        assert row["hmm_realized_vol_252d_ann"] == pytest.approx(expected_vol, abs=1e-4)
        assert row["hmm_realized_return_252d_ann"] == pytest.approx(expected_mean, abs=1e-4)

    def test_hmm_realized_columns_absent_without_diagnostic_fields(self):
        """Like hmm_regime / hmm_multiplier, the disambiguation columns
        are diagnostics -- they must not appear when diagnostics are off."""
        df = _rank(_runner(), include_diagnostic_fields=False)
        assert "hmm_realized_vol_252d_ann" not in df.columns
        assert "hmm_realized_return_252d_ann" not in df.columns

    def test_sector_column_present_on_all_survivor_rows(self):
        """S31 F2 / F6 regression: every surviving row carries a `sector`
        column, sourced from engine.risk_manager.DEFAULT_SECTOR_MAP (the
        same map check_sector_cap aggregates by). Unknown tickers default
        to "Unknown" so the column is never NaN."""
        df = _rank(_runner())
        assert "sector" in df.columns
        # _GBMConn uses synthetic tickers (AAA..EEE) not in
        # DEFAULT_SECTOR_MAP, so the lookup falls back to "Unknown".
        # Whatever the value, it must be a populated string (no NaN).
        assert df["sector"].notna().all()
        assert df["sector"].apply(lambda s: isinstance(s, str) and len(s) > 0).all()

    def test_sector_column_uses_default_sector_map(self):
        """The sector lookup must use DEFAULT_SECTOR_MAP — same source as
        check_sector_cap — so the trader sees the GICS sector the gate
        would aggregate by. Using a known ticker ('AAPL' is GICS
        Information Technology) verifies the wiring."""
        from engine.risk_manager import DEFAULT_SECTOR_MAP

        runner = _runner(tickers=["AAPL"])
        df = _rank(runner, tickers=["AAPL"])
        if df.empty:
            return  # _GBMConn synth path may yield no survivor for AAPL alone
        expected = DEFAULT_SECTOR_MAP.get("AAPL", "Unknown")
        assert (df["sector"] == expected).all()

    def test_sector_column_is_present_even_without_diagnostic_fields(self):
        """sector is a CORE column (per-ticker fact, useful for
        portfolio reasoning regardless of diagnostic depth) -- it must
        appear when include_diagnostic_fields=False, unlike the
        hmm_*/skew_* diagnostic columns."""
        df = _rank(_runner(), include_diagnostic_fields=False)
        assert "sector" in df.columns

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
    """Section-2 invariants on the ranker's evaluate-call count.

    The original spirit: drop-capture diagnostics (the .attrs["drops"]
    feature) introduce zero extra EVEngine.evaluate calls. A drop logged
    at a pre-EV gate (history, data, event, ...) skips evaluate entirely;
    a drop logged at the post-EV gate (ev_threshold) reuses the single
    evaluate already paid for.

    F4 Fix B1+C update (2026-05-27): the ranker now optionally runs a
    SECOND EVEngine.evaluate when the regime widening factor > 1.0 (the
    HMM flagged a cold-tail regime), per the worst-of-two design in
    docs/F4_TAIL_RISK_DIAGNOSTIC.md §11. The two calls share trade /
    market_structure inputs; only forward_log_returns differs. The
    ranker surfaces the more conservative ev_dollars result.

    Updated bound: <= 2x evaluate calls per surviving candidate. Lower
    bound is still N (one per survivor). The drop-capture invariant
    is preserved — pre-EV drops still cost 0, and ev_threshold drops
    reuse the existing call(s).
    """

    def test_clean_run_evaluates_at_most_twice_per_survivor(self):
        """One EVEngine.evaluate call per surviving candidate baseline;
        up to one additional call per candidate when F4 Fix B1+C
        widening fires (the worst-of-two pair from
        docs/F4_TAIL_RISK_DIAGNOSTIC.md §11)."""
        with _spy_evaluate() as spy:
            df = _rank(_runner())
        assert len(df) == 5
        assert 5 <= spy.call_count <= 10, (
            f"evaluate called {spy.call_count} times for 5 survivors; "
            f"expected 5 (no widening) or up to 10 (F4 Fix B1+C "
            f"worst-of-two when widening fires on every survivor)."
        )

    def test_pre_evaluate_drop_triggers_no_evaluate(self):
        """A drop recorded at a pre-evaluation gate (history) costs zero
        EVEngine.evaluate calls -- the drop is logged without evaluating.
        """
        with _spy_evaluate() as spy:
            df = _rank(_runner(), min_history_days=10_000_000)
        assert df.empty
        assert len(df.attrs["drops"]) == 5
        assert spy.call_count == 0

    def test_post_evaluate_drop_reuses_existing_evaluate_calls(self):
        """A drop recorded at a post-evaluation gate (ev_threshold) reuses
        the EVResult(s) already produced -- recording the reason adds no
        further evaluate calls. Five tickers means at most 10 evaluate
        calls (5 primary + 5 alt under Fix B1+C worst-of-two when
        widening fires); recording the drops does not add an 11th.
        """
        with _spy_evaluate() as spy:
            df = _rank(_runner(), min_ev_dollars=1e9)
        assert df.empty
        assert len(df.attrs["drops"]) == 5
        assert 5 <= spy.call_count <= 10, (
            f"evaluate called {spy.call_count} times for 5 ev_threshold "
            f"drops; expected 5 (no widening) or up to 10 (F4 Fix B1+C "
            f"worst-of-two when widening fires on every survivor). "
            f"Drop-capture itself must not add an 11th call."
        )


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
        the regime multiplier.

        F4 Fix B1+C (2026-05-27): the ranker may invoke EVEngine.evaluate
        TWICE per survivor when widening fires (worst-of-two; see
        docs/F4_TAIL_RISK_DIAGNOSTIC.md §11). The captured list then
        holds two EVResults per ticker — primary and alt. The row's
        ev_raw comes from the WINNING (lower-ev_dollars) result, so the
        assertion checks that ev_raw matches ONE of the captured
        mean_pnl values for that ticker (rounded to 2dp).
        """
        ctx, captured = _capture_evaluate()
        with ctx:
            df = _rank(_runner())
        assert not df.empty
        captured_tickers = {t for t, _ in captured}
        assert captured_tickers == set(df["ticker"])
        for _, row in df.iterrows():
            ticker = row["ticker"]
            candidate_means = {round(res.mean_pnl, 2) for t, res in captured if t == ticker}
            assert row["ev_raw"] in candidate_means, (
                f"{ticker}: ev_raw={row['ev_raw']} not in any captured "
                f"mean_pnl: {sorted(candidate_means)} (F4 Fix B1+C worst-of-two)"
            )

    def test_ev_dollars_is_ev_raw_times_the_final_regime_multiplier(self):
        """The relationship ev_raw exists to expose. Inside the engine
        ev_dollars == mean_pnl * regime_multiplier, where
        regime_multiplier is the engine's *final* multiplier (the
        clamped overlay product, x heavy-tail penalty, x dealer mult) --
        not the runner's raw combined_regime_mult. The identity holds
        exactly on every captured EVResult, and within rounding on the
        row's exposed columns (ev_dollars == ev_raw *
        regime_multiplier)."""
        ctx, captured = _capture_evaluate()
        with ctx:
            df = _rank(_runner())
        assert not df.empty
        assert len(captured) >= 2
        # Per-EVResult identity holds regardless of which call won the
        # worst-of-two pick.
        for _ticker, res in captured:
            assert res.ev_dollars == pytest.approx(res.mean_pnl * res.regime_multiplier)
        # Row-level identity uses the WINNING res's regime_multiplier,
        # which the ranker surfaces directly in the regime_multiplier
        # column (verified by the dedicated test below).
        for _, row in df.iterrows():
            if abs(row["ev_raw"]) < 1e-6:
                continue
            implied = row["ev_dollars"] / row["ev_raw"]
            assert implied == pytest.approx(row["regime_multiplier"], abs=0.01), (
                f"{row['ticker']}: ev_dollars/ev_raw={implied:.4f} != "
                f"regime_multiplier column {row['regime_multiplier']:.4f}"
            )

    def test_regime_multiplier_column_matches_engine_final_multiplier(self):
        """S31 F9 regression: the ranker output must expose the engine's
        final regime_multiplier as a column, not just the four component
        multipliers (hmm / skew / news / credit) the trader would need
        to multiply manually. The combined value differs from the input
        product by the engine's clamp to [0.0, 1.25], heavy_tail_penalty
        if heavy_tail, and dealer_mult. Without this column, a trader
        verifying composition (ev_dollars / ev_raw) gets a value that
        does NOT equal hmm * skew * news * credit -- the S31 driver
        author hit exactly this confusion.

        F4 Fix B1+C: when widening fires, two EVResults per ticker may be
        captured (primary + alt). The row's regime_multiplier matches the
        WINNING one — checked by asserting equality with ONE of the
        captured regime_multiplier values for that ticker."""
        ctx, captured = _capture_evaluate()
        with ctx:
            df = _rank(_runner())
        assert not df.empty
        assert "regime_multiplier" in df.columns, (
            "regime_multiplier column must be present in ranker output"
        )
        for _, row in df.iterrows():
            ticker = row["ticker"]
            candidate_rms = [res.regime_multiplier for t, res in captured if t == ticker]
            assert candidate_rms, f"no captured res for {ticker}"
            assert any(
                row["regime_multiplier"] == pytest.approx(rm, abs=1e-4) for rm in candidate_rms
            ), (
                f"{ticker}: regime_multiplier column {row['regime_multiplier']:.6f} "
                f"matches none of the captured values {candidate_rms}"
            )
            # And ev_dollars / ev_raw reconstructs the same multiplier
            # (within the rounding of both ev_raw and ev_dollars to 2 dp).
            if abs(row["ev_raw"]) > 1e-6:
                implied = row["ev_dollars"] / row["ev_raw"]
                assert implied == pytest.approx(row["regime_multiplier"], abs=0.01)
