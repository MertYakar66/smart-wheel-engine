"""Anchor tests: real ranker rows feed real ``consume_ranker_row``.

PR #174 wired the production chain ``rank_candidates_by_ev`` →
``issue_ev_authority_token`` → ``open_short_put(current_ev_dollars=...)``
via :meth:`engine.wheel_tracker.WheelTracker.consume_ranker_row`. Its
test file ``tests/test_decision_layer_wiring.py`` pinned the contract
on hand-built ``_ev_row(...)`` dicts that *claim* to mirror the
``rank_candidates_by_ev`` tail without any test that backs the claim —
the canonical #166 B1 pattern at small scale, surfaced as the headline
realism finding in Terminal C's Phase 1 audit of #174.

This file closes that gap. Every test pulls a row from a real
:class:`~engine.wheel_runner.WheelRunner` call against the real
:class:`~engine.data_connector.MarketDataConnector` and the real
Bloomberg CSVs (``sp500_ohlcv.csv`` / ``sp500_vol_iv_full.csv``),
then feeds the unmodified row into ``consume_ranker_row``. If the
ranker's row schema drifts away from what ``consume_ranker_row``
expects, every test here fails immediately.

Tickers: AAPL is positive-EV post-#179 (IV-PIT fix landed
2026-05-24), META is negative-EV at the same ``as_of=2026-03-20``
(the CSV cutoff per ``CLAUDE.md §4``). The +EV / -EV pair lets us
exercise both the happy-path and the D16 refusal branch on real
data.

NAV is $10M for all positive-path tests — matches
:class:`tests.test_decision_layer_wiring.TestD17HardBlocks`'
$10M fixture: D17's portfolio-delta cap scales at $300 per $100k NAV,
and a 35-DTE ~25-delta AAPL short put's dollar delta (~$8k) needs the
larger NAV to clear the cap structurally.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.wheel_runner import WheelRunner
from engine.wheel_tracker import EVAuthorityRefused, WheelTracker

_AS_OF = "2026-03-20"  # CSV cutoff per CLAUDE.md §4
_ENTRY_DATE = date(2026, 3, 20)
_AAPL = "GOOGL"  # positive-EV at _AS_OF post-#179 + F4-Fix-B1; was AAPL pre-F4-fix but AAPL's HMM at 2026-03-20 fires bear=0.61 -> widening flips EV negative. GOOGL HMM is normal -> positive EV preserved.
_META = "META"  # negative-EV at _AS_OF post-#179
_NAV_LARGE = 10_000_000.0  # clears D17 portfolio-delta cap


# ----------------------------------------------------------------------
# Shared real-ranker helpers
# ----------------------------------------------------------------------
def _real_ranker_row(ticker: str, *, min_ev_dollars: float = 0.0) -> dict:
    """Pull the top row from a real :meth:`rank_candidates_by_ev` call.

    Always uses the real :class:`MarketDataConnector` (default in
    ``WheelRunner.__init__``) and the real Bloomberg CSVs. Returns
    the row as a plain dict so the test can pass it through
    ``consume_ranker_row`` unmodified.
    """
    runner = WheelRunner()
    df = runner.rank_candidates_by_ev(
        tickers=[ticker],
        top_n=1,
        as_of=_AS_OF,
        min_ev_dollars=min_ev_dollars,
        include_diagnostic_fields=True,
    )
    assert not df.empty, (
        f"expected a row for {ticker} at as_of={_AS_OF} with "
        f"min_ev_dollars={min_ev_dollars}; drops={df.attrs.get('drops', [])}"
    )
    return df.iloc[0].to_dict()


# ======================================================================
# 1. Schema completeness — the inoculation the audit said was missing
# ======================================================================
class TestRankerRowKeysCoverConsumeExpectations:
    """``consume_ranker_row`` reads a fixed set of keys from the row it
    receives. Pin that the ranker's actual output covers all of them —
    closes the #174 audit's headline realism gap (the hand-built
    ``_ev_row`` in ``test_decision_layer_wiring.py`` *claimed* schema
    compatibility without a test that proves it).
    """

    def test_real_ranker_row_carries_all_keys_consume_ranker_row_uses(self):
        row = _real_ranker_row(_AAPL)
        # The eight keys consume_ranker_row references when invoking
        # issue_ev_authority_token + open_short_put (per PR #174's
        # diff on engine/wheel_tracker.py). Schema drift on any of
        # these would silently break the production wire — this test
        # catches it.
        required_keys = {
            "ticker",
            "strike",
            "premium",
            "dte",
            "ev_dollars",
            "iv",
            "prob_profit",
            "distribution_source",
        }
        missing = required_keys - set(row.keys())
        assert not missing, (
            f"rank_candidates_by_ev row is missing keys consume_ranker_row "
            f"needs: {missing}. Schema drift between ranker and tracker."
        )

    def test_real_ranker_row_types_are_compatible_with_consume(self):
        """Beyond key presence: the type of each required field must match
        what consume_ranker_row expects (ticker str, numeric strike/premium/
        ev_dollars/prob_profit/iv, int dte, str distribution_source).
        """
        row = _real_ranker_row(_AAPL)
        assert isinstance(row["ticker"], str)
        assert isinstance(row["strike"], (int, float))
        assert isinstance(row["premium"], (int, float))
        assert isinstance(row["dte"], (int, float))  # numpy int is acceptable
        assert isinstance(row["ev_dollars"], (int, float))
        assert isinstance(row["iv"], (int, float))
        assert isinstance(row["prob_profit"], (int, float))
        assert isinstance(row["distribution_source"], str)


# ======================================================================
# 2. Happy path — positive-EV real row consumed end-to-end
# ======================================================================
class TestRealRankerHappyPath:
    """A real positive-EV row from ``rank_candidates_by_ev`` flows
    cleanly through ``consume_ranker_row`` and lands as a tracker
    position."""

    def test_real_ranker_top_row_consumes_successfully(self):
        row = _real_ranker_row(_AAPL)
        assert row["ev_dollars"] > 0, (
            f"expected AAPL positive-EV at as_of={_AS_OF} post-PR #179; "
            f"got {row['ev_dollars']}. Smoke baseline may have shifted."
        )
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        ok = tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE)
        assert ok is True
        assert _AAPL in tracker.positions
        # The opened position must reflect the row's strike / dte.
        pos = tracker.positions[_AAPL]
        assert pos.put_strike == row["strike"]
        # expiration_date defaults to entry_date + dte (per #174 audit).
        from datetime import timedelta

        assert pos.put_expiration_date == _ENTRY_DATE + timedelta(days=int(row["dte"]))


# ======================================================================
# 3. D16 refusal — negative-EV real row refused at issuance
# ======================================================================
class TestRealRankerNegativeEvRefused:
    """The D16 contract: negative-EV rows raise ``EVAuthorityRefused``
    at the token-issuance step. The hand-built version of this test
    exists in ``test_decision_layer_wiring.py``; this is the real-input
    version proving the contract holds when the row actually came from
    the ranker."""

    def test_real_negative_ev_row_raises_authority_refused(self):
        # min_ev_dollars=-1e9 keeps negatives; META negative at _AS_OF
        # per the smoke.
        row = _real_ranker_row(_META, min_ev_dollars=-1e9)
        assert row["ev_dollars"] < 0, (
            f"expected META negative-EV at as_of={_AS_OF} post-PR #179; "
            f"got {row['ev_dollars']}. Smoke baseline may have shifted."
        )
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        with pytest.raises(EVAuthorityRefused):
            tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE)
        # Nothing landed in the tracker.
        assert _META not in tracker.positions
        # The refusal is in the audit log with the right reason.
        assert any(
            entry.get("action") == "refuse_issue" and entry.get("reason") == "non_positive_ev"
            for entry in tracker._ev_authority_log
        )


# ======================================================================
# 4-5. Round-trips — consume then read the chain's downstream state
# ======================================================================
class TestRealRankerRoundTrips:
    """After ``consume_ranker_row`` lands a position, the tracker's
    downstream readers (``mark_to_market`` and
    ``portfolio_context_snapshot``) reflect it consistently."""

    def test_consume_then_mark_to_market_with_real_spot(self):
        row = _real_ranker_row(_AAPL)
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        assert tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE) is True

        # Mark at the same spot the ranker used (row["spot"] is the
        # PIT spot at as_of — production-realistic).
        spot = float(row["spot"])
        nav = tracker.mark_to_market(current_date=_ENTRY_DATE, prices={_AAPL: spot})
        # mark_to_market returns NAV as a float directly.
        assert isinstance(nav, float)
        # Reasonable bounds: a single 1-contract CSP entered today and
        # marked at the same spot moves NAV by at most a few thousand
        # dollars (credit minus mark-to-market liability). Allow ±$50k
        # of initial capital.
        assert nav > 0
        assert abs(nav - _NAV_LARGE) < 50_000.0, (
            f"NAV={nav:.2f} unreasonably far from initial capital "
            f"${_NAV_LARGE:.0f} after one CSP consumed + marked at "
            f"the entry spot"
        )

    def test_consume_then_portfolio_context_snapshot_reflects_position(self):
        row = _real_ranker_row(_AAPL)
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        assert tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE) is True
        spot = float(row["spot"])
        ctx = tracker.portfolio_context_snapshot(
            spot_prices={_AAPL: spot},
            today=_ENTRY_DATE,
        )
        assert len(ctx.held_option_positions) == 1
        leg = ctx.held_option_positions[0]
        assert leg["symbol"] == _AAPL
        assert leg["option_type"] == "put"
        assert leg["is_short"] is True
        assert leg["strike"] == row["strike"]


# ======================================================================
# 6. Empty universe — no row, no consume call, tracker unchanged
# ======================================================================
class TestEmptyRankerOutput:
    """An impossible ``min_ev_dollars`` threshold yields zero rows; no
    ``consume_ranker_row`` call is made; tracker state is unchanged.
    Pins the empty-universe handling on the real ranker."""

    def test_impossible_min_ev_returns_empty_then_no_position_changes(self):
        runner = WheelRunner()
        df = runner.rank_candidates_by_ev(
            tickers=[_AAPL, _META],
            top_n=10,
            as_of=_AS_OF,
            min_ev_dollars=1e9,  # impossible — every row gets filtered
            include_diagnostic_fields=True,
        )
        assert df.empty, f"expected empty ranker output at min_ev_dollars=1e9; got {len(df)} rows"
        # Tracker stays empty.
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        assert len(tracker.positions) == 0
        # No rows to consume — nothing to call. Sanity: cash unchanged.
        ctx = tracker.portfolio_context_snapshot(today=_ENTRY_DATE)
        assert ctx.held_option_positions == []
        assert ctx.nav == pytest.approx(_NAV_LARGE, abs=1.0)


# ======================================================================
# 7. Duplicate consume — same real row twice returns False second time
# ======================================================================
class TestDuplicateConsume:
    """Consuming the same real row twice: first call opens the position,
    second call returns False because ``open_short_put`` refuses a
    duplicate ticker. Pins behavior with a real ranker row (the
    hand-built version lives in ``test_decision_layer_wiring.py``)."""

    def test_duplicate_consume_of_real_row_returns_false_second_time(self):
        row = _real_ranker_row(_AAPL)
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        assert tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE) is True
        # Second attempt with a later entry_date — still the same ticker
        # so open_short_put rejects.
        from datetime import timedelta

        second = tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE + timedelta(days=1))
        assert second is False
        # Still exactly one position.
        assert len(tracker.positions) == 1


# ======================================================================
# 8. Strict mode — real row drives the full token-issue + consume chain
# ======================================================================
class TestStrictModeFullChain:
    """Strict mode (``require_ev_authority=True``) demands that
    ``open_short_put`` only accept rows arriving via a fresh
    ``issue_ev_authority_token``. ``consume_ranker_row`` is the helper
    that satisfies that contract. With a real positive-EV row, the full
    chain works."""

    def test_strict_mode_consume_real_row_logs_issue_and_consume(self):
        row = _real_ranker_row(_AAPL)
        tracker = WheelTracker(initial_capital=_NAV_LARGE, require_ev_authority=True)
        ok = tracker.consume_ranker_row(row, entry_date=_ENTRY_DATE)
        assert ok is True
        assert _AAPL in tracker.positions
        # Token issued + consumed → tokens dict drained (one-shot semantics).
        assert len(tracker._ev_authority_tokens) == 0
        actions = [entry.get("action") for entry in tracker._ev_authority_log]
        assert "issue" in actions
        assert "consume" in actions
