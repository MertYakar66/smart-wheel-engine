"""Regression tests for the ranker→tracker production wire.

Closes **C3** and **C4** from
``docs/END_TO_END_REVIEW_2026_05_25.md`` and TERMINAL_A_AUDIT.md
cross-cutting #4:

* **C3** — :meth:`engine.wheel_runner.WheelRunner.build_candidate_dossiers`
  now accepts a ``portfolio_context`` parameter and threads it
  through to :func:`engine.candidate_dossier.build_dossiers`. The
  underlying helper already accepted the parameter; the runner
  method just did not pass it. Without this, R7 (portfolio VaR)
  and R8 (stress + dealer regime) soft-warns were dormant on every
  call site that went through ``build_candidate_dossiers``,
  including the ``/api/tv/dossier`` HTTP endpoint.

* **C4** — :meth:`engine.wheel_runner.WheelRunner.consume_into_tracker`
  is the end-to-end production wire from ``rank_candidates_by_ev``
  through ``issue_ev_authority_token`` (D16 launch gate) and
  ``open_short_put`` with ``current_ev_dollars`` (D16 fresh-EV
  check + D17 portfolio-risk hard-blocks). Per-row outcomes are
  captured into a list of dicts; refusals at any stage are caught
  (the helper is loop-safe so one bad row doesn't abort a campaign).

These tests pin the contract end-to-end against real Bloomberg
fixtures at ``as_of=2026-03-20`` (the CSV cutoff per CLAUDE.md §4)
plus structural unit tests for refusal-path coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner
from engine.wheel_tracker import WheelTracker

_AS_OF = "2026-03-20"
_ENTRY_DATE = date(2026, 3, 20)
_NAV_LARGE = 10_000_000.0  # clears D17 portfolio-delta cap


# ======================================================================
# C3 — portfolio_context threading through build_candidate_dossiers
# ======================================================================


@dataclass
class _FakePortfolioContext:
    """Minimal :class:`PortfolioContext` shape sufficient for R7/R8.

    Lives in the test file so the test does not depend on the real
    portfolio-risk-gate construction path. The reviewer reads
    ``held_option_positions``, ``spot_prices``, ``nav``,
    ``returns_data``, ``correlation_matrix``, ``volatilities``,
    ``dealer_regime_by_ticker`` — all optional, all missing-skip safe.
    """

    held_option_positions: list = field(default_factory=list)
    spot_prices: dict = field(default_factory=dict)
    nav: float = 0.0
    returns_data: Any = None
    correlation_matrix: Any = None
    volatilities: Any = None
    dealer_regime_by_ticker: dict = field(default_factory=dict)


class TestPortfolioContextThreadedThroughBuildCandidateDossiers:
    """The new ``portfolio_context`` parameter on
    ``WheelRunner.build_candidate_dossiers`` is threaded into the
    underlying :func:`build_dossiers` and attached to every dossier.
    """

    def test_portfolio_context_attached_to_every_dossier(self):
        """When the caller passes a context, every produced
        :class:`CandidateDossier` carries it on
        ``dossier.portfolio_context``. Pre-fix the parameter did not
        exist on the runner method.
        """
        runner = WheelRunner()
        ctx = _FakePortfolioContext(nav=_NAV_LARGE)

        dossiers = runner.build_candidate_dossiers(
            tickers=["AAPL", "MSFT"],
            top_n=2,
            min_ev_dollars=-1e9,
            as_of=_AS_OF,
            portfolio_context=ctx,
        )
        assert dossiers, f"expected at least one dossier from 2-ticker run at as_of={_AS_OF}; got 0"
        for d in dossiers:
            assert d.portfolio_context is ctx, (
                f"dossier {d.ticker} did not receive portfolio_context — C3 threading broken"
            )

    def test_no_context_means_no_dossier_context_attached(self):
        """When ``portfolio_context=None`` (the default), the
        dossiers carry ``portfolio_context=None`` — preserves
        today's behaviour for callers that don't supply one. Pin
        the no-op default so a future "always-attach" refactor
        is caught.
        """
        runner = WheelRunner()
        dossiers = runner.build_candidate_dossiers(
            tickers=["AAPL", "MSFT"],
            top_n=2,
            min_ev_dollars=-1e9,
            as_of=_AS_OF,
            # portfolio_context omitted → defaults to None
        )
        assert dossiers
        for d in dossiers:
            assert d.portfolio_context is None, (
                f"dossier {d.ticker} got a portfolio_context when none was "
                f"passed; got {d.portfolio_context!r}"
            )

    def test_parameter_exposed_with_correct_default(self):
        """Schema check — the parameter is on the method and defaults
        to None. Catches a future refactor that drops the parameter.
        """
        import inspect

        sig = inspect.signature(WheelRunner.build_candidate_dossiers)
        assert "portfolio_context" in sig.parameters, (
            "WheelRunner.build_candidate_dossiers lost the "
            "portfolio_context parameter — C3 reverted?"
        )
        assert sig.parameters["portfolio_context"].default is None, (
            f"portfolio_context default must be None for backward "
            f"compatibility; got "
            f"{sig.parameters['portfolio_context'].default!r}"
        )


# ======================================================================
# C4 — consume_into_tracker end-to-end wire
# ======================================================================


class TestConsumeIntoTrackerHappyPath:
    """Real ranker + real tracker. Positive-EV rows from the ranker
    flow into the tracker as positions; outcomes mark ``opened=True``.
    """

    def test_real_positive_ev_rows_open_positions(self):
        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        outcomes = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE,
            rank_kwargs={
                "tickers": ["AAPL"],
                "as_of": _AS_OF,
                "top_n": 1,
            },
        )
        assert outcomes, f"expected at least one outcome for AAPL at as_of={_AS_OF}; got 0"
        opened_outcomes = [o for o in outcomes if o["opened"]]
        assert opened_outcomes, (
            f"expected at least one position to open; all refused: "
            f"{[(o['ticker'], o['refusal_reason']) for o in outcomes]}"
        )
        for o in opened_outcomes:
            assert o["refusal_reason"] is None
            assert o["ticker"] in tracker.positions, (
                f"outcome marked opened=True but {o['ticker']} not in tracker positions"
            )

    def test_outcome_schema_is_stable(self):
        """Pin the per-outcome dict shape so a future refactor that
        renames or drops fields is caught."""
        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        outcomes = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE,
            rank_kwargs={"tickers": ["AAPL"], "as_of": _AS_OF, "top_n": 1},
        )
        assert outcomes
        for o in outcomes:
            assert set(o.keys()) == {
                "ticker",
                "ev_dollars",
                "opened",
                "refusal_reason",
            }, f"outcome dict shape drift: keys={set(o.keys())}"
            assert isinstance(o["ticker"], str)
            assert isinstance(o["ev_dollars"], (int, float))
            assert isinstance(o["opened"], bool)
            assert o["refusal_reason"] is None or isinstance(o["refusal_reason"], str)


class TestConsumeIntoTrackerRefusalPaths:
    """Refusal paths at each stage of the wire produce a structured
    outcome rather than raising. The helper is loop-safe — one bad
    row does not abort the campaign.
    """

    def test_negative_ev_row_records_ev_authority_refused(self):
        """Lower ``min_ev_dollars`` so a negative-EV row survives the
        ranker filter, then verify the EVAuthorityRefused at the
        token-issuance step is caught into ``refusal_reason``.

        META is the canonical negative-EV row at as_of=2026-03-20
        post-PR #179 (the IV-PIT fix); same fixture as
        ``tests/test_consume_ranker_row_anchor.py``.
        """
        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        outcomes = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE,
            rank_kwargs={
                "tickers": ["META"],
                "as_of": _AS_OF,
                "top_n": 1,
                "min_ev_dollars": -1e9,
            },
        )
        # META might be event-blocked at this as_of — skip if no row.
        if not outcomes:
            pytest.skip(
                f"META produced no ranker row at as_of={_AS_OF} — likely "
                f"event-gated. Refusal path covered by the synthetic test."
            )
        # If a negative-EV row landed, the refusal should be
        # ev_authority_refused and nothing should be in the tracker.
        for o in outcomes:
            if o["ev_dollars"] <= 0:
                assert o["opened"] is False
                assert o["refusal_reason"] == "ev_authority_refused", (
                    f"negative-EV row should be EVAuthorityRefused at "
                    f"the launch gate; got {o['refusal_reason']!r}"
                )
        # Nothing landed in the tracker for this run.
        assert "META" not in tracker.positions

    def test_synthetic_negative_ev_row_records_ev_authority_refused(self):
        """Drive consume_into_tracker via a stubbed runner that
        returns a known-negative-EV row, independent of upstream
        data. Pins the EVAuthorityRefused → outcome translation
        deterministically.
        """
        from unittest.mock import patch

        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)

        negative_row = pd.DataFrame(
            [
                {
                    "ticker": "FAKE",
                    "spot": 100.0,
                    "strike": 95.0,
                    "premium": 1.5,
                    "dte": 35,
                    "iv": 0.22,
                    "ev_dollars": -25.0,  # forces EVAuthorityRefused
                    "ev_per_day": -0.71,
                    "prob_profit": 0.5,
                    "prob_assignment": 0.5,
                    "distribution_source": "empirical_non_overlapping",
                }
            ]
        )

        with patch.object(WheelRunner, "rank_candidates_by_ev", return_value=negative_row):
            outcomes = runner.consume_into_tracker(
                tracker,
                entry_date=_ENTRY_DATE,
                rank_kwargs={"top_n": 1, "min_ev_dollars": -1e9},
            )

        assert len(outcomes) == 1
        assert outcomes[0]["ticker"] == "FAKE"
        assert outcomes[0]["opened"] is False
        assert outcomes[0]["refusal_reason"] == "ev_authority_refused"
        assert "FAKE" not in tracker.positions

    def test_duplicate_ticker_records_tracker_rejected(self):
        """Open a position, then run consume_into_tracker again on
        the same ticker — the second consume returns False
        (open_short_put refuses duplicate ticker) and outcome
        records ``tracker_rejected``.
        """
        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)

        # First pass — should open.
        first = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE,
            rank_kwargs={"tickers": ["AAPL"], "as_of": _AS_OF, "top_n": 1},
        )
        opened = [o for o in first if o["opened"]]
        if not opened:
            pytest.skip(
                f"first pass produced no opens; outcomes={first}. Test "
                "premise broken — likely data shifted."
            )
        ticker_opened = opened[0]["ticker"]

        # Second pass — same ticker, second day; tracker should refuse
        # duplicate.
        second = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE + timedelta(days=1),
            rank_kwargs={"tickers": [ticker_opened], "as_of": _AS_OF, "top_n": 1},
        )
        assert second
        for o in second:
            assert o["opened"] is False
            assert o["refusal_reason"] == "tracker_rejected"
        # Exactly one position still in the tracker.
        assert len(tracker.positions) == 1

    def test_loop_continues_after_a_refusal(self):
        """A refusal on row N does not abort the loop — row N+1 still
        gets a consume attempt. Pin loop-safety via a mixed batch.
        """
        from unittest.mock import patch

        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)

        mixed_batch = pd.DataFrame(
            [
                {
                    "ticker": "BAD",
                    "spot": 100.0,
                    "strike": 95.0,
                    "premium": 0.01,
                    "dte": 35,
                    "iv": 0.22,
                    "ev_dollars": -10.0,  # refuses
                    "ev_per_day": -0.28,
                    "prob_profit": 0.5,
                    "prob_assignment": 0.5,
                    "distribution_source": "empirical_non_overlapping",
                },
                {
                    "ticker": "GOOD",
                    "spot": 100.0,
                    "strike": 95.0,
                    "premium": 1.5,
                    "dte": 35,
                    "iv": 0.22,
                    "ev_dollars": 42.0,  # opens
                    "ev_per_day": 1.20,
                    "prob_profit": 0.78,
                    "prob_assignment": 0.18,
                    "distribution_source": "empirical_non_overlapping",
                },
            ]
        )

        with patch.object(WheelRunner, "rank_candidates_by_ev", return_value=mixed_batch):
            outcomes = runner.consume_into_tracker(
                tracker,
                entry_date=_ENTRY_DATE,
                rank_kwargs={"top_n": 2, "min_ev_dollars": -1e9},
            )

        assert len(outcomes) == 2
        # Order preserved.
        assert outcomes[0]["ticker"] == "BAD"
        assert outcomes[1]["ticker"] == "GOOD"
        # BAD refused; GOOD opened.
        assert outcomes[0]["opened"] is False
        assert outcomes[0]["refusal_reason"] == "ev_authority_refused"
        assert outcomes[1]["opened"] is True
        assert outcomes[1]["refusal_reason"] is None
        # The GOOD position landed even though BAD refused first.
        assert "GOOD" in tracker.positions
        assert "BAD" not in tracker.positions


class TestConsumeIntoTrackerSchemaAndDefaults:
    """Method-level contracts: forces diagnostic fields, honours
    ``top_n_to_consume``, returns empty list on empty ranker output.
    """

    def test_empty_ranker_output_returns_empty_outcomes(self):
        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)
        outcomes = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE,
            rank_kwargs={
                "tickers": ["AAPL"],
                "as_of": _AS_OF,
                "top_n": 1,
                "min_ev_dollars": 1e9,  # impossible — every row filtered
            },
        )
        assert outcomes == []
        assert tracker.positions == {}

    def test_top_n_to_consume_bounds_per_row_attempts(self):
        """Even if the ranker returns more rows, only the top
        ``top_n_to_consume`` are fed into the tracker.
        """
        from unittest.mock import patch

        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)

        rows = pd.DataFrame(
            [
                {
                    "ticker": f"T{i}",
                    "spot": 100.0,
                    "strike": 95.0,
                    "premium": 1.5,
                    "dte": 35,
                    "iv": 0.22,
                    "ev_dollars": 50.0 - i,  # all positive
                    "ev_per_day": 1.4,
                    "prob_profit": 0.78,
                    "prob_assignment": 0.18,
                    "distribution_source": "empirical_non_overlapping",
                }
                for i in range(5)
            ]
        )
        with patch.object(WheelRunner, "rank_candidates_by_ev", return_value=rows):
            outcomes = runner.consume_into_tracker(
                tracker,
                entry_date=_ENTRY_DATE,
                rank_kwargs={"top_n": 5, "min_ev_dollars": -1e9},
                top_n_to_consume=2,
            )
        # Only the first 2 rows attempted.
        assert len(outcomes) == 2
        assert [o["ticker"] for o in outcomes] == ["T0", "T1"]
        assert len(tracker.positions) == 2

    def test_helper_forces_include_diagnostic_fields(self):
        """The token-hash canonicalisation needs ``distribution_source``,
        which is only emitted when ``include_diagnostic_fields=True``.
        The helper must force that flag regardless of the caller's
        ``rank_kwargs``.
        """
        from unittest.mock import patch

        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE)

        observed_kwargs: dict[str, Any] = {}

        def _spy(self, **kw):
            observed_kwargs.update(kw)
            return pd.DataFrame()

        with patch.object(WheelRunner, "rank_candidates_by_ev", _spy):
            runner.consume_into_tracker(
                tracker,
                entry_date=_ENTRY_DATE,
                # Caller explicitly disables diagnostic fields — helper
                # must override.
                rank_kwargs={"include_diagnostic_fields": False},
            )
        assert observed_kwargs.get("include_diagnostic_fields") is True, (
            f"consume_into_tracker must force include_diagnostic_fields=True; "
            f"observed {observed_kwargs.get('include_diagnostic_fields')!r}"
        )

    def test_signature_exposes_documented_kwargs(self):
        import inspect

        sig = inspect.signature(WheelRunner.consume_into_tracker)
        params = sig.parameters
        # Positional + first kwarg
        assert "tracker" in params
        assert "entry_date" in params
        # All advertised kwargs are present
        for kw in (
            "rank_kwargs",
            "top_n_to_consume",
            "expiration_date",
        ):
            assert kw in params, f"consume_into_tracker missing advertised kwarg {kw!r}"
            assert params[kw].default is None, (
                f"{kw} default must be None; got {params[kw].default!r}"
            )


class TestConsumeIntoTrackerStrictMode:
    """In strict mode (``require_ev_authority=True``), the wire
    structurally enforces D16 token + D17 hard-blocks. Pin that the
    helper continues to work — the consume_ranker_row already issues
    + consumes the token internally; no caller boilerplate required.
    """

    def test_strict_mode_flow_opens_position_and_drains_token(self):
        runner = WheelRunner()
        tracker = WheelTracker(initial_capital=_NAV_LARGE, require_ev_authority=True)
        outcomes = runner.consume_into_tracker(
            tracker,
            entry_date=_ENTRY_DATE,
            rank_kwargs={"tickers": ["AAPL"], "as_of": _AS_OF, "top_n": 1},
        )
        assert outcomes
        opened_outcomes = [o for o in outcomes if o["opened"]]
        if not opened_outcomes:
            pytest.skip(
                f"strict mode opened no positions; outcomes={outcomes}. "
                "Likely D17 cap or stale-EV at fire time."
            )
        # Token-hash one-shot semantics — tokens dict drained after consume.
        assert len(tracker._ev_authority_tokens) == 0, (
            f"strict-mode consume should drain the token; remaining={tracker._ev_authority_tokens}"
        )
        actions = [entry.get("action") for entry in tracker._ev_authority_log]
        assert "issue" in actions
        assert "consume" in actions
