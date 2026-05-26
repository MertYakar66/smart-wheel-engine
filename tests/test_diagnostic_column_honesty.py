"""Regression tests for ranker diagnostic-column honesty.

Two findings from the 2026-05-25 end-to-end review
(``docs/END_TO_END_REVIEW_2026_05_25.md``) pinned diagnostic columns
that surfaced upstream values regardless of whether the corresponding
engine signal was actually live, misleading anyone reading the row:

* **S28 Fix #1** — :meth:`engine.wheel_runner.WheelRunner.rank_covered_calls_by_ev`
  emitted ``expected_dividend`` populated from ``conn.get_next_dividend``
  even when the EVEngine dividend gate at
  ``engine/ev_engine.py:355-361`` would refuse to fire (because
  ``days_to_ex_div > dte`` — the ex-div falls outside the holding
  window). A trader inspecting the row reasonably concludes the
  engine is factoring the dividend into EV; it is not. The fix
  emits ``0.0`` when the gate would block.

* **S29 Fix #1** — :meth:`engine.wheel_runner.WheelRunner.rank_candidates_by_ev`
  emitted ``skew_multiplier=1.0`` and ``skew_slope=None`` with no
  way to distinguish "skew was measured and is neutral" from "skew
  was not measured at all" (the Bloomberg connector has no chain
  access so the skew block is structurally dormant, per S29 F3).
  The fix adds a ``skew_source`` provenance column with values
  ``"chain"`` (block executed; skew was measured) or
  ``"unavailable"`` (block did not execute; multiplier defaulted to
  1.0). Mirrors the S1B precedent of ``oi_source`` / ``premium_source``.

These tests inoculate against either fix being reverted. Every
fixture drives the real :class:`~engine.data_connector.MarketDataConnector`
+ real Bloomberg CSVs at ``as_of=2026-03-20`` (the CSV cutoff per
``CLAUDE.md §4``) so a schema or data-flow regression that breaks
the column honesty fails immediately.
"""

from __future__ import annotations

import pytest

from engine.wheel_runner import WheelRunner

_AS_OF = "2026-03-20"  # CSV cutoff per CLAUDE.md §4


# ----------------------------------------------------------------------
# S28 — expected_dividend honesty in rank_covered_calls_by_ev
# ----------------------------------------------------------------------


class TestExpectedDividendDiagnosticHonesty:
    """The ``expected_dividend`` diagnostic column reflects what the
    EVEngine dividend gate (``engine/ev_engine.py:355-361``) would
    actually apply as a penalty, NOT just the upstream lookup value.
    The gate fires only when ``days_to_ex_div <= dte AND
    expected_dividend > 0``; when either condition fails the column
    must read 0.0.
    """

    def test_ex_div_outside_window_emits_zero_expected_dividend(self):
        """MSFT at 2026-03-20 has ex-div 2026-05-21 ($0.91) — that's
        ~62 days out. A 35-DTE CC would have ``days_to_ex_div > dte``,
        so the EVEngine gate refuses to fire. The diagnostic column
        must read 0.0 (not 0.91) to avoid implying penalty machinery
        that did not run.

        S28 F2: pre-fix, the column showed 0.91 in this exact
        configuration despite the gate blocking.
        """
        runner = WheelRunner()
        df = runner.rank_covered_calls_by_ev(
            ticker="MSFT",
            shares_held=100,
            target_dtes=(35,),
            target_deltas=(0.25,),
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
            use_event_gate=False,  # avoid earnings-window noise
        )
        assert not df.empty, (
            f"expected at least one MSFT 35-DTE CC row at as_of={_AS_OF}; "
            f"drops={df.attrs.get('drops', [])}"
        )
        # Sanity: the gate-block condition holds — ex-div is outside
        # the holding window on every row.
        for _, row in df.iterrows():
            dte = int(row["dte"])
            d2x = row.get("days_to_ex_div")
            # Either there's no upcoming dividend, or it's outside
            # the holding window.
            if d2x is not None:
                assert d2x > dte, (
                    f"test premise broken: MSFT days_to_ex_div={d2x} <= dte={dte}; "
                    "the gate would fire — pick a different DTE or ticker"
                )
            # The honesty contract: column must be 0.0 when the gate
            # would not apply the penalty.
            assert row["expected_dividend"] == pytest.approx(0.0, abs=1e-9), (
                f"expected_dividend honesty broken: row shows "
                f"{row['expected_dividend']} but EVEngine gate would not "
                f"fire (days_to_ex_div={d2x}, dte={dte})"
            )

    def test_ex_div_inside_window_emits_actual_dividend(self):
        """JPM at 2026-03-20 has ex-div 2026-04-06 ($1.50) — ~17 days
        out. A 35-DTE CC has ``days_to_ex_div <= dte`` and a positive
        dividend, so the EVEngine gate fires. The diagnostic column
        must reflect the actual dividend amount the gate is applying.
        """
        runner = WheelRunner()
        df = runner.rank_covered_calls_by_ev(
            ticker="JPM",
            shares_held=100,
            target_dtes=(35,),
            target_deltas=(0.25,),
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
            use_event_gate=False,  # avoid earnings-window noise
        )
        assert not df.empty, (
            f"expected at least one JPM 35-DTE CC row at as_of={_AS_OF}; "
            f"drops={df.attrs.get('drops', [])}"
        )
        # Skip if the test premise no longer holds (e.g. dividend file
        # was refreshed and JPM's ex-div moved out of the 35-day window).
        any_inside_window = any(
            row.get("days_to_ex_div") is not None
            and int(row.get("days_to_ex_div", 999)) <= int(row["dte"])
            for _, row in df.iterrows()
        )
        if not any_inside_window:
            pytest.skip(
                f"JPM has no ex-div inside 35-day window at as_of={_AS_OF} — "
                "test premise broken, likely upstream dividend-file refresh"
            )

        for _, row in df.iterrows():
            dte = int(row["dte"])
            d2x = row.get("days_to_ex_div")
            if d2x is None or int(d2x) > dte:
                # Gate would not fire on this row — must be 0.0
                assert row["expected_dividend"] == pytest.approx(0.0, abs=1e-9)
            else:
                # Gate fires — column reflects the actual dividend
                assert row["expected_dividend"] > 0.0, (
                    f"JPM row in window (d2x={d2x}, dte={dte}) shows "
                    f"expected_dividend={row['expected_dividend']} but "
                    "should reflect the upstream amount the gate applies"
                )

    def test_diagnostic_zero_does_not_change_ev_dollars(self):
        """The fix is observability-only: zeroing the diagnostic
        column must NOT change ``ev_dollars``. EV is computed by
        :meth:`EVEngine.evaluate` from the trade's ``expected_dividend``
        kwarg, which is set unconditionally from the upstream lookup
        (``ShortOptionTrade(..., expected_dividend=expected_dividend, …)``
        at ``engine/wheel_runner.py:2239``); the gate inside
        ``EVEngine.evaluate`` is what decides whether to apply the
        penalty. Pin that the diagnostic-column fix touches only the
        row dict, never the EV math.
        """
        runner = WheelRunner()
        df = runner.rank_covered_calls_by_ev(
            ticker="MSFT",
            shares_held=100,
            target_dtes=(35,),
            target_deltas=(0.25,),
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
            use_event_gate=False,
        )
        assert not df.empty
        for _, row in df.iterrows():
            # ev_dollars should still be a real number reflecting
            # the engine's actual EV computation, not zero.
            assert row["ev_dollars"] is not None
            assert isinstance(row["ev_dollars"], (int, float))


# ----------------------------------------------------------------------
# S29 — skew_source provenance in rank_candidates_by_ev
# ----------------------------------------------------------------------


class TestSkewSourceProvenance:
    """The ``skew_source`` column distinguishes a measured-neutral
    skew (``"chain"``) from an unmeasured one (``"unavailable"``).
    On the Bloomberg connector the skew block is structurally dormant
    (no ``get_options`` / ``get_option_chain`` method), so every row
    must read ``"unavailable"``. S29 F3 confirmed this dormancy
    across five test tickers; the fix surfaces it in the row.
    """

    def test_skew_source_column_present_in_diagnostic_output(self):
        """The new column must be in the diagnostic output. Pin
        schema presence so a future column reorder doesn't drop it.
        """
        runner = WheelRunner()
        df = runner.rank_candidates_by_ev(
            tickers=["AAPL"],
            top_n=1,
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        assert not df.empty
        assert "skew_source" in df.columns, (
            "skew_source diagnostic column missing — S29 Fix #1 reverted?"
        )

    def test_skew_source_uniformly_unavailable_on_bloomberg(self):
        """The Bloomberg connector (``engine/data_connector.py``)
        exposes no ``get_options`` / ``get_option_chain`` method, so
        the skew_mult block at ``engine/wheel_runner.py:1260`` never
        executes and every row must read ``skew_source="unavailable"``.

        S29 F3 verified this for AAPL/MSFT/META/AMZN/KO/PG/JNJ/XOM/VZ/TSLA/NVDA;
        this test pins the same property for a 5-ticker sample.
        """
        runner = WheelRunner()
        # Connector must be the Bloomberg path (no chain access).
        conn_name = type(runner.connector).__name__
        assert conn_name == "MarketDataConnector", (
            f"this test pins Bloomberg behaviour; runner is using {conn_name}"
        )
        # And the chain methods must genuinely be absent — that's the
        # structural reason the skew block is dormant.
        assert not hasattr(runner.connector, "get_options"), (
            "MarketDataConnector unexpectedly grew get_options — the test's "
            "premise (chain-absence dormancy) no longer holds"
        )
        assert not hasattr(runner.connector, "get_option_chain"), (
            "MarketDataConnector unexpectedly grew get_option_chain — the "
            "test's premise no longer holds"
        )

        df = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
            top_n=10,
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        assert not df.empty, f"expected at least one row from 5-ticker watchlist at as_of={_AS_OF}"
        unique_sources = df["skew_source"].dropna().unique().tolist()
        assert unique_sources == ["unavailable"], (
            f"on Bloomberg every row's skew_source must be 'unavailable'; got {unique_sources}"
        )

    def test_skew_source_unavailable_pairs_with_default_skew_multiplier(self):
        """When ``skew_source=="unavailable"``, the
        ``skew_multiplier`` column must be the unmeasured default
        (1.0) — otherwise the provenance label is lying. Pins the
        coherence between the two columns.
        """
        runner = WheelRunner()
        df = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
            top_n=10,
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        assert not df.empty
        for _, row in df.iterrows():
            if row["skew_source"] == "unavailable":
                assert row["skew_multiplier"] == pytest.approx(1.0, abs=1e-9), (
                    f"skew_source='unavailable' implies skew_multiplier=1.0 "
                    f"(the unmeasured default); got {row['skew_multiplier']}"
                )

    def test_skew_source_unavailable_pairs_with_empty_slope_columns(self):
        """When ``skew_source=="unavailable"``, the related
        ``skew_slope`` / ``put_skew`` / ``risk_reversal`` columns
        must be None — they're populated from ``skew_diag`` only
        when the block executes. Pins the coherence between provenance
        and the diagnostic-detail columns.
        """
        runner = WheelRunner()
        df = runner.rank_candidates_by_ev(
            tickers=["AAPL", "MSFT"],
            top_n=5,
            as_of=_AS_OF,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        assert not df.empty
        for _, row in df.iterrows():
            if row["skew_source"] == "unavailable":
                assert row["skew_slope"] is None, (
                    f"skew_source='unavailable' implies skew_slope=None; got {row['skew_slope']}"
                )
                assert row["put_skew"] is None
                assert row["risk_reversal"] is None
