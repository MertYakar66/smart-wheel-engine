"""Tests for engine/event_gate.py — hard event lockout for the EV engine."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from engine.event_gate import EventGate, ScheduledEvent


class TestEventGateBasics:
    def test_no_events_no_block(self):
        gate = EventGate()
        blocked, reason = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert not blocked
        assert reason == ""

    def test_add_single_event_in_window_blocks(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)))
        blocked, reason = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert blocked
        assert "earnings" in reason
        assert "2026-05-10" in reason
        assert "5d" in reason  # buffer in reason

    def test_event_outside_window_does_not_block(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 7, 1)))
        blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert not blocked

    def test_other_ticker_event_does_not_block(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("MSFT", "earnings", date(2026, 5, 10)))
        blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert not blocked

    def test_wildcard_event_blocks_any_ticker(self):
        gate = EventGate(macro_buffer_days=1)
        gate.add_event(ScheduledEvent("*", "fomc", date(2026, 5, 10)))
        blocked_aapl, _ = gate.is_blocked("AAPL", date(2026, 5, 9), date(2026, 6, 1))
        blocked_msft, _ = gate.is_blocked("MSFT", date(2026, 5, 9), date(2026, 6, 1))
        assert blocked_aapl
        assert blocked_msft

    def test_case_insensitive_ticker_match(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("aapl", "earnings", date(2026, 5, 10)))
        blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert blocked

    def test_buffer_extends_window(self):
        gate = EventGate(earnings_buffer_days=5)
        # Trade window ends 5/5; event is on 5/10 — within 5-day buffer.
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)))
        blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 5, 5))
        assert blocked

    def test_buffer_does_not_extend_indefinitely(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 20)))
        # Window ends 5/5; event is 5/20 (15 days later, > 5 day buffer)
        blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 5, 5))
        assert not blocked


class TestBufferKinds:
    def test_macro_buffer_for_fomc_cpi_nfp_pce_ecb_boe(self):
        gate = EventGate(earnings_buffer_days=99, macro_buffer_days=1)
        for kind in ("fomc", "cpi", "nfp", "pce", "ecb", "boe"):
            gate.clear()
            gate.add_event(ScheduledEvent("*", kind, date(2026, 5, 10)))
            # Window ends 5/8; event 5/10 = 2 days out, beyond 1-day buffer
            blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 5, 8))
            assert not blocked, f"{kind} should not block (2-day gap > 1-day buffer)"
            # Window ends 5/9; event 5/10 = 1 day out, within 1-day buffer
            blocked2, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 5, 9))
            assert blocked2, f"{kind} should block (1-day gap == 1-day buffer)"

    def test_dividend_buffer(self):
        gate = EventGate(dividend_buffer_days=1)
        gate.add_event(ScheduledEvent("AAPL", "dividend", date(2026, 5, 10)))
        blocked, reason = gate.is_blocked("AAPL", date(2026, 5, 9), date(2026, 6, 1))
        assert blocked
        assert "dividend" in reason

    def test_split_buffer(self):
        gate = EventGate(split_buffer_days=3)
        gate.add_event(ScheduledEvent("AAPL", "split", date(2026, 5, 10)))
        blocked, reason = gate.is_blocked("AAPL", date(2026, 5, 7), date(2026, 6, 1))
        assert blocked
        assert "split" in reason

    def test_custom_kind_uses_zero_buffer(self):
        # 'custom' falls through to the zero-buffer branch
        gate = EventGate()
        gate.add_event(ScheduledEvent("AAPL", "custom", date(2026, 5, 10)))
        # Event must be inside the trade window itself
        blocked_inside, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 5, 20))
        assert blocked_inside
        blocked_outside, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 5, 9))
        assert not blocked_outside


class TestMultipleEvents:
    def test_first_by_date_returned_when_multiple_match(self):
        gate = EventGate(earnings_buffer_days=10)
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 20)))
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 12)))
        blocked, reason = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert blocked
        assert "2026-05-12" in reason  # first chronologically

    def test_add_events_batch(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_events(
            [
                ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)),
                ScheduledEvent("MSFT", "earnings", date(2026, 6, 10)),
            ]
        )
        assert len(gate.events) == 2

    def test_clear_removes_all_events(self):
        gate = EventGate()
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)))
        assert len(gate.events) == 1
        gate.clear()
        assert len(gate.events) == 0


class TestFilterCandidates:
    def test_keeps_unaffected_candidate(self):
        gate = EventGate(earnings_buffer_days=5)
        cands = [
            {"ticker": "AAPL", "trade_date": date(2026, 5, 1), "expiration": date(2026, 6, 1)},
            {"ticker": "XOM", "trade_date": date(2026, 5, 1), "expiration": date(2026, 6, 1)},
        ]
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)))
        kept, blocked = gate.filter_candidates(cands)
        assert len(kept) == 1 and kept[0]["ticker"] == "XOM"
        assert len(blocked) == 1 and blocked[0]["ticker"] == "AAPL"
        assert "event_lockout_reason" in blocked[0]

    def test_missing_dates_passes_through_kept(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)))
        cands = [{"ticker": "AAPL"}]  # no trade_date / expiration
        kept, blocked = gate.filter_candidates(cands)
        assert len(kept) == 1
        assert len(blocked) == 0

    def test_datetime_inputs_normalised_to_date(self):
        gate = EventGate(earnings_buffer_days=5)
        gate.add_event(ScheduledEvent("AAPL", "earnings", date(2026, 5, 10)))
        cands = [
            {
                "ticker": "AAPL",
                "trade_date": datetime(2026, 5, 1, 9, 30),
                "expiration": datetime(2026, 6, 1, 16, 0),
            }
        ]
        kept, blocked = gate.filter_candidates(cands)
        assert len(blocked) == 1


class TestFromBloombergCalendar:
    def test_earnings_only(self):
        earnings = pd.DataFrame(
            [
                {"ticker": "AAPL", "announcement_date": pd.Timestamp("2026-05-10")},
                {"ticker": "MSFT", "announcement_date": pd.Timestamp("2026-06-15")},
            ]
        )
        gate = EventGate.from_bloomberg_calendar(earnings)
        assert len(gate.events) == 2
        assert all(e.kind == "earnings" for e in gate.events)
        assert gate.earnings_buffer_days == 5  # default

    def test_macro_events(self):
        macro = pd.DataFrame(
            [
                {"event": "fomc", "date": pd.Timestamp("2026-05-10")},
                {"event": "CPI", "date": pd.Timestamp("2026-05-15")},
                {"event": "wibble", "date": pd.Timestamp("2026-05-20")},  # falls back to custom
            ]
        )
        gate = EventGate.from_bloomberg_calendar(earnings_df=None, macro_df=macro)
        kinds = [e.kind for e in gate.events]
        assert "fomc" in kinds
        assert "cpi" in kinds
        assert "custom" in kinds  # 'wibble' normalised
        assert all(e.ticker == "*" for e in gate.events)

    def test_dividends(self):
        divs = pd.DataFrame(
            [
                {"ticker": "AAPL", "ex_date": pd.Timestamp("2026-05-10")},
            ]
        )
        gate = EventGate.from_bloomberg_calendar(earnings_df=None, dividends_df=divs)
        assert len(gate.events) == 1
        assert gate.events[0].kind == "dividend"

    def test_skips_rows_with_missing_dates(self):
        # Defensive: rows with None or pandas NaT in the date column are
        # filtered at ingest, not admitted to the gate (which would crash
        # is_blocked() on a NaT vs date comparison).
        earnings = pd.DataFrame(
            [
                {"ticker": "AAPL", "announcement_date": pd.Timestamp("2026-05-10")},
                {"ticker": "MSFT", "announcement_date": None},
                {"ticker": "GOOG", "announcement_date": pd.NaT},
            ]
        )
        gate = EventGate.from_bloomberg_calendar(earnings)
        assert len(gate.events) == 1
        assert gate.events[0].ticker == "AAPL"
        # is_blocked must not crash on any subsequent query
        blocked, _ = gate.is_blocked("AAPL", date(2026, 5, 1), date(2026, 6, 1))
        assert blocked

    def test_empty_dataframes_produce_empty_gate(self):
        gate = EventGate.from_bloomberg_calendar(pd.DataFrame())
        assert gate.events == []

    def test_buffer_overrides_propagate(self):
        earnings = pd.DataFrame(
            [{"ticker": "AAPL", "announcement_date": pd.Timestamp("2026-05-10")}]
        )
        gate = EventGate.from_bloomberg_calendar(
            earnings,
            earnings_buffer_days=10,
            macro_buffer_days=2,
            dividend_buffer_days=2,
        )
        assert gate.earnings_buffer_days == 10
        assert gate.macro_buffer_days == 2
        assert gate.dividend_buffer_days == 2
