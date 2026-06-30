"""#3A — market-wide macro-release event lockout (Phase-3A, macro half).

Wires the dormant ``engine.event_gate`` macro kinds (``fomc`` / ``cpi`` / ``nfp``
/ ``pce``) to the broad-pull macro-release calendar
(``data/bloomberg/broad_pull/macro_calendar/sp500_macro_calendar.csv``) via a new
connector accessor ``get_macro_events`` and the module-level ranker helper
``engine.wheel_runner._register_macro_events``. The gate hard-blocks any wheel
whose holding window touches one of the four highest-impact, distribution-breaking
releases — the FOMC rate decision, CPI, NFP, PCE — events near which the empirical
forward distribution is mis-specified (jump regimes, IV crush).

These events affect *every* candidate, so they register **once per run** with the
wildcard ticker ``"*"`` (unlike the per-ticker earnings / corporate-action
registration). Lower-tier prints on the same calendar (ISM, initial jobless
claims, retail sales, GDP, unemployment) are deliberately NOT gated.

§2: remove-only / evaluate-input-correctness — this can only DROP a candidate
(via the existing hard event lockout, which runs BEFORE EV is computed), never
rescue one. No PIT *announcement* question: scheduled releases are published far
in advance.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from engine.event_gate import EventGate
from engine.wheel_runner import _MACRO_EVENT_KIND, _register_macro_events

# --------------------------------------------------------------------------
# 1. The event→kind map is well-formed (every value is a macro EventKind)
# --------------------------------------------------------------------------


class TestMacroEventKindMap:
    def test_map_targets_only_the_four_macro_kinds(self):
        assert set(_MACRO_EVENT_KIND.values()) == {"fomc", "cpi", "nfp", "pce"}

    def test_every_mapped_kind_uses_the_macro_buffer(self):
        # The invariant the map relies on: every kind it produces resolves to
        # macro_buffer_days in EventGate (not 0 / "custom"). A drift to a
        # non-macro kind would silently change the lockout buffer.
        g = EventGate(macro_buffer_days=2)
        for kind in set(_MACRO_EVENT_KIND.values()):
            assert g._buffer_for(kind) == 2, kind

    def test_core_cpi_collapses_onto_cpi(self):
        assert _MACRO_EVENT_KIND["core_cpi_yoy"] == "cpi"
        assert _MACRO_EVENT_KIND["cpi_yoy"] == "cpi"

    def test_lower_tier_prints_are_not_in_the_map(self):
        for ev in (
            "ism_manufacturing",
            "ism_services",
            "initial_jobless_claims",
            "retail_sales_mom",
            "gdp_qoq",
            "unemployment_rate",
        ):
            assert ev not in _MACRO_EVENT_KIND


# --------------------------------------------------------------------------
# 2. The ranker helper — registration, scoping, no-op safety
# --------------------------------------------------------------------------


class _MacroConn:
    """Minimal connector exposing get_macro_events from an in-memory frame."""

    def __init__(self, rows):
        self._df = pd.DataFrame(rows)

    def get_macro_events(self, start_date=None, end_date=None):
        df = self._df.copy()
        if df.empty:
            return df
        df["release_date"] = pd.to_datetime(df["release_date"])
        if start_date is not None:
            df = df[df["release_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["release_date"] <= pd.Timestamp(end_date)]
        return df[["event", "release_date"]].reset_index(drop=True)


# Anchored around a 2026-03-04 decision date; a mix of mapped + unmapped events.
_ROWS = [
    {"event": "fed_funds_target", "release_date": "2026-03-18"},  # -> fomc
    {"event": "cpi_yoy", "release_date": "2026-03-11"},  # -> cpi
    {"event": "core_cpi_yoy", "release_date": "2026-03-11"},  # -> cpi
    {"event": "nonfarm_payrolls", "release_date": "2026-03-06"},  # -> nfp
    {"event": "pce_yoy", "release_date": "2026-03-27"},  # -> pce
    {"event": "ism_manufacturing", "release_date": "2026-03-02"},  # NOT gated
    {"event": "initial_jobless_claims", "release_date": "2026-03-05"},  # NOT gated
    {"event": "gdp_qoq", "release_date": "2026-03-26"},  # NOT gated
]


class TestRegisterMacroEvents:
    def test_registers_mapped_events_as_wildcard(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(g, _MacroConn(_ROWS), as_of="2026-03-04")
        assert g.events, "expected mapped macro events to be registered"
        assert all(e.ticker == "*" for e in g.events)
        kinds = {e.kind for e in g.events}
        assert kinds == {"fomc", "cpi", "nfp", "pce"}

    def test_lower_tier_events_are_not_registered(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(g, _MacroConn(_ROWS), as_of="2026-03-04")
        # ISM / jobless / GDP carry no macro EventKind -> never on the gate.
        registered_dates = {e.event_date.isoformat() for e in g.events}
        assert "2026-03-02" not in registered_dates  # ism
        assert "2026-03-05" not in registered_dates  # jobless
        assert "2026-03-26" not in registered_dates  # gdp

    def test_core_cpi_registers_as_cpi(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(
            g,
            _MacroConn([{"event": "core_cpi_yoy", "release_date": "2026-03-11"}]),
            as_of="2026-03-04",
        )
        assert [(e.kind, e.event_date.isoformat()) for e in g.events] == [("cpi", "2026-03-11")]

    def test_future_within_horizon_registers_without_announcement_filter(self):
        # Scheduled releases are not PIT-announcement-gated: a release dated in
        # the future relative to as_of (but within the horizon) registers.
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(
            g,
            _MacroConn([{"event": "fed_funds_target", "release_date": "2026-06-17"}]),
            as_of="2026-03-04",
        )
        assert [(e.kind, e.event_date.isoformat()) for e in g.events] == [("fomc", "2026-06-17")]

    def test_beyond_horizon_is_not_registered(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(
            g,
            _MacroConn([{"event": "fed_funds_target", "release_date": "2030-01-29"}]),
            as_of="2026-03-04",
            horizon_days=400,
        )
        assert g.events == []

    def test_noop_when_gate_is_none(self):
        # must not raise
        _register_macro_events(None, _MacroConn(_ROWS), as_of="2026-03-04")

    def test_noop_when_connector_lacks_accessor(self):
        class _Bare:
            pass

        g = EventGate()
        _register_macro_events(g, _Bare(), as_of="2026-03-04")
        assert g.events == []

    def test_survives_connector_errors(self):
        class _Boom:
            def get_macro_events(self, *a, **k):
                raise RuntimeError("down")

        g = EventGate()
        _register_macro_events(g, _Boom(), as_of="2026-03-04")
        assert g.events == []

    def test_noop_on_empty_calendar(self):
        g = EventGate()
        _register_macro_events(g, _MacroConn([]), as_of="2026-03-04")
        assert g.events == []


# --------------------------------------------------------------------------
# 3. End-to-end — the registered wildcard events block every candidate
# --------------------------------------------------------------------------


class TestEndToEndBlocking:
    def test_window_into_fomc_is_blocked(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(g, _MacroConn(_ROWS), as_of="2026-03-04")
        # a window 2026-03-15..2026-03-20 (±1d -> [03-14, 03-21]) straddles ONLY
        # the 03-18 FOMC (the 03-11 CPI and 03-27 PCE fall outside the buffer).
        blk, reason = g.is_blocked("AAPL", date(2026, 3, 15), date(2026, 3, 20))
        assert blk and "event_lockout:fomc@2026-03-18" in reason

    def test_macro_lockout_applies_to_any_ticker(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(g, _MacroConn(_ROWS), as_of="2026-03-04")
        # wildcard "*": the SAME macro event blocks two unrelated names
        for tk in ("AAPL", "XOM"):
            blk, _ = g.is_blocked(tk, date(2026, 3, 12), date(2026, 3, 26))
            assert blk, tk

    def test_window_clear_of_macro_events_is_not_blocked(self):
        g = EventGate(macro_buffer_days=1)
        _register_macro_events(g, _MacroConn(_ROWS), as_of="2026-03-04")
        # a short window in a gap with no mapped release nearby
        blk, reason = g.is_blocked("AAPL", date(2026, 3, 20), date(2026, 3, 23))
        assert blk is False and reason == ""


# --------------------------------------------------------------------------
# 4. Connector accessor on the real broad-pull macro calendar
# --------------------------------------------------------------------------

_MACRO_CSV = Path("data/bloomberg/broad_pull/macro_calendar/sp500_macro_calendar.csv")
needs_macro = pytest.mark.skipif(
    not _MACRO_CSV.exists(), reason="needs broad_pull/macro_calendar/sp500_macro_calendar.csv"
)


@needs_macro
def test_get_macro_events_returns_mapped_releases():
    from engine.data_connector import MarketDataConnector

    c = MarketDataConnector()
    me = c.get_macro_events(start_date="2026-01-01", end_date="2026-12-31")
    assert me is not None and not me.empty
    assert list(me.columns) == ["event", "release_date"]
    events = set(me["event"].unique())
    # the four gated releases are all present in the real calendar
    assert {"fed_funds_target", "cpi_yoy", "nonfarm_payrolls", "pce_yoy"} <= events


@needs_macro
def test_get_macro_events_respects_date_range():
    from engine.data_connector import MarketDataConnector

    c = MarketDataConnector()
    me = c.get_macro_events(start_date="2026-06-01", end_date="2026-06-30")
    assert me is not None
    assert (me["release_date"] >= pd.Timestamp("2026-06-01")).all()
    assert (me["release_date"] <= pd.Timestamp("2026-06-30")).all()


# --------------------------------------------------------------------------
# 5. The ranker flag — default OFF (capability dormant), ON empties the book
# --------------------------------------------------------------------------

_OHLCV_CSV = Path("data/bloomberg/sp500_ohlcv.csv")
needs_ranker_data = pytest.mark.skipif(
    not (_OHLCV_CSV.exists() and _MACRO_CSV.exists()),
    reason="needs bloomberg OHLCV + broad_pull macro calendar",
)
_SMOKE = ["AAPL", "MSFT", "JPM", "XOM", "UNH"]


@needs_ranker_data
def test_macro_gate_is_off_by_default():
    # Default path: the macro lockout is NOT registered, so the smoke book is
    # non-empty (byte-identical to pre-#3A behaviour — no re-baseline impact).
    from engine.wheel_runner import WheelRunner

    df = WheelRunner().rank_candidates_by_ev(
        tickers=_SMOKE, top_n=10, min_ev_dollars=-1e9, include_diagnostic_fields=True
    )
    assert len(df) > 0


@needs_ranker_data
def test_macro_gate_on_empties_the_book_monthly_print_infeasibility():
    # The documented finding (docs/WIRING_CAMPAIGN.md §3A): with the EventGate's
    # whole-holding-window semantic, gating ~monthly macro prints blocks every
    # 21-63 DTE wheel (any such window contains a CPI/NFP/PCE). This pins WHY the
    # flag ships default-off — activating it as-is empties the book.
    from engine.wheel_runner import WheelRunner

    df = WheelRunner().rank_candidates_by_ev(
        tickers=_SMOKE,
        top_n=10,
        min_ev_dollars=-1e9,
        include_diagnostic_fields=True,
        use_macro_event_gate=True,
    )
    assert len(df) == 0


@needs_macro
def test_end_to_end_gate_blocks_into_real_fomc():
    from engine.data_connector import MarketDataConnector

    c = MarketDataConnector()
    me = c.get_macro_events(start_date="2026-01-01", end_date="2026-12-31")
    fomc_dates = sorted(d.date() for d in me[me["event"] == "fed_funds_target"]["release_date"])
    assert fomc_dates, "expected at least one 2026 FOMC date in the real calendar"
    target = fomc_dates[len(fomc_dates) // 2]
    as_of = (target - timedelta(days=10)).isoformat()

    g = EventGate(macro_buffer_days=1)
    _register_macro_events(g, c, as_of=as_of)
    ts = date.fromisoformat(as_of)
    te = ts + timedelta(days=35)  # 35-DTE wheel straddles the FOMC
    blk, reason = g.is_blocked("AAPL", ts, te)
    assert blk and "event_lockout:" in reason
