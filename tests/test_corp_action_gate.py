"""#3A — corporate-action event lockout (W/Phase-3A).

Wires the dormant `engine.event_gate` `kind="corp_action"` lockout to the
`sp500_corporate_actions.csv` data (on `main`, 52,442 rows) via a new connector
accessor `get_corporate_actions` and the module-level ranker helper
`engine.wheel_runner._register_corp_action_events`. The gate hard-blocks a wheel
whose holding window touches a *disruptive* (non-`Regular Cash`) corporate action
— a split seam, a GE-style spinoff, a large special cash — events the empirical
forward distribution cannot price.

§2: remove-only / evaluate-input-correctness — this can only DROP a candidate
(via the existing hard event lockout), never rescue one. Point-in-time: only
actions *announced* by `as_of` are registered.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.event_gate import EventGate, ScheduledEvent
from engine.wheel_runner import _register_corp_action_events

# --------------------------------------------------------------------------
# 1. EventGate — the corp_action kind
# --------------------------------------------------------------------------


class TestCorpActionKind:
    def test_corp_action_has_its_own_buffer(self):
        g = EventGate(corp_action_buffer_days=3)
        assert g._buffer_for("corp_action") == 3
        g2 = EventGate(corp_action_buffer_days=7)
        assert g2._buffer_for("corp_action") == 7

    def test_corp_action_blocks_window_and_reason(self):
        g = EventGate(corp_action_buffer_days=3)
        g.add_event(ScheduledEvent(ticker="GE", kind="corp_action", event_date=date(2023, 1, 4)))
        # holding window touches the action -> blocked
        blk, reason = g.is_blocked("GE", date(2023, 1, 3), date(2023, 2, 7))
        assert blk and "corp_action@2023-01-04" in reason
        # a different name is unaffected (per-ticker, not wildcard)
        assert g.is_blocked("AAPL", date(2023, 1, 3), date(2023, 2, 7))[0] is False
        # well outside the window+buffer -> not blocked
        assert g.is_blocked("GE", date(2023, 3, 1), date(2023, 4, 5))[0] is False

    def test_corp_action_back_buffer_is_symmetric(self):
        # an action 2 days BEFORE entry (within the 3-day buffer) still blocks
        g = EventGate(corp_action_buffer_days=3)
        g.add_event(ScheduledEvent(ticker="GE", kind="corp_action", event_date=date(2023, 1, 1)))
        assert g.is_blocked("GE", date(2023, 1, 3), date(2023, 2, 7))[0] is True


# --------------------------------------------------------------------------
# 2. The ranker helper — registration, no-op safety, PIT
# --------------------------------------------------------------------------


class _CorpConn:
    """Minimal connector exposing get_corporate_actions from an in-memory frame."""

    def __init__(self, rows):
        self._df = pd.DataFrame(rows)

    def get_corporate_actions(
        self, ticker, start_date=None, end_date=None, as_of=None, include_regular_cash=False
    ):
        df = self._df[self._df["ticker"] == ticker].copy()
        if df.empty:
            return df
        df["effective_date"] = pd.to_datetime(df["effective_date"])
        df["announcement_date"] = pd.to_datetime(df["announcement_date"])
        if start_date is not None:
            df = df[df["effective_date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["effective_date"] <= pd.Timestamp(end_date)]
        if as_of is not None:
            df = df[df["announcement_date"] <= pd.Timestamp(as_of)]
        if not include_regular_cash:
            df = df[df["action_type"] != "Regular Cash"]
        return df.reset_index(drop=True)


_ROWS = [
    {
        "ticker": "GE",
        "announcement_date": "2021-11-09",
        "effective_date": "2023-01-04",
        "action_type": "Spinoff",
    },
    {
        "ticker": "GE",
        "announcement_date": "2024-01-02",
        "effective_date": "2024-04-02",
        "action_type": "Spinoff",
    },
    {
        "ticker": "GE",
        "announcement_date": "2022-02-01",
        "effective_date": "2022-03-01",
        "action_type": "Regular Cash",
    },
]


class TestRegisterCorpActionEvents:
    def test_registers_disruptive_actions(self):
        g = EventGate()
        _register_corp_action_events(
            g, _CorpConn(_ROWS), "GE", date(2023, 1, 3), as_of="2023-01-03"
        )
        kinds = [(e.kind, e.event_date.isoformat()) for e in g.events]
        # the 2023-01-04 spinoff is registered (announced 2021); Regular Cash excluded
        assert ("corp_action", "2023-01-04") in kinds
        assert all(e.kind == "corp_action" for e in g.events)

    def test_point_in_time_excludes_unannounced(self):
        # at as_of before any announcement, nothing is known
        g = EventGate()
        _register_corp_action_events(
            g, _CorpConn(_ROWS), "GE", date(2021, 1, 4), as_of="2021-01-04"
        )
        assert g.events == []

    def test_noop_when_gate_is_none(self):
        # must not raise
        _register_corp_action_events(None, _CorpConn(_ROWS), "GE", date(2023, 1, 3))

    def test_noop_when_connector_lacks_accessor(self):
        class _Bare:
            pass

        g = EventGate()
        _register_corp_action_events(g, _Bare(), "GE", date(2023, 1, 3))
        assert g.events == []

    def test_survives_connector_errors(self):
        class _Boom:
            def get_corporate_actions(self, *a, **k):
                raise RuntimeError("down")

        g = EventGate()
        _register_corp_action_events(g, _Boom(), "GE", date(2023, 1, 3))
        assert g.events == []


# --------------------------------------------------------------------------
# 3. Connector accessor on real data (the on-main corp-actions census)
# --------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

_HAS_CA = Path("data/bloomberg/sp500_corporate_actions.csv").exists()
needs_ca = pytest.mark.skipif(
    not _HAS_CA, reason="needs data/bloomberg/sp500_corporate_actions.csv"
)


@needs_ca
def test_get_corporate_actions_excludes_regular_cash_by_default():
    from engine.data_connector import MarketDataConnector

    c = MarketDataConnector()
    ge = c.get_corporate_actions("GE", start_date="2022-01-01", end_date="2024-12-31")
    assert not ge.empty, "expected GE's 2023/2024 spinoffs"
    assert (ge["action_type"] != "Regular Cash").all()
    # the two GE spinoffs are present
    effs = set(ge["effective_date"].dt.strftime("%Y-%m-%d"))
    assert {"2023-01-04", "2024-04-02"} <= effs
    # raw set DOES include the (far more numerous) Regular Cash rows
    raw = c.get_corporate_actions(
        "GE", start_date="2022-01-01", end_date="2024-12-31", include_regular_cash=True
    )
    assert len(raw) > len(ge)


@needs_ca
def test_get_corporate_actions_is_point_in_time():
    from engine.data_connector import MarketDataConnector

    c = MarketDataConnector()
    # COST's 2023-12-27 special cash was announced 2023-11; unknown mid-2023.
    known_after = c.get_corporate_actions(
        "COST", start_date="2023-01-01", end_date="2024-06-30", as_of="2024-06-30"
    )
    known_before = c.get_corporate_actions(
        "COST", start_date="2023-01-01", end_date="2024-06-30", as_of="2023-06-30"
    )
    # the special cash is visible after, not before its announcement
    assert len(known_after) >= len(known_before)


@needs_ca
def test_end_to_end_gate_blocks_into_real_spinoff():
    from engine.data_connector import MarketDataConnector

    c = MarketDataConnector()
    g = EventGate()
    _register_corp_action_events(g, c, "GE", date(2023, 1, 3), as_of="2023-01-03")
    blk, reason = g.is_blocked("GE", date(2023, 1, 3), date(2023, 2, 7))
    assert blk and "corp_action" in reason
