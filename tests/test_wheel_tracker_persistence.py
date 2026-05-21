"""Tests for WheelTracker / WheelPosition persistence.

Issue #118 P4. S2 logged that a ``WheelTracker`` could not be
serialised or resumed. ``to_dict`` / ``from_dict`` / ``save`` / ``load``
give it a JSON round-trip.

Pinned here:
  * ``WheelPosition`` round-trips — ``PositionState`` enum and the six
    ``date`` fields reconstruct as real objects, not strings;
  * ``WheelTracker.to_dict`` is fully JSON-serialisable;
  * a tracker with mixed-state positions, closed trades and an equity
    curve round-trips structurally identical;
  * the ``connector`` is not serialised and is re-attached on load;
  * a reloaded tracker keeps operating (positions are live objects);
  * ``save`` / ``load`` round-trip through a real JSON file.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date

from engine.wheel_tracker import PositionState, WheelPosition, WheelTracker

_ENTRY = date(2026, 1, 5)
_EXPIRY = date(2026, 2, 20)
_CC_EXPIRY = date(2026, 4, 17)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _open_put(t: WheelTracker, ticker: str, strike: float = 200.0, premium: float = 3.0) -> None:
    assert t.open_short_put(ticker, strike, premium, _ENTRY, _EXPIRY, 0.25), (
        f"open_short_put({ticker}) failed"
    )


def _make_populated_tracker() -> WheelTracker:
    """A tracker with a position in each lifecycle state, a closed trade
    and an equity-curve point — exercises every serialised field."""
    t = WheelTracker(initial_capital=150_000.0)
    _open_put(t, "PUT1", strike=200.0)  # stays SHORT_PUT
    _open_put(t, "STK1", strike=150.0)
    t.handle_put_assignment("STK1", _EXPIRY, 148.0)  # → STOCK_OWNED
    _open_put(t, "CC1", strike=90.0)
    t.handle_put_assignment("CC1", _EXPIRY, 88.0)
    t.open_covered_call("CC1", 99.0, 2.0, _EXPIRY, _CC_EXPIRY, 0.25)  # → COVERED_CALL
    _open_put(t, "EXP1", strike=100.0)
    t.handle_put_expiration("EXP1", _EXPIRY, 120.0)  # expires worthless → closed_positions
    t.mark_to_market(date(2026, 1, 20), {"PUT1": 198.0, "STK1": 150.0, "CC1": 95.0})
    return t


# ======================================================================
# 1. WheelPosition serialisation
# ======================================================================
class TestWheelPositionSerialization:
    def test_date_fields_constant_is_not_a_dataclass_field(self):
        """_DATE_FIELDS has no annotation, so it must stay a plain class
        constant — never become a dataclass field."""
        field_names = {f.name for f in dataclasses.fields(WheelPosition)}
        assert "_DATE_FIELDS" not in field_names

    def test_to_dict_is_json_safe(self):
        t = WheelTracker(initial_capital=100_000.0)
        _open_put(t, "AAA", strike=180.0)
        d = t.positions["AAA"].to_dict()
        json.dumps(d)  # must not raise

    def test_short_put_position_round_trips(self):
        t = WheelTracker(initial_capital=100_000.0)
        _open_put(t, "AAA", strike=180.0)
        pos = t.positions["AAA"]
        back = WheelPosition.from_dict(pos.to_dict())
        assert back == pos  # dataclass equality across every field

    def test_covered_call_position_round_trips(self):
        """Exercises the call-leg date fields."""
        t = _make_populated_tracker()
        pos = t.positions["CC1"]
        assert pos.state == PositionState.COVERED_CALL
        assert WheelPosition.from_dict(pos.to_dict()) == pos

    def test_enum_and_dates_reconstruct_as_objects(self):
        t = WheelTracker(initial_capital=100_000.0)
        _open_put(t, "AAA", strike=180.0)
        back = WheelPosition.from_dict(t.positions["AAA"].to_dict())
        assert isinstance(back.state, PositionState)
        assert isinstance(back.entry_date, date)
        assert isinstance(back.put_expiration_date, date)
        assert isinstance(back.put_expiration_date, date) and back.put_expiration_date == _EXPIRY

    def test_from_dict_ignores_unknown_keys(self):
        t = WheelTracker(initial_capital=100_000.0)
        _open_put(t, "AAA", strike=180.0)
        payload = t.positions["AAA"].to_dict()
        payload["a_field_from_a_newer_schema"] = 123
        back = WheelPosition.from_dict(payload)  # must not raise
        assert back.ticker == "AAA"

    def test_from_dict_fills_missing_optionals_with_defaults(self):
        minimal = {"ticker": "ZZZ", "state": "no_position", "entry_date": "2026-01-05"}
        back = WheelPosition.from_dict(minimal)
        assert back.ticker == "ZZZ"
        assert back.state == PositionState.NO_POSITION
        assert back.put_strike is None
        assert back.stock_shares == 0
        assert back.notes == []


# ======================================================================
# 2. WheelTracker.to_dict
# ======================================================================
class TestWheelTrackerToDict:
    def test_to_dict_is_fully_json_serialisable(self):
        t = _make_populated_tracker()
        json.dumps(t.to_dict())  # must not raise

    def test_to_dict_carries_expected_keys_and_omits_connector(self):
        t = _make_populated_tracker()
        d = t.to_dict()
        for key in (
            "schema_version",
            "initial_capital",
            "cash",
            "require_ev_authority",
            "positions",
            "closed_positions",
            "equity_curve",
            "ev_authority_tokens",
            "ev_authority_log",
        ):
            assert key in d
        # the live connector object is never serialised
        assert "connector" not in d


# ======================================================================
# 3. WheelTracker round-trip via to_dict / from_dict
# ======================================================================
class TestWheelTrackerRoundTrip:
    def test_empty_tracker_round_trips(self):
        t = WheelTracker(initial_capital=75_000.0)
        back = WheelTracker.from_dict(t.to_dict())
        assert back.cash == t.cash
        assert back.initial_capital == t.initial_capital
        assert back.positions == {}

    def test_populated_tracker_round_trips_structurally(self):
        t = _make_populated_tracker()
        back = WheelTracker.from_dict(t.to_dict())
        # to_dict is the canonical form — a clean structural identity
        assert back.to_dict() == t.to_dict()

    def test_position_states_and_strikes_survive(self):
        t = _make_populated_tracker()
        back = WheelTracker.from_dict(t.to_dict())
        assert {tk: p.state for tk, p in back.positions.items()} == {
            tk: p.state for tk, p in t.positions.items()
        }
        assert back.positions["PUT1"].put_strike == 200.0
        assert back.positions["CC1"].call_strike == 99.0

    def test_closed_positions_round_trip_with_date_objects(self):
        t = _make_populated_tracker()
        assert t.closed_positions, "fixture should have a closed trade"
        back = WheelTracker.from_dict(t.to_dict())
        assert len(back.closed_positions) == len(t.closed_positions)
        rec = back.closed_positions[0]
        assert isinstance(rec["entry_date"], date)
        assert isinstance(rec["exit_date"], date)
        assert rec["entry_date"] == t.closed_positions[0]["entry_date"]

    def test_equity_curve_round_trips_with_date_objects(self):
        t = _make_populated_tracker()
        assert t.equity_curve, "fixture should have an equity-curve point"
        back = WheelTracker.from_dict(t.to_dict())
        assert len(back.equity_curve) == len(t.equity_curve)
        assert isinstance(back.equity_curve[0]["date"], date)
        assert back.equity_curve[0]["date"] == t.equity_curve[0]["date"]

    def test_ev_authority_tokens_and_log_round_trip(self):
        t = WheelTracker(initial_capital=100_000.0, require_ev_authority=True)
        token = t.issue_ev_authority_token(
            {"ticker": "AAA", "strike": 180.0, "premium": 2.0, "dte": 35}
        )
        back = WheelTracker.from_dict(t.to_dict())
        assert back.require_ev_authority is True
        assert token in back._ev_authority_tokens
        assert back._ev_authority_log == t._ev_authority_log

    def test_connector_not_serialised_and_reattached(self):
        sentinel = object()
        t = WheelTracker(initial_capital=100_000.0, connector=sentinel)
        # connector is dropped by to_dict ...
        back_no_conn = WheelTracker.from_dict(t.to_dict())
        assert back_no_conn.connector is None
        # ... and re-attached when supplied to from_dict
        back_conn = WheelTracker.from_dict(t.to_dict(), connector=sentinel)
        assert back_conn.connector is sentinel

    def test_reloaded_tracker_keeps_operating(self):
        """The reconstructed positions are live WheelPosition objects —
        a reloaded tracker can still be traded and queried."""
        t = _make_populated_tracker()
        back = WheelTracker.from_dict(t.to_dict())
        # available_buying_power reads put_strike off the live positions
        assert back.available_buying_power() == t.available_buying_power()
        # and a new trade can still be opened on the reloaded tracker
        assert back.open_short_put("NEW1", 120.0, 1.5, _ENTRY, _EXPIRY, 0.25)
        assert back.positions["NEW1"].state == PositionState.SHORT_PUT


# ======================================================================
# 4. save / load through a JSON file
# ======================================================================
class TestSaveLoad:
    def test_save_writes_valid_json(self, tmp_path):
        t = _make_populated_tracker()
        path = tmp_path / "tracker.json"
        t.save(path)
        assert path.exists()
        with open(path, encoding="utf-8") as fh:
            reloaded = json.load(fh)  # valid JSON
        assert reloaded["schema_version"] == 1

    def test_save_load_round_trip(self, tmp_path):
        t = _make_populated_tracker()
        path = tmp_path / "tracker.json"
        t.save(path)
        back = WheelTracker.load(path)
        assert back.to_dict() == t.to_dict()

    def test_load_reattaches_connector(self, tmp_path):
        sentinel = object()
        t = WheelTracker(initial_capital=100_000.0)
        _open_put(t, "AAA", strike=180.0)
        path = tmp_path / "tracker.json"
        t.save(path)
        back = WheelTracker.load(path, connector=sentinel)
        assert back.connector is sentinel

    def test_load_accepts_str_path(self, tmp_path):
        t = WheelTracker(initial_capital=50_000.0)
        path = tmp_path / "t.json"
        t.save(str(path))
        back = WheelTracker.load(str(path))
        assert back.cash == t.cash
