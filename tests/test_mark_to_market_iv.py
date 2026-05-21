"""Tests for the mark-to-market IV staleness fix.

Issue #118 P4. S2 / S8 logged that ``WheelTracker.mark_to_market``, when
no ``current_ivs`` dict is passed, marks short-option liabilities at the
position's *entry* IV — stale, so a position held through a vol regime
change is mis-marked. The fix plumbs the connector's as-of IV: the
fallback chain is now

    explicit current_ivs  →  connector as-of ATM IV  →  entry IV.

Pinned here:
  * ``_connector_atm_iv`` — composite (put+call)/2, percent→decimal,
    defensive (no connector / no method / failure / empty → None);
  * ``_resolve_mark_iv`` — the three-step priority;
  * ``mark_to_market`` marks at the connector IV when ``current_ivs``
    is absent — identical to passing that IV explicitly, and distinct
    from the stale entry-IV fallback.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.wheel_tracker import PositionState, WheelTracker

_ENTRY = date(2026, 1, 5)
_EXPIRY = date(2026, 3, 20)
_CC_EXPIRY = date(2026, 5, 15)
_MARK_DATE = date(2026, 1, 20)


# ----------------------------------------------------------------------
# Connector stubs
# ----------------------------------------------------------------------
class _FakeIVConn:
    """Connector stub exposing only ``get_iv_history``."""

    def __init__(
        self,
        iv_pct: float = 35.0,
        *,
        raises: bool = False,
        empty: bool = False,
        missing_cols: bool = False,
        rows: tuple[float, ...] | None = None,
    ) -> None:
        self._iv_pct = iv_pct
        self._raises = raises
        self._empty = empty
        self._missing_cols = missing_cols
        self._rows = rows

    def get_iv_history(self, ticker, start_date=None, end_date=None):
        if self._raises:
            raise RuntimeError("iv history unavailable")
        if self._empty:
            return pd.DataFrame()
        vals = list(self._rows) if self._rows is not None else [self._iv_pct, self._iv_pct]
        idx = pd.to_datetime([f"2026-01-{5 + i:02d}" for i in range(len(vals))])
        if self._missing_cols:
            return pd.DataFrame({"volatility_30d": vals}, index=idx)
        return pd.DataFrame(
            {"hist_put_imp_vol": vals, "hist_call_imp_vol": vals},
            index=idx,
        )


class _NoIVConn:
    """A connector with no get_iv_history at all (mirrors ThetaConnector)."""

    def get_ohlcv(self, ticker):  # present, but not the IV method
        return pd.DataFrame()


def _tracker(connector=None, capital: float = 100_000.0) -> WheelTracker:
    return WheelTracker(initial_capital=capital, connector=connector)


def _with_short_put(t: WheelTracker, *, entry_iv: float = 0.20) -> WheelTracker:
    assert t.open_short_put("AAA", 200.0, 3.0, _ENTRY, _EXPIRY, entry_iv)
    return t


# ======================================================================
# 1. _connector_atm_iv
# ======================================================================
class TestConnectorAtmIv:
    def test_percent_iv_normalised_to_decimal(self):
        t = _tracker(_FakeIVConn(iv_pct=40.0))
        assert t._connector_atm_iv("AAA", _MARK_DATE) == pytest.approx(0.40)

    def test_already_decimal_iv_passes_through(self):
        t = _tracker(_FakeIVConn(iv_pct=0.28))
        assert t._connector_atm_iv("AAA", _MARK_DATE) == pytest.approx(0.28)

    def test_uses_the_most_recent_row(self):
        # the as-of value is the last row of the (date-filtered) history
        t = _tracker(_FakeIVConn(rows=(25.0, 30.0, 38.0)))
        assert t._connector_atm_iv("AAA", _MARK_DATE) == pytest.approx(0.38)

    def test_none_when_no_connector(self):
        assert _tracker(None)._connector_atm_iv("AAA", _MARK_DATE) is None

    def test_none_when_connector_lacks_get_iv_history(self):
        assert _tracker(_NoIVConn())._connector_atm_iv("AAA", _MARK_DATE) is None

    def test_none_when_get_iv_history_raises(self):
        assert _tracker(_FakeIVConn(raises=True))._connector_atm_iv("AAA", _MARK_DATE) is None

    def test_none_when_history_empty(self):
        assert _tracker(_FakeIVConn(empty=True))._connector_atm_iv("AAA", _MARK_DATE) is None

    def test_none_when_iv_columns_absent(self):
        assert _tracker(_FakeIVConn(missing_cols=True))._connector_atm_iv("AAA", _MARK_DATE) is None

    def test_none_when_iv_degenerate(self):
        # 600(%) → 6.0 decimal, outside the (0, 5] sanity band
        assert _tracker(_FakeIVConn(iv_pct=600.0))._connector_atm_iv("AAA", _MARK_DATE) is None


# ======================================================================
# 2. _resolve_mark_iv — the three-step priority
# ======================================================================
class TestResolveMarkIv:
    def test_current_ivs_override_wins(self):
        t = _tracker(_FakeIVConn(iv_pct=35.0))
        # current_ivs present → used even though a connector IV exists
        assert t._resolve_mark_iv("AAA", _MARK_DATE, 0.20, {"AAA": 0.55}) == pytest.approx(0.55)

    def test_connector_iv_used_when_current_ivs_absent(self):
        t = _tracker(_FakeIVConn(iv_pct=35.0))
        assert t._resolve_mark_iv("AAA", _MARK_DATE, 0.20, {}) == pytest.approx(0.35)

    def test_entry_iv_is_the_last_resort(self):
        # no connector → neither override nor connector IV → stale entry IV
        t = _tracker(None)
        assert t._resolve_mark_iv("AAA", _MARK_DATE, 0.20, {}) == pytest.approx(0.20)

    def test_nonpositive_current_iv_is_skipped(self):
        t = _tracker(_FakeIVConn(iv_pct=35.0))
        # a 0.0 override is not a real IV → fall through to the connector
        assert t._resolve_mark_iv("AAA", _MARK_DATE, 0.20, {"AAA": 0.0}) == pytest.approx(0.35)


# ======================================================================
# 3. mark_to_market integration
# ======================================================================
class TestMarkToMarketUsesConnectorIv:
    def test_connector_iv_mark_equals_explicit_iv_mark(self):
        """With a connector and no current_ivs, mark_to_market marks at
        the connector's as-of IV — the same value, and the same
        portfolio mark, as passing that IV explicitly."""
        conn_tracker = _with_short_put(_tracker(_FakeIVConn(iv_pct=35.0)), entry_iv=0.20)
        explicit_tracker = _with_short_put(_tracker(None), entry_iv=0.20)

        pv_connector = conn_tracker.mark_to_market(_MARK_DATE, {"AAA": 198.0})
        pv_explicit = explicit_tracker.mark_to_market(
            _MARK_DATE, {"AAA": 198.0}, current_ivs={"AAA": 0.35}
        )

        assert pv_connector == pytest.approx(pv_explicit)

    def test_connector_iv_differs_from_stale_entry_iv_mark(self):
        """The whole point: the connector IV (0.35) produces a different
        mark than the stale entry IV (0.20) — staleness actually moved."""
        conn_tracker = _with_short_put(_tracker(_FakeIVConn(iv_pct=35.0)), entry_iv=0.20)
        stale_tracker = _with_short_put(_tracker(None), entry_iv=0.20)

        pv_connector = conn_tracker.mark_to_market(_MARK_DATE, {"AAA": 198.0})
        pv_stale = stale_tracker.mark_to_market(_MARK_DATE, {"AAA": 198.0})

        assert pv_connector != pytest.approx(pv_stale)
        # higher IV → larger short-put liability → lower portfolio value
        assert pv_connector < pv_stale

    def test_explicit_current_ivs_still_overrides_the_connector(self):
        conn_tracker = _with_short_put(_tracker(_FakeIVConn(iv_pct=35.0)), entry_iv=0.20)
        override_tracker = _with_short_put(_tracker(None), entry_iv=0.20)

        pv_override = conn_tracker.mark_to_market(
            _MARK_DATE, {"AAA": 198.0}, current_ivs={"AAA": 0.50}
        )
        pv_expected = override_tracker.mark_to_market(
            _MARK_DATE, {"AAA": 198.0}, current_ivs={"AAA": 0.50}
        )

        assert pv_override == pytest.approx(pv_expected)

    def test_no_connector_falls_back_to_entry_iv(self):
        """Legacy behaviour preserved: no connector, no current_ivs →
        the entry IV is still used."""
        no_conn = _with_short_put(_tracker(None), entry_iv=0.22)
        explicit = _with_short_put(_tracker(None), entry_iv=0.99)
        pv_fallback = no_conn.mark_to_market(_MARK_DATE, {"AAA": 198.0})
        pv_at_entry_iv = explicit.mark_to_market(
            _MARK_DATE, {"AAA": 198.0}, current_ivs={"AAA": 0.22}
        )
        assert pv_fallback == pytest.approx(pv_at_entry_iv)

    def test_covered_call_leg_also_uses_resolved_iv(self):
        """The call leg goes through the same _resolve_mark_iv path."""

        def _covered_call_tracker(connector) -> WheelTracker:
            t = _tracker(connector)
            assert t.open_short_put("AAA", 200.0, 3.0, _ENTRY, _EXPIRY, 0.20)
            assert t.handle_put_assignment("AAA", _EXPIRY, 198.0)
            assert t.open_covered_call("AAA", 215.0, 2.0, _EXPIRY, _CC_EXPIRY, 0.20)
            assert t.positions["AAA"].state == PositionState.COVERED_CALL
            return t

        conn_tracker = _covered_call_tracker(_FakeIVConn(iv_pct=35.0))
        explicit_tracker = _covered_call_tracker(None)
        mark_date = date(2026, 4, 1)  # inside the covered-call window
        pv_connector = conn_tracker.mark_to_market(mark_date, {"AAA": 210.0})
        pv_explicit = explicit_tracker.mark_to_market(
            mark_date, {"AAA": 210.0}, current_ivs={"AAA": 0.35}
        )
        assert pv_connector == pytest.approx(pv_explicit)

    def test_equity_curve_still_recorded(self):
        """Regression: the IV-resolution refactor did not break the
        equity-curve append."""
        t = _with_short_put(_tracker(_FakeIVConn(iv_pct=35.0)), entry_iv=0.20)
        t.mark_to_market(_MARK_DATE, {"AAA": 198.0})
        assert len(t.equity_curve) == 1
        assert t.equity_curve[0]["date"] == _MARK_DATE
