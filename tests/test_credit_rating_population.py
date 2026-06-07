"""R0a regression guard: ``analyze_ticker`` must populate ``credit_rating`` from
the ``get_credit_risk()`` ``"sp_rating"`` key.

The connector maps the raw Bloomberg field ``rtg_sp_lt_lc_issuer_credit`` to the
friendly key ``sp_rating`` (``data_connector.get_credit_risk``). The pre-fix code
read the *raw* field name, so ``credit_rating`` was silently ``""`` for every
ticker (a dead read — ``sp500_credit_risk.csv`` wasted). This pins that we read
the key the connector actually returns.

This field is OFF the EV-authoritative path: ``credit_rating`` feeds only the
legacy heuristic ``_compute_wheel_score()`` / ``screen_candidates()`` and the
memo / API display — never ``rank_candidates_by_ev`` / ``EVEngine.evaluate`` — so
this fix moves no EV ranking and no backtest snapshot.
"""

from __future__ import annotations

import pandas as pd

from engine.wheel_runner import WheelRunner


class _CreditOnlyConnector:
    """Minimal connector stub. ``analyze_ticker`` wraps every other connector
    call in try/except, so only ``get_fundamentals`` / ``get_credit_risk`` /
    ``get_ohlcv`` need to exist and not raise; the credit assignment happens
    before the no-data early return."""

    def __init__(self, credit: dict | None) -> None:
        self._credit = credit

    def get_fundamentals(self, ticker: str):  # noqa: ARG002
        return None

    def get_credit_risk(self, ticker: str):  # noqa: ARG002
        return self._credit

    def get_ohlcv(self, ticker, *args, **kwargs):  # noqa: ARG002
        return pd.DataFrame()


def _credit_rating_for(credit: dict | None) -> str:
    runner = WheelRunner()
    # `connector` is a lazy read-only property backed by `_connector`; inject the
    # stub before first access so analyze_ticker uses it.
    runner._connector = _CreditOnlyConnector(credit)
    return runner.analyze_ticker("TEST").credit_rating


def test_credit_rating_populated_from_sp_rating():
    """The fix: read the friendly ``sp_rating`` key the connector returns."""
    assert _credit_rating_for({"sp_rating": "A+", "altman_z_score": 3.0}) == "A+"


def test_credit_rating_ignores_raw_bloomberg_field_name():
    """Regression guard: the pre-fix dead-read keyed off the raw Bloomberg field
    name. The connector never returns that key (it maps it to ``sp_rating``), so
    reading it yields ``""``. If this assertion ever fails, the code regressed to
    reading the raw field."""
    assert _credit_rating_for({"rtg_sp_lt_lc_issuer_credit": "A+"}) == ""


def test_credit_rating_empty_when_no_credit_data():
    """No credit dict -> credit_rating stays at its "" default."""
    assert _credit_rating_for(None) == ""
