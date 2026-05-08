"""Targeted coverage tests for engine/theta_connector.py — D10 floor lift.

Background
----------
The 2026-05-08 audit pinned engine.theta_connector at 79% line coverage,
1pp below the global ``--cov-fail-under=80`` gate. This module's existing
test surface (``test_theta_connector_v3.py``, 87 tests) already covers
the D11 PerEndpointFailure contract end-to-end and most happy paths.

This file extends coverage to the previously-untested edge branches:

  * Cache-eviction loop in ``_set_cached_chain``.
  * OHLCV column-padding when Theta returns close-only.
  * ``get_option_chain`` mid-pricing with bid-only / no-quote chains.
  * ``get_fundamentals`` happy path that injects a live ATM IV, plus
    the generic-Exception branch where the chain produces a non-numeric
    delta.
  * ``get_iv_rank`` live path with the SAME requests-mock URL the
    connector actually hits (the existing test misses 672-728 because
    its regex requires ``greeks/first_order`` BEFORE ``snapshot``,
    which is the wrong ordering for ``/v3/option/snapshot/greeks/first_order``).
  * ``get_vix_regime`` no-price-column branch + PerEndpointFailure
    propagation + generic-exception branch.
  * ``get_vol_risk_premium`` happy path that returns a finite
    IV - RV value, plus PerEndpointFailure propagation.
  * ``get_iv_surface`` sub-branches for empty chains and missing
    ``strike``/``right`` columns.
  * ``get_atm_term_structure`` puts-empty branch.
  * ``get_skew_snapshot`` ``_pick`` returns None branches.
  * ``get_vix_family`` no-price-column / empty-numeric / per-symbol
    Exception branches and CBOE/Yahoo failure paths.

Tests are hermetic: every requests call is mocked. Bloomberg fallback
paths use an empty ``tmp_path`` so super() returns empty results,
proving the fallback was taken without needing real CSVs (PR #67 set
that pattern).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import requests_mock as rm_module

from engine.theta_connector import (
    _CHAIN_CACHE_TTL,
    PerEndpointFailure,
    ThetaConnector,
)

THETA_BASE = "http://127.0.0.1:25503"
THETA_RE = re.compile(r"http://127\.0\.0\.1:25503/v3/.*")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock():
    with rm_module.Mocker() as m:
        yield m


@pytest.fixture
def connector(tmp_path: Path) -> ThetaConnector:
    """Connector pointed at an empty tmp_path so Bloomberg fallbacks
    return empty / NaN — we can detect "fell back to CSV" by the
    absence of fresh-Theta values."""
    return ThetaConnector(data_dir=str(tmp_path), base_url=THETA_BASE)


def _expirations_csv(dates: list[str]) -> str:
    return "expiration\n" + "\n".join(dates)


def _greeks_for_chain(strikes: list[float], expiry: str = "20260221") -> str:
    """Greeks CSV with both calls and puts at each strike, valid IV."""
    header = "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
    rows = []
    for k in strikes:
        # Put delta scales linearly so 150 → -0.50 (ATM).
        put_delta = -0.50 - (k - 150.0) * 0.01
        call_delta = 0.50 + (150.0 - k) * 0.01
        rows.append(f"AAPL,{expiry},{k},call,{call_delta},0.02,-0.05,0.20,0.01,0.25,1.50,1.55")
        rows.append(f"AAPL,{expiry},{k},put,{put_delta},0.02,-0.05,0.20,0.01,0.27,1.40,1.45")
    return header + "\n".join(rows)


def _quotes_for_chain(strikes: list[float], expiry: str = "20260221") -> str:
    header = "symbol,expiration,strike,right,bid,ask,bid_size,ask_size\n"
    rows = []
    for k in strikes:
        rows.append(f"AAPL,{expiry},{k},call,1.50,1.55,10,20")
        rows.append(f"AAPL,{expiry},{k},put,1.40,1.45,12,18")
    return header + "\n".join(rows)


def _oi_for_chain(strikes: list[float], expiry: str = "20260221") -> str:
    header = "symbol,expiration,strike,right,open_interest\n"
    rows = []
    for k in strikes:
        rows.append(f"AAPL,{expiry},{k},call,5000")
        rows.append(f"AAPL,{expiry},{k},put,3500")
    return header + "\n".join(rows)


def _stub_chain_endpoints(
    mock: rm_module.Mocker,
    strikes: list[float],
    expiry: str = "20260221",
) -> None:
    """Stub the three endpoints get_option_chain hits.

    Uses requests_mock URL patterns that match the ACTUAL Theta v3 path
    (``snapshot/greeks/first_order``, NOT ``history/greeks/first_order``).
    """
    mock.get(
        re.compile(r".*snapshot/greeks/first_order.*"),
        text=_greeks_for_chain(strikes, expiry),
    )
    mock.get(
        re.compile(r".*snapshot/quote.*"),
        text=_quotes_for_chain(strikes, expiry),
    )
    mock.get(
        re.compile(r".*snapshot/open_interest.*"),
        text=_oi_for_chain(strikes, expiry),
    )


# ---------------------------------------------------------------------------
# _set_cached_chain — stale-eviction loop (line 332)
# ---------------------------------------------------------------------------


class TestCacheEviction:
    """Pin the eviction loop in ``_set_cached_chain`` (line 332).

    Without an entry old enough to be stale, the eviction `for k in stale`
    loop body never executes; that's the missing line from the audit.
    """

    def test_stale_entry_is_evicted(self, connector: ThetaConnector):
        df = pd.DataFrame({"a": [1]})

        # Inject a stale entry directly so the eviction sweep finds it.
        # _CHAIN_CACHE_TTL * 2 + 1 seconds in the past = stale.
        import time as _time

        stale_ts = _time.time() - (_CHAIN_CACHE_TTL * 2 + 1)
        connector._chain_cache["STALE|20200101|0"] = (stale_ts, df)

        # Now set a new entry — that triggers the eviction sweep.
        connector._set_cached_chain("AAPL", "20260221", df)

        # Stale entry is gone; fresh entry stays.
        assert "STALE|20200101|0" not in connector._chain_cache
        # Fresh entry is reachable.
        assert connector._get_cached_chain("AAPL", "20260221") is not None


# ---------------------------------------------------------------------------
# get_ohlcv — column padding when Theta omits a column (line 425)
# ---------------------------------------------------------------------------


class TestGetOhlcvColumnPadding:
    def test_missing_volume_column_padded_with_nan(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Theta returns date/open/high/low/close (no volume) → connector
        pads volume with NaN rather than dropping the row."""
        text = (
            "symbol,date,open,high,low,close\n"
            "AAPL,20260102,150.0,152.0,149.0,151.0\n"
            "AAPL,20260103,151.0,153.5,150.5,153.0\n"
        )
        mock.get(re.compile(r".*stock/history/eod.*"), text=text)
        df = connector.get_ohlcv("AAPL")
        assert not df.empty
        # Volume column exists and is NaN.
        assert "volume" in df.columns
        assert df["volume"].isna().all()


# ---------------------------------------------------------------------------
# get_option_chain — mid-pricing branches (lines 561-564)
# ---------------------------------------------------------------------------


class TestOptionChainMidBranches:
    def test_mid_uses_bid_when_no_ask(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """Greeks endpoint returns bid only (no ask). The connector strips
        bid from greeks so it's sourced from quotes — but if quotes only
        has bid, the mid path takes the ``elif "bid"`` branch."""
        # Greeks: NO bid/ask cols at all; will rely on quotes for them.
        greeks_text = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv\n"
            "AAPL,20260221,150.0,call,0.5,0.02,-0.05,0.20,0.01,0.25\n"
            "AAPL,20260221,150.0,put,-0.5,0.02,-0.05,0.20,0.01,0.27\n"
        )
        # Quotes: only bid, no ask.
        quotes_text = (
            "symbol,expiration,strike,right,bid\n"
            "AAPL,20260221,150.0,call,1.50\n"
            "AAPL,20260221,150.0,put,1.40\n"
        )
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=greeks_text)
        mock.get(re.compile(r".*snapshot/quote.*"), text=quotes_text)
        mock.get(re.compile(r".*snapshot/open_interest.*"), status_code=404)

        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert not df.empty
        assert "mid" in df.columns
        # mid == bid because ask is missing
        assert df["mid"].dropna().iloc[0] == pytest.approx(1.50)

    def test_mid_is_nan_when_no_quotes_and_no_bid_in_greeks(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Greeks has no bid/ask, quotes is empty → mid is NaN."""
        greeks_text = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv\n"
            "AAPL,20260221,150.0,call,0.5,0.02,-0.05,0.20,0.01,0.25\n"
        )
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=greeks_text)
        mock.get(re.compile(r".*snapshot/quote.*"), status_code=404)
        mock.get(re.compile(r".*snapshot/open_interest.*"), status_code=404)

        df = connector.get_option_chain("AAPL", expiration="20260221")
        assert not df.empty
        # mid column exists but is NaN
        assert "mid" in df.columns
        assert df["mid"].isna().all()


# ---------------------------------------------------------------------------
# get_fundamentals — happy path with live ATM IV (lines 593-604)
#                  + generic-Exception branch (lines 612-613)
# ---------------------------------------------------------------------------


class TestGetFundamentalsLiveIVPath:
    def test_atm_iv_overlay_when_chain_has_valid_puts(self, mock: rm_module.Mocker, tmp_path: Path):
        """Bloomberg CSV has the ticker → super() returns dict; then chain
        produces a put close to -0.50 delta with valid IV → connector
        injects implied_vol_atm and volatility_30d.

        Hits lines 593-604 (the put-selection + IV inject branch)."""
        # Write a minimal fundamentals CSV that super() will load.
        bloom_dir = tmp_path
        # MarketDataConnector loads via _load — it expects a specific name.
        # Easier to monkey-patch super().get_fundamentals to return a real dict.
        connector = ThetaConnector(data_dir=str(bloom_dir), base_url=THETA_BASE)

        with patch.object(
            ThetaConnector.__bases__[0],
            "get_fundamentals",
            return_value={"ticker": "AAPL", "pe_ratio": 25.0},
        ):
            future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
            mock.get(
                re.compile(r".*list/expirations.*"),
                text=_expirations_csv([future]),
            )
            _stub_chain_endpoints(mock, [140.0, 150.0, 160.0])

            result = connector.get_fundamentals("AAPL")

        assert result is not None
        # Live IV got injected — both fields set to the put-IV value.
        assert "implied_vol_atm" in result
        assert "volatility_30d" in result
        # Original csv field preserved
        assert result["pe_ratio"] == 25.0
        # Sanity: IV is a decimal in (0, 3].
        assert 0 < result["implied_vol_atm"] <= 3.0

    def test_atm_iv_overlay_creates_dict_when_super_returns_none(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Bloomberg CSV missing → super() returns None; chain still has IV
        → connector creates a {ticker, implied_vol_atm} dict.

        Hits the ``if base is None: base = {"ticker": ticker}`` branch
        on line 600-601.
        """
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        _stub_chain_endpoints(mock, [140.0, 150.0, 160.0])

        result = connector.get_fundamentals("AAPL")

        assert result is not None
        assert result["ticker"] == "AAPL"
        assert "implied_vol_atm" in result

    def test_invalid_iv_does_not_inject(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """If the ATM-row IV is outside (0, 3] (e.g., zero or negative), the
        connector skips injection. Combined with super() returning None,
        the final result is None.
        """
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        # Build greeks where IV is zero — fails the ``0 < live_iv <= 3.0`` guard.
        zero_iv_greeks = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
            "AAPL,20260221,150.0,put,-0.50,0.02,-0.05,0.20,0.01,0.0,1.40,1.45\n"
        )
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=zero_iv_greeks)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_for_chain([150.0]))
        mock.get(re.compile(r".*snapshot/open_interest.*"), text=_oi_for_chain([150.0]))

        result = connector.get_fundamentals("AAPL")
        # super() returned None and we did NOT inject (IV=0 fails guard) →
        # base is still None.
        assert result is None

    def test_generic_exception_during_chain_fetch_swallowed(self, connector: ThetaConnector):
        """A non-PerEndpointFailure raised by get_option_chain is swallowed
        so the CSV-only fundamentals dict still flows out (lines 612-613).
        """
        with patch.object(
            ThetaConnector.__bases__[0],
            "get_fundamentals",
            return_value={"ticker": "AAPL", "pe_ratio": 25.0},
        ):
            with patch.object(
                connector,
                "get_option_chain",
                side_effect=ValueError("boom"),
            ):
                result = connector.get_fundamentals("AAPL")
        # CSV dict survived, no IV overlay since chain raised.
        assert result == {"ticker": "AAPL", "pe_ratio": 25.0}


# ---------------------------------------------------------------------------
# get_iv_rank — live happy path + propagation branches (lines 635-642)
# ---------------------------------------------------------------------------


class TestGetIvRankLivePath:
    def test_live_path_with_valid_history_returns_rank(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """End-to-end live path: chain → ATM strike → 12 monthly chunks of
        IV history → rank returned in [0, 1].

        Covers 635-637 (the rank computation) AND most of 672-728 (the
        ``_fetch_iv_history`` body, which the existing test missed because
        its regex didn't match the live ``snapshot/greeks/first_order``
        URL).
        """
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        _stub_chain_endpoints(mock, [140.0, 150.0, 160.0])

        # Build IV history with enough points (≥20 across all monthly chunks).
        # The IV history endpoints take params and return per-bar rows.
        # We mock both the dedicated IV endpoint (returns iv directly) and
        # the first_order fallback. Either path should populate >= 20 bars.
        # Use the ``timestamp`` column the connector recognises.
        iv_rows = []
        base_ts = pd.Timestamp("2025-06-01")
        for i in range(60):
            ts = base_ts + pd.Timedelta(days=i)
            # Cyclic IV pattern in (0.20, 0.30) so rank lands strictly in (0, 1).
            iv = 0.20 + 0.10 * ((i % 5) / 4.0)
            iv_rows.append(f"{ts.isoformat()},{iv}")
        iv_history_text = "timestamp,iv\n" + "\n".join(iv_rows)

        mock.get(
            re.compile(r".*history/greeks/implied_volatility.*"),
            text=iv_history_text,
        )
        mock.get(
            re.compile(r".*history/greeks/first_order.*"),
            text=iv_history_text,
        )

        rank = connector.get_iv_rank("AAPL")
        assert isinstance(rank, float)
        assert 0.0 <= rank <= 1.0
        # Sanity: rank is rounded to 4 decimals.
        assert rank == round(rank, 4)

    def test_per_endpoint_failure_propagates(self, connector: ThetaConnector):
        """A PerEndpointFailure raised by _fetch_iv_history must propagate
        through get_iv_rank (line 638-639), NOT be swallowed by the generic
        Exception clause.

        We patch ``_fetch_iv_history`` directly to raise — easier than
        sequencing requests-mock matchers around the SPY-probe URL pattern,
        which collides with the per-symbol expirations URL.
        """
        from engine.theta_connector import FailureRecord

        record = FailureRecord(
            timestamp_utc="2026-05-08T00:00:00+00:00",
            endpoint="/v3/option/list/expirations",
            params={"symbol": "AAPL"},
            reason="ConnectionError: simulated",
        )
        with patch.object(
            connector,
            "_fetch_iv_history",
            side_effect=PerEndpointFailure(record),
        ):
            with pytest.raises(PerEndpointFailure):
                connector.get_iv_rank("AAPL")

    def test_generic_exception_falls_back_to_super(self, connector: ThetaConnector):
        """A non-PerEndpointFailure raised inside the live path falls back
        to super().get_iv_rank (lines 640-642)."""
        with patch.object(connector, "_fetch_iv_history", side_effect=ValueError("boom")):
            # super() returns NaN with no CSV in tmp_path.
            rank = connector.get_iv_rank("AAPL")
        # Either NaN or some real fallback value — what matters is that
        # the call did NOT raise.
        assert isinstance(rank, float)


# ---------------------------------------------------------------------------
# _fetch_iv_history — direct invocation for branches not reached above
# ---------------------------------------------------------------------------


class TestFetchIvHistoryDirect:
    def test_returns_none_when_no_expiration(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """If _nearest_expiration returns None, _fetch_iv_history returns
        None at line 666."""
        mock.get(re.compile(r".*list/expirations.*"), text="expiration\n")
        out = connector._fetch_iv_history("AAPL")
        assert out is None

    def test_returns_none_when_chain_lacks_delta(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Chain has no ``delta`` column → _fetch_iv_history returns None
        on line 670."""
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # Greeks without delta column.
        no_delta = (
            "symbol,expiration,strike,right,gamma,theta,vega,rho,iv,bid,ask\n"
            "AAPL,20260221,150.0,put,0.02,-0.05,0.20,0.01,0.27,1.40,1.45\n"
        )
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=no_delta)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_for_chain([150.0]))
        mock.get(re.compile(r".*snapshot/open_interest.*"), status_code=404)

        out = connector._fetch_iv_history("AAPL")
        assert out is None

    def test_returns_none_when_no_puts_in_chain(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Chain has only calls (no puts) → puts.empty → returns None
        at line 674."""
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # Only calls with valid delta.
        calls_only = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
            "AAPL,20260221,150.0,call,0.5,0.02,-0.05,0.20,0.01,0.25,1.50,1.55\n"
        )
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=calls_only)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_for_chain([150.0]))
        mock.get(re.compile(r".*snapshot/open_interest.*"), status_code=404)

        out = connector._fetch_iv_history("AAPL")
        assert out is None


# ---------------------------------------------------------------------------
# get_vix_regime — branches at lines 749, 758-762
# ---------------------------------------------------------------------------


class TestGetVixRegimeBranches:
    def test_no_price_column_falls_back_to_super(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Theta response has columns but none are recognised price columns
        → falls back to super() (line 749)."""
        # Response has columns 'symbol' and 'foo' but no price/close/last/value.
        mock.get(re.compile(r".*index/snapshot/price.*"), text="symbol,foo\nVIX,bar\n")
        out = connector.get_vix_regime()
        # super() was called → dict shape preserved
        assert isinstance(out, dict)
        # The Theta-injected vix key is absent because we never reached it.
        # (super() in tmp_path returns its own structure with vix key, so
        # we just check we got a dict back without crashing.)

    def test_per_endpoint_failure_propagates(self, connector: ThetaConnector):
        """A PerEndpointFailure during VIX fetch propagates (line 758-759).

        Patches ``_fetch`` directly so we don't have to sequence the
        SPY-probe URL ordering against the index-snapshot URL in
        requests-mock.
        """
        from engine.theta_connector import FailureRecord

        record = FailureRecord(
            timestamp_utc="2026-05-08T00:00:00+00:00",
            endpoint="/v3/index/snapshot/price",
            params={"symbol": "VIX"},
            reason="Timeout: simulated",
        )
        with patch.object(connector, "_fetch", side_effect=PerEndpointFailure(record)):
            with pytest.raises(PerEndpointFailure):
                connector.get_vix_regime()

    def test_generic_exception_falls_back_to_super(self, connector: ThetaConnector):
        """A non-PerEndpointFailure raised in the live path falls back to
        super() (lines 760-762)."""
        with patch.object(connector, "_fetch", side_effect=ValueError("boom")):
            out = connector.get_vix_regime()
        assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# get_vol_risk_premium — happy path + propagation (lines 786-790)
# ---------------------------------------------------------------------------


class TestGetVolRiskPremiumBranches:
    def test_full_path_returns_finite_premium(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """30+ days of OHLCV + chain returns ATM IV → premium = IV - RV."""
        # Build 35 daily bars with controlled returns so realised vol is
        # finite and small.
        base = pd.Timestamp("2026-01-01")
        rows = []
        for i in range(35):
            d = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
            close = 150.0 + 0.1 * (i % 3)
            rows.append(f"AAPL,{d},{close - 1},{close + 1},{close - 2},{close},1000000")
        ohlcv_text = "symbol,date,open,high,low,close,volume\n" + "\n".join(rows)
        mock.get(re.compile(r".*stock/history/eod.*"), text=ohlcv_text)

        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        _stub_chain_endpoints(mock, [140.0, 150.0, 160.0])

        result = connector.get_vol_risk_premium("AAPL")
        assert isinstance(result, float)
        # Result should be finite — IV is 0.27, realised vol from low-noise
        # series is small, so premium > 0.
        import math

        assert math.isfinite(result)

    def test_per_endpoint_failure_propagates(self, connector: ThetaConnector):
        """A PerEndpointFailure during OHLCV fetch propagates (787-788).

        Patches ``get_ohlcv`` directly so we don't have to sequence URL
        matchers around the SPY probe.
        """
        from engine.theta_connector import FailureRecord

        record = FailureRecord(
            timestamp_utc="2026-05-08T00:00:00+00:00",
            endpoint="/v3/stock/history/eod",
            params={"symbol": "AAPL"},
            reason="ReadTimeout: simulated",
        )
        with patch.object(connector, "get_ohlcv", side_effect=PerEndpointFailure(record)):
            with pytest.raises(PerEndpointFailure):
                connector.get_vol_risk_premium("AAPL")

    def test_generic_exception_falls_back_to_super(self, connector: ThetaConnector):
        """A non-PerEndpointFailure raised in the live path falls back
        to super() (lines 789-790)."""
        with patch.object(connector, "get_ohlcv", side_effect=ValueError("boom")):
            out = connector.get_vol_risk_premium("AAPL")
        assert isinstance(out, float)


# ---------------------------------------------------------------------------
# get_iv_surface — empty chain, missing strike/right, fill columns (831/835/840/846)
# ---------------------------------------------------------------------------


class TestGetIvSurfaceBranches:
    def test_chain_empty_skipped(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """Expirations endpoint returns valid future date but the chain
        endpoints all 404 → chain.empty → continue (line 831)."""
        future = (datetime.now(UTC).date() + timedelta(days=45)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # Chain endpoints all 404
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), status_code=404)
        mock.get(re.compile(r".*snapshot/quote.*"), status_code=404)
        mock.get(re.compile(r".*snapshot/open_interest.*"), status_code=404)
        df = connector.get_iv_surface("AAPL")
        # All expirations had empty chains → final frames is empty.
        assert df.empty

    def test_chain_missing_strike_or_right_skipped(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """If the chain DataFrame has neither ``strike`` nor ``right``,
        the iteration continues without appending (line 835)."""
        future = (datetime.now(UTC).date() + timedelta(days=45)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))

        # Patch get_option_chain to return a DF without strike/right.
        bad_chain = pd.DataFrame({"symbol": ["AAPL"], "expiration": ["20260221"], "iv": [0.25]})
        with patch.object(connector, "get_option_chain", return_value=bad_chain):
            df = connector.get_iv_surface("AAPL")
        assert df.empty

    def test_fills_missing_columns_with_nan(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """If the chain has strike+right but lacks delta/iv/mid, the
        connector fills those with NaN (line 840)."""
        future = (datetime.now(UTC).date() + timedelta(days=45)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))

        partial_chain = pd.DataFrame(
            {
                "strike": [150.0],
                "right": ["put"],
                "iv": [0.25],
                # delta and mid missing
            }
        )
        with patch.object(connector, "get_option_chain", return_value=partial_chain):
            df = connector.get_iv_surface("AAPL")
        assert not df.empty
        assert "delta" in df.columns
        assert df["delta"].isna().all()
        assert "mid" in df.columns
        assert df["mid"].isna().all()


# ---------------------------------------------------------------------------
# get_atm_term_structure — puts-empty branch (line 860)
# ---------------------------------------------------------------------------


class TestGetAtmTermStructureBranches:
    def test_puts_empty_returns_empty_df(self, connector: ThetaConnector):
        """If the surface has no puts (or no rows where right==put with
        valid delta + iv), the function returns an empty DF (line 860)."""
        # Build a surface that only has calls.
        calls_only = pd.DataFrame(
            {
                "strike": [150.0],
                "right": ["call"],
                "delta": [0.50],
                "iv": [0.25],
                "mid": [1.50],
                "expiration": [pd.Timestamp("2026-02-21")],
                "dte": [45],
            }
        )
        with patch.object(connector, "get_iv_surface", return_value=calls_only):
            df = connector.get_atm_term_structure("AAPL")
        assert df.empty


# ---------------------------------------------------------------------------
# get_skew_snapshot — _pick None branches (lines 887, 899)
# ---------------------------------------------------------------------------


class TestGetSkewSnapshotBranches:
    def test_returns_empty_dict_when_a_pick_is_none(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """If any of the 25Δ-put / ATM / 25Δ-call picks fails (e.g.,
        because all rows have NaN delta), the function returns {} on
        line 899."""
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # All puts have valid delta but call side has NaN delta → 25Δ call
        # pick returns None → final dict is empty.
        rows_text = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
            # Put with -0.25 delta
            "AAPL,20260221,140.0,put,-0.25,0.02,-0.05,0.20,0.01,0.30,1.40,1.45\n"
            # Put with -0.50 delta
            "AAPL,20260221,150.0,put,-0.50,0.02,-0.05,0.20,0.01,0.27,1.40,1.45\n"
            # Call with NO valid delta — IV is fine but delta is NaN
            "AAPL,20260221,160.0,call,,0.02,-0.05,0.20,0.01,0.28,1.50,1.55\n"
        )
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=rows_text)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_for_chain([140.0, 150.0, 160.0]))
        mock.get(
            re.compile(r".*snapshot/open_interest.*"), text=_oi_for_chain([140.0, 150.0, 160.0])
        )

        result = connector.get_skew_snapshot("AAPL")
        # 25Δ call is None → final dict is empty
        assert result == {}

    def test_pick_returns_none_when_dataframe_empty_after_dropna(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Cover line 887: ``_pick`` returns None when the dropna leaves
        an empty DataFrame.

        We construct a chain where puts have IV==0 (fails the >0 guard
        in _pick) and calls have valid data — so iv_25d_put is None and
        the final result is {}.

        Actually 887 is: ``if d.empty: return None``. We trigger this by
        making puts with NaN delta (so ``dropna(subset=["delta", "iv"])``
        leaves an empty DataFrame).
        """
        future = (datetime.now(UTC).date() + timedelta(days=35)).isoformat()
        mock.get(re.compile(r".*list/expirations.*"), text=_expirations_csv([future]))
        # Both put rows have NaN delta → dropna → empty d → None
        rows_text = (
            "symbol,expiration,strike,right,delta,gamma,theta,vega,rho,iv,bid,ask\n"
            "AAPL,20260221,140.0,put,,0.02,-0.05,0.20,0.01,0.30,1.40,1.45\n"
            "AAPL,20260221,150.0,put,,0.02,-0.05,0.20,0.01,0.27,1.40,1.45\n"
            "AAPL,20260221,160.0,call,0.25,0.02,-0.05,0.20,0.01,0.28,1.50,1.55\n"
        )
        mock.get(re.compile(r".*snapshot/greeks/first_order.*"), text=rows_text)
        mock.get(re.compile(r".*snapshot/quote.*"), text=_quotes_for_chain([140.0, 150.0, 160.0]))
        mock.get(
            re.compile(r".*snapshot/open_interest.*"), text=_oi_for_chain([140.0, 150.0, 160.0])
        )

        result = connector.get_skew_snapshot("AAPL")
        assert result == {}


# ---------------------------------------------------------------------------
# get_vix_family — branches 945, 948, 950-951, 965-967, 991-992
# ---------------------------------------------------------------------------


class TestGetVixFamilyBranches:
    def test_no_price_column_skips_symbol(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """Theta returns CSV without a recognised price column → continue
        (line 945)."""
        # Each symbol gets columns that don't include any of price/close/last/value.
        mock.get(re.compile(r".*index/snapshot/price.*"), text="symbol,foo\nVIX,bar\n")
        # CBOE/Yahoo also fail
        mock.get(re.compile(r".*cboe\.com.*"), status_code=404)
        mock.get(re.compile(r".*query[12]?\.finance\.yahoo\.com.*"), status_code=404)

        out = connector.get_vix_family()
        # No symbol populated from Theta (no price col) — fallbacks also failed.
        # Either empty dict or a partial — what matters is no crash.
        assert isinstance(out, dict)

    def test_empty_value_after_to_numeric_skips_symbol(
        self, mock: rm_module.Mocker, connector: ThetaConnector
    ):
        """Theta returns price column but values coerce to all-NaN → val
        empty → continue (line 948)."""
        # Price column is non-numeric so ``pd.to_numeric(..., errors='coerce')``
        # produces NaN; ``.dropna()`` empties the series.
        mock.get(
            re.compile(r".*index/snapshot/price.*"),
            text="symbol,price\nVIX,not-a-number\n",
        )
        mock.get(re.compile(r".*cboe\.com.*"), status_code=404)
        mock.get(re.compile(r".*query[12]?\.finance\.yahoo\.com.*"), status_code=404)

        out = connector.get_vix_family()
        assert isinstance(out, dict)

    def test_per_symbol_exception_does_not_crash_loop(self, connector: ThetaConnector):
        """A non-PerEndpointFailure exception inside the per-symbol loop
        is caught and the next symbol is tried (lines 950-951).

        We patch _fetch to raise on every call so the inner ``try/except``
        is the only thing keeping the loop alive."""
        from engine.theta_connector import ThetaConnector as _TC

        with patch.object(_TC, "_fetch", side_effect=ValueError("boom")):
            with patch("engine.external_data.cboe_adapter.CBOEAdapter") as cboe_cls:
                # Force CBOE to raise immediately so the CBOE block's
                # except branch runs (covers 965-967).
                cboe_cls.side_effect = RuntimeError("cboe boom")
                with patch("engine.external_data.yfinance_adapter.YFinanceAdapter") as yf_cls:
                    yf_cls.side_effect = RuntimeError("yf boom")
                    out = connector.get_vix_family()
        assert isinstance(out, dict)

    def test_cboe_block_failure_swallowed(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """If the CBOE adapter raises during construction or .latest(),
        the except clause logs and moves on (lines 965-967)."""
        # Theta 403s for everything → all symbols fall to CBOE.
        mock.get(re.compile(r".*index/snapshot/price.*"), status_code=403)

        with patch("engine.external_data.cboe_adapter.CBOEAdapter") as cboe_cls:
            cboe_cls.side_effect = RuntimeError("cboe broken")
            with patch("engine.external_data.yfinance_adapter.YFinanceAdapter") as yf_cls:
                yf_cls.side_effect = RuntimeError("yf broken")
                out = connector.get_vix_family()
        assert isinstance(out, dict)

    def test_yahoo_block_failure_swallowed(self, mock: rm_module.Mocker, connector: ThetaConnector):
        """If the Yahoo adapter raises during construction or .latest_close(),
        the except clause logs and moves on (lines 991-992)."""
        # Theta 403, CBOE returns nothing useful, Yahoo blows up.
        mock.get(re.compile(r".*index/snapshot/price.*"), status_code=403)

        with patch("engine.external_data.cboe_adapter.CBOEAdapter") as cboe_cls:
            cboe_inst = cboe_cls.return_value
            # Return NaN so symbol stays in `still_missing`.
            import math

            cboe_inst.latest.return_value = math.nan
            with patch("engine.external_data.yfinance_adapter.YFinanceAdapter") as yf_cls:
                yf_cls.side_effect = RuntimeError("yf broken")
                out = connector.get_vix_family()
        assert isinstance(out, dict)
