"""Tests for the real-premium ranker wiring (Phase-2 EV unlock).

Covers the split-adjustment that maps the RAW Theta larder into the engine's
split-adjusted frame, the ``_resolve_real_premium`` snap helper, and the
end-to-end ranker behaviour: when the option-premium rail covers a candidate the
ranker uses the real market mid (``premium_source="market_mid"``); when it is
absent the synthetic-BSM path is unchanged (byte-identical fallback).

§2: this only changes the *premium input* to ``EVEngine.evaluate`` — never
bypasses evaluate, never rescues a candidate. The empirical forward distribution
(realized returns) is unchanged, so real skew premium is not double-counted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from engine.data_connector import (
    OPTION_PREMIUM_COLUMNS,
    MarketDataConnector,
    _cumulative_split_factor,
)
from engine.wheel_runner import WheelRunner, _resolve_real_premium

REPO = Path(__file__).resolve().parent.parent
_OHLCV = REPO / "data" / "bloomberg" / "sp500_ohlcv.csv"


# ---------------------------------------------------------------------------
# Split-factor math
# ---------------------------------------------------------------------------


class TestCumulativeSplitFactor:
    def test_no_splits_all_ones(self):
        f = _cumulative_split_factor(pd.to_datetime(["2023-01-01", "2024-01-01"]), [], [])
        assert (f == 1.0).all()

    def test_split_after_date_applies_before_only(self):
        dates = pd.to_datetime(["2023-05-25", "2024-07-01"])
        eff = pd.to_datetime(["2024-06-10"]).to_numpy()
        f = _cumulative_split_factor(dates, eff, [10.0])
        assert f[0] == 10.0  # quote before split → raw / 10
        assert f[1] == 1.0  # quote after split → unchanged

    def test_quote_on_effective_date_is_post_split(self):
        eff = pd.to_datetime(["2024-06-10"]).to_numpy()
        f = _cumulative_split_factor(pd.to_datetime(["2024-06-10"]), eff, [10.0])
        assert f[0] == 1.0

    def test_multiple_splits_compound(self):
        eff = pd.to_datetime(["2014-06-09", "2020-08-31"]).to_numpy()
        f = _cumulative_split_factor(pd.to_datetime(["2005-01-01"]), eff, [7.0, 4.0])
        assert f[0] == pytest.approx(28.0)  # both splits ahead → 7 * 4


# ---------------------------------------------------------------------------
# Split-adjustment on load (accessor maps raw → engine frame)
# ---------------------------------------------------------------------------


def _write_produced(d: Path, ticker: str, rows: list[dict]):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df[list(OPTION_PREMIUM_COLUMNS)].to_parquet(d / f"{ticker}.parquet", index=False)


class TestSplitAdjustOnLoad:
    def test_load_back_adjusts_raw_strikes(self, tmp_path, monkeypatch):
        d = tmp_path / "op"
        d.mkdir()
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(d))
        # A raw quote dated BEFORE a 10:1 split: strike 300, mid 10.2.
        _write_produced(
            d,
            "SPL",
            [
                {
                    "date": "2024-01-02",
                    "expiration": "2024-02-16",
                    "dte": 45,
                    "strike": 300.0,
                    "right": "put",
                    "bid": 10.0,
                    "ask": 10.4,
                    "mid": 10.2,
                    "close": 10.1,
                    "volume": 5,
                    "open_interest": 100,
                }
            ],
        )
        c = MarketDataConnector()
        fake_splits = pd.DataFrame(
            {
                "announcement_date": [pd.Timestamp("2024-05-01")],
                "effective_date": [pd.Timestamp("2024-06-10")],
                "action_type": ["Stock Split"],
                "ratio": [10.0],
                "amount": [float("nan")],
            }
        )
        monkeypatch.setattr(c, "get_corporate_actions", lambda *a, **k: fake_splits)
        chain = c.get_option_premium_chain("SPL", "2024-02-16", as_of="2024-01-05")
        assert not chain.empty
        row = chain.iloc[0]
        assert row["strike"] == pytest.approx(30.0)  # 300 / 10
        assert row["mid"] == pytest.approx(1.02)  # 10.2 / 10
        assert row["bid"] == pytest.approx(1.0)

    def test_no_corp_actions_leaves_raw_unchanged(self, tmp_path, monkeypatch):
        d = tmp_path / "op"
        d.mkdir()
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(d))
        _write_produced(
            d,
            "NOS",
            [
                {
                    "date": "2024-01-02",
                    "expiration": "2024-02-16",
                    "dte": 45,
                    "strike": 100.0,
                    "right": "put",
                    "bid": 2.0,
                    "ask": 2.4,
                    "mid": 2.2,
                    "close": 2.1,
                    "volume": 5,
                    "open_interest": 100,
                }
            ],
        )
        c = MarketDataConnector()
        monkeypatch.setattr(c, "get_corporate_actions", lambda *a, **k: pd.DataFrame())
        chain = c.get_option_premium_chain("NOS", "2024-02-16", as_of="2024-01-05")
        assert chain.iloc[0]["strike"] == pytest.approx(100.0)
        assert chain.iloc[0]["mid"] == pytest.approx(2.2)


# ---------------------------------------------------------------------------
# _resolve_real_premium snap helper
# ---------------------------------------------------------------------------


class _StubConn:
    def __init__(self, exps, quote):
        self._exps = exps
        self._q = quote

    def list_option_expirations(self, ticker, as_of=None, **k):
        return self._exps

    def get_option_premium(self, ticker, expiry, strike, right, as_of=None, strike_tol=None):
        return self._q


class TestResolveRealPremium:
    Q = {"mid": 2.2, "bid": 2.0, "ask": 2.4, "strike": 100.0, "right": "put"}

    def test_none_when_no_conn(self):
        assert (
            _resolve_real_premium(None, "X", pd.Timestamp("2024-02-16"), 100, "put", None) is None
        )

    def test_none_when_conn_lacks_method(self):
        class C:
            pass

        assert _resolve_real_premium(C(), "X", pd.Timestamp("2024-02-16"), 100, "put", None) is None

    def test_returns_quote_within_expiry_tol(self):
        conn = _StubConn([pd.Timestamp("2024-02-16")], self.Q)
        out = _resolve_real_premium(conn, "X", pd.Timestamp("2024-02-15"), 100, "put", None)
        assert out is self.Q

    def test_none_when_nearest_expiry_too_far(self):
        conn = _StubConn([pd.Timestamp("2024-03-30")], self.Q)
        out = _resolve_real_premium(
            conn, "X", pd.Timestamp("2024-02-15"), 100, "put", None, expiry_tol_days=7
        )
        assert out is None

    def test_none_when_no_quote(self):
        conn = _StubConn([pd.Timestamp("2024-02-16")], None)
        assert (
            _resolve_real_premium(conn, "X", pd.Timestamp("2024-02-15"), 100, "put", None) is None
        )

    def test_none_on_nonpositive_mid(self):
        conn = _StubConn([pd.Timestamp("2024-02-16")], {"mid": 0.0, "bid": 0.0, "ask": 0.0})
        assert (
            _resolve_real_premium(conn, "X", pd.Timestamp("2024-02-15"), 100, "put", None) is None
        )

    def test_none_when_no_expirations(self):
        conn = _StubConn([], self.Q)
        assert (
            _resolve_real_premium(conn, "X", pd.Timestamp("2024-02-15"), 100, "put", None) is None
        )


# ---------------------------------------------------------------------------
# D1-1 (adversarial review 2026-07-01): quote/spot date + DTE coherence
# ---------------------------------------------------------------------------


class _CoherenceStub:
    """Stub connector that also records every ``as_of`` it is queried with."""

    def __init__(self, exps, quote):
        self._exps = exps
        self._q = quote
        self.seen_as_of: list = []

    def list_option_expirations(self, ticker, as_of=None, **k):
        self.seen_as_of.append(as_of)
        return self._exps

    def get_option_premium(self, ticker, expiry, strike, right, as_of=None, strike_tol=None):
        self.seen_as_of.append(as_of)
        return self._q


class TestQuoteSpotCoherence:
    """A served market quote must come from the SAME EOD session as the spot
    it is paired with (``spot_date``), and its market DTE must match the
    modeled horizon (``dte_target``) — the two refuse-only guards that close
    the frontier-skew defect (live EV inflated 4-18x at ``as_of=None``)."""

    EXP = pd.Timestamp("2024-02-16")

    def _q(self, date="2024-01-10", dte=37, **extra):
        q = {
            "mid": 2.2,
            "bid": 2.0,
            "ask": 2.4,
            "strike": 100.0,
            "right": "put",
            "date": pd.Timestamp(date) if date is not None else None,
            "expiration": self.EXP,
            "dte": dte,
        }
        q.update(extra)
        return {k: v for k, v in q.items() if v is not None}

    def _resolve(self, quote, conn=None, **kw):
        conn = conn or _CoherenceStub([self.EXP], quote)
        return _resolve_real_premium(conn, "X", pd.Timestamp("2024-02-15"), 100, "put", None, **kw)

    def test_rejects_quote_from_different_session(self):
        # the live defect shape: quote snapshot newer than the spot bar
        q = self._q(date="2024-01-12")
        assert self._resolve(q, spot_date=pd.Timestamp("2024-01-10")) is None

    def test_accepts_same_session_quote(self):
        q = self._q(date="2024-01-10")
        assert self._resolve(q, spot_date=pd.Timestamp("2024-01-10")) is q

    def test_rejects_quote_missing_date_when_spot_date_given(self):
        q = self._q(date=None)
        assert self._resolve(q, spot_date=pd.Timestamp("2024-01-10")) is None

    def test_rejects_dte_skew(self):
        # same-session-but-both-stale case: quote's market time (62d) vs the
        # engine's modeled horizon (35d) — date-matching alone cannot catch it
        q = self._q(dte=62)
        assert self._resolve(q, dte_target=35) is None

    def test_accepts_weekend_lag_dte(self):
        # Monday as_of pricing off Friday's bar: |44 - 35| = 9 <= tol 10
        q = self._q(dte=44)
        assert self._resolve(q, dte_target=35) is q

    def test_rejects_beyond_dte_tol(self):
        q = self._q(dte=46)
        assert self._resolve(q, dte_target=35) is None

    def test_dte_derived_from_expiration_when_missing(self):
        # 2024-02-16 - 2024-01-10 = 37 days; |37 - 35| <= 10 -> accepted
        q = self._q(dte=None)
        assert self._resolve(q, spot_date=pd.Timestamp("2024-01-10"), dte_target=35) is q

    def test_legacy_call_without_coherence_kwargs_unchanged(self):
        # compat contract: callers that supply neither kwarg (test stubs whose
        # quotes carry no date/dte) keep the pre-guard behaviour
        q = self._q(date="2024-01-12", dte=62)
        assert self._resolve(q) is q

    def test_as_of_none_queries_market_as_of_spot_bar(self):
        # with no explicit as_of the connector must be queried as of the spot
        # bar — never "latest larder snapshot"
        q = self._q(date="2024-01-10")
        conn = _CoherenceStub([self.EXP], q)
        out = self._resolve(q, conn=conn, spot_date=pd.Timestamp("2024-01-10"))
        assert out is q
        assert conn.seen_as_of == ["2024-01-10", "2024-01-10"]


# ---------------------------------------------------------------------------
# End-to-end: the puts ranker uses real premiums when the rail covers a name
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _OHLCV.exists(), reason="Bloomberg OHLCV not present")
class TestPutsRankerRealPremium:
    AS_OF = "2023-05-25"
    TKR = "AAPL"  # no split since 2020 → raw == adjusted at this as_of

    def _build_rail(self, d: Path):
        spot = float(
            MarketDataConnector().get_ohlcv(self.TKR, end_date=self.AS_OF).iloc[-1]["close"]
        )
        rows = []
        for exp in ["2023-06-20", "2023-06-23", "2023-06-30", "2023-07-07", "2023-07-14"]:
            e = pd.Timestamp(exp)
            dte = (e - pd.Timestamp(self.AS_OF)).days
            for k in range(int(spot * 0.70), int(spot * 1.05)):
                for right in ("put", "call"):
                    rows.append(
                        {
                            "date": pd.Timestamp(self.AS_OF),
                            "expiration": e,
                            "dte": dte,
                            "strike": float(k),
                            "right": right,
                            "bid": 9.9,
                            "ask": 10.1,
                            "mid": 10.0,
                            "close": 10.0,
                            "volume": 50,
                            "open_interest": 500,
                        }
                    )
        _write_produced(d, self.TKR, rows)

    def test_ranker_uses_market_mid(self, tmp_path, monkeypatch):
        d = tmp_path / "op"
        d.mkdir()
        self._build_rail(d)
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(d))
        out = WheelRunner().rank_candidates_by_ev(
            tickers=[self.TKR],
            top_n=25,
            min_ev_dollars=-1e9,
            as_of=self.AS_OF,
            include_diagnostic_fields=True,
        )
        assert "premium_source" in out.columns
        assert (out["premium_source"] == "market_mid").any(), "expected ≥1 real-premium candidate"
        mm = out[out["premium_source"] == "market_mid"]
        # the matched candidates carry the injected market mid (10.0), not a BSM value
        assert all(v == pytest.approx(10.0) for v in mm["premium"])

    def test_ranker_falls_back_when_rail_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(tmp_path / "absent"))
        out = WheelRunner().rank_candidates_by_ev(
            tickers=[self.TKR],
            top_n=25,
            min_ev_dollars=-1e9,
            as_of=self.AS_OF,
            include_diagnostic_fields=True,
        )
        if len(out):
            assert (out["premium_source"] == "synthetic_bsm").all()


# ---------------------------------------------------------------------------
# End-to-end: quote/spot coherence through the real ranker (D1-1 pins)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _OHLCV.exists(), reason="Bloomberg OHLCV not present")
class TestRankerQuoteSpotCoherenceE2E:
    """Pins the two live shapes of the frontier-skew defect against the real
    ranker + real OHLCV: (a) a dated ``as_of`` where the larder's freshest
    quote comes from an EARLIER session than the spot bar (previously served —
    up to 7 days of market move booked as phantom edge); (b) ``as_of=None``
    where the larder frontier runs BEYOND the OHLCV frontier (the exact
    2026-07-01 configuration that inflated live EV 4-18x). Both must degrade
    to the synthetic-BSM premium."""

    TKR = "AAPL"

    def _build_rail(self, d, quote_date: str, expirations: list[str]):
        spot = float(
            MarketDataConnector().get_ohlcv(self.TKR, end_date=quote_date).iloc[-1]["close"]
        )
        qd = pd.Timestamp(quote_date)
        rows = []
        for exp in expirations:
            e = pd.Timestamp(exp)
            for k in range(int(spot * 0.70), int(spot * 1.05)):
                for right in ("put", "call"):
                    rows.append(
                        {
                            "date": qd,
                            "expiration": e,
                            "dte": (e - qd).days,
                            "strike": float(k),
                            "right": right,
                            "bid": 9.9,
                            "ask": 10.1,
                            "mid": 10.0,
                            "close": 10.0,
                            "volume": 50,
                            "open_interest": 500,
                        }
                    )
        _write_produced(d, self.TKR, rows)

    def test_dated_as_of_rejects_prior_session_quote(self, tmp_path, monkeypatch):
        # as_of Thu 2023-05-25 (spot bar 2023-05-25); larder quoted Mon
        # 2023-05-22 — inside the connector's 7-day PIT window, so the OLD
        # code served it against Thursday's spot. Expirations bracket
        # as_of+35d so the expiry snap and DTE tolerance both pass; the
        # ONLY violated guard is same-session date coherence.
        d = tmp_path / "op"
        d.mkdir()
        self._build_rail(d, "2023-05-22", ["2023-06-23", "2023-06-30"])
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(d))
        out = WheelRunner().rank_candidates_by_ev(
            tickers=[self.TKR],
            top_n=25,
            min_ev_dollars=-1e9,
            as_of="2023-05-25",
            include_diagnostic_fields=True,
        )
        assert len(out)
        assert (out["premium_source"] == "synthetic_bsm").all(), (
            "a quote from a different EOD session than the spot bar must not be served"
        )

    def test_as_of_none_skewed_larder_frontier_refused(self, tmp_path, monkeypatch):
        # The 2026-07-01 live defect: larder frontier 13 days beyond the
        # OHLCV frontier. At as_of=None the rail must be queried as of the
        # SPOT bar (the OHLCV frontier), where these future-dated quotes do
        # not exist yet -> synthetic fallback, regardless of wall-clock date.
        conn = MarketDataConnector()
        frontier = pd.Timestamp(conn.get_ohlcv(self.TKR).index.max())
        quote_d = frontier + pd.Timedelta(days=13)
        exps = [(quote_d + pd.Timedelta(days=n)).date().isoformat() for n in (30, 37, 44)]
        d = tmp_path / "op"
        d.mkdir()
        # build against the frontier spot (get_ohlcv end_date must be <= frontier)
        spot = float(conn.get_ohlcv(self.TKR).iloc[-1]["close"])
        qd = quote_d
        rows = []
        for exp in exps:
            e = pd.Timestamp(exp)
            for k in range(int(spot * 0.70), int(spot * 1.05)):
                rows.append(
                    {
                        "date": qd,
                        "expiration": e,
                        "dte": (e - qd).days,
                        "strike": float(k),
                        "right": "put",
                        "bid": 9.9,
                        "ask": 10.1,
                        "mid": 10.0,
                        "close": 10.0,
                        "volume": 50,
                        "open_interest": 500,
                    }
                )
        _write_produced(d, self.TKR, rows)
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(d))
        out = WheelRunner().rank_candidates_by_ev(
            tickers=[self.TKR],
            top_n=25,
            min_ev_dollars=-1e9,
            include_diagnostic_fields=True,
        )
        if len(out):
            assert (out["premium_source"] == "synthetic_bsm").all(), (
                "quotes from beyond the spot frontier must never pair with a stale spot"
            )
