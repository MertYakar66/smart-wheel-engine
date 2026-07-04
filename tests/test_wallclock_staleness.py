"""Tests for the D1-2/D3-2 wall-clock frontier-staleness layers.

Adversarial review 2026-07-01, CRITICAL-2 compound (b): a 27-day-stale
OHLCV frontier was INVISIBLE at runtime. The only staleness gate was
frontier-RELATIVE (each ticker vs the universe frontier), so when the
whole universe is stale every per-ticker gap reads 0 — no drop, no
warning, no output field, anywhere — while the event gate uses the REAL
wall clock (mixed clocks: month-old spots, today's lockout windows).

The fix is layered, mirroring the #464 earnings-overlay treatment:

* **Layer 0 (pre-existing)** — the deterministic ``EXPECTED_FRONTIER``
  preflight pin catches stale TREES.
* **Layer 1** — ``MarketDataConnector.get_data_frontier`` logs a
  once-per-connector warning when the frontier is more than
  ``_OHLCV_FRONTIER_STALE_DAYS`` (7) behind today. WARN-only: the
  return value never changes (a default refuse would blank every
  ``as_of=None`` book — the #462 lesson).
* **Layer 2** — the three rankers attach ``attrs["staleness"]``
  (structured: frontier / wall clock / age / threshold / stale flag)
  so the API and dashboard can SHOW data currency. attrs-only —
  survivor rows byte-identical.
* **Layer 3** — opt-in hard refuse: ``refuse_stale_live=True`` (param)
  or ``SWE_REFUSE_STALE_LIVE=1`` (env). Universe-wide single drop,
  BEFORE the per-ticker loop; drop-only (§2 — a refusal can never
  rescue). Default OFF.

All ranker tests use synthetic duck-typed connectors (the
``test_asof_none_staleness`` idiom); Layer-1 tests use a tmp-dir real
``MarketDataConnector`` so the actual warn path runs. Drop-reason
strings are pinned DISTINCT from the two existing staleness reasons
("beyond latest data", "behind universe data frontier").
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from engine.wheel_runner import WheelRunner, _frontier_staleness_info


@pytest.fixture(autouse=True)
def _hermetic_refusal_env(monkeypatch):
    """An operator box that has armed SWE_REFUSE_STALE_LIVE (exactly what
    the feature invites) must not flip the default-path tests — the arm is
    exercised explicitly via setenv where intended."""
    monkeypatch.delenv("SWE_REFUSE_STALE_LIVE", raising=False)


# ── synthetic OHLCV ──────────────────────────────────────────────────────────

_TODAY = pd.Timestamp(date.today())
_FRESH_END = _TODAY  # frontier age 0 — never stale
_STALE_END = _TODAY - pd.Timedelta(days=27)  # the motivating 27-day case


def _ohlcv(end: pd.Timestamp, n_rows: int = 600, seed: int = 42) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=n_rows)
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_rows)))
    return pd.DataFrame({"close": prices}, index=idx)


def _run_kwargs(**extra) -> dict:
    return {
        "top_n": 10,
        "min_ev_dollars": -1e9,
        "use_dealer_positioning": False,
        "use_news_sentiment": False,
        "use_credit_regime": False,
        "use_skew_dynamics": False,
        **extra,
    }


def _make_runner(end: pd.Timestamp, tickers=("AAA",)) -> WheelRunner:
    """Stub connector whose universe ends at ``end`` (= the frontier)."""
    ohlcv_map = {t: _ohlcv(end, seed=7 + i) for i, t in enumerate(tickers)}
    frontier = end

    class _Conn:
        _OHLCV_FRONTIER_STALE_DAYS = 7

        def get_ohlcv(self, ticker):
            return ohlcv_map.get(ticker, pd.DataFrame())

        def get_fundamentals(self, ticker):
            return {"implied_vol_atm": 0.28, "volatility_30d": 0.25, "dividend_yield": 0.01}

        def get_risk_free_rate(self, as_of=None):
            return 0.05

        def get_next_earnings(self, ticker, as_of=None):
            return None

        def get_universe(self):
            return sorted(ohlcv_map.keys())

        def get_data_frontier(self, dataset="ohlcv"):
            return frontier

    r = WheelRunner()
    r._connector = _Conn()
    return r


# ── Layer 1: connector warn (real MarketDataConnector, tmp dir) ─────────────


def _write_min_bloomberg(tmp_path, end: pd.Timestamp) -> pd.Timestamp:
    """Minimal OHLCV csv so a real connector's get_data_frontier resolves.

    Returns the ACTUAL max date written (``bdate_range(end=weekend)`` snaps
    back to the prior Friday — assert against this, not ``end``)."""
    idx = pd.bdate_range(end=end, periods=40)
    df = pd.DataFrame(
        {
            "date": idx,
            "ticker": "AAA UW",
            "px_open": 100.0,
            "px_high": 101.0,
            "px_low": 99.0,
            "px_last": 100.5,
            "px_volume": 1e6,
        }
    )
    df.to_csv(tmp_path / "sp500_ohlcv.csv", index=False)
    return idx.max()


class TestConnectorFrontierWarn:
    def test_stale_frontier_warns_once_per_connector(self, tmp_path, caplog):
        from engine.data_connector import MarketDataConnector

        actual_max = _write_min_bloomberg(tmp_path, _STALE_END)
        conn = MarketDataConnector(data_dir=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="engine.data_connector"):
            f1 = conn.get_data_frontier()
            f2 = conn.get_data_frontier()
        warns = [r for r in caplog.records if "behind the wall clock" in r.message]
        assert len(warns) == 1, "must warn exactly once per connector"
        # WARN-only: the return value is still the true frontier, both calls.
        assert f1 is not None and f2 is not None
        assert pd.Timestamp(f1).normalize() == actual_max.normalize()

    def test_fresh_frontier_no_warn(self, tmp_path, caplog):
        from engine.data_connector import MarketDataConnector

        _write_min_bloomberg(tmp_path, _FRESH_END)
        conn = MarketDataConnector(data_dir=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="engine.data_connector"):
            f = conn.get_data_frontier()
        assert f is not None
        assert not any("behind the wall clock" in r.message for r in caplog.records)


# ── Layer 2: structured attrs on ranker output ───────────────────────────────


class TestStalenessAttrs:
    def test_attrs_present_and_stale_at_asof_none(self):
        runner = _make_runner(_STALE_END)
        df = runner.rank_candidates_by_ev(tickers=["AAA"], as_of=None, **_run_kwargs())
        s = df.attrs.get("staleness")
        assert s is not None and s["checked"] is True
        assert s["stale"] is True
        assert s["frontier_age_days"] >= 27
        assert s["threshold_days"] == 7
        assert s["data_frontier"] == _STALE_END.date().isoformat()

    def test_attrs_fresh_not_stale(self):
        runner = _make_runner(_FRESH_END)
        df = runner.rank_candidates_by_ev(tickers=["AAA"], as_of=None, **_run_kwargs())
        s = df.attrs.get("staleness")
        assert s is not None and s["checked"] is True and s["stale"] is False

    def test_attrs_sentinel_at_explicit_asof(self):
        """Dated queries must never read the wall clock — explicit audit
        sentinel, and byte-identity for dated backtests."""
        runner = _make_runner(_STALE_END)
        df = runner.rank_candidates_by_ev(
            tickers=["AAA"],
            as_of=_STALE_END.date().isoformat(),
            **_run_kwargs(),
        )
        assert df.attrs.get("staleness") == {"checked": False}

    def test_attrs_only_survivors_untouched(self):
        """Rows are identical with and without the staleness layers armed
        OFF — attrs ride-along only (CLAUDE.md §2 drops precedent)."""
        runner = _make_runner(_STALE_END)
        df1 = runner.rank_candidates_by_ev(tickers=["AAA"], as_of=None, **_run_kwargs())
        df2 = runner.rank_candidates_by_ev(
            tickers=["AAA"], as_of=None, refuse_stale_live=False, **_run_kwargs()
        )
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))

    def test_cc_and_strangle_attach_staleness(self):
        runner = _make_runner(_STALE_END)
        cc = runner.rank_covered_calls_by_ev("AAA", as_of=None, min_ev_dollars=-1e9)
        st = runner.rank_strangles_by_ev("AAA", as_of=None, min_ev_dollars=-1e9)
        for frame in (cc, st):
            s = frame.attrs.get("staleness")
            assert s is not None and s["checked"] is True and s["stale"] is True


# ── Layer 3: opt-in hard refuse ──────────────────────────────────────────────


class TestRefuseStaleLive:
    def test_param_refuses_universe_wide(self):
        runner = _make_runner(_STALE_END, tickers=("AAA", "BBB"))
        df = runner.rank_candidates_by_ev(
            tickers=["AAA", "BBB"], as_of=None, refuse_stale_live=True, **_run_kwargs()
        )
        assert df.empty
        drops = df.attrs["drops"]
        assert len(drops) == 1 and drops[0]["ticker"] == "*"  # ONE drop, not per-ticker spam
        assert drops[0]["gate"] == "data"
        reason = drops[0]["reason"]
        assert "behind the wall clock" in reason and "refuse_stale_live" in reason
        # Distinct from the two pre-existing staleness reasons (pinned):
        assert "behind universe data frontier" not in reason
        assert "beyond latest data" not in reason
        assert df.attrs["staleness"]["stale"] is True

    def test_env_var_arms_refusal(self, monkeypatch):
        monkeypatch.setenv("SWE_REFUSE_STALE_LIVE", "1")
        runner = _make_runner(_STALE_END)
        df = runner.rank_candidates_by_ev(tickers=["AAA"], as_of=None, **_run_kwargs())
        assert df.empty and df.attrs["drops"][0]["ticker"] == "*"

    def test_param_false_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SWE_REFUSE_STALE_LIVE", "1")
        runner = _make_runner(_STALE_END)
        df = runner.rank_candidates_by_ev(
            tickers=["AAA"], as_of=None, refuse_stale_live=False, **_run_kwargs()
        )
        assert not df.empty  # explicit param wins over the env arm

    def test_default_is_fail_open(self):
        """Default (no param, no env): stale universe still ranks — the
        engine warns and surfaces attrs, but never silently blanks the
        book (#462 lesson)."""
        runner = _make_runner(_STALE_END)
        df = runner.rank_candidates_by_ev(tickers=["AAA"], as_of=None, **_run_kwargs())
        assert not df.empty

    def test_fresh_frontier_not_refused_even_when_armed(self):
        runner = _make_runner(_FRESH_END)
        df = runner.rank_candidates_by_ev(
            tickers=["AAA"], as_of=None, refuse_stale_live=True, **_run_kwargs()
        )
        assert not df.empty

    def test_explicit_asof_never_refused_even_when_armed(self):
        """The refusal is a LIVE gate: dated backtests are untouched even
        with the arm on (wall-clock reads gate on as_of is None)."""
        runner = _make_runner(_STALE_END)
        df = runner.rank_candidates_by_ev(
            tickers=["AAA"],
            as_of=_STALE_END.date().isoformat(),
            refuse_stale_live=True,
            **_run_kwargs(),
        )
        assert not df.empty

    def test_cc_and_strangle_share_the_refusal(self):
        runner = _make_runner(_STALE_END)
        cc = runner.rank_covered_calls_by_ev(
            "AAA", as_of=None, refuse_stale_live=True, min_ev_dollars=-1e9
        )
        st = runner.rank_strangles_by_ev(
            "AAA", as_of=None, refuse_stale_live=True, min_ev_dollars=-1e9
        )
        for frame in (cc, st):
            assert frame.empty
            assert any("refuse_stale_live" in d["reason"] for d in frame.attrs["drops"])


# ── helper unit ──────────────────────────────────────────────────────────────


class TestFrontierStalenessInfo:
    def test_none_ref_unchecked(self):
        assert _frontier_staleness_info(None, None, object()) == {"checked": False}

    def test_dated_unchecked(self):
        assert _frontier_staleness_info("2026-06-04", pd.Timestamp("2026-06-04"), object()) == {
            "checked": False
        }

    def test_threshold_read_from_connector(self):
        class _C:
            _OHLCV_FRONTIER_STALE_DAYS = 3

        ref = pd.Timestamp(date.today() - timedelta(days=5))
        info = _frontier_staleness_info(None, ref, _C())
        assert info["threshold_days"] == 3 and info["stale"] is True

    def test_methodless_stub_defaults_to_seven(self):
        ref = pd.Timestamp(date.today() - timedelta(days=5))
        info = _frontier_staleness_info(None, ref, object())
        assert info["threshold_days"] == 7 and info["stale"] is False
