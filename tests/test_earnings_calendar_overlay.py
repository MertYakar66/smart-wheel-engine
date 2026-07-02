"""Tests for the D3-1 earnings-calendar overlay + D6-1 event-gate de-silencing.

Context (adversarial review 2026-07-01, findings D3-1 + D6-1):

* **D3-1 (critical).** ``get_next_earnings`` read only
  ``sp500_earnings.csv``, whose *forward* coverage is ~39/511 names —
  so at live ``as_of=None`` the earnings hard-lockout silently no-op'ed
  for ~92 % of the universe going into the July/August earnings season,
  while richer forward calendars sat unwired on disk. Fix: the connector
  now overlays the broad-pull per-name snapshot's ``next_earnings_dt``
  (100 % coverage, stamped with its own ``asof`` knowledge date),
  point-in-time gated so the overlay participates only when
  ``as_of >= asof`` — every dated query before the snapshot date is
  byte-identical to the pre-overlay engine.

* **D6-1 (high).** Each ranker wrapped its whole event-exclusion block
  (forward lookup + back-buffer + corp-action registration + soft skip)
  in one bare ``except Exception``, so a single raising stage silently
  disabled the entire hard lockout for that ticker — indistinguishable
  from "no earnings scheduled". Fix: per-stage guards
  (``_fetch_next_earnings`` / ``_fetch_recent_earnings`` /
  ``_earnings_event_date``) that LOG on failure, keep the other stages
  alive, and leave the soft-skip control flow outside any try.

Pinned here:

  1. Overlay unit semantics on tmp fixtures — forward serve, PIT gate,
     earlier-date-wins merge, tie prefers the richer base row, back-buffer
     (just-reported) serve, hermeticity (no repo-calendar bleed into
     tmp-dir connectors), malformed/unstamped rows refused.
  2. Real-data pins at dated ``as_of`` (deterministic, no wall clock).
  3. End-to-end: overlay -> ranker -> EventGate -> EVEngine short-circuit
     -> ``df.attrs["drops"]`` event reason, plus the PIT-gate control.
  4. D6-1: a raising forward lookup is LOGGED and no longer kills the
     back-buffer stage (the lockout can still fire); malformed dates are
     logged, not swallowed; method-less connectors stay warning-free.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.data_connector import MarketDataConnector
from engine.wheel_runner import WheelRunner

_OFFLINE = {
    "use_dealer_positioning": False,
    "use_news_sentiment": False,
    "use_credit_regime": False,
    "use_skew_dynamics": False,
}

_TICKERS = ["AAA", "BBB"]

_REPO_DATA = Path("data/bloomberg")
HAS_BLOOMBERG_DATA = (_REPO_DATA / "sp500_earnings.csv").exists() and (
    _REPO_DATA / "broad_pull" / "per_name" / "sp500_snapshot_bdp.csv"
).exists()


# ----------------------------------------------------------------------
# fixtures — a minimal Bloomberg-shaped data dir + broad-pull snapshot
# ----------------------------------------------------------------------
def _write_base_earnings(data_dir: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(data_dir / "sp500_earnings.csv", index=False)


def _write_snapshot_bdp(
    data_dir: Path,
    *,
    asof: str | None,
    next_by_ticker: dict[str, str | None],
) -> None:
    """Write a broad-pull per-name snapshot with Bloomberg-style tickers.

    Tickers are written as ``"<root> UW"`` to prove the overlay bridges the
    snapshot's terminal format to the engine's plain roots via
    ``ticker_normalized``.
    """
    per_name = data_dir / "broad_pull" / "per_name"
    per_name.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "asof": [asof] * len(next_by_ticker),
            "ticker": [f"{t} UW" for t in next_by_ticker],
            "next_earnings_dt": list(next_by_ticker.values()),
        }
    ).to_csv(per_name / "sp500_snapshot_bdp.csv", index=False)


@pytest.fixture
def overlay_dir(tmp_path: Path) -> Path:
    """Base file: AAA has ONLY a past row (the D3-1 shape — 92 % of the
    universe looks like this); snapshot (asof 2026-06-01) knows AAA's next
    date. BBB is absent from the snapshot."""
    _write_base_earnings(
        tmp_path,
        [
            {"ticker": "AAA", "announcement_date": "2026-01-05", "year/period": "2025 Q4"},
            {"ticker": "BBB", "announcement_date": "2026-01-08", "year/period": "2025 Q4"},
        ],
    )
    _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "2026-07-05"})
    return tmp_path


# ----------------------------------------------------------------------
# 1. Overlay unit semantics (tmp fixtures)
# ----------------------------------------------------------------------
class TestSnapshotOverlayUnit:
    def test_overlay_serves_forward_date_when_base_has_none(self, overlay_dir):
        conn = MarketDataConnector(data_dir=str(overlay_dir))
        nxt = conn.get_next_earnings("AAA", as_of="2026-06-20")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-07-05")
        assert nxt["source"] == "snapshot_bdp"

    def test_pit_gate_refuses_before_snapshot_asof(self, overlay_dir):
        """A query dated before the snapshot's knowledge date must not see
        it — the no-lookahead half of the fix. Byte-identical to the
        pre-overlay engine for every dated backtest at as_of < asof."""
        conn = MarketDataConnector(data_dir=str(overlay_dir))
        assert conn.get_next_earnings("AAA", as_of="2026-05-20") is None

    def test_ticker_absent_from_snapshot_stays_base_only(self, overlay_dir):
        conn = MarketDataConnector(data_dir=str(overlay_dir))
        assert conn.get_next_earnings("BBB", as_of="2026-06-20") is None

    def test_base_earlier_date_wins(self, tmp_path):
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-07-01", "year/period": "2026 Q2"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "2026-07-20"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        nxt = conn.get_next_earnings("AAA", as_of="2026-06-20")
        assert nxt["announcement_date"] == pd.Timestamp("2026-07-01")
        assert nxt["source"] == "earnings_csv"

    def test_overlay_earlier_date_wins(self, tmp_path):
        """The base file's thin forward rows can sit a full quarter out
        (banks pre-schedule to 2028); the snapshot's nearer date must win —
        otherwise the gate would miss the imminent event."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-10-10", "year/period": "2026 Q3"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "2026-07-20"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        nxt = conn.get_next_earnings("AAA", as_of="2026-06-20")
        assert nxt["announcement_date"] == pd.Timestamp("2026-07-20")
        assert nxt["source"] == "snapshot_bdp"

    def test_tie_prefers_richer_base_row(self, tmp_path):
        _write_base_earnings(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "announcement_date": "2026-07-20",
                    "year/period": "2026 Q2",
                    "estimate_eps": 1.25,
                }
            ],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "2026-07-20"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        nxt = conn.get_next_earnings("AAA", as_of="2026-06-20")
        assert nxt["source"] == "earnings_csv"
        assert nxt["estimate_eps"] == 1.25

    def test_recent_overlay_serves_just_reported(self, overlay_dir):
        """A snapshot date that has PASSED by as_of is the just-reported /
        IV-crush case — served by get_recent_earnings (back-buffer), not
        get_next_earnings. Complementary cutoff preserved."""
        conn = MarketDataConnector(data_dir=str(overlay_dir))
        recent = conn.get_recent_earnings("AAA", as_of="2026-07-08", lookback_days=5)
        assert recent is not None
        assert recent["announcement_date"] == pd.Timestamp("2026-07-05")
        assert recent["source"] == "snapshot_bdp"
        assert conn.get_next_earnings("AAA", as_of="2026-07-08") is None

    def test_recent_overlay_respects_lookback_window(self, overlay_dir):
        conn = MarketDataConnector(data_dir=str(overlay_dir))
        assert conn.get_recent_earnings("AAA", as_of="2026-07-25", lookback_days=5) is None

    def test_recent_overlay_pit_gate(self, tmp_path):
        """PIT gate applies to the back-buffer too: a query dated before
        the snapshot's asof must not see its date even as a past event."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2020-01-05", "year/period": "2019 Q4"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "2026-05-28"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        # as_of 2026-05-30: the snapshot (known 06-01) is in this query's
        # future -> refused, even though 05-28 is inside the lookback.
        assert conn.get_recent_earnings("AAA", as_of="2026-05-30", lookback_days=5) is None
        # as_of 2026-06-02 (>= asof): now knowable -> served.
        recent = conn.get_recent_earnings("AAA", as_of="2026-06-02", lookback_days=7)
        assert recent is not None
        assert recent["source"] == "snapshot_bdp"

    def test_recent_prefers_most_recent_across_sources(self, tmp_path):
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-06-28", "year/period": "2026 Q2"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "2026-07-01"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        recent = conn.get_recent_earnings("AAA", as_of="2026-07-02", lookback_days=7)
        assert recent["announcement_date"] == pd.Timestamp("2026-07-01")
        assert recent["source"] == "snapshot_bdp"

    def test_no_broad_pull_dir_no_bleed(self, tmp_path):
        """Hermeticity: a connector on a data dir WITHOUT broad_pull/ must
        not bleed the repo's real calendar in (several existing tests pin
        ``get_next_earnings(...) is None`` on tmp dirs)."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-01-05", "year/period": "2025 Q4"}],
        )
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_next_earnings("AAA", as_of="2026-06-20") is None
        assert conn.get_next_earnings("AAPL", as_of="2026-06-20") is None

    def test_malformed_next_earnings_dt_refused(self, tmp_path):
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-01-05", "year/period": "2025 Q4"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"AAA": "not-a-date"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_next_earnings("AAA", as_of="2026-06-20") is None

    def test_unstamped_snapshot_refused(self, tmp_path):
        """No asof stamp -> no PIT gate possible -> the overlay must refuse
        (an unstamped calendar cannot prove it was knowable at as_of)."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-01-05", "year/period": "2025 Q4"}],
        )
        _write_snapshot_bdp(tmp_path, asof=None, next_by_ticker={"AAA": "2026-07-05"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        assert conn.get_next_earnings("AAA", as_of="2026-06-20") is None

    def test_share_class_slash_ticker_bridged(self, tmp_path):
        """2026-07-02 refuter panel (ops): the connector's normalize_ticker
        keeps the slash ('BRK/B') while the loader's ticker_normalized
        column uses dots ('BRK.B') — unbridged, the two slash names in the
        snapshot silently stayed in the D3-1 no-op state. Both sides must
        compare in dot-form."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "BRK/B", "announcement_date": "2026-01-05", "year/period": "2025 Q4"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-06-01", next_by_ticker={"BRK/B": "2026-08-03"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        nxt = conn.get_next_earnings("BRK/B", as_of="2026-06-20")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-08-03")
        assert nxt["source"] == "snapshot_bdp"

    def test_multi_asof_panel_serves_newest_eligible(self, tmp_path):
        """2026-07-02 refuter panel (§2): if a future broad-pull refresh
        APPENDS a new asof instead of replacing rows, the overlay must serve
        the NEWEST snapshot knowable at ref — iloc[0] on the asof-ascending
        panel would silently serve the decayed calendar while the preflight
        pin (max asof) stayed green."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-01-05", "year/period": "2025 Q4"}],
        )
        per_name = tmp_path / "broad_pull" / "per_name"
        per_name.mkdir(parents=True)
        pd.DataFrame(
            {
                "asof": ["2026-03-01", "2026-06-01"],
                "ticker": ["AAA UW", "AAA UW"],
                "next_earnings_dt": ["2026-04-20", "2026-07-05"],
            }
        ).to_csv(per_name / "sp500_snapshot_bdp.csv", index=False)
        conn = MarketDataConnector(data_dir=str(tmp_path))
        # Both snapshots knowable -> the newer one's date.
        nxt = conn.get_next_earnings("AAA", as_of="2026-06-20")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-07-05")
        # Only the older snapshot knowable -> its date (PIT-correct).
        nxt_old = conn.get_next_earnings("AAA", as_of="2026-03-15")
        assert nxt_old is not None
        assert nxt_old["announcement_date"] == pd.Timestamp("2026-04-20")

    def test_loader_failure_is_logged_not_silent(self, overlay_dir, monkeypatch, caplog):
        """2026-07-02 refuter panel (§2): a broad-pull import/loader failure
        silently un-armed the restored lockout (the D6-1 class one seam
        higher). It must degrade to the base calendar WITH a warning."""
        import data.broad_pull_loaders as bpl

        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("simulated loader failure")

        monkeypatch.setattr(bpl, "BroadPullLoader", _Boom)
        conn = MarketDataConnector(data_dir=str(overlay_dir))
        with caplog.at_level(logging.WARNING, logger="engine.data_connector"):
            nxt = conn.get_next_earnings("AAA", as_of="2026-06-20")
        assert nxt is None  # degraded to the (empty-forward) base calendar
        assert any("earnings-calendar overlay unavailable" in r.message for r in caplog.records)

    def test_stale_overlay_warns_once_per_connector(self, tmp_path, caplog):
        """Runtime staleness alarm: a snapshot >45d older than the query
        date warns ONCE per connector (no per-ticker spam) — the threshold
        is 45d because per-name forward-lockout decay becomes material at
        snapshot age ~51d (quarterly cadence minus the 40d gate lookahead),
        so a 90d alarm would sit silent through ~40 days of un-armed names."""
        _write_base_earnings(
            tmp_path,
            [{"ticker": "AAA", "announcement_date": "2026-01-05", "year/period": "2025 Q4"}],
        )
        _write_snapshot_bdp(tmp_path, asof="2026-05-01", next_by_ticker={"AAA": "2026-08-01"})
        conn = MarketDataConnector(data_dir=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="engine.data_connector"):
            conn.get_next_earnings("AAA", as_of="2026-06-20")  # age 50d > 45d
            conn.get_next_earnings("AAA", as_of="2026-06-21")
        stale = [r for r in caplog.records if "days older than the query date" in r.message]
        assert len(stale) == 1
        # Fresh snapshot: no warning.
        caplog.clear()
        conn2 = MarketDataConnector(data_dir=str(tmp_path))
        with caplog.at_level(logging.WARNING, logger="engine.data_connector"):
            conn2.get_next_earnings("AAA", as_of="2026-05-20")  # age 19d
        assert not [r for r in caplog.records if "days older" in r.message]


# ----------------------------------------------------------------------
# 2. Real-data pins (dated as_of — deterministic, no wall clock)
# ----------------------------------------------------------------------
@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="bundled Bloomberg data absent")
class TestSnapshotOverlayRealData:
    """Pins on the committed CSVs. The repo snapshot's asof is 2026-06-18;
    dated queries either side of it pin the exact value-add of the fix."""

    @pytest.fixture(scope="class")
    def conn(self):
        return MarketDataConnector()

    def test_aapl_forward_date_served_after_asof(self, conn):
        # AAPL is in the 92 % (no forward row in sp500_earnings.csv).
        nxt = conn.get_next_earnings("AAPL", as_of="2026-06-20")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-07-31")
        assert nxt["source"] == "snapshot_bdp"

    def test_aapl_none_before_asof(self, conn):
        """Byte-identity with the pre-overlay engine for dated queries
        before the snapshot date — including the 2026-06-04 data frontier
        every dated regression pin runs at."""
        assert conn.get_next_earnings("AAPL", as_of="2026-06-10") is None
        assert conn.get_next_earnings("AAPL", as_of="2026-06-04") is None

    def test_jpm_base_row_still_wins(self, conn):
        # JPM is in the ~8 % with real forward rows; the base must keep
        # serving it (richer row, same-or-earlier date).
        nxt = conn.get_next_earnings("JPM", as_of="2026-06-20")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-07-14")
        assert nxt["source"] == "earnings_csv"

    def test_tsla_just_reported_back_buffer(self, conn):
        recent = conn.get_recent_earnings("TSLA", as_of="2026-07-03", lookback_days=5)
        assert recent is not None
        assert recent["announcement_date"] == pd.Timestamp("2026-07-02")
        assert recent["source"] == "snapshot_bdp"

    def test_brk_b_share_class_served(self, conn):
        """The slash mega-cap the 2026-07-02 refuter panel caught falling
        through the normalize mismatch — pinned on the real snapshot."""
        nxt = conn.get_next_earnings("BRK/B", as_of="2026-06-20")
        assert nxt is not None
        assert nxt["announcement_date"] == pd.Timestamp("2026-08-03")
        assert nxt["source"] == "snapshot_bdp"


# ----------------------------------------------------------------------
# 3. End-to-end: overlay -> ranker -> EventGate -> drop
# ----------------------------------------------------------------------
class _BareMarketStub:
    """Synthetic market data (same shape as test_event_gate_back_buffer's
    stub) with NO earnings API at all — the hasattr-gated legacy shape."""

    def __init__(self, tickers, *, default_days: int = 3000) -> None:
        self._tickers = list(tickers)
        self._ohlcv: dict[str, pd.DataFrame] = {}
        for i, t in enumerate(tickers):
            idx = pd.date_range("2016-01-01", periods=default_days, freq="B")
            rng = np.random.default_rng(100 + i)
            base = 80.0 * (1.0 + 0.45 * i)
            close = base * np.exp(np.cumsum(rng.normal(0.0003, 0.011, default_days)))
            self._ohlcv[t] = pd.DataFrame({"close": close}, index=idx)

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        return self._ohlcv[ticker]

    def get_fundamentals(self, ticker: str) -> dict:
        return {"implied_vol_atm": 28.0, "volatility_30d": 28.0, "dividend_yield": 0.0}

    def get_risk_free_rate(self, as_of=None) -> float:
        return 0.05

    def get_universe(self) -> list[str]:
        return list(self._tickers)


class _MarketStub(_BareMarketStub):
    """Earnings methods DELEGATE to a real MarketDataConnector on a tmp
    fixture, so the e2e exercises the overlay's actual read path."""

    def __init__(self, tickers, earnings_conn, *, default_days: int = 3000) -> None:
        super().__init__(tickers, default_days=default_days)
        self._earnings_conn = earnings_conn

    def get_next_earnings(self, ticker: str, as_of=None) -> dict | None:
        return self._earnings_conn.get_next_earnings(ticker, as_of)

    def get_recent_earnings(self, ticker: str, as_of=None, lookback_days=7) -> dict | None:
        return self._earnings_conn.get_recent_earnings(ticker, as_of, lookback_days=lookback_days)


def _runner(conn) -> WheelRunner:
    r = WheelRunner()
    r._connector = conn
    return r


def _rank(runner, **extra):
    kw = dict(tickers=_TICKERS, top_n=20, min_ev_dollars=-1e9, **_OFFLINE)
    kw.update(extra)
    return runner.rank_candidates_by_ev(**kw)


class TestRankerLockoutE2E:
    def test_overlay_date_locks_candidate_through_gate(self, overlay_dir):
        """AAA's next earnings is known ONLY to the snapshot overlay; the
        ranker must lock it out (D3-1's exact fix, end to end). BBB (no
        scheduled earnings) must survive."""
        stub = _MarketStub(_TICKERS, MarketDataConnector(data_dir=str(overlay_dir)))
        df = _rank(_runner(stub), as_of="2026-06-20", dte_target=30)
        drops = df.attrs["drops"]
        aaa = [d for d in drops if d["ticker"] == "AAA" and d["gate"] == "event"]
        assert aaa, f"AAA (earnings 2026-07-05, inside the 30d window) must lock: {drops}"
        assert "earnings@2026-07-05" in aaa[0]["reason"]
        assert "BBB" in set(df["ticker"]) if len(df) else True

    def test_pit_gate_control_no_lock_before_asof(self, overlay_dir):
        """Same fixture, as_of BEFORE the snapshot's knowledge date: the
        overlay must not participate, so AAA must NOT event-lock — pins
        that the e2e lock above comes from the PIT-gated overlay, not some
        other path."""
        stub = _MarketStub(_TICKERS, MarketDataConnector(data_dir=str(overlay_dir)))
        df = _rank(_runner(stub), as_of="2026-05-20", dte_target=30)
        aaa = [d for d in df.attrs["drops"] if d["ticker"] == "AAA" and d["gate"] == "event"]
        assert not aaa, f"overlay leaked through the PIT gate: {aaa}"


# ----------------------------------------------------------------------
# 4. D6-1 — registration failures are logged, isolated, and non-fatal
# ----------------------------------------------------------------------
class _RaisingForwardConn(_MarketStub):
    """get_next_earnings raises; get_recent_earnings still works."""

    def __init__(self, tickers, *, recent: dict[str, str] | None = None) -> None:
        super().__init__(tickers, earnings_conn=None)
        self._recent = dict(recent or {})

    def get_next_earnings(self, ticker: str, as_of=None) -> dict | None:
        raise RuntimeError("simulated calendar read failure")

    def get_recent_earnings(self, ticker: str, as_of=None, lookback_days=7) -> dict | None:
        d = self._recent.get(ticker)
        if d is None:
            return None
        ref = pd.Timestamp(as_of) if as_of else pd.Timestamp.now().normalize()
        ev = pd.Timestamp(d)
        if ref - pd.Timedelta(days=int(lookback_days)) <= ev <= ref:
            return {"announcement_date": ev}
        return None


class TestEventGateRegistrationErrors:
    def test_raising_forward_lookup_is_logged_and_fail_open(self, caplog):
        """Fail-open is deliberate (an error is not evidence of an event),
        but it must be LOUD: pre-fix this produced no log, no drop record,
        and days_to_earnings=None — indistinguishable from 'no earnings'."""
        conn = _RaisingForwardConn(_TICKERS)
        with caplog.at_level(logging.WARNING, logger="engine.wheel_runner"):
            df = _rank(_runner(conn), as_of="2026-03-15")
        assert any("get_next_earnings failed" in r.message for r in caplog.records)
        assert set(df["ticker"]) == set(_TICKERS)  # still ranks — fail-open

    def test_raising_forward_does_not_kill_back_buffer(self, caplog):
        """THE D6-1 behavioral fix: pre-fix, one blanket try meant a raising
        forward lookup also killed the back-buffer registration — AAA with
        earnings 2 days ago would rank. Now the back-buffer stage survives
        and locks it."""
        conn = _RaisingForwardConn(_TICKERS, recent={"AAA": "2026-03-13"})
        with caplog.at_level(logging.WARNING, logger="engine.wheel_runner"):
            df = _rank(_runner(conn), as_of="2026-03-15", earnings_buffer_days=5)
        aaa = [d for d in df.attrs["drops"] if d["ticker"] == "AAA" and d["gate"] == "event"]
        assert aaa, (
            f"back-buffer lockout must survive a raising forward lookup; drops={df.attrs['drops']}"
        )
        assert "earnings@2026-03-13" in aaa[0]["reason"]

    def test_connector_without_earnings_methods_is_quiet(self, caplog):
        """A connector legitimately lacking the earnings API (hasattr gate)
        is a configuration, not an error — no warning spam."""
        conn = _BareMarketStub(_TICKERS)
        with caplog.at_level(logging.WARNING, logger="engine.wheel_runner"):
            df = _rank(_runner(conn), as_of="2026-03-15")
        assert not any("failed" in r.message for r in caplog.records)
        assert set(df["ticker"]) == set(_TICKERS)

    def test_malformed_announcement_date_is_logged_not_swallowed(self, caplog):
        class _GarbageDate(_MarketStub):
            def __init__(self, tickers):
                super().__init__(tickers, earnings_conn=None)

            def get_next_earnings(self, ticker, as_of=None):
                return {"announcement_date": "garbage"} if ticker == "AAA" else None

            def get_recent_earnings(self, ticker, as_of=None, lookback_days=7):
                return None

        conn = _GarbageDate(_TICKERS)
        with caplog.at_level(logging.WARNING, logger="engine.wheel_runner"):
            df = _rank(_runner(conn), as_of="2026-03-15")
        assert any("unparseable earnings announcement_date" in r.message for r in caplog.records)
        assert set(df["ticker"]) == set(_TICKERS)

    def test_truthy_non_dict_return_is_logged_not_fatal(self, caplog):
        """2026-07-02 refuter panel (§2 note): a connector returning a
        truthy non-dict (a DataFrame -> ValueError on truthiness, a list ->
        AttributeError on .get) must degrade with a log, not crash the
        ranking run — the old blanket except handled these silently."""

        class _WeirdReturn(_MarketStub):
            def __init__(self, tickers):
                super().__init__(tickers, earnings_conn=None)

            def get_next_earnings(self, ticker, as_of=None):
                if ticker == "AAA":
                    return pd.DataFrame({"announcement_date": ["2026-03-18"]})
                return ["2026-03-18"]

            def get_recent_earnings(self, ticker, as_of=None, lookback_days=7):
                return None

        conn = _WeirdReturn(_TICKERS)
        with caplog.at_level(logging.WARNING, logger="engine.wheel_runner"):
            df = _rank(_runner(conn), as_of="2026-03-15")
        assert any("unparseable earnings announcement_date" in r.message for r in caplog.records)
        assert set(df["ticker"]) == set(_TICKERS)


# ----------------------------------------------------------------------
# 5. Live-deployment preflight (opt-in; wall-clock dependent BY DESIGN)
# ----------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("SWE_LIVE_PREFLIGHT", "").strip() != "1",
    reason="wall-clock live preflight — opt-in via SWE_LIVE_PREFLIGHT=1 before live use",
)
@pytest.mark.skipif(not HAS_BLOOMBERG_DATA, reason="bundled Bloomberg data absent")
def test_live_earnings_calendar_is_fresh_enough():
    """The overlay FAILS OPEN as it ages (dates fall behind the wall clock
    and stop registering), silently reverting the lockout toward the ~8 %
    baseline — the D3-1 failure class. Run this (SWE_LIVE_PREFLIGHT=1)
    before trusting a live as_of=None ranking; CI stays deterministic via
    the pinned-asof guard in tests/test_preflight_environment.py."""
    snap = pd.read_csv(
        _REPO_DATA / "broad_pull" / "per_name" / "sp500_snapshot_bdp.csv", usecols=["asof"]
    )
    asof = pd.to_datetime(snap["asof"]).max()
    age = (pd.Timestamp.now().normalize() - asof).days
    assert age <= MarketDataConnector._SNAPSHOT_EARNINGS_STALE_DAYS, (
        f"broad-pull earnings snapshot is {age} days old (asof {asof.date()}) — the live "
        "earnings lockout has decayed. Re-pull broad_pull (per_name/sp500_snapshot_bdp.csv) "
        "and bump EXPECTED_EARNINGS_CALENDAR_ASOF in tests/test_preflight_environment.py."
    )
