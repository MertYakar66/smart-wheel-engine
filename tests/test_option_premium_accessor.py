"""Tests for the real EOD option-premium rail.

Covers the pure distillation transform + the ``produce_ticker`` writer
(``scripts/produce_option_premiums.py``) and the connector accessors
(``get_option_premium`` / ``get_option_premium_chain`` /
``list_option_expirations``). The rail is the data half of the real-premium
producer that lets skew/VRP become EV-relevant (docs/PHASE2_SKEW_EXECUTION_SPEC.md);
the EV-moving ranker wiring is a separate change and is NOT exercised here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.produce_option_premiums as prod
from engine.data_connector import OPTION_PREMIUM_COLUMNS, MarketDataConnector

REPO = Path(__file__).resolve().parent.parent
_LARDER = REPO / "data_processed" / "theta" / "option_history"


# ---------------------------------------------------------------------------
# Pure transform: distill_expiration_frame
# ---------------------------------------------------------------------------


def _raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw Theta-shaped expiration frame from compact row dicts."""
    return pd.DataFrame(rows)


class TestDistillExpirationFrame:
    def test_mid_right_and_dte(self):
        raw = _raw(
            [
                # created (snapshot) 30 DTE before expiry; valid two-sided market
                {
                    "created": "2024-01-15T16:00:00",
                    "expiration": "20240216",
                    "strike": 100.0,
                    "right": "PUT",
                    "bid": 2.0,
                    "ask": 2.4,
                    "close": 2.2,
                    "volume": 10,
                    "open_interest": 500,
                },
                {
                    "created": "2024-01-15T16:00:00",
                    "expiration": "20240216",
                    "strike": 105.0,
                    "right": "CALL",
                    "bid": 1.0,
                    "ask": 1.2,
                    "close": 1.1,
                    "volume": 5,
                    "open_interest": 300,
                },
            ]
        )
        out = prod.distill_expiration_frame(raw)
        assert list(out.columns) == list(OPTION_PREMIUM_COLUMNS)
        assert len(out) == 2
        put = out[out["right"] == "put"].iloc[0]
        assert put["mid"] == pytest.approx(2.2)  # (2.0 + 2.4) / 2
        assert put["dte"] == 32  # 2024-01-15 -> 2024-02-16
        assert set(out["right"]) == {"put", "call"}  # normalized from PUT/CALL

    def test_dte_belt_filters_far_expiry(self):
        raw = _raw(
            [
                {
                    "created": "2024-01-15",
                    "expiration": "20240216",
                    "strike": 100.0,
                    "right": "PUT",
                    "bid": 2.0,
                    "ask": 2.4,
                },
                {
                    "created": "2024-01-15",
                    "expiration": "20240620",  # ~157 DTE
                    "strike": 100.0,
                    "right": "PUT",
                    "bid": 5.0,
                    "ask": 5.4,
                },
            ]
        )
        out = prod.distill_expiration_frame(raw, dte_min=0, dte_max=75)
        assert len(out) == 1
        assert out.iloc[0]["dte"] == 32

    def test_crossed_and_empty_markets_dropped(self):
        raw = _raw(
            [
                # crossed (ask < bid) -> dropped
                {
                    "created": "2024-01-15",
                    "expiration": "20240216",
                    "strike": 100.0,
                    "right": "PUT",
                    "bid": 3.0,
                    "ask": 2.0,
                },
                # both sides zero -> mid 0 -> dropped
                {
                    "created": "2024-01-15",
                    "expiration": "20240216",
                    "strike": 90.0,
                    "right": "PUT",
                    "bid": 0.0,
                    "ask": 0.0,
                },
                # legitimate 0-bid wide market -> kept (mid = ask/2 > 0)
                {
                    "created": "2024-01-15",
                    "expiration": "20240216",
                    "strike": 80.0,
                    "right": "PUT",
                    "bid": 0.0,
                    "ask": 0.10,
                },
            ]
        )
        out = prod.distill_expiration_frame(raw)
        assert len(out) == 1
        assert out.iloc[0]["strike"] == 80.0
        assert out.iloc[0]["mid"] == pytest.approx(0.05)

    def test_dedup_keeps_last_snapshot(self):
        raw = _raw(
            [
                {
                    "created": "2024-01-15T10:00:00",
                    "expiration": "20240216",
                    "strike": 100.0,
                    "right": "PUT",
                    "bid": 2.0,
                    "ask": 2.4,
                },
                {
                    "created": "2024-01-15T16:00:00",
                    "expiration": "20240216",
                    "strike": 100.0,
                    "right": "PUT",
                    "bid": 2.1,
                    "ask": 2.5,
                },
            ]
        )
        out = prod.distill_expiration_frame(raw)
        # both normalize to date 2024-01-15 -> one row, last (mid 2.3) wins
        assert len(out) == 1
        assert out.iloc[0]["mid"] == pytest.approx(2.3)

    def test_empty_and_missing_columns(self):
        assert prod.distill_expiration_frame(pd.DataFrame()).empty
        assert prod.distill_expiration_frame(
            pd.DataFrame({"created": ["2024-01-15"], "strike": [100.0]})
        ).empty
        out = prod.distill_expiration_frame(None)
        assert list(out.columns) == list(OPTION_PREMIUM_COLUMNS)


# ---------------------------------------------------------------------------
# Writer: produce_ticker on a tiny synthetic larder
# ---------------------------------------------------------------------------


class TestProduceTicker:
    def _write_partition(self, larder: Path, ticker: str, expiration: str, df: pd.DataFrame):
        d = larder / f"ticker={ticker}" / f"expiration={expiration}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "data.parquet", index=False)

    def test_round_trip(self, tmp_path):
        larder = tmp_path / "larder"
        out = tmp_path / "out"
        self._write_partition(
            larder,
            "ZZZ",
            "20240216",
            _raw(
                [
                    {
                        "created": "2024-01-15",
                        "expiration": "20240216",
                        "strike": 100.0,
                        "right": "PUT",
                        "bid": 2.0,
                        "ask": 2.4,
                        "close": 2.2,
                        "volume": 7,
                        "open_interest": 100,
                    },
                ]
            ),
        )
        self._write_partition(
            larder,
            "ZZZ",
            "20240315",
            _raw(
                [
                    {
                        "created": "2024-02-15",
                        "expiration": "20240315",
                        "strike": 105.0,
                        "right": "CALL",
                        "bid": 1.0,
                        "ask": 1.4,
                        "close": 1.2,
                        "volume": 3,
                        "open_interest": 50,
                    },
                ]
            ),
        )
        stats = prod.produce_ticker("ZZZ", larder, out)
        assert stats["rows"] == 2
        assert stats["n_expirations"] == 2
        assert stats["date_min"] == "2024-01-15"
        assert stats["date_max"] == "2024-02-15"

        got = pd.read_parquet(out / "ZZZ.parquet")
        assert list(got.columns) == list(OPTION_PREMIUM_COLUMNS)
        assert len(got) == 2
        # sorted by date -> first row is the January put
        assert got.iloc[0]["right"] == "put"
        assert got.iloc[0]["mid"] == pytest.approx(2.2)

    def test_no_usable_rows_writes_nothing(self, tmp_path):
        larder = tmp_path / "larder"
        out = tmp_path / "out"
        self._write_partition(
            larder,
            "ZZZ",
            "20240620",  # far DTE -> filtered out of the belt
            _raw(
                [
                    {
                        "created": "2024-01-15",
                        "expiration": "20240620",
                        "strike": 100.0,
                        "right": "PUT",
                        "bid": 2.0,
                        "ask": 2.4,
                    },
                ]
            ),
        )
        stats = prod.produce_ticker("ZZZ", larder, out, dte_max=75)
        assert stats["rows"] == 0
        assert not (out / "ZZZ.parquet").exists()


# ---------------------------------------------------------------------------
# Accessors (against a synthetic produced parquet via SWE_OPTION_PREMIUM_DIR)
# ---------------------------------------------------------------------------


@pytest.fixture
def premium_dir(tmp_path, monkeypatch):
    """A produced option-premium dir wired into the connector via env var."""
    d = tmp_path / "option_premium"
    d.mkdir()
    monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(d))
    return d


def _write_produced(d: Path, ticker: str, rows: list[dict]):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df = df[list(OPTION_PREMIUM_COLUMNS)]
    df.to_parquet(d / f"{ticker}.parquet", index=False)


def _grid(dates, expiration, strikes):
    rows = []
    exp = pd.Timestamp(expiration)
    for dt in dates:
        for k in strikes:
            for right, base in (("put", 2.0), ("call", 1.5)):
                rows.append(
                    {
                        "date": dt,
                        "expiration": expiration,
                        "dte": (exp - pd.Timestamp(dt)).days,
                        "strike": float(k),
                        "right": right,
                        "bid": base,
                        "ask": base + 0.4,
                        "mid": base + 0.2,
                        "close": base + 0.1,
                        "volume": 10,
                        "open_interest": 100,
                    }
                )
    return rows


class TestOptionPremiumAccessor:
    def test_chain_pit_selects_latest_on_or_before_as_of(self, premium_dir):
        _write_produced(
            premium_dir,
            "AAA",
            _grid(["2024-01-02", "2024-01-03", "2024-01-10"], "2024-02-16", [95, 100, 105]),
        )
        c = MarketDataConnector()
        chain = c.get_option_premium_chain("AAA", "2024-02-16", as_of="2024-01-05")
        assert not chain.empty
        assert chain["date"].nunique() == 1
        assert chain["date"].iloc[0] == pd.Timestamp("2024-01-03")  # latest <= as_of
        assert len(chain) == 6  # 3 strikes x 2 rights

    def test_chain_excludes_future_quotes(self, premium_dir):
        _write_produced(
            premium_dir,
            "AAA",
            _grid(["2024-01-10"], "2024-02-16", [100]),
        )
        c = MarketDataConnector()
        assert c.get_option_premium_chain("AAA", "2024-02-16", as_of="2024-01-05").empty

    def test_chain_staleness_window(self, premium_dir):
        _write_produced(
            premium_dir,
            "AAA",
            _grid(["2024-01-02"], "2024-02-16", [100]),
        )
        c = MarketDataConnector()
        # 2024-01-02 quote is 10 calendar days stale vs as_of -> excluded at 7
        assert c.get_option_premium_chain(
            "AAA", "2024-02-16", as_of="2024-01-12", max_staleness_days=7
        ).empty
        assert not c.get_option_premium_chain(
            "AAA", "2024-02-16", as_of="2024-01-12", max_staleness_days=14
        ).empty

    def test_get_option_premium_nearest_strike(self, premium_dir):
        _write_produced(
            premium_dir,
            "AAA",
            _grid(["2024-01-03"], "2024-02-16", [95, 100, 105]),
        )
        c = MarketDataConnector()
        q = c.get_option_premium("AAA", "2024-02-16", 101.0, "put", as_of="2024-01-05")
        assert q is not None
        assert q["strike"] == 100.0  # nearest listed
        assert q["mid"] == pytest.approx(2.2)
        assert q["right"] == "put"

    def test_strike_tol_rejects_far(self, premium_dir):
        _write_produced(
            premium_dir,
            "AAA",
            _grid(["2024-01-03"], "2024-02-16", [95, 100, 105]),
        )
        c = MarketDataConnector()
        assert (
            c.get_option_premium(
                "AAA", "2024-02-16", 101.0, "put", as_of="2024-01-05", strike_tol=0.5
            )
            is None
        )
        assert (
            c.get_option_premium(
                "AAA", "2024-02-16", 101.0, "p", as_of="2024-01-05", strike_tol=2.0
            )
            is not None
        )  # "p" normalizes to put

    def test_missing_side_and_missing_ticker(self, premium_dir):
        _write_produced(premium_dir, "AAA", _grid(["2024-01-03"], "2024-02-16", [100]))
        c = MarketDataConnector()
        # ticker with no produced file -> None (caller falls back to synthetic)
        assert c.get_option_premium("ZZZ", "2024-02-16", 100.0, "put") is None
        assert c.get_option_premium_chain("ZZZ", "2024-02-16").empty

    def test_list_expirations_pit_and_dte_belt(self, premium_dir):
        rows = (
            _grid(["2024-01-03"], "2024-02-16", [100])  # 44 DTE
            + _grid(["2024-01-03"], "2024-03-15", [100])  # 72 DTE
            + _grid(["2024-01-03"], "2024-06-21", [100])  # 170 DTE
        )
        _write_produced(premium_dir, "AAA", rows)
        c = MarketDataConnector()
        exps = c.list_option_expirations("AAA", as_of="2024-01-03", min_dte=30, max_dte=75)
        assert exps == [pd.Timestamp("2024-02-16"), pd.Timestamp("2024-03-15")]

    def test_absent_dir_degrades(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWE_OPTION_PREMIUM_DIR", str(tmp_path / "does_not_exist"))
        c = MarketDataConnector()
        assert c.get_option_premium("AAA", "2024-02-16", 100.0, "put") is None
        assert c.get_option_premium_chain("AAA", "2024-02-16").empty
        assert c.list_option_expirations("AAA") == []

    def test_malformed_parquet_degrades_not_raises(self, premium_dir):
        # A present-but-malformed file (missing a contract column) must degrade
        # like an absent one — the accessors filter on expiration/right/date, so
        # an unvalidated frame would KeyError and crash the ranker. Connector
        # contract: missing data degrades, never crashes.
        bad = pd.DataFrame({"date": ["2024-01-03"], "strike": [100.0], "mid": [2.2]})
        bad.to_parquet(premium_dir / "BAD.parquet", index=False)  # no expiration/right
        c = MarketDataConnector()
        assert c.get_option_premium_chain("BAD", "2024-02-16", as_of="2024-01-05").empty
        assert c.get_option_premium("BAD", "2024-02-16", 100.0, "put") is None
        assert c.list_option_expirations("BAD", as_of="2024-01-05") == []


# ---------------------------------------------------------------------------
# Data-backed: the distiller works on a real larder partition (guarded)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_LARDER / "ticker=AAPL").exists(),
    reason="Theta option-history larder not present (laptop/sandbox-only)",
)
def test_distill_real_aapl_partition():
    parts = sorted((_LARDER / "ticker=AAPL").glob("expiration=2023*/data.parquet"))
    assert parts, "expected at least one 2023 AAPL expiration partition"
    raw = pd.read_parquet(parts[len(parts) // 2])
    out = prod.distill_expiration_frame(raw)
    if out.empty:
        pytest.skip("selected partition had no in-belt two-sided quotes")
    assert list(out.columns) == list(OPTION_PREMIUM_COLUMNS)
    assert (out["mid"] > 0).all()
    assert (out["ask"] >= out["bid"]).all()
    assert set(out["right"]).issubset({"put", "call"})
    assert (out["dte"] >= 0).all() and (out["dte"] <= 75).all()
