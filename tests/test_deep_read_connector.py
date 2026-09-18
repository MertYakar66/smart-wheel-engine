"""R2 deep-read connector tests — docs/DATA_LAYER_DEEP_READ_DESIGN.md Part A.

Two tiers:

* **CI-runnable** (committed repo data, which has no ``deep/`` slices): the
  ``deep_history`` flag plumbing, and that ``deep_history=True`` *degrades to the
  monolith* when the deep slices are absent (so ON == OFF with no deep data).
* **Local-only** (skipped unless ``SWE_DEEP_TEST_DATA`` points at a dir that
  contains ``deep/`` slices): assembly reaches 1994, default-OFF ignores deep even
  when present, a delisted name returns its last bar, the rotation invariant holds
  post-assembly, and schema parity.

The deep + delisted gz are not committed (they live on the buffer branch), so the
assembly assertions run locally against a materialized data dir and skip in CI.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from engine import paths
from engine.data_connector import MarketDataConnector

# The served monoliths (2018+) under the data root (D31). On the desktop root
# the deep/ slices sit next to them; in a sandbox or CI they usually do not.
REPO_DATA = str(paths.bloomberg_dir())
_MONOLITH = Path(REPO_DATA) / "sp500_ohlcv.csv"

# The assembly tier needs deep/ next to the monoliths: the data root itself
# once the slices are materialized there, or an ad-hoc dir via
# SWE_DEEP_TEST_DATA (e.g. C:\tmp\deepdata\bloomberg).
_DEEP = os.environ.get("SWE_DEEP_TEST_DATA") or REPO_DATA

deep_data = pytest.mark.skipif(
    not (_DEEP and (Path(_DEEP) / "deep" / "sp500_ohlcv__1994_2018.csv.gz").exists()),
    reason="deep/ slices absent under the data root (materialize them, D31) and SWE_DEEP_TEST_DATA unset",
)


# --------------------------------------------------------------------------
# CI-runnable: flag plumbing + graceful degrade
# --------------------------------------------------------------------------


def test_default_is_deep_off():
    """Default-OFF is the whole safety contract — adoption is a re-baseline event."""
    assert MarketDataConnector()._deep_history is False


def test_env_var_enables_deep(monkeypatch):
    monkeypatch.setenv("SWE_DEEP_HISTORY", "1")
    assert MarketDataConnector()._deep_history is True
    # An explicit argument always wins over the env var.
    assert MarketDataConnector(deep_history=False)._deep_history is False


def test_env_var_falsey_stays_off(monkeypatch):
    monkeypatch.setenv("SWE_DEEP_HISTORY", "0")
    assert MarketDataConnector()._deep_history is False
    monkeypatch.setenv("SWE_DEEP_HISTORY", "")
    assert MarketDataConnector()._deep_history is False


@pytest.mark.skipif(not _MONOLITH.exists(), reason="served OHLCV monolith absent (no data root)")
def test_deep_on_without_slices_degrades_to_monolith(tmp_path):
    """deep_history=True against a dir holding the monolith but no deep/:
    missing slices are logged + skipped, so the result equals the OFF result.
    Built in tmp_path (a copy of the OHLCV monolith only) so the guarantee is
    tested even on a data root where the deep slices are present (D31)."""
    lone = tmp_path / "bloomberg"
    lone.mkdir()
    shutil.copyfile(_MONOLITH, lone / "sp500_ohlcv.csv")
    assert not (lone / "deep").exists()
    on = MarketDataConnector(str(lone), deep_history=True).get_ohlcv("AAPL")
    off = MarketDataConnector(str(lone), deep_history=False).get_ohlcv("AAPL")
    assert not off.empty
    pd.testing.assert_frame_equal(on, off)


# --------------------------------------------------------------------------
# Local-only: real assembly against materialized deep data
# --------------------------------------------------------------------------


@deep_data
def test_default_off_ignores_deep_even_when_present():
    """The critical default-OFF guarantee: with deep/ present on disk,
    deep_history=False STILL loads only the 2018+ monolith — no pre-2018 rows,
    no delisted names."""
    conn = MarketDataConnector(_DEEP, deep_history=False)
    aapl = conn.get_ohlcv("AAPL")
    assert aapl.index.min().year >= 2018
    # A delisted-only name (Lehman) is absent from the monolith.
    assert conn.get_ohlcv("LEHMQ").empty


@deep_data
def test_assembled_ohlcv_reaches_1994_and_dedups():
    df = MarketDataConnector(_DEEP, deep_history=True).get_ohlcv("AAPL")
    assert df.index.min() <= pd.Timestamp("1994-12-31")
    assert df.index.max() >= pd.Timestamp("2026-01-01")
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates  # dedup keep-first held


@deep_data
def test_delisted_lehman_returns_last_bar():
    """Lehman (PIT key LEHMQ) flows through get_ohlcv like any name; last bar
    2008-09-12, rotation-fixed close ~3.65 (stored high column)."""
    df = MarketDataConnector(_DEEP, deep_history=True).get_ohlcv("LEHMQ")
    assert not df.empty
    assert df.index.max() == pd.Timestamp("2008-09-12")
    last = df.iloc[-1]
    assert last["close"] == pytest.approx(3.65, abs=0.01)


@deep_data
def test_rotation_invariant_holds_post_assembly():
    """After concat + the get_ohlcv rename, high is the row max and low the row
    min — 0 violations across AAPL's full assembled history."""
    df = MarketDataConnector(_DEEP, deep_history=True).get_ohlcv("AAPL")
    df = df.dropna(subset=["open", "high", "low", "close"])
    rowmax = df[["open", "high", "low", "close"]].max(axis=1)
    rowmin = df[["open", "high", "low", "close"]].min(axis=1)
    assert ((df["high"] - rowmax).abs() < 1e-6).all(), "high is not the row max"
    assert ((df["low"] - rowmin).abs() < 1e-6).all(), "low is not the row min"


@deep_data
def test_schema_parity_monolith_vs_deep():
    base = Path(_DEEP)
    mono = pd.read_csv(base / "sp500_ohlcv.csv", nrows=1).columns.tolist()
    for rel in (
        "deep/sp500_ohlcv__1994_2018.csv.gz",
        "deep/sp500_ohlcv__delisted.csv.gz",
    ):
        cols = pd.read_csv(base / rel, nrows=1, compression="gzip").columns.tolist()
        assert cols == mono, f"{rel} schema {cols} != monolith {mono}"


@deep_data
def test_vol_iv_assembled_reaches_1994():
    df = MarketDataConnector(_DEEP, deep_history=True).get_iv_history("AAPL")
    assert not df.empty
    assert df.index.min() <= pd.Timestamp("1994-12-31")
