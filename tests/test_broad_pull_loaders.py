"""Tests for the Phase-0B broad-pull loaders (``data/broad_pull_loaders.py``).

Two layers:

* **Synthetic unit tests** (always run) — exercise the loader logic on tiny
  in-memory / tmp fixtures: gz reading, winsorization (clamp + log + row-count
  preservation), float32 downcast, date parsing, PIT ``series`` slicing,
  ``category_series``, ``snapshot_row``, registry consistency, and the §2
  boundary guard (no production module imports this loader).
* **Real-data tests** (skipped when ``data/bloomberg/broad_pull/`` is absent,
  mirroring ``HAS_BLOOMBERG_DATA`` in ``test_data_integrity_bloomberg.py``) —
  pin every dataset's row count / date range / ticker count / schema to the
  byte-verified manifest (``docs/DATA_INVENTORY.md`` §6), confirm gz panels
  load, and verify the flagged columns are genuinely winsorized vs the raw bytes.

Policy mirrors the integrity suite: a test that fails because the data is
genuinely wrong is a SUCCESS — we do not soften assertions to go green.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.broad_pull_loaders import (
    SPECS,
    BroadPullLoader,
    get_broad_pull_loader,
)

BROAD_PULL_DIR = Path("data/bloomberg/broad_pull")
HAS_BROAD_PULL_DATA = (BROAD_PULL_DIR / "macro_vol" / "sp500_vol_indices.csv").exists()

# Byte-verified facts (scan of the staged bytes == BROAD_PULL_MANIFEST.md ==
# docs/DATA_INVENTORY.md §6). rows / date range / distinct tickers / must-have
# columns. tickers=None for market-level (wide/long) series.
EXPECTED: dict[str, dict] = {
    "iv_surface": {
        "rows": 1_944_699,
        "dmin": "2010-01-04",
        "dmax": "2026-06-17",
        "tickers": 509,
        "cols": ("date", "ticker", "iv_30d_100", "iv_365d_110"),
    },
    "vol_term_rv": {
        "rows": 1_963_364,
        "dmin": "2010-01-04",
        "dmax": "2026-06-18",
        "tickers": 510,
        "cols": ("date", "ticker", "atm_iv_30d", "atm_iv_730d", "rv_10d", "rv_260d"),
    },
    "macro_calendar": {
        "rows": 352,
        "dmin": "2025-01-02",
        "dmax": "2027-12-08",
        "tickers": None,
        "cols": ("event", "release_date", "release_time"),
        "categories": 11,
    },
    "macro_releases": {
        "rows": 4_724,
        "dmin": "2015-01-01",
        "dmax": "2026-06-17",
        "tickers": None,
        "cols": ("event", "date", "actual"),
        "categories": 11,
    },
    "vol_indices": {
        "rows": 5_847,
        "dmin": "2004-01-01",
        "dmax": "2026-06-17",
        "tickers": None,
        "cols": ("date", "vix", "vvix", "skew", "move", "cvix"),
    },
    "spx_correlation": {
        "rows": 5_146,
        "dmin": "2006-01-03",
        "dmax": "2026-06-17",
        "tickers": None,
        "cols": ("date", "cor1m", "cor3m", "cor6m"),
    },
    "credit_spreads": {
        "rows": 5_647,
        "dmin": "2004-01-02",
        "dmax": "2026-06-16",
        "tickers": None,
        "cols": ("date", "ig_oas", "hy_oas"),
    },
    "vix_futures_curve": {
        "rows": 5_150,
        "dmin": "2006-01-03",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "ux1", "ux7"),
    },
    "ois_sofr_curve": {
        "rows": 6_393,
        "dmin": "2001-12-04",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "ois_1m", "ois_30y", "sofr_on"),
    },
    "real_yields": {
        "rows": 6_900,
        "dmin": "2000-01-03",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "tips_2y", "infl_swap_10y"),
    },
    "fed_funds": {
        "rows": 6_850,
        "dmin": "2000-01-03",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "fed_target", "ff_fut_front"),
    },
    "macro_surprise": {
        "rows": 6_044,
        "dmin": "2003-01-01",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "citi_surprise_usd", "citi_surprise_g10"),
    },
    "fx": {
        "rows": 6_904,
        "dmin": "2000-01-03",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "dxy", "eurusd", "usdjpy", "gbpusd"),
    },
    "commodities": {
        "rows": 6_652,
        "dmin": "2000-01-04",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "wti", "gold", "copper", "natgas"),
    },
    "global_vol": {
        "rows": 6_880,
        "dmin": "2000-01-03",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "vstoxx", "cdx_ig_5y", "cdx_hy_5y"),
    },
    "sector_factor_etfs": {
        "rows": 94_646,
        "dmin": "1998-01-02",
        "dmax": "2026-06-18",
        "tickers": None,
        "cols": ("date", "etf", "open", "close", "volume"),
        "categories": 15,
    },
    "returns_micro": {
        "rows": 1_874_882,
        "dmin": "2010-01-04",
        "dmax": "2026-06-18",
        "tickers": 511,
        "cols": ("date", "ticker", "tot_return", "px_bid", "px_ask"),
    },
    "beta_shares": {
        "rows": 93_605,
        "dmin": "2010-01-29",
        "dmax": "2026-05-29",
        "tickers": 510,
        "cols": ("date", "ticker", "beta_raw", "shares_out"),
    },
    "fundamentals_q": {
        "rows": 31_479,
        "dmin": "2010-01-01",
        "dmax": "2026-05-31",
        "tickers": 511,
        "cols": ("date", "ticker", "revenue", "eps", "roe"),
    },
    "fundamentals_ext_q": {
        "rows": 31_470,
        "dmin": "2010-01-01",
        "dmax": "2026-05-31",
        "tickers": 511,
        "cols": ("date", "ticker", "roic", "net_margin", "debt_to_equity"),
    },
    "estimates_m": {
        "rows": 92_680,
        "dmin": "2010-01-29",
        "dmax": "2026-05-29",
        "tickers": 511,
        "cols": ("date", "ticker", "best_eps", "best_target", "analyst_count"),
    },
    "estimates_fwd": {
        "rows": 93_169,
        "dmin": "2010-01-29",
        "dmax": "2026-05-29",
        "tickers": 511,
        "cols": ("date", "ticker", "best_eps_1bf", "best_eps_2bf"),
    },
    "valuation_m": {
        "rows": 89_079,
        "dmin": "2010-01-29",
        "dmax": "2026-05-29",
        "tickers": 509,
        "cols": ("date", "ticker", "px_to_book", "ev_to_ebitda", "pe", "peg"),
    },
    "options_sentiment": {
        "rows": 1_998_083,
        "dmin": "2010-01-01",
        "dmax": "2026-06-18",
        "tickers": 511,
        "cols": ("date", "ticker", "pc_oi_ratio", "pc_vol_ratio", "oi_call", "oi_put", "news_sent"),
    },
    "dividend_pit": {
        "rows": 72_461,
        "dmin": "2010-01-29",
        "dmax": "2026-05-29",
        "tickers": 421,
        "cols": ("date", "ticker", "dvd_yld_12m", "dvd_yld_ind", "dvd_sh_12m"),
    },
    "short_interest": {
        "rows": 134_035,
        "dmin": "2015-01-15",
        "dmax": "2026-05-29",
        "tickers": 509,
        "cols": ("date", "ticker", "short_interest", "short_int_ratio"),
    },
    "snapshot_bdp": {
        "rows": 511,
        "dmin": "2026-06-18",
        "dmax": "2026-06-18",
        "tickers": 511,
        "cols": ("asof", "ticker", "rtg_sp", "gics_sector", "next_earnings_dt"),
    },
}

# Datasets whose flagged columns are winsorized (must match SPECS.winsor_cols).
_FLAGGED = {name: spec.winsor_cols for name, spec in SPECS.items() if spec.winsor_cols}


# ===================================================================== #
# Synthetic unit tests (always run — no real data needed)
# ===================================================================== #


def test_every_expected_dataset_has_a_spec_and_vice_versa():
    assert set(EXPECTED) == set(SPECS), "EXPECTED and SPECS must enumerate the same datasets"


def test_specs_registry_consistent():
    seen_paths: set[str] = set()
    valid_kinds = {"ticker_ts", "wide_ts", "long_ts", "snapshot"}
    for name, spec in SPECS.items():
        assert spec.name == name
        assert spec.kind in valid_kinds
        assert spec.relpath not in seen_paths, f"duplicate relpath {spec.relpath}"
        seen_paths.add(spec.relpath)
        if spec.kind in ("ticker_ts", "snapshot"):
            assert spec.ticker_col is not None
        if spec.kind == "long_ts":
            assert spec.category_col is not None


def test_named_loader_exists_for_every_spec():
    loader = BroadPullLoader()
    for name in SPECS:
        assert hasattr(loader, f"load_{name}"), f"missing load_{name}()"


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        BroadPullLoader().load("does_not_exist")


def test_missing_file_returns_none(tmp_path):
    loader = BroadPullLoader(data_dir=tmp_path)
    assert loader.load("vol_indices") is None


def test_reads_plain_csv(tmp_path):
    (tmp_path / "macro_vol").mkdir(parents=True)
    df = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "VIX": [13.1, 14.2]})
    df.to_csv(tmp_path / "macro_vol" / "sp500_vol_indices.csv", index=False)
    out = BroadPullLoader(data_dir=tmp_path).load("vol_indices")
    assert out is not None and len(out) == 2
    assert "vix" in out.columns  # lowercased
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_reads_gzip(tmp_path):
    """gz panels (iv_surface, vol_term_rv) read via compression='infer'."""
    (tmp_path / "iv_surface").mkdir(parents=True)
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "ticker": ["AAPL", "AAPL"],
            "iv_30d_100": [0.25, 0.26],
        }
    )
    df.to_csv(tmp_path / "iv_surface" / "sp500_iv_surface.csv.gz", index=False)
    out = BroadPullLoader(data_dir=tmp_path).load("iv_surface")
    assert out is not None and len(out) == 2 and "iv_30d_100" in out.columns


def test_winsorize_clamps_and_preserves_rows():
    df = pd.DataFrame({"x": list(range(1000)) + [10_000_000, -10_000_000]})
    n_before = len(df)
    out = BroadPullLoader._winsorize(df.copy(), ("x",))
    assert len(out) == n_before, "winsorize must not drop rows"
    assert out["x"].max() < 10_000_000 and out["x"].min() > -10_000_000


def test_winsorize_logs_when_clamping(caplog):
    df = pd.DataFrame({"x": list(range(1000)) + [10_000_000]})
    with caplog.at_level(logging.WARNING, logger="data.broad_pull_loaders"):
        BroadPullLoader._winsorize(df, ("x",))
    assert any("winsorized" in r.message for r in caplog.records)


def test_winsorize_skips_missing_and_all_nan_columns():
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
    out = BroadPullLoader._winsorize(df.copy(), ("x", "absent"))  # must not raise
    assert out["x"].isna().all()


def test_downcast_floats():
    df = pd.DataFrame({"a": pd.Series([1.0, 2.0], dtype="float64"), "b": [1, 2]})
    out = BroadPullLoader._downcast_floats(df.copy())
    assert out["a"].dtype == np.float32
    assert out["b"].dtype != np.float32  # ints untouched


def test_series_pit_filter():
    loader = BroadPullLoader()
    loader._panels["returns_micro"] = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "tot_return": [1.0, 1.1, 2.0],
        }
    )
    full = loader.series("returns_micro", "aapl")
    assert list(full["tot_return"]) == [1.0, 1.1]  # only AAPL, sorted ascending
    pit = loader.series("returns_micro", "AAPL", as_of="2024-01-02")
    assert list(pit["tot_return"]) == [1.0]  # no-look-ahead


def test_series_rejects_non_ticker_dataset():
    with pytest.raises(ValueError):
        BroadPullLoader().series("vol_indices", "AAPL")


def test_category_series():
    loader = BroadPullLoader()
    loader._panels["sector_factor_etfs"] = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "etf": ["XLK", "XLE"],
            "close": [1.0, 2.0],
        }
    )
    out = loader.category_series("sector_factor_etfs", "XLK")
    assert len(out) == 1 and out.iloc[0]["close"] == 1.0


def test_snapshot_row():
    loader = BroadPullLoader()
    loader._panels["snapshot_bdp"] = pd.DataFrame(
        {"asof": pd.to_datetime(["2026-06-18"]), "ticker": ["AAPL"], "rtg_sp": ["AA+"]}
    )
    row = loader.snapshot_row("snapshot_bdp", "aapl")
    assert row is not None and row["rtg_sp"] == "AA+"
    assert loader.snapshot_row("snapshot_bdp", "ZZZZ") is None


def test_singleton_is_stable():
    assert get_broad_pull_loader() is get_broad_pull_loader()


def test_module_not_consumed_by_production():
    """§2 boundary, enforced structurally: no decision-path module imports the
    broad-pull loader. If a future wiring step adds such an import it is
    EV-moving (supervised) and this test must be updated deliberately.
    """
    forbidden = [
        Path("engine/ev_engine.py"),
        Path("engine/wheel_runner.py"),
        Path("engine/candidate_dossier.py"),
        Path("engine/data_connector.py"),
        Path("engine/skew_dynamics.py"),
        Path("engine/event_gate.py"),
        Path("engine/regime_detector.py"),
        Path("engine/option_pricer.py"),
        Path("engine/risk_manager.py"),
        Path("data/consolidated_loader.py"),
    ]
    for path in forbidden:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "broad_pull_loaders" not in text, (
                f"{path} imports/references broad_pull_loaders — that is EV-moving "
                f"Phase 1-3 wiring, out of scope for Phase 0B"
            )


# ===================================================================== #
# Real-data tests (skipped when broad_pull/ is absent)
# ===================================================================== #

pytestmark_real = pytest.mark.skipif(
    not HAS_BROAD_PULL_DATA, reason="broad-pull data not present (data/bloomberg/broad_pull)"
)


def _fresh() -> BroadPullLoader:
    """A fresh loader per test so only one panel is resident at a time."""
    return BroadPullLoader()


@pytestmark_real
@pytest.mark.parametrize("name", list(EXPECTED))
def test_real_dataset_matches_manifest(name):
    exp = EXPECTED[name]
    spec = SPECS[name]
    df = _fresh().load(name)
    assert df is not None, f"{name} failed to load"

    # row count (byte-pinned)
    assert len(df) == exp["rows"], f"{name}: rows {len(df)} != {exp['rows']}"

    # date range
    dates = pd.to_datetime(df[spec.date_col], errors="coerce")
    assert str(dates.min().date()) == exp["dmin"], f"{name}: dmin {dates.min().date()}"
    assert str(dates.max().date()) == exp["dmax"], f"{name}: dmax {dates.max().date()}"

    # schema
    for col in exp["cols"]:
        assert col in df.columns, f"{name}: missing column {col}"

    # ticker count
    if exp["tickers"] is not None:
        assert spec.ticker_col in df.columns
        assert df[spec.ticker_col].nunique() == exp["tickers"], (
            f"{name}: tickers {df[spec.ticker_col].nunique()} != {exp['tickers']}"
        )

    # category count (long_ts)
    if exp.get("categories") is not None and spec.category_col:
        assert df[spec.category_col].nunique() == exp["categories"], (
            f"{name}: categories {df[spec.category_col].nunique()} != {exp['categories']}"
        )


@pytestmark_real
@pytest.mark.parametrize("name", ["iv_surface", "vol_term_rv"])
def test_real_gz_panels_load(name):
    """The two gzipped panels read correctly through compression='infer'."""
    df = _fresh().load(name)
    assert df is not None and len(df) == EXPECTED[name]["rows"]
    assert SPECS[name].relpath.endswith(".gz")


@pytestmark_real
@pytest.mark.parametrize("name", list(_FLAGGED))
def test_real_winsorization_applied(name):
    """Flagged columns are genuinely clipped vs the raw bytes (not silent)."""
    cols = list(_FLAGGED[name])
    loaded = _fresh().load(name)
    assert loaded is not None
    raw = pd.read_csv(BROAD_PULL_DIR / SPECS[name].relpath, usecols=cols)
    for col in cols:
        rs = pd.to_numeric(raw[col], errors="coerce")
        lo, hi = rs.quantile(0.001), rs.quantile(0.999)
        if pd.isna(lo) or pd.isna(hi):
            continue
        ls = pd.to_numeric(loaded[col], errors="coerce")
        tol = abs(hi) * 1e-3 + abs(lo) * 1e-3 + 1e-6
        assert ls.max() <= hi + tol, f"{name}.{col}: loaded max {ls.max()} > clip hi {hi}"
        assert ls.min() >= lo - tol, f"{name}.{col}: loaded min {ls.min()} < clip lo {lo}"
        # if the raw tail exceeded the clip bound, clamping must have lowered the max
        if rs.max() > hi + tol:
            assert ls.max() < rs.max(), f"{name}.{col}: clamp did not reduce the max"


@pytestmark_real
def test_real_date_columns_are_datetime():
    for name in EXPECTED:
        df = _fresh().load(name)
        spec = SPECS[name]
        assert pd.api.types.is_datetime64_any_dtype(df[spec.date_col]), (
            f"{name}: {spec.date_col} not parsed to datetime"
        )


@pytestmark_real
def test_real_series_pit_and_snapshot():
    loader = _fresh()
    # a ticker_ts series is sorted ascending and PIT-filterable
    s = loader.series("returns_micro", "AAPL")
    assert s is not None and len(s) > 100
    assert s["date"].is_monotonic_increasing
    early = loader.series("returns_micro", "AAPL", as_of="2015-01-01")
    assert early["date"].max() <= pd.Timestamp("2015-01-01")
    # snapshot row
    loader2 = _fresh()
    row = loader2.snapshot_row("snapshot_bdp", "AAPL")
    assert row is not None and "gics_sector" in row


@pytestmark_real
def test_real_bid_ask_ordering_intact():
    """returns_micro: px_bid <= px_ask where both present (manifest: 0.0000% inverted)."""
    df = _fresh().load("returns_micro")
    both = df.dropna(subset=["px_bid", "px_ask"])
    inverted = (both["px_bid"] > both["px_ask"]).mean()
    assert inverted == 0.0, f"px_bid>px_ask in {inverted:.4%} of rows"
