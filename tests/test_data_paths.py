"""engine.paths — SWE_DATA_ROOT re-roots the conventional data prefixes (D31).

Pins: unset root → paths unchanged (CWD-relative, the legacy behaviour);
set root → ``data/``, ``data_raw/``, ``data_processed/`` land under it;
absolute paths and unrelated relative paths are never touched; the narrower
overrides (``SWE_DATA_PROCESSED_DIR``, ``SWE_IBKR_DATA_DIR``,
``SWE_OPTION_PREMIUM_DIR``, ``SWE_SIM_DATA_DIR``) keep winning inside their
scope; and the connector honours the root through WheelRunner's untouched
``"data/bloomberg"`` default.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine import paths

_ROOT_VARS = (
    "SWE_DATA_ROOT",
    "SWE_DATA_PROCESSED_DIR",
    "SWE_IBKR_DATA_DIR",
    "SWE_OPTION_PREMIUM_DIR",
    "SWE_SIM_DATA_DIR",
)


@pytest.fixture
def clean_env(monkeypatch):
    for v in _ROOT_VARS:
        monkeypatch.delenv(v, raising=False)
    return monkeypatch


def test_unset_root_leaves_relative_paths_alone(clean_env):
    assert paths.data_root() is None
    assert paths.resolve("data/bloomberg") == Path("data/bloomberg")
    assert paths.bloomberg_dir() == Path("data/bloomberg")
    assert paths.raw_dir() == Path("data_raw")
    assert paths.processed_dir() == paths.repo_root() / "data_processed"
    assert paths.ibkr_dir() == paths.repo_root() / "data_processed" / "ibkr"


def test_set_root_reroots_the_three_prefixes(clean_env, tmp_path):
    clean_env.setenv("SWE_DATA_ROOT", str(tmp_path))
    assert paths.data_root() == tmp_path.resolve()
    assert paths.resolve("data/bloomberg") == tmp_path.resolve() / "data" / "bloomberg"
    assert paths.resolve("data_raw/ohlcv") == tmp_path.resolve() / "data_raw" / "ohlcv"
    assert paths.processed_dir() == tmp_path.resolve() / "data_processed"
    assert paths.theta_dir() == tmp_path.resolve() / "data_processed" / "theta"
    assert paths.ibkr_dir() == tmp_path.resolve() / "data_processed" / "ibkr"
    assert paths.sim_dir() == tmp_path.resolve() / "data_processed" / "sim"
    assert paths.option_premium_dir() == tmp_path.resolve() / "data_processed" / "option_premium"
    assert paths.deep_dir() == tmp_path.resolve() / "data" / "bloomberg" / "deep"


def test_absolute_and_unrelated_paths_are_untouched(clean_env, tmp_path):
    clean_env.setenv("SWE_DATA_ROOT", str(tmp_path))
    abs_path = tmp_path / "elsewhere" / "x.csv"
    assert paths.resolve(abs_path) == abs_path
    assert paths.resolve("tests/fixtures/ibkr") == Path("tests/fixtures/ibkr")
    assert paths.resolve("database/x") == Path("database/x")  # prefix must match a whole component


def test_narrow_overrides_win_and_are_themselves_rerooted(clean_env, tmp_path):
    clean_env.setenv("SWE_DATA_ROOT", str(tmp_path))
    elsewhere = tmp_path.parent / "abs-ibkr"  # absolute on every OS ("/abs" is not, on Windows)
    clean_env.setenv("SWE_IBKR_DATA_DIR", str(elsewhere))
    clean_env.setenv("SWE_OPTION_PREMIUM_DIR", str(tmp_path / "rail"))
    clean_env.setenv("SWE_SIM_DATA_DIR", "data_processed_b/sim")  # relative → under the root
    clean_env.setenv(
        "SWE_DATA_PROCESSED_DIR", "data_processed_b"
    )  # any relative value, not only data*
    root = tmp_path.resolve()
    assert paths.ibkr_dir() == elsewhere  # absolute: as given
    assert paths.option_premium_dir() == tmp_path / "rail"
    assert paths.sim_dir() == root / "data_processed_b" / "sim"
    assert paths.processed_dir() == root / "data_processed_b"
    assert paths.theta_dir() == root / "data_processed_b" / "theta"


def test_relative_overrides_stay_relative_without_a_root(clean_env):
    """No root: a relative override keeps the legacy CWD-relative meaning."""
    clean_env.setenv("SWE_DATA_PROCESSED_DIR", "data_processed_b")
    clean_env.setenv("SWE_SIM_DATA_DIR", "scratch/sim")
    assert paths.processed_dir() == Path("data_processed_b")
    assert paths.sim_dir() == Path("scratch/sim")
    assert paths.ibkr_dir() == Path("data_processed_b") / "ibkr"


def test_connector_follows_the_root_through_the_trio_default(clean_env, tmp_path):
    """WheelRunner passes "data/bloomberg"; the connector must land under the root."""
    from engine.data_connector import MarketDataConnector

    clean_env.setenv("SWE_DATA_ROOT", str(tmp_path))
    conn = MarketDataConnector("data/bloomberg")
    assert conn._data_dir == tmp_path.resolve() / "data" / "bloomberg"
    assert conn._option_premium_dir == tmp_path.resolve() / "data_processed" / "option_premium"
    explicit = MarketDataConnector(str(tmp_path / "explicit"))
    assert explicit._data_dir == tmp_path / "explicit"
