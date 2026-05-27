"""Backtest regression tests — lock S27/S32/S34/S35 against current engine.

Runs each reproducer end-to-end and compares the metrics dict to the
committed snapshot in ``backtests/regression/snapshots/<id>.json``.
Failure means the engine output has drifted from the documented
backtest — investigate before re-baselining (see ``TESTING.md`` §
"Backtest regression — re-baseline workflow").

These tests are slow (S27 ~20 min, S32 ~30 min, S34 ~3.5 h, S35 ~40
min). They are gated behind the ``backtest_regression`` marker and
excluded from the default per-PR CI lane. Run manually via the
``.claude/commands/backtest-regression.md`` skill or the dedicated
``.github/workflows/backtest-regression.yml`` workflow.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import pytest

from backtests.regression._common import load_snapshot
from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100

# ---------------------------------------------------------------------------
# Tolerance schema
# ---------------------------------------------------------------------------
# Each metric key maps to a (kind, tolerance) tuple. ``exact`` requires
# integer equality (counts must not drift); ``abs`` and ``rel`` use
# ``pytest.approx`` semantics. Per-bucket keys (rho, p, hit, pnl_mean,
# ev_mean, n, iv_mean, mean_realized) reuse the same scale as the
# top-level aggregate.

_TOLERANCES: dict[str, tuple[str, float]] = {
    # Counts — exact
    "row_count": ("exact", 0),
    "executed_trades": ("exact", 0),
    "put_assignments": ("exact", 0),
    "open_at_end": ("exact", 0),
    "n": ("exact", 0),
    # Correlations / hit rates — absolute (statistical noise floor)
    "spearman_rho": ("abs", 0.005),
    "rho": ("abs", 0.005),
    "hit_rate": ("abs", 0.005),
    "hit": ("abs", 0.005),
    # P-values — relative (large dynamic range near 1e-67)
    "spearman_p": ("rel", 0.01),
    "p": ("rel", 0.01),
    # Dollars / IV — tight relative
    "final_nav": ("rel", 1e-5),
    "final_cash": ("rel", 1e-5),
    "mean_realized": ("rel", 1e-4),
    "ev_mean": ("rel", 1e-4),
    "pnl_mean": ("rel", 1e-4),
    "iv_mean": ("abs", 1e-4),
}

_FINGERPRINT_REQUIRED = (
    "capital",
    "tickers",
    "universe_size",
    "start",
    "end",
    "seed",
    "friction_level",
    "data_csv_sha256",
)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _assert_metric(expected: Any, actual: Any, key: str, path: str) -> None:
    """Compare a single metric value against the snapshot using ``_TOLERANCES``.

    Unknown keys are skipped silently — adding new metrics to the
    snapshot schema doesn't require updating the test.
    """
    spec = _TOLERANCES.get(key)
    if spec is None:
        return
    if isinstance(expected, float) and math.isnan(expected):
        assert isinstance(actual, float) and math.isnan(actual), (
            f"{path}: snapshot is NaN, got {actual!r}"
        )
        return
    kind, tol = spec
    if kind == "exact":
        assert expected == actual, f"{path}: expected {expected!r}, got {actual!r}"
    elif kind == "abs":
        assert actual == pytest.approx(expected, abs=tol), (
            f"{path}: expected {expected} ±{tol} (abs), got {actual}"
        )
    elif kind == "rel":
        assert actual == pytest.approx(expected, rel=tol), (
            f"{path}: expected {expected} ±{tol * 100:g}% (rel), got {actual}"
        )
    else:  # pragma: no cover — defensive
        raise ValueError(f"Unknown tolerance kind {kind!r} for {key}")


def _compare_dict(expected: dict, actual: dict, path: str = "") -> None:
    """Recursively walk a snapshot dict, comparing leaf values."""
    for key, exp_val in expected.items():
        sub_path = f"{path}/{key}" if path else key
        if isinstance(exp_val, dict):
            assert isinstance(actual, dict) and key in actual, (
                f"{sub_path}: missing in actual output"
            )
            _compare_dict(exp_val, actual[key], sub_path)
        else:
            _assert_metric(
                exp_val, actual.get(key) if isinstance(actual, dict) else None, key, sub_path
            )


# ---------------------------------------------------------------------------
# Structural tests (fast, not behind marker)
# ---------------------------------------------------------------------------


def test_universes_match_connector():
    """``UNIVERSE_100`` is asserted to remain `get_universe()[:100]`.

    If the connector's universe drifts (ticker added/removed/reordered),
    snapshots silently invalidate. This test fails fast so the drift is
    explicit.
    """
    from engine.data_connector import MarketDataConnector

    derived = tuple(MarketDataConnector().get_universe()[:100])
    assert derived == UNIVERSE_100, (
        f"UNIVERSE_100 has drifted. Connector now returns "
        f"{derived[:5]}...{derived[-3:]}; pinned starts {UNIVERSE_100[:5]}... "
        f"and ends {UNIVERSE_100[-3:]}. Regenerate the constant and re-baseline."
    )
    # UNIVERSE_24 sanity — every name must exist in the connector's universe
    full = set(MarketDataConnector().get_universe())
    missing = [t for t in UNIVERSE_24 if t not in full]
    assert not missing, f"UNIVERSE_24 has tickers no longer in connector: {missing}"


@pytest.mark.parametrize(
    "snapshot_id",
    ["s27_ivpit_24t_100k", "s32_friction_24t_1m", "s34_universe_100t_1m", "s35_oos_24t_100k"],
)
def test_snapshot_fingerprints_have_required_keys(snapshot_id):
    """Every committed snapshot must carry the fingerprint keys the
    re-baseline workflow depends on."""
    try:
        snap = load_snapshot(snapshot_id)
    except FileNotFoundError:
        pytest.skip(f"snapshot {snapshot_id}.json not yet committed")
    fp = snap.get("fingerprint", {})
    missing = [k for k in _FINGERPRINT_REQUIRED if k not in fp]
    assert not missing, f"{snapshot_id}: fingerprint missing keys {missing}"


# ---------------------------------------------------------------------------
# The slow regression tests (behind marker)
# ---------------------------------------------------------------------------

_BACKTESTS = [
    ("s27_ivpit_24t_100k", "backtests.regression.s27_ivpit_24t_100k"),
    ("s32_friction_24t_1m", "backtests.regression.s32_friction_24t_1m"),
    ("s34_universe_100t_1m", "backtests.regression.s34_universe_100t_1m"),
    ("s35_oos_24t_100k", "backtests.regression.s35_oos_24t_100k"),
]


@pytest.mark.backtest_regression
@pytest.mark.slow
@pytest.mark.parametrize("snapshot_id,reproducer", _BACKTESTS)
def test_backtest_matches_snapshot(snapshot_id, reproducer):
    """Run the reproducer against current engine; assert metrics match the
    committed snapshot within the per-metric tolerance band.

    Skips when the snapshot file is absent (e.g. S35 pending PR4
    re-baseline). Failure indicates engine drift — see TESTING.md
    "Backtest regression — re-baseline workflow" before regenerating.
    """
    try:
        snap = load_snapshot(snapshot_id)
    except FileNotFoundError:
        pytest.skip(f"snapshot {snapshot_id}.json not yet committed")

    module = importlib.import_module(reproducer)
    result = module.run()  # uses CANONICAL args

    # ``module.build_payload`` wraps whatever ``run()`` returns into the
    # snapshot payload shape — same call path the reproducer's CLI uses
    # under ``--update-snapshot``, so we compare apples to apples.
    actual_payload = module.build_payload(result)

    # Compare metric sections; fingerprint differs by ``generated_at`` /
    # ``engine_sha_at_snapshot_lock`` between runs, so we compare only
    # the metric leaves.
    for section in ("aggregate", "per_year", "per_quartile"):
        if section in snap:
            assert section in actual_payload, (
                f"{snapshot_id}: actual output missing section {section!r}"
            )
            _compare_dict(snap[section], actual_payload[section], path=section)

    # Per-friction-level (S32/S34/S35 only)
    if "per_friction_level" in snap:
        assert "per_friction_level" in actual_payload, (
            f"{snapshot_id}: actual output missing per_friction_level"
        )
        for level, level_snap in snap["per_friction_level"].items():
            assert level in actual_payload["per_friction_level"], (
                f"{snapshot_id}: actual output missing friction level {level!r}"
            )
            for section in ("aggregate", "per_year", "per_quartile"):
                if section in level_snap:
                    _compare_dict(
                        level_snap[section],
                        actual_payload["per_friction_level"][level].get(section, {}),
                        path=f"per_friction_level/{level}/{section}",
                    )
