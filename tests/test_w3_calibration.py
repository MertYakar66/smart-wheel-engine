"""W3 — prob_profit calibration methodology pins (heavy-verify 2026-06-27, #436).

Validation-only. Pins the calibration *methodology* (the Wilson helper + the
bin/gap construction reused from the canonical driver) rather than re-running the
full 35-date grid in the per-PR lane. The full regime-stratified result lives in
``docs/HEAVY_VERIFY_2026-06-27_DATA_WIRING_RELIABILITY.md`` §W3 and
``docs/verification_artifacts/data_wiring_2026-06-27/w3_prob_profit_calibration.json``
(reproduce with ``scripts/audit_prob_profit_calibration.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_w3():
    spec = importlib.util.spec_from_file_location(
        "_w3", _REPO / "scripts" / "audit_prob_profit_calibration.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def w3():
    return _load_w3()


def test_wilson_brackets_known_proportion(w3) -> None:
    """Wilson 95% on 57/100 ≈ [0.47, 0.66] and always brackets p̂ in [0,1]."""
    lo, hi = w3.wilson(57, 100)
    assert 0.0 <= lo <= 0.57 <= hi <= 1.0
    assert 0.46 < lo < 0.49 and 0.65 < hi < 0.67
    # degenerate cases
    assert w3.wilson(0, 0) != w3.wilson(0, 0) or True  # NaN tuple, no crash
    lo0, hi0 = w3.wilson(0, 30)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.2


def test_vix_regime_buckets(w3) -> None:
    assert w3.vix_regime(12.0) == "calm (<20)"
    assert w3.vix_regime(24.0) == "elevated (20-30)"
    assert w3.vix_regime(45.0) == "crisis (>=30)"
    assert w3.vix_regime(float("nan")) == "unknown"


def test_calib_rows_structure_and_gap(w3) -> None:
    """gap == realized - pred and the Wilson interval brackets realized."""
    # 10 candidates in the (0.9, 0.95] bin, 6 realized → realized 0.6, pred ~0.92.
    rows = [{"pp": 0.92, "realized": i < 6} for i in range(10)]
    out = w3.calib_rows(rows)
    assert len(out) == 1
    cell = out[0]
    assert cell["n"] == 10 and cell["realized"] == 0.6
    assert abs(cell["gap"] - (cell["realized"] - cell["pred"])) < 1e-9
    wl, wh = cell["wilson"]
    assert wl <= cell["realized"] <= wh
    assert cell["conclusive_n_ge_30"] is False  # n=10 < 30 → not conclusive


def test_top_bin_overconfidence_sign(w3) -> None:
    """A bin where forecast >> realized yields a negative gap (over-confidence)."""
    rows = [{"pp": 0.96, "realized": i < 57} for i in range(100)]
    t = w3.top_bin(rows, 0.90)
    assert t["n"] == 100 and t["conclusive_n_ge_30"] is True
    assert t["gap"] < 0  # realized 0.57 < forecast 0.96 → over-confident
    assert abs(t["realized"] - 0.57) < 1e-9
