"""Pins for ``scripts/check_coverage_floors.py`` (the per-file coverage
ratchet — adversarial review 2026-07-01: the aggregate ``--cov-fail-under=80``
lets a single decision-layer file shed 20pp without tripping).

Behavior pinned, not shape (the xfail-false-green lesson): pass path,
below-floor failure naming the file, MISSING-file loud failure (a rename
must update FLOORS in the same PR — silent skip would be a false-green
gate), and the missing-json exit code. Synthetic coverage.json fixtures —
no real coverage run needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_coverage_floors as ccf  # noqa: E402


def _write_cov(tmp_path: Path, per_file: dict[str, float], aggregate: float = 85.0) -> Path:
    payload = {
        "files": {rel: {"summary": {"percent_covered": pct}} for rel, pct in per_file.items()},
        "totals": {"percent_covered": aggregate},
    }
    p = tmp_path / "coverage.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _all_at(pct_offset: float) -> dict[str, float]:
    return {rel: floor + pct_offset for rel, floor in ccf.FLOORS.items()}


def test_all_above_floors_passes(tmp_path):
    p = _write_cov(tmp_path, _all_at(+2.0))
    assert ccf.main(["prog", str(p)]) == 0


def test_exactly_at_floor_passes(tmp_path):
    p = _write_cov(tmp_path, _all_at(0.0))
    assert ccf.main(["prog", str(p)]) == 0


def test_one_below_floor_fails_and_names_it(tmp_path, capsys):
    cov = _all_at(+2.0)
    cov["engine/wheel_runner.py"] = ccf.FLOORS["engine/wheel_runner.py"] - 0.01
    p = _write_cov(tmp_path, cov)
    assert ccf.main(["prog", str(p)]) == 1
    err = capsys.readouterr().err
    assert "engine/wheel_runner.py" in err and "floor" in err


def test_missing_file_fails_loud(tmp_path, capsys):
    cov = _all_at(+2.0)
    del cov["engine/candidate_dossier.py"]
    p = _write_cov(tmp_path, cov)
    assert ccf.main(["prog", str(p)]) == 1
    assert "missing from coverage.json" in capsys.readouterr().err


def test_windows_path_keys_normalized(tmp_path):
    """coverage.json on Windows may key files with backslashes."""
    cov = {rel.replace("/", "\\"): floor + 2.0 for rel, floor in ccf.FLOORS.items()}
    p = _write_cov(tmp_path, cov)
    assert ccf.main(["prog", str(p)]) == 0


def test_missing_json_exits_2(tmp_path):
    assert ccf.main(["prog", str(tmp_path / "nope.json")]) == 2


def test_floors_match_ci_floored_set():
    """The floors dict covers exactly the decision trio + engine core the
    review flagged; extending it requires MEASURING first (other files in
    the tree sit lower and would go red on day one)."""
    assert set(ccf.FLOORS) == {
        "engine/ev_engine.py",
        "engine/wheel_runner.py",
        "engine/candidate_dossier.py",
        "engine/data_connector.py",
        "engine/event_gate.py",
        "engine/wheel_tracker.py",
        "engine/portfolio_risk_gates.py",
    }
