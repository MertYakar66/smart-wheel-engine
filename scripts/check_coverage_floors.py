#!/usr/bin/env python3
"""Per-file coverage floors — a ratchet against decay (adversarial review
2026-07-01: "no per-file coverage floor — wheel_runner 75.3% under
aggregate-80 green").

The CI Test Suite job's aggregate ``--cov-fail-under=80`` spans ~15,320
statements, so a single decision-layer file can shed 20pp of coverage
without tripping it. This script reads the ``coverage.json`` that the CI
job already writes (``coverage json``, branch-inclusive) and exits 1 when
any floored file sits below its floor.

Floors = measured-on-main minus 2pp — provably green on day one, red only
on decay. Baseline: CI run 28638294523 @ ``4c5a1a4`` (2026-07-03;
py3.11 and py3.12 artifacts identical to 4 decimal places on all 83
files). Recalibrate DELIBERATELY (bump floors in the same PR that raises
or legitimately shifts coverage — e.g. a change to the Test Suite's
``-m`` selection or the omit list); never loosen silently.

A file missing from coverage.json FAILS LOUD (a rename/split must update
``FLOORS`` in the same PR) — the xfail-false-green lesson: a gate that
silently skips its subject is worse than no gate.

Mechanism origin: the rescued 2026-06-15 design (``c1fdbd4``) used three
bare ``coverage report --fail-under`` lines; this replaces it because
(a) pyproject's ``fail_under = 80`` is inherited by any bare
``coverage report`` call that forgets the flag, silently mis-gating a
single file at 80; (b) the json path compares full-precision floats;
(c) one greppable floors dict; (d) missing-file fails loud. Its stale
thresholds (wheel_runner 54 vs today's 79.35) are discarded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Measured on main @ 4c5a1a4 (CI run 28638294523, coverage.py 7.15.0,
# branch=true, Test Suite selection -m "not backtest_regression").
FLOORS: dict[str, float] = {
    "engine/ev_engine.py": 93.0,  # measured 95.53
    "engine/wheel_runner.py": 77.0,  # measured 79.35
    "engine/candidate_dossier.py": 89.0,  # measured 91.21
    "engine/data_connector.py": 88.0,  # measured 90.69
    "engine/event_gate.py": 95.0,  # measured 97.33
    "engine/wheel_tracker.py": 81.0,  # measured 83.43
    "engine/portfolio_risk_gates.py": 96.0,  # measured 98.16
}


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("coverage.json")
    if not path.is_file():
        print(f"FAIL: {path} not found - run `coverage json` first", file=sys.stderr)
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    files = {k.replace("\\", "/"): v for k, v in data["files"].items()}
    failures: list[str] = []
    for rel, floor in sorted(FLOORS.items()):
        entry = files.get(rel)
        if entry is None:
            failures.append(f"{rel}: missing from coverage.json (renamed/split? update FLOORS)")
            continue
        pct = entry["summary"]["percent_covered"]
        print(f"{'ok  ' if pct >= floor else 'FAIL'} {rel:40s} {pct:6.2f}% (floor {floor:.0f}%)")
        if pct < floor:
            failures.append(f"{rel}: {pct:.2f}% < floor {floor:.0f}%")
    print(
        f"     aggregate {data['totals']['percent_covered']:.2f}% (gated at 80 by --cov-fail-under)"
    )
    if failures:
        print("\nPer-file coverage floor violations:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
