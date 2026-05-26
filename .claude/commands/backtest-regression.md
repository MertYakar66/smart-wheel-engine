---
description: Run the backtest-regression harness — locks the S27/S32/S34/S35 claims against the current engine
---

Run the backtest regression harness, which pins the headline numbers
from the four committed backtest docs (S27, S32, S34, S35) against
the current engine. This is the slow lane — runtime is ~4–5 hours
total, dominated by S34.

Execute exactly:

```bash
pytest tests/test_backtest_regression.py -v -m backtest_regression --tb=long
```

Report pass/fail per-snapshot. If anything fails, surface the failing
metric, the snapshot value, the observed value, and the snapshot doc
path — do not summarize. A failure means either (a) a real engine
regression, or (b) the snapshot is stale and needs explicit
re-baselining (see `TESTING.md` § "Backtest regression — re-baseline
workflow").

For changes to `engine/ev_engine.py`, `engine/wheel_runner.py`,
`engine/forward_distribution.py`, `engine/dealer_positioning.py`, or
`engine/tail_risk.py`, run this harness in addition to the launch
blockers — the four backtests are downstream of all five files.

The test file uses `pytest.skip` for any snapshot whose JSON is
absent from `backtests/regression/snapshots/`. If a backtest reports
"skipped", check whether the snapshot has been generated yet (only
S35 should skip pre-PR4).

This command is a thin wrapper around the documented invocation; it
adds no logic of its own.
