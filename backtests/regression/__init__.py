"""Regression backtests for the Smart Wheel Engine.

Four reproducers (S27, S32, S34, S35) pin the headline numbers from the
committed backtest docs against the current engine. Each module exposes
``CANONICAL`` (the canonical run kwargs) and ``run(**overrides)`` that
returns a metrics dict matching the snapshot schema in
``backtests/regression/snapshots/``.

Snapshots, pytest harness, and the on-fail re-baseline workflow live
under PR2 — see ``docs/LAUNCH_READINESS.md`` §6 and
``.claude/commands/backtest-regression.md`` once they land.
"""
