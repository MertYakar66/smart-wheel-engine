# Verification artifacts

Raw artifacts from engine verification runs — kept in the repo so
future agents can:

1. **Observe** the exact outputs that backed a given verification doc
   without having to re-run the engine.
2. **Re-run** the same verification on a future commit by invoking
   the driver, then diff the new output against the captured one.

Each verification doc in `docs/` that has measurable outputs gets a
companion subset here. The doc itself is the human-readable narrative;
the artifact files here are the raw observable data.

## Index

| Doc | Driver | Raw output | Captured on commit |
|---|---|---|---|
| [`ENGINE_REALISM_VERIFICATION_2026-05-26.md`](../ENGINE_REALISM_VERIFICATION_2026-05-26.md) | [`realism_verify_driver.py`](realism_verify_driver.py) | [`realism_2026-05-26_raw_output.txt`](realism_2026-05-26_raw_output.txt) | `9f0afaf` |
| **F4 pre-fix baseline** (companion to [`F4_TAIL_RISK_DIAGNOSTIC.md`](../F4_TAIL_RISK_DIAGNOSTIC.md)) | [`f4_baseline_driver.py`](f4_baseline_driver.py) | [`f4_baseline_2026-05-26_raw_output.txt`](f4_baseline_2026-05-26_raw_output.txt) | `70fdb78` |

**The F4 baseline is the pre-fix snapshot for Terminal A's incoming
`claude/fix-f4-regime-conditioned-widening` branch.** Re-run the driver
against the post-fix engine and diff `prob_profit` / `cvar_5` /
`heavy_tail` against the captured baseline to validate the fix.

## How to re-run

```bash
# From any worktree:
python docs/verification_artifacts/realism_verify_driver.py
```

The driver self-bootstraps `sys.path` to the current worktree
(`C:\Users\merty\Desktop\swe-terminal-b` is hardcoded for this run
because of the [sys-path worktree shadow](../../.claude/projects/...)
note — agents in other worktrees should edit the `WORKTREE` constant
at the top of the driver to match their own path before running).

## Conventions

- **File naming:** `{verification-name}_{YYYY-MM-DD}_raw_output.txt`
  for outputs; `{verification-name}_driver.py` for drivers.
- **Determinism:** drivers should be fully self-contained
  (no network, no time-of-day, no RNG without an explicit seed).
- **No backtest harnesses here.** Backtest drivers live under
  `%TEMP%` per the Sn convention because they emit gigabytes of
  per-candidate ledger rows. Only short-runtime verification
  artifacts go in this directory.
- **No engine modification.** Drivers are read-only clients of the
  production engine.
