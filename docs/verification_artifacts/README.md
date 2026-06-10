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
| [`ENGINE_REALISM_VERIFICATION_2026-05-26.md`](../../archive/2026-05/ENGINE_REALISM_VERIFICATION_2026-05-26.md) (archived) | [`realism_verify_driver.py`](realism_verify_driver.py) | [`realism_2026-05-26_raw_output.txt`](realism_2026-05-26_raw_output.txt) | `9f0afaf` (pre-#260) |
| [`ENGINE_REVERIFY_S46_POST_F4_R10.md`](../ENGINE_REVERIFY_S46_POST_F4_R10.md) §2 | (same driver, re-run) | [`realism_2026-05-28_raw_output.txt`](realism_2026-05-28_raw_output.txt) | `56d8e5c` (post-#260+#262) |
| **F4 pre-fix baseline** (companion to [`F4_TAIL_RISK_DIAGNOSTIC.md`](../F4_TAIL_RISK_DIAGNOSTIC.md)) | [`f4_baseline_driver.py`](f4_baseline_driver.py) | [`f4_baseline_2026-05-26_raw_output.txt`](f4_baseline_2026-05-26_raw_output.txt) | `70fdb78` (pre-#260) |
| **F4 post-fix re-run** ([`ENGINE_REVERIFY_S46_POST_F4_R10.md`](../ENGINE_REVERIFY_S46_POST_F4_R10.md) §3) | (same driver, re-run) | [`f4_baseline_2026-05-28_raw_output.txt`](f4_baseline_2026-05-28_raw_output.txt) | `56d8e5c` (post-#260+#262) |
| **S41 — F4 fix validation (post-#260)** (companion to [`ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md`](../ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md)) | [`s41_f4_validation_driver.py`](s41_f4_validation_driver.py) | [`s41_f4_validation_2026-05-28_raw_output.txt`](s41_f4_validation_2026-05-28_raw_output.txt) | `56d8e5c` (post-#260+#262) |
| **V&V campaign — efficiency / realism / reliability** (companion to [`VNV_CAMPAIGN_2026-06-01.md`](../VNV_CAMPAIGN_2026-06-01.md)) | [`vnv_funnel_tier_report.py`](../../scripts/vnv_funnel_tier_report.py), [`vnv_prob_profit_calibration.py`](../../scripts/vnv_prob_profit_calibration.py) | [`vnv_2026-06-01/`](vnv_2026-06-01/) — funnel JSON, calibration JSON, profile txt | `1c69062` |

**The F4 baseline + post-fix pair** is the captured pre/post evidence
for PR #260's RV30/RV252 widening. Diff the two raw outputs to see
the fix's effect: UNH 2024-11-11 widens (ev_dollars −$6.28, cvar_5
−3.2%); COST 2022-04-04 is unchanged (rv30/rv252 below the 1.30
firing threshold); AAPL control byte-identical.

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
