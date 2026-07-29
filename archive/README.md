# Archive

Point-in-time and superseded artifacts, retained for history but **no
longer part of the live documentation set**. Treat everything here as a
historical snapshot — these files are not maintained and should not be
read as current reference. Subdirectories are named by archive date
(`YYYY-MM`), not by the artifact's own vintage.

## 2026-05 — D14 repository restructure

| Archived file | Original path | Reason |
|---|---|---|
| `2026-05/OptionsEngine.txt` | repo root | A narrative end-to-end usage walkthrough. It is a point-in-time document, not maintained; a paired accuracy audit of it is preserved at `archive/2026-05/optionsengine_audit_2026-05-17.md`. Archived rather than kept at the orientation root. |
| `2026-05/ARCHITECTURE.md` | `docs/` | Describes a planned `src/`-based module layout (`src/data`, `src/models`, `src/execution`, …) that does not match the actual repository tree. Superseded as the architecture reference by `MODULE_INDEX.md`. |
| `2026-05/DATA_COLLECTION_REPORT.md` | `docs/` | A dated data-collection phase report (2026-03-22) describing the collection effort as in progress. The phase is complete; the report is no longer current. |
| `2026-05/bloomberg_excel_extractor.bas` | `scripts/` | V1 of the Bloomberg Excel VBA extractor. V2 explicitly self-describes as a "FIXED VERSION with longer wait times"; `docs/bloomberg_refresh_runbook.md` is the live path. |
| `2026-05/download_ohlcv.py` | `scripts/` | An early yfinance OHLCV downloader. Superseded by `scripts/download_yf_ohlcv.py`, which adds multi-index header cleanup. |

See `DECISIONS.md` D14 for the restructure rationale.

## 2026-05 — verification-doc consolidation (post-2026-05 campaign)

Twelve point-in-time review / verification snapshots from the 2026-05
deployment-readiness campaign, moved here because the live verification
state now lives in a single canonical index: `docs/VERIFICATION_INDEX_2026-05-28.md`.
Each headline finding is carried forward into the index's Tested-surfaces
table; the originals are preserved here for the per-PR detail. None of
these are maintained; treat them as historical snapshots tied to the
engine SHA they were captured against.

| Archived file | Original path | Reason |
|---|---|---|
| `2026-05/END_TO_END_REVIEW_2026_05_25.md` | `docs/` | Four-pass end-to-end product review against `origin/main` @ `e83eaca` (pre-#260 / pre-#262). Tally and follow-ups (R1 +inf bypass, ranker→tracker auto-wire, F4, dossier/webhook divergence) have either shipped or are tracked in `docs/PRODUCTION_READINESS.md`. |
| `2026-05/LAUNCH_READINESS_ANALYSIS_2026-05-26.md` | `docs/` | 2026-05-26 launch-readiness analysis pulling S22/S27/S32/S34/S35 + four review PRs into one verdict. Pre-#260 / pre-#262 engine; the live deployment gate is `docs/PRODUCTION_READINESS.md`. |
| `2026-05/SOUNDNESS_REVIEW_2026-05-26.md` | `docs/` | Second-pass critical re-verification (PR #229). Equity-beta-dominance and BKNG-concentration findings now folded into `docs/PRODUCTION_READINESS.md` §1 headline. |
| `2026-05/PREDICTIVE_VALIDITY_REVIEW.md` | `docs/` | PR #197 meta-verification of S22 + S27 (P1–P9). ρ ≈ 0.22 floor carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `2026-05/RELIABILITY_ARC_REVIEW.md` | `docs/` | PR #194 independent verification of the reliability arc (S18 / S19 / S20). "PASS-with-caveat" carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `2026-05/AUDIT_OF_AUDIT_REVIEW.md` | `docs/` | PR #195 meta-verification of `archive/2026-05/TERMINAL_A_AUDIT.md`. "22/22 SOLID, 0 §2 breaches missed" carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `2026-05/ENGINE_SUBSYSTEM_AUDIT.md` | `docs/` | Structural read-through audit of 46 `engine/` + 10 `advisors/` files. "No new bugs" finding carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `2026-05/TERMINAL_A_AUDIT.md` | `docs/` | Independent engineering audit of Terminal A's 22-PR coordinated run on board #113. Per-PR detail; tally carried forward in `docs/VERIFICATION_INDEX_2026-05-28.md`. |
| `2026-05/SESSION_REPORT_2026-05-26.md` | `docs/` | Machine-readable session ledger for the 2026-05-26 deployment-readiness campaign. Superseded by `docs/VERIFICATION_INDEX_2026-05-28.md` as the campaign-level reference. |
| `2026-05/ENGINE_REALISM_VERIFICATION_2026-05-26.md` | `docs/` | 2026-05-26 realism + reliability battery against `origin/main` @ 9f0afaf. Pre-#260 / pre-#262 engine snapshot; superseded on the live surface by `docs/REALISM_VERIFICATION_2026-05-28.md` (post-F4 + R9 + R10). |
| `2026-05/optionsengine_audit_2026-05-17.md` | `docs/` | Accuracy audit of the (also-archived) `OptionsEngine.txt` walkthrough. Point-in-time, paired with `2026-05/OptionsEngine.txt`. |
| `2026-05/data_inventory_2026-05-17.md` | `docs/` | Point-in-time data-inventory analysis report. |

## 2026-06 — D27 repo restructure (docs/ pass)

| Archived file | Original path | Reason |
|---|---|---|
| `2026-06/SESSION_HANDOFF.md` | `docs/` | Point-in-time session handoff (2026-05-18) carrying its own SUPERSEDED banner since 2026-05-22; every claim it makes is now owned by `PROJECT_STATE.md` / `DECISIONS.md` / `docs/TRADINGVIEW_MCP_INTEGRATION.md`. Unique residue (the 13-script `sys.stdout` reassignment anti-pattern list) preserved here. |
| `2026-06/Claude_Prompting_Master_Guide.md` | `docs/` | Generic Claude prompt-engineering reference with zero inbound references; not about this repository. Repo-specific agent conventions live in `CLAUDE.md` / `AGENTS.md`. |
| `2026-06/DATA_SPECIFICATION.md` | `docs/` | Aspirational partitioned-Parquet data-layer design (2026-03-19) that self-declares it does not match on-disk reality. The current authorities are `docs/DATA_POLICY.md` (procedures + tiers) and `docs/DATA_INVENTORY.md` (verified census). |
| `2026-06/pull.bat` | repo root | Windows double-click launcher (stash → pull `origin/main` → restore). Zero code or doc references; the operator confirmed it is not part of the live workflow (agent sessions + the documented Python entry points are). |
| `2026-06/pull_branch.bat` | repo root | Windows double-click launcher for checking out a feature branch (edit-in-file branch name; its default pointed at a long-dead branch). Same non-use rationale as `pull.bat`. |
| `2026-06/fetch_data.bat` | repo root | Windows double-click launcher wrapping `theta_health_check` + `theta_backfill`. The documented Theta workflow (`docs/THETA_INSTRUCTIONS.md`) invokes the Python scripts directly. |

See `DECISIONS.md` D27 for the restructure rationale.

## 2026-07-28 — docs/ top-level structure run (filed by artifact vintage per Operator ruling)

Dated, completed campaign/audit/verification reports moved out of the
`docs/` top level so a stateless agent can judge currency from location.
Classification evidence: the 2026-07-28 docs-structure run (worklog
fragment `docs/worklog/` `docs-structure`; 108-file classification table
in the run record). Nothing deleted; live-surface references were updated
in the same commit as each move.

| Archived file | Original path | Reason |
|---|---|---|
| `2026-05/CODE_REVIEW_2026-05-30.md` | `docs/` | Dated code-review findings ledger; decisions it produced live in `DECISIONS.md` D20+. Sole citation was CHANGELOG narrative. |
| `2026-05/ENGINE_BACKTEST_S32_REBASELINE_POST260.md` | `docs/` | Post-#260 snapshot re-baseline record; superseded as snapshot provenance by the 2026-06-06 re-pin (#338). |
| `2026-05/ENGINE_BACKTEST_S34_REBASELINE_POST260.md` | `docs/` | Same wave as the S32 re-baseline; superseded as snapshot provenance by the 2026-06-06 re-pin (#338). |
| `2026-05/HEAVY_NEWS_CALIBRATION_REVERIFY.md` | `docs/` | HT-C heavy-verify report (2026-05-30), complete; worklog-only citations. |
| `2026-05/HEAVY_PERSONA_WALKTHROUGH.md` | `docs/` | HT-A persona walkthrough report (2026-05-30), complete; findings triaged at merge. |
| `2026-05/HEAVY_PIT_REALISM.md` | `docs/` | HT-B PIT-realism report (2026-05-30), complete; its committed driver stays under `docs/verification_artifacts/`. |
| `2026-05/REPO_EFFICIENCY_AUDIT.md` | `docs/` | Repo-efficiency audit whose executable output became `docs/REPO_MAP.md` (the live router); safe tier executed 2026-05-30. |
| `2026-05/REVERIFICATION_REPORT_2026-05-26.md` | `docs/` | Dated re-verification report; durable content carried into `docs/USAGE_TEST_LEDGER.md` per its own text. |
| `2026-06/ADVERSARIAL_WEAKNESS_REVIEW_2026-06-15.md` | `docs/` | Self-declared point-in-time 9-dimension adversarial review; fixes shipped separately (2026-06/07 remediation waves). |
| `2026-06/DATA_ACQUISITION_PLAN_2026-06-14.md` | `docs/` | Dated Bloomberg-lab pull compilation; live pull-side plan is `docs/DATA_ACQUISITION_ROADMAP.md` + `docs/BLOOMBERG_PULL_LIST.md`. |
| `2026-06/DATA_FIX_2026-06-28_OHLCV_SPLIT_SCALE_439.md` | `docs/` | Completed OHLCV split-scale fix record (#439); splice work closed by the 2026-07 remediation campaign (#472). |
| `2026-06/DATA_LAYER_ACTIVATION_ROADMAP.md` | `docs/` | Data-layer activation plan; landed as PRs #332-#337. Design contract survives in `docs/DATA_LAYER_DEEP_READ_DESIGN.md`. |
| `2026-06/ENGINE_TOP20_VALIDATION_2026-06-17.md` | `docs/` | Completed top-20 validation report ("all claims SURVIVED"); zero inbound references. |
| `2026-06/ENGINE_TRADER_STRESS_TEST_2026-06-15.md` | `docs/` | Role-played trader stress test; crisis-overconfidence theme carried forward by later cited studies (W3/W6, R11/D23). |
| `2026-06/HEAVY_VERIFY_2026-06-29_R11_REFINEMENT.md` | `docs/` | Verdict-negative R11 refinement study (#442 band-split — outcome: keep R11 unchanged, no static carve-out). |
| `2026-06/HEAVY_VERIFY_FINDINGS_2026-06-09.md` | `docs/` | Dated heavy-verify findings register; fixes shipped in #405-#410 wave. |
| `2026-06/THETA_PULL_DATA_LOG.md` | `docs/` | Theta pull log, complete 2026-06-17; canonical inventory is `docs/DATA_INVENTORY.md`. |
| `2026-06/VNV_CAMPAIGN_2026-06-01.md` | `docs/` | Read-only V&V sweep report (2026-06-01), complete; worklog-only citations. |
| `2026-06/bloomberg_refresh_runbook.md` | `docs/` | Pre-salvage Bloomberg refresh runbook (2026-06-08 batch decision); salvage completed 2026-07-04 (#477), data current census in `docs/DATA_POLICY.md` §5. |
| `2026-07/DATA_SUFFICIENCY_REVIEW_2026-07-21.md` | `docs/` | Dated pull-vs-wire-vs-blocked data-sufficiency review (2026-07-21), complete; worklog-only citations. |
