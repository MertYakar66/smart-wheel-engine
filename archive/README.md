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
