---
id: docs-structure
title: docs/ top-level structure run — classify 108, archive 20, zero deletions
kind: docs
status: complete
terminal:
pr:
decisions: []
date: 2026-07-28
headline: All 108 top-level docs/*.md classified with per-file evidence (LIVE 36 / EVIDENCE 48+ / HISTORICAL / UNCERTAIN-resolved); 20 completed dated reports archived to root archive/{2026-05,2026-06,2026-07}/ via git mv with live-surface references updated same-commit; 3 Theta docs deferred post-#507; zero deletions; worklog-README + REPO_MAP repoints landed.
surface: []
---

## Goal

`docs/` had ~108 top-level files mixing live runbooks, evidence for live
invariants, and completed dated campaign reports — a stateless agent could not
judge currency from names/location. Classify every file with evidence, archive
the completed historical set (never delete), keep every reference resolving,
change no code behaviour.

## What we tried

Two-layer classification: (1) my own repo-wide citation scans (external:
OPERATING_MODEL/README/PROJECT_STATE/DECISIONS/root docs/scripts/tests/hooks/
skills/engine/backtests/snapshots; plus docs-internal incl. worklog/audits) and
line-level PROJECT_STATE/DECISIONS context reads; (2) a 12-agent header
inspection (first 30+ lines, status-marker grep, citation lookup) over all 108
files. Agent proposals cross-checked against my citation map (zero conflicts);
12 rows overridden on my own verification.

## What worked

- Category totals at the gate: LIVE 36, EVIDENCE 48, HISTORICAL 23,
  UNCERTAIN 5 (all resolved with recommendations). Actions approved:
  **ARCHIVE 20, DEFER 3, KEEP 85, MERGE 0, RENAME 0, DELETE 0.**
- Operator rulings: root `archive/<yyyy-mm>/` (not docs/archive/ — the card
  was corrected); path rewrites gated on CITING-file liveness (live → update,
  historical record → retain old path, enumerated); worklog-README exception
  covers both refs; REPO_MAP gains the OPERATING_MODEL route + the stale
  "CLAUDE.md §2" rule-text route fixed to OPERATING_MODEL.md §9.2/§7.
- Batches: 8 → archive/2026-05/ (refs: REPO_MAP:7, CHANGELOG:299);
  11 → archive/2026-06/ (refs: PROJECT_STATE:88, DATA_POLICY:197,
  NEXT_DATA_SESSION_RUNBOOK ×2, SUPERVISED_BLOCK_WORKLIST ×2,
  WIRING_CAMPAIGN:104, DATA_LAYER_DEEP_READ_DESIGN ×2,
  CODEBASE_AUDIT_2026-07-15_PROPOSALS:169); 1 → archive/2026-07/ (no live
  citers). FILE_MANIFEST rows relocated to the archive section same-commit;
  archive/README.md index rows added per file (original path + reason).

## What didn't

- The classify workflow initially crashed (args delivered as a JSON string →
  `args.files` undefined); fixed with a string-parse guard and resumed.
- The batch-2 manifest relocation asserted on a pre-existing DUPLICATE
  manifest row for `DATA_FIX_2026-06-28_OHLCV_SPLIT_SCALE_439.md` (two rows,
  different descriptions — a #455→#472-era leftover), and a `tail`-masked
  exit code let the commit land red; caught immediately, fixed (both rows
  relocated, duplication retained + reported), amended before push.

## How we fixed it

See batches above. Deferred (second-stop avoided by planning):
`THETA_ENRICH_RUNBOOK_2026-06-17`, `THETA_ENTITLEMENT_RETEST_2026-06-17`,
`THETA_PULL_AUDIT_2026-06-15` — archive-worthy but cited from
`docs/DATA_INVENTORY.md`, which draft PR #507 owns; queue post-#507.

## Evidence

- Classification evidence per file lives in the run's gate table (Run
  Summary); every LIVE/EVIDENCE claim rests on a named citing file:line.
- check_manifest_coverage 0 uncovered / 0 orphans after every batch;
  engine/scripts/tests/backtests untouched (`git diff origin/main --stat`
  empty on those paths).

## Unresolved / handoff

- REPORT-ONLY (code out of scope): `backtests/regression/s34_universe_100t_1m.py:4,:23`
  and committed snapshot `s34_universe_100t_1m.json:3` still cite
  `docs/SOUNDNESS_REVIEW_2026-05-26.md`, which moved to `archive/2026-05/` in a
  PRIOR wave — stale-path precedent; needs a follow-up code run (snapshot is
  immutable evidence; the .py docstring/DOC constant is the fixable part).
- Pre-existing duplicate FILE_MANIFEST rows for the archived
  `DATA_FIX_2026-06-28_OHLCV_SPLIT_SCALE_439.md` (both relocated, kept) —
  dedup is an operator/manifest-owner call.
- Historical records deliberately retain old `docs/` paths (Operator ruling 2);
  enumerated in the Run Summary grep output.
- Post-#507 follow-up: archive the three deferred Theta docs.
- archive/README.md states subdirs are named by ARCHIVE date; this wave filed
  by artifact vintage per Operator ruling — convention line may want amending.
- Proposed DECISIONS entry (not written, per ruling 5): record the
  dated-docs-archive practice (departure from D28's in-place-banner practice).
