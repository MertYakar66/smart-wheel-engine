---
id: operating-model-consolidation
title: Operating Model consolidation — one governance doc, CLAUDE.md becomes loader
kind: docs
status: complete
terminal:
pr:
decisions: []
date: 2026-07-28
headline: Governance consolidated from seven scattered files into OPERATING_MODEL.md v2 (Operator-supplied skeleton + ten directed amendments + every still-valid source rule); CLAUDE.md reduced to the auto-loaded two-line pointer; six source files deleted and every live inbound reference repointed; 227 source rules enumerated by a 9-agent adversarial audit, 10 partial carries restored before deletion.
surface: []
---

## Goal

Executors twice failed to find the Run Summary format by grepping AGENTS.md,
docs/PARALLEL_SESSIONS.md and docs/MAJOR_SESSION_PROMPT.md, and improvised —
the retrieval failure this run fixes. Consolidate every governance/operating
document into ONE authoritative file (`OPERATING_MODEL.md`), apply the
Operator's ten amendments, keep `CLAUDE.md` only as the auto-loaded two-line
loader (it is the sole file Claude Code injects automatically — deleting it
would strand fresh agents), delete the six superseded sources, and repoint
every inbound reference.

## What we tried

Single assembly pass, verification-heavy: (1) save the Operator-pasted v1
skeleton verbatim; (2) apply amendments A1–A10 exactly as directed; (3) merge
the still-valid rules of seven sources (pre-consolidation CLAUDE.md, AGENTS.md,
COMMIT_GUIDE.md, docs/PARALLEL_SESSIONS.md, docs/GOVERNANCE.md,
docs/MAJOR_SESSION_PROMPT.md, docs/CONTRIBUTING.md) by meaning into §2.4 and
§9.1–§9.11; (4) truncate CLAUDE.md; then HARD STOP with a rule coverage map
before any deletion.

## What worked

- A 9-agent adversarial workflow (7 per-source rule-loss hunters + a
  v1-verbatim differ + a ten-amendment checker) enumerated **227 distinct
  rules** and confirmed the amendments were applied faithfully; a mechanical
  `diff` of §§1–8 against the v1 paste showed divergence only at the directed
  amendments.
- The audit caught **10 partial/lost carries** (board #113 identity,
  claim-comment condition, manifest-owner rule strength, the < 15 min and
  [0.01, 5.0] framework bounds, "track transformation steps", the missing-data
  defaults, the `docs/MODEL_CARDS.md` checkpoint, the risk-hat monitoring
  duty, the venv recommendation, the blanket add-tests rule) and **1 real
  amendment defect** ("§5 Three rules" above four bullets — the repo's known
  rule-count-drift failure mode). All restored/fixed before the gate.

## What didn't

- The Operating Model v1 text did not arrive with the original task card
  (context promised it "immediately after this prompt"); the run hard-stopped
  and requested it rather than reconstructing policy — the card forbids
  authoring.
- The verbatim checker flagged the A8 parenthetical "(two executors have
  already failed to locate a bare section reference)" as fabricated — a false
  alarm caused by an abbreviated transcription in the checker's spec file; the
  sentence is verbatim from the card's A8.

## How we fixed it

- `OPERATING_MODEL.md` v2 created; ten amendments applied; §9 carries the
  consolidated project reference. Operator gate rulings applied: C1 — no model
  name/version in commits/PRs/comments; C2 — branch standard `claude/<slug>`
  (topical prefixes permitted for non-agent work); P1 — Leg 2 retitled "(a
  strategy session)" + concurrency sentence; P2 — Run Summary routed to the
  issuing lane's Strategist; P3 + allocator ruling — the allocator is a hat
  one Strategist wears, one holder at a time, named explicitly by the
  Operator, and decides cross-lane disagreements.
- CLAUDE.md replaced with the exact two-line loader text from the card.
- Six sources deleted (post-approval). Live inbound references repointed:
  `.claude/hooks/session_start.sh` + `.codex/hooks/session_start.sh` (banner →
  OPERATING_MODEL.md §9.5), `.github/workflows/ci.yml` decision-layer-claim
  comment, `.github/pull_request_template.md`, `scripts/check_lane_claim.py`
  (docstring + error message), `tests/test_check_lane_claim.py`,
  `scripts/setup-terminal.{sh,ps1}`, `scripts/check_manifest_coverage.py`,
  both `.agents/skills/source-command-*` skills, `docs/LAUNCH_READINESS.md`,
  `docs/FRESH_LAB_BOX_SETUP.md`.
- FILE_MANIFEST: `OPERATING_MODEL.md` row added, `CLAUDE.md` row rewritten,
  six deleted-file rows removed.
- Disposition record: the 2026-07 "RETIRED" banners on the two
  parallel-session docs were NOT carried — the Operator confirmed the
  multi-terminal protocol is live again (C3 ruling at the gate).

## Evidence

- `python scripts/check_manifest_coverage.py` → 0 uncovered / 0 orphans.
- `python scripts/gen_worklog_index.py --check` → OK; `check_doc_currency.py`
  → OK; `python -c "import engine.wheel_runner"` → OK;
  `git diff origin/main -- engine/` → empty; ruff check + format --check on
  the three touched .py files → clean.
- Post-repoint grep of the six deleted names over live surfaces (hooks, CI,
  scripts, tests, skills, live docs) → zero matches.
- Verification workflow: 9 agents, 227 rules, coverage 217/227 pre-fix →
  restorations committed; amendment checker verdict CLEAN.

## Unresolved / handoff

- References to the deleted files remain in **out-of-scope historical
  records** (docs/worklog fragments incl. `docs/worklog/README.md` lines
  15/35, dated campaign/audit/backtest reports under docs/, `archive/`,
  CHANGELOG/DECISIONS/PROJECT_STATE/ROADMAP) and in **README.md**, which the
  card listed do-not-touch — README still routes newcomers to the deleted
  AGENTS.md. A small follow-up run should repoint README.md (and optionally
  docs/worklog/README.md) to OPERATING_MODEL.md.
- `docs/REPO_MAP.md` never referenced the deleted files but should gain an
  OPERATING_MODEL.md route entry in a follow-up.
- PROJECT_STATE.md is 26 days stale (doc-currency WARN, pre-existing).
