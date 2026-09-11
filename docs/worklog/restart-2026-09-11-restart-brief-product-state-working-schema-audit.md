---
id: restart-2026-09-11
title: Restart brief — product state + working-schema audit after the summer break
kind: docs
status: completed
terminal: remote-sandbox
pr:
decisions: []
date: 2026-09-11
headline: Two-part re-onboarding brief (docs/RESTART_BRIEF_2026-09-11.md) from 8 readers + 8 adversarial verifiers + 2 critics over main@ec1c5c5 plus a sandbox runtime check; found the doc-currency CI gate failing (71d/62d), reconciled 27 unrecorded merges into CHANGELOG, added PROJECT_STATE §0; no code/data/trio change.
surface: [docs/RESTART_BRIEF_2026-09-11.md, PROJECT_STATE.md, CHANGELOG.md, FILE_MANIFEST.md]
---

## Goal

The Operator returned after ~6 weeks away (last merge #522 on 2026-07-28, PR #523
left open) with no memory of the product or of the AI-agent operating model, and
asked for a deep, honest re-onboarding: what the product is and where it stands,
and how the "working schema" (OPERATING_MODEL.md as written) compares with how
agents actually worked. Deliver it as one evidence-labelled document that is also
durable repository memory, and repair the memory decay found on the way.

## What we tried

1. Inline scouting per the repo's own read order: OPERATING_MODEL.md in full,
   PROJECT_STATE.md, README, ROADMAP, worklog INDEX, GitHub state (open PRs #523
   and #507, 13 open issues, 6 non-main branches), git history (unshallowed with
   `--filter=blob:none` so `git log` covers 2025-11-22 → 2026-07-28).
2. Two background workflows (the sandbox has 4 CPUs, so each runs 2 agents at a
   time; running two side by side doubled throughput):
   - product: 5 readers (decision+quant, data, interface/operator, validation
     evidence, in-flight queue) → 1 adversarial verifier each (source-check +
     currency lenses, 8 headline claims each) → synthesizer → completeness critic;
   - schema: 3 readers (doc as written + the six deleted sources via
     `git show ec1c5c5^:<path>`; practice on #113/#493/#494/#517/#436;
     worklogs/decisions/automation) → verifier each → synthesizer → critic.
3. Runtime evidence gathered directly: `pip install -e ".[dev]"`, provider
   bring-up, the §9.4 5-ticker EV smoke, the launch-blocker subset,
   `pytest --collect-only`, and the four stdlib guard scripts.
4. Wrote the brief myself from the verified findings (applying every verifier
   correction), built an HTML artifact from the same markdown, committed the
   markdown as the canonical copy.

## What worked

- The reader → adversarial-verifier pattern caught real errors before they
  reached the brief: wrong PR attribution (#473 vs #472 for the off-box tail),
  a "low" column quoted as "close", stale line numbers for the F4 sites, a
  "read-only auditors" framing that omitted their later fix branches, "all ten
  fixes brain-side" (four were terminal-side), 780 vs 35 commits ahead, 8 vs 7
  remote branches, the "#494 fully complete" claim (CMD 2 adapter fix is
  stranded on origin/Dashboard).
- Recomputed counts over the GitHub channels (367 + 104 comments) settled
  questions prose could not: task cards used exactly twice (2026-05-30), zero
  ripple notices, zero Run Summaries anywhere on GitHub, 31 of 60 July merges
  without a worklog fragment, 27 merges missing from CHANGELOG, 64 fragments
  marked in-flight with 60 merged.
- The repo's own guards were the best onboarding signal: `check_doc_currency.py`
  exit 1 told the story of the break in one line.

## What didn't

- The EV smoke as documented ("five rows means healthy") is date-dependent:
  today JPM is correctly dropped by the event gate (earnings 2026-10-13 inside
  the 35-DTE window), so 4 rows + 1 logged event drop is the healthy result.
  Do not chase a "missing" row.
- `python -m pytest` must be used in this sandbox; bare `pytest` resolved before
  the install. The suite's default `addopts` does not deselect the slow
  backtest lane (TESTING.md callout) — the launch-blocker subset was run
  explicitly instead of the full suite.
- The clone arrived shallow (50 commits); early git-based velocity claims were
  wrong until `git fetch --unshallow --filter=blob:none` (cheap: no blobs).
- Workflow result labels are not attached to journal `result` lines; mapping
  results back to readers by completion order mis-labelled two readers until
  checked by content. Verify by summary keywords before trusting the mapping.
- The `Co-Authored-By` model-name trailer this harness mandates conflicts with
  OPERATING_MODEL.md §9.7 (ruling 2026-07-28). This run followed §9.7 (generic
  co-authorship + session link, no model name) and flags the conflict as an
  Operator decision (Part 2, decision 9).

## How we fixed it

- Wrote `docs/RESTART_BRIEF_2026-09-11.md` (≈11.7k words; §0 health check,
  repo at a glance, mental-model corrections, Part 1 product, Part 2 schema
  with drift table + 8 ranked proposals + 15 Operator decisions + 8 sharpening
  questions, what-to-do-first, method appendix). Published the same content
  as an HTML artifact for reading.
- Cleared the failing doc-currency gate by refreshing `PROJECT_STATE.md`
  (`Last updated` + new §0 "Restart 2026-09-11") and adding a `## 2026-09-11`
  CHANGELOG section that reconciles the 27 unrecorded merges (titles from
  `git log`) and lists the two open PRs. Added the FILE_MANIFEST row for the
  brief. No code, data, or decision-layer file touched.

## Evidence

- Provider: `SWE_DATA_PROVIDER='bloomberg' resolved to MarketDataConnector`.
- EV smoke (11.8 s): MSFT 110.13 / XOM 80.14 / AAPL −12.61 / UNH −20.69
  `ev_dollars`; `drops_summary {'total_dropped': 1, 'by_gate': {'event': 1}}`;
  JPM `event_lockout:earnings@2026-10-13 (±5d buffer)`; connector warnings:
  OHLCV frontier 2026-07-02 is 71 days behind; earnings overlay 70 days old.
- `python -m pytest <launch-blocker subset> -q` → `118 passed, 2 warnings in 43.73s`.
- `python -m pytest tests/ --collect-only -q` → `3720 tests collected`.
- `check_manifest_coverage.py` → 1338 tracked / 957 entries / 0 uncovered / 0 orphans.
- `gen_worklog_index.py --check` → OK. `check_lane_claim.py --base origin/main` → OK.
- `check_doc_currency.py` before: `FAIL PROJECT_STATE.md … (71d ago)`,
  `FAIL CHANGELOG.md newest section 2026-07 (62d behind)`, exit 1; after: OK, exit 0.
- GitHub (2026-09-11): main protected=false, empty rulesets; #523 9/9 checks
  success, mergeable clean; #507 draft, Test Suite 3.11/3.12 failure by design.
- Workflow usage: schema run 8 agents / 311 tool uses / ≈1.82M tokens / 50 min;
  product run 12 agents / 623 tool uses / ≈2.6M tokens / 66 min, plus a 15-min critic re-run.

## Unresolved / handoff

- The product workflow's completeness critic failed once on a session usage
  limit (11 of 12 agents done); it was re-run via `resumeFromRunId` (cached
  readers/verifiers/synthesizer, only the critic re-executed) and its 38
  missing / 19 unverified / 17 contradiction / 35 clarity findings were
  applied to Part 1 in the follow-up commit.
- Not done (deliberately, outside a docs run): flipping the 60 stale
  `in-flight` worklog statuses, marking the audit register/worklist as shipped,
  refreshing `docs/PRODUCTION_READINESS.md`, closing evidence-complete issues,
  rebasing the stranded `origin/Dashboard` adapter fix — all queued in the
  brief (Part 1 §1.7, Part 2 P1).
- Operator decisions the brief asks for: Part 1 §1.8 (10) and Part 2 §2.8 (15);
  the eight §2.9 sharpening questions gate the next run.
- Next agent: start from `docs/RESTART_BRIEF_2026-09-11.md` §0 and "What to do
  first"; do not re-derive the evidence tables — they are labelled and cited.
