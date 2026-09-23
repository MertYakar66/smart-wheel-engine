---
id: working-structure-v4-2026-09-23
title: "Working structure v4: four roles, marks from commands, the close (D32)"
kind: refactor
status: completed
terminal: sandbox
pr: 531
decisions: [D32, D29, D31]
date: 2026-09-23
headline: ORCA's working structure adopted and fitted to this repository. Four roles, with Codex as a read-only second opinion. A first line filled from commands, whose pen version now carries the data age from a manifest frontier. A close, CLAUDE.md as the checklist, and AGENTS.md carrying it word for word for Codex. A CI check keeps them true.
surface: [CLAUDE.md, AGENTS.md, OPERATING_MODEL.md, docs/PROMPTING_STANDARD.md, PROJECT_STATE.md, DECISIONS.md, TESTING.md, docs/deadlines.md, scripts/session_open.py, scripts/check_working_structure.py, scripts/data_manifest.py, data/DATA_MANIFEST.json, .github/workflows/ci.yml, .claude/hooks/session_start.sh, .codex/hooks/session_start.sh]
---

## Goal

The Operator pasted the working structure of the ORCA project and asked for it
here ("I want this system to be implemented"). The pen compared it with
Operating Model v3, which was already about 70% of it. It then asked five
questions. The Operator answered on 2026-09-23:

"1. yes 2. no you can still merge if necessary 3. yes keep sir 4. yes add the
data dates (how stale they are) to the report sentence 5. change it right away
and tell other agents about the change"

Success means:
- a fresh Claude or Codex session opens with a correct mark without being told;
- the terminal refuses a prompt without line 1 (the run mode) and line 2 (the
  confirmed request);
- CI fails when the two checklists drift apart, or when `PROJECT_STATE.md` loses
  its main hash;
- the first structure scenarios pass.

## What we tried

1. **Read ORCA's actual files, not the summary.** The repository
   `MertYakar66/Orca-Project` was attached to the session read-only. Its
   `CLAUDE.md` checklist, `AGENTS.md` outline, `docs/deadlines.md`,
   `tests/check-docs.mjs` and TESTING §10 scenario table were read as data.
   Nothing there changed.
2. **Mapped each ORCA piece against v3.**
   - Kept from v3: the §7 invariants, the evidence tiers, the lane claim, the
     worklogs, the file manifest, the Python guards.
   - Added from ORCA: the marks, the words, the close, the checklist files, the
     refusal rule, the twelve headings, and the scenarios.
   - Skipped: ORCA's Node tooling, its separate changelog, its website rules and
     its folder counts.
3. **One script prints the marks instead of a bash block.** The desktop runs
   PowerShell or Git Bash, where ORCA's `grep`/`${VAR:?}` block does not run as
   written. ORCA's own scenario 4 also failed when marks were filled by eye.
   `scripts/session_open.py` computes every slot.
4. **The data age had to be readable without the data.** The pen works in a
   cloud sandbox that holds no data (D31). So `scripts/data_manifest.py build`
   now records a `frontier` (the last date of the price and IV files), and the
   mark reads it from git. On the desktop, the script also flags a data root
   that runs past the recorded date.

## What worked

- Opening lines produced by `python scripts/session_open.py`, run before this
  commit:

  ```
  Report from the Strategist, Sir — main `f1c0066` (2026-09-23) · docs 0 commits behind · 6 other branches · data 2026-07-02 (83 days old) · nearest deadline: none open
    data frontier: prices 2026-07-02 (83 days), iv 2026-07-02 (83 days) (data/DATA_MANIFEST.json)
  Report from the Executor, Sir — branch `claude/project-restart-ai-agents-kot5jr` · HEAD `5db6847` · pushed `5db6847` · behind/ahead of main 0/1 · run mode change
  Second opinion from Codex, Sir — reviewing the v4 draft · main `f1c0066` (2026-09-23) · verified myself: nothing
  ```

- The frontier dates come from bytes proven equal to the manifest rows. Checked
  against this sandbox's copies before the edit:

  ```
  data/bloomberg/sp500_ohlcv.csv sha256 matches manifest: True | last date: 2026-07-02
  data/bloomberg/sp500_vol_iv_full.csv sha256 matches manifest: True | last date: 2026-07-02
  ```

- `python scripts/check_working_structure.py` gave `working structure: 6 of 6
  checks pass`. Before the files existed, it failed exactly where they were
  missing, and each message named its fix.
- The `AGENTS.md` Appendix is generated from `CLAUDE.md` §1–§6 by script, so it
  is identical from the first commit.

## What didn't

- **The deadlines check first read every table in the file.** The new "Data
  dates" table was flagged as malformed deadline rows. Both scripts now read
  only the table under the `| Due | What | Owner | Status | Source |` header.
- **No real dated deadline exists in the repository.** There is no Flex token
  expiry and no ThetaData renewal date, so the slot reads `none open`. Those two
  rows are there, Undated and Conditional, for the Operator to date.

## How we fixed it

- Deadline parsing is scoped to the Due table (`parse_deadlines`,
  `check_deadlines`), with tests for a second table in the same file.

## Evidence

```
tests/test_session_open.py ........                                      [ 26%]
tests/test_check_working_structure.py .......                            [ 50%]
tests/test_data_manifest.py .............                                [ 93%]
tests/test_testing_md_taxonomy.py ..                                     [100%]
============================== 30 passed in 1.83s ==============================
```

## Unresolved / handoff

1. **Behavioural scenarios 5–8** (TESTING.md) are "not yet run". They are the
   refusal, Codex blind, drift-at-open and question-without-writes cases. Each
   runs the first time the situation arises; record the date and result.
2. **The sessions running when this landed** (the desktop round 3 and the
   MacBook) were told by a note from the pen (D32 ruling 8). New sessions pick
   the change up from `CLAUDE.md` and `AGENTS.md` by themselves.
3. **A D31 gap found on the way:** git still tracks 20 data fragments under
   `staging/` (about 7 MB). They sit outside the data trees that
   `test_git_tracks_no_market_data` covers. This is proposed, not authorized
   (`PROJECT_STATE.md` §0 B).
