---
id: d38-pm-role
title: D38: the PM/RE above the engine; the mark leaves its branches out
kind: feature
status: in-flight
terminal: desktop
pr:
decisions: [D38]
date: 2026-10-15
headline: the pen's mark no longer counts Risk_Management or rm/ branches as other branches; the Appendix sentinel test follows the new CLAUDE.md wording; the pm-request label
surface: [scripts/session_open.py, tests/test_session_open.py, tests/test_check_working_structure.py, FILE_MANIFEST.md, TESTING.md]
---

## Goal
Carry out PM1 Part 5 (i) (`origin/Risk_Management:DECISIONS.md`) for D38, on top
of the pen's rule-book commit `d203cdd`, in one PR into `main`. The Operator
confirmed it on 2026-10-15: "carry it out as one change run through your normal
process, and ask me for the explicit yes your rules need for the §7 change, the
D38 entry and the label"; then "1. yes 2. yes 3. yes 4. they". Three pieces:
- `scripts/session_open.py`: "other branches" leaves out the Portfolio Manager /
  Risk Engineer's branches, which never merge into `main`, as `CLAUDE.md` §2 and
  `OPERATING_MODEL.md` §3.2 now say;
- the `tests/test_check_working_structure.py` sentinel that the new `CLAUDE.md` §1
  wording broke ("Four roles." no longer exists);
- the `pm-request` label on GitHub, which the PM/RE uses for its requests to the pen.

## What we tried
The rule as the prompt specified it: leave out exactly
`refs/heads/Risk_Management`, and every head that starts with `refs/heads/rm/`.
Every other head except `main` still counts. That includes names that only look
alike: `rm`, `rmx`, `rm-x`, `Risk_Management2`, `risk_management`, and
`claude/rm/x`, which has `rm/` inside it but not at the start. The comparison is
exact and case-sensitive, like git's own ref names.

## What worked
- `is_other_branch(head)`, a small pure function, with two module constants
  (`PM_BRANCH`, `PM_PREFIX`). `other_branches()` counts the heads it accepts. The
  module docstring's "other branches" line says the same, and names D38.
- Tests use the file's existing `_git_stub` to fake `git ls-remote`, so there is no
  network:
  - the PM/RE's three sample heads with `main` count 0, and with `claude/x` added
    they count 1;
  - each look-alike, parametrized, counts exactly 1 beside `main` and the PM/RE's
    heads. That fails both when the PM/RE's heads are counted and when the
    look-alike is wrongly left out;
  - the pen's whole mark reads ` · 6 other branches · ` for the six look-alikes
    beside the PM/RE's heads.
- The sentinel replaces "Six roles:" with "Seven roles:" in the Appendix copy only.
  The test still asserts that the untouched copy passes first, so a sentinel that
  no longer exists would fail it. That is how this run found the old one.

## What didn't
Nothing in the code. Two pre-existing points:
- `test_a_mark_missing_from_claude_md_fails` fails on this Windows desktop, on
  `main` too (`9c84a44`). The test writes `CLAUDE.md` with `write_text()` and no
  encoding, so Windows writes the em dash as cp1252 byte `0x97`, and the check's
  UTF-8 read raises `UnicodeDecodeError`. Left as it is, as the prompt says.
- The new sentinel test fails at `d203cdd` alone, by design (the pen's commit
  message says so). It passes after this run's commit.

## How we fixed it
See "What worked". The registries describe the two roles above the engine (Leg 4,
D38) and the branch rule: `FILE_MANIFEST.md`'s rows for `CLAUDE.md`, `AGENTS.md`,
`OPERATING_MODEL.md`, `scripts/session_open.py` and `tests/test_session_open.py`,
and `TESTING.md`'s row for `test_session_open.py`. Governance scenario 3's pass
condition now names the D38 rule, and it was re-run. That was an opportunistic
fix, labelled in the Run Summary.

## Evidence
- Fail-before: with `d203cdd`'s `scripts/session_open.py` in place, the new tests
  give `8 failed, 10 passed`. With this run's script they give `18 passed`.
- `python -m pytest tests/test_session_open.py tests/test_check_working_structure.py -q`:
  `1 failed, 24 passed`. The one failure is `test_a_mark_missing_from_claude_md_fails`,
  the encoding failure above, and `main`'s own files fail it the same way.
- `python scripts/session_open.py` after `git fetch origin`: `2 other branches`
  (before this branch was pushed). `git ls-remote --heads origin` listed
  `Risk_Management`, `claude/card3-code-part1`,
  `claude/project-restart-ai-agents-kot5jr`, `main` and `rm/standing-rule`.
- The full fast lane, ruff and the four registry checks: pasted in the PR's Run
  Summary.

## Unresolved / handoff
- The pen sets `status: merged` and `pr:` here when the PR merges, at the close.
- The Windows encoding failure of `test_a_mark_missing_from_claude_md_fails` is
  still open. Writing with `encoding="utf-8"` in the test would fix it, in its own
  run.
