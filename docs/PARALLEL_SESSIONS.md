# Parallel sessions — how this repo is worked

This repo is often developed by **two Claude Code terminals at once**, to
clear the backlog faster. Read this before starting work so you don't
collide with the other terminal.

## The four roles

| Role | What it does |
|---|---|
| **Terminal A** | Executor. Runs tasks, edits code, opens/merges PRs. |
| **Session A**  | Verifier for Terminal A. Drafts task prompts; verifies A's output against source/data. Does not execute. |
| **Terminal B** | Executor. |
| **Session B**  | Verifier for Terminal B. |

"Terminal" = the executor that touches the repo. "Session" = its verifier —
checks claims, re-runs numbers, drafts prompts; does not run the tasks itself.

## Lanes (default ownership)

| Lane | Owner | Surface |
|---|---|---|
| Ranker / sizing / tracker / strategy | Terminal A | `wheel_runner.py` ranking, `select_book`, `wheel_tracker.py`, `strangle_timing.py` |
| Gates / news / regime / webhook | Terminal B | usage tests over `ev_engine` gates, `news_sentiment`, `regime_hmm`, `engine_api` webhook |

Cross-lane work is allowed only after claiming it on the board.

## The coordination board

A pinned GitHub issue titled **"[coordination] Parallel-session work board"**
(currently **#113**) is the live work board. **Before `git checkout -b`**, each
terminal: (1) reads the board and open claims; (2) posts a claim — terminal,
task, branch name, files it expects to touch; (3) posts the merge SHA and marks
the claim done when its PR merges.

The board — not merged `main` — is the source of truth for *in-flight* work
(claimed-but-unmerged branches and scenario numbers).

## Rules

1. **Each terminal works in its own git worktree / clone — never two terminals
   in one directory.** Two terminals sharing a working tree share one git index,
   one `HEAD`, one checked-out branch: one terminal's `git checkout` / `rebase`
   silently moves the other's branch out from under it, and uncommitted edits
   leak across. Terminal A keeps the primary clone; Terminal B runs from a
   separate worktree (`git worktree add ../swe-terminal-b`). They share only
   `.git/objects` — separate index, HEAD, branch, working files.
2. **Claim before you start.** If a claim overlaps an open one, coordinate first.
3. **One branch + one PR per task.** Descriptive branch names.
4. **Decision-layer code** — `ev_engine.py`, `wheel_runner.py`,
   `candidate_dossier.py` — is edited by **one terminal at a time**, claimed on
   the board first. Usage tests only *read* it.
5. **`USAGE_TEST_LEDGER.md` is append-only.** Add your own `### Sn` entry; never
   edit another terminal's. On conflict, rebase and keep both.
6. **`git fetch origin && git rebase origin/main` before every push.**
7. **Scenario numbers (`Sn`) are global and consumed in parallel.** Take the
   next free number from the *board's claims*, not the merged ledger — the
   other terminal may hold an unmerged number. Never hardcode `Sn`; read it at
   runtime.

## Recurring hazards (learned the hard way)

- **Shared working tree.** Two terminals in one directory: a fix commit landed
  on the wrong branch (`main`) when the tree was checked out from under the
  terminal mid-task, and one terminal's uncommitted files showed up in the
  other's `git status`. Fixed by rule 1 — separate worktrees.
- **Stale `main`.** Rebasing onto a `main` that predates the other terminal's
  just-merged PR. `git fetch` immediately before reasoning about `main`.
- **Parallel number collision.** Both terminals picking the same `Sn` (rule 7).
- **Silent supersede.** Two branches implementing the same feature — `select_book`
  was built twice (PR #107 vs #109). Claim the file on the board before feature work.

## See also

- `USAGE_TEST_LEDGER.md` — the usage-test record.
- `CLAUDE.md` §2 — the EV-authority invariant; holds regardless of which terminal works.
