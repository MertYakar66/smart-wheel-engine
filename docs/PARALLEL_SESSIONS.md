# Parallel sessions — how this repo is worked

This repo is often developed by **N Claude Code terminals at once**, to clear
the backlog faster. The pattern is the same regardless of N: one executor
terminal per letter, one verifier session per terminal, each terminal in its
own git worktree, all coordinating through GitHub issue **#113**. Read this
file before starting work so you don't collide with another terminal.

The current live count — which letters are spun up *right now* — lives on
board #113's "Live claims" section. This doc is the pattern; the board is the
truth.

## The two roles

| Role | What it does |
|---|---|
| **Terminal X** | Executor. Runs tasks, edits code, opens/merges PRs. |
| **Session X**  | Verifier for Terminal X. Drafts task prompts; verifies X's output against source/data. Does not execute. |

"Terminal" = the executor that touches the repo. "Session" = its verifier —
checks claims, re-runs numbers, drafts prompts; does not run the tasks itself.
There is **one Session per Terminal**, and the pair count scales with how
many terminals are live. The first executor is Terminal A; the second is
Terminal B; a third is Terminal C; and so on. Letters are assigned in board
order and don't need to be reused if a terminal goes idle.

## Lanes (default ownership)

| Lane | Owner | Surface |
|---|---|---|
| Ranker / sizing / tracker / strategy | Terminal A | `wheel_runner.py` ranking, `select_book`, `wheel_tracker.py`, `strangle_timing.py` |
| Gates / news / regime / webhook | Terminal B | usage tests over `ev_engine` gates, `news_sentiment`, `regime_hmm`, `engine_api` webhook |

These are **default ownership** assignments for the two-terminal case the
project ran first; the table is illustrative, not exhaustive. Cross-lane work
is allowed only after claiming it on the board. A third or later terminal
picks up a new lane (or shares one by claim) when it spins up.

## The coordination board

A pinned GitHub issue titled **"[coordination] Parallel-session work board"**
(currently **#113**) is the live work board. **Before `git checkout -b`**, each
terminal: (1) reads the board and open claims; (2) posts a claim — terminal,
task, branch name, files it expects to touch; (3) posts the merge SHA and marks
the claim done when its PR merges.

The board — not merged `main` — is the source of truth for *in-flight* work
(claimed-but-unmerged branches, branch/PR identifiers, and — once merged —
scenario / D-numbers).

### Glanceable live state + cycle allocation

The board's pinned **body** carries a *Live state* table — one row per active
terminal (`terminal · current task · branch · status · updated`) that each
terminal edits in place, so the current state is readable at a glance without
scrolling the comment log. The comment log stays the durable, append-only
audit trail (claims, done-notices, merge SHAs); the body table is just the
scannable summary of it.

At the **start of a cycle** a single *cycle allocator* (a dedicated session,
or the operator) posts one plan: a non-overlapping task per active terminal,
each with a pre-assigned branch and the files it owns. Terminals start from
the plan rather than self-selecting — this is what prevents two terminals
doing the same work (the `select_book` double-build, the S43/S44 duplicate
backtest). The allocator allocates and steps out; it does not execute.

## Env vars per terminal

Each terminal gets its own port, coverage file, and pytest cache so parallel
runs don't stomp each other. The setup is parametrised by terminal letter and
sourced into each shell:

```bash
# bash / Git Bash / WSL
source scripts/setup-terminal.sh a       # or b, c, ...
```

```powershell
# native PowerShell
. .\scripts\setup-terminal.ps1 a         # or b, c, ...
```

That sets:

| Var | Value (for letter `<x>`) | Status |
|---|---|---|
| `SWE_API_PORT` | `8787 + (<x> - 'a')` — A=8787, B=8788, C=8789, … | honoured by `engine_api.py` and `audit.py` (default 8787) |
| `SWE_DATA_PROCESSED_DIR` | shared `data_processed/` | shared by default; switch to per-terminal only on write contention |
| `SWE_MODELS_DIR` | shared `models/` | same |
| `COVERAGE_FILE` | `.coverage.<x>` | real today — coverage.py reads it |
| `PYTEST_CACHE_DIR` | `.pytest_cache_<x>` | real today — pytest reads it |
| `SWE_DATA_PROVIDER` | `bloomberg` | silences the SessionStart warning; matches the CLAUDE.md §1 default |

`SWE_DATA_PROCESSED_DIR` and `SWE_MODELS_DIR` are **conventions** —
the engine does not read them yet. The contract exists so new code that
does respect them lands in a consistent shape; today they're markers,
not bindings. `SWE_API_PORT` is honoured today by `engine_api.py`'s
`_resolve_port()` and by `audit.py`'s client `BASE` — multi-instance
launches just set the env per terminal. `COVERAGE_FILE` and
`PYTEST_CACHE_DIR` are honoured automatically by the test tooling and
are what actually keep parallel `pytest` runs from corrupting each
other.

## Rules

1. **Each terminal works in its own git worktree — including Terminal A.**
   The primary clone (`smart-wheel-engine/`) is reserved for Sessions,
   orchestration, and safety; **no executor lives in it**. Spin up your
   worktree from the primary:

   ```bash
   git worktree add ../swe-terminal-<x> -b claude/<your-branch> origin/main
   ```

   Two terminals sharing a working tree share one git index, one `HEAD`, one
   checked-out branch: one terminal's `git checkout` / `rebase` silently
   moves the other's branch out from under it, and uncommitted edits leak
   across. Separate worktrees share only `.git/objects` — independent index,
   HEAD, branch, working files.
2. **Claim before you start.** If a claim overlaps an open one, coordinate first.
3. **One branch + one PR per task.** Descriptive branch names.
4. **Decision-layer code** — `ev_engine.py`, `wheel_runner.py`,
   `candidate_dossier.py` — is edited by **one terminal at a time**, claimed on
   the board first. Usage tests only *read* it.
5. **`USAGE_TEST_LEDGER.md` is append-only.** Add your own `### Sn` entry; never
   edit another terminal's. On conflict, rebase and keep both.
6. **`git fetch origin && git rebase origin/main` before every push.**
7. **Scenario numbers (`Sn`) and D-numbers (`D<N>`) are assigned at MERGE,
   not at work-start.** While a task is in flight it is identified by its
   **branch name + PR number** only — both are globally unique and need no
   coordination, so there is no shared counter to race on. The
   human-readable `Sn` / `D<N>` is allocated by whoever merges, as
   `max(merged numbers in the canonical file) + 1`, at the moment of merge.
   Because merges are serialised through the operator, this allocation is
   race-free. Never claim a global number at work-start; never hardcode one.

   *Why this replaced the old "take the next free number from the board"
   rule:* the old rule allocated the number at the concurrent moment (work
   start), so two terminals reading the board minutes apart both grabbed the
   same `Sn` — three such collisions happened on 2026-05-28 alone. Moving
   allocation to the serialised merge point removes the race by construction
   instead of asking terminals to share a counter more carefully (a
   convention that demonstrably failed).

## Recurring hazards (learned the hard way)

- **Shared working tree.** Two terminals in one directory: a fix commit landed
  on the wrong branch (`main`) when the tree was checked out from under the
  terminal mid-task, and one terminal's uncommitted files showed up in the
  other's `git status`. Fixed by rule 1 — separate worktrees for every
  terminal, primary reserved for Sessions.
- **Stale `main`.** Rebasing onto a `main` that predates another terminal's
  just-merged PR. `git fetch` immediately before reasoning about `main`.
- **Parallel number collision.** Two terminals picking the same `Sn` or
  `D<N>` (rule 7).
- **Silent supersede.** Two branches implementing the same feature —
  `select_book` was built twice (PR #107 vs #109). Claim the file on the
  board before feature work.

## See also

- `USAGE_TEST_LEDGER.md` — the usage-test record.
- `DECISIONS.md` D15 — why the doc went N-generic and A moved out of the primary.
- `CLAUDE.md` §2 — the EV-authority invariant; holds regardless of which terminal works.
