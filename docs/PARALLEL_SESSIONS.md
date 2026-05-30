# Parallel sessions — how this repo is worked

This repo is developed by **up to 4 Claude Code terminals at once**. Running
four agents against one project only stays fast if collisions are *designed
out* rather than negotiated away comment-by-comment. This doc is the pattern;
**GitHub issue [#113](https://github.com/MertYakar66/smart-wheel-engine/issues/113)
is the live truth** for who is doing what right now.

The model has one organising idea:

> **The cheapest cross-terminal communication is the communication you never
> need.** Give every task a *disjoint* surface and most coordination
> disappears — two terminals that cannot touch the same file cannot collide.
> A single allocator guarantees the disjointness up front; a CI gate catches
> the one drift humans miss (the decision-layer trio).

This is the 2026-05 evolution of `DECISIONS.md` D15 (worktree-per-terminal,
N-generic). D15's "rotating cycle allocator" is now a **persistent Major
Session**, and "claim the decision-layer file on the board" is now a
**CI gate**, not a prose promise.

---

## 1. Roles

| Role | Count | What it does |
|---|---|---|
| **Major Session** | 1 (persistent) | The allocator + reconciler. Decomposes each cycle into **disjoint task cards**, one per terminal; guarantees no two cards own the same file; routes cross-terminal ripple; assigns `Sn`/`D` numbers and merges. Does **not** write task code. |
| **Terminal X** | up to 4 (A–D) | Executor. Pulls *its* card, works in *its* worktree, opens one PR, posts a one-line done-notice. Never self-selects a task. |
| **Session X** | 1 per terminal | Verifier for Terminal X. Drafts/sharpens the card's acceptance check, re-runs the numbers, reviews the PR. Does not execute. |

Topology is a **star**, not a mesh: terminals talk to the Major Session and
the board, not to each other. With disjoint surfaces there is almost nothing
to say between terminals — and the little that remains (§6) goes through one
channel.

---

## 2. The cycle (the whole loop)

1. **Allocate (Major Session).** Post one **cycle plan** to #113: N task cards
   (§3), each owning a *disjoint* set of files. The Major Session checks the
   `owns` sets are pairwise non-overlapping *before* posting — this is where
   "don't pick similar/same tasks" is guaranteed. Terminals do not choose
   tasks; they receive them.
2. **Pull (Terminal X).** Read your card. If anything is ambiguous, ask the
   Major Session — do not improvise into a neighbour's surface.
3. **Branch (Terminal X).** In *your* worktree (§7):
   `git worktree add ../swe-terminal-<x> -b claude/<slug> origin/main`.
4. **Work.** Edit only files in your card's `owns`. You may *read* anything in
   `reads`. If you discover you must edit a file outside `owns`, **stop and
   tell the Major Session** — that is a re-allocation, not a judgment call.
5. **Document (Terminal X).** Write your card's **worklog fragment** — the one
   place the work is recorded so other agents learn from it. Scaffold it with
   `python scripts/new_worklog.py <card-id> --title "..." --kind <kind>`, fill
   the fixed sections (*what we tried / what worked / what didn't / how we fixed
   it / unresolved-handoff*), then `python scripts/gen_worklog_index.py` to
   refresh `docs/worklog/INDEX.md` (CI fails if it is stale). Format spec:
   `docs/worklog/README.md`. See "Where to document" below for the one-home-per-
   record rule.
6. **PR.** One branch, one PR. Body follows `.github/pull_request_template.md`
   and links your worklog fragment. If your card is `decision_layer: true`, the
   body **must** carry a `lane-claim` block (§5) — CI fails otherwise.
7. **Verify (Session X).** Re-run the acceptance check; confirm the diff stays
   inside `owns`; confirm `§2` (CLAUDE.md) holds.
8. **Done-notice (Terminal X).** One compact comment on #113 with the merge
   SHA (§8). Edit your live-state row to `done`.
9. **Reconcile (Major Session) at cycle close.** Assign `Sn`/`D` numbers
   (rule R7), merge the per-task CHANGELOG lines / DECISIONS pointers, close the
   cycle, post the next plan.

Allocation is a **batch** at cycle start, so terminals never block on the
Major Session per task — they run 4-wide with no required mid-cycle sync.

### Where to document (so other agents learn)

One home per kind of record — do **not** scatter the same content across five
files:

| What | Where |
|---|---|
| **What you tried / worked / didn't / how you fixed it** — the learning record | a **worklog fragment** `docs/worklog/<card>.md` (`scripts/new_worklog.py`) — the canonical place |
| Shipping summary for review | the **PR body** (`.github/pull_request_template.md`) — *link* the fragment, don't re-type it |
| A *structural* decision + rejected alternatives | a `DECISIONS.md` D-entry that links the fragment — only when you actually made one |
| One-line "what shipped" | a `CHANGELOG.md` bullet (the Major Session reconciles these at merge) |

The worklog fragment is the durable record; everything else points at it. The
old 490 KB `docs/USAGE_TEST_LEDGER.md` is **frozen** — never reopen it; its
`S1`–`S46` entries are now fragments under `docs/worklog/`.

---

## 3. The task card

Every task is a card. The Major Session posts the whole cycle's cards in one
fenced block on #113 so they are machine-scannable. Schema:

```yaml
- id: C7-A                      # cycle 7, Terminal A. Unique by construction
  terminal: A                   #   (one allocator) — no number race.
  title: Sever verbal news from the EV path
  owns:                         # THE key field. Disjoint across ALL open cards.
    paths:   [engine/news_sentiment.py, tests/test_news_severance.py]
  reads:    [engine/wheel_runner.py]   # may read; must NOT edit
  decision_layer: false         # true iff owns ev_engine/wheel_runner/candidate_dossier
  depends_on: []                # card ids that must merge before this starts
  acceptance: "sentiment_multiplier returns 1.0 for every (sentiment, n) input"
  worklog: docs/worklog/C7-A.md # the task's learning record (docs/worklog/README.md)
```

The Major Session's allocation invariant: **the union of `owns.paths` over all
open cards has no overlap, and at most one open card has
`decision_layer: true`.** Two terminals cannot be handed the same file, so
"similar/same task" is impossible — not discouraged, *impossible*.

---

## 4. Worked example — "give me the news tests" without collisions

A request like *"create the news tests"* is decomposed into four cards with
provably disjoint surfaces:

| Card | `owns.paths` | Cannot collide because… |
|---|---|---|
| C-A — news severance invariant | `engine/news_sentiment.py`, `tests/test_news_severance.py` | only card touching `news_sentiment.py` |
| C-B — EDGAR earnings layer | `engine/external_data/edgar_adapter.py`, `tests/test_external_data_edgar.py` | disjoint path |
| C-C — FRED credit layer | `engine/external_data/fred_adapter.py`, `tests/test_external_data_fred.py` | disjoint path |
| C-D — adversarial / sanitizer news | `news_pipeline/security/`, `tests/test_adversarial_news.py` | disjoint path |

Four terminals, four surfaces, zero overlap. The Major Session spent ~10
minutes allocating instead of refereeing collisions all cycle.

---

## 5. Decision-layer hard gate (the one CI-enforced lane)

The three decision-layer files — `engine/ev_engine.py`,
`engine/wheel_runner.py`, `engine/candidate_dossier.py` — are the highest-
contention surface and the one place concurrent edits silently corrupt each
other (the `select_book` double-build, PR #107 vs #109). They are edited
**one terminal at a time**.

This is enforced, not just documented. `scripts/check_lane_claim.py` runs in
CI on every PR: **if the diff touches a decision-layer file, the PR
description must carry a `lane-claim` block naming that file**, or the build
fails.

```
<!-- lane-claim
files: engine/wheel_runner.py
board: https://github.com/MertYakar66/smart-wheel-engine/issues/113#issuecomment-NNN
-->
```

A decision-layer path counts as claimed if its exact repo-relative path
appears inside the block. The `board:` link points at *your* claim comment so
a reviewer can audit it.

**Scope of the gate (deliberately narrow).** It fires *only* on the trio.
Non-decision-layer lane ownership stays advisory — the Major Session's
allocation + the board are the guarantee there, so routine refactors that
touch many files never fight a gate. The gate also does not *itself* prove no
*other* open PR holds the same file; that mutual exclusion is the Major
Session's allocation invariant (one allocator ⇒ no race). The gate closes the
common failure — editing a decision-layer file without coordinating at all.

Run it locally before pushing:

```bash
python scripts/check_lane_claim.py --base origin/main --claim-file <pr-body.md>
```

---

## 6. Communication — the star and the one cross-terminal signal

Disjoint surfaces remove ~all need for terminals to talk. The residue is a
**ripple**: a change to a *shared interface* whose behaviour another
terminal's in-flight card assumes (e.g. the real S3 case — Terminal A's
`suggest_rolls` EV-metric fix changed values Terminal B was about to read).

Ripples go through one channel, never terminal-to-terminal DMs:

- Post a `ripple:` notice on #113 naming the interface and the affected card.
- The **Major Session** routes it: pauses/re-sequences the affected card, or
  adds a `depends_on` so the dependent card rebases after the source merges.

If you find yourself needing to coordinate a file two cards both touch, the
allocation was wrong — that is a Major-Session fix (re-cut the cards), not a
terminal-to-terminal negotiation.

---

## 7. Worktrees + per-terminal env (mandatory)

**Every terminal works in its own git worktree** — including Terminal A. The
primary clone (`smart-wheel-engine/`) is reserved for the Major Session,
Sessions, and orchestration; **no executor runs in it.**

```bash
git worktree add ../swe-terminal-<x> -b claude/<your-branch> origin/main
```

Two terminals in one working tree share one index, one HEAD, one checked-out
branch — one terminal's `checkout`/`rebase` silently moves the other's branch
and leaks uncommitted edits (the S12 incident). Separate worktrees share only
`.git/objects`.

**Source your per-terminal env in every shell** so parallel test runs do not
stomp each other:

```bash
source scripts/setup-terminal.sh a      # b, c, d, …
```

That sets a per-terminal `SWE_API_PORT` (8787 + offset), `COVERAGE_FILE`
(`.coverage.<x>`), and `PYTEST_CACHE_DIR` (`.pytest_cache_<x>`) — the three
that actually keep concurrent `pytest` / `engine_api.py` runs isolated. The
SessionStart hook warns if you are in a worktree without these set.

---

## 8. The board (#113) — structure + compact templates

The pinned issue body carries a **Live state** table (one row per terminal,
edited in place) for glanceable state; the comment log is the durable
append-only audit trail. Keep comments compact — the long freeform essays of
the old protocol are replaced by these:

**Claim (only needed for a decision-layer card or a ripple; normal cards are
pre-allocated in the cycle plan):**

```
[Terminal C] claim C7-C — FRED credit layer
owns: engine/external_data/fred_adapter.py, tests/test_external_data_fred.py
decision_layer: no   branch: claude/news-fred-credit
```

**Done-notice:**

```
[Terminal C] done C7-C — PR #NNN merged @ <sha>. §2 holds. acceptance: green.
```

**Ripple:**

```
[Terminal A] ripple: suggest_rolls roll_ev metric changes → affects C7-B (reads wheel_tracker)
```

---

## 9. Numbering — `Sn` / `D` assigned at merge

Scenario numbers (`Sn`) and decision numbers (`D<N>`) are allocated **at
merge**, as `max(merged in the canonical file) + 1`, by whoever merges.
In-flight work is keyed by **branch + PR number** only — both globally
unique, no shared counter to race on. Never claim a global number at
work-start; never hardcode one (the S13 double-grab on 2026-05-28 is why).

---

## 10. Shared "magnet" files (interim)

`USAGE_TEST_LEDGER.md`, `FILE_MANIFEST.md`, and the temporal docs
(`CHANGELOG.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `ROADMAP.md`) are touched
by nearly every task and are the main rebase-contention points. Until the
**documentation redesign (Part B)** lands — per-task worklog *fragments* +
generated indexes, which removes these as shared files entirely — the interim
rules hold:

- **One `FILE_MANIFEST.md` owner per cycle** (named by the Major Session).
  Others list new files in their PR body; the owner reconciles, or a
  `python scripts/sync_manifest.py --fix` pass closes the gaps at cycle close.
- The temporal docs are reconciled by the **Major Session at cycle close** —
  individual cards add a CHANGELOG line / DECISIONS pointer in their own PR
  body for the Major Session to merge, rather than four terminals editing
  `CHANGELOG.md` concurrently.
- On any conflict in these files, **rebase and re-apply** — never
  `git checkout --theirs` (it silently reverted a 12-doc archive move on
  2026-05-29).

---

## 11. Rules (the short list)

1. **Worktree per terminal**, including A; the primary clone is reserved.
2. **Don't self-select.** Work the card the Major Session allocated. New
   need? Ask the Major Session — it re-cuts cards; you don't improvise.
3. **Edit only `owns`; read `reads`.** Drifting outside `owns` is a
   re-allocation, not a judgment call.
4. **Decision-layer trio is CI-gated.** A PR touching it needs a `lane-claim`
   block (§5). One terminal at a time, by the allocation invariant.
5. **One branch, one PR per card.** Body per `.github/pull_request_template.md`.
6. **`git fetch origin && git rebase origin/main` before every push.**
7. **`Sn` / `D` numbers assigned at merge** (§9), never at work-start.
8. **Ripples go through the board + Major Session** (§6), never terminal-to-
   terminal.

---

## 12. Recurring hazards (learned the hard way)

- **Shared working tree.** Two terminals in one dir corrupt each other on the
  next checkout/rebase. → Rule 1 (worktrees).
- **Unclaimed decision-layer edit.** Two terminals edit `wheel_runner.py` at
  once; one silently supersedes the other. → §5 (now a CI failure, not a
  hope).
- **Self-selected duplicate work.** `select_book` was built twice (#107 vs
  #109). → §2–§3 (Major Session allocates disjoint cards).
- **Number collision.** Two terminals grab the same `Sn`/`D`. → Rule 7
  (merge-time assignment).
- **Stale `main`.** Rebasing onto a `main` that predates another terminal's
  just-merged PR. → Rule 6 (`fetch` immediately before reasoning about main).
- **Magnet-file `--theirs` revert.** Bulk-resolving `FILE_MANIFEST.md` reverted
  a concurrent restructure. → §10 (rebase, never `--theirs`).

---

## See also

- `DECISIONS.md` D15 — worktree-per-terminal + the N-generic model this
  extends. (The Major-Session / task-card / CI-gate evolution is recorded as
  the 2026-05 extension under D15.)
- `scripts/check_lane_claim.py` — the decision-layer gate; `tests/test_check_lane_claim.py` pins it.
- `.github/pull_request_template.md` — the PR body + `lane-claim` block.
- `scripts/setup-terminal.{sh,ps1}` — per-terminal env.
- `CLAUDE.md` §2 — the EV-authority invariant; holds regardless of which terminal works.
