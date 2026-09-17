# Prompting standard — how a request becomes an Execution Prompt

**Status:** the project's prompting standard, referenced by `OPERATING_MODEL.md`
§4.3. One page of rules, one template, one question bank, one short form for the
Run Summary. Applies identically to every Strategist implementation (Claude Code,
ChatGPT Codex) and every Executor.

---

## 1. The rule in one paragraph

Every run starts from a **sharpened request**, not from the Operator's first
sentence. The Strategist turns the request into a precise, bounded, verifiable
job: what "done" means, what is out of scope, which invariants apply, where the
Executor must stop, and which commands prove the work. Precision is the goal,
not ambition: a sharpened request is never *larger* than what the Operator asked
for, only clearer. The Execution Prompt is self-contained because the Executor
may be a fresh session with no memory of the discussion.

## 2. Sharpening (Operator → Strategist), the gate

When the request has depth behind it, ask **3 to 5 short questions** before
writing anything. Ask about intent and priority, never about implementation the
Strategist should determine itself. Skip the questions when the task is small,
obvious, or fully specified, and say so in the block below ("not warranted,
because ..."). The answers are recorded; they travel with the work.

Question bank (pick what fits, never all of them):

- What is the underlying goal this request serves?
- What would count as success for this one run?
- What has already been tried, and what happened?
- What must not break or change?
- What is the acceptable scope of one run: one file, one module, one campaign?
- Is this a question (read-only) or a change?
- Who verifies, and what evidence will you accept?

## 3. The Execution Prompt template

Copy this shape. Keep every section; brevity comes from short sections, never
from missing ones. Tags in angle brackets are the fixed section names.

```
<run-mode>       read-only | change
<request-as-sharpened>
  Goal:            one sentence, the Operator's intent in plain words
  Success:         what the Operator will accept as done
  Tried before:    what exists already / what failed (or "nothing")
  Must not break:  the invariants and behaviours that stay fixed
  Scope of run:    one coherent piece of work; name it
  Operator answers: the sharpening answers, or "not warranted, because ..."
<objective>      one paragraph: what this run accomplishes and what "done" means
<context>        what a cold session needs: files, decisions, prior runs, data state
<scope>          explicit list of what is in scope
<out-of-scope>   explicit list of what is NOT in scope (mandatory)
<owns>           the files this run may edit (disjoint from any other open run)
<reads>          files it may read but not edit
<invariants>     the relevant OPERATING_MODEL.md §8 items, restated
<gates>          exact points where the Executor stops for the Operator
<verification>   commands to run and the output that counts as a pass
<push>           branch name; push before every gate and before the summary
<summary>        "Write the Run Summary per OPERATING_MODEL.md §4.4 as a PR comment."
<constraints>    what the Executor must not do under any circumstances
```

Rules the template encodes: objective before detail; scope and out-of-scope
separately; constraints restated rather than assumed; a gate before every
irreversible step (merge, history rewrite, deletion, decision-layer edit, CI or
environment change, more than about eight files); verification stated as
commands plus expected output, never as intent.

## 4. Run modes

- **read-only**: "This is a question. Report the answer. Change no files. If you
  notice a problem while looking, describe it and stop." No branch, no commit.
- **change**: modifies the repository under the scope and gates above. Work on a
  branch; push before every handoff; open one PR per run.

## 5. The Run Summary, short form (Executor → Operator)

Posted as a comment on the PR (so it exists on GitHub), in this order:

1. **Header** — date, branch, base commit, HEAD, pushed commit id, "went badly?"
2. **Objective as I understood it** — one paragraph; echo the sharpened request
3. **What I did** — one entry per change: role, why, confidence with reason
4. **Commands I ran and their actual output** — pasted, decisive portion
5. **What I did NOT do** — every in-scope item not completed, with the reason
6. **Gates hit and how I proceeded** — quote the approval for anything irreversible
7. **Invariant check** — each relevant §8 item: checked / not applicable, with the proof
8. **Risks, current state, next action, open questions** — one short paragraph each

Prohibited: claiming a check you did not run; describing expected instead of
actual output; "should work" / "appears correct"; omitting a failure a later step
worked around; burying an unrequested change; presenting a passing suite as
proof of correctness; asserting a runtime result without its output.

## 6. Parity across Strategist implementations

The same Execution Prompt must work for Claude Code and for ChatGPT Codex. Do not
rely on tool names, slash commands, or harness features one of them lacks; state
commands as shell commands and files as repository paths. A prompt that only one
implementation can run is not to standard.

## 7. Worked example (abridged)

```
<run-mode> change
<request-as-sharpened>
  Goal: the Operator wants the ranker to offer 7/14/21/28-day puts instead of 35.
  Success: rank_candidates_by_ev accepts a DTE menu and returns one row per (ticker, dte).
  Tried before: dte_target is already a parameter (default 35); select_book iterates a dtes tuple.
  Must not break: EVEngine.evaluate authority; downgrade-only reviewers; the event gate stays on by default.
  Scope of run: the menu plumbing only; no change to the forward distribution or the event policy.
  Operator answers: menu = (7, 14, 21, 28); events stay blocked in this run.
<objective> Add a dte_menu parameter ... done when the new tests pass and the four regression snapshots are unchanged with the default menu.
<scope> engine/wheel_runner.py (rank_candidates_by_ev signature + loop), tests/test_dte_menu.py
<out-of-scope> forward_distribution.py, event_gate.py, candidate_dossier.py, snapshots, docs beyond the worklog
<owns> engine/wheel_runner.py, tests/test_dte_menu.py, docs/worklog/dte-menu.md
<invariants> Decision integrity; Engineering (import smoke, launch blockers); Honesty.
<gates> before editing engine/wheel_runner.py (trio, lane claim); before opening the PR.
<verification> python -m pytest tests/test_dte_menu.py tests/test_audit_invariants.py -q → all pass; python -c "import engine.wheel_runner" → no error.
<push> claude/dte-menu
<summary> Run Summary per OPERATING_MODEL.md §4.4 as a PR comment.
<constraints> No edit outside <owns>; no snapshot regeneration; no merge.
```
