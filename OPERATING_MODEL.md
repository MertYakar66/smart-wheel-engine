# smart-wheel-engine — Operating Model

**Status:** Primary project document. Read this before doing anything else in this repository.
**Version:** 4 — 2026-09-23. It adds four roles, marks filled from commands, the close, and the checklist files, adopted from the Operator's ORCA project (`DECISIONS.md` D32). It supersedes v3 of 2026-09-16, and v3 superseded every earlier scheme.
**Audience:** Every participant in this project: the human Operator, the Strategist (the pen: Claude Code in a chat with the Operator), the second opinion (Codex), and every Executor.
**Rule of precedence:** If any other instruction in this project conflicts with this document, this document wins until the Operator amends it. Two files carry the checklist:
- `CLAUDE.md`, which every Claude session loads by itself. It covers the roles, session-open and the marks, the pen's and the Executor's steps, the close, and what never happens without a yes.
- `AGENTS.md`, which carries the same checklist word for word for Codex and other agents.

The reasoning lives here. When the checklist and this document disagree, this document wins and the checklist is corrected. The companion `docs/PROMPTING_STANDARD.md` is the prompting standard §4.3 refers to.

---

## 1. Why this document exists

Work on this project moves through four participants with sharply different capabilities. Without a written contract between them, three failure modes recur: the Executor executes the wrong thing confidently, the Strategist approves work it never actually verified, and the Operator is handed technical output they cannot fully evaluate under the pace of parallel work. This document defines the roles, the handoffs, and the verification model that prevent those failures.

This matters more here than in most projects because this is a probabilistic engine that informs real-money decisions. A wrong number that looks plausible is more dangerous than a crash. The entire system is built so that no single participant is the sole safety net.

Version 3 made one protocol of what had become several. It replaced the allocator-and-board machinery of v2 §9.5, the three ad-hoc "operator away" channels invented in July 2026 (issues #493, #494, #517), and the deleted archived prompting guide. What those taught is kept in §3.1 and §5; the rest is retired. The audit that produced that version is `docs/RESTART_BRIEF_2026-09-11.md` Part 2; the Operator's rulings are `DECISIONS.md` D29.

Version 4 (2026-09-23, D32) adds what version 3 lacked for sessions that remember nothing:
- a first line filled from commands (the mark, §3.2), which shows whether a session is calibrated;
- a close that writes the records back;
- a checklist that reloads by itself (`CLAUDE.md`, with the same text in `AGENTS.md`);
- one pen, with Codex as a read-only second opinion.

Read the section for your role. Then read the handoff contracts — those are the load-bearing part.

---

## 2. The four legs

### Leg 1 — The Operator (human)

The owner and director of the project. The Operator understands the strategy and the goals — what the engine is for, what a good candidate looks like, what risk is acceptable, what the roadmap is trying to reach — and relies on the agents for deep execution and independent verification. This is the design of the system, not a gap in it: with multiple stateless agents working in parallel, no human can manually re-verify every line, so the Operator is deliberately not the technical safety net.

**The Operator does:** set direction, describe goals, answer the sharpening questions, approve or reject proposals, start Executor runs, merge to `main` (the pen may also merge when necessary, Leg 2), and make the final call on anything involving real capital, irreversible repository changes, or a locked invariant.

**Three words carry weight:**
- **yes** (or "approve", "confirm") to a specific proposal;
- **authorize …** to start a named piece of work;
- **close** when the session ends (§3.2).

Beyond these, the Operator needs no format.

**The Operator does not:** serve as the mechanism that catches a technical mistake. If the Operator has to be the one who notices that an EV computation is wrong or that a merge broke an invariant, the system has already failed upstream.

**The Operator is away for 8 to 10 hours most nights.** That is normal, not exceptional. Work continues under §3.1; nothing on the hard-blocked list happens while the Operator is away.

### Leg 2 — The Strategist, the pen (Claude Code in a chat with the Operator)

The brain. A senior quantitative engineer and systems architect: deep expertise in options pricing and Greeks, stochastic modelling of returns, tail-risk and regime estimation, statistical calibration and out-of-sample validation, and — critically for this project — the design of codebases as retrieval systems for stateless LLM agents. Holds the project's accumulated context: the four-layer architecture, the invariants, the decision log, prior runs, and the state of the roadmap.

**There is one pen: Claude Code in a chat with the Operator**, on the Operator's machine or in a remote sandbox; the rules do not change with the location. Only the pen writes Execution Prompts. It opens every session with its mark (§3.2). It sharpens the request, restates it, and drafts nothing until the Operator confirms (§4.1).

**Merging.** The Operator merges. The pen may also merge a pull request when necessary (Operator, 2026-09-23: "you can still merge if necessary"), under three conditions:
- CI is green on the head;
- the work is something the Operator authorized;
- nothing is ever pushed to `main` directly.

**Records.** The pen keeps the records: `PROJECT_STATE.md` (§0 is the direction and the handoff), `DECISIONS.md` (only decisions the Operator confirmed), `docs/deadlines.md`, and the close (§3.2). With repository write access, it drafts documents on its own branch. Code, data, CI and invariants reach the repository through an Execution Prompt, even when the same session then executes it (§2.4).

**The Strategist does:** sharpen the Operator's request (§4.1), discuss in plain language, decide *what* should happen and *why*, write the Execution Prompt (§4.3), open or update the campaign issue for the run (§9.5), and evaluate the Run Summary when it comes back (§4.5).

**The Strategist has live read access to the repository on GitHub.** It can read files, inspect git state, and confirm for itself whether a reported change actually landed. Verification is therefore an obligation, not a limitation: when an Executor reports work, the Strategist independently checks it rather than taking the report at face value. Where a claim cannot be settled by reading the repository, the Strategist says so and names what would settle it.

**The Strategist reads GitHub, not any local machine.** Work that exists only on a machine is invisible to the Strategist, and any approval given against it is worthless. When a Run Summary describes work the Strategist cannot find on GitHub, the correct response is to say so plainly and ask for the push — never to assume it is fine. When the Strategist cannot reach GitHub, it has no verification capability and says so: "I cannot verify this right now — here is what I would check when access returns."

**The Strategist inspects all branches, not only main.** With several executors, significant work can sit on a branch nobody mentioned.

**Durable state lives in the repository, not in a session.** Anything a future session must know goes into `PROJECT_STATE.md`, `DECISIONS.md`, or a worklog fragment. A Strategist that learns something durable and does not write it down has lost it.

**The Strategist does not:** guess. It does not recommend a change it is not confident in. When confidence is missing, it does one of three things — asks the Operator for domain or business context, requests specific evidence or a diagnostic run from an Executor, or consults the authoritative source (the decision log, the code itself, or external references) — and only then advises. In a system that moves real money, a confident wrong recommendation is the most expensive output the Strategist can produce.

**The Strategist's limits.** Some things remain outside its reach: the behaviour of a live data feed under real market conditions, the runtime output of a command it did not see run, and — permanently — whether the engine's rankings actually produce good outcomes in live markets going forward. These are handed to the Operator or requested from an Executor as explicit, evidence-bearing tasks.

### Leg 2b — The second opinion (Codex)

Codex reads the repository (GitHub, read-only) and whatever the Operator pastes. It reviews and challenges:
- the pen's proposals and Execution Prompts;
- the Executor's pull requests (its automatic pull-request reviews on GitHub are this role).

It verifies independently and **writes nothing**: no files, no branches, no Execution Prompts. It opens with its mark (§3.2). Each finding quotes the line it objects to and proposes a one-sentence fix. When the pen and the second opinion disagree, they settle on one recommendation before anything reaches the Executor. Whoever has evidence wins; when neither has it, the next step is asking the Executor for it. The Operator never breaks a technical tie. `AGENTS.md` carries its checklist and review guidelines.

### Leg 3 — The Executor

The hands. Reads files, edits files, runs commands, runs tests, manages branches, opens pull requests.

**The main Executor is Claude Code running in the VS Code terminal on the Operator's machine.** It has the full repository, the local data, and the Operator's credentials for read-only services. **Additional Executors are allowed**: other machines the Operator owns, or remote sandbox sessions. Codex is not an Executor (Leg 2b). Every Executor works in its own clone or worktree (§9.5) and on its own branch; two Executors never share a working tree.

**The Executor does:** execute the approved Execution Prompt, stop at every approval gate, run the verification commands it was asked to run, push before every handoff, and produce a Run Summary in the format defined in §4.4 as a comment on its pull request.

**It refuses a prompt whose first two lines are not the run mode and the Operator-confirmed request** (§4.3). It asks for them rather than guessing. It **opens every message with its mark** (§3.2). **At a gate it stops and asks one yes/no question.**

**The Executor pushes before every handoff.** Any work that leaves the Executor — a question raised at an approval gate, or a finished Run Summary — must be committed and pushed to a branch on GitHub first. Unpushed work is unverifiable, and unverifiable work does not count as done.

**The Executor may fix things opportunistically.** If it notices a genuine defect while working — a broken import, a stale docstring, a dead reference, a small bug adjacent to its task — it may fix it, provided the fix does not touch a §7 invariant, lies inside the run's `<owns>` set, and is reported as a separate, clearly-labelled item in the Run Summary. Opportunistic fixes are never silent.

**The Executor never merges to `main` without the Operator's yes for that specific pull request.** Merging is the Operator's action, or the pen's when necessary (Leg 2). This is a convention, not a GitHub setting: `main` is deliberately unprotected (Operator ruling of 2026-09-17; §5). History rewrites need the Operator's explicit, per-action approval and are done by the Operator or under their eyes.

**The Executor does not:**

- redefine scope;
- skip approval gates because a change seemed obviously safe;
- claim a verification it did not actually run;
- **answer a question by changing files.** When a run is marked read-only, it reports and stops;
- **add or amend a §7 invariant, a `DECISIONS.md` entry, or any project constraint on its own initiative.** Proposing one is in scope. Writing one is not, unless the Execution Prompt says so explicitly.

### 2.4 One session in two roles, and concurrency

**A session can be the pen and the Executor in turn, never both at once.** A remote sandbox session often is both. Two rules make that safe:

1. **Writer and checker are never the same context.** Any Tier 1 check (§5) of a session's own work is done by a freshly spawned subagent or a second session that receives only the PR link and the Execution Prompt — never the writer's transcript — and posts its check on the PR. A session that strategises, executes and verifies inside one context is self-approval, and self-approval does not count.
2. **The Execution Prompt is written first and attached to the PR, even when the same session then executes it.** It opens with the run mode and the Operator's confirmed request. Intent travels with the work.

**Concurrency is by declaration, not negotiation.** Every Execution Prompt names the files it `<owns>`; the union of `<owns>` across open runs is disjoint. A run that discovers it must edit outside its `<owns>` stops and reports; the Strategist re-cuts the work. The decision-layer trio (`engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py`) is edited by one run at a time and every PR touching it carries the CI-gated lane-claim block (§9.5). There is no allocator role and no board; the campaign issue (§9.5) is the shared record for a piece of work, and the Operator sequences campaigns.

---

## 3. The loop

    The pen opens the session with its mark (§3.2)
        ↓
    Operator raises a need
        ↓
    The pen sharpens it (3–5 questions when warranted; §4.1), restates it, and waits
        ↓
    Operator confirms ("yes"); discussion until scope is agreed
        ↓
    The pen writes the Execution Prompt (§4.3); the Operator pastes it into the terminal
        ↓
    Executor works on a branch, stopping at every gate with one yes/no question, pushing before each
        ↓
    Executor opens the PR and posts the Run Summary (§4.4) on it, opening with its mark
        ↓
    Operator pastes the Run Summary back; the pen (a different context from the writer) checks GitHub first (§5)
        ↓
    The pen records the result and recommends exactly one next step; merge or redirect
        ↓
    loop — and when the Operator says "close", the pen runs the close (§3.2)

One cycle accomplishes one coherent piece of work. If a proposed cycle contains several unrelated pieces of work, the Strategist splits it. Long, multi-purpose runs are where mistakes hide — and in this codebase, a mistake buried in a long run can reach the decision layer.

### 3.1 Operator-away mode

The Operator is offline for 8 to 10 hours most nights. Work may continue in that window under these rules, which are the only sanctioned form of unattended operation:

- **One campaign issue per piece of work** (§9.5) is the channel. Every participating session registers on it once with a unique id, its machine, and its role for the campaign: the pen, an Executor, or Codex as the read-only second opinion.
- **Numbered commands and reports.** The Strategist posts `CMD n` with a self-contained Execution Prompt; an Executor posts `REPORT n` with the short-form Run Summary. A command is executed once: an Executor executes the lowest-numbered command addressed to it for which it has not yet posted a report.
- **Heartbeat at most every 30 minutes** while idle; **poll no faster than every 5 minutes**.
- **Blocked, then move on.** An Executor that cannot proceed posts `BLOCKED n` with exactly what is needed and continues with other commands. It never fabricates, never guesses past a gate, and never reaches outside its `<owns>`.
- **Two polls without an answer to a question, then documented best judgment** — only for decisions outside the hard-blocked list; the decision is written into the report.
- **The hard-blocked list while the Operator is away:** merging to `main`; any history rewrite; any edit to the decision-layer trio; writing a `DECISIONS.md` entry or a §7 invariant; deleting files; changing CI or environment configuration; anything touching real capital or a brokerage credential. These wait for the Operator, full stop.
- **Stand-down.** When the Operator returns or the campaign closes, every session deletes its timers and loops and posts that it did so. No loop outlives its campaign.
- **The Operator reads the campaign issue first thing in the morning.** Everything that happened is there; nothing is in a chat window only.

### 3.2 The marks, the words, and the close

**The mark** is the first line of every session, and of every Executor message. It is filled from `python scripts/session_open.py`, never from memory. The Operator reads it to see that the session is calibrated; wrong or missing numbers mean it is not. `CLAUDE.md` §2 gives the three shapes (the pen, the Executor, Codex) and the rules for each slot.

The pen's mark carries five facts:
- where `main` is;
- how many content commits `PROJECT_STATE.md` is behind;
- how many other branches exist;
- how old the data is (the frontier `data/DATA_MANIFEST.json` records);
- the nearest open deadline in `docs/deadlines.md`.

A slot whose source fails reads `unknown`. After a context compaction the session runs session-open again: a compacted session that has not re-read this document is not following it.

**The words.** "yes" approves a specific proposal, "authorize …" starts a named piece of work, and "close" ends a session (§2, Leg 1).

**The close.** It runs when the Operator says "close", and after every merge to `main`:
- **`PROJECT_STATE.md`:** the stamp, the Branches line ("`main` is at `<hash>`"), and §0 B, the handoff (done, remains, next action, authorized or proposed).
- **`CHANGELOG.md`:** one bullet per merge that landed.
- **`docs/deadlines.md`:** what is new, moved or closed, with evidence.
- **`python scripts/check_working_structure.py`:** green.
- **One line to the Operator:** what is unmerged, and what the next session opens with.

Close commits are typed `docs(close): …`, so the next session's drift count skips them. A pen that opens with a non-zero drift count runs the close for the previous session before new work. A pen without repository access hands the close to the Executor as its last Execution Prompt. A forgotten close then costs one session-open, not weeks.

---

## 4. Handoff contracts

These five interfaces are where the system succeeds or fails. Each has a required shape.

### 4.1 Operator → Strategist, and the sharpening gate

No required format from the Operator. The Operator describes the need in plain terms.

**The Strategist's obligation on receipt: sharpen the request before acting on it, and record the result.** A request will often be narrower than the real opportunity behind it, or broader than one clean run should hold. Sharpening makes it *precise*, never larger.

When the request has depth behind it, the Strategist asks **3 to 5 short follow-up questions** designed to surface what the Operator did not think to specify: the underlying goal, what would count as success, what has already been tried, what must not break, and what the acceptable scope of one run is. Two rules govern this:

- **Only when warranted.** If the task is small, obvious, or fully specified, skip the questions and write "not warranted, because …" in the record.
- **Questions must be answerable without re-deriving the whole system.** Ask about intent and priority, not about implementation the Strategist should determine itself.

**Restate, then wait.** After the questions, the pen restates the request in one paragraph: what we would do, why, and what counts as success. If that differs from what was asked, it says what changed and why. It drafts nothing until the Operator confirms. For a fully specified request, the Operator's own words are both the restatement and the confirmation, and the pen quotes them.

**The record** is the Operator's confirmed words, quoted on line 2 of every Execution Prompt, together with the `<request-as-sharpened>` block defined in `docs/PROMPTING_STANDARD.md`. Both are copied into the PR body, and the Run Summary's "Objective as I understood it" echoes them. A PR without them is not to standard.

### 4.2 Strategist → Operator (discussion)

**Clear, direct, formal, simple, and above all concise.** Short sentences. Plain words. Technical vocabulary only when there is no alternative, and then with a one-line gloss. Be sharp and brief by default; the Operator will ask for a fuller explanation if one is wanted.

Every proposal states, in this order, as briefly as it can be stated:

1. **What we would do** — in plain terms.
2. **Why** — what improves.
3. **What it costs** — time, risk, complexity.
4. **What could go wrong** — honestly.
5. **My recommendation** — one clear call.

If several viable paths exist, present at most three, one sentence of tradeoff each, and **still recommend one**. The Operator is never asked to break a tie between technical options. **Never advise without confidence.** If a decision requires knowledge the Strategist does not have — an acceptable risk threshold, whether a modelling assumption matches the Operator's intent, whether a result is good enough to build on — stop and ask.

### 4.3 Strategist → Executor (the Execution Prompt)

Written to the project's prompting standard, `docs/PROMPTING_STANDARD.md`, which defines the template and the run modes. The prompt is self-contained: the Executor may be a fresh session with no memory of the discussion that produced it.

**Line 1 is the run mode** (**read-only** or **change**). **Line 2 is the Operator's confirmed request, quoted.** An Executor refuses a prompt that does not open this way.

Then come the fixed sections, in the standard's tags:
- the sharpened request;
- objective, context, scope, and out-of-scope;
- `<owns>` / `<reads>`;
- the relevant §7 invariants, restated;
- approval gates, and push-before-handoff;
- verification commands with their expected output;
- the Run Summary requirement, citing **OPERATING_MODEL.md §4.4** by that full path;
- constraints.

Batch-size discipline: a run that would touch more than roughly eight files, delete anything, add a new top-level file, modify CI or environment configuration, or change any file in the decision layer must contain an approval gate before that step. Proportionality: a small run still carries every required element; brevity comes from each element being short, never from omitting one.

### 4.4 Executor → Operator (the Run Summary)

**This is the most important artifact in the system.** It is posted as a comment on the run's pull request so that it exists on GitHub and directs the verification. Write it for a reader who will independently confirm every claim. It opens with the Executor's mark (§3.2), then twelve headings, in order:

1. **Run header**: date, branch, base commit, current HEAD, and the commit id pushed to GitHub. A Run Summary without a pushed commit id is incomplete and is returned unread. If the run went badly, say so here, first.
2. **Objective as I understood it**: one paragraph. Echo the confirmed request and the sharpened request, and state any difference from the prompt.
3. **What I did**: one entry per change. Give what changed (its role and purpose, not only the filename), why, and a confidence of high, medium or low with a reason.
4. **Opportunistic fixes**: each one separately and labelled, or "none".
5. **Commands and their real output**: pasted, not summarised. Give the decisive portion if long, and say what was truncated. For anything touching engine logic, include the import smoke-check and the test subset.
6. **What I did not do**: every in-scope item not completed, with the reason. An empty section here is almost always a sign something was missed.
7. **Gates hit**: the yes/no question asked, what the Operator answered, and how you proceeded. Quote the approval for anything irreversible.
8. **Invariant check**: each §7 item relevant to this run, marked checked or not applicable, with the command or inspection that proves it.
9. **Risks**: possible regressions, ranked.
10. **Current state**: branch, tree clean, pushed, PR URL, whether CI ran, and each check's conclusion.
11. **What must happen next**, and who takes it.
12. **Open questions** to the pen.

**Prohibited in a Run Summary:** claiming a check you did not run; describing a command's expected output instead of its actual output; "should work", "appears correct", "likely fine"; omitting a failure because a later step worked around it; smoothing over a skipped gate; burying an opportunistic fix; presenting a passing test suite as proof of correctness (state what the tests cover); asserting a runtime result without quoting the output that shows it. **A truthful summary of a partially failed run is more valuable than a confident summary of an ambiguous one.**

### 4.5 Strategist → Operator (evaluation)

A Run Summary is evaluated by a Strategist context that did not write the run (§2.4). The pen opens with its mark. It **first checks GitHub itself** — the diff, the file contents, the git state, the branches, the PR and its checks — and only then writes:

1. **Plain-language recap** — two or three sentences.
2. **What I verified** — claims checked directly against the repository, and what the check showed.
3. **What I could not verify** — anything requiring a runtime result or forward market performance; what would settle it and who has to do it.
4. **Discrepancies** — anything done that was not asked, asked that was not done, or stated that contradicts the repository. A summary that disagrees with the repository is a serious finding.
5. **Risk assessment** — ranked.
6. **Recommended next step** — exactly one.
7. **The next Execution Prompt** — if scope is already clear; otherwise the questions that must be answered first.

---

## 5. Verification model

Claims fall into three tiers. Every claim the Strategist repeats to the Operator is labelled with its tier, in substance if not in name.

**Tier 1 — Verifiable by the Strategist directly on GitHub.** Whether a file changed and how, git and branch state, whether the code matches what was described, whether an invariant still holds in the source, whether the work matches what was asked. Reporting a Tier 1 claim without having checked it is a breach of this document.

**Tier 2 — Requires evidence the Strategist cannot produce alone.** Command output, test results, coverage figures, backtest output, data-pull behaviour, anything runtime. The Strategist requests the specific evidence rather than inferring it.

**Tier 3 — Requires the live system or the market itself.** The behaviour of a live data feed under real conditions, and — permanently — whether the engine's rankings produce good outcomes in live markets going forward.

Rules on top of the tiers:

- **Claims about runtime results are never Tier 1.** They are settled by reading the actual output.
- **A passing test suite does not prove the model is correct.** It proves the code behaves as the tests expect.
- **The central validity question — whether high-EV candidates outperform a simpler benchmark out of sample — is permanently Tier 3.** A backtest is evidence, never proof.
- **No CI run means not verified.** A pull request with no completed CI run is unverified regardless of local checks. A PR stacked on a non-`main` base triggers no CI — retarget it to `main` before it is treated as verified.
- **Writer and checker are different contexts** (§2.4). The strongest verification pattern on record is a second executor independently re-running fail-before / pass-after checks in an isolated worktree; use it for anything touching the decision layer.
- **`main` is not branch-protected (Operator ruling, 2026-09-17).** The Operator alone merges, and only a PR whose head is green in CI; nobody pushes to `main` directly. This is a convention the Operator enforces personally, not a GitHub setting — the Operator chose not to turn protection on.

---

## 6. Failure modes and recovery

**A mark is wrong, missing, or filled from memory.** The session is not calibrated. It runs `python scripts/session_open.py` again before anything else, and the Operator treats its earlier numbers as unverified.

**The docs drift count is not zero at session-open.** `PROJECT_STATE.md` is behind `main`. The pen runs the close for the previous session (§3.2) before new work; `git log <recorded hash>..origin/main` says what happened in between.

**A prompt reaches the Executor without the run mode on line 1 and the confirmed request on line 2.** The Executor refuses and asks for them. It changes no file.

**The Executor reports work it has not pushed.** The Strategist cannot verify it and must not pretend otherwise. Ask for the push; evaluate nothing until it arrives.

**The Executor answered a question by changing files.** The run exceeded its mode. Do not merge. Assess the change on its merits separately; if unnecessary, close it — already-done is not a reason.

**The Executor wrote a new invariant, constraint, or `DECISIONS` entry on its own initiative.** Treat the reasoning as unverified until the Strategist has checked the premise. A false rule in a governing document outlives any mistaken line of code. Remove it unless the premise holds.

**The Executor did something out of scope.** Declared and outside the invariants: review on its merits, accept or revert. Undeclared: a process breach — flag it, assess harm, write a revert prompt if warranted.

**The summary is thin, evasive, or disagrees with the repository.** Do not proceed on it. The repository is the truth; establish what actually happened before any further work.

**A pull request shows no checks.** Not verified, however clean it looks. Retarget to `main`, let CI pass, then evaluate.

**`FILE_MANIFEST.md` or `docs/worklog/INDEX.md` conflicts on merge.** Expected: every PR appends rows to both. Resolve mechanically — take the union of both sides' rows in `FILE_MANIFEST.md`, regenerate `docs/worklog/INDEX.md` with `python scripts/gen_worklog_index.py`, prove it with `python scripts/check_manifest_coverage.py`. Never `git checkout --theirs` on a magnet file (it once silently reverted a 12-doc archive move).

**The Strategist's plan turns out wrong once the Executor sees the actual state.** The Executor stops, reports, and does not improvise a redesign.

**The Operator does not understand a recommendation.** The Strategist's failure. Re-explain from a different angle, more simply. Never proceed on an approval the Operator did not actually understand.

**The Strategist is unsure and advises anyway.** The most damaging failure in the system, because the Operator cannot detect it. Resolve the uncertainty first.

**Two runs conflict, or the state has drifted from what everyone believed.** Stop the loop. Inspect every branch and every open PR; report the true state; resume planning only after that.

**Something touched a §7 invariant or reached the decision layer without a gate.** An incident. Halt other work. Verify the invariant is intact, restore it if not, then continue.

**A session is still polling after its campaign closed.** Post the stand-down, delete the timer, and record it on the campaign issue. Unattended loops exist only inside §3.1.

---

## 7. Project invariants

Both the Strategist and the Executor must honour these. The Strategist must not propose work that violates one; the Executor must refuse work that would.

**Decision integrity.** No tradeable candidate is produced except through the authoritative EV evaluation (`EVEngine.evaluate`). Ranking happens only through the single ranking entry point (`rank_candidates_by_ev`). Reviewers may downgrade a candidate, never upgrade it — no reviewer path raises the EV. The dealer positioning multiplier is clamped to its fixed range and scales only the EV dollar figure, nothing else. No path bypasses these, however obviously safe a shortcut looks.

**Data integrity.** The active data provider is logged at the start of every run. A per-endpoint failure of a live provider raises the defined failure rather than being silently substituted with the CSV provider; a globally-down provider may fall back, a single failing endpoint may not. No computation that feeds a decision may use data timestamped after the decision moment — point-in-time discipline is absolute, and any claimed look-ahead fix must be shown to hold. Split-adjusted and raw price series are never mixed.

**Brokerage safety.** Every path that touches the brokerage is read-only. No participant places, modifies, or cancels an order, and no order-routing or OMS surface is built, however sandboxed it looks. Simulated fills are permitted only in explicitly-labelled paper surfaces that cannot reach a broker. Real account data lives only in the gitignored data directory, is never committed, and is owned by the named terminal responsible for it.

**Modelling.** Every Greek consumed anywhere in the system respects the single unit contract at every call site; unit drift is a silent correctness failure. Dormant tooling is not wired into the decision path without a named missing-data contract first being chosen. Deprecated trees are not extended; new logic goes in the live modules. No news, sentiment, or narrative input reaches the EV path; when a news layer is redesigned it enters, if at all, as a downgrade-only reviewer under an explicit decision.

**Engineering.** The decision layer is covered by tests at or above the project gate, and that gate is not lowered to make a run pass. The engine remains importable as a whole — the import smoke-check passes after any structural change. Growth in surface area without a decision behind it is a design smell, not a milestone.

**Validation honesty.** No claim that the engine "works" is made on the strength of internal consistency alone. Internal invariants prove the engine is self-consistent; they do not prove its outputs are realistic. The central out-of-sample question is treated as open until a valid, contamination-free result says otherwise, and even then the result is evidence, not proof. Every engine-versus-market number is stated with its benchmark and its horizon; a result at one expiry does not transfer to another.

**Process.** Work happens on a branch. `main` is unprotected by the Operator's choice; the Operator merges, or the pen when necessary with CI green (D32); nobody pushes to `main` directly, and nobody rewrites history without the Operator's explicit approval for that specific action. No unreviewed deletion. New decisions are appended to the decision log as the next free entry; existing decisions are updated, never silently reversed. Gitignored data and secrets are never committed. Nothing on the §3.1 hard-blocked list happens while the Operator is away.

**Honesty.** No fabricated numbers. A figure the Executor cannot produce with a command and paste as output is not stated as fact. Every repeated claim is labelled as proven, attested, or unverified.

**Governance of this list.** This section is amended by the Operator, on the Strategist's recommendation, after the reasoning has been checked. The Executor may propose an addition in a Run Summary; it may not write one. Every invariant here must trace to a real, verified failure or a genuine risk.

---

## 8. Quick reference

| | Operator | Strategist, the pen (Claude Code) | Second opinion (Codex) | Executor |
|---|---|---|---|---|
| **Is** | Project owner and director | Quant and systems brain | Independent reviewer | Hands in a terminal (main: Claude Code in VS Code) |
| **Sees** | The goals, the strategy, the campaign issue | GitHub, all branches | GitHub (read-only) and what the Operator pastes | Its clone or worktree and filesystem |
| **Decides** | Direction, merges, final approval | What to do and why | Nothing: finds and challenges | Nothing: executes the approved plan |
| **Produces** | yes / authorize / close, answers to sharpening questions | Marks, restatements, Execution Prompts, evaluations, the records | Findings that quote the line and propose a fix | Pushed branches, PRs, Run Summaries |
| **Must never** | Be the mechanism that catches a technical error | Advise without being sure; verify its own writing; push to `main` | Write anything | Skip a gate, hide a fix, touch the decision layer unclaimed, merge without a yes, or hand over unpushed work |

**The six sentences that matter most:**

1. Every session opens with its mark, filled from commands; every run starts from the Operator's confirmed request, quoted on line 2 of its prompt.
2. The Strategist checks the repository itself, on GitHub, across all branches — in a context that did not write the run.
3. The Executor pushes before every handoff; unpushed work cannot be verified, and unverified work does not count as done.
4. The Run Summary lives on the PR and contains evidence, not reassurance.
5. While the Operator is away, nothing on the hard-blocked list happens.
6. No tradeable output ever bypasses the authoritative EV evaluation, and no claim that the engine "works" rests on internal consistency alone.

---

## 9. Project reference

### 9.1 The system — four layers

The product is a probabilistic expected-value (EV) decision engine for wheel strategies (short cash-secured puts → covered calls, plus timing-gated strangles) on S&P 500 names. Direction as of 2026-09-16 (`DECISIONS.md` D29): tradeable expiries are **7, 14, 21 and at most 28 days** (the 35-day default is being replaced); trading close to scheduled events to capture elevated premium is a goal, gated behind an event-conditioned distribution and its validation; an **exit evaluator** (position management that learns from closed trades) and a **strategist commentary layer** are being added. The plan of record is `docs/RESTART_PLAN_2026-09-16.md`.

1. **Data layer** (`data/`, `data_processed/`, `scripts/pull_*.py`) — OHLCV, option chains, IV, fundamentals, macro. The Bloomberg panels under `data/bloomberg/` are a frozen 2018-01-02 → 2026-07-02 history; **the Bloomberg Terminal is no longer available** and the live path is being rebuilt from online sources and subscriptions (Track A of the restart plan). **The data lives under the data root on the Operator's desktop (`SWE_DATA_ROOT`), never in git — D31**; git holds `data/DATA_MANIFEST.json`, the checksum ledger of every dataset. Provider selection via `SWE_DATA_PROVIDER`; capability matrix in `docs/DATA_POLICY.md` §2, data root and first fill in §6.
2. **Quant layer** (`engine/`) — Black-Scholes-Merton pricing + Greeks to 3rd order, empirical forward distributions (non-overlapping → overlapping → block bootstrap → HAR-RV → lognormal fallback), POT-GPD tail risk, 4-state Gaussian HMM regime, Nelson-Siegel skew, Student-t copula CVaR, dealer GEX / walls / gamma-flip (Theta chain data only). SVI surface tooling is live but fail-loud and off the EV ranking path (`DECISIONS.md` D9).
3. **Decision layer — the authoritative ranker.** `engine/ev_engine.py` (`EVEngine.evaluate`: event lockout → cost model → fair value → forward distribution → statistics → clamped regime and dealer multipliers → `EVResult`), `engine/wheel_runner.py` (`WheelRunner.rank_candidates_by_ev`, the one ranker every tradeable path routes through), and `engine/candidate_dossier.py` (`EnginePhaseReviewer`, the downgrade rules R1–R11 in §9.2).
4. **Interface layer** — `engine_api.py` (HTTP on `:8787`), `dashboard/` (Next.js: cockpit, portfolio, terminal), `engine/tradingview_bridge.py` (chart providers — sanity check, not a decider). The news stacks, the local-AI integrations (the browser agent, the Ollama memo and chat) and the advisor committee (`advisors/`) were removed on 2026-09-16 and 2026-09-17 (D29).

### 9.2 Decision authority — the hard invariant and the reviewer register

**No tradeable candidate bypasses `EVEngine.evaluate`.** Chart providers, the dealer positioning overlay, and the TradingView bridge can all **downgrade** a verdict (negative → blocked, proceed → review, proceed → skip). None of them can **rescue** a negative-EV trade. The dealer multiplier (`dealer_regime_multiplier` in `engine/dealer_positioning.py`) is clamped to `[0.70, 1.05]` and only scales the final `ev_dollars` — it never touches `ev_raw`.

If you add a new input (a data source, an advisor, a chart feed), wire it as a participant in a chained provider or as a downgrade-only reviewer. Do not introduce a code path that converts a non-tradeable candidate into a tradeable one without a fresh `EVEngine.evaluate` call.

The `EnginePhaseReviewer` rules (they run when a dossier is built, not inside the ranker):

- **R1:** negative OR non-finite EV → blocked (**R1a** guards `±inf` / `NaN` before the sign check, `verdict_reason="ev_non_finite"`). On the default dossier path R1 is pre-empted because the ranker already drops EV below zero.
- **R2:** chart missing or errored → a **note**, not a stop (D30, 2026-09-17): R3/R4 are skipped and the ladder continues to R5–R11, so a headless run still reaches the EV threshold and the portfolio gates.
- **R3:** spot mismatch > 2% → skip.
- **R4:** phase contradiction → skip *(dormant — no chart provider populates a phase).*
- **R5:** EV above threshold (`min_proceed_ev`, 10.0) → proceed, else review.
- **R6:** short-gamma regime + strike at/above put wall, or dealer regime near gamma flip → review *(needs an option chain; dormant on the CSV provider).*
- **R7–R10** *(D17 soft-warns; need an attached `PortfolioContext`; act only on proceed):* R7 portfolio VaR_95 above `max_var_pct × NAV` (default 5%); R8 stress drawdown > 8% NAV or `short_gamma_amplifying`; R9 sector cap (`max_sector_pct`, default 25% NAV); R10 single-name cap (`max_single_name_pct`, default 10% NAV). The tracker applies R9/R10 as HARD refusals at `open_short_put` when armed via `make_live_book_tracker`.
- **R11:** VIX above 25 and `prob_profit` above 0.90 → review (elevated-vol top-bin size-down; `DECISIONS.md` D23). Kept external to the HMM on purpose: the measured miscalibration sits at regime transitions the HMM has not caught.

Enforced structurally by `tests/test_audit_invariants.py`, `tests/test_dossier_invariant.py`, `tests/test_r11_elevated_vol.py`, `tests/test_authority_hardening.py`, and the rest of the launch-blocker set (`TESTING.md`). Never move or merge a change touching these without the Operator's sign-off.

### 9.3 Out of scope by design

Get explicit Operator consent before crossing any of these:

- **No auto-execution, broker wiring, OMS, or order routing** (§7 Brokerage safety).
- **No tick-level order flow / sub-minute features.**
- **No non-S&P-500 universe and no strategies beyond the wheel** (short puts, covered calls, timing-gated strangles).
- **No news or sentiment input to the EV path** until a redesign is ruled (D29).
- **No expiry beyond 28 days as a default tradeable menu** (D29); longer horizons remain available to research harnesses.

### 9.4 Fresh-session bring-up

0. Point at the data. `SWE_DATA_ROOT` names the desktop data root (unset = the repository folder). `python scripts/data_manifest.py check --root <root> --size-only` says in seconds whether the root is complete (D31). An environment without a data root — CI, a Cowork sandbox — runs the unit lane only: the `requires_data` tests skip visibly and steps 2–3 below cannot run there.

1. Confirm the provider. Default is `bloomberg` (the frozen CSVs); verify with:

   ```python
   from engine.wheel_runner import WheelRunner
   print(type(WheelRunner().connector).__name__)
   ```

   Expect `MarketDataConnector` on the CSV path. **Always log which provider was actually selected.**

2. Sanity-check the data layer with the 5-ticker smoke test:

   ```python
   from engine.wheel_runner import WheelRunner
   df = WheelRunner().rank_candidates_by_ev(
       tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
       top_n=10, min_ev_dollars=-1e9,
       include_diagnostic_fields=True,
   )
   print(df.attrs.get("drops_summary"))
   ```

   Runs in seconds. Healthy means: every ticker either ranks with non-null `ev_dollars`, `iv`, and `premium`, or appears in `attrs["drops"]` with a named gate (an earnings lockout is the usual reason). The count of rows is date-dependent; do not chase a "missing" row.

3. Read the staleness warnings the connector prints. A frontier more than 7 days behind the wall clock means a live `as_of=None` rank is pricing off an old close; set `SWE_REFUSE_STALE_LIVE=1` to hard-refuse instead of warn-and-rank.

4. For any change touching the decision layer: run the launch-blocker subset and the fast lane, `python -m pytest tests/ -m "not backtest_regression" -q`. A bare `pytest tests/` pulls the 4-to-5-hour backtest-regression lane in locally; use the marker.

For sandbox-vs-machine capability differences, see `docs/DATA_POLICY.md` §7.

### 9.5 Concurrency mechanics

- **Worktrees or clones.** Every Executor works in its own worktree (`git worktree add ../swe-<id> -b claude/<slug> origin/main`) or its own clone. Two Executors in one working tree share one index and one HEAD; one's `checkout` silently moves the other's branch. When several Executors share a machine, source `scripts/setup-terminal.sh <id>` (or the `.ps1`) so each gets its own `SWE_API_PORT`, `COVERAGE_FILE`, and `PYTEST_CACHE_DIR`.
- **One branch, one PR per run.** Branch names `claude/<short-slug>` (Codex writes no branches, D32). `git fetch origin && git rebase origin/main` before every push on your own branch; never rewrite a branch someone else has checked out.
- **The campaign issue.** Each piece of work that spans more than one run, or that will run while the Operator is away, gets one GitHub issue: the sharpened request, the Execution Prompts as numbered comments, the reports, the blocked items, the stand-down. Its number goes in the lane-claim block's `campaign:` line and in every PR body for the campaign. The old board (#113) and the July channels are closed and are history.
- **The decision-layer hard gate (CI-enforced).** The trio is edited one run at a time. `scripts/check_lane_claim.py` runs in CI on every PR: if the diff touches a decision-layer file, the PR description must carry a `lane-claim` block naming that file, or the build fails. Run it locally before pushing: `python scripts/check_lane_claim.py --base origin/main --claim-file <pr-body.md>`.
- **Worklog fragment per run** (`python scripts/new_worklog.py <id> --title "..." --kind <kind>`, then `python scripts/gen_worklog_index.py`; CI fails if the index is stale). Format: `docs/worklog/README.md`. Set `status` and `pr:` when the PR merges; a fragment left `in-flight` after merge is a defect.
- **Where to document.** The learning record goes in the worklog fragment; the PR body links it; a structural decision with rejected alternatives becomes a `DECISIONS.md` entry (only when one was actually made, and only when the Operator directed it); the one-line "what shipped" is a `CHANGELOG.md` bullet added in the same PR. `PROJECT_STATE.md`'s date line is refreshed by any PR that changes durable state; the doc-currency CI gate fails when it or `CHANGELOG.md` is more than 45 days old.
- **Magnet files.** `FILE_MANIFEST.md`, `docs/worklog/INDEX.md`, `CHANGELOG.md`, `PROJECT_STATE.md` collide by construction; resolve per §6 (union of rows, regenerate the index), never by taking one side.
- **Retired with v3:** the allocator role, task cards, the live board, ripple notices, lettered terminals A–D, scenario (`Sn`) numbering (campaigns are keyed by name and PR), and the July command-bus variants (their surviving rules are §3.1).
- **Retired with v4 (D32):** Codex as an equal Strategist or as an Executor (it is the read-only second opinion), and `CLAUDE.md` as a bare pointer (it is the checklist).

### 9.6 Onboarding and read order

`CLAUDE.md` is auto-loaded on every Claude Code session and carries the checklist; Codex loads `AGENTS.md`, whose Appendix is the same checklist word for word. Any AI agent entering this repo first runs session-open (`CLAUDE.md` §2) and then follows the same read order:

1. **This document**, §§1–8 in full; §9 as reference.
2. **`PROJECT_STATE.md`** §0 (the direction, the handoff, the recorded `main` commit) and §1 — what is authoritative right now.
3. **`AGENTS.md`** before executing; **`docs/PROMPTING_STANDARD.md`** before writing or executing any prompt.

For "where does X live / what tests cover Y" lookups, `docs/REPO_MAP.md` routes each question to the one owning doc.

**What you can change without asking:** docstrings; new tests under `tests/` that exercise existing public surface; a worklog fragment for your run; documentation (`docs/`, root `*.md`) **except** the rule-books (`CLAUDE.md`, `AGENTS.md`, this document, `DECISIONS.md`), which the pen drafts and the Operator approves; files under your active feature branch; updates to `CHANGELOG.md` / `PROJECT_STATE.md` / `ROADMAP.md` that reflect the work you just did.

**What requires an explicit ask:** everything on the `CLAUDE.md` §6 list; any change under `engine/ev_engine.py`, `engine/wheel_runner.py`, or `engine/candidate_dossier.py`; any new data-provider class, advisor, or chart provider; adding broker / OMS / order-routing surface; editing a rule-book; writing a `DECISIONS.md` entry; committing to `main`; deleting files. Market data is never committed at all (D31).

When in doubt, ask before doing.

### 9.7 Commit and pull-request standard

The repo's history is its second test suite. A commit message that explains *why* lets the next agent recover context months later from `git log` alone.

**Commit message format:**

```
type(scope): one-line summary in imperative voice

Changed:
- bullet (files + behaviour)

Why:
- bullet (the motivation, the constraint, the past incident)

Tested:
- bullet (the exact command you actually ran — copy-pasteable)

Tried but rejected:
- bullet (omit if none)

Unresolved:
- bullet (omit if none)

AI handoff:
- bullet (omit if none)
```

Omit a section entirely if it is empty. **Allowed `type` values:** `feat` | `fix` | `docs` | `refactor` | `test` | `chore` | `audit` | `perf`. **`scope`** is the smallest module the commit really touches. **Close commits** are `docs(close): …` (§3.2); session-open's drift count skips them.

**PR format:** the `.github/pull_request_template.md` sections (Summary / Changes / Why / Tests / §2 surface / Tried but rejected / Unresolved / AI handoff), the `<request-as-sharpened>` block, and the lane-claim block when the trio is touched. The Run Summary (§4.4) is a PR comment. PR titles follow `type(scope): summary`, under 70 characters.

**Attribution (ruling 2026-09-16, D29, reversing 2026-07-28):** commit trailers may carry the harness's co-authorship line, including a model name, and the session link; they are the only per-session audit trail the record has. Existing history is left untouched.

**Never edit `main` directly.** Even a one-line typo fix gets a branch + PR. **Anti-patterns:** "WIP" messages; many "fix typo" commits (squash before opening the PR); `--no-verify`; force-pushing a shared branch; committing secrets (`git diff --staged` before commit).

### 9.8 Engineering practice

**Prerequisites:** Python 3.11+, Node.js 18+ (dashboard only), Git. `pip install -e ".[dev]"` (what CI uses) inside a virtual environment; `pre-commit install`.

**Style:** PEP 8 + Ruff defaults; Ruff is the formatter. Type hints on all functions. Docstrings on public functions. Greek units are pinned in `docs/GREEKS_UNIT_CONTRACT.md`; read it before touching any Greek code.

**Testing:** tests live in `tests/`, flat `test_*.py`, descriptive names. Add tests for new functionality. For decision-layer changes, add a test that pins the invariant, not just one that covers the new path; the launch-blocker family (`TESTING.md`) is what enforces the hard EV invariant. Every test file needs its one-line row in `TESTING.md` (CI-gated). Fast lane: `python -m pytest tests/ -m "not backtest_regression" -q`. Backtest regression (4–5 h): `.claude/commands/backtest-regression.md`, required after changes to `ev_engine.py`, `wheel_runner.py`, `forward_distribution.py`, `dealer_positioning.py`, or `tail_risk.py`.

**Lint (CI hard-fails):** `ruff format --check .` and `ruff check .`.

**CI:** all jobs must pass before merge (environment validation, lint and type check, security scan, test suites, quantitative validation, integration tests, manifest coverage, worklog index, doc currency, working structure, decision-layer lane claim). The rule is *all green*.

**Keeping `FILE_MANIFEST.md` in sync:** `python scripts/sync_manifest.py --fix` appends rows for new files into the "Untriaged additions" tail; move each row to its section and write a real description. Orphans are flagged, never auto-removed.

**License:** MIT.

### 9.9 Model governance

A **single-operator project**: every governance role is performed by the Operator, backed by the CI gates and the decision-integrity tests. "Committee approval" means the Operator's own review recorded in `DECISIONS.md` and the PR trail. The binding standards: model risk tiers (Tier 1 option pricing / VaR / sizing; Tier 2 volatility estimators / regime; Tier 3 indicators / transforms); every model documented in `docs/MODEL_CARDS.md` with its mathematical basis, inputs and valid ranges, outputs and bounds, assumptions, and a reference; unit tests at every tier and property tests plus textbook validation for Tier 1; minimum coverage 70% line, 80% for Tier 1; walk-forward validation with embargo periods, transaction costs included, slippage documented, and every result stated with its horizon and benchmark; material changes (formula, new model, new horizon) need an Operator decision in `DECISIONS.md`, parameter recalibrations need a recorded review, bug fixes need CI. Approved references: Hull; Wilmott; Natenberg; Sinclair; Taleb.

### 9.10 Named terminals

If the Operator addresses a session by a **role name**, it adopts that role and reads its runbook in full before acting:

| Operator says | You are | Read first |
|---|---|---|
| **"You are responsible for the Dashboard"** (or `dashboard`) | the **Dashboard** terminal — owns the live IBKR portfolio viewer; on **"update"** it refreshes all live data | **`docs/DASHBOARD_TERMINAL.md`** |

The Dashboard terminal is strictly read-only (§7 Brokerage safety). Other sessions leave the dashboard, `data_processed/ibkr/`, and the `/api/portfolio/*` pipeline to it.

### 9.11 Where to look next

| For… | Read |
|---|---|
| What is authoritative / WIP / deprecated right now | `PROJECT_STATE.md` |
| The restart analysis and the rulings behind v3 | `docs/RESTART_BRIEF_2026-09-11.md`, `DECISIONS.md` D29, `docs/RESTART_PLAN_2026-09-16.md` |
| The checklist, the marks and the close (v4) | `CLAUDE.md`, `AGENTS.md`, `DECISIONS.md` D32, `docs/deadlines.md`, `scripts/session_open.py` |
| How to write a prompt or a Run Summary | `docs/PROMPTING_STANDARD.md` |
| Fast where/what/authoritative router | `docs/REPO_MAP.md` |
| Per-module purpose + decision-layer role | `MODULE_INDEX.md` |
| Test taxonomy + launch-blocker subset | `TESTING.md` |
| Every tracked file by purpose — grep, don't read | `FILE_MANIFEST.md` |
| Data layer (tiers, provider matrix, refresh, sandbox caveats) | `docs/DATA_POLICY.md` |
| Launch-blocker invariants before merging | `docs/LAUNCH_READINESS.md` |
| Real-money deployment gate | `docs/PRODUCTION_READINESS.md` |
| Greek units (canonical) | `docs/GREEKS_UNIT_CONTRACT.md` |
| *Why* a structural choice was made | `DECISIONS.md` |
| What's intentionally next | `ROADMAP.md` |
| What shipped recently | `CHANGELOG.md` |
| Every run's learning record | `docs/worklog/INDEX.md` |
| Dashboard terminal — live IBKR viewer + "update" | `docs/DASHBOARD_TERMINAL.md` |
