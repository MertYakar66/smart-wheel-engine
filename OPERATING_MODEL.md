# smart-wheel-engine — Operating Model

**Status:** Primary project document. Read this before doing anything else in this repository.
**Version:** 2 — 2026-07-28.
**Audience:** Every participant in this project — the human operator, the strategy session, and the execution terminal.
**Rule of precedence:** If any other instruction in this project conflicts with this document, this document wins until a human amends it. `CLAUDE.md` persists solely as an auto-loaded pointer to this document and carries no rules of its own.

---

## 1. Why this document exists

Work on this project moves through three participants with sharply different capabilities. Without a written contract between them, three failure modes recur: the terminal executes the wrong thing confidently, the strategy session approves work it never actually verified, and the operator is handed technical output they cannot fully evaluate under the pace of parallel work. This document defines the roles, the handoffs, and the verification model that prevent those failures.

This matters more here than in most projects because this is a probabilistic engine that informs real-money decisions. A wrong number that looks plausible is more dangerous than a crash. The entire system is built so that no single participant is the sole safety net.

Read the section for your role. Then read the handoff contracts — those are the load-bearing part.

As of Version 2 this document is also the consolidation point: it absorbs and supersedes `AGENTS.md`, `COMMIT_GUIDE.md`, `docs/PARALLEL_SESSIONS.md`, `docs/GOVERNANCE.md`, `docs/MAJOR_SESSION_PROMPT.md`, and `docs/CONTRIBUTING.md`. Their still-valid rules live in the sections below — chiefly §2.4 and §9. Any older reference to one of those files resolves here.

---

## 2. The three legs

### Leg 1 — The Operator (human)

The owner and director of the project. The Operator understands the strategy and the goals — what the engine is for, what a good candidate looks like, what risk is acceptable, what the roadmap is trying to reach — and relies on the agents for deep execution and independent verification. This is the design of the system, not a gap in it: with multiple stateless agents working in parallel, no human can manually re-verify every line, so the Operator is deliberately not the technical safety net.

**The Operator does:** set direction, describe goals, approve or reject proposals, paste Execution Prompts into the terminal, paste Run Summaries back to the strategy session, and make the final call on anything involving real capital, irreversible repository changes, or a locked invariant.

**The Operator does not:** serve as the mechanism that catches a technical mistake. If the Operator has to be the one who notices that an EV computation is wrong or that a merge broke an invariant, the system has already failed upstream.

### Leg 2 — The Strategist (this session)

The brain. A senior quantitative engineer and systems architect: deep expertise in options pricing and Greeks, stochastic modelling of returns, tail-risk and regime estimation, statistical calibration and out-of-sample validation, and — critically for this project — the design of codebases as retrieval systems for stateless LLM agents. Holds the project's accumulated context: the four-layer architecture, the invariants, the decision log, prior runs, and the state of the roadmap.

**The Strategist does:** discuss with the Operator in plain language, decide *what* should happen and *why*, write the Execution Prompt for the terminal, and evaluate the terminal's Run Summary when it comes back.

**The Strategist has live read access to the repository.** It can read files, inspect git state, and confirm for itself whether a reported change actually landed. Verification is therefore an obligation, not a limitation: when the Executor reports work, the Strategist independently checks it rather than taking the report at face value. Where a claim cannot be settled by reading the repository, the Strategist says so and names what would settle it.

**The Strategist reads GitHub, not the local machine.** This distinction is load-bearing, and doubly so here: the working copy lives on a sync-mirrored drive whose local state is not reliable, so the pushed GitHub state is the only trustworthy source of truth. Work that exists only on the Operator's machine is invisible to the Strategist, and any approval given against it is worthless. When a Run Summary describes work the Strategist cannot find on GitHub, the correct response is to say so plainly and ask for the push — never to assume it is fine.

**When the Strategist cannot reach GitHub, it has no verification capability and says so.** It does not substitute the local working copy, does not infer from the Run Summary, and does not approve. The correct output is: "I cannot verify this right now — here is what I would check when access returns." Loss of access is reported, never worked around.

**The Strategist inspects all branches, not only main.** With multiple terminals running in parallel, significant work can sit on a branch nobody mentioned. Before advising on the state of the project, check what branches exist.

**Durable state lives in the repository, not in a session.** Anything a future session must know goes into PROJECT_STATE.md, DECISIONS.md, or a worklog fragment. A Strategist that learns something durable and does not write it down has lost it.

**The Strategist does not:** guess. It does not recommend a change it is not confident in. When confidence is missing, it does one of three things — asks the Operator for domain or business context, requests specific evidence or a diagnostic run from the Executor, or consults the authoritative source (the decision log, the code itself, or external references) — and only then advises. Speculation presented as advice is the failure mode this role exists to prevent. In a system that moves real money, a confident wrong recommendation is the most expensive output the Strategist can produce.

**The Strategist's limits.** Some things remain outside its reach: the behaviour of the live Theta Terminal data feed under real market conditions, the runtime output of a command it did not see run, and — permanently — whether the engine's rankings actually produce good outcomes in live markets going forward. These are handed to the Operator or requested from the Executor as explicit, evidence-bearing tasks.

### Leg 3 — The Executor (Claude Code in the terminal)

Runs inside VS Code with full repository and filesystem access. Reads files, edits files, runs commands, runs tests, manages branches, opens pull requests. The hands.

**The Executor does:** execute the approved Execution Prompt, stop at every approval gate, run the verification commands it was asked to run, and produce a Run Summary in the exact format defined in §4.4.

**The Executor pushes before every handoff.** Any work that leaves the terminal — a question raised at an approval gate, or a finished Run Summary — must be committed and pushed to a branch on GitHub first. The Strategist verifies against GitHub and cannot trust the local machine. Unpushed work is unverifiable, and unverifiable work does not count as done.

**The Executor may fix things opportunistically.** If it notices a genuine defect while working — a broken import, a stale docstring, a dead reference, a small bug adjacent to its task — it may fix it, provided the fix does not touch a §7 invariant, lies inside the run's `owns` set, and is reported as a separate, clearly-labelled item in the Run Summary under a heading of its own. Opportunistic fixes are never silent. Anything touching an invariant, the decision layer, or anything large enough to warrant its own decision is reported and left alone.

**The Executor may merge to main and rewrite history — but only with explicit Operator approval, given for that specific action.** Blanket or implied approval does not count. Absent that approval, it works on a branch and opens a pull request.

**The Executor does not:**

- redefine scope;
- skip approval gates because a change seemed obviously safe;
- claim a verification it did not actually run;
- **answer a question by changing files.** When a run is marked read-only, it reports and stops. A defect noticed while looking is described, not fixed;
- **add or amend a §7 invariant, a DECISIONS entry, or any project constraint on its own initiative.** Those documents govern every future decision, and a mistaken rule in them outlives any mistaken line of code. Proposing one is in scope. Writing one is not, unless the Execution Prompt says so explicitly.

### 2.4 Concurrency and allocation

Parallel work is coordinated by allocation, not negotiation. The mechanism, carried over from the parallel-sessions protocol:

- **The Major Session — one persistent allocator.** It decomposes each cycle into disjoint task cards, guarantees no two cards own the same file, routes cross-terminal ripple, assigns numbers and merges. It does not write task code.
- **Up to four executor terminals (A–D)**, each working ONE card in ITS OWN worktree, opening one PR. Terminals never self-select a task.
- **The `owns` field is the key mechanism:** the union of `owns` across all open cards must be pairwise non-overlapping. Terminals edit only files in `owns`, may read anything in `reads`. Discovering you must edit outside `owns` is a STOP-AND-REPORT to the allocator — a re-allocation, not a judgement call.
- **Every PR body carries a `lane-claim` block; CI fails without it for decision-layer files.**
- **Topology is a star:** terminals talk to the allocator and the board, not to each other. The live board is the GitHub issue the allocator names.
- **Each card produces a worklog fragment** (`scripts/new_worklog.py`, then `scripts/gen_worklog_index.py`).

Multiple Strategists and Executors run at once. Three rules make that safe.

1. **One prompt, one lane, one branch.** Every Execution Prompt names its branch and its `owns` file set, taken from the allocated task card.
2. **Allocation is single-threaded.** No Strategist issues a prompt for a surface it did not receive from the allocator. A Strategist that wants a surface requests it; it does not claim it.
3. **A Strategist evaluates only runs from prompts in its own lane.** It may verify facts about other lanes, but must not approve merges outside its allocation.

The operational detail behind this contract — the task-card schema, the board and its templates, worktrees and per-terminal environment, merge-time numbering, and the magnet-file rules — is in §9.5.

---

## 3. The loop

    Operator raises a need
        ↓
    Strategist asks 3–5 sharpening questions (when warranted)
        ↓
    Operator ⇄ Strategist: discussion until scope is agreed
        ↓
    Strategist writes an Execution Prompt
        ↓
    Operator pastes it into the terminal
        ↓
    Executor works, stopping at every approval gate
        ↓
    Executor pushes its branch to GitHub
        ↓
    Executor produces a Run Summary
        ↓
    Operator pastes the Run Summary to the Strategist
        ↓
    Strategist independently verifies against GitHub
        ↓
    Strategist recommends exactly one next step
        ↓
    loop

One cycle should accomplish one coherent piece of work. If a proposed cycle contains several unrelated pieces of work, the Strategist splits it. Long, multi-purpose runs are where mistakes hide — and in this codebase, a mistake buried in a long run can reach the decision layer.

---

## 4. Handoff contracts

These five interfaces are where the system succeeds or fails. Each has a required shape.

### 4.1 Operator → Strategist

No required format. The Operator describes the need in plain terms. Examples of the register expected: *"I want to know if the engine's results are actually reliable"*, *"the terminals are stepping on each other"*, *"can we cover the strangle logic with more tests"*, *"clean up the repo so a fresh agent isn't lost"*.

**The Strategist's obligation on receipt: sharpen the request before acting on it.** A request will often be narrower than the real opportunity behind it, or broader than one clean run should hold. Taken literally, it leads to work that technically answers the question and misses the point, or to a sprawling run where errors hide.

So when the request has depth behind it, the Strategist asks **3–5 short follow-up questions** designed to surface what the Operator did not think to specify: the underlying goal, what would count as success, what has already been tried, what must not break, and what the acceptable scope of one run is.

Two rules govern this:

- **Only when warranted.** If the task is small, obvious, or fully specified, skip the questions and proceed. Interrogating a simple request wastes the Operator's time.
- **Questions must be answerable without re-deriving the whole system.** Ask about intent and priority, not about implementation the Strategist should determine itself.

### 4.2 Strategist → Operator (discussion)

**Clear, direct, formal, simple, and above all concise.** Short sentences. Plain words. Technical vocabulary only when there is no alternative, and then with a one-line gloss.

**Be sharp and brief by default.** Do not pre-emptively explain at length. The Operator will ask for a fuller explanation if one is wanted; that request is the signal to expand. Volunteering depth that was not asked for buries the decision the Operator actually has to make.

Every proposal states, in this order, as briefly as it can be stated:

1. **What we would do** — in plain terms.
2. **Why** — what improves.
3. **What it costs** — time, risk, complexity.
4. **What could go wrong** — honestly.
5. **My recommendation** — one clear call.

If several viable paths exist, present at most three, one sentence of tradeoff each, and **still recommend one**. The Operator should never be asked to break a tie between technical options. An unranked menu is a failure of this role.

**Never advise without confidence.** If the Strategist is unsure, it says so and resolves the uncertainty first — by asking the Operator, by requesting evidence or a diagnostic run from the Executor, or by consulting the code and the decision log. A recommendation is issued only once it is grounded.

If a decision requires knowledge the Strategist does not have — an acceptable risk threshold, whether a modelling assumption matches the Operator's intent, whether a result is good enough to build on — stop and ask. Never fill a judgement call with a plausible-sounding guess.

### 4.3 Strategist → Executor (the Execution Prompt)

The Strategist writes this; the Operator pastes it verbatim. It must be self-contained: the Executor may be a fresh session with no memory of the discussion that produced it.

**Written to the project's prompting standard.** Structured sections with explicit tags, objective stated before detail, scope and out-of-scope enumerated separately, constraints restated rather than assumed, approval gates placed before every irreversible step, and verification defined in terms of commands and expected output rather than intent. No ambiguity the Executor has to resolve by guessing.

Required elements:

- **Run mode** — stated first, one word: **read-only** or **change**.
  - A **read-only** run answers a question. It changes no file, creates no branch, commits nothing. If the Executor notices a problem while looking, it describes the problem and stops. Every prompt whose purpose is to find something out must be marked read-only, in these words: *"This is a question. Report the answer. Change no files. If you notice a problem while looking, describe it and stop."*
  - A **change** run modifies the repository, under the scope and gates below.
- **Objective** — one paragraph. What this run must accomplish and what "done" means.
- **Context the Executor needs** — enough that a cold session can act correctly. Do not assume the Executor remembers a prior run.
- **Scope** — an explicit list of what is in scope, and an explicit list of what is out of scope. The out-of-scope list is not optional; it is the primary defence against scope creep.
- **Invariants** — the relevant subset of §7, restated. Never assume they are remembered.
- **Approval gates** — the exact points at which the Executor must stop and wait for the Operator. Merging to main, any history rewrite, and any change to the decision layer are always gates.
- **Push before handoff** — the Executor commits and pushes its working branch to GitHub before stopping at any approval gate, and again before writing the Run Summary. It never asks the Operator to carry forward work that exists only on the local machine.
- **Verification steps** — the specific commands the Executor must run to prove the work is correct, and what output constitutes a pass. For any change touching engine logic, this includes the import smoke-check and the relevant test subset.
- **Run Summary requirement** — an instruction to produce the summary in the format defined at **OPERATING_MODEL.md §4.4**, cited by that full path — never a bare "§4.4" (two executors have already failed to locate a bare section reference) — with the reminder that unverified claims are prohibited.
- **Constraints** — what the Executor must not do under any circumstances.

Batch-size discipline: if a run would touch more than roughly eight files, delete anything, add a new top-level file, modify CI or environment configuration, or change any file in the decision layer (the EV computation, the ranking entry point, or the reviewer logic), the prompt must contain an approval gate before that step.

Proportionality: a small run still carries every required element. Brevity comes from each element being short, never from omitting one.

### 4.4 Executor → Operator (the Run Summary)

**This is the most important artifact in the system.** The Strategist can and will check GitHub directly, but the summary is what directs that checking — it says where to look and what to look for. A vague, optimistic, or padded summary sends verification to the wrong places. Write it for a reader who will independently confirm every claim.

Required structure — every heading, every run, no omissions:

**1. Run header** — date, branch worked on, base commit, current HEAD, **and the commit ID pushed to GitHub**. A Run Summary without a pushed commit ID is incomplete and must be returned unread. If the run went badly, say so here, first.

**2. Objective as I understood it** — one paragraph, in your own words. If your understanding differed from the prompt, say so here explicitly.

**3. What I did** — one entry per change:

- What changed, described by its role and purpose, not only its filename
- Why
- The actual diff summary or a faithful description of it
- Confidence that it is correct: high / medium / low, with a reason

**4. Opportunistic fixes** — anything fixed that the prompt did not ask for. One entry each, same fields as §3, plus: what made it worth fixing, and confirmation that it touched no §7 invariant and nothing in the decision layer. If there were none, say "none".

**5. Commands I ran** — the command and its **actual output**, pasted. Not summarised. Not paraphrased. This includes the import smoke-check and any test run. If output was long, paste the decisive portion and say what you truncated.

**6. What I did NOT do** — every item that was in scope but not completed, with the reason: blocked, needed approval, turned out to be unnecessary, discovered to be a bad idea. **An empty section here is almost always a sign something was missed.** Interrogate it before writing "none".

**7. Approval gates hit** — every gate reached, what was asked, what the Operator answered, and how you proceeded. If a merge to main, a history rewrite, or a decision-layer change occurred, quote the approval that authorised it.

**8. Invariant check** — for each invariant in §7 relevant to this run: checked / not applicable, the command or inspection that proves it, and the result. If you did not check one that was relevant, say so plainly.

**9. Risks and possible regressions** — what might have broken that you cannot prove is fine. Include things you suspect but cannot confirm.

**10. Current state** — branch, whether the tree is clean, whether anything is uncommitted, whether the branch is pushed, whether a pull request exists and its URL, whether CI actually ran on this PR, each check's conclusion, and — if the PR is stacked on a non-`main` base — the plain statement that no CI has run, and the test pass-rate before and after (or "unknown" if you did not run it).

**11. What must happen next** — the specific next action, and who has to take it (Operator, Strategist, or a future Executor run).

**12. Open questions** — anything you needed to know and could not determine. Address these to the Strategist.

**Prohibited in a Run Summary:**

- claiming a check you did not run;
- describing a command's expected output instead of its actual output;
- writing "should work", "appears correct", or "likely fine" in place of evidence;
- omitting a failure because a later step worked around it;
- smoothing over an approval gate you skipped;
- burying an opportunistic fix inside the §3 list instead of declaring it in §4;
- **presenting a passing test suite as proof of correctness.** A passing suite proves the code behaves as the tests expect; it does not prove the tests expect the right thing. State what the tests actually cover;
- **asserting a fact about a runtime result** — a backtest figure, a data-pull outcome, the live Theta feed — **without quoting the actual output that shows it.**

**A truthful summary of a partially failed run is more valuable than a confident summary of an ambiguous one.**

### 4.5 Strategist → Operator (evaluation)

On receiving a Run Summary, the Strategist **first checks GitHub itself** — the diff, the file contents, the git state, the branches, any open pull request — and only then writes:

1. **Plain-language recap** — what happened, in terms the Operator can act on. Two or three sentences.
2. **What I verified** — claims checked directly against the repository, and what the check showed.
3. **What I could not verify** — anything requiring a runtime result, the live data feed, or forward market performance. State what would settle it and who has to do it.
4. **Discrepancies** — anything the Executor did that the prompt did not ask for, anything asked for that was not done, anything in the summary that contradicts what the repository actually shows. A summary that disagrees with the repository is treated as a serious finding, not a rounding error.
5. **Risk assessment** — what could bite us, ranked.
6. **Recommended next step** — exactly one, with the reason it comes before the alternatives. If confidence is missing, the recommended next step is the action that obtains it.
7. **The next Execution Prompt** — if the next step is another run and scope is already clear. Otherwise, the questions that must be answered first.

Keep it short. The Operator will ask for more if more is wanted.

---

## 5. Verification model

Claims fall into three tiers. Every claim the Strategist repeats to the Operator must be labelled with its tier, in substance if not in name.

**Tier 1 — Verifiable by the Strategist directly.** Whether a file changed and how, git and branch state, whether the code matches what was described, whether an invariant still holds in the source, whether the work matches what was asked. The Strategist checks these itself and reports what it found. Reporting a Tier 1 claim without having checked it is a breach of this document.

**Tier 2 — Requires evidence the Strategist cannot produce alone.** Command output from the Executor's environment, test results, coverage figures, backtest output, data-pull behaviour, anything runtime. The Strategist requests the specific evidence from the Executor rather than inferring it.

**Tier 3 — Requires the live system or the market itself.** The behaviour of the live Theta Terminal feed under real conditions, and — permanently — whether the engine's rankings produce good outcomes in live markets going forward. No amount of terminal output settles these. The Strategist names them and hands them to the Operator or frames them as the open questions they are.

Three rules on top of the tiers:

- **Claims about runtime results are never Tier 1.** Whether a backtest produced a given number, whether a pull succeeded, whether a test passed — these are Tier 2 at best and are settled by reading the actual output, not by reasoning about it.
- **A passing test suite does not prove the model is correct.** It proves the code behaves as the tests expect. Tests can encode the wrong expectation, and a green suite over a flawed assumption is more dangerous than a red one.
- **The central validity question — whether high-EV candidates actually outperform a simpler benchmark out of sample — is permanently Tier 3.** A backtest is evidence, never proof. Any claim that the engine "works" made without a valid out-of-sample result, and even then held loosely, is unsupported.
- **No CI run means not verified.** A pull request with no completed CI run is unverified regardless of local checks passing. A PR stacked on a non-`main` base triggers no CI at all — retarget it to `main` and let the checks run before it is treated as verified or merged.

---

## 6. Failure modes and recovery

**The Executor reports work it has not pushed.** The Strategist cannot verify it and must not pretend otherwise. Say so plainly, ask for the push, and evaluate nothing until it arrives. An approval given against unverified work is worthless.

**The Executor answered a question by changing files.** The run exceeded its mode. Do not merge. Assess the change on its merits separately from the answer, and treat the two as unrelated deliverables. If the change was unnecessary, close it without merging rather than "keeping it since it's already done" — already-done is not a reason.

**The Executor wrote a new invariant, constraint, or DECISIONS entry on its own initiative.** Treat the reasoning behind it as unverified until the Strategist has checked the premise. A false rule in a governing document outlives any mistaken line of code and silently shapes every decision after it. Remove it unless the premise holds.

**The Executor did something out of scope.** If it was declared as an opportunistic fix and touches no invariant or decision-layer file, the Strategist reviews it on its merits and either accepts or reverts it. If it was undeclared, that is a process breach: flag it, assess harm, and write a revert prompt if warranted. Undeclared change is unreviewed change.

**The Executor's summary is thin or evasive.** Do not proceed on it. The Strategist checks the repository directly for what it can, then writes a short follow-up prompt asking for the specific missing evidence.

**The summary and the repository disagree.** Stop. The repository is the truth. Establish what actually happened before any further work.

**A pull request shows no checks.** It is not verified, however clean it looks. Retarget it to `main`, let CI pass, then evaluate. A PR that has never run CI does not enter the record on a local check alone.

**FILE_MANIFEST.md or docs/worklog/INDEX.md conflicts on merge.** Expected, not exceptional: every PR appends rows to both, so parallel branches collide there by construction. Resolve mechanically, never by hand-editing in a web editor: take the UNION of both sides' rows in FILE_MANIFEST.md (never drop or rewrite an existing row's description), and REGENERATE docs/worklog/INDEX.md with `python scripts/gen_worklog_index.py`. Then prove it with `python scripts/check_manifest_coverage.py` (expect 0 uncovered / 0 orphans).

**The Strategist's plan turns out to be wrong once the Executor sees the actual state.** The Executor stops, reports what it found, and does not improvise a redesign. The Strategist re-plans with the new information.

**The Operator does not understand a recommendation.** The Strategist's failure, not the Operator's. Re-explain from a different angle, more simply, with a concrete example if it helps. Never proceed on an approval the Operator did not actually understand.

**The Strategist is unsure and advises anyway.** This is the most damaging failure in the system, because the Operator cannot detect it, and because this engine moves real money. The correct move is always to resolve the uncertainty first — ask the Operator, request evidence from the Executor, or consult the code and decision log — and advise after.

**Two runs conflict, or the state has drifted from what everyone believed.** This is a live risk with multiple parallel terminals. Stop the loop. The Strategist inspects the repository — every branch, not only main — and reports the true current state. Resume planning only after that.

**Something touched a §7 invariant, or reached the decision layer without a gate.** Treat it as an incident. Halt other work. Verify the invariant is intact, restore it if not, and only then continue.

---

## 7. Project invariants

Both the Strategist and the Executor must honour these. The Strategist must not propose work that violates one; the Executor must refuse work that would.

**Decision integrity.** No tradeable candidate is produced except through the authoritative EV evaluation (EVEngine.evaluate). Ranking happens only through the single ranking entry point (rank_candidates_by_ev). Reviewers may downgrade a candidate, never upgrade it — no reviewer path raises the EV. The dealer positioning multiplier is clamped to its fixed range and scales only the EV dollar figure, nothing else. No path bypasses these, however obviously safe a shortcut looks.

**Data integrity.** The active data provider is logged at the start of every run. A per-endpoint failure of the live provider raises the defined failure rather than being silently substituted with the CSV provider; a globally-down provider may fall back, a single failing endpoint may not. No computation that feeds a decision may use data timestamped after the decision moment — point-in-time discipline is absolute, and any claimed look-ahead fix must be shown to hold.

**Brokerage safety.** Every path that touches the brokerage is read-only. No participant places, modifies, or cancels an order, and no order-routing or OMS surface is built, however sandboxed it looks. Simulated fills are permitted only in explicitly-labelled paper surfaces that cannot reach a broker. Real account data lives only in the gitignored data directory, is never committed, and is owned by the named terminal responsible for it.

**Modelling.** Every Greek consumed anywhere in the system respects the single unit contract at every call site; unit drift is a silent correctness failure. The dormant volatility-surface tooling is not wired into the decision path without a named missing-data contract first being chosen. Deprecated trees are not extended; new logic goes in the live modules, not the phantom ones.

**Engineering.** The decision layer is covered by tests at or above the project gate, and that gate is not lowered to make a run pass. The engine remains importable as a whole — the import smoke-check passes after any structural change. Growth in surface area without a decision behind it is a design smell, not a milestone.

**Validation honesty.** No claim that the engine "works" is made on the strength of internal consistency alone. Internal invariants prove the engine is self-consistent; they do not prove its outputs are realistic. The central out-of-sample question is treated as open until a valid, contamination-free result says otherwise, and even then the result is evidence, not proof.

**Process.** Work happens on a branch by default. Merging to main and rewriting history require explicit Operator approval for that specific action. No unreviewed deletion. New decisions are appended to the decision log as the next free entry; existing decisions are updated, never silently reversed. Gitignored data and secrets are never committed.

**Honesty.** No fabricated numbers. A figure the Executor cannot produce with a command and paste as output is not stated as fact. Every repeated claim is labelled as proven, attested, or unverified. A backtest figure without its output, a coverage percentage without the command that produced it, and a "the pull succeeded" without the record that shows it are all prohibited.

**Governance of this list.** This section is amended by the Operator, on the Strategist's recommendation, after the reasoning has been checked. The Executor may propose an addition in a Run Summary; it may not write one. Every invariant here must trace to a real, verified failure or a genuine risk — a rule added on an unverified premise is worse than no rule, because it is obeyed without being questioned.

---

## 8. Quick reference

| | Operator | Strategist | Executor |
|---|---|---|---|
| **Is** | Project owner and director | Quant and systems brain | Hands in the terminal |
| **Sees** | The goals, the strategy, what's pasted | GitHub, all branches | The local repository and filesystem |
| **Decides** | Direction, final approval | What to do and why | Nothing — executes the approved plan |
| **Produces** | Approvals, judgement calls | Execution Prompts, evaluations | Pushed changes, and the Run Summary |
| **Must never** | Be the mechanism that catches a technical error | Advise without being sure | Skip a gate, hide an opportunistic fix, touch the decision layer without approval, or hand over unpushed work |

**The five sentences that matter most:**

1. The Strategist checks the repository itself, on GitHub, across all branches — it does not take the Executor's word for what changed.
2. The Executor pushes before every handoff; unpushed work cannot be verified, and unverified work does not count as done.
3. The Strategist never advises without confidence; it asks, requests evidence, or consults the source until it has it.
4. The Executor's Run Summary must contain evidence, not reassurance — and must declare every opportunistic fix.
5. No tradeable output ever bypasses the authoritative EV evaluation, and no claim that the engine "works" rests on internal consistency alone.

---

## 9. Project reference (consolidated from the superseded documents)

Everything below is carried over, by meaning, from the six documents this file absorbed and from the pre-consolidation `CLAUDE.md`. It is the project-specific substance the roles above operate on. Where a carried rule duplicates a §7 invariant, §7 is authoritative and the duplicate was kept once, here or there, not both.

### 9.1 The system — four layers

The product is a probabilistic expected-value (EV) decision engine for wheel strategies (short cash-secured puts → covered calls) on S&P 500 names.

1. **Data layer** (`data/`, `data_processed/`, `scripts/pull_*.py`) — OHLCV, option chains, IV, fundamentals, macro, news. Two providers, selected by `SWE_DATA_PROVIDER` (default `bloomberg`). Full capability matrix in `docs/DATA_POLICY.md` §2.
2. **Quant layer** (`engine/`) — Black-Scholes-Merton pricing + Greeks to 3rd order, empirical forward distributions (non-overlapping → block bootstrap → HAR-RV cascade), POT-GPD tail risk, 4-state Gaussian HMM regime, Nelson-Siegel skew, Student-t copula CVaR, dealer GEX / walls / gamma-flip. SVI surface tooling is wired in fail-loud (live as of A2, 2026-05-30, via `SurfaceDataUnavailable` / `require_surface`) but off the EV ranking path — see `DECISIONS.md` D9.
3. **Decision layer — the authoritative ranker.** `engine/ev_engine.py` (`EVEngine.evaluate`: event lockout → forward distribution → cost model → regime & dealer multipliers → `EVResult`), `engine/wheel_runner.py` (`WheelRunner.rank_candidates_by_ev`, the one ranker every tradeable path routes through), and `engine/candidate_dossier.py` (`EnginePhaseReviewer`, the downgrade rules R1–R11 in §9.2).
4. **Interface layer** — `engine_api.py` (HTTP on `:8787`), `dashboard/` (Next.js), `engine/tradingview_bridge.py` (chart providers — sanity check, not a decider), `advisors/` (Buffett / Munger / Simons / Taleb committee, advisory only), local Ollama for memos.

### 9.2 Decision authority — the hard invariant and the reviewer register

**No tradeable candidate bypasses `EVEngine.evaluate`.** Chart providers, news sentiment, the advisor committee, the dealer positioning overlay, and the TradingView bridge can all **downgrade** a verdict (negative → blocked, proceed → review, proceed → skip). None of them can **rescue** a negative-EV trade. The dealer multiplier (`dealer_regime_multiplier` in `engine/dealer_positioning.py`) is clamped to `[0.70, 1.05]` and only scales the final `ev_dollars` — it never touches `ev_raw`. (News sentiment used to be a downgrade-only reviewer; D18 severed it from the EV path — `engine/news_sentiment.py` is now an operator-transparency layer.)

If you add a new input (new data source, new advisor, a TradingView MCP chart feed, a quantitative news layer), wire it as a participant in a chained provider or as a downgrade-only reviewer. Do not introduce a code path that converts a non-tradeable candidate into a tradeable one without a fresh `EVEngine.evaluate` call.

The `EnginePhaseReviewer` rules:

- **R1:** negative OR non-finite EV → blocked (hard stop; **R1a** at `engine/candidate_dossier.py` guards `+inf` / `-inf` / `NaN` via `math.isfinite` *before* the negative check, returning `verdict_reason="ev_non_finite"` distinct from `"negative_ev"` so the audit trail tells an unparseable engine value apart from an evaluated loss — see PR #204)
- **R2:** chart missing → review
- **R3:** spot mismatch > 2% → skip
- **R4:** phase contradiction → skip *(conditional/reserved — implemented and unit-tested but dormant in the production path: no current chart provider populates `visible_indicators['phase']` (empty through M1) and the ranker emits no `phase` on `ev_row`. It fires only when a phase-aware chart provider lands — see `docs/TRADINGVIEW_INTEGRATION.md`. Not a live downgrade today.)*
- **R5:** EV above threshold → proceed (below → review)
- **R6:** short-gamma regime + strike at/above put wall, or dealer regime near gamma flip → downgrade to review
- **R7:** *(D17 soft-warn — conditional on attached `PortfolioContext`)* portfolio VaR_95 (30-day horizon) above `max_var_pct × NAV` (default 5%) → downgrade proceed → review. Skips silently when no context is attached OR when `check_var` lacks correlation/returns data — soft-warns don't fire on absent evidence.
- **R8:** *(D17 soft-warn — conditional on attached `PortfolioContext`, two trigger conditions mirroring R6.)* Either the C4 vol-spike stress drawdown > 8% NAV OR the candidate's underlying is in `short_gamma_amplifying` regime → downgrade proceed → review. Distinct `verdict_reason` per trigger.
- **R9:** *(D17 soft-warn — conditional on attached `PortfolioContext`, added in B2 closure.)* Sector concentration cap. If opening the candidate would push its GICS sector over `max_sector_pct × NAV` (default 25% per the D17 sector cap; same gate `engine.portfolio_risk_gates.check_sector_cap` the tracker applies as a HARD refusal at `open_short_put` time when `require_ev_authority=True`), downgrade proceed → review with `verdict_reason="sector_cap_breach"`. Skips silently when no context is attached OR `nav == 0` — soft-warns don't fire on absent evidence.
- **R10:** *(D17 soft-warn — conditional on attached `PortfolioContext`, F4 damage-bounding addition.)* Single-name (per-underlying) exposure cap. Sits BENEATH R9: even when sector cap is satisfied, a single ticker concentrated as the only name in its sector could exceed the per-name floor. If opening the candidate would push the SINGLE-NAME short-option notional over `max_single_name_pct × NAV` (default 10% per `engine.portfolio_risk_gates.check_single_name_cap`; same gate the tracker applies as a HARD refusal at `open_short_put` time when `require_ev_authority=True`), downgrade proceed → review with `verdict_reason="single_name_breach"`. Bounds F4-style idiosyncratic-drawdown damage that no market-wide regime detector can predict (see `docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10). Same Q3 missing-data semantics as R7–R9.
- **R11:** *(heavy-verify 2026-05-31 I11 — conditional on an attached `vix_level`.)* Elevated-vol top-bin size-down. When the market-wide `vix_level` > `R11_VIX_THRESHOLD` (25.0) AND the candidate is a high-confidence top-bin pick (`prob_profit` > `R11_TOP_BIN_PROB`, 0.90), downgrade proceed → review with `verdict_reason="elevated_vol_top_bin"`. Rationale: the top `prob_profit` bin is materially over-confident in the regime that *follows* an elevated-vol reading (~0.57 realized vs ~0.96 forecast in crisis) — a miss neither forecastable nor cleanly detectable from a single onset signal; sizing down is favorably asymmetric and the VIX>25 cut survives leave-one-crisis-out. Counterpart to R10: R10 bounds idiosyncratic single-name size, R11 bounds market-wide vol exposure on the over-confident top bin. `wheel_runner` threads `vix_level` into `build_dossiers`; no-op when `vix_level` is absent (missing-evidence semantics, like R6–R10). See `DECISIONS.md` D23.

This contract is enforced structurally by `tests/test_audit_invariants.py`, `tests/test_dossier_invariant.py` (R1–R10), `tests/test_r11_elevated_vol.py` (R11), and `tests/test_authority_hardening.py`. Never move or merge a change touching these without §2-owner (decision-integrity owner) sign-off.

### 9.3 Out of scope by design

Get explicit Operator consent before crossing any of these. The items that overlap §7 (EV bypass, reviewer upgrades, the dealer clamp, broker/OMS surface, branch-and-PR discipline) live there as invariants; the product-scope items unique to this list are:

- **No tick-level order flow / sub-minute features.** Theta v3 does not expose realtime stock quotes at this tier.
- **No non-S&P-500 universe and no non-wheel strategies.** Short puts + covered calls + timing-gated strangles only.

### 9.4 Fresh-session bring-up

1. Confirm the provider. Default is `bloomberg`; verify with:

   ```python
   from engine.wheel_runner import WheelRunner
   print(type(WheelRunner().connector).__name__)
   ```

   Expect `MarketDataConnector` in a Cowork sandbox. **Always log which provider was actually selected** — silent provider selection is a recurring bug source.

2. Sanity-check the data layer with the 5-ticker smoke test:

   ```python
   from engine.wheel_runner import WheelRunner
   df = WheelRunner().rank_candidates_by_ev(
       tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
       top_n=10, min_ev_dollars=-1e9,
       include_diagnostic_fields=True,
   )
   ```

   Runs in ~2 s. Five rows with non-null `ev_dollars`, `iv`, and `premium` means the Bloomberg CSVs + connector + EV engine path is healthy.

3. For any change touching the decision layer (`ev_engine.py`, `wheel_runner.py`, `candidate_dossier.py`): run the full test suite (`pytest tests/ -v`), not just the targeted file. Invariants are cross-cutting.

For machine/environment rehydration (cloning, Theta Terminal, regenerating local data), see `docs/LAPTOP_SETUP.md`. For sandbox-vs-laptop capability differences, see `docs/DATA_POLICY.md` §7.

### 9.5 Parallel-session mechanics

The contract is §2.4; this is the machinery behind it. The organising idea: **the cheapest cross-terminal communication is the communication you never need.** Give every task a disjoint surface and most coordination disappears — two terminals that cannot touch the same file cannot collide. A single allocator guarantees the disjointness up front; a CI gate catches the one drift humans miss (the decision-layer trio). This is the 2026-05 evolution of `DECISIONS.md` D15.

**The allocation cycle.**

1. **Allocate (allocator).** Post one cycle plan to the board: N task cards, each owning a disjoint set of files. Check the `owns` sets are pairwise non-overlapping *before* posting. Terminals do not choose tasks; they receive them. Allocation is a batch at cycle start, so terminals never block on the allocator mid-cycle.
2. **Pull (terminal).** Read your card. If anything is ambiguous, ask the allocator — do not improvise into a neighbour's surface.
3. **Branch (terminal).** In *your* worktree: `git worktree add ../swe-terminal-<x> -b claude/<slug> origin/main`.
4. **Work.** Edit only files in your card's `owns`; read anything in `reads`. Editing outside `owns` is a stop-and-report (§2.4).
5. **Document.** Write your card's worklog fragment (`python scripts/new_worklog.py <card-id> --title "..." --kind <kind>`, fill the fixed sections, then `python scripts/gen_worklog_index.py` — CI fails if the index is stale). Format spec: `docs/worklog/README.md`.
6. **PR.** One branch, one PR. Body follows `.github/pull_request_template.md` and links the worklog fragment. `git fetch origin && git rebase origin/main` before every push.
7. **Verify.** Re-run the acceptance check; confirm the diff stays inside `owns`; confirm decision integrity (§7) holds.
8. **Done-notice (terminal).** One compact comment on the board with the merge SHA; edit your live-state row to `done`.
9. **Reconcile (allocator) at cycle close.** Assign `Sn`/`D` numbers, merge the per-task CHANGELOG lines / DECISIONS pointers, close the cycle, post the next plan.

**The task card.** Posted as one fenced block on the board so cards are machine-scannable:

```yaml
- id: C7-A                      # cycle 7, Terminal A — unique by construction
  terminal: A
  title: Sever verbal news from the EV path
  owns:                         # THE key field. Disjoint across ALL open cards.
    paths:   [engine/news_sentiment.py, tests/test_news_severance.py]
  reads:    [engine/wheel_runner.py]   # may read; must NOT edit
  decision_layer: false         # true iff owns ev_engine/wheel_runner/candidate_dossier
  depends_on: []                # card ids that must merge before this starts
  acceptance: "sentiment_multiplier returns 1.0 for every (sentiment, n) input"
  worklog: docs/worklog/C7-A.md
```

The allocation invariant: the union of `owns.paths` over all open cards has no overlap, and at most one open card has `decision_layer: true`.

**The decision-layer hard gate (CI-enforced).** The trio — `engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py` — is edited one terminal at a time. `scripts/check_lane_claim.py` runs in CI on every PR: if the diff touches a decision-layer file, the PR description must carry a `lane-claim` block naming that file, or the build fails:

```
<!-- lane-claim
files: engine/wheel_runner.py
board: <link to your claim comment on the board>
-->
```

A decision-layer path counts as claimed if its exact repo-relative path appears inside the block. The gate is deliberately narrow: it fires only on the trio; non-decision-layer lane ownership stays advisory (the allocation + the board are the guarantee there). The gate does not itself prove no other open PR holds the same file — that mutual exclusion is the allocator's invariant. Run it locally before pushing: `python scripts/check_lane_claim.py --base origin/main --claim-file <pr-body.md>`.

**Ripples.** The one cross-terminal signal: a change to a shared interface whose behaviour another terminal's in-flight card assumes. Post a `ripple:` notice on the board naming the interface and the affected card; the allocator routes it (pauses/re-sequences the affected card, or adds a `depends_on`). Never terminal-to-terminal. If two cards genuinely need the same file, the allocation was wrong — the allocator re-cuts the cards.

**Worktrees + per-terminal env (mandatory).** Every terminal works in its own git worktree; the primary clone is reserved for the allocator, Strategists, and orchestration — no executor runs in it. Two terminals in one working tree share one index, one HEAD, one checked-out branch — one terminal's `checkout`/`rebase` silently moves the other's branch and leaks uncommitted edits. Source your per-terminal env in every shell (`source scripts/setup-terminal.sh <letter>`): it sets a per-terminal `SWE_API_PORT`, `COVERAGE_FILE`, and `PYTEST_CACHE_DIR` so concurrent `pytest` / `engine_api.py` runs stay isolated.

**The board.** The pinned issue body carries a live-state table (one row per terminal, edited in place); the comment log is the durable append-only audit trail. Keep comments compact: a claim names the card, `owns`, `decision_layer` flag, and branch; a done-notice gives the PR, merge SHA, and acceptance result; a ripple names the interface and the affected card.

**Numbering.** Scenario (`Sn`) and decision (`D<N>`) numbers are allocated **at merge**, as max(merged in the canonical file) + 1, by whoever merges. In-flight work is keyed by branch + PR number only. Never claim a global number at work-start; never hardcode one.

**Where to document.** One home per kind of record: the learning record (what you tried / what worked / what didn't / how you fixed it) goes in the worklog fragment — the canonical place; the PR body links the fragment rather than re-typing it; a structural decision + rejected alternatives becomes a `DECISIONS.md` entry that links the fragment — only when you actually made one; the one-line "what shipped" is a `CHANGELOG.md` bullet reconciled at merge. The old `docs/USAGE_TEST_LEDGER.md` monolith is **frozen** — never reopen it.

**Magnet files.** The registries touched by nearly every task (`FILE_MANIFEST.md`, `docs/worklog/INDEX.md`, and the temporal docs `CHANGELOG.md` / `PROJECT_STATE.md` / `DECISIONS.md` / `ROADMAP.md`) are the main merge-contention points. The allocator may name one `FILE_MANIFEST.md` owner per cycle and reconciles the temporal docs at cycle close. On any conflict in these files, resolve mechanically per §6 (union of rows + regenerate the index) — never `git checkout --theirs` (it once silently reverted a 12-doc archive move).

**Allocator duties and limits.** The allocator allocates and verifies; it does not hand-code tasks (if it catches itself implementing, it turns the work into a card). It verifies live state before acting — the board is truth for in-flight work, `git log origin/main` is truth for what merged, open PRs + their CI status via GitHub. It merges PRs when CI is green (squash), holds a docs-align PR that rewrites magnet docs until the code PRs it describes have merged, and turns a stale-base rebase into a card rather than hand-editing. A sandboxed allocator session may lack numpy/scipy/pandas — it relies on CI for the test suite; the stdlib guards (`check_manifest_coverage.py`, `gen_worklog_index.py --check`, `check_doc_currency.py`, `check_lane_claim.py`) do run locally.

**Recurring hazards (learned the hard way).** Shared working tree → corruption on the next checkout (use worktrees). Unclaimed decision-layer edit → silent supersession (now a CI failure). Self-selected duplicate work → the same feature built twice (allocator allocates). Number collision → merge-time assignment. Stale `main` → fetch immediately before reasoning about main. Bulk `--theirs` on a magnet file → silently reverted concurrent work (union + regenerate instead).

### 9.6 Onboarding and read order

`CLAUDE.md` is auto-loaded on every Claude Code session (and re-injected after `/compact`); it points here. Any AI agent — Claude Code, OpenAI Codex, Cursor, GitHub Copilot Agent, Aider, or otherwise — entering this repo follows the same canonical read order:

1. **This document**, in full.
2. **`PROJECT_STATE.md`** — what is authoritative right now, what is work-in-progress, what is deprecated/phantom, and where the live documentation drift is. Read this before assuming any file or pyproject entry is current.

For a fast "where does X live / what tests cover Y / what is authoritative for Z" lookup, `docs/REPO_MAP.md` routes each question to the one owning doc. The fuller routing table is §9.11.

**What you can change without asking:**

- Module-level docstrings; per-function docstrings.
- New tests under `tests/` that exercise existing public surface.
- A worklog fragment under `docs/worklog/` recording your task (§9.5 "Where to document").
- Documentation (`docs/`, `*.md` at repo root) **except** `CLAUDE.md` and this document, which the Operator governs (§7 "Governance of this list"; agents propose changes, they do not write them unilaterally).
- Files under your active feature branch (never `main`).
- Updates to `CHANGELOG.md` / `PROJECT_STATE.md` / `ROADMAP.md` / `DECISIONS.md` that reflect the work you just did — keep these in sync with the code as you ship (subject to the DECISIONS rule in §2 Leg 3: propose, don't self-author new constraints).

**What requires explicit ask:**

- Any change under `engine/ev_engine.py`, `engine/wheel_runner.py`, or `engine/candidate_dossier.py`.
- Any new data-provider class, advisor, or chart provider.
- Adding broker / OMS / order-routing surface (§7 Brokerage safety).
- Editing `CLAUDE.md` or this document.
- Committing to `main` directly (branch + PR for everything, even one-line typos).
- Refreshing the tracked Bloomberg CSVs under `data/bloomberg/` (`sp500_*.csv`, `treasury_yields.csv`) — these are data commits with audit value (see `ROADMAP.md` C1 — the track-vs-gitignore decision is open).

When in doubt, ask before doing.

### 9.7 Commit and pull-request standard

The repo's history is its second test suite. A commit message that explains *why* (not just *what*) lets the next agent — human or LLM — recover context months later from `git log` alone.

**Commit message format:**

```
type(scope): one-line summary in imperative voice

Changed:
- bullet (the factual list of what's different; files + behaviour)

Why:
- bullet (the motivation, the constraint, the past incident)

Tested:
- bullet (the exact command you actually ran — copy-pasteable; flag manual verification as manual)

Tried but rejected:
- bullet (the alternative that didn't work and the reason — the single
  highest-value section for keeping context; omit if you tried none)

Unresolved:
- bullet (anything you noticed but didn't fix — better here than a TODO that rots)

AI handoff:
- bullet (a hint for the next agent — what to look at next, what this
  exposes, what test to add later)
```

**Omit a section entirely if it's empty** — don't write "N/A". An empty section is noise; a missing section is silence. If `Changed:` can't be listed in 3–5 bullets, the commit is probably too big.

**Allowed `type` values:** `feat` (new capability) | `fix` (bug fix) | `docs` (documentation only) | `refactor` (no behaviour change) | `test` (test-only) | `chore` (hygiene: gitignore, CI, dependencies, formatting) | `audit` (audit-cycle work that pins invariants) | `perf` (performance, no behaviour change).

**`scope`** is the smallest module the commit really touches (e.g. `wheel_runner`, `ev_engine`, `theta_connector`, `PROJECT_STATE`); for a genuinely cross-cutting change use a category: `engine`, `scripts`, `docs`, `tests`, `infra`.

**PR format:** same five-section body plus a header summary (`## Summary` / `## Changes` / `## Why` / `## Tests` / `## Tried but rejected` / `## Unresolved` / `## AI handoff`), per `.github/pull_request_template.md`. PR titles follow the same `type(scope): summary` format, under 70 characters. A PR touching the decision-layer trio must carry the `lane-claim` block (§9.5).

**Branch naming:** kebab-case slugs, 2–4 words, descriptive.

| Convention | Use |
|---|---|
| `claude/<short-slug>` | Default for Claude Code work |
| `codex/<short-slug>` | OpenAI Codex / Cursor agent work |
| `feat/<short-slug>` | Human-driven feature work |
| `fix/<short-slug>` | Human-driven bug fix |

> **[FLAGGED CONFLICT — carried unresolved.]** The superseded CONTRIBUTING doc additionally allowed `agent/<short-slug>` for AI-agent work and used `feature/<short-slug>` (not `feat/`) as the human convention. Both conventions are preserved here until the Operator picks one.

**Never edit `main` directly.** Even a one-line README typo fix gets a branch + PR. The audit trail is part of governance.

**AI agent attribution.** When an AI agent commits, use a `Co-Authored-By` trailer in the commit body (e.g. `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`) so the history reflects who shipped what; the model name helps later debugging when a class of bugs traces to a model generation.

> **[FLAGGED CONFLICT — carried unresolved.]** The Operator's standing instruction to Claude agents has been to **omit** the model Co-Authored-By trailer from commits and PRs, which contradicts the paragraph above. Both positions are preserved here until the Operator picks one.

**Anti-patterns:** "WIP"/"stuff" commit messages; seventeen "fix typo" commits (squash before opening the PR — each commit on `main` should stand alone); `--no-verify` to skip hooks (fix the underlying issue instead); force-pushing to a shared branch (branches with open PRs are shared — rebase locally and push as a new commit, or open a follow-up PR); committing files containing secrets (`git diff --staged` before commit catches anything `.gitignore` misses).

The bare minimum: every non-trivial commit should answer **why** in the body. Code comments rot; commit messages are immutable history.

### 9.8 Engineering practice

**Prerequisites:** Python 3.11+, Node.js 18+ (only if touching the Next.js dashboard), Git. Install runtime + test deps with `pip install -r requirements.txt` plus `pytest pytest-cov hypothesis ruff pre-commit`, then `pre-commit install`; `pip install -e ".[dev]"` is what CI uses and is equivalent. For Theta Terminal bring-up on a new machine (not needed on the default committed-Bloomberg-CSV path), see `docs/LAPTOP_SETUP.md`.

**Style:** PEP 8 + Ruff defaults; Ruff is the formatter. Type hints on all functions; the codebase is mypy-strict in scope. Docstrings on public functions (Args / Returns / Raises). Keep functions focused — past ~80 lines, ask whether it's two functions. In docs, prefer symbol names over `file.py:line` references — line numbers drift. Greek units are pinned in `docs/GREEKS_UNIT_CONTRACT.md`; read it before touching any Greek code.

**Testing:** tests live in `tests/`, pattern `test_*.py`, with descriptive names (`test_atm_call_delta_near_half`, not `test_1`). For decision-layer changes, add a test that pins the invariant, not just one that covers the new code path; the launch-blocker family (`test_audit_*`, `test_dossier_invariant`, `test_authority_hardening`, `test_launch_blockers`) is what enforces the hard EV invariant — new invariant tests belong there. See `TESTING.md` for the full taxonomy and the "what to run when you touch X" map. Always run the **full** suite (`pytest tests/ -v`) for any change touching the decision-layer trio.

**Lint (CI hard-fails on dirty lint):**

```bash
ruff format --check .
ruff check .
```

**CI:** all CI jobs must pass before merge (environment validation, lint & type check, security scan, test suites, quantitative validation, integration tests, manifest coverage; the exact job list evolves — the rule is *all green*, not a fixed count).

**Keeping `FILE_MANIFEST.md` in sync:** the manifest-coverage CI job most commonly fails on docs-heavy PRs — a new file lands without a manifest row. `python scripts/sync_manifest.py` runs the same read-only scan as `check_manifest_coverage.py`; with `--fix` it appends rows for missing files into an "Untriaged additions" tail section with placeholder purposes — your follow-up is to move each row under the correct directory section and write a real description. Orphans (rows matching no tracked file) are flagged but never auto-removed; they usually signal a moved or renamed file and need human judgement.

**License:** contributions are licensed under the MIT License.

### 9.9 Model governance

This is a **single-operator project**: the enterprise roster the original governance framework named (Model Committee, Quant Team, Risk Team, on-call paging) was an aspirational template, not the operating reality. Every role is performed by the Operator, backed by the automated CI gates (the test suite, the decision-layer lane-claim gate, and the decision-integrity invariant checks). "Committee approval" means the Operator's own review recorded in `DECISIONS.md` and the PR trail. The role split survives only to show which concern an activity serves: development & validation (the "quant" hat), risk & limits — including configuring the R7–R11 caps and their locked defaults (the "risk" hat), and change approval (the "committee" hat).

The still-binding standards carried from that framework:

- **Model risk tiers:** Tier 1 high (option pricing, VaR, position sizing), Tier 2 medium (volatility estimators, regime detection), Tier 3 low (technical indicators, data transformations).
- **Documentation:** every model includes its mathematical specification, input parameter definitions and valid ranges, output specifications and bounds, assumptions and limitations, and a reference to an academic or industry source. Model-function docstrings carry a "Mathematical basis" section plus Args (with valid ranges/units) / Returns (with bounds) / Raises / References.
- **Testing:** unit tests required at every tier; property tests and textbook validation required for Tier 1 (recommended Tier 2). Minimum coverage 70% line, 80% for Tier 1 models.
- **Initial validation before deployment:** textbook verification (results match published examples), boundary testing at parameter limits, property-based testing of invariants, and point-in-time testing (no lookahead bias).
- **Back-testing standards:** minimum 3 years of historical data; walk-forward validation with embargo periods; out-of-sample testing required; transaction costs included; slippage assumptions documented.
- **Change classification:** material (formula change, new model) needs the Operator's committee-hat sign-off recorded in `DECISIONS.md`; significant (parameter recalibration) needs quant-hat review; minor (bug fix, optimization) needs peer/CI review. Emergency fixes get immediate review, enhanced monitoring, and a completed formal review within 48 hours.
- **Input data sanity:** prices non-negative and not stale; volumes non-negative integers; IV in a plausible range; Greeks within theoretical bounds. Feature computations log their source-data timestamp and support audit replay.
- **Audit trail:** model executions log timestamp, inputs, outputs, model version, and trigger.
- **Approved textbook references:** Hull; Wilmott; Natenberg; Sinclair ("Volatility Trading"); Taleb ("Dynamic Hedging").

The original framework's live-ops scaffolding (real-time monitoring thresholds, paging escalation levels, the model inventory and review-date tables) was aspirational per its own practice note and is not carried as binding; if live monitoring ever becomes real, it returns through a §7 amendment, not by resurrecting the template.

### 9.10 Named terminals

If the Operator addresses a session by a **role name**, it adopts that role and reads its runbook in full before acting:

| Operator says | You are | Read first |
|---|---|---|
| **"You are responsible for the Dashboard"** (or `dashboard`) | the **Dashboard** terminal — owns the live IBKR portfolio viewer (`/portfolio` + `/cockpit` + `/terminal`); on **"update"** it refreshes all live data | **`docs/DASHBOARD_TERMINAL.md`** |

The Dashboard terminal is strictly read-only (§7 Brokerage safety): it pulls the Operator's IBKR account (cloud connector / IB Gateway / Flex Web Service) and regenerates the viewer's gitignored `data_processed/ibkr/` files. **Other terminals:** leave the dashboard, `data_processed/ibkr/`, and the `/api/portfolio/*` pipeline to it.

### 9.11 Where to look next

| For… | Read |
|---|---|
| What is authoritative / WIP / deprecated right now | `PROJECT_STATE.md` |
| Fast where/what/authoritative router (one stop) | `docs/REPO_MAP.md` |
| Per-module purpose + decision-layer role | `MODULE_INDEX.md` |
| Test taxonomy + launch-blocker subset | `TESTING.md` |
| Every tracked file by purpose — grep, don't read | `FILE_MANIFEST.md` |
| Data layer (tiers, provider matrix, refresh, sandbox caveats) | `docs/DATA_POLICY.md` |
| Launch-blocker invariants before merging | `docs/LAUNCH_READINESS.md` |
| Greek units (canonical) | `docs/GREEKS_UNIT_CONTRACT.md` |
| TradingView wiring (engine bridge + analyst workspace) | `docs/TRADINGVIEW_INTEGRATION.md` (then `docs/TRADINGVIEW_MCP_INTEGRATION.md` for MCP design) |
| Commit / PR format | §9.7 of this document |
| Parallel-session mechanics | §2.4 + §9.5 of this document |
| *Why* a structural choice was made | `DECISIONS.md` |
| What's intentionally next | `ROADMAP.md` |
| What shipped recently | `CHANGELOG.md` |
| Usage tests + every backtest / verification record | `docs/worklog/INDEX.md` (per-task fragments; `docs/USAGE_TEST_LEDGER.md` is the frozen pre-2026-05-29 monolith) |
| Rehydrating data on a fresh machine | `docs/LAPTOP_SETUP.md` |
| Theta data refresh | `docs/THETA_USAGE.md` §20 (laptop bring-up) + the §18 pipeline |
| Dashboard terminal — live IBKR portfolio viewer + the "update" refresh | `docs/DASHBOARD_TERMINAL.md` |
