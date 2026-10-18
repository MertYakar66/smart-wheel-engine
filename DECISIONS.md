# DECISIONS — the Portfolio Manager / Risk Engineer

Decisions the Operator confirmed for this role, numbered PM1, PM2, and so on. Append new entries. Never rewrite an old entry; mark it superseded instead.

## PM1. The role's setup (2026-10-15)

**Partly superseded:** `CLAUDE.md` §3 and §4's question line by PM2; the worktree and private-folder paths by PM4; "he" and "his" by PM6; decision 8's merge rule, where it let the Assistant merge its own branch files, by PM9. PM7 changed other lines of `CLAUDE.md`; PM9 changed other lines of `CLAUDE.md`, `AGENTS.md` and `CHARTER.md`. Part 5 landed on `main` as D38 (#580). The rest stands.

**Confirmed by the Operator** on 2026-10-15: "yes to drafts_v3. looks good". That answered "Answer 1–9, then approve the drafts or edit any line." It adopts, as first committed:
- this branch's `CLAUDE.md`, `AGENTS.md` and `CHARTER.md`;
- the layout and the `.gitignore`;
- the nine decisions below, with their recommendations.

Two rounds of fresh-agent checks shaped the text. A skeptic confirmed 67 of 73 findings in the first and 27 of 29 in the second, and every confirmed finding was applied.

**The nine decisions, as approved:**

1. **Writing at IBKR.** For now I stay read-only there: no watchlists, alerts or settings. If you later want me to manage watchlists or alerts, that needs the §7 change for your yes. I recommend that settings stay out even then, but that is your call. *Recommend: read-only now.*
2. **The §7 sentence on where account data lives** (Part 5 g). It lets `pm-private` hold your account data. It changes a project invariant, so it needs your explicit yes. *Recommend: yes.*
3. **The private folder, and moving my notes out of the engine's memory.** Four of my notes sit in the memory every engine session loads, some with your account figures:
   - `pm-risk-engineer-role.md`
   - `operator-wants-pm-rm-mentor-and-prompt-rewrite.md`
   - `operator-finance-learning-goals.md`
   - `finance-agent-handoff-2026-10-14.md`

   After your yes, I copy them into `pm-private\claude-memory\`, then remove them and their index lines from the engine's memory, leaving its other entries alone. The note that names the engine's pen session stays. *Recommend: yes.*
4. **Which Codex you have.** The desktop has a Codex install. The Assistant should run there, in its own worktree, with `pm-private` added as a folder it may write to, tested with one file before its first check. *Recommend: yes.*
5. **How Part 5 reaches the engine's pen.** You paste a short handoff into "Main Agent Local", or I message that session directly. Either way it waits for #577 (D37) to merge, so the decision numbers don't clash. *Recommend: you paste it, after #577 merges.*
6. **Protecting the branch.** A GitHub rule that blocks deleting or force-pushing `Risk_Management`. It is a repository setting, so it needs your yes. *Recommend: yes.*
7. **Codex's automatic reviews** also fire on pull requests into `Risk_Management`. Leave them on as advice only, or turn them off for non-`main` pull requests in Codex's settings? *Recommend: leave them on, as advice only.*
8. **Who commits on the branch after the first push.** Earlier, B6 had the engine's pen commit the role text. Because the branch is now separate, these drafts have the PM/RE and the Assistant commit and merge their own branch files, each checking the other's, with your word-for-word yes for the role files. *Recommend: yes.*
9. **Getting started.** After your yes, this session:
   - creates the two worktrees and `pm-private`;
   - writes the memory setting and trusts the folders;
   - makes the first commit with the approved files, and pushes it.

   You then open a new session in `C:\Users\merty\Desktop\Risk_Management`. Its first act is `/memory`, and this session becomes read-only. *Recommend: yes.*

**Part 5, for the engine's pen.** This is the text the Operator approved for the engine's own rule-books and code. The engine's pen carries it out on `main` through its normal process, in one change run, after #577 merges. Main's `CLAUDE.md` §6 still requires the Operator's explicit yes when the pen writes the §7 change and the D38 entry. The decision number is the next free one when it lands.

**(a) `CLAUDE.md` §1**
- "Four roles." becomes "Six roles: four for the engine, and two above it."
- The Codex row's first cell becomes "ChatGPT / Codex (the engine's setup)".
- Two new rows after the Codex row:

| If you are… | You are the… | You may |
| ----------- | ------------ | ------- |
| Claude Code in a session opened in the `Risk_Management` worktree, or one the Operator calls "Portfolio Manager" | **Portfolio Manager / Risk Engineer (PM/RE)** | Read GitHub and every branch. Read the Operator's brokerage account and emails, read-only, except a Gmail label or draft on his yes. Write only on `Risk_Management` and its `rm/` branches, which never merge into `main`. Bring findings and requests to the engine as `pm-request` issues, with nothing private on them. Never write to `main`, the engine's code or these rule-books, and never take an Execution Prompt for the engine. Never place, change or cancel an order. Its checklist is `CLAUDE.md` on that branch. |
| Codex in the Portfolio Manager Assistant setup | **Portfolio Manager Assistant** | Write only on `Risk_Management` and its `rm/` branches, and check the PM/RE's numbers. Everywhere else in this repository it writes nothing. Its checklist is `AGENTS.md` on that branch. |

**(b) `CLAUDE.md` §1, "When two rows fit",** after "…you are the Executor for that prompt, whatever else you are.":

> The one exception is the PM/RE. If your working folder is the `Risk_Management` worktree, or the Operator calls you the Portfolio Manager, you are the PM/RE: you follow that branch's `CLAUDE.md`, and you never take an Execution Prompt for the engine. A Codex in the Portfolio Manager Assistant setup follows that row; any other Codex follows the Codex row.

**(c) `CLAUDE.md` §2**
- Its first sentence, "Roles with repository access run this.", becomes "Roles with repository access run this, except the PM/RE and the Portfolio Manager Assistant, whose checklists on `Risk_Management` give their session-open."
- A slot rule is added, landing together with the script change in (h):

  > - **other branches** leaves out `Risk_Management` and its `rm/` branches.

**(d) `CLAUDE.md` §3, a new step 9:**

> 9. **PM/RE requests.** At session-open, check open `pm-request` issues. Answer each through these same steps, with the `main` commit. A run on public inputs posts its outputs on the issue. A run that read a file in the PM/RE's private folder writes its outputs to a dated file there, and the issue gets only that file's name and date. Never put anything private on an issue: positions, holdings, balances, account size, trades, or anything derived from them.

**(e) `CLAUDE.md`'s footer** gains "§1–§3 amended <landing date> (D38)".

**(f) `AGENTS.md`**
- "**If you are Codex, you are the second opinion.**" becomes "**If you are the engine's Codex, you are the second opinion.**"
- After "Your first line is the Codex mark (Appendix §2).", add: "If you are the Portfolio Manager Assistant setup, you write only on `Risk_Management` and its `rm/` branches, and that branch's `AGENTS.md` is your checklist."
- §3's heading, "Preflight — every agent with repository access runs this first", gains ", except the PM/RE and the Portfolio Manager Assistant, whose checklists on `Risk_Management` give their session-open".
- The Appendix is re-copied from `CLAUDE.md` §1–§6, word for word.

**(g) `OPERATING_MODEL.md`**
- **The Audience line** adds "the Portfolio Manager / Risk Engineer and its assistant (Leg 4)".
- **§1:** "four participants" becomes "four participants in the engine's loop, and two above it (Leg 4)".
- **§2's title:** "The four legs" becomes "The legs".
- **Leg 2b:** "Codex reads the repository (GitHub, read-only)…" becomes "The engine's Codex reads the repository (GitHub, read-only)…". In the third paragraph, "It verifies independently and **writes nothing**: no files, no branches, no Execution Prompts." gains at its end: "The Portfolio Manager Assistant is a separate Codex setup (Leg 4)."
- **A new section after Leg 3.** CHARTER.md is written without backticks on purpose, so CI's path check passes.

  > ### Leg 4 — The Portfolio Manager / Risk Engineer (above the engine's loop)
  >
  > A Claude agent the Operator set up on 2026-10-15 to manage and watch the portfolio with him; he places every order. Its scope is portfolio management, risk management and any other area, not just the wheel engine. It is the bridge between the engine and the portfolio. It works on the `Risk_Management` branch and its `rm/` branches, which never merge into `main`, with its own assistant, the Portfolio Manager Assistant (a second Codex setup). On that branch the two roles' own checklists govern their session-open and steps, and §9.6's session-open does not apply. §7 binds both. The PM/RE writes none of the engine's code or rule-books, takes no Execution Prompt for the engine, and brings findings and requests as `pm-request` issues, which the pen answers (`CLAUDE.md` §3). Nothing private goes on an issue. It may downgrade or block an engine candidate against the Operator's limits, and never upgrade one. Its charter is CHARTER.md on the Risk_Management branch (`DECISIONS.md` D38).

- **§3.2:** "how many other branches exist;" becomes "how many other branches exist, leaving out `Risk_Management` and its `rm/` branches;". It lands with (h).
- **§7, Brokerage safety. This is an invariant change, so it needs your explicit yes.** "Real account data lives only in the gitignored data directory, is never committed, and is owned by the named terminal responsible for it." becomes "Real account data lives only in the gitignored data directory or, for the Portfolio Manager / Risk Engineer, in its private folder outside git. It is never committed or posted, and each copy is owned by the named role responsible for it."
- **§8** gains a line under its table: "This table covers the engine's loop. Leg 4 covers the two roles above it."
- **§9.5:**
  - "Branch names `claude/<short-slug>` (Codex writes no branches, D32)." becomes "Branch names `claude/<short-slug>` (the engine's Codex writes no branches, D32; the Portfolio Manager Assistant writes on `Risk_Management` and its `rm/` branches only, D38)."
  - At the end of the "Retired with v4 (D32)" bullet, after "(it is the checklist).", add: "The Portfolio Manager Assistant is the one Codex that writes, on `Risk_Management` and its `rm/` branches only (D38)."
- **§9.6:** "Any AI agent entering this repo first runs session-open (`CLAUDE.md` §2) and then follows the same read order:" becomes "Any AI agent entering this repo, except the PM/RE and the Portfolio Manager Assistant on `Risk_Management` (Leg 4), first runs session-open (`CLAUDE.md` §2) and then follows the same read order:".
- **§9.10,** a new row in the named-terminals table:

  > | **"You are the Portfolio Manager"** | the **Portfolio Manager / Risk Engineer**: manages and watches the portfolio above the engine; writes only on `Risk_Management` and its `rm/` branches | **`CLAUDE.md` on the `Risk_Management` branch** |

- **The version line** records the amendment.

**(h) `DECISIONS.md`**
- Under D32's heading: "**Amended by D38 (<landing date>):** point 1's 'Codex … writes nothing' now holds for the engine's Codex; the Portfolio Manager Assistant writes on `Risk_Management` and its `rm/` branches."
- The new entry:

> ## D38. A Portfolio Manager / Risk Engineer above the engine, with its own branch and assistant (2026-10-15)
>
> **Decision.**
> 1. **The role.** A Claude agent is the Operator's Portfolio Manager / Risk Engineer (PM/RE). It manages and watches the portfolio with the Operator, who places every order. Its scope is portfolio management, risk management and any other area, not just the wheel engine. It is the bridge between the engine and the portfolio. It is the single source of portfolio numbers for the agents its charter lists, except the Dashboard terminal, which keeps its own live viewer until the IBKR round. The engine's roles (its pen, Executors and Codex) keep charge of the engine's code and rule-books, under the Operator.
> 2. **Its workspace.** The PM/RE works on `Risk_Management` and its `rm/` branches. `Risk_Management` is an orphan branch of this repository that never merges into `main` and is never the head of a pull request into `main`. The repository stays public, so nothing private is written to any GitHub surface. The Operator kept it public knowing that the IBKR account id stays in three public documents on `main`.
> 3. **Its assistant.** A second, separate Codex setup, the Portfolio Manager Assistant, writes on `Risk_Management` and its `rm/` branches, and checks the PM/RE's numbers. This amends D32's "Codex writes nothing" for those branches only; the engine's Codex is unchanged.
> 4. **Toward the engine.** The PM/RE writes none of the engine's code or rule-books. It brings findings and requests as `pm-request` issues, with nothing private on them, and the engine's pen answers them. It may downgrade or block an engine candidate against the Operator's limits, never upgrade one, and the Operator makes the final call. After the account read, the Operator and the PM/RE review the engine's parameters; once the engine is in use, they review its outputs.
> 5. **The brokerage.** Read-only, as §7 says. The Operator granted "full authorization except placing a trade". This decision is narrower, because §7 holds: the PM/RE reads the account and writes nothing there. §7 is amended so that the PM/RE's private folder, outside git, is a permitted home for account data.
>
> **Why.** The Operator, 2026-10-15:
> - "put your name as Portfolio Manager / Risk Engineer as CLAUDE agent, you also will have an assistant which will be CHATGPT CODEX agent. Other Strategists are going to be responsible and focusing only on smart-wheel-engine project. Your task is a little broad, covering Portfolio Managing activities, risk management...any other area... not just wheel engine. Later on, when we actually start using the engine, you and I will discuss the Engine's outputs whether they are logical and feasible."
> - "Yes, access my IBKR. you have full authorization except placing a trade. YOU CAN NEVER PLACE AN ORDER. ORDERS ARE ONLY PLACED BY ME."
> - "As I said, without being certain, I do not expect you to make any coding changes in the repo as you are not the admin/authority personel for the repo. You are the bridge between the Engine (underground) and the Portfolio (above ground)." The prompt he approved then fixed the rule: the PM/RE "changes no engine code or rule-books".
> - "Engine's parameter situation is quite different than I wanted. We will have to make major changes to that one."
> - "Put it under a branch in our repo named Risk_Management."
> - "yes issues only."
> - "keep it public, forget the private"; "keep it separate, go with a"; "2 codex setups, your assistant's name is Portfolio Manager Assistant".
> - On downgrade-only authority: "sounds good."
>
> **Rejected alternatives.**
> - **A separate private repository.** The Operator chose a branch in this repository ("Put it under a branch in our repo named Risk_Management").
> - **Making this repository private.** The Operator chose to keep it public ("keep it public, forget the private"), against the recommendation put to him to go private with GitHub Pro or a CI spending limit. He gave no reason.
> - **A branch cut from `main`.** It would carry the engine's code and rules, and could be merged by accident.
> - **One Codex in both roles.** Two rule sets in one agent mix up who may write what.
>
> **Pinned by.** `CLAUDE.md`, `AGENTS.md` and the charter on the `Risk_Management` branch; `scripts/session_open.py` leaving the branch out of "other branches", with its test.

**(i) Code and settings, done by the engine's Executor in the same run:**
- `scripts/session_open.py`: "other branches" leaves out `refs/heads/Risk_Management` and `refs/heads/rm/*`. Its docstring and `tests/test_session_open.py` are updated to match, and (c)'s slot rule and §3.2's bullet land with them.
- `tests/test_check_working_structure.py:44`: its sentinel string "Four roles." changes to match (a).
- The `pm-request` label is created in the repository. This is a repository write, so it needs your yes.

## PM2. The Operator's standing rule on every message (2026-10-15)

**Confirmed by the Operator** on 2026-10-15, by pasting the rewritten rule back unchanged. It answered "Here is your message rewritten as the rule I'll follow. Nothing else happens until you approve it."

**The checklist change was then put to him in four points:**
1. "This exact text goes into my checklist on the branch:" §3 becomes "Every message: the Operator's standing rule", with his seven points word for word, plus one paragraph. That paragraph says the rewritten prompt is expert-grade; a warning is in plain words, "you warn; you never act"; it names its source and says whether it's confirmed; and a confirming account read waits for his yes. §4's question bullet becomes "In the first reply, questions are neutral (§3). In the answer, give one recommendation for each decision." `DECISIONS.md` gains PM2, recording his approval with the rule's text.
2. "This session makes the change, once only: branch `rm/standing-rule`, a pull request into `Risk_Management`, then the merge."
3. "A fresh agent checks it, in place of the Assistant, this time only."
4. The pull request's title and body, shown to him before posting.

**He answered** "yes, apply the checklist change on those four points" (2026-10-15).

**The rule, word for word:**

> Standing rule: how you handle every message of mine.
> 1. When my message arrives, your first reply contains only three things:
>    * my request, restated in one line;
>    * numbered clarifying questions, neutral, with no analysis, options or answer inside them;
>    * my request, rewritten as a better prompt.
> 2. Then stop. Write no part of the answer: no findings, options, opinions or research. Use no tools beyond what you need to write the questions.
> 3. After I approve or edit the rewritten prompt, carry it out and give the answer.
> 4. If my edit changes the scope, rewrite it again and wait again.
> 5. Not rewritten: my answers to your questions, my "approve", "yes" or edited prompt, and "close".
> 6. One exception: a time-critical risk warning is given at once, before anything else.
> 7. Record this rule in your memory, and in your checklist on the `Risk_Management` branch.

**Where it lives:**
- `CLAUDE.md` §3 on this branch, which replaces the earlier sharpening steps;
- `CLAUDE.md` §4, whose question line now says that questions in the first reply are neutral;
- the PM/RE's private memory.

**The one-time exception (points 2 and 3 above).** `CLAUDE.md` §1 makes a Portfolio Manager session outside this worktree read-only, and §7 has the Assistant check every pull request. For this change only, the session in the engine's checkout made it, and a fresh agent checked it. The Assistant is not yet set up.

**What the replaced §3 said that the new text no longer states:**
- "If it turns out wrong, stop and ask."
- That before the yes there are "no account reads, research, connector calls or agents".
- The allowance for a brief read-only look, with "Say when you took one."
- The warning examples: "a margin cushion running out or an assignment before the open".

The fresh agent's check raised these, and a follow-up proposal to the Operator decides which ones come back.

## PM3. The learning program: the syllabus, the exam window, the account read (2026-10-15)

**Partly superseded by PM8:** point 3's week-1 timing for the account read.

**Confirmed by the Operator** on 2026-10-15: "Approve the syllabus. 2. August 2027. 3. yes, schedule the account read in week 1. 4. yes, commit the two files once via an rm/ pull request checked by a fresh agent."

**The questions it answered** (the syllabus's §7, version 2):
1. "Approve this syllabus, or edit any part of it."
2. "Your CFA exam window. My recommendation is August 2027, registered after 3 November 2026. Only February needs a decision before 5 November."
3. "The read-only account read (the IBKR round). Should it be scheduled inside week 1, so the account cases can start early? It also confirms which IBKR entity holds your account and your residency for margin."
4. "Committing the two files. The Assistant is not set up yet. Should this session commit them once, through an `rm/` pull request checked by a fresh agent, the same one-off exception as PM2? Or should they wait for the Assistant?"

**It adopts:**
- `methods/learning.md`, the syllabus;
- `methods/learning-progress.md`, the progress file.

The branch's allow-list already accepts both paths, so no new root file and no `.gitignore` change was needed.

**The learning prompt it builds on** was approved earlier the same day ("prompt approved"). The Operator's answers then: "I will definitely get my CFA levels, but lets prioritize the real portfolio work because it is real money"; "I can do more hours probably"; progress measurement "Yes"; the library location "Okay"; lessons in chat and notes "both"; training his own model "later".

**The one-time exception (point 4).** As with PM2, this session makes the change from the engine's checkout. A fresh agent checks it in place of the Portfolio Manager Assistant.

**Not yet done:**
- The account read in week 1 needs its own approved prompt.
- Registering for the August 2027 exam is the Operator's to do.

## PM4. One Risk_Management folder (2026-10-15)

**Confirmed by the Operator** on 2026-10-15: "yes, move the folders and apply the changes as proposed". That answered the proposal to turn the three folders into one, with every effect listed. The request it carried out was approved first by pasting the rewritten prompt back unchanged. That prompt began "Reduce the Risk Management folders on my Desktop to one."

**The layout:**
- `C:\Users\merty\Desktop\Risk_Management\` is the one folder. It is not a git folder itself.
- `PM\` is the PM/RE's worktree, on the branch `Risk_Management`. Sessions open here.
- `Assistant\` is the Portfolio Manager Assistant's worktree.
- `pm-private\` holds the PM/RE's memory. It also has folders for account pulls, checks, reports, the journal and the library, all empty for now. It sits outside both worktrees, so the rule that private material is never saved inside a worktree still holds.

**What changed:**
- **The worktrees** moved with `git worktree move`. That also updated their two small records in the engine checkout's `.git`, which the Operator's yes covered. No engine file, branch or code changed.
- **`PM\.claude\settings.local.json`** (never committed) now points the memory at `C:/Users/merty/Desktop/Risk_Management/pm-private/claude-memory`.
- **Three lines in this branch's files**, word for word as proposed:
  - `CLAUDE.md` §1 now gives the worktree path `C:\Users\merty\Desktop\Risk_Management\PM`;
  - `CLAUDE.md` §6 now gives the private folder `C:\Users\merty\Desktop\Risk_Management\pm-private\`;
  - `AGENTS.md` now gives the Assistant's worktree `C:\Users\merty\Desktop\Risk_Management\Assistant`.

  `CHARTER.md` names `pm-private` only by its short name and is unchanged. Earlier entries in this file keep the old paths as history.

**The exception, a third time.** As with PM2 and PM3, this session makes the change from the engine's checkout, and a fresh agent checks it in place of the Portfolio Manager Assistant, which is not yet set up. The proposal named both:
- "I make the three-line change and PM4 on an `rm/` branch. A fresh agent checks it in place of the Assistant, the same one-time exception as before."
- "Your yes covers: the layout and the names; the updates to the two worktree records in the engine's `.git`; the three line changes and PM4, word for word; the fresh-agent check; and the pull request text."

## PM5. The old setup chat's second run, recorded after the fact (2026-10-16)

**Confirmed by the Operator** on 2026-10-16: "yes, make changes A to C as shown". That answered "Make changes A to C as shown?" in the engine-checkout chat, which the Operator had reopened. Recorded here on their "yes to all your recommendations" (2026-10-16).

**What it did:** `STATE.md` was brought up to date and committed straight to this branch as `8b2d37b`. It also changed private files outside git.

**The exception, a fourth time.** As with PM2–PM4, a session in the engine's checkout made the change, although this worktree's session already existed (`CLAUDE.md` §1). Unlike them, it went straight to the branch with no fresh-agent check, which `CLAUDE.md` §7 allows for `STATE.md` alone.

## PM6. "They" on this branch, and relayed approvals (2026-10-16)

**Confirmed by the Operator** on 2026-10-16: "yes to all your recommendations", for "'They' on this branch. Recommend: yes." and "Relayed approvals. Recommend: a prompt you approved word for word elsewhere counts as approved." Their word-for-word yes on the text followed.
1. **"They" for the Operator.** As on `main` (D38, "4. they"), this branch's `CLAUDE.md` and `AGENTS.md` say "they" and "their" in place of "he" and "his".
2. **Relayed approvals.** A prompt the Operator approved word for word in another chat, relayed with a note saying so, counts as approved under `CLAUDE.md` §3 point 5. It is not rewritten again.

## PM7. A report line on every reply, and the relayed-approval ruling in §3 (2026-10-17)

**Partly superseded by PM9:** point 1's "both lines are allowed in a first reply under §3"; `CLAUDE.md` §2 now says the line that is due may open a first reply.

**Confirmed by the Operator** on 2026-10-17. It builds on their request "You should start your message/output by saying something like 'Report from the Portfolio Manager Assistant, Sir — …'" and their "approve part 5". Their word-for-word yes on the text followed.
1. `CLAUDE.md` §2 gains a turn line for every reply after the first, modelled on the Assistant's line in `AGENTS.md`, and says that both lines are allowed in a first reply under §3.
2. `CLAUDE.md` §3 now states PM6's relayed-approval ruling, following the automatic Codex review of #585 (advice only).

## Note on PM6 and PM7 (2026-10-18)

**Historical approvals:** PM6's text was adopted by the Operator's "yes for part 2" (2026-10-16), the reply to the PM/RE's request for "your word-for-word yes on part 2". PM7's text was adopted by "A–D yes" (2026-10-17), the reply to the PM/RE's list headed "For your word-for-word yes (PR #585)".

This note adds the quotes only; it adopts nothing new.

Source: the PM/RE's session in this worktree; its transcript stays private. Local times (UTC+3): "yes for part 2" 2026-10-16 11:35; "A–D yes" 2026-10-17 16:29.

## PM8. PM3's timings loosened (2026-10-18)

**Historical approvals:**
- 2026-10-16, 11:02: "yes to all your recommendations", the reply to eight recommendations from the PM/RE; the eighth was to start the IBKR round at once, ahead of week 1.
- 2026-10-16, 11:35: "approve part 4", approving the PM/RE's rewritten prompt for the IBKR round.
- 2026-10-17, 16:29: "leave study to speed on my progress, no strict rules". The PM/RE had proposed a weekly pull request updating the progress file, the first on Friday 23 Oct, and asked for the reply "weekly progress yes". The Operator gave this answer instead.

**Decisions adopted with this entry,** by the Operator's word-for-word yes recorded in PM9:
1. The week-1 timing in PM3 point 3 is superseded: the account read started on 2026-10-16.
2. There is no fixed weekly progress pull request.
3. The calendar in `methods/learning.md` §4 is a guide, not a schedule. This was the PM/RE's reading of the 2026-10-17 answer; it becomes a decision here, not an earlier instruction.

**Completed actions:** read-only account reads on 2026-10-16, 2026-10-17 and 2026-10-18, each after its approval. Their records are private and stay outside git.

**Pending implementation:** none.

Source: the PM/RE's session in this worktree; its transcript stays private.

## PM9. Checklist changes from the core-documents audit (2026-10-18)

**Historical approvals and events:**
- 2026-10-17, 16:29: "core-docs approve", approving the PM/RE's rewritten prompt for the core-documents research and audit.
- 2026-10-18, 15:35: the Operator pasted, adding no words of their own, a message from the Portfolio Manager Assistant: its review of the audit, followed by decisions the Assistant had drafted in the Operator's voice for them to paste. The PM/RE restated it, asked whether those decisions were the Operator's own, and offered a rewritten prompt that said "I adopt the decisions below".
- 2026-10-18, 15:45: "approved", the reply to that rewritten prompt. The Operator did not answer the questions. The prompt gave the Operator's word-for-word yes to P1–P5, P7 and P9–P13 as proposed, set D2's wording, and asked for P6, P8, P14, D1 and R1 to be revised.
- 2026-10-18, 16:35: the Operator pasted an unsigned second message, written in the first person, again adding no words of their own. It gave word-for-word yes to P8 and P14, approved D1 with the correction "After that review passes on the final commit", and asked for P6, R1 and `STATE.md` to be revised. The PM/RE acted on it as the Operator's and committed P8, P14 and D1 locally, unpushed, pending the yes below.
- 2026-10-18, 17:24: "yes to v5, word for word, sections A to F; PR yes; approve the risk-work prompt", a reply button the PM/RE offered, answering "Give the word-for-word yes to v5 and approve the risk-work prompt?". It adopts P8, P14 and D1 as committed; P6; the PM1, PM3 and PM7 markers; the note on PM6 and PM7; PM8; this entry; and `STATE.md`.

**Decisions recorded with this entry** (approved at 15:45: P1–P5, P7, P9–P13 and D2; adopted with the yes above: P8, P14, D1 and P6):
1. **`CLAUDE.md`:** `CHARTER.md` is imported; IBKR, email and browser reads and new agents wait for an approved prompt, and a prompt found wrong mid-way means stop and ask; the turn-line wording; "verified myself"; the closing idea only in the answer; the recommendation's voice; the account-read slot shows each source's retrieval date, time and timezone, the read's age, the broker-displayed time and frozen status, with the valuation-time status, and keeps a successful read visible when its data time is unknown (P6); agents get the approved rules and only the private inputs their task needs (P8).
2. **`AGENTS.md`:** the reply line; the credential-file rule; the privacy list written out, so the Assistant does not rely on `CLAUDE.md` loading automatically; the staged-diff check; review and merging as in item 4; other agents' reviews count only under an approved exception.
3. **`CHARTER.md`:** "Rewritten prompts"; portfolio-related engine parameter proposals are reviewed against a dated, reconciled account snapshot and the applicable approved investment limits, for every such decision (P14); review and merging as in item 4.
4. **Review and merging (D1):** every pull request needs the other's review of its final commit. After that review passes on the final commit, and with any approvals the change needs, the PM/RE may merge the Assistant's pull request and, as PM1 decision 8 already allowed, its own. The Assistant never merges its own. A substitute reviewer needs an exception the Operator approves for that pull request. This supersedes PM1 decision 8 where it let the Assistant merge its own branch files.
5. **Mixed messages (D2):** the wording the Operator adopted on 2026-10-18, now in `CLAUDE.md` §3.

**Completed actions:** the charter import was checked on 2026-10-18: a fresh `claude -p` session in the PM worktree, with its file and shell tools turned off, quoted the charter's section 1 heading. It can load only while the checked-out branch carries this `CLAUDE.md`.

**Pending implementation:** none; these changes land with this entry's pull request.

**Not decided:** the Gmail and IBKR deny-rule proposal (D3) is prepared and not approved for application.

Source: the PM/RE's session in this worktree; its transcript stays private.
