# STATE — the Portfolio Manager / Risk Engineer

**Last updated:** 2026-11-16, by the PM/RE, at the close of the day's second session. PM1–PM13 are on this branch (PM13 from the merge of #610, `22dc437`).

Nothing private is written here. This branch is public.

## In force

PM1, the role and its checklists; PM3, the learning program (on hold, see Open 6); PM4, one Risk_Management folder; PM5, the old setup chat's second run recorded; PM6, "they" on this branch (its ruling on relayed approvals was amended by PM13); PM7, the turn line on every reply; PM8, PM3's timings loosened; PM9, the checklist changes from the core-documents audit; PM13, the Operator's ten-point standing rule, the same text as the engine's (`main`, D32's amendment of 2026-11-15); it replaced the active text of PM10 and PM11, which stay as history. Lasting plans for saved account data in force: none. PM12 was a one-time reviewer exception for #592 and has ended.

Working practice since 2026-11-14 (recorded with PM11; PM13 made its first item rule text, and a material error found in the agreed design is told at once): the PM/RE decides small things and reports each in one line; the agreed engine design is not reviewed again until D40; each message says where the five steps stand.

## The work now, and the five steps

Since 2026-10-31 the work is the engine's data: settling what data the engine needs, and being ready to choose a source. Account work is set aside until the Operator says otherwise. The five steps are D40, the vendor pick, the first data update, the join, and the first pass. None is done yet. Step 1 is in motion: on 2026-11-16 the Operator had the engine's pen take #604 and #606 through the standing rule, in a pen session of its own. The Operator said the same day that the Theta Data subscription is not active, as `main`'s `PROJECT_STATE.md` also records, so no plan may assume a live Theta feed.

## Completed since 2026-10-18

- Read-only account reads on 2026-10-20 and 2026-10-21. Their records are private in `pm-private\account\`. The latest saved read is dated 2026-10-21.
- The standing rule was revised (PM10, 2026-10-31) and extended (PM11, 2026-11-14). #605 tells the engine's pen about PM10.
- Input for D40 on #604: the first post (2026-10-31), parts one, two and three (2026-11-05, 2026-11-06, 2026-11-14) and note 1 (2026-11-14). The Operator agreed to the two design documents on 2026-11-06 and to amendment 1 on 2026-11-14. They are private files, which the posts identify by checksum.
- #606 (2026-11-05): a request for a weekly job that keeps free dated copies before D40.
- #592 was merged on 2026-11-14 as `8280b87`, after a fresh read-only agent passed its final commit under PM12.
- The engine recorded the Operator's ten-point standing rule on 2026-11-15 (`main` at `c7601c0`, #607). This branch recorded the same text as PM13: #610 was merged on 2026-11-16 as `22dc437`, after a fresh, independent read-only agent passed its final commit `99878a8` under PM13's decision 10.
- 2026-11-16, second session: the PM/RE advised the Operator, in chat only, on the engine's read-only startup audit and its connector-and-skills review, which the pen ran in its own chat. The PM/RE changed nothing for it on this branch or in memory, and read no account data.

## Waiting on others

- **The engine's pen:** D40's draft, shown to the Operator in chat first, with the two clarifications of note 1; and #606's exception text. Both are now in the pen's D40 session, which first needs the Operator's approval of its two plans (next item). The pen answered #605 on 2026-11-15; #605 is still open. No reply is posted on #604 or #606 as of 2026-11-16. No look-again date is set for #604 or #606, which were posted before PM13. A correction: until 2026-11-16 this file said D40's start waited on the pen, while `main`'s `PROJECT_STATE.md` recorded #604 and #606 as waiting on the Operator. The Operator's word to the pen on 2026-11-16 settled it.
- **The Operator:** approval of the two plans in the pen's D40 session (one for #606, one for D40's draft), as the pen reported them on 2026-11-16, which the PM/RE has not seen; line-by-line answers on the investment-limits draft of 2026-10-31; a yes or no to changing the PM/RE's memory note, which still holds the rule of PM10 and PM11, and to removing account figures from memory notes, each as its own proposal; a plan to write the metadata file for the newest saved snapshot, without which the opening line gives only the folder's date; a lasting plan for questions about the book, if they want one; their decision on the session-advisor trial (Open 7); a decision on further block lines for the brokerage connector (see "Carried over"); the "setup review" of hooks and skills, which is offered; and reading the vendor's terms, before which no script runs on the held option-chain data.

## Open

1. **The paid-sources comparison.** The Operator said "Not yet" on 2026-11-16. It follows D40, under its own approved plan.
2. **Risk limits (the investment policy).** A second draft (2026-10-31) is private and waits for the Operator's answers. Nothing is approved. Its statuses were measured on the 2026-10-21 read and are not current.
3. **The risk work approved on 2026-10-18 (PM9).** Its approval stays valid for exactly the work it authorizes, and a read it does not name needs a plan first (PM13, decision 6). Its state was not re-checked at this close.
4. **The engine's parameters.** Not yet reviewed. Portfolio-related proposals are reviewed against a dated, reconciled account snapshot and the applicable approved investment limits (`CHARTER.md` §1).
5. **The Portfolio Manager Assistant** is not reviewing at present. The exceptions for #592 (PM12) and for #610 (PM13, decision 10) have both ended. Until that changes, each pull request on this branch needs the Operator's exception for a substitute reviewer.
6. **Lessons.** On hold since 2026-10-18, by the Operator. Lesson 1.0's check question stays open.
7. **A session-advisor skill** is on a three-session trial on the Operator's machine (approved 2026-11-14). It is not on this branch. Three sessions are counted (2026-11-14, 2026-11-15 and 2026-11-16). The PM/RE's recommendation was given at the close of the third: keep it, with two revisions. The Operator's decision is pending, and the skill was not run in the session that followed.
8. **The reviewer's notes on #592** are settled by PM13: the limits on questions no longer conflict; the mixed-message line; "Plans and final prompts" in `CHARTER.md` §2; markers on PM1, PM2, PM6, PM7, PM9, PM10 and PM11. One is left as it is: a possible typing slip inside a quote in PM10, which is history.
9. **The reviewer's notes on #610,** none blocking, for a later change: the sentence in `AGENTS.md` on tracked files that hold account data could be clearer; the word-for-word yes for a `DECISIONS.md` entry is in `CLAUDE.md` §3 but not in §7's list or in `CHARTER.md` §1; the pull request's text says "Decided by the Operator" of thirteen points, of which four are the PM/RE's proposals that the Operator approved with the text.

## Carried over from 2026-10-18, not re-checked at this close

- Deny rules for both tool-naming forms: a proposal waits for the Operator's decision. On 2026-11-16 three further tool names were found that the block does not cover; the private handoff of that date names them.
- The IBKR data setup recommended on 2026-10-16 (one Activity Flex Query and a scheduled PortfolioAnalyst report), which the Operator builds if they choose.
- CFA registration for August 2027 opened on 3 Nov 2026 (early price until 20 Jan 2027). The Operator registers.
- Private copies outside `pm-private`: the Operator deletes them.

## Next

Support the Operator on D40, in a fresh chat: check the pen's two plans, and then D40's draft, against #604, #606 and the agreed private documents. The paid-sources comparison follows, on the Operator's word, under its own approved plan. The newest private handoff in `pm-private\advisor-trial\` is dated 2026-11-16 19:34. The earlier ones remain: 2026-11-14 15:08, for the data work, and 2026-11-15 18:08, for PM13, which is finished.
