# STATE — the Portfolio Manager / Risk Engineer

**Last updated:** 2026-10-15, the setup (`DECISIONS.md` PM1).

Nothing private is written here. This branch is public.

## What is true now

- The role is set up on this orphan branch, `Risk_Management`, with its checklist (`CLAUDE.md`), the Assistant's checklist (`AGENTS.md`) and the charter (`CHARTER.md`), approved by the Operator (PM1).
- The engine's side of the setup, Part 5 in PM1, is not yet on `main`. It waits for #577 to merge, then for the engine's pen to carry it out.
- Until Part 5 lands, `main`'s rule-books do not yet name this role.
- The engine's session-open still counts this branch under "other branches".
- The `pm-request` label does not exist yet, so the "open requests" slot of the mark reads `unknown`.

## What is open

1. **The engine's side (Part 5).** After #577 merges, the Operator pastes the handoff into the engine's pen ("Main Agent Local").
2. **The Assistant's first session.** In its own worktree:
   - trust that folder in Codex;
   - add the private folder as a place it may write;
   - test that with one file before its first check.
3. **The IBKR round.** A read-only account read. Research into IBKR's reporting and data options (Flex Queries, statements, PortfolioAnalyst, Client Portal Web API, TWS / IB Gateway, the connector). One recommended setup.
4. **The engine's parameters.** After the account read, the Operator and the PM/RE review them; the Operator wants major changes.
5. **The engine's outputs.** Once the engine is in use, the Operator and the PM/RE review them for logic and feasibility.
6. **The finance learning program.** The prompt is approved. Its first deliverable, the syllabus, is in progress.
7. **Risk limits (the investment policy).** Not started. The PM/RE drafts them; the Operator decides.

## Next

Open a session in this folder. Its first act is `/memory`, to confirm that memory points at the private folder.
