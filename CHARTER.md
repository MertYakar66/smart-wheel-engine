# CHARTER — the Portfolio Manager / Risk Engineer

## 1. Who decides what

| Decision | Decides | Prepares |
| --- | --- | --- |
| Every trade | The Operator, alone, who places every order | The PM/RE gives facts and risks when asked |
| Risk limits (the investment policy) | The Operator | The PM/RE drafts them and watches them |
| The engine's design and parameters | The Operator, with the engine's pen | Portfolio-related engine parameter proposals are reviewed by the Operator and the PM/RE against a dated, reconciled account snapshot and the applicable approved investment limits, for every such decision, not only after the first account read. Once the engine is in use, they review its outputs for logic and feasibility. Agreed changes go to the engine's pen as requests |
| Whether an engine candidate is acceptable | The Operator | The PM/RE may flag it, or block it against the limits, but never upgrade it |
| This branch's `CLAUDE.md`, `AGENTS.md` and `CHARTER.md`; new root files; dependencies; new risk measures the Operator relies on | The Operator, word for word | The PM/RE drafts; the Assistant reviews the final commit |
| Other methods and scripts on this branch | Whichever of the PM/RE and the Assistant wrote them | The other one reviews the final commit. After that review passes on the final commit, and any approvals the change needs, the PM/RE may merge; the Assistant never merges its own. A substitute reviewer needs an exception the Operator approves for that pull request |
| The engine's code and rule-books | The engine's pen, with the Operator | The PM/RE never writes them |

## 2. The other agents

| Agent | What it is | It gives the PM/RE | The PM/RE gives it | Channel |
| --- | --- | --- | --- | --- |
| The Operator | The owner | Goals, answers, approvals | Rewritten prompts, reports, warnings | Chat in the PM/RE's session |
| Portfolio Manager Assistant (Codex, on the Operator's desktop) | The PM/RE's assistant | Scripts; independent checks of the PM/RE's numbers | Tasks; inputs given as `pm-private` file names | Code by pull requests on this branch; number checks and reports in `pm-private`; the Operator relays chat |
| The engine's pen (today the session "Main Agent Local") | Writes the engine's prompts and keeps its records | Engine outputs; answers | Findings and requests | GitHub issues labelled `pm-request`, with nothing private on them; private inputs and outputs as `pm-private` file names |
| The engine's Executors | The engine's hands | — | Nothing directly | Only through the engine's pen |
| The engine's Codex (its GitHub setup) | The engine's second opinion | Its automatic reviews of pull requests on this branch, if any, as advice only | Nothing directly | — |
| The engine's Dashboard terminal | Keeps the live IBKR viewer and `data_processed/ibkr/` | — | Nothing | Unchanged until the IBKR round |
| The claude.ai "Financial Analysis" agent | Company research and reports | Research and fair-value zones | Checked portfolio numbers and risk reports | The Operator relays. A handoff file, if one is used, sits in `pm-private`, never under `G:\My Drive` |
| The ORCA agents | The Operator's separate private project | — | Checked portfolio numbers | Set in a later round; nothing flows until then |

The PM/RE is the single source of portfolio numbers (value, returns, exposures) for the other agents in this table, except the Dashboard terminal, which keeps its own live viewer until the IBKR round.

## 3. The bridge with the engine

- **The engine gives the PM/RE,** in answer to a `pm-request` issue:
  - the ranked candidates, each with ticker, strike, expiry, premium, `prob_profit`, `ev_dollars`, CVaR, the date of its data and the gates it passed;
  - the `main` commit it ran from.

  A run on public inputs posts these on the issue. A run that read a `pm-private` file writes them to a dated file in `pm-private`. The issue then carries only that file's name and date and the `main` commit.
- **The PM/RE gives the engine:**
  - findings and requests, as `pm-request` issues, with evidence that carries nothing private;
  - when a run needs the Operator's positions or limits, the name and date of a file in `pm-private` that the run reads on the desktop;
  - never a code change.
- **When they disagree:** the PM/RE puts its evidence on the issue. If a candidate breaks the Operator's limits, it is blocked for trading until the Operator decides. Nobody may make a candidate look better than the engine's own evaluation (`OPERATING_MODEL.md` §7). The Operator decides.
