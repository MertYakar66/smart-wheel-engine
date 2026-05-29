---
id: S13
title: Dashboard end-to-end
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise the Next.js dashboard (`dashboard/`) as a user
would, and answer the §2 question: does the dashboard faithfully
display the engine's EV verdicts, or is there client-side logic that
recomputes or overrides them — anything that could present a
non-tradeable candidate as tradeable?

**Setup.** Terminal B worktree `../swe-terminal-b`,
`SWE_DATA_PROVIDER=bloomberg`. Fresh `npm install`; `engine_api.py` on
`:8787` + the Next.js dev server (`npm run dev`) on `:3000`. Exercised
at the HTTP level — every page route and the `/api/engine` bridge —
plus a full source read of the API → UI data path. No
browser-automation tool was available, so the post-JS rendered UI was
not visually driven (stated plainly, per the task). No code changes.

**Path.** `dashboard/` is a Next.js app (originally the "finance-news"
aggregator) with the wheel engine bolted on. It reaches the engine
**only** through `dashboard/src/app/api/engine/route.ts` — a
server-side proxy: each `action` does `fetchEngine(...)` →
`NextResponse.json(data)`, forwarding `engine_api.py` responses
verbatim. `useEngineData` fetches `/api/engine?action=candidates`
(→ engine `/api/candidates` → `rank_candidates_by_ev` →
`EVEngine.evaluate`) and stores the result; components render it.

**Status.** Done. The dashboard builds and runs — all page routes
(`/`, `/top`, `/terminal`, `/feed`, `/watchlist`, `/calendar`) and the
`/api/engine` bridge (`status`, `candidates`, `regime`, `vix`) served
HTTP 200 with live EV-authoritative engine data (FIX `evDollars`
2263.5, regime ELEVATED, VIX 28.97). §2 holds — no client-side verdict
computation. No §2 violation, no bug. All findings logged.

**Findings:**

- **§2 holds — the dashboard is a display layer with no verdict
  authority.** The engine is reached only via `api/engine/route.ts`, a
  verbatim proxy (no transformation of any kind). `useEngineData` is
  fetch-and-store. A repo-wide grep of `dashboard/src` finds **no**
  client-side EV or verdict computation — no `ev_dollars` recompute,
  no proceed / skip / tradeable logic; the engine fields (`evDollars`,
  `probProfit`, `cvar5`, …) are rendered as received.
  `/api/engine?action=candidates` empirically returned the engine's EV
  output (FIX `evDollars` 2263.5). The only client-side ranking,
  `services/exposure-ranking.ts`, ranks **news stories** by
  user-exposure relevance — not trade candidates. The dashboard cannot
  turn a non-tradeable candidate tradeable. **Logged.**

- **The terminal renders hardcoded placeholder data with no "demo"
  labelling.** `(terminal)/terminal/page.tsx` feeds `MarketOverview`
  (indices SPX 5234.18, futures, commodities) and `AgentPanel` from
  `PLACEHOLDER_*` constants — static fake numbers; a user sees what
  looks like live index / futures / commodity quotes. (`AgentPanel`
  at least gets `connected=false`; `MarketOverview` gets
  `loading=false` and no honesty flag.) Likewise, when no Daytona
  sandbox is configured the research-chat code-execution path falls
  back to a `TemplateExecutor` returning canned tables (event-study
  rows hardcoded `-0.3% / +2.1% / …`). Misleading displays. **Logged.**

- **The OptionsPanel portfolio summary is permanently zero.**
  `useEngineData` initialises `portfolio` to
  `{openPositions:0, totalPremiumCollected:0, winRate:0, avgDaysHeld:0}`
  and never calls `setPortfolio` — the hook has no portfolio fetch.
  The terminal's portfolio summary always shows 0 positions / $0 /
  0 % win rate, regardless of state. **Logged.**

- **README and `package.json` are stale.** `dashboard/package.json`
  (`"name": "finance-news"`) and `README.md` ("FinanceNews — AI
  Financial News Platform") describe only the original news
  aggregator; the README architecture diagram omits the entire
  engine-wired trading terminal (`(terminal)/`, `api/engine`,
  `api/exposure`, payoff diagrams, strike recommendations) added later
  — per the `dashboard/` git log ("Wire dashboard to engine", "Add …
  interactive terminal", …). A fresh reader of the docs would not know
  the trading surface exists. **Logged.**

- **Silent error states.** Several dashboard fetches swallow failures
  silently — `catch { /* silent */ }` in `useTickerAnalysis`,
  `fetchAlerts`, `checkOllama`; others only `console.error`. On a
  fresh DB the news / watchlist / events routes return `[]` (no
  ingestion) — empty, not erroring — but the UI cannot distinguish
  "empty" from "failed to load." Same silent-failure family S1 / S2 /
  S9 logged on the engine side. **Logged.**

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — `dashboard/src/app/api/engine/route.ts` is verbatim-proxy; no client-side recomputation. `useEngineData` is fetch-and-store. Live probe: `/api/engine?action=candidates` returned top row `{ticker:"FIX", evDollars:2547.97, probProfit:0.9714, ...}` — values pass through unmodified from `engine_api.py`.
  - Qualitative verdict: match — full dashboard stack runs (Node v22.14.0 + npm 10.9.2 installed; `npm install` and `npm run dev` both succeed on the worktree). `/api/engine?action=status` → 200 with `universe_size: 503`, `vix: 28.97`. `/api/engine?action=regime` → `"ELEVATED"`. `/api/engine?action=vix` → 28.97. `/top` → 200 HTML. The "dashboard is a display layer with no verdict authority" finding holds.
  - Numerical drift > 5%:
    - metric `FIX_evDollars` (top dashboard candidate): orig `2263.5` → new `2547.97` (`+12.6%`); attributable to **PR #179** (IV-PIT fix in `rank_candidates_by_ev` — FIX's PIT IV is higher than the snapshot, raising premium and EV) plus possibly PR #208 (HMM regime label disambiguation). FIX is still #1 candidate, so the ordering signal is preserved.
  - Notes: regime label and VIX value bit-identical to original (`ELEVATED`, `28.97`). Dashboard's `OptionsPanel.portfolio` is still hardcoded to zeros — the documented `useEngineData` initialization at `{openPositions:0,...}` with no `setPortfolio` is unchanged. Terminal `MarketOverview` still uses `PLACEHOLDER_*` constants. Both findings remain Logged-by-design.

---
