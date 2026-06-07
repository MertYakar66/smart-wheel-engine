---
id: dashboard-launch-polish
title: Dashboard launch polish — cockpit front door, data honesty, a11y, chat/SSE fixes
kind: feature
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-02
headline: Launch-grade polish of the Next.js dashboard from a read-only repo audit — the Decision Cockpit is now the front door + in the nav, fake "LIVE"/mock data is labelled honestly or wired to real data, the Research chat + SSE leak are fixed, and the cockpit is keyboard/mobile accessible. Interface layer only; no engine logic touched.
surface: []
---

## Goal
Make the analyst dashboard launch-grade (the user's top external priority) without
touching the §2 decision layer. Driven by an 11-dimension read-only repo audit;
this PR closes the dashboard-side findings.

## What we tried
Audited the full `dashboard/` tree (cockpit + terminal + news app) for: navigability,
data honesty, accessibility, broken integrations, and brand consistency. Each fix was
made in an isolated worktree off `origin/main` and verified with `tsc --noEmit`,
`next lint`, and a full `next build`.

## What worked
- **Front door (launch blocker).** The Decision Cockpit was unreachable except by typing
  `/cockpit`. Added it as the first, accented nav entry (`nav.tsx`), made `/` redirect to
  `/cockpit` (`page.tsx`), and made navigation bidirectional: a Cockpit·Terminal·News strip
  in the cockpit header and the terminal status-bar brand now links back to the cockpit.
- **Data honesty (the biggest credibility risk for a finance launch).**
  - `market-overview.tsx` now badges placeholder index/futures/commodity tiles **DEMO**
    (amber) instead of green **LIVE**; gated by a new `isLive` prop (default false).
  - The terminal status-bar "AGENT ●online" was hardcoded; now reports **offline**
    (the agent runtime is not wired — `connected={false}`).
  - The ticker page rendered a `Math.random()` random-walk chart labelled "Price History
    (30D)" (also the only 2 ESLint *errors* — impure-during-render). Replaced with **real
    daily closes** from the engine OHLCV endpoint (`/api/engine?action=chart`), with an
    honest empty state when the engine is down — never fabricated data.
  - The macro calendar's static projected FOMC/CPI/jobs/GDP dates now carry a **Projected**
    badge + "approximate dates" note.
- **Broken integrations fixed.**
  - Research chat: client parsed legacy `0:`-prefixed data-stream framing while the server
    sends a raw text stream (`toTextStreamResponse`), so it always showed "No response."
    Now appends the decoded text chunk directly (`chat-panel.tsx`).
  - SSE `/api/stream` leaked a 10s DB-poll `setInterval` per client (cleanup never wired).
    `GET` now takes `Request`, registers `request.signal` abort + a stream `cancel()`.
  - `checkOllama()` fetched Ollama from the browser (CORS-blocked → always "disconnected");
    now proxies the same-origin `/api/engine?action=ollama_status` (engine checks it
    server-side, which is who actually calls Ollama).
- **Accessibility.** Cockpit table rows are now keyboard-operable (`role=button`,
  `tabIndex`, Enter/Space `onKeyDown`, focus-visible ring); the dossier drawer renders as a
  mobile/tablet bottom-sheet below `lg` (previously `hidden lg:block` → a dead click).
- **Staleness signal.** The cockpit's stale `as_of` default (2026-03-20) now shows a
  client-computed "(N days old)" chip (amber >7d, red >30d) in the regime banner.
- **Brand + docs.** Unified the brand to **YAKAR TERMINAL** (the umbrella name in root
  metadata; the news nav read "FinanceNews"). Added `ENGINE_API_URL` to `.env.example`.
  Corrected `dashboard/README.md` (Next.js 16, Node 20+, the `(main)`=news /
  `(terminal)`=cockpit route map).

## What didn't
- Considered removing the ticker chart entirely (audit's "remove until wired" option) — but
  the engine already serves real OHLCV via `/api/engine?action=chart` (shape `{data:[{date,
  close}]}`, confirmed against `chart-panel.tsx`), so wiring real data with a graceful
  fallback was strictly better than deleting a feature.
- `NEXT_PUBLIC_OLLAMA_URL` was going to be added to `.env.example` (audit rank 13), but the
  CORS fix removed its only reader — adding it would document a phantom var, so only
  `ENGINE_API_URL` was added.

## How we fixed it
Targeted edits to 16 files under `dashboard/`, interface-layer only. No engine logic, no new
endpoints, no change to how candidates are evaluated — the dashboard still renders only what
the engine API returns, so the §2 invariant is unaffected.

## Evidence
- `npx tsc --noEmit`: clean (0 type errors).
- `next lint`: **0 errors** (was 2 — the ticker impure-render), 26 pre-existing unused-var
  warnings remain in untouched files (noted for a separate hygiene pass).
- `npm run build`: ✓ compiled successfully, 25 routes (incl. `/cockpit`), exit 0.

## Unresolved / handoff
- ~26 pre-existing `no-unused-vars` warnings in untouched dashboard files (services/, top,
  watchlist, alerts, categories, useEngineData, etc.) — harmless dead code; a dedicated
  lint-hygiene PR should clear them (some, e.g. `watchlist-panel onAddTicker` and
  `useEngineData setPortfolio`, signal genuinely-unwired features and need a decision, not a
  blind delete).
- Full reconciliation of the two visual languages + a theme toggle (audit ranks 23/24, L
  effort) deferred — brand name is unified, but the news app stays light-shadcn while the
  cockpit/terminal stay dark.
- Wiring real index/futures/commodity quotes (to flip the DEMO badge to LIVE) and EDGAR
  filings on the ticker page remain future enhancements.
