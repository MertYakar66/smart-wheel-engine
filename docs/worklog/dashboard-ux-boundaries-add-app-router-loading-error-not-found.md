---
id: dashboard-ux-boundaries
title: Dashboard App Router boundaries (loading / error / not-found)
kind: feature
status: complete
terminal: dash-ux
pr:
decisions: []
date: 2026-06-03
headline: Added the missing Next.js App Router special files — graceful loading skeletons, error boundaries (retry, no blank screen), and a branded 404 — across the news-app and terminal route groups. Additive-only; zero engine/decision-layer surface.
surface: []
---

## Goal

The dashboard shipped with **no** `loading.tsx`, `error.tsx`, `global-error.tsx`,
or `not-found.tsx` anywhere in the App Router. Consequences for a launch-facing
UI: navigation showed no loading state, an unhandled render/fetch error fell
through to Next's default overlay (a blank screen in production), and an unknown
URL hit the bare default 404. Close those gaps without touching any existing
file (so it can't collide with in-flight cockpit work — see the pending
`claude/dashboard-prob-profit-ci`).

## What worked

Purely additive — six new files, no edits to existing components/pages:

- `src/app/not-found.tsx` — branded 404, links back to the Cockpit / TOP.
- `src/app/global-error.tsx` — root error boundary (renders its own
  `<html>/<body>`, since it replaces the root layout) with a retry + a digest
  ref line for support.
- `src/app/(main)/loading.tsx` + `error.tsx` — skeleton list / retry boundary
  for the news-app routes; the `Nav` stays usable on error so the user can
  navigate away.
- `src/app/(terminal)/loading.tsx` + `error.tsx` — monospace skeleton / retry
  boundary matching the `terminal-body` shell; the error copy points at the
  engine on `:8787` (the usual cause of a terminal-view failure).

Reused the existing idioms: `Skeleton`, `Button`/`buttonVariants`, the
zinc/blue/amber palette, `lucide-react` icons, and `dark:` variants.

## How we fixed it

Surveyed `src/app` for the special files (none present), read the root +
group layouts, `skeleton.tsx`, `button.tsx`, and `nav.tsx` to match style, then
wrote the six boundaries. Added their `FILE_MANIFEST.md` rows (dashboard app
files are listed individually, not glob-covered).

## Evidence

- `npx tsc --noEmit` clean; `npx eslint` clean on all six files.
- `npm run build` succeeded — `/_not-found` now in the route table; the error
  / loading boundaries compiled into the segment trees.
- `python scripts/check_manifest_coverage.py` and
  `python scripts/gen_worklog_index.py --check` pass.
- No `.py`, no `engine/`, no decision-layer file touched — §2 untouched by
  construction.

## Unresolved / handoff

- These boundaries are deliberately generic. Per-route loading skeletons that
  mirror each page's exact layout (e.g. the TOP grid) could be a later polish.
- The terminal error hint assumes the engine on `:8787`; if the proxy base
  becomes configurable, update the copy.
