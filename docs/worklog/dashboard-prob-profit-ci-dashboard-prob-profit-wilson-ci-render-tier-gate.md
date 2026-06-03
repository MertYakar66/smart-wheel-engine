---
id: dashboard-prob-profit-ci
title: Dashboard prob_profit Wilson-CI render (tier-gated)
kind: feature
status: in-flight
terminal:
pr:
decisions: []
date: 2026-06-01
headline: Decision-cockpit renders the prob_profit Wilson 95% sampling CI + N, gated to the IID non-overlapping forward tier so it never shows false precision.
surface:
  - dashboard/src/types/cockpit.ts
  - dashboard/src/lib/cockpit-trust.ts
  - dashboard/src/components/cockpit/calibrated-prob.tsx
  - dashboard/src/components/cockpit/verdict-card.tsx
  - dashboard/src/components/cockpit/cockpit-table.tsx
---

## Goal
Complete the last mile of the `prob_profit` honesty chain. The engine half
(branches `claude/prob-profit-ci-honesty` → `claude/prob-profit-ci-propagate`)
makes `prob_profit` honest by emitting `n_scenarios` + a Wilson 95% sampling CI
(`prob_profit_ci_low/high`), camelCased over the wire by `/api/candidates` as
`nScenarios` / `probProfitCiLow` / `probProfitCiHigh`. But the **decision
cockpit still rendered a bare `86%`** — the honesty fix never reached the human's
eyes. This change is the *consumer half*: surface the interval (`86% [71–94%]
N=35`) so a k/N frequency over ~35 forward scenarios is never read as a precise
2-dp figure.

## What we tried
1. Mapped the cockpit render path (Explore agent): `/api/candidates` →
   `EngineCandidate` (`types/cockpit.ts`) → `CockpitTable` → `CalibratedProb`
   (the "Confidence" column) and `VerdictCard` (dossier header metric grid).
   Formatters live in `lib/cockpit-trust.ts`; the file already encodes a
   *calibration* trust layer (top-bin over-confidence, crisis-realized ~0.57).
2. Added the three fields as **optional** on `EngineCandidate`, a `fmtProbCi`
   formatter, a faint CI band on the existing 0–100% track, a caption, and the
   metric-grid bracket — all **defensive** (degrade to the bare dot when the CI
   is absent), so the branch sits on `origin/main` independent of the API
   branch's merge order. tsc + eslint clean.
3. Ran a 3-lens adversarial review (correctness / quant-UX-honesty /
   convention-scope) before committing.

## What worked
- The whole render is pure presentation: it reads `probProfit` + the new fields
  and only draws them. It cannot alter ranking or verdicts — the §2 invariant is
  untouched (confirmed by the convention-scope reviewer).
- Defensive degradation is a true no-op when the CI is absent: `showBand=false`,
  `fmtProbCi` returns `""`, `nStr=null` → the bare point estimate, unchanged.
- The proxy (`/api/engine`) forwards the JSON body verbatim, so no route change
  was needed; the field names match end-to-end (verified against
  `engine_api.py:747-749` and `wheel_runner.py:1706-1714` on the unmerged
  producer branches).

## What didn't
The review caught a **real quant defect** that the naïve render shared with the
engine half: a Wilson interval is only an honest *sampling* CI when
`n_scenarios` is a genuine independent-trial count. The production ranker feeds
the engine via `best_available_forward_distribution`, a 4-tier cascade
(`forward_distribution.py:342-386`): `empirical_non_overlapping` (~35 IID
windows — honest), `empirical_overlapping` (≥60 autocorrelated — effective N ≪
count), `block_bootstrap` (n=5000 resamples), `har_rv` (5000 synthetic). The
engine sets `n_scenarios = len(pnls)` for **all** tiers, so a Wilson CI over
2000–5000 draws renders a deceptively *tight* band — false precision, the exact
opposite of the goal. (Verified against source, not taken on the reviewer's
word.)

## How we fixed it
- **Tier gate** `samplingCiHonest(source)` → renders the CI only when
  `distributionSource === "empirical_non_overlapping"`; on every other tier the
  CI/band/N are suppressed and the row degrades to the bare estimate. Both
  surfaces (`CalibratedProb`, `VerdictCard`) consume it identically.
- **Defensive formatting** in `fmtProbCi`: order bounds (min/max, never
  inverts) and widen on display (lower floored, upper ceiled) so the caption
  never claims tighter precision than Wilson gives.
- **`fmtN`** centralizes the `N>0` guard (no more `N=0` / negative labels) and
  removes the duplicated finiteness check both components had.
- **Axis disambiguation**: the visible caption reads `sampling 95% CI […]` so it
  is never fused with the orthogonal calibration `realized ~57% in crisis` line.

## Evidence
- `npx tsc --noEmit` → exit 0; `eslint` on the 5 files → exit 0 (node_modules
  junctioned from the primary clone; package.json byte-identical).
- 3-lens review wf_ee2f9131-0ff: 0 §2 / scope breaches; both "blockers" were the
  sequencing fact (consumer half ahead of producer) + the tier false-precision
  concern, both now addressed.
- `git diff --stat`: 5 files, +150/-6, all under `dashboard/`.

## Unresolved / handoff
- **Lights up on merge of the producer branches.** Until
  `claude/prob-profit-ci-propagate` (carries the `/api/candidates` emission)
  merges, the cockpit shows the same bare `%` as before (graceful no-op). This
  is the consumer half — **not** a live honesty improvement on its own.
- **Engine-half root-cause fix is the operator's §2 call (NOT self-authorized).**
  The cleanest fix is to null the Wilson CI *at the engine* when
  `distribution_source != "empirical_non_overlapping"` (or emit an effective
  sample size), which would also make the **API JSON and the Ollama trade memo**
  honest — both still emit/render the tight CI on non-IID tiers. That touches
  the §2 trio (`ev_engine.py` / `wheel_runner.py`) and needs the lane-claim +
  independent §2 second-read process. This dashboard gate is belt-and-suspenders
  in the meantime.
- Possible follow-up: a one-decimal CI near the 0.90 trust boundary (whole-pct
  rounding muddies the exact R11 bin edge).
