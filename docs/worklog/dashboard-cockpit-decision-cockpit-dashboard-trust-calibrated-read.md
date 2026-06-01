---
id: dashboard-cockpit
title: Decision-cockpit dashboard — trust-calibrated read of the engine verdict
kind: feature
status: complete
terminal: ultracode
pr:
decisions: []
date: 2026-06-01
headline: A read-top-to-bottom decision cockpit in dashboard/src that encodes WHAT TO TRUST and WHAT TO DISTRUST about the engine output — distribution-not-point-EV, calibration-flagged top bin, R11 regime banner. Pure display layer; no engine changes.
surface:
  - dashboard/src/app/(terminal)/cockpit/page.tsx
  - dashboard/src/components/cockpit/
  - dashboard/src/lib/cockpit-trust.ts
  - dashboard/src/types/cockpit.ts
  - dashboard/src/app/api/engine/route.ts
---

## Goal

Build a decision **cockpit** (read top-to-bottom and act) — not a browse
portal — that visualizes the smart-wheel-engine's short-put verdict and, above
all, **encodes what the verification campaign proved trustworthy vs
misleading**:

- Trust mid-range `prob_profit` (0.60–0.85). **Distrust the top bin** (> 0.90):
  crisis-realized ~0.57 vs ~0.96 forecast (heavy-verify I1). Never draw a
  confident green "96%".
- `ev_dollars` is a **ranking score, not a dollar forecast** (~0 correlation
  with realized $). Show the **distribution** (p25/p50/p75 + cvar5), not the
  point estimate, as the headline.
- It's a **defensive premium sleeve**, not a market-beater.

Pure display-layer work in `dashboard/src/` + read-only consumption of the
existing `/api/candidates` and `/api/tv/dossier` endpoints. No engine changes.

## What we tried

Grounded on the REAL wire shapes first (no designing off assumptions):

- Read `engine_api.py::_handle_candidates` / `_handle_tv_dossier` for the exact
  emitted fields, and hit the live engine at `as_of=2026-03-20` to capture real
  values. Key finding that drove the headline visual: for a short put,
  `pnlP25 == pnlP50 == pnlP75` frequently **pin at the max-premium ceiling**
  while `cvar5` carries a large negative tail (e.g. BKNG p25=p50=p75=$8,799 vs
  cvar5 = −$38,612). A naïve p25–p75 box collapses to a flat line and HIDES the
  risk — so the bar must foreground the cvar5 tail.
- Confirmed `/api/candidates` does NOT emit `vix_level`, `verdict_reason`,
  `drops_summary`, or book exposures; those live in the dossier path / on
  `frame.attrs` (not serialized). Built on what exists; flagged the rest as
  follow-ups rather than touching the engine.

## What worked

Shipped (each its own component, incrementally committed):

1. **Regime banner** — VIX level + regime label, R11 active/dormant
   ("ELEVATED VOL — size down top bin"), as_of, universe-scan summary,
   defensive-posture note.
2. **Candidate cockpit table** — each row a decision unit: verdict badge →
   **P&L distribution bar** → **calibration-aware confidence** →
   strike/premium/DTE/IV/collateral/ROC/CVaR5 → de-emphasized **EV·rank**.
3. **Distribution bar** (the headline visual) — cvar5 whisker · p25/p50/p75
   box · breakeven (zero) line. The long red tail dominates when the body
   pins at the premium ceiling.
4. **Calibration-aware prob** — 0–1 dot, green mid-range, amber/red top bin;
   in elevated vol a ghost marker at the crisis-realized ~0.57 with the gap
   drawn. Never a confident green high-confidence reading.
5. **Dossier drawer** — verdict card + plain-language reviewer-chain trace
   (R1–R11) derived from the EV row + market VIX using the engine's own R11
   thresholds; chart/book-dependent rules shown as needs-chart / needs-book.
6. **Funnel** — universe → scanned → ranked → shown.
7. **Concentration meters** — single-name %NAV vs the (now-armed) 10% R10 cap.
8. **Verdict card** + trust legend + loading/empty/error states + metric
   tooltips with trust caveats.

The trust thresholds live in one place (`lib/cockpit-trust.ts`) mirroring
`engine/candidate_dossier.py` R11 (VIX 25 / top-bin 0.90), so the UI caution
boundary equals the engine's size-down boundary.

## What didn't

- **`drops_summary` is not API-exposed.** The engine computes per-gate drops
  (chain-quality / event-gate / EV-threshold) on `frame.attrs`, but
  `/api/candidates` iterates rows, not attrs. The funnel therefore shows
  universe → scanned → ranked → shown and explicitly flags the per-gate
  breakdown as a follow-up — it does not invent numbers.
- **R9 sector concentration needs the engine sector map** (DEFAULT_SECTOR_MAP),
  absent from `/api/candidates`. Rather than guess sectors client-side, the
  meter ships single-name (R10) — fully supported by candidate strikes — and
  flags sector bars + the live R9/R10 verdict (via `/api/tv/dossier`) as a
  follow-up.
- **Banner VIX uses `/api/vix` (latest), not PIT-at-arbitrary-as_of** — the vix
  endpoint takes no `as_of`. Correct for the intended `as_of=2026-03-20`
  (=latest data → 28.97); a PIT `as_of` on `/api/vix` is an engine follow-up.
- **Dossier R2 (chart-missing) swamps the sandbox.** A real per-ticker dossier
  fetch with no screenshot returns verdict=review/reason=chart_missing for
  everything, masking R11. So the drawer derives the R1–R11 trace from the
  candidate fields + the engine's R11 constants instead — illuminating about
  R11 without the chart-missing noise, and honest about which rules need a
  chart/book.

## How we fixed it

Extended the existing `/api/engine` proxy (dashboard-side, allowed) to forward
the full PIT parameter set for `candidates` (as_of/dte/delta/min_ev/
universe_limit) and added a `dossier` action. Everything else consumes the
endpoints as-is. `engine/` diff is empty.

## Evidence

- `npm run build` (Next 16, Turbopack): **EXIT 0**, `/cockpit` in the route
  manifest. (One TS fix mid-build: nullable `cvarPct` narrowing in the
  distribution bar.)
- Live data path verified end-to-end: `next start -p 3001` →
  `GET /api/engine?action=candidates&as_of=2026-03-20&universe_limit=60` →
  engine `:8787` returned **HTTP 200** with real candidates (8 trades, scanned
  60/503). `GET /cockpit` → **HTTP 200** ("Decision Cockpit" present).
- Trust story renders on real data: at `as_of=2026-03-20` the engine VIX is
  **28.97 (> 25 → R11 active)**, and the returned set includes top-bin picks
  **AZO prob_profit 0.91** and **AJG 0.94** — exactly the candidates the
  CalibratedProb flags red ("crisis-realized ~0.57") and the dossier R11 row
  marks FIRES. AXON (0.83) renders as trusted green. The distribution bar shows
  AJG's tiny cvar5 (−$68, deep OTM) vs AZO's −$8,877 tail.
- `engine/` diff empty; FILE_MANIFEST coverage OK; the Decision-Layer Lane
  Claim gate does not apply (no decision-layer files touched).

## Unresolved / handoff

Engine-side follow-ups (do NOT implement without consent; out of this display
PR's scope):

1. Serialize `frame.attrs["drops_summary"]` into `/api/candidates` so the
   funnel can show per-gate drop reasons.
2. Add an `as_of` param to `/api/vix` (and/or emit `vix_level` per candidate
   row) so the banner VIX is PIT at arbitrary as_of.
3. Emit per-sector / per-name book exposure (or a sector for each candidate) so
   R9 sector bars can be continuous rather than verdict-only.

Dashboard follow-ups: wire the dossier drawer to the real `/api/tv/dossier`
when a screenshots dir + book are available (R2–R10 live); add a TradingView
chart pane driven by the selected row (TradingView stays the chart surface).
