---
id: Phase-2-spec
title: Phase 2 skew-surface execution spec — turnkey file:line plan for the IV-skew wiring
kind: feature
status: held
terminal: autonomous session 2026-06-23
pr:
decisions: []
date: 2026-06-23
headline: docs-only turnkey spec for the supervised, EV-moving Phase 2 (moneyness IV skew surface → connector accessor → skew_dynamics sizing + option_pricer/ev_engine BSM-IV seam) verified vs origin/main @ 21e489d; no engine code
surface: [docs/PHASE2_SKEW_EXECUTION_SPEC.md]
---

## Goal

Make the supervised, EV-moving Phase 2 of the wiring campaign (`docs/WIRING_CAMPAIGN.md`
lines 182–200) fast and low-risk by producing a **turnkey, file:line-accurate** spec for wiring
the banked **moneyness IV skew surface** (`#417`'s `data/bloomberg/broad_pull/iv_surface/`,
1.94M rows, dormant `BroadPullLoader`) into its three consumers — a new
`data_connector.get_iv_surface` accessor → `engine/skew_dynamics.py` (the `skew_mult` sizing
path) + the `engine/option_pricer.py` / `engine/ev_engine.py` BSM fair-value IV seam. Docs only;
the consume is a PANEL, EV-moving, re-baseline-coupled change that lands under §2 supervision.

## What we tried

- **Verified every line against `origin/main` @ `21e489d`** (post #416/#417/#418), via a 5-agent
  read-only research workflow (skew_dynamics · option_pricer · connector+loader+schema · tests ·
  governance/re-baseline), each survey **independently adversarially re-checked** by a second
  agent. Then spot-verified the load-bearing anchors first-hand in a fresh main worktree before
  writing.
- **Adjudicated the campaign doc, did not relay it.** The `WIRING_CAMPAIGN.md` Phase-2 row
  (line 191) names deliverables the code already has or lacks differently than stated — the spec
  corrects them rather than propagating.

## What worked

- **Three campaign corrections caught by the adversarial pass (the load-bearing finding):**
  - **`vanna/charm/volga` already exist** on the scalar path (`option_pricer.py:524/530-536/540`,
    consumed by `dealer_positioning`) — not a Phase-2 deliverable; only the *vectorized* path
    (`vectorized_bs_all_greeks` :1181) lacks them.
  - **Butterflies do NOT exist in `skew_dynamics`** (only `risk_reversal` :167) — a butterfly
    lives only in the dormant SVI module `engine/volatility_surface.py`. Phase-2 "butterflies" is
    new code, not wiring.
  - **No true 25Δ column** in the 5×5 surface (`{90,95,100,105,110}`; wings `{80,120}` empty) —
    the connector must map `{90}`→put/`{110}`→call as a **labeled moneyness proxy**; the emitted
    RR is a 90/110-moneyness RR, not a literal 25Δ RR.
- **Pinned the EV-moving seam exactly:** `sigma=trade.iv` at `ev_engine.py:376` (fair value,
  `edge_vs_fair` :380-381) is the single highest-leverage change; the distribution vol at
  `:671` shares `trade.iv` (a deliberate strike-vs-ATM decision for the panel).
- **Surfaced two structural facts:** the connector's `_load`/`_FILES` path **cannot read the
  broad-pull `.gz`** (must delegate to `BroadPullLoader.series` :352, float32); and the
  **lane-claim CI gate is silent** for `option_pricer`/`skew_dynamics`/`data_connector`
  (`check_lane_claim.py:80-83` = trio only) → the human §2 panel is the sole governance gate.

## What didn't

- The campaign's "vanna/charm/volga + butterflies" framing would have produced a spec for work
  that is already done (greeks) or absent (butterflies) — the adversarial verify is what flipped
  those from assumptions to corrections.
- Two decision points are left to the §2 panel (not guessed): strike-IV vs ATM-IV for the
  distribution vol (`ev_engine.py:671`), and whether to scope butterflies / the vectorized-greek
  gap into Phase 2 at all.

## How we fixed it

Shipped `docs/PHASE2_SKEW_EXECUTION_SPEC.md` (per-consumer finding/fix/§2-role/PANEL, tests to
add + keep, the §2-panel checklist, the ATM single-source reconciliation + `#378`-first
ordering, and Phase R coupling) + a FILE_MANIFEST row + this worklog. Held for review; no engine
code.

## Evidence

- Anchors re-verified on `origin/main` @ `21e489d` (grep + read): `skew_dynamics.py:57/151/167/177/229`,
  `option_pricer.py:126/435/524/1073/1181`, `ev_engine.py:88/116/147/321/371/376/380/671`,
  `data_connector.py:479/507/528/563/900` (no `get_iv_surface`), `wheel_runner.py:812/1538/1540/1557/1561`,
  `broad_pull_loaders.py:269/346/352/387`, `check_lane_claim.py:80-83`.
- Re-baseline reproducers confirmed present: `backtests/regression/{s27:84,s32:84,s34:80,s35:78}`.

## Unresolved / handoff

- Execution is the supervised session: land **`#378` first**, then Phase 2 as a single PANEL PR
  (connector accessor → skew_dynamics sizing → option_pricer/ev_engine BSM-IV seam), held + §2
  panel, **before** the single Phase-R re-baseline that absorbs its `ev_raw` shift.
- The ATM `{100}` slice must reconcile with `implied_vol_atm` (`data_connector.py:900`) and the
  Phase 3C term structure — pin the overlap so the engine carries one ATM IV, not three.
