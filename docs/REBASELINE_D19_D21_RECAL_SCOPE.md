# Scope — Coordinated Re-baseline: D19 (exit costs) + D21 (horizon units) + Probability Recalibration

**Status: DRAFT for operator review. Planning only — no code changes, no implementation.**
**Date: 2026-06-01. Author: engine session (read-only research, 7-agent map).**
**This file is an untracked draft (not committed, not a PR). The running Theta pull was not touched.**

> Line numbers below were read from the on-disk working copy and should be re-confirmed at
> execution time (the codebase moves). Items the research could not verify exactly are marked
> **[CONFIRM]** rather than asserted.

---

## 0. TL;DR

Three deferred decision-layer changes are **entangled and must re-base in one pass**:

- **D21** — the forward-distribution sampler is fed **calendar DTE** but indexes **trading-day bars**, so the modelled horizon is **~46 % too long** (35 cal → 24 bars), over-dispersing the terminal distribution by **~21 %**. This currently **inflates `prob_assignment`** and **deflates `prob_profit`**.
- **D19** — `EVEngine.evaluate` computes the exit-leg commission + slippage (~$1–4/contract) into `total_transaction_cost` but **never subtracts it from `ev_raw`**, so EV is mildly overstated.
- **Recalibration** — the top `prob_profit` bin is over-confident (I1: ~0.96 forecast vs ~0.57 realized in crisis, **−27 pp** miss). A prior recalibration map (W2-C) cut ECE 3.17 → 1.29 pp on calm tape **but failed leave-one-crisis-out (LOCO)** — and crucially, **it was fit on D21-deflated `prob_profit`**.

**The entanglement, precisely:** D21's over-long horizon *deflates* `prob_profit` today, which drags the forecast **down toward** the realized rate and therefore **masks part of the top-bin over-confidence**. Fixing D21 alone *raises* `prob_profit` (verified example **0.833 → 0.886**) → the measured calibration gap gets **worse**, not better. So D21 cannot ship without re-measuring calibration and re-attempting the recalibration on the corrected probabilities; and because both D21 and D19 move EV-authority output, every backtest baseline and the published calibration matrix re-base together. One commit, one backtest re-run, one baseline refresh.

---

## 1. Standing constraints (carried into this work)

- **Planning only now.** Nothing in this doc is to be implemented until you green-light it.
- **Do not stop the data pull** for any reason (standing operator directive). None of this work goes near the running worker, its larder, or the Theta connection.
- **§2 invariant holds throughout** (CLAUDE.md §2): no tradeable candidate bypasses `EVEngine.evaluate`; reviewers downgrade-only; dealer clamp `[0.70, 1.05]` immutable.
- **Branch + PR, human merge, no auto-merge.** Decision-trio change → full suite + independent §2 read (see §7).

---

## 2. Current status of the three pieces

| Piece | What it is | Status | Pinned at |
|---|---|---|---|
| **D21** | calendar→trading-bar horizon conversion | **Confirmed, fix authored, NOT applied** (deferred 2026-05-30) | helper `calendar_days_to_trading_bars` `engine/forward_distribution.py:50-63`; deferral docstring `:334-340`; `DECISIONS.md:1179-1214`; arithmetic test `tests/test_audit_improvements.py:109-120` |
| **D19** | net the expected exit-leg cost into `ev_raw` | **Confirmed, fix authored, NOT applied** (deferred 2026-05-30) | insertion-point note `engine/ev_engine.py:477-485`; `DECISIONS.md:1109-1145` |
| **Recalibration** | close/quantify top-bin `prob_profit` over-confidence | **Researched, no decision entry yet**; prior map fails LOCO | `i1/i6/i9` under `docs/verification_artifacts/campaign_2026-05-31/`; `docs/HEAVY_VERIFY_2026-05-31_I9_FIX_GENERALIZATION.md` |

---

## 3. What each change touches

### 3.1 D21 — forward-distribution horizon units
- **Root cause.** Callers pass **calendar** DTE; the three samplers index **trading-day bars**:
  - empirical non-overlapping — `engine/forward_distribution.py:128` (`log_prices[h:] - log_prices[:-h]`)
  - block bootstrap — `:182` + accumulation loop `:196-203`
  - HAR-RV cascade — `:299` (`standard_t(df, size=(n, h))`)
- **Entry points (calendar DTE in).** `engine/wheel_runner.py:1294` (`rank_candidates_by_ev`, `horizon_days=dte_target`), `:2475` (strangle leg), `:3014` (covered-call leg) — all into `best_available_forward_distribution` (`:310-386`).
- **Fix scope (boundary-only, no algorithm change).** One conversion at the orchestrator entry (`best_available_forward_distribution`, ~`:341`): `horizon_days = calendar_days_to_trading_bars(horizon_days)` — then all three samplers receive the correct bar count. The helper already exists and is unit-tested.
- **Direction of effect.** Correcting the over-long horizon **narrows** the terminal distribution → **`prob_assignment` ↓** (less left-tail mass below strike) and **`prob_profit` ↑** (fewer downside paths). Magnitude: ratio 35/24 ≈ 1.458 (~46 % excess horizon), √1.458 ≈ 1.208 (~21 % excess std). Verified candidate: `prob_profit` 0.833 → 0.886.
- **§2 surface.** Upstream of `evaluate`; a units fix in the sampler. No new path that rescues a candidate. Reviewers stay downgrade-only.

### 3.2 D19 — exit-leg cost netting
- **Defect.** Entry commission + slippage are netted into `net_premium_in` (`engine/ev_engine.py:327`) and flow into the P&L (`:353`). Exit commission + slippage are **computed** (`:320-324`, exit slippage = entry slippage by symmetry) and stored in metadata (`total_transaction_cost`, `:552`) but **never subtracted** — the comment at `:354-362` states the intent, the subtraction is missing. So `ev_raw = mean(pnls)` (`:383`) carries only the entry-side penalty.
- **Fix scope.** At the deferred-note site (`:477-485`, after `expected_days_held`, before the regime multiplier): subtract `expected_exit_cost = min(1.0, prob_profit + prob_stop_terminal) * (exit_commission + exit_slippage)` from `ev_raw`. A **mean-shift only** — leaves `prob_profit`, `prob_assignment`, and all distribution-shape fields (`std_pnl, skew_pnl, cvar_5, omega`) untouched. Cost model: `engine/transaction_costs.py:30-46` (commission $0.65) and `:77-155` (Almgren-Chriss / Kyle-λ slippage).
- **Direction of effect.** `ev_dollars` ↓ by ~$1–4/contract (example `+$5.50 → +$4.15`); marginal candidates near `min_proceed_ev` slide proceed → review.
- **§2 surface.** `expected_exit_cost ≥ 0` always → D19 can only **reduce** `ev_raw`; it can never rescue a negative-EV trade. Invariant preserved by construction.

### 3.3 Recalibration — the top-bin over-confidence
- **The finding (I1).** Calibrated in mid-bins (±5 pp); over-confident at the top, worst in crisis (~−27 pp; LOCO-2020 predicted 0.84 vs realized 0.72; regime-holdout 0.843 vs 0.670). Crisis top-bin realized rate is **unstable across crises: 0.37–0.93 (56.5 pp spread)** — which is why a single static histogram map struggles to generalize.
- **The prior attempt (W2-C, `i6_deepening.py:89-164`).** Per-bin and bin×regime map; ECE 3.17 → 1.29 pp on calm 2024-26 OOS. **But the LOCO gate (`i9_calibration_generalization.py`) failed** — and it was fit on **D21-deflated** `prob_profit`. That is the key reason it deserves a clean re-run (see §6), not a write-off.
- **Fix scope.** Re-measure calibration on D21-corrected probabilities, then re-fit + re-gate the map (procedure in §6). If the map still fails LOCO on corrected inputs, escalate to the **structural POT-GPD route** (recommended-but-untested in `HEAVY_VERIFY_2026-05-31_I9_FIX_GENERALIZATION.md`) — a decision point, not assumed in-scope here.

---

## 4. Dependency order

**D19 and D21 are operationally entangled but logically independent;** recalibration is **serial — strictly after D21.** No circular dependency exists. The clean pipeline:

```
Step 1  Apply D21 (horizon units)         ── code, forward_distribution.py
Step 2  Apply D19 (exit-cost netting)     ── code, ev_engine.py   (order vs Step 1 is free)
Step 3  Re-MEASURE calibration on the     ── re-run i1 on D21-corrected prob_profit;
        D21-corrected probabilities          the masked gap now shows at its true size
Step 4  Re-ATTEMPT the recalibration map  ── re-fit i6 W2-C + re-gate i9 LOCO  (see §6)
        └─ passes LOCO → ship the map
        └─ fails  LOCO → escalate decision: POT-GPD structural fix, or accept gap +
                          keep R11 (downgrade-only) as the operational mitigation
Step 5  Re-run full backtest suite        ── with D21+D19 (and map, if shipped) applied
Step 6  Refresh PROB_PROFIT_CALIBRATION    ── the published matrix
Step 7  Re-pin fixture-locked EV tests    ── new expected ev_dollars/prob_profit
Step 8  DECISIONS.md: flip D19/D21 to      ── add a recalibration decision entry
        Resolved + post-ship validation
```

Two things to keep honest about the ordering:
- **Why "together" rather than three point-fixes:** churn minimization. Each of D19 and D21 alone would flip the byte-identical backtest baselines once; shipping solo means flipping them twice. It is an operational constraint, not a logical one — D19 does **not** require D21's conversion to be present.
- **Why recalibration cannot precede D21:** the map must be learned on the corrected (higher) `prob_profit`. Fitting on today's D21-deflated values is exactly the prior failure.

---

## 5. What re-bases (every `prob_assignment` / `prob_profit` consumer that moves)

Producers: `prob_assignment` `engine/ev_engine.py:363-364, 544`; `prob_profit` `:393, 543`.

| Downstream | How it moves | Where |
|---|---|---|
| **Backtest numbers** (S27/S32/S34/S35/S38/S40/S43/S44/S46 campaigns) | every candidate's EV/rank shifts (D21 narrows dist, D19 lowers EV) → ρ, NAV deltas, hit-rate all re-base; **byte-identical regression baselines flip** | `docs/ENGINE_BACKTEST_*.md`, `backtests/regression/_common.py:198,504-505,865-866` — **[CONFIRM canonical snapshot registry/IDs]** |
| **Calibration band** (top-bin over-confidence, the decile reliability table) | D21 raises `prob_profit` → measured top-bin gap **widens** (over-confidence looks worse) before any map is applied | `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`; `i1_calibration.py` |
| **S-claims** (deployment/capability/structural framings) | the "engine systematically over-confident on top-bin" claim and the 10-config MAD numbers (6.16 pp mean; top-bin −14.8 to −17.6 pp) are *the* framing behind the F4 + R10/R11 bundle; all re-base | `PROJECT_STATE.md:29-32,162-192`; `LAUNCH_READINESS.md:186-194`; `HEAVY_VERIFY_2026-05-31_INDEX.md:100` |
| **Premium-correction pilot risk axis** (Refinement 2) | pilot's `gap = realized_itm − eng_prob_itm` is currently biased **downward** because `eng_prob_itm` is D21-inflated; D21 fix lowers `eng_prob_itm` → the axis **shifts up** (less negative). Pilot's "risk looks aligned, no under-modelling" conclusion is **conditional on D21 unfixed** — re-run the pilot read after re-baseline | `docs/PREMIUM_CORRECTION_PILOT.md:139-198`; `studies/premium_correction/` |
| **R11** (elevated-vol top-bin size-down: `prob_profit > 0.90` AND `vix > 25`) | **direct consumer** — D21 raises `prob_profit`, so the >0.90 population *grows* and R11 fires **more**; note R11 is itself the interim mitigation for the very over-confidence being re-based | `engine/candidate_dossier.py:534-565`; D23 `DECISIONS.md:1255-1330` |
| **R1 / R5** (block on neg/non-finite EV; EV threshold) | indirect via `ev_dollars`: D19 lowers EV → marginal candidates slide proceed→review (R5), a few borderline → block (R1) | `engine/candidate_dossier.py` (R1 `:305-312`, R5 `:348-356`) |
| **API / dashboard** | `/api/candidates` surfaces `prob_profit`/`prob_assignment`; displayed confidence shifts | `engine_api.py:687-739` |
| **Fixture-pinned EV tests** | hard-coded expected `ev_dollars` change | `tests/test_f4_rv_widening.py`, `tests/test_strangle_ev_ranker.py` — **[CONFIRM exact test/class names; the names cited in `DECISIONS.md` (`TestRanksNeverRescues`, `TestF4CasesRanker`) did not grep-match and must be re-located]** |
| **`expected_days_held` / `ev_per_day`** | hold-time weights on `prob_profit` → D21 shift cascades into per-day scaling | `engine/ev_engine.py:467-474, 542` |

**Consumer-completeness note:** beyond the above, `prob_profit` is read by `wheel_tracker.py` (win-rate display, not authority) and the dashboard (display). The critic's sweep found **no consumer that bypasses `EVEngine.evaluate`** — the blast radius is fully inside the EV authority surface. **[CONFIRM with a fresh grep at execution time.]**

---

## 6. Validation — re-attempt LOCO recalibration on D21-corrected probabilities

The prior recalibration is **not written off**; it was fit on biased inputs and earns a clean re-run.

**Procedure (after Steps 1–2 land D21+D19):**
1. **Re-realize** all calibration rows against the D21-corrected forward distributions — re-run `i1_calibration.py` (`load_realized` `:68-94`); outputs the corrected reliability table (Wilson-CI bins, Brier, ECE). This is the **re-measure** step — expect the top-bin gap to *widen* vs today.
2. **Re-fit** the recalibration map on a LOCO-train split (2020-2023 excluding the held-out crisis), per-bin and bin×regime — `i6_deepening.py` W2-C (`:127-141`).
3. **Re-gate** with leave-one-crisis-out — `i9_calibration_generalization.py` T1 walk-forward (`:135-140`), T2 LOCO **for both 2020 and 2022** (`:143-147`), T3 regime-holdout crisis-only (`:150-152`).

**Success criterion (propose; confirm with operator):** the re-fit map must **hold up out-of-crisis on a crisis it never saw** — i.e., T2/T3 top-bin predicted-vs-realized within a pre-registered tolerance (e.g. ≤ ~5 pp), not just an improved calm-tape ECE. The prior map's failure was specifically T2/T3, so calm-tape ECE alone is **not** a passing bar.

**Two honest caveats to carry:**
- **The D21 contribution to the −27 pp is not yet decomposed.** I1 measures the *total* miss; no document isolates "how much is the D21 horizon artifact vs. structural tail under-modelling." The masking is a mechanically sound, well-motivated hypothesis — **the re-baseline is the experiment that decomposes it.** State it as a hypothesis-under-test, not a settled fact.
- **The map may still fail on corrected inputs.** The crisis realized-rate spread (0.37–0.93) is a property of the *outcomes*, not of D21. If LOCO fails again post-correction, that is itself a finding: it points to the **structural POT-GPD route** (untested) rather than a histogram map, and that becomes an explicit operator decision (wire POT-GPD now, or defer it and lean on R11). Do not silently ship a map that only helps calm tape.

---

## 7. Test + §2 plan (decision-trio change)

- **Full suite, not a subset.** D21 touches `forward_distribution.py`; D19 touches `ev_engine.py` — both are decision-layer. Per `TESTING.md:183-184` and `LAUNCH_READINESS.md:98-122`, `ev_engine.py`/`wheel_runner.py` changes run the **full** suite (invariants cross-cut), after the launch-blocker subset: `pytest tests/test_audit_invariants.py tests/test_dossier_invariant.py tests/test_authority_hardening.py tests/test_audit_viii_unit_invariants.py tests/test_audit_viii_e2e.py tests/test_audit_viii_real_data_smoke.py tests/test_launch_blockers.py -v` (`TESTING.md:145-152`).
- **Backtest-regression gate (long pole).** `LAUNCH_READINESS.md:186-194` requires the backtest-regression harness green for any change touching `ev_engine.py`/`wheel_runner.py`/`forward_distribution.py`. This harness is **~4–5 h** and is **excluded from per-PR CI** — it must be run **serially before merge**, off the normal CI loop. Schedule accordingly.
- **Independent §2 read.** A separate reviewer confirms: (a) no new path converts a non-tradeable candidate to tradeable; (b) D19 only ever subtracts (`expected_exit_cost ≥ 0`); (c) D21 is a sampler-units fix upstream of `evaluate`; (d) dealer clamp `[0.70,1.05]` and downgrade-only reviewer contract untouched.
- **Lane-claim.** Decision-layer edits require the lane-claim block on board #113 (CI gate `scripts/check_lane_claim.py`); claim `ev_engine.py` + `forward_distribution.py` (and `wheel_runner.py` if call sites are touched) before branching. High-contention files — keep the board armed for the whole re-baseline to avoid a double-build (precedent: #107 vs #109).
- **Branch + PR, human merge, NO auto-merge.** This is the opposite of the #316 quick-win; nothing here auto-merges. Commit/PR per `COMMIT_GUIDE.md` (Changed/Why/Tested/§2 surface/Tried-but-rejected/Unresolved).
- **DECISIONS.md annotation.** On landing, flip D19 and D21 from DEFERRED → Resolved with a dated **Post-ship validation** note (the D23 entry `:1255-1330` is the template: status, book-level impact, reframe, pinned-by test), and add a new decision entry for the recalibration outcome (map shipped, or POT-GPD escalated/deferred).
- **No implementation now.** This doc is the plan; execution waits on your go.

---

## 8. Open questions to settle before execution

1. **Recalibration route if LOCO fails again:** wire the structural POT-GPD fix into `prob_profit` (untested — needs its own prototype + LOCO gate), or defer it and keep R11 as the operational mitigation with the gap documented? (§6)
2. **Pre-register the LOCO pass/fail tolerance** (proposed ≤ ~5 pp on T2/T3 top bin) so the re-run can't be graded post-hoc.
3. **Confirm the canonical backtest snapshot registry** — which exact `ENGINE_BACKTEST_*` docs/IDs and rolling-window campaigns re-base, and whether all ~2,600 regression baselines flip or a subset. (§5 **[CONFIRM]**)
4. **Re-locate the fixture-pinned EV tests** — the class names in `DECISIONS.md` did not grep-match; identify the real tests carrying hard-coded `ev_dollars`. (§5 **[CONFIRM]**)
5. **Decompose intent:** do you want the re-baseline to explicitly report the D21-artifact-vs-structural split of the −27 pp (a deliverable), or just the corrected, recalibrated numbers? (§6)
6. **Re-run the premium-correction pilot read** post-re-baseline (its risk axis shifts up) — fold into this work or keep separate?

---

## 9. Guardrails / non-goals

- No implementation until green-lit; this is planning.
- Do not patch D21 or D19 inside the premium-correction pilot or any study — they re-base engine-wide, once, here.
- Do not lower the 504-day OHLCV history floor to chase coverage.
- Do not ship D19 or D21 **solo** in a routine bug sweep — solo D21 makes the measured calibration gap visibly worse with no offset; solo D19 trips the baselines a second time. They land together or not at all.
- Do not touch the running data pull.
