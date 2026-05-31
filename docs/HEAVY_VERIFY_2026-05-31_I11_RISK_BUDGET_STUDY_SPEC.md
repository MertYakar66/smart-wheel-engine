# Heavy-Verify Campaign 2026-05-31 — I11 (SPEC): R11 risk-budget parameter study

**Status:** observe-only **design spec — execution pending operator nod.** `engine/`
is NOT touched by this study; it reads existing campaign artifacts + the VIX CSVs and
prints tables. The R11 reviewer it parameterizes is a **separate gated card**
(explicit-ask + decision-layer lane-claim + §2 second-read), not built here.

**Predecessors:** I9 (recalibration does not generalize across crises) and I10 (no
simple PIT signal cleanly *detects* the transition). Together they close the
precision-fix path: you can neither **calibrate** the unstable crisis quantity (I9) nor
**gate** it (I10). The remaining move is to **manage exposure regardless** — the
§2-clean, downgrade-only **risk-budget rule** I10 routed to as the robust default
(REMEDIATION cat. B): *cap top-bin (`prob_profit > 0.90`) size in elevated-vol regimes.*
This study picks that rule's parameters by measurement, not guess.

---

## 1. Objective

Choose the parameters of the **R11 risk-budget reviewer**:
1. the **"elevated vol" trigger** — a VIX-*level* threshold θ, and
2. confirm the **confidence cutoff** (`prob_profit > 0.90`, inherited from I1/I9/I10),

by measuring, **leave-one-crisis-out (LOCO)**, the trade-off the rule actually makes:
**2020-onset bleed averted** vs **premium forgone** at the 2020 recovery and across the
2022 bear. The output is either a coarse θ that makes that trade-off favorably
asymmetric, **or** the null-result fallback (§7).

R11 is **downgrade-only** (`proceed → review` on a matched candidate); it introduces no
position-size output and never touches `ev_raw`/`ev_dollars` (§2-clean). The size-down
is the reviewer's effect at review time, carried by a **computed, regime-matched warning
payload** (§6), not a new sizing channel.

## 2. Why this is not redundant with I10

I10 tested `rv_ratio` (RV30/RV252), trailing drawdown, `rv_accel`, and `rv30` level —
**never raw VIX level** — and its decisive failure was a **ratio artifact**: `rv_ratio`
is *higher at the 2020 recovery (2.69) than at the crash onset (1.69)*, so a gate on it
refuses the best entry. **VIX level inverts that ordering correctly** (~80 at the
March-2020 onset vs ~40–50 at the April recovery). Steer: the trigger must be a **level,
not a ratio** — a normalized/differenced vol feature (incl. `iv_rank`) re-inherits I10's
failure. So:

- **Primary signal:** VIX **level**, `data/bloomberg/sp500_vix_full.csv` (`close`,
  `instrument=="vix"`), joined to each candidate's `as_of`.
- **Secondary signal (one, tested only for the §6 honesty check):** term-structure
  **backwardation**, `data/bloomberg/vix_term_structure.csv` (`vix > vix_3m`) — a
  market-*state* object, not a trailing ratio; the one feature that might flag a
  *calm-level* onset if such a cell exists.
- **Excluded as primary:** `iv_rank` / any percentile-normalized vol (ratio-like).

## 3. Inputs (all observe-only, already on disk)

| Input | Source | Use |
|---|---|---|
| Per-trade realized rows | regenerate via `i1_calibration.py` (`load_realized()` → `realized_pnl_synth`, `prob_profit`, `as_of`, `hmm_regime`, `ticker`); the `.parquet` is gitignored, the driver re-derives it | the P&L the rule acts on |
| VIX level | `data/bloomberg/sp500_vix_full.csv` `close` | the trigger θ |
| Term structure | `data/bloomberg/vix_term_structure.csv` `vix,vix_3m` | §6 secondary check |
| Crisis cells | `hmm_regime=="crisis"`, years 2020/2021/2022/2025 (well-powered: 2020 n=158, 2022 n=86) | LOCO folds |

## 4. Method — LOCO cost/benefit of the size-down rule

For the top bin (`prob_profit > 0.90`):

1. **Define the rule** at threshold θ: a candidate whose `as_of` VIX `close > θ` is
   *downgraded* (in the study, "downgraded" = removed from the booked set; the realized
   `realized_pnl_synth` it carried is the P&L the rule forgoes-or-averts).
2. **LOCO:** hold out one crisis year; (θ is a fixed coarse grid value, §5 — *not* fit on
   the others, so there is no per-fold optimization to overfit); apply to the held-out
   crisis and to the adjacent recovery months.
3. **Score, per held-out crisis:**
   - **bleed averted** = −Σ(`realized_pnl_synth` < 0 on downgraded onset trades),
   - **premium forgone** = Σ(`realized_pnl_synth` > 0 on downgraded recovery/benign/
     sustained-bear trades),
   - report both per-contract and book-aggregate, with the count downgraded.
4. **The headline deliverable is the 2022 cell.** I10 P3 established 2022 vol is elevated
   *all year*, so a VIX-level rule **fires through 2022** — there is **no detection miss**
   here. 2022's cost is therefore a **false-positive opportunity cost**: the rule sizes
   down a year that netted **+$5k** for the wheel. The study must quantify that forgone
   premium and weigh it against the 2020-onset bleed averted (−$1,305/contract at 0.57
   realized). The decision is **asymmetric and tail-aware**: a *bounded, near-certain*
   opportunity cost to avoid a *large, unforecastable* loss is the correct treatment of a
   tail you can neither predict (I9) nor detect (I10).

## 5. Threshold discipline — coarse, robust-not-optimal

LOCO over 2–3 well-powered crises is 2–3 folds; optimizing θ against that overfits to
*which crisis was held out*. So θ is searched on a **coarse round-number grid**
`{20, 22.5, 25, 27.5, 30}` and the deliverable is the **simplest cut that survives every
fold** (e.g. "VIX > 25"), reported as **robust-not-optimal**. No fine tuning (no
"VIX > 23.7"). Same discipline that tightened the I10 headline from 56pp to ≥26pp.

## 6. Honesty check — does any real-VIX cell show low level + fat tail?

The in-sample crises do **not** (2020: high VIX; 2022: VIX elevated all year). But the
claim "VIX-level works" must be distinguished from "VIX-level works on the data we have."
So the driver **scans the full real VIX series** for any `as_of` with **VIX `close` below
θ AND a fat realized left tail** (e.g. mean `realized_pnl_synth` in the bottom decile
among top-bin trades). If such a cell exists, that is the residual gap — **name it**, and
test whether **term-structure backwardation** (§2 secondary) flags it where the level
misses. If none exists, state that explicitly with the scan window.

## 7. Null-result exit (defined fallback — not an open question)

If **no** coarse θ makes the LOCO trade-off favorably asymmetric — i.e. forgone premium
swamps averted bleed across folds, or §6 surfaces an un-flagged low-level fat tail the
secondary can't catch — the study **concludes detection-by-VIX-level fails** and R11
falls back to the **unconditional top-bin haircut**: downgrade `prob_profit > 0.90`
`proceed → review` in **all** regimes, **no detector**. This is still §2-clean, needs no
signal, and costs some calm-regime premium. It is the honest answer if detection fails,
and it routes there *by design*.

## 8. Acceptance criteria → the R11 form it produces

| Study outcome | R11 form (built in the separate gated card) |
|---|---|
| A coarse θ with favorable LOCO asymmetry | **Conditional** R11: downgrade top-bin when VIX `close > θ`, payload carries the matched-cell loss |
| Null result (§7) | **Unconditional** R11: downgrade top-bin in all regimes, no VIX gate |

In **both** outcomes R11 is a downgrade-only reviewer carrying a **computed,
regime-matched** warning string — the matched-cell realized loss (e.g. the −$1,305/
contract at 0.57 assignment for the crisis top bin), **computed from the realized rows at
build time, not hardcoded**, so it stays honest as data updates. This recovers most of a
hard size-cap's protective guarantee (a downgrade otherwise delegates sizing to human
discretion at the exact onset where judgment fails) without introducing a position-size
output.

## 9. Driver contract (to author at execution)

`docs/verification_artifacts/campaign_2026-05-31/i11_vix_budget_study.py`, observe-only,
prints to `raw_output/i11_vix_budget_study_RAW.txt`. It must print, with Wilson/CI where
applicable: (a) the per-crisis LOCO cost/benefit table (§4), (b) the 2022 forgone-premium
headline (§4.4), (c) the coarse-θ survival table (§5), (d) the §6 low-level-fat-tail scan
result. Reproduce stub:

```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py        # realized rows
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i11_vix_budget_study.py   # this study
```

## 10. Scope / §2 boundary

Observe-and-document only. No `engine/` change, no re-baseline (R11 is a refusal-only
reviewer; it never alters `ev_raw`/`ev_dollars` — same class as R6–R10). The R11 build is
gated on this study's result and goes through the decision-layer lane-claim + §2
second-read. This spec commits nothing beyond the study itself.
