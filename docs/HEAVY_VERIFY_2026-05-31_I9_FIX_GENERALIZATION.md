# Heavy-Verify Campaign 2026-05-31 — I9: does the calibration fix generalize? (B-verification gate)

**Investigation:** the gate before any EV-authority re-baseline — does the recalibration
fix that I6-C demonstrated (ECE 3.17→1.29pp out-of-sample) actually generalize to an
**unseen crisis**, the regime where the top-bin over-confidence actually bites?
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i9_calibration_generalization.py`
(analysis on `raw_output/i1_realized_rows.parquet`). **Raw:** `raw_output/i9_generalization_RAW.txt`.
**Status:** observe-and-document; `engine/` not modified.

**Scope:** this tests the **recalibration-layer** fix (a histogram map on `prob_profit`,
the one I6-C demonstrated). It does **NOT** test wiring POT-GPD into `prob_profit` —
that needs an engine prototype and is out of observe-only scope. Conclusions below
apply to recalibration.

---

## VERDICT — the gate is NOT cleared; this qualifies I6-C

> **The recalibration fix does NOT generalize to unseen crises.** It helps in
> calm/recent regimes (where the over-confidence is mild) but UNDER-corrects crises
> (where it is severe) and OVER-corrects benign transitions. The decisive reason: the
> (crisis-regime, top-bin) realized rate is **wildly unstable across crises — 0.57
> (2020) / 0.93 (2021) / 0.83 (2022) / 0.37 (2025), a 56.5pp spread** — so neither a
> static nor a regime-conditional histogram map can capture it. I6-C's promising
> result was an artifact of a benign 2024-2026 test window. **Do NOT build the
> re-baseline around a recalibration layer.** The remaining candidate is the
> structural route (POT-GPD into `prob_profit`, which responds to *current* realized
> tail conditions rather than a backward-fit label) — but that is **untested** and
> needs its own prototype + validation before it earns a re-baseline.**

Confidence: **high** on the decisive cells (leave-one-crisis-out 2020 n=358, 2022
n=270; regime-holdout crisis n=300). The instability conclusion rests on the large
2020 vs 2022 cells (0.57 vs 0.83 = 26pp alone); the small 2021/2025 cells (n=15/19)
amplify but are not load-bearing.

This **vindicates the "verify the fix generalizes before committing to a re-baseline"
discipline** — and saves the re-baseline cycle from being built on a fix that doesn't work where it matters.

---

## T1 — Walk-forward (train on years < Y, test on Y)

| test year | raw ECE | +bin-recal | +bin×regime | top-bin (>0.90) raw → realized → recal |
|---|---|---|---|---|
| 2021 (bull) | 5.5pp | 7.6 ✗ | 7.2 ✗ | 0.920 → **0.906** (already calibrated!) → 0.74 (**over-corrected**) |
| **2022 (bear)** | 12.0pp | 13.6 ✗ | 12.1 ≈ | 0.922 → 0.785 → 0.81 (top-bin nudged, overall ECE worse) |
| 2023 | 4.1pp | 1.5 ✓ | 6.5 ✗ | 0.923 → 0.853 → 0.80 |
| 2024 | 2.9pp | 1.0 ✓ | 1.2 ✓ | 0.925 → 0.793 → 0.80 |
| 2025 | 3.0pp | 0.7 ✓ | 0.8 ✓ | 0.923 → 0.796 → 0.81 |

Recalibration **helps once enough calm history accumulates (2023-2025: this is the
regime I6-C tested)** but **fails at the regime transitions** — it makes the 2021 bull
*worse* (over-correcting a well-calibrated 0.906 down to 0.74) and doesn't fix the
2022 bear's overall ECE.

## T2 — Leave-one-crisis-out (train EXCLUDING the crisis, test on it — the hard test)

| test | raw ECE | +bin | +bin×regime | top-bin raw → realized → recal |
|---|---|---|---|---|
| **2020 crash** (train all-but-2020) | 6.2pp | 3.2 | 2.2 | 0.929 → **0.723** → 0.84 (**under-corrects by ~12pp**) |
| **2022 bear** (train all-but-2022) | 12.0pp | 10.8 | 10.8 | 0.922 → **0.785** → 0.81 (barely moved) |

Even with *every other year* in the training set, the recalibration leaves the crisis
top-bins materially over-confident (predicts 0.84 / 0.81 when reality is 0.72 / 0.79).

## T3 — Regime-holdout (train on non-crisis, test on `crisis` regime)

raw ECE 11.0 → recal 9.1 (modest). **Top bin: raw 0.927 → realized 0.670 → recal
0.843.** A recalibration learned without crisis data still predicts **0.843 when crisis
reality is 0.670** — a 17pp residual miss. The fix needs crisis data it won't have
in advance.

## The decisive diagnostic — crisis realized rate is not stable across crises

>0.90 bin, `hmm_regime == "crisis"`, realized `engine_exact` rate by year:

| year | n | realized |
|---|---|---|
| 2020 (crash) | 158 | **0.570** |
| 2021 (bull; false-alarm "crisis") | 15 | 0.933 |
| 2022 (bear) | 86 | 0.826 |
| 2025 | 19 | **0.368** |

**Cross-crisis spread: 56.5pp.** The HMM "crisis" label maps to realized outcomes
anywhere from 0.37 to 0.93 — because it fires on high *vol* regardless of *direction*
(a 2021 vol spike that resolved benignly looks like the 2020 crash to the label). A
recalibration map keyed on `(bin, regime)` therefore can't generalize: "crisis → 0.57"
learned from 2020 is badly wrong for 2021/2022/2025.

## Implications (for the re-baseline decision)

1. **Recalibration is insufficient.** It is a *post-hoc, backward-looking* correction;
   crises differ too much for a learned map to transfer. Shipping it would help the
   calm tape (cosmetic) and leave the crisis over-confidence (dangerous) largely intact.
2. **The structural route is the remaining candidate.** A *forward-looking* tail
   adjustment that reads current realized-tail conditions — POT-GPD wired into
   `prob_profit` (it exists in `engine/tail_risk.py`, unwired) — could in principle
   respond to a crisis as it unfolds rather than relying on a stale label. **This is
   untested**; it needs an engine prototype + its own generalization study before a
   re-baseline. Do not assume it works because recalibration didn't.
3. **The gate did its job.** Per the triage's sequencing, this verification was the
   precondition for the B re-baseline. Result: **the demonstrated fix does not clear
   it.** B is not ready; the re-baseline should wait for a structural prototype that
   passes this same leave-one-crisis-out test.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py   # writes realized rows
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i9_calibration_generalization.py
```
All numbers in `raw_output/i9_generalization_RAW.txt`.
