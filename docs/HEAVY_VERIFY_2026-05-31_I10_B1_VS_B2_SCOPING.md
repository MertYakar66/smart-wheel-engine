# Heavy-Verify Campaign 2026-05-31 — I10: B1-vs-B2 scoping (route the fix before building)

**Investigation:** before commissioning any prototype, route the crisis-over-confidence
fix between **B1** (structural probability fix, POT-GPD→`prob_profit`) and **B2**
(behavioral regime-transition gate, a downgrade-only reviewer). (1) Is the crisis
instability real or small-n? (2) Can any PIT signal *detect* the transition cleanly —
B2's prerequisite? (3) Does a leave-one-crisis-out gate net-cut the bleed?
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i10_b1_vs_b2_scoping.py`.
**Raw:** `raw_output/i10_b1_vs_b2_RAW.txt`. **Status:** observe-and-document; `engine/` not modified.

---

## VERDICT

> **Neither naive fix clears the bar. B1's lesson ("you can't calibrate an unstable
> quantity," I9) has a twin: "you can't gate what you can't detect" (B2, here).** The
> crisis instability is **real** (well-powered cells 0.57 vs 0.83 = 26pp, CIs
> non-overlapping; the 56pp headline was inflated by thin cells). But **no simple PIT
> signal achieves the 3-way separation B2 needs** — `rv_ratio` is *highest at the 2020
> recovery* (the wheel's best entry) and the 2022 bear-onset looks *calm* by drawdown,
> so a single-signal gate gates the wrong months. A crude LOCO gate helps in the sharp
> 2020 crash (the bleed dominates) but forgoes the recovery and over-fires the 2022
> bear (a net-positive year). **Do NOT commission either naive fix.** If a fix is
> pursued, it must be a *multi-feature onset detector* whose acceptance bar is the 3-way
> separation under leave-one-crisis-out — and the simple features tested fail it. The
> robust default in the meantime is a **risk-budget posture** (cap top-bin size in any
> elevated-vol regime), which reinforces the defensive-sleeve conclusion.**

Confidence: **high** on P1 (Wilson CIs) and P2 (the 3-way failure is clear on the
decisive, well-powered onset/recovery months); **directional** on P3 (the qualitative
conflation is robust; the $ magnitudes are tail-driven and not load-bearing).

---

## P1 — The instability is real, but ≥26pp, not 56pp (your verify-first check)

>0.90 bin, `hmm_regime=="crisis"`, realized rate by year with Wilson 95% CIs:

| year | n | realized | Wilson 95% | powered? |
|---|---|---|---|---|
| 2020 (crash) | 158 | 0.570 | [0.49, 0.64] | yes |
| 2022 (bear) | 86 | 0.826 | [0.73, 0.89] | yes |
| 2021 | 15 | 0.933 | [0.70, 0.99] | THIN |
| 2025 | 19 | 0.368 | [0.19, 0.59] | THIN |

The two **well-powered** crisis cells differ **26pp with non-overlapping CIs** → the
instability is genuine, not sampling noise. The full 63pp spread (incl. thin 2021/2025)
is inflated by small-n. **Honest statement: the crisis top-bin realized rate moves ≥26pp
crisis-to-crisis** — enough that no static or regime-keyed map can transfer (confirms I9
on solid cells), and the headline should not lean on n=15.

## P2 — No PIT signal achieves the 3-way separation (the B2 prerequisite fails)

Rank-correlation of each candidate detector with the per-date realized P&L of the
engine's positive-EV puts (a good detector → strong **negative** corr):

| signal | Spearman vs per-date realized P&L |
|---|---|
| rv_ratio (RV30/RV252, F4's signal) | −0.035 |
| trailing drawdown | −0.028 |
| rv_accel | −0.016 |
| rv30 level | −0.081 |

All ≈ 0. The 3-way table shows *why* — the signal must separate onset (gate) from benign
vol (don't) from **recovery** (don't — best entries), and it doesn't:

| date | tag | realized/contract | rv_ratio | drawdown | rv_accel |
|---|---|---|---|---|---|
| 2020-03-02 | **ONSET (crash)** | **−1,305** | 1.69 | −0.092 | 1.41 |
| 2020-04-01 | **RECOVERY** | **+302** | **2.69** | −0.307 | 2.21 |
| 2022-01-03 | **ONSET (bear)** | **−1,466** | 1.37 | **−0.002** | 0.19 |
| 2021-11-01 | benign vol | +5 | 0.90 | 0.000 | −0.02 |

- **`rv_ratio` is HIGHER at the recovery (2.69) than at the crash-onset (1.69)** — a gate
  on it would refuse the wheel's *best* entry (2020-04, +$302, 100% win).
- **The 2022 bear-onset looks calm by drawdown (−0.002)** and has low `rv_accel` (0.19) —
  a gate keyed on those would miss it entirely.

So the detection problem is as hard as I9's prediction problem. Bleed-vs-non-bleed mean
separations are weak (rv_ratio +0.17, drawdown +0.02).

## P3 — A crude LOCO gate conflates the regimes (directional)

Threshold `rv_ratio` on all-but-one-crisis, apply to the held-out crisis:
- **2020 crash:** the gate fires on 6/11 months — *including the 2020-04/05 recovery*.
  It nets positive only because the −$1,305 onset bleed dominates the book $ (tail-driven,
  not robust), but it **forgoes the recovery** it should keep.
- **2022 bear:** `rv_ratio` is elevated *all year*, so the gate fires on **12/12 months**
  — gating a year that was net-**positive** (+$5k) for the wheel, i.e. it **hurts**.

The gate can't tell acute onset from sustained-elevated-vol from recovery. (The dollar
figures are dominated by a few high-notional names and are not the takeaway; the
*conflation* is.)

## What this routes (the actual recommendation)

1. **Do NOT commission a single-signal transition gate** (naive B2) **or** the
   recalibration/probability fix (naive B1, already killed by I9). Both fail the gate.
2. **If a fix is pursued, it is a multi-feature *onset* detector** — something that
   captures *acceleration into a drawdown* and separates it from *recovery* (drawdown
   deep but vol falling) and *sustained bear* (vol elevated but not accelerating). Its
   acceptance bar is the **3-way separation under leave-one-crisis-out** in P2. The
   simple features here fail it; a real detector must beat them on that test before it
   earns an engine prototype. This is now a *research* task, not a wiring task.
3. **The robust default that needs no detector:** a **risk-budget posture** — cap
   top-bin (`prob_profit > 0.90`) position size in any elevated-vol regime, accepting
   you forgo some recovery trades for robustness. This is a sizing/reviewer rule, not a
   precision gate, and it is §2-clean (downgrade-only). It operationalizes the
   defensive-sleeve conclusion: *if you can neither predict nor cleanly detect the
   crisis, don't bet big at the top of the confidence range when vol is up.*

> **Strategic confirmation:** the engine structurally cannot be trusted on
> high-confidence trades at crisis onset — and I10 shows you cannot cheaply detect your
> way around it either. The honest posture is to *size that exposure down*, not to
> believe a corrected number or a single-signal gate. This is a better defensive sleeve,
> not a precision instrument.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i1_calibration.py   # writes realized rows
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i10_b1_vs_b2_scoping.py
```
All numbers in `raw_output/i10_b1_vs_b2_RAW.txt`.
