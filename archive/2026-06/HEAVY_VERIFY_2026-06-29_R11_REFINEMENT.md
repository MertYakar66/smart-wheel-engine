# Heavy-verify 2026-06-29 — R11 refinement study (#442 → persistence-N)

**Verdict: keep R11 exactly as it is. Neither candidate refinement clears the bar.**
A static entry-VIX band carve-out (the naïve reading of #442) is wrong on four
independent grounds, and the persistence-N condition from the
`r11-onset-aware-trigger` research card is **refuted** — every whole-book
difference is within noise (paired daily-return |t| ≤ 0.5), and a strictly
*consecutive* N-day condition *erodes* R11's only real protection (the 2022
bear) because that regime's elevation was choppy, not sustained.

**No decision-trio change is proposed.** This is a measurement-only study;
`engine/candidate_dossier.py` R11 is untouched. Closes the R11-refinement
thread of #442 and the `r11-onset-aware-trigger` research card.

---

## 0. Context

- **R11 (live, D23):** downgrades a *proceed* top-bin pick (`prob_profit > 0.90`)
  to *review* when `vix_level > 25.0`. Downgrade-only; §2-safe; never rescues a
  negative-EV trade. Its threshold was CV-tuned in I11 (leave-one-crisis-out;
  θ≥27.5 fails the 2022 fold → 25 is the robust-not-optimal floor).
- **#442 (Mac/W3):** top-bin over-confidence is **calm/elevated-ENTRY ~−16pp**,
  but **crisis-entry (VIX≥30) is well-calibrated (+0.8pp)** — routed to this
  (Windows) lane as input for a possible band-aware R11. (Note: the "R11b"
  label in #442/#436 refers to a *proposed* skew-edge gate, #437, which was
  A/B-tested net-costly OOS and **closed/rejected** — R11b does not exist in the
  engine; only R11 does.)
- **D23 post-ship validation (2026-06-01):** a full $1M/100t dollar A/B found
  R11's whole-book NAV/Sharpe impact **statistically indistinguishable from
  zero** (|t| = 0.62/0.24, NAV flips sign by window), and flagged the natural
  refinement — a **persistence/onset-aware trigger** — as a research card.

This study answers both open questions: (1) does #442 motivate a static band
carve-out? and (2) does the persistence-N trigger beat the level trigger?

---

## Part 1 — #442 band-split: no static carve-out

A static "spare VIX≥30" carve-out fails on four counts, each grounded in
existing evidence:

1. **R11's firing zone is net-costly, not mis-targeted.** W6 (#448) measured the
   VIX>25 top-bin at **−$158/contract** (n=704, monthly grid) — R11 sizes down a
   *losing* population.
2. **The bands R11 *misses* are net-positive.** W6: VIX 15-25 top-bin **+$125**,
   calm ≤15 **+$266** — over-confident but profitable. Gating them is the exact
   #437 mistake (rejected, −3.43% OOS in the 2020 crash).
3. **The "≥30 well-calibrated" basis was a single date.** W3's 35-date grid had
   exactly one date ≥30 (2020-06-15) → cluster-n=1.
4. **Win-rate calibration ≠ tail/net-dollar safety** (R11's concern, via `cvar_5`).

A dense weekly **VIX≥25 re-run** (`r11_band_split.py`, n=15,738 over 68 dates —
28 of them ≥30, vs W3's one) gives the multi-date ≥30 net-dollar W3 could not:

| VIX-at-entry band | top-bin net $/contract (full friction) | n / dates | 95% CI |
|---|---:|---|---|
| highelev 25-30 (grind) | **−$27.82** | 1488 / 38 | [−393.5, +188.3] |
| crisis ≥30 (spike) | **+$148.86** | 1838 / 27 | [−135.1, +321.4] |
| current R11 (>25) | +$69.82 | 3326 / 65 | [−146.1, +229.5] |

The sign split (≥30 net-positive, 25-30 net-negative) is the **spike-vs-grind**
distinction — but **both CIs straddle zero** and the ≥30 positive is ~half
2020-COVID-V-recovery (one regime). It does **not** support a static rule; it
points to the *persistence* hypothesis, which Part 2 tests directly.

---

## Part 2 — persistence-N study (the research card)

### Method
`r11_persist_driver.py` extends the shipped `r11_dollar_impact` harness with
three persistence arms, all sharing one daily rank (only the open-step gate
differs). Gate replicated from the engine's own `R11_*` constants (cannot
drift). Arms:

- **suppressed** — pre-R11 open policy (EV>0, top-N, BP-gated, `max_new/day`).
- **active** — live R11: block top-bin when `vix > 25`.
- **persist{5,10,20}** — active AND `vix > 25` has held **N consecutive trading
  days** (a pure PIT look-back; §2-clean, downgrade-only).

Windows: **W3 2020-2024**, **W4 2021-2025**; $1M, UNIVERSE_100, full friction,
dte 35 / delta 0.25 / ≤3 opens-per-day. Blocked sets are forward-replayed to
expiry (the averted-vs-forgone counterfactual). §2 scan: zero non-positive-EV
opens in any arm.

### Whole-book results

| Arm | W3 ΔNAV (t vs supp) | W3 Sharpe | W4 ΔNAV (t vs supp) | W4 Sharpe |
|---|---:|---:|---:|---:|
| **active** | **+$44,300** (+0.34) | 0.517 | **+$62,773** (+0.47) | 0.673 |
| persist5 | −$11,765 (−0.03) | 0.442 | +$17,976 (+0.17) | 0.587 |
| persist10 | −$13,731 (−0.33) | 0.457 | **+$70,788** (+0.49) | 0.676 |
| persist20 | +$45 (−0.06) | 0.469 | +$1,783 (+0.11) | 0.626 |

- **Everything is within noise.** All paired daily-return |t| ≤ 0.5 (vs ~2 for
  significance). persist10's eye-catching W4 result (+$70.8k, nominally beating
  active) is **t = +0.14 vs active** — pure noise — and it is *negative* in W3
  (sign-flip = not robust). No arm is statistically distinguishable from active
  or from suppressed. (Confirms + extends D23: R11 and every persistence variant
  are net-neutral-to-the-book.)

### Why consecutive-N fails — the mechanism

R11's only real value is the **2022 bear** (active averts the CSP-leg loss);
its only real cost is the **2020 V-recovery** (forgone gain). The card's
hypothesis was that persistence keeps the former and drops the latter. The
blocked-set counterfactual shows it does the **opposite**:

(2022 averted loss = W3 figures; W4 is within ~$2k of these. 2020 forgone gain
is W3-only — W4 excludes the 2020 crash.)

| Arm | 2022 averted loss (want to KEEP) | % of active | 2020 forgone gain (want to CUT) |
|---|---:|---:|---:|
| active | −$268,104 | 100% | +$47,731 |
| persist5 | −$167,680 | **63%** | +$48,453 (not cut) |
| persist10 | −$174,115 | **65%** | +$35,101 (cut 26%) |
| persist20 | −$1,837 | **0.7%** | +$13,863 (cut 71%) |

No N retains ≥80% of the 2022 protection (the card's success criterion); only
persist20 cuts the 2020 forgone gain, and it does so by barely firing at all
(blocks 118 vs active's 429) — killing the 2022 protection.

The root cause is in the VIX-streak coverage:

| Regime | days VIX>25 | run≥5 | run≥10 | run≥20 |
|---|---:|---:|---:|---:|
| 2020 crash (Feb-Jun) | 91 | 83 | 73 | **55 (61%)** |
| 2022 bear (full year) | 135 | 100 | 73 | **33 (24%)** |

**2020's elevation was sustained** (even run≥20 covers 61% of its elevated days)
while **2022's was choppy** (run≥20 covers only 24% — VIX repeatedly dipped under
25). So "VIX>25 for N consecutive days" fires *more* in the regime you want to
skip (2020) and *less* in the one you want to keep (2022) — backwards from the
card's premise that 2020 was the brief spike and 2022 the sustained grind.

---

## Reconciliation with D23 (data drift)

On *current* data R11 active is whole-book **positive in both windows**
(+$44.3k / +$62.8k) vs D23's 2026-06-01 net-neutral/sign-flipping (−$37.6k /
+$21.7k). The harness is faithful (same logic, same gate, same universe/params);
the difference is **data drift** — the committed OHLCV SHA changed
(`c3d5443158b12ec5` → `7a3e77a4fab07f85`) across the refreshes since 2026-06-01.
Both are within noise, so the *conclusion* (net-neutral-to-the-book) is
unchanged; R11 simply looks no-worse, possibly marginally-better, on current
data.

---

## Honesty caveats

- All whole-book Δ are **within noise** (|t| ≤ 0.5); none of the point estimates
  (active's included) should be read as a real edge.
- **W3 and W4 overlap ~80%** (shared 2021-2024) — "positive in both" is not two
  independent confirmations; both lean on the same 2022 bear + calm years.
- Per-regime blocked-set counts are small (2022 bear n≈70-114) and
  regime-concentrated; treat the averted/forgone figures as directional.
- The one **untested** variant is a **non-consecutive** condition ("VIX>25 on ≥N
  of the last M days"), which would target choppy-2022 better. But since the
  whole book is noise-dominated (the wheel's recovery leg dilutes the real
  per-contract 2022 protection to immateriality — D23's structural reframe,
  reconfirmed), it is unlikely to move the whole-book needle. Filed as the
  residual open item; not pursued.

---

## §2 statement

Measurement-only. Every candidate routes through `EVEngine.evaluate` via
`rank_candidates_by_ev`; the replicated R11/persistence gates only ever *remove*
an open (downgrade-only), never rescue a non-tradeable one. No `engine/` file is
modified. R11 ships unchanged.

## Reproduce

```
py -3.12 docs/verification_artifacts/r11_refinement_2026-06-29/r11_persist_driver.py \
    --start 2020-01-02 --end 2024-12-31 --out-dir <tmp>/w3
py -3.12 .../r11_persist_driver.py --start 2021-01-02 --end 2025-12-31 --out-dir <tmp>/w4
py -3.12 .../r11_persist_driver.py --analyze --out-dir <tmp>/w3   # by-regime
```

Artifacts: `docs/verification_artifacts/r11_refinement_2026-06-29/`
(`w3_summary.json`, `w4_summary.json`, `w{3,4}_daily_nav.csv`, `band_split_442.json`,
`r11_persist_driver.py`, `r11_band_split.py`).
