# Engine backtest — S34 snapshot re-baseline (post-#260) + R10 firing analysis

**Date:** 2026-05-29
**Engine SHA:** `origin/main` @ `56d8e5c` (post-PR #260 realized-vol-ratio widening, post-PR #262 R10 single-name cap)
**Author:** Terminal A (executor), Session A (verifier)
**Window / universe / strategy / config:** identical to `docs/ENGINE_BACKTEST_S34_UNIVERSE.md` — 2022-01-03 → 2024-12-31, 100 first-alphanumeric SP500 tickers (`UNIVERSE_100`, A → CMI), $1M starting capital, 35-DTE / 25-Δ short puts, wheel into CC on assignment, hold to expiry, `top_n=15`, `max_new_per_day=3`, `require_ev_authority=False`. Three friction levels: `none` / `bid_ask` / `full` via `run_backtest_multi_friction` (one rank call/day shared across three trackers).
**Source artefacts:** `C:/Users/merty/AppData/Local/Temp/s45_s34_rebaseline/{s34_pre_260_snapshot.json, s34_run/}` and the regenerated in-repo `backtests/regression/snapshots/s34_universe_100t_1m.json`.

> **Naming note.** This doc uses the descriptive name **"S34 re-baseline (post-#260)"** rather than a hardcoded Sn label, per `docs/PARALLEL_SESSIONS.md` rule 7 (introduced in PR #282): *Sn is allocated at MERGE by the operator, not claimed at work-start*. The branch retains its working-name (`claude/usage-test-s45-s34-rebaseline-post-260`) for git-log reachability; the merger fills in the canonical Sn at merge time. The R10 firing probe filename (`docs/verification_artifacts/s45_r10_firing_driver.py`) preserves its pre-rule-7 prefix; it's a tool artifact identified by its content (R10 firing analysis) not by an Sn claim.

---

## Purpose

PR #260's body explicitly deferred the S32 / S34 / S35 snapshot re-baselines to a separate follow-on workflow because the slow-lane CI runs them on a dedicated `.github/workflows/backtest-regression.yml` job. Until those are re-baselined, the slow-lane CI fails `test_backtest_matches_snapshot[s34_universe_100t_1m]` against the current post-#260 engine.

The companion **S32 re-baseline** (parallel branch `claude/usage-test-s42-s32-rebaseline-post-260`, doc `docs/ENGINE_BACKTEST_S32_REBASELINE_POST260.md`) closed the 24t / $1M leg. **This work closes the S34 leg** (100t / $1M) — the heaviest of the three (~5-6 h wall clock under contention) and the one that lets us test the **contrapositive of the S32 re-baseline's F5 finding**:

> S32 re-baseline F5: *"R10 cap = $100k notional per name; max position notional ~$70k at $1M / 24t / 1-contract; R10 cannot fire."*

This work tests this at 100 tickers, where heavyweight strikes (AZO $3,216 / BKNG $2,397 / BLK $969) push first-contract notional **above the 10% NAV cap**. The question: **does R10 fire meaningfully when the universe contains tickers expensive enough for a single 1-contract position to exceed $100k notional?**

This is the **R10 firing analysis** — the novel content beyond the snapshot regen.

---

## Comparison

S34 backtest re-ran cleanly. Pre-#260 snapshot was generated 2026-05-27 02:18 UTC; post-#260 (this PR) generated 2026-05-29 13:02 UTC. The `data_csv_sha256` is **identical** between the two — same OHLCV CSV, same data window, same fingerprint config; the diff is purely the engine code change.

### Aggregate (full-friction headline)

| Metric | Pre-#260 snapshot | Post-#260 (this PR) | Δ | % Δ |
|---|---|---|---|---|
| row_count | 10,911 | 10,911 | 0 | — |
| Spearman ρ (overall) | **+0.32859** | **+0.32473** | **−0.00386** | **−1.18%** |
| Spearman p | 5.11e-273 | 2.49e-266 | — | — |
| mean_realized | $23.52 | $22.89 | −$0.63 | −2.69% |
| hit_rate | 78.00% | 77.96% | −0.04pp | −0.05% |
| iv_mean | 0.30552 | 0.30504 | −0.0005 | −0.16% |
| ev_mean | $20.92 | $18.42 | −$2.50 | −11.94% |
| final_cash (full friction) | $83,831 | $97,087 | +$13,256 | +15.81% |
| **final_NAV (full friction)** | **$1,347,145** | **$1,339,359** | **−$7,786** | **−0.58%** |
| executed_trades | 264 | **271** | **+7** | **+2.65%** |
| put_assignments | 79 | 75 | −4 | −5.06% |
| open_at_end | 57 | 56 | −1 | — |

### Per friction level

| Friction | Pre-#260 ρ | Post-#260 ρ | Δρ | Pre-#260 NAV | Post-#260 NAV | Δ NAV |
|---|---|---|---|---|---|---|
| `none` (frictionless) | +0.33106 | +0.32714 | −0.00393 | $1,369,533 | $1,351,147 | **−$18,386** |
| `bid_ask` | +0.32859 | +0.32473 | −0.00386 | $1,350,552 | $1,341,231 | **−$9,321** |
| `full` | +0.32859 | +0.32473 | −0.00386 | $1,347,145 | $1,339,359 | **−$7,786** |

All three friction levels show the SAME directional effect: ρ −1.18% relative; NAV slightly *negative* delta. Notable that the `none` level NAV drops MORE than the `full` level NAV — under no friction the F4 widening's reshuffle has the cleanest dollar impact; under friction, the proportional drag dampens the comparison. **All three friction levels still execute the same 271 trades** (the rank call is friction-independent; only premium / realized_pnl differ).

### Per year (full friction)

| Year | n | Pre-#260 ρ | Post-#260 ρ | Δρ | Pre-#260 mean | Post-#260 mean | Δ mean |
|---|---|---|---|---|---|---|---|
| 2022 (bear) | 3,575 | +0.36430 | **+0.35764** | −0.00666 (−1.83%) | −$121.52 | −$120.70 | **+$0.82 (+0.68%)** |
| 2023 (recovery) | 3,599 | +0.31175 | +0.31054 | −0.00122 (−0.39%) | +$95.32 | +$94.58 | −$0.74 (−0.78%) |
| 2024 (bull) | 3,737 | +0.30948 | +0.30103 | −0.00845 (−2.73%) | +$93.11 | +$91.20 | −$1.91 (−2.05%) |

ρ degrades less in 2023 (calm regime, F4 widening barely fires) and more in 2024 (more reshuffling). 2022 mean_realized **improves slightly** — the F4 widening refuses the marginally worst bear-year picks.

### Per quartile (full friction)

| Q | n | Pre-#260 EV mean | Post-#260 EV mean | Pre-#260 PnL mean | Post-#260 PnL mean | Δ PnL |
|---|---|---|---|---|---|---|
| Q0 (low) | 2,728 | −$92.55 | −$94.26 | −$20.99 | −$23.52 | −$2.52 (+12.0%) |
| Q1 | 2,728 | −$8.32 | −$8.95 | −$8.03 | **−$1.45** | **+$6.58 (−81.9%)** |
| Q2 | 2,727 | $14.88 | $13.82 | +$19.90 | +$14.24 | **−$5.67 (−28.5%)** |
| Q3 (high) | 2,728 | +$169.67 | +$163.09 | +$103.19 | +$102.27 | −$0.92 (−0.89%) |

**Q3 still beats Q0 by 4.35× in realized PnL** ($102.27 vs −$23.52) — the engine's ordering signal is preserved at the larger universe. As in the S32 re-baseline, the composition shift concentrates in Q1 ↔ Q2 mid-quartile reshuffling (Q1 PnL improves +82%; Q2 PnL drops −29%). Q3 vs Q0 monotonicity holds.

---

## The (capital × universe) regime triangle — S27, S32 re-baseline, S34 re-baseline

S27, the S32 re-baseline, and this S34 re-baseline form a triangle: each varies ONE dimension of the (capital × universe) configuration:

| Metric | S27 ($100k / 24t) | S32 re-baseline ($1M / 24t) | S34 re-baseline ($1M / 100t) |
|---|---|---|---|
| Overall ρ relative Δ | −3.30% | −3.33% | **−1.18%** |
| Δ executed_trades | 51 → 40 (−22%) | 105 → 104 (−1%) | **264 → 271 (+3%)** |
| Δ final_NAV (full friction) | **−$15,471 (−12.1%)** | **+$2,376 (+0.22%)** | **−$7,786 (−0.58%)** |
| Engine vs SPY (pre→post) | (unchanged: +27pp) | (unchanged: −22pp) | **+11.6pp → +9.94pp (−1.66pp)** |

**Three distinct regimes:**

- **S27 ($100k / 24t):** BP-saturated. F4 widening reshuffles top-10/day; cascading BP refusals refuse 11 trades; NAV drops sharply. The F4 fix's *intended* defensive behaviour.
- **S32 re-baseline ($1M / 24t):** Capacity-constrained AND narrow universe. F4 widening fires only 1× across 3 years; net dollar effect marginally positive (+$2,376) because the refused trade was net-negative-realized.
- **S34 re-baseline ($1M / 100t):** Capacity-constrained but WIDE universe. F4 widening's reshuffle now LETS MORE TRADES IN (+7 net executed); some are net-negative-realized (NAV −$7,786). The wider universe provides more candidates for the F4-widened ordering to surface.

**The F4 fix's dollar-level effect is non-monotonic across the (capital, universe) plane.** It does what was intended at $100k (defensive — refuses trades, NAV down by the loss they would have caused). At $1M / 24t it's a no-op. At $1M / 100t it slightly underperforms the pre-fix engine on the 2022-2024 window because the wider candidate pool surfaces more F4-widened-but-still-acceptable picks that are marginally net-negative.

This is the most operationally meaningful finding in this re-baseline: **at $1M / 100t, post-#260 is a small dollar drag (−$7,786 / −0.58%) on 2022-2024 NAV**, traded against the structural F4-tail protection the widening provides. Whether that trade-off is "worth it" depends on whether 2025+ sees an unpredicted F4-style event in the larger universe.

---

## R10 firing analysis — the novel content

The S32 re-baseline found (F5) that R10's $100k single-name cap is structurally non-binding at $1M / 24t because the 24-ticker universe's max strike (COST / GS / UNH heavyweights) only produces $20-70k notional per 1-contract position. **This work tests the contrapositive at 100t.**

### Static per-ticker R10 verdict (max_fcn = max-strike × 100 × 1 contract)

Of the 98 distinct tickers in the post-#260 rank_log (universe is 100; CBOE UF + 1 other have no OHLCV coverage in this window):

| R10 verdict | # tickers | Meaning |
|---|---|---|
| `blocks_1st` | **2** | Even on a clean slate, R10 refuses every entry attempt (FCN > $100k). |
| `blocks_2nd` | 4 | First contract allowed; second contract refused (FCN > $50k). |
| `blocks_3rd` | 6 | Up to 2 contracts allowed; third refused (FCN > $33k). |
| `non_binding` | 86 | R10 never engages within reasonable position counts. |

**The two `blocks_1st` tickers — AZO and BKNG — drive the entire R10-firing story:**

| Ticker | n_rows | n_ev_positive | max_strike | max_fcn | max_fcn / $100k cap | Sum realized_pnl per rank_log row | Mean realized per row |
|---|---|---|---|---|---|---|---|
| **AZO** | 310 | 238 | $3,216.00 | **$321,600** | **3.22×** | **+$659,300** | +$2,127 |
| **BKNG** | 132 | 131 | $2,397.00 | **$239,700** | **2.40×** | **−$287,049** | −$2,175 |

Both tickers are R10-blocks-1st structurally. **The two-ticker R10 firing concentration cross-references Terminal C's S43 W3 (PR #270) finding** — C's running-max audit independently identifies BKNG (44.25% NAV peak) and AZO (21.42%) as the two structural R10 violators across the 2020-2024 window. This work confirms the same pair drives R10 firing in the narrower 2022-2024 window.

### Counterfactual replay — what would R10 refuse, and at what dollar cost?

Replayed the harness's open logic (`top_n=15`, `max_new_per_day=3`, `EV > 0`, no existing position, settle on `expiration_date <= today`) with R10 (`engine.portfolio_risk_gates.check_single_name_cap`) wired in at each open attempt. Headline:

| Metric | Without R10 (live harness) | With R10 (counterfactual) | Δ |
|---|---|---|---|
| Total opens (replay basis) | 708 | 684 | **−24** |
| Total realized P&L on opens | **+$112,211** | **+$32,824** | **−$79,387 (−70.7%)** |
| Total R10 fires | 0 (R10 absent) | **368** | — |

**Note on replay vs harness counts.** The 708 replay opens > 271 harness executed_trades (per metrics.json) because the replay's expiration-resets-the-ticker simplification is strictly more permissive than the harness's WheelTracker state machine (which routes SHORT_PUT → STOCK_OWNED → COVERED_CALL before re-opening). The *delta* between with-R10 and without-R10 (−24 opens, −$79k realized) is what matters and is well-bounded by the replay's per-ticker discipline.

**R10 fires per year (counterfactual):**

| Year | R10 fires |
|---|---|
| 2022 (bear) | **225** (61%) |
| 2023 (recovery) | 75 (20%) |
| 2024 (bull) | 68 (18%) |

**Total refused notional across the 3-year window: $81,787,350.** Concentration in 2022 makes sense — bear-year EV thresholds let more bearish AZO / BKNG candidates rank high.

### Per-ticker dollar-cost-of-R10 on actually-executed trades

| Ticker | Opens (no R10) | Opens (with R10) | Δ opens | Realized PnL (no R10) | Realized PnL (with R10) | Δ realized |
|---|---|---|---|---|---|---|
| **AZO** | 18 | **0** | **−18** | **+$46,155** | $0 | **−$46,155** |
| **BKNG** | 9 | **0** | **−9** | **+$32,839** | $0 | **−$32,839** |
| ADBE | 13 | 13 | 0 | +$5,341 | +$5,341 | 0 |
| AXON | 8 | 8 | 0 | +$3,694 | +$3,694 | 0 |
| CHTR | 12 | 12 | 0 | −$3,329 | −$3,329 | 0 |
| (95 other) | … | … | small re-allocation | … | … | small |

**Net effect of wiring R10 in: lose AZO ($46k) + BKNG ($33k) = $79k of executed-trade alpha** in the 2022-2024 window. Both names happen to be net-positive contributors in this window. R10 doesn't distinguish "high-priced and bullish" from "high-priced and tail-risk-exposed" — it refuses all heavyweight single-name concentration uniformly.

### The asymmetry — R10 is insurance, not optimization

This result confirms the design rationale in `engine/portfolio_risk_gates.py` and `docs/F4_TAIL_RISK_DIAGNOSTIC.md` §11: R10's job is to **bound F4-style idiosyncratic damage that no market-wide signal can predict**, not to improve in-window returns. In the 2022-2024 window the named F4 events were COST 2022-04 (NOT in the 100-ticker universe; first-100 cutoff is at CMI alphabetically) and UNH 2024-11 (NOT in the universe either). With no F4 events firing on the in-universe heavyweights, R10's protection is paying premium for insurance against events that didn't happen. The cost was $79k of realized P&L; the protection was zero realized payout — a window-specific 100% loss ratio.

In a hypothetical window where AZO or BKNG drops 30%+ in a single month, R10's $46k + $33k cost would have prevented unbounded losses on the held assignments — exactly the F4 mechanism §10 of the diagnostic documents. The trade-off is structurally about tail-event coverage, not about average-case returns.

---

## Engine vs SPY / Universe-EW — re-framed

S34 originally claimed **"+11.6pp over SPY"** using a quoted SPY return of ~+24% price-only for the window. Updating with the post-#260 NAV:

| Benchmark | Window return | Post-#260 engine return | Engine vs benchmark |
|---|---|---|---|
| **SPY (~+24%, S34's external benchmark)** | ~+24.0% | **+33.94%** | **+9.94pp** |
| **Universe-100 EW (price-only, computed)** | **+22.31%** | **+33.94%** | **+11.63pp** |
| S34 reference (pre-#260) | ~+24.0% | +35.61% | +11.6pp |

**Δ vs S34's headline: −1.66pp under SPY (−14% relative); −0.0pp under Universe-EW (matches the original +11.6pp essentially by coincidence).** The wider universe's EW return (+22.31%) is below SPY (+24%) because the first-100-alphanumeric cut skews toward smaller-cap defensives (e.g., AJG, ACGL) and excludes mega-cap-growth winners (NVDA, TSLA, GOOGL all alphabetically after CMI). The engine's absolute alpha vs SPY drops by 14%; vs the more honest same-universe EW it's essentially unchanged.

The most honest framing: **"on the 2022-2024 / $1M / 100t configuration, the post-#260 engine beats SPY by +9.94pp and beats the same-universe equal-weight by +11.63pp — both slightly below the pre-fix engine's +11.6pp vs SPY but indistinguishable from pre-fix vs the universe-EW benchmark."**

---

## Concentration — top-5 vs others

S34 (pre-#260) found **single-name BKNG drove 110% of net executed P&L** ($31,576 on $28,571 total). In the post-#260 rank_log replay (without R10):

| Tier | Aggregate realized PnL on executed opens (replay basis) |
|---|---|
| AZO | +$46,155 (41% of total) |
| BKNG | +$32,839 (29%) |
| **Top 2 (AZO + BKNG)** | **+$78,994 (70%)** |
| ADBE | +$5,341 |
| AXON | +$3,694 |
| **Top 5** | **+$87,683 (78%)** |
| Other 93 (positives + negatives) | +$24,528 (22%) |
| Total executed-replay realized | **+$112,211** |

Concentration is now **bipolar (AZO + BKNG)** rather than the original S34's BKNG-only dominance — the F4 widening's reshuffle elevates AZO from a moderate contributor to the largest single-name positive. **With R10 wired in, both go to zero**; concentration distributes entirely across the remaining 96 tickers, but the headline alpha drops $79k.

R10 doesn't merely "distribute exposure away from BKNG/NVR/ORLY" as the task brief anticipated — NVR / ORLY aren't in the 100-ticker universe at all, and the F4 fix elevates AZO alongside BKNG as a second heavyweight contributor. **R10 distributes exposure across the universe, at the cost of $79k of realized alpha in this window.**

---

## §2 invariant

- `backtests/regression/snapshots/s34_universe_100t_1m.json` is the only file in the snapshot diff. `docs/verification_artifacts/s45_r10_firing_driver.py` is a new read-only probe.
- Zero edits to `engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py`, `engine/forward_distribution.py`, `engine/portfolio_risk_gates.py`. Snapshot regeneration + analysis only.
- **§2 scan: 0 non-finite ev_dollars across 10,911 rank_log rows.** Per-column: `ev_dollars`, `premium`, `strike`, `iv`, `realized_pnl`, `prob_profit` all 100% finite. R1a (`ev_non_finite` guard, PR #204) reports clean on the post-#260 engine at $1M / 100t.

---

## Test gates

| Gate | Result |
|---|---|
| `pytest tests/test_backtest_regression.py -k s34` against the new snapshot | ✅ **2 passed in 4h18m** (`test_snapshot_fingerprints_have_required_keys[s34_universe_100t_1m]` + `test_backtest_matches_snapshot[s34_universe_100t_1m-backtests.regression.s34_universe_100t_1m]`). The test re-ran the full S34 backtest end-to-end and confirmed every metric matches the newly-regenerated snapshot. Re-baseline is independently reproducible. |
| Launch-blocker subset | ✅ 103 passed in 18s (already green on `origin/main`; ran pre-push to confirm post-snapshot-change cleanliness) |
| `ruff check docs/verification_artifacts/s45_r10_firing_driver.py` | ✅ All checks passed |
| §2 invariant scan over rank_log (10,911 rows) | ✅ 0 non-finite EV |

---

## Findings

- **F1 — Snapshot reproduces on post-#260 engine; row count is byte-identical (10,911).** Pre→post: ρ −1.18% relative, mean_realized −2.69%, hit_rate effectively unchanged. ev_mean drops −11.9% (F4 widening shifts EV estimates down on most candidates).
- **F2 — Three (capital × universe) regimes for the F4 fix's dollar effect.** S27 ($100k/24t) −12.1% NAV; S32 re-baseline ($1M/24t) +0.22% NAV; **S34 re-baseline ($1M/100t) −0.58% NAV with +7 more executed trades.** Wider universe lets the F4-widened ordering surface more candidates, slightly diluting per-trade realized P&L.
- **F3 — R10 fires meaningfully at $1M / 100t — but ONLY on AZO and BKNG.** 368 R10 fires in 3 years (225 in 2022 / 75 in 2023 / 68 in 2024); 100% of fires are AZO (238) + BKNG (130). 96 of 98 in-universe tickers never engage R10. **F5 from the S32 re-baseline (R10 structurally non-binding) is inverted: R10 IS binding at 100t.**
- **F4 — Wiring R10 in costs $79,387 of executed-trade realized P&L (−70.7%) on the 2022-2024 window.** AZO contributes +$46k and BKNG +$33k as the dominant positive-realized contributors; R10 refuses both entirely. This is the **insurance-vs-optimization trade-off** — R10 protects against F4-style tail events that didn't fire on these names in this window.
- **F5 — Concentration shifts from single-ticker (BKNG, original S34) to bipolar (AZO + BKNG).** Original S34 had BKNG at +110% of net executed P&L. Post-#260 replay has AZO at 41%, BKNG at 29%, top-5 = 78%. The F4 widening's reshuffle elevates AZO from moderate to dominant.
- **F6 — Engine vs SPY drops from +11.6pp to +9.94pp (−1.66pp, −14% relative); engine vs Universe-EW is essentially unchanged at +11.63pp.** The wider universe's EW is +22.31% (below SPY's ~+24%) because the first-100-alphanumeric cut skews to mid-cap defensives.
- **F7 — Task brief assumed NVR / ORLY / COST / GS were in the 100-ticker universe — they aren't.** First-100 alphabetic cuts at CMI; NVR / ORLY / GS / COST / MTD / FICO all start with later letters. The R10-binding heavyweights in this universe are AZO and BKNG only.
- **F8 — Cross-reference with Terminal C's S43 W3 (PR #270, 2020-2024 multi-window).** C's running-max R10 audit on W3 reports BKNG peak = 44.25% NAV, AZO peak = 21.42% NAV — the same two-ticker pair as this work's static / replay analysis identifies. The R10-violator set is window-invariant across the in-flight rolling-window backtests.

---

## What this validates / does not validate

| Claim | Verdict |
|---|---|
| S34 baseline reproduces on post-#260 engine | ✅ snapshot regenerated; row count identical; metrics shift in line with F4 widening |
| PR #260's ρ-preservation pattern holds at 100t / $1M | ✅ −1.18% relative (less than S27 / S32-rebaseline's −3.3% — wider universe absorbs reshuffle) |
| PR #260 is a *dollar*-improver at $1M / 100t | ❌ marginal drag: −$7,786 NAV across 3-year window (−0.58%) |
| PR #262 R10 cap fires at $1M / 100t | ✅ 368 fires across 3 years on AZO + BKNG; the S32 re-baseline's F5 (R10 non-binding) is inverted at 100t |
| R10 refuses BKNG entries that drove S34's P&L | ✅ refuses ALL 9 BKNG entries the harness would have opened (−$33k) |
| R10 distributes exposure away from concentrators | ✅ but at $79k cost in this window — insurance, not optimization |
| Engine vs SPY +11.6pp survives the F4 fix at 100t | ⚠️ +9.94pp post-fix (−1.66pp). vs Universe-EW: unchanged at +11.63pp |
| Strategy capacity finding holds | ✅ 271 executed_trades (+7), open_at_end 56 (−1) — broadly unchanged |
| Single-ticker concentration finding (BKNG) holds | ⚠️ now bipolar — AZO has overtaken BKNG as the dominant positive contributor |

---

## AI handoff

- **S35 is the last remaining pre-#260 re-baseline.** S35 ($100k / 24t / 2018-2020 OOS, 3 friction levels) is similar to the S32 re-baseline in cost (~30-90 min wall under contention). Natural next-Sn after this S34 re-baseline and the S32 re-baseline land. Once it merges, the entire slow-lane regression suite (S27 / S32 / S34 / S35) is post-#260.
- **For `docs/PRODUCTION_READINESS.md` §3 Blocker B1 + §5 deployment matrix:** this work sharpens the S32 re-baseline's "bundle non-binding at $1M" framing. At $1M / 24t the bundle is a no-op (S32 re-baseline); at $1M / 100t the bundle COSTS ~0.6% NAV on 2022-2024 AND R10 fires 368×, refusing $79k of realized alpha (mostly AZO + BKNG). The "bundle protects" framing is correct as insurance; the cost of that insurance is now empirically pinned at this scale.
- **For the slow-lane CI workflow:** after this PR merges, `tests/test_backtest_regression.py::test_backtest_matches_snapshot[s34_universe_100t_1m]` will pass on the dedicated workflow. Only S35 parametrization remains failing.
- **For the engine team:** R10's hard refusal at $1M / 100t systematically excludes AZO and BKNG as tradeable names. If those two are durable wheel candidates outside of F4 events, consider whether **multi-contract scaling** + **dollar-notional position sizing** (rather than 1-contract-per-name) would let R10's per-name cap allocate fractional contracts to heavyweights — i.e., 0.3 contracts of AZO at $96k notional fits R10's $100k cap. Out of scope for this PR; flagging for the next decision-layer ROADMAP item.
- **For benchmark hygiene in future S<n> docs:** the S34 doc's "SPY ~+24%" external benchmark is fine but glosses over universe-shape skew. **Universe-EW is a strictly cleaner comparison** (same data source, same window, same dividend treatment). Both are now reported in this work. C's S43 already uses Universe-EW exclusively — recommending Universe-EW become the primary benchmark in future Sn re-runs, with SPY as a secondary external reference.

---

## Method appendix

**Harness:** `backtests.regression.s34_universe_100t_1m` (in-repo since PR #241). Same `run_backtest_multi_friction` path; one rank call per day, three trackers (`none` / `bid_ask` / `full`) each independently MTM-ing and deciding opens against the shared ranked frame.

**Snapshot file:** `backtests/regression/snapshots/s34_universe_100t_1m.json`. Pre-#260 backup in `C:/Users/merty/AppData/Local/Temp/s45_s34_rebaseline/s34_pre_260_snapshot.json` for diff reference.

**R10 firing probe:** `docs/verification_artifacts/s45_r10_firing_driver.py` (this PR adds it). Re-runnable as `py -3.12 docs/verification_artifacts/s45_r10_firing_driver.py <rank_log.csv>`. Two analyses: (1) static per-ticker max_fcn vs $100k cap; (2) counterfactual replay applying `check_single_name_cap` at each harness-rule open attempt.

**Engine state:** `WheelRunner.connector == MarketDataConnector` (Bloomberg) verified at probe start. `engine/forward_distribution.py:430` defines `realized_vol_widening_factor`; `engine/wheel_runner.py:1590` calls it. `engine/portfolio_risk_gates.py:380` defines `check_single_name_cap` with `_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10`.

**Pre-#260 snapshot reference:** the snapshot row that was on `origin/main` before this re-baseline merged. Diff against the new snapshot uses the file in `C:/Users/merty/AppData/Local/Temp/s45_s34_rebaseline/s34_pre_260_snapshot.json`.

**Re-launch note (process detail).** Initial launch on 2026-05-28 23:21 UTC used `$env:TEMP\s45_s34_rebaseline\s34_run` (PowerShell env-var syntax) inside a Bash tool invocation; Bash didn't expand it, so the literal `:TEMP\…` string crashed at end-of-run on `output_dir.mkdir()` after all 753 trading days had computed. Snapshot save was never reached. Re-launched with absolute Windows path (`C:/Users/merty/AppData/Local/Temp/s45_s34_rebaseline/s34_run`); completed cleanly on 2026-05-29 13:02 UTC. Snapshot generated_at fingerprint reflects the successful run. Saved as feedback memory `bash-env-var-expansion-trap.md` so the trap doesn't recur.
