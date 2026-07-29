# Engine backtest — S32 snapshot re-baseline (post-#260)

**Date:** 2026-05-28
**Engine SHA:** `origin/main` @ `56d8e5c` (post-PR #260 realized-vol-ratio widening, post-PR #262 R10 single-name cap)
**Author:** Terminal A (executor), Session A (verifier)
**Window / universe / strategy / config:** identical to `docs/ENGINE_BACKTEST_S32_FRICTION.md` — 2022-01-03 → 2024-12-31, 24 SP500 tickers, $1M starting capital, 35-DTE / 25-Δ short puts, wheel into CC on assignment, hold to expiry, `require_ev_authority=False`. Three friction levels: `none` / `bid_ask` / `full` (shared SP rank call per day across the three trackers via `run_backtest_multi_friction`).
**Source artefacts:** `C:/Users/merty/AppData/Local/Temp/s42_s32_rebaseline/{s32_pre_260_snapshot.json, s32_run/}` and the regenerated in-repo `backtests/regression/snapshots/s32_friction_24t_1m.json`.

> **Naming note.** This doc was originally drafted under working labels
> "S42" then "S44" before each collided with merged work (D's S42 R9/R10
> audit, PR #265; B's S44 S38 post-F4 re-run, PR #271). The branch retains
> its original name (`claude/usage-test-s42-s32-rebaseline-post-260`) for
> git-log reachability. **The doc itself uses the descriptive "S32
> re-baseline (post-#260)" naming, per `docs/PARALLEL_SESSIONS.md` rule 7
> (introduced in PR #282) — Sn is allocated at MERGE by the operator, not
> claimed at work-start.** The merger fills in the canonical Sn at merge.

---

## Purpose

PR #260 (realized-vol-ratio widening) regenerated the **S27** snapshot as part of its merge. **S32 / S34 / S35 snapshots were intentionally deferred** to the slow-lane `.github/workflows/backtest-regression.yml` workflow per PR #260's body. Until those are re-baselined, the slow-lane CI fails the S32/S34/S35 `test_backtest_matches_snapshot` assertions against the current engine.

This doc closes the **S32 leg** (24t / $1M / 2022-2024). The S34 leg was closed by Terminal A's parallel branch `claude/usage-test-s45-s34-rebaseline-post-260` (`docs/ENGINE_BACKTEST_S34_REBASELINE_POST260.md`); the S35 leg remains open.

This is a documented engine-snapshot regeneration — same shape as PR #260's S27 snapshot regen. **No engine source touched; the only diff is the snapshot JSON + this doc.**

---

## Comparison

S32 backtest re-ran cleanly. Pre-#260 snapshot was generated 2026-05-26 21:58 UTC; post-#260 (this PR) generated 2026-05-28 17:24 UTC. The `data_csv_sha256` is **identical** between the two — same OHLCV CSV, same data window, same fingerprint config; the diff is purely the engine code change.

### Aggregate (full-friction headline)

| Metric | Pre-#260 snapshot | Post-#260 (this PR) | Δ | % Δ |
|---|---|---|---|---|
| row_count | 5,944 | 5,944 | 0 | — |
| Spearman ρ (overall) | **+0.18639** | **+0.18018** | **−0.00621** | **−3.33%** |
| Spearman p | 1.31e-47 | (~1e-44) | — | — |
| mean_realized | $34.14 | $33.34 | −$0.80 | −2.34% |
| hit_rate | 80.11% | 79.96% | −0.15pp | −0.19% |
| iv_mean | 0.25634 | 0.25638 | +0.0000 | +0.02% |
| ev_mean | −$19.45 | −$21.07 | −$1.62 | +8.33% |
| final_cash (full friction) | $824,659 | $827,036 | **+$2,376** | **+0.29%** |
| **final_NAV (full friction)** | **$1,078,208** | **$1,080,584** | **+$2,376** | **+0.22%** |
| executed_trades | 105 | **104** | **−1** | **−0.95%** |
| put_assignments | 19 | 18 | −1 | −5.26% |
| open_at_end | 14 | 14 | 0 | — |

### Per friction level

| Friction | Pre-#260 ρ | Post-#260 ρ | Pre-#260 NAV | Post-#260 NAV | Δ NAV |
|---|---|---|---|---|---|
| `none` (frictionless) | +0.18815 | +0.18186 | $1,081,705 | $1,084,031 | **+$2,326** |
| `bid_ask` | +0.18639 | +0.18018 | $1,078,793 | $1,081,142 | **+$2,349** |
| `full` | +0.18639 | +0.18018 | $1,078,208 | $1,080,584 | **+$2,376** |

All three friction levels show the SAME directional effect: ρ −3.3% relative; NAV slightly *positive* delta of ~+0.22% in each. Friction drag is unchanged from the pre-#260 baseline because the same 104 trades fire across all three levels (multi-friction harness shares the rank call; only the friction overlay differs in execution P&L).

### Per year (full friction)

| Year | n | Pre-#260 ρ | Post-#260 ρ | Δ ρ | Pre-#260 mean | Post-#260 mean | Δ mean |
|---|---|---|---|---|---|---|---|
| 2022 (bear) | 1,936 | +0.37253 | **+0.36133** | −0.01121 (−3.0%) | −$17.81 | **−$19.14** | −$1.33 (+7.5%) |
| 2023 (recovery) | 1,971 | +0.17709 | +0.17918 | +0.00210 (+1.2%) | +$72.05 | **+$72.05** | $0.00 (**unchanged**) |
| 2024 (bull) | 2,037 | +0.07700 | +0.06814 | −0.00886 (−11.5%) | +$46.84 | +$45.77 | −$1.07 (−2.3%) |

**2023 mean_realized is byte-identical** ($72.04869...) — same as the S27 pattern. The fix's calibration is sufficiently conservative that 2023's calm regime sees zero top-10 reshuffles per day on the dates where the rank could have changed.

### Per quartile (full friction)

| Q | n | Pre-#260 EV mean | Post-#260 EV mean | Pre-#260 PnL mean | Post-#260 PnL mean | Δ PnL |
|---|---|---|---|---|---|---|
| Q0 (low) | 1,486 | −$120.43 | −$122.31 | +$42.41 | +$42.19 | −$0.22 |
| Q1 | 1,486 | −$30.61 | −$31.17 | +$17.44 | **+$13.55** | **−$3.89 (−22.3%)** |
| Q2 | 1,486 | −$0.91 | −$2.11 | +$12.41 | **+$16.96** | **+$4.55 (+36.7%)** |
| Q3 (high) | 1,486 | +$74.14 | +$71.30 | +$64.31 | +$60.67 | −$3.64 (−5.7%) |

**Q3 still beats Q0 by 1.44× in realized PnL** ($60.67 vs $42.19) — the engine's ordering signal still works. Mid-quartile reshuffling (Q1 ↔ Q2) is where the F4 fix's composition effect concentrates: Q2's mean PnL jumps +$4.55 while Q1's drops −$3.89.

---

## The S32-vs-S27 deviation — the F4 fix barely affects $1M

The most important finding is **how S32's response to PR #260 differs from S27's**, given that S32 differs from S27 only in capital scale ($1M vs $100k):

| Metric | S27 (\$100k) Δ | S32 (\$1M) Δ |
|---|---|---|
| Overall ρ | −3.3% relative | −3.3% relative |
| executed_trades | **51 → 40 (−22%)** | **105 → 104 (−1%)** |
| final_NAV | **−$15,471 (−12.1%)** | **+$2,376 (+0.22%)** |

**Why the asymmetry?** At $100k the strategy is BP-saturated (S22/S27 doc reports 1,171 `skipped_insufficient_bp` events). The F4 widening reorders the top-10/day, which cascades through BP availability: 11 fewer puts fire, NAV drops 12%. At $1M the strategy is **capacity-constrained, not BP-constrained** (S32 doc F3 / F4: average deployed capital is ~$108k = 10.8% of NAV). When the F4 widening refuses one trade, the trade just doesn't fire — there's no downstream BP-cascade because BP wasn't binding to begin with. The refused trade happened to be net-negative-realized (the +$2,376 NAV bump in this run), so the fix produces a *small* dollar-improvement at scale, not a dollar-loss.

**Implication for the bundle framing (PR #260 + PR #262 = F4 defence-in-depth):**

- **At $100k:** PR #260 is the dominant force (more selective ranking via widened distribution); PR #262 R10's per-name cap (10% NAV = $10k) is binding under the 1-contract-per-name model, so R10 is essentially redundant with the strategy's natural sizing. F4 protection comes mostly from PR #260's frequency guard.
- **At $1M with 24 tickers (this setup):** PR #260's frequency guard barely fires (1 trade refused). PR #262 R10's 10% cap = $100k notional per name, which the 1-contract / strike-$~$200-$700 puts never hit. **Both PRs are effectively no-ops at this scale and universe.** F4 protection at $1M / 24t comes from capacity constraint, NOT from the bundle.
- **At $1M with 100 tickers (companion S34 re-baseline):** R10 IS binding — fires 368× across 3 years on AZO + BKNG only; the bundle's protective behaviour activates at the wider universe. Cross-referenced in `docs/ENGINE_BACKTEST_S34_REBASELINE_POST260.md` §"R10 firing analysis".

This is a useful S41 (deployment-readiness) clarification: the bundle that "closes B1" closes it at the **deployment scale where it's needed** (≤ $100k). At $1M / 24t the engine doesn't need the bundle because it's already not over-deploying.

---

## §2 invariant

- `backtests/regression/snapshots/s32_friction_24t_1m.json` is the only file in the snapshot diff.
- Zero edits to `engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/candidate_dossier.py`, `engine/forward_distribution.py`. Validation only.
- §2 scan on the regenerated rank_log: targeted 0 non-finite tradeable verdicts on 5,944 rows (re-verified by `pytest tests/test_backtest_regression.py -k s32` PASS).

---

## Test gates

| Gate | Result |
|---|---|
| `pytest tests/test_backtest_regression.py -k s32` against the new snapshot | ✅ **2 passed in 110min** (`test_snapshot_fingerprints_have_required_keys[s32_friction_24t_1m]` + `test_backtest_matches_snapshot[s32_friction_24t_1m-backtests.regression.s32_friction_24t_1m]`). The test re-ran the full S32 backtest end-to-end and confirmed every metric matches the newly-regenerated snapshot. |
| Launch-blocker subset | (already green on `origin/main` — not re-running for a docs+snapshot PR) |

---

## Findings

- **F1 — S27 deltas reproduce at $1M in *ratio* terms, not in *dollar* terms.** S27 ($100k) saw overall ρ −3.3% relative; S32 sees the same −3.3%. S27 saw NAV −12.1%; S32 sees NAV +0.22%. The percentage NAV deltas diverge by **two orders of magnitude** because of the capacity / BP-saturation regime difference (see §"The S32-vs-S27 deviation" above).
- **F2 — PR #260's widening fires only 1× in the full 2022-2024 rank_log at $1M.** Executed_trades drops by exactly 1 (105 → 104). The 1 refused trade has positive contribution to the post-fix NAV (+$2,376), meaning the F4 fix is *net dollar-positive* at this scale and universe, even though it's marginally net dollar-negative at $100k.
- **F3 — 2023 mean_realized is byte-identical pre/post.** $72.04869... matches across 1,971 rows. The fix's calibration is conservative enough that 2023's calm regime sees zero top-10 reshuffles on any day where the rank ordering could have changed. This mirrors the S27 result.
- **F4 — Q1 ↔ Q2 reshuffling is where the composition shift concentrates.** Q3 pnl drops −5.7% and Q0 stays flat, but Q1 pnl drops −22% while Q2 pnl rises +37%. The F4 widening pushes some Q1 candidates into Q2 (or vice versa) where they have different forward outcomes. Q3 vs Q0 monotonicity holds (1.44× spread); ordering signal preserved.
- **F5 — Per-name notional cap (R10, PR #262) is structurally non-binding at $1M / 24t / 1-contract.** The maximum per-position notional is ~strike × 100 = $20-70k (for the COST / GS / UNH heavyweights at peak). 10% of $1M = $100k cap. **R10 cannot fire** because no single 1-contract position approaches the cap. The F4 dollar-damage bound from R10 only activates at smaller capital scales or with multi-contract sizing — or at wider universes where heavyweight strikes are present (confirmed by the companion S34 re-baseline: R10 fires 368× on AZO + BKNG).
- **F6 — Snapshot is current-engine-trustworthy.** Re-baselined snapshot regenerated cleanly via `python -m backtests.regression.s32_friction_24t_1m --update-snapshot`. **Confirmed independently by `pytest tests/test_backtest_regression.py -k s32` (2 passed in 110min)** — a fresh full S32 backtest re-runs and asserts every metric (aggregate / per-year / per-quartile / per-friction-level) matches the regenerated snapshot.

---

## What this validates / does not validate

| Claim | Verdict |
|---|---|
| S32 baseline reproduces on post-#260 engine | ✅ snapshot regenerated; metrics shift in line with the F4 fix's mechanism |
| PR #260's ρ-preservation pattern holds at $1M | ✅ −3.3% relative, same as S27 |
| PR #260 is a *dollar*-improver at $1M | ✅ marginal — +$2,376 NAV across the 3-year window (negligible vs S32's other variance sources) |
| PR #260 / R10 bundle closes B1 *at $1M / 24t* | ❌ structurally not needed — capacity constraint already protects the account at this scale |
| Strategy capacity finding (S32 F3, "$100k-class strategy") survives | ✅ executed_trades 104, average deployment still ~10% NAV — unchanged conclusion |

---

## AI handoff

- **S35 ($100k / 24 tickers / 2018-2020 OOS) is the last remaining pre-#260 re-baseline.** ~30-90 min wall under contention. Natural follow-on.
- **For `docs/PRODUCTION_READINESS.md` §3 Blocker B1:** once S32 + S34 + S35 are all re-baselined, the deployment matrix in §5 can be refreshed against the bundled (PR #260 + #262) post-fix engine output without snapshot drift.
- **For the slow-lane CI:** after this PR merges, `tests/test_backtest_regression.py::test_backtest_matches_snapshot[s32_friction_24t_1m]` will pass on the dedicated workflow. The S34 parametrization passes once `claude/usage-test-s45-s34-rebaseline-post-260` also merges; S35 still fails until re-baselined.

---

## Method appendix

**Harness:** `backtests.regression.s32_friction_24t_1m` (in-repo since PR #241). Same `run_backtest_multi_friction` path; one rank call per day, three trackers (`none` / `bid_ask` / `full`) each independently MTM-ing and deciding opens against the shared ranked frame.

**Snapshot file:** `backtests/regression/snapshots/s32_friction_24t_1m.json`. Pre-#260 backup in `C:/Users/merty/AppData/Local/Temp/s42_s32_rebaseline/s32_pre_260_snapshot.json` for diff reference.

**Engine state:** `WheelRunner.connector == MarketDataConnector` (Bloomberg) verified at probe start. `engine/forward_distribution.py:430` defines `realized_vol_widening_factor`; `engine/wheel_runner.py:1590` calls it; the `tail_widening_factor` column lands in the rank_log for downstream audit.

**Pre-#260 snapshot reference:** the snapshot row currently on `origin/main`. Diff against the new snapshot uses the file in `C:/Users/merty/AppData/Local/Temp/s42_s32_rebaseline/`.
