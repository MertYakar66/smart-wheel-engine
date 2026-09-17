# Heavy-test validation + defensive-guard audit — 2026-06-09

Autonomous executor run (8h window). Validates the three (E) fixes (#382/#384/#386)
and sweeps the quant layer for the same class of defect. **Read-only audit + heavy
tests; no engine merges, no PRs, no decision-trio edits.** All findings are
probe-backed (every claim has an executed Python-3.12 probe with output).

Baseline engine = `origin/main @ 1985547` (worktree `swe-main`, detached).
Integration = `1985547` + #382/#384/#386 engine+test files (throwaway branch
`claude/efix-integration-validation` in worktree `swe-efixes`).

---

## 1. Bottom line

- **All 3 (E) fixes verified correct, minimal, and §2-safe.** Recommend merge.
- **#386 is determinism-null for `ev_dollars` / the backtest fingerprint, AND fixes a
  real operator-facing bug on a real name** (BIIB fabricated 'crisis' → honest
  'unknown'). See §3.
- **The defense rests on data integrity + upstream IV filters, not on the on-path
  primitives failing loud.** Several on-path quant primitives swallow NaN/inf the
  same way #382/#384/#386 did; today they're unreachable only because real OHLCV is
  ~clean and `wheel_runner`/`R1a` strip bad IV before pricing. The 4 real NaN OHLCV
  bars (§4) show the "clean data" assumption is not absolute. (§5)
- **17 sweep findings, all mechanically reproduced; only 1 reachable on real data**
  (the BIIB HMM case, already fixed on the live path by #386). The rest are
  off-path/dormant hygiene. (§5–§6)

---

## 2. (E)-fix verdicts (workflow `wf_546f05d6-22f`, 26 agents, probe-backed)

| Issue | Verdict | Orig bug reproduced | Resolves | Valid input byte-identical | On evaluate path | §2-safe |
|---|---|---|---|---|---|---|
| **#382** realized_vol `_log_ratio` | **solid** | yes (+ a 3rd, worse mode found) | yes | yes | **no** (proven via import tracer) | yes |
| **#384** copula `_validate_t_copula_df` | **solid-with-notes** | yes | yes | yes | no | yes |
| **#386** HMM fit non-finite guard | **solid-with-notes** | yes | yes | yes | **yes** (but determinism-null) | yes |

Notes that matter:
- **#382**: the verifier reproduced a **third corruption mode beyond the two in the
  issue** — `open=0` in Garman-Klass drives the window variance to `-inf`, which
  `max(var,0.0)` clamps to **0.0**, silently returning **zero realized vol** →
  `vol_risk_premium_bundle` reports **maximal premium richness** (`consensus_vrp =
  iv`). The fix turns all such bars into an honest NaN. Off the evaluate path
  (confirmed: `engine.realized_vol` is imported by **no** production engine module;
  a `builtins.__import__` tracer over a live 5-ticker rank showed it never loads).
- **#384**: precise framing — `df<=0` already raised cleanly on baseline; the genuinely
  new closes are `df=nan/inf` (was a cryptic `IndexError`) and the silent `0<df<=2`
  infinite-variance compute. Sibling operands (marginals/weights/corr/n_samples) are
  still unguarded but pre-existing and off-path (see findings #9/#10).
- **#386**: determinism-null is real (ev_engine already clamps a non-finite
  `regime_multiplier` to 1.0). Its true value is **diagnostic correctness** — see §3.

---

## 3. Real-data determinism proof (BIIB NaN bar)

`BIIB` is in `UNIVERSE_100` (S34). The connector serves `BIIB` close on
`2023-06-09` and `2020-11-06` as **NaN** (it does not drop them). Ranking BIIB at
`as_of=2020-11-20` (504-day HMM tail contains the NaN), baseline vs fixed:

```
                 baseline(main)     fixed(#386)
ev_dollars       237.68             237.68      <-- IDENTICAL
ev_raw           237.68             237.68
regime_multiplier 1.0               1.0
hmm_multiplier   nan                1.0
hmm_regime       'crisis'  (FAKE)   'unknown'   <-- honest
dealer_multiplier 1.0               1.0
```

- `ev_dollars` identical → **#386 cannot move the backtest fingerprint**
  (`ev_dollars`/`realized_pnl`/`nav`/counts are the only snapshot metrics; the
  diagnostic `hmm_multiplier`/regime label are not in the rank log). The sandbox
  `MarketDataConnector` has no option chain, so `dealer_multiplier` is always 1.0 →
  no scenario where the baseline clamp (NaN→1.0) eats a non-trivial dealer factor.
- Baseline fabricates a **'crisis'** regime label for BIIB purely because the HMM
  choked on a NaN bar; #386 makes `fit` raise → `wheel_runner` neutral fallback →
  honest **'unknown'**. `predict_proba` is never reached on the live path because
  `fit` is called first with the same tail (so finding #1's residual is not live).

**Implication:** merging #386 is safe (zero fingerprint impact) and corrects a real
misleading output on a real S&P 500 name. The full 5h backtest A/B
(`swe-main` vs `swe-efixes`, all 4 reproducers) is running to confirm byte-identity
across the whole 2020–2024 execution + report drift vs the committed snapshots.

---

## 4. Data-integrity scan (`data/bloomberg/sp500_ohlcv.csv`, 1,014,920 rows, 2018→2026)

- No zero/negative OHLC anywhere (min close 1.42).
- **Exactly 4 NaN close bars**: `BIIB 2020-11-06`, `BIIB 2023-06-09` (both in
  range; BIIB ∈ U100 → exercised by S34), `TPL 2019-05-16`, `TPL 2019-07-09`
  (pre-window). So the "log-returns of positive finite prices are always finite"
  premise is *mostly* true but **not absolute** — NaN bars exist and reach the HMM
  tail for BIIB. This is what makes the on-path findings (§5) non-theoretical.

---

## 5. On-evaluate-path findings (same class as the fixes; currently contained)

All reproduced; all currently **unreachable in production** only because of data
integrity + upstream filters — not because the primitive fails loud. Listed because
the BIIB NaN bar shows the containment is thinner than assumed.

| # | File:line | Defect | Why contained today |
|---|---|---|---|
| 1 | `regime_hmm.py` predict_proba/position_multiplier | NaN obs → NaN posterior → fabricated 'crisis'/poisoned cache | **#386 fixes the live path** (fit raises first). Residual = fit-clean-then-predict-poisoned, not on wheel_runner path. |
| 2 | `forward_distribution.py:560-562` (`realized_vol_widening_factor`) | a **zero/negative** close → `log`→`-inf` survives → NaN rv-ratio → `min(1.15,NaN)` falls through → silent **max widening 1.15** | **Probe-confirmed NOT reachable on the real data.** `realized_vol_ratio` handles **NaN** closes (returns finite): BIIB @2020-11-20 ratio=2.207→widen=1.15 is *legitimate* (real Nov-2020 BIIB aducanumab-crash vol spike, NaN day excluded); AAPL stays 1.0. Only a literal zero/neg close would misfire, and there are **none** in the data. Untouched by the 3 fixes; latent defense-in-depth only. |
| 3 | `forward_distribution.py:254,288-302` (`har_rv_conditional_distribution`) | non-positive close at window edge → all-NaN scenario set | needs thin history (~60 bars) AND a bad edge bar; cascade rarely selects har tier |
| 4 | `skew_dynamics.py:168` (`skew_slope`) | `max(iv_atm,1e-6)` guards 0 but not NaN/inf → NaN slope, no flag | `wheel_runner.py:1511` pre-filters `all(0<v<=3.0)` → strips NaN/inf IV first |
| 5 | `option_pricer.py:57` validator → BSM price/Greeks | `_validate_inputs` lets NaN sigma/T past the `sigma<0` contract → silent NaN price/Greeks; reached from `ev_engine.py:371` | `wheel_runner` IV guard + R1a (`ev_non_finite`→blocked) strip it before/after |
| 6/7 | `dealer_positioning.py:327-333,416-417` | non-finite OI → `int()` `OverflowError` (caught → overlay dropped); negative OI → flips gex sign, mislabels regime, drives mult→0.70 (clamp **not** breached) | sandbox connector exposes no chain; data-quality gate. Self-degrading. |

**Architectural note:** `forward_distribution.py:127` (`np.log(prices)`) — the *actual*
on-path empirical-vol log — is unguarded the same way `_log`/`_log_ratio` are; the
accompanying test asserts non-finite returns are filtered before EV, so it's
contained, but it's the on-path twin of #382. A consolidated "fail-loud the on-path
quant primitives (forward_distribution log + widening + BSM sigma/T validator)"
follow-up would move the safety from "data happens to be clean" to "primitive
refuses bad input." **Held for user direction (decision-layer-adjacent → §2).**

---

## 6. Off-path findings (dormant / hygiene; low priority)

8 `block_bootstrap` NaN leak (tier never selected) · 9 copula corr-matrix non-finite
skips PSD-repair → cryptic crash · 10 copula NaN marginals/weights → NaN CVaR masked
as **`'negligible_tail_dependence'`** (falsely reassuring) · 11 `gpd_var_cvar`
`confidence>1` → uninterpretable complex/`TypeError` (evaluate hardcodes 0.99) · 12
`RegimeDetector` 0.20 vol default / FLAT-on-garbage term structure / fabricated
abs-threshold percentile · **13 `SplineVolSurface.get_iv` returns flat 0.20 on empty
data** — the suspected 4th-(E); confirmed real, **but the class is never instantiated
and it's already a known/deferred MP-D item** (`docs/worklog/mp-d-...:66`) · 14
`estimate_iv_for_delta` returns `(spot,0.20)` on T≤0 (zero callers) · 15
`NelsonSiegel.fit` fabricates `beta0=0.20` on <2 points (converged=False exposed) ·
16 `american_option_greeks` dS=0 → NaN; binomial n_steps=0 → `ZeroDivisionError` · 17
vectorized BS substitutes `0.2` for negative sigma (scalar raises) — diverges, off-path.

---

## 7. Prioritized recommendations (for user triage — nothing actioned here)

1. **Merge #382, #384, #386** — all verified; #386 has a demonstrated real benefit
   (BIIB) with zero fingerprint impact. (Pending the 5h backtest A/B confirmation.)
2. **Consolidated on-path fail-loud follow-up** (medium): `forward_distribution.py`
   :127 log + `realized_vol_widening_factor` NaN-ratio guard + `option_pricer`
   `_validate_inputs` finite-sigma/T check. Same class as the 3 fixes, on the live
   path, contained-not-safe. Decision-layer-adjacent → §2 review.
3. **#1 predict_proba/viterbi defense-in-depth guard** (low; #386 already covers the
   live path).
4. **#10 honest copula degenerate verdict** (low, off-path) — don't label a NaN CVaR
   `'negligible_tail_dependence'`.
5. **#13 SplineVolSurface fail-loud** (low) — already tracked as deferred MP-D.

---

## 8. Reproduction

- Workflow: `wf_546f05d6-22f` (transcript under the session subagents dir).
- Probes: `_data_integrity_probe.py`, `_biib_ab_probe.py` (run per-root),
  `_efix_compare.py` (A/B + drift), `_efix_backtest_driver.py` (per worktree).
- Backtest A/B logs: `swe-main/baseline_run.log`, `swe-efixes/integ_run.log`;
  payloads under each `.efix-validation/`. (All scratch files are untracked.)

## 9. Backtest A/B determinism + snapshot-drift result (completed 2026-06-10)

Run 1 was killed by a Windows-Update reboot at 01:29:55 (EventLog 6006/6005; the
"exit code 4" was the external kill, not a code failure). Surviving payloads
(integ s27/s32/s35 + main s34) were kept; only the complements were re-run.

**A/B DETERMINISM: OVERALL PASS — all 4 backtests output-identical.**

| Reproducer | Metric leaves | A/B | data sha match |
|---|---|---|---|
| s27_ivpit_24t_100k | 46 | byte-identical | yes |
| s32_friction_24t_1m | 184 | byte-identical | yes |
| s34_universe_100t_1m | 184 | byte-identical | yes |
| s35_oos_24t_100k | 136 | byte-identical | yes |

The #386 determinism-null claim (§3) is confirmed end-to-end on full 2020–2024
executions. The 3 fixes were merged the same day: PR #397 (#382), #398 (#384),
#399 (#386), each 9/9 CI green, after PR #400 unblocked the lint job that D27
had left red on main's own `tests/test_testing_md_taxonomy.py`.

**Snapshot drift (pre-existing on main, NOT from the fixes — identical leaf-for-leaf
in both arms):** s35 GREEN; s27 6 / s32 24 / s34 32 drifting leaves, all confined to
`ev_mean` (rel ≈ 3e-4..2e-3 vs tol 1e-4), `pnl_mean` (s34 only), and
`spearman_p`/`p` (rel ≈ 0.08–0.11 vs tol 0.01; `spearman_rho` itself within
tolerance everywhere). Snapshots were last baselined at #338; the prime suspect for
the in-window movement is the #363 IV-sentinel connector gate (nulls implied vol
outside (3.0, 10000]), which is the main engine-visible data-path change since.
Needs the standard revert-isolation attribution + re-pin
(`--update-snapshot`) per the slow-lane drift protocol — filed as a follow-up issue.
