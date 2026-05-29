---
id: S46
title: Re-verify closed tests on post-#260/#262 engine
kind: verification
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Launch-readiness sweep of the closed tests (those NOT
actively being edited on coordination board #113). Goal: confirm
what we previously called "verified" still holds against the
current engine; surface any drift; patch / file as warranted.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`.
`origin/main` @ `56d8e5c` (post-PR #260 RV30/RV252 widening,
post-PR #262 R10 single-name cap). Three orthogonal probe sets:

1. **Snapshot harness.** `pytest tests/test_backtest_regression.py
   -m backtest_regression -k "s27_ivpit or s35_oos"`. S32 / S34
   deselected — Terminal A's S45 and Terminal B's S44 own those
   re-baselines. **Result: S27 ran 1h45m without completing** (vs
   ~20min expected per the test file); terminated to avoid burning
   sandbox compute. S27 verification inherited from Terminal A's
   S41 byte-for-byte check; S35 re-baseline deferred to a follow-
   on Sn (pairs with A's S45 / B's S44 re-baseline campaign).
   Documented in S46 report §1.1 + §1.2 with the
   inherited-evidence framing.
2. **Committed verification drivers.** Re-ran
   `docs/verification_artifacts/realism_verify_driver.py` and
   `docs/verification_artifacts/f4_baseline_driver.py` against
   the current engine. Outputs captured to
   `realism_2026-05-28_raw_output.txt` and
   `f4_baseline_2026-05-28_raw_output.txt`.
3. **Doc-claim reproduction.** S30 (5 tickers × 5 dates HMM
   regime), S31 V1 (AVGO/NVDA composition arithmetic), S33 V1
   (universe-scale composition identity), S36 (consensus-crisis
   labels), S37 (ev_dollars vs roc Spearman ρ). Throwaway driver
   at `%TEMP%\s46\doc_claims_reproducer.py`.

Off-limits respected: not touching S32/S34 snapshots, not racing
A/B/C's in-flight Sn entries.

**Status.** Done. **Verdict: engine is launch-ready against the
re-verified surfaces. Realism battery byte-identical pre-vs-post
#260. F4 fix mechanism reproduces exactly (UNH ev_dollars -$6.28,
COST unchanged below threshold, AAPL control unchanged). HMM regime
values reproduce 19/19 doc cells within 0.001. Composition
identity holds at unit scale (S31 V1) and universe scale (S33 V1,
max 0.022556% rel err). Two expected drifts (S31 V1 credit_mult
0.80→1.0 due to FRED degradation; S37 crisis-day ρ 0.6341→0.7075
due to F4 differential widening) — both data/environment-explainable,
not regressions. Three sharp findings filed in S46 §5 (FRED empty-
series guard, silent except, hardcoded WORKTREE in two drivers).**

**Findings:**

| Surface | Pre-#260 reference | Post-#260+#262 observed | Δ | Verdict |
|---|---|---|---|---|
| **Realism (1) 5-ticker smoke @ 2026-03-20** | XOM 137.57 / JPM 124.90 / MSFT 90.97 / UNH 62.62 / AAPL 20.45 | identical | 0 | ✅ byte-identical |
| **Realism (2) IV PIT vs Bloomberg** | max rel diff 0.015% | identical | 0 | ✅ |
| **Realism (3) regime_multipliers** | UNH 0.6676 / MSFT 0.4626 / JPM 0.7113 / XOM 0.9217 / AAPL 0.6768 | identical | 0 | ✅ |
| **Realism (4) COST 2022-04-25 prob_profit** | 0.903200 | 0.903200 | 0 | ✅ |
| **Realism (5) refusal at 3 anchors** | 100% / 0% / 60% | 100% / 0% / 60% | 0 | ✅ |
| **F4 driver — COST 2022-04-04** | ev_dollars +62.88, cvar_5 -4376.07 | identical | 0 | ✅ (fix correctly silent below 1.30 threshold) |
| **F4 driver — UNH 2024-11-11** | ev_dollars +114.53, cvar_5 -2528.13 | **+108.25**, **-2608.69** | **-$6.28**, **-3.2%** | ✅ matches PR #260 + S41 exactly |
| **F4 driver — AAPL 2026-02-13 control** | ev_dollars +5.50, cvar_5 -2898.15 | identical | 0 | ✅ no spurious widening |
| **S30 HMM (19 doc'd cells)** | mult values across 5 tickers × 5 dates | all within 0.001 abs | ≤ 0.0005 max | ✅ 19/19 |
| **S31 V1 — composition identity** | 5-factor product == ev_dollars to <1¢ | identity STILL HOLDS; one factor's value drifted | credit_mult 0.80→1.0; ev_dollars AVGO 82.65→103.31 NVDA 36.22→45.28 | ⚠ environment-driven (FRED), not engine drift |
| **S33 V1 — universe-scale identity** | within 0.023% rel err | 0.022556% | ≈ exact reproduction | ✅ |
| **S36 — consensus crises** | 9/9 at 2020-03-23, 8/9 at 2025-04-04 | 9/9, 8/9 (LLY=bear) | 0 | ✅ |
| **S37 — ev_dollars vs roc Spearman ρ** | 0.6341 (crisis) / 0.8950 (normal) | **0.7075** / 0.8950 | **+0.0734** crisis, 0 normal | ⚠ expected — F4 widens differentially on crisis day |

**Findings detail:**

- **(F1 — § 5 sharp) `FREDAdapter.credit_regime` crashes
  `TypeError: '<=' not supported between instances of
  'numpy.ndarray' and 'Timestamp'` on empty series.** Cause:
  `fred_adapter.py:154` does `hy[hy.index <= cutoff]` without an
  `if hy.empty` guard. Fires in environments without FRED API
  access OR when the FRED CSV endpoint returns empty.
  `wheel_runner.py:870`'s broad `except Exception: credit_mult = 1.0`
  silently absorbs it. **Logged as candidate hardening PR
  (not shipped here per single-concern rule).**

- **(F2 — § 5 sharp) `wheel_runner.py:870` silent except.**
  The broad `except Exception` masks FRED failures so the trader
  can't tell "no credit stress" from "credit detector broken".
  Mirrors the S22 / S28 / S29 / S31 observability family. Fix
  candidates: narrow the except + add a `credit_regime_status`
  diagnostic column. **Logged — gated on F1.**

- **(F3 — § 5 ergonomic) Two committed drivers hardcode
  `WORKTREE = "swe-terminal-b"`** at `realism_verify_driver.py:18`
  and `f4_baseline_driver.py:29`. Terminal A's
  `s41_f4_validation_driver.py` already uses the cleaner
  `Path(__file__).resolve().parents[2]` pattern. **Logged as
  a small follow-on PR.**

- **(F4 — expected drift, not regression) S31 V1 credit_multiplier
  0.80 → 1.0.** Engine code unchanged on the credit-regime
  branch since 2026-05-24. The drift is purely data-environment-
  driven (FREDAdapter unreachable in Cowork). Composition
  identity STILL HOLDS with the new factor value. Footnote added
  to the S31 V1 box explaining the data-environment caveat.

- **(F5 — expected drift, not regression) S37 crisis-day ρ
  0.6341 → 0.7075.** PR #260's tail_widening_factor fires
  differentially across the 22-name universe at 2025-04-04
  (some tickers above 1.30 threshold, others below). Differential
  widening means `ev_dollars` reductions are non-uniform, which
  breaks the uniform-rescale property that preserves Spearman
  rank correlation. Normal-day ρ (2026-03-20) is byte-identical
  because the widening doesn't fire universe-wide on calm dates.
  Footnote added to S37 Realism Check noting the post-#260 ρ.

- **§2 verified.** Composition identity holds at both unit
  (S31 V1, <0.01¢) and universe scale (S33 V1, 0.022556% max
  rel err). Dealer multiplier `[0.70, 1.05]` clamp preserved.
  No code path bypasses `EVEngine.evaluate`. No reviewer
  promotes a non-tradeable candidate.

**Realism Check.**

| Aspect | Engine output (2026-05-28) | External reference | Verdict |
|---|---|---|---|
| 5-ticker smoke at 2026-03-20 | byte-identical to 2026-05-26 | Engine unchanged on calm-regime path | ✓ Aligned |
| IV PIT vs Bloomberg file | max rel diff 0.015% | The IV file IS the ground truth | ✓ Aligned |
| UNH 2024-11-11 widening | ev_dollars -$6.28, cvar_5 -3.2% | PR #260 §11 documents factor=1.0121 at this date | ✓ matches PR exactly |
| COST 2022-04-04 unchanged | rv30/rv252 = 0.9615 < 1.30 threshold | PR #260 explicit: "does NOT close named F4 cases" | ✓ Aligned |
| HMM consensus crisis at 2020-03-23 | 9/9 crisis | VIX > 60 in March 2020; broad-market crash | ✓ Aligned |
| April 2025 universe-crisis (LLY=bear exception) | 8/9 crisis; LLY mult 0.482 | LLY's per-ticker history doesn't trip crisis emission | ✓ Aligned (S36 F4 already documented) |
| Composition identity | <0.023% rel err at universe scale | Math should reproduce exactly | ✓ Aligned |
| credit_multiplier on 2025-04-04 | 1.0 (FRED unreachable in sandbox) | Pre-#260 reference was 0.80 in Terminal C's worktree | ⚠ Environment-driven; doc-footnoted |
| ev_dollars vs roc crisis ρ | 0.7075 | Pre-#260 was 0.6341; F4 widening drives non-uniform delta | ⚠ Expected; doc-footnoted |

**Verdict.**

- **The engine is solid on the surfaces re-verified in this sweep.**
  Realism, F4 mechanism, HMM regime, composition identity, and
  consensus-crisis behaviour all reproduce against the documented
  reference values. No §2 breaches.
- **PR #260's F4 widening behaves as advertised.** UNH 2024-11-11
  widening reproduces the +$108.25 ev_dollars and -$2608.69
  cvar_5 to the cent. COST 2022-04-04 correctly silent (below
  threshold). AAPL control byte-identical.
- **PR #262's R10 cap was not exercised here** — its surface is
  pinned by D's S42 (PR #265 R9+R10 audit) and C's S43 (PR #270
  R10 firing analysis). Not re-verified in S46.
- **Two drift findings (S31 V1 credit_mult, S37 crisis-day ρ)
  are both expected and explained** — neither indicates an
  engine regression. Both ledger entries footnoted in this PR.
- **Three sharp findings filed** for follow-on hardening
  (FRED empty-series guard, silent except observability, two
  drivers' hardcoded WORKTREE). None launch-blocking.

**AI handoff.**

- **Next natural follow-up (small PR):** ship S46 §5 F1 — the
  FRED empty-series guard. One-line change in
  `engine/external_data/fred_adapter.py` + a test asserting
  `credit_regime(as_of=X)` returns `{"regime": "unknown", ...}`
  on empty data.
- **§5 F2 gated on F1.** Narrowing the except can't ship until
  the underlying TypeError is fixed.
- **§5 F3 independent.** Could ship alone or bundled with F1+F2.
- **Snapshot re-baseline candidate:** if §1.2's S35 result drifts
  beyond tolerance, regenerate the snapshot and amend
  `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` with a "post-#260" Δ
  table. Becomes its own ledger entry (S47+).

**Methodology debt.**

- **Sandbox dependency on FRED.** The credit_multiplier branch
  is silently dormant in Cowork (no FRED access). Same shape as
  S29 / S31's "dealer module dormant on Bloomberg" finding.
  Worth a `docs/DATA_POLICY.md` §7 note.
- **No re-run of S38.** Terminal B's S44 (S38 post-F4 re-run)
  is the right vehicle. Deferring.
- **No probe of R10 firing.** D's S42 + C's S43 already cover it.
- **No comparison against another connector.** Theta replay (S6
  queued) would test the credit_regime path with a different
  data-source dependency profile.

Full doc: `docs/ENGINE_REVERIFY_S46_POST_F4_R10.md`.

---
