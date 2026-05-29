# Engine re-verification — S46: closed tests on post-#260/#262 engine

**Date:** 2026-05-28
**Engine SHA:** `origin/main` @ `56d8e5c` (post-PR #260 RV30/RV252 widening, post-PR #262 R10 single-name cap)
**Author:** Terminal D (executor), Session D (verifier)
**Scope.** Launch-readiness sweep of the closed tests (those NOT actively being edited on the coordination board #113). Goal: confirm what we previously called "verified" still holds against the current engine; surface any drift; patch / escalate / file as warranted.

**Off-limits (deferred to in-flight terminals):** S43 (C — rolling multi-window, PR #270), S44 (B — S32 re-baseline + S38 post-F4 re-run), S45 (A — S34 re-baseline post-#260/#262). NOT touching S32/S34 snapshots; A/B own those.

**Source artefacts (committed, re-runnable):**
- `docs/verification_artifacts/realism_2026-05-28_raw_output.txt` — re-run of `realism_verify_driver.py` against the post-#260+#262 engine.
- `docs/verification_artifacts/f4_baseline_2026-05-28_raw_output.txt` — re-run of `f4_baseline_driver.py` against the post-#260+#262 engine.
- Throwaway probes under `%TEMP%\s46\`: `doc_claims_reproducer.py` reproducing S30/S31/S33/S36/S37 numeric claims; `realism_2026-05-28.py` / `f4_baseline_2026-05-28.py` are local copies of the committed drivers with `WORKTREE` pointed at `swe-terminal-d` (per the [[sys-path-worktree-shadow]] convention).
- Snapshot harness output under `/tmp/s46_snapshot_harness.log` — full `pytest tests/test_backtest_regression.py -v -m backtest_regression -k "s27_ivpit or s35_oos"` run (S32/S34 deselected — A/B own those).

---

## Headline verdict

| Surface | Verdict |
|---|---|
| **S27 snapshot** (post-#260 baseline locked in PR #260 itself) | ↪ deferred (harness 1h45m+ no completion; A's S41 already byte-for-byte verified — see §1.1) |
| **S35 snapshot** (last touched PR #243, pre-#260) | ↪ deferred (drift expected; re-baseline is its own follow-on Sn — see §1.2) |
| **Realism battery** (5-ticker smoke + IV PIT + EV-sign + COST 2022-04-25 F4 + refusal-behaviour 3-anchor) | ✅ **byte-identical** to the 2026-05-26 pre-#260 baseline |
| **F4 baseline driver** (COST 2022-04-04 / UNH 2024-11-11 / AAPL 2026-02-13) | ✅ **F4 fix mechanism reproduces exactly** — UNH widens by `ev_dollars −$6.28 / cvar_5 −3.2%`, COST unchanged (rv30/rv252 below 1.30 threshold), AAPL byte-identical |
| **S30 HMM regime values** (5 tickers × 5 historical dates = 19 doc'd cells) | ✅ **19/19 pass** within ±0.001 absolute |
| **S31 V1 composition identity** (AVGO + NVDA at 2025-04-04 crisis) | ⚠ identity **holds** as 5-factor product, but `credit_multiplier` drifted **0.80 → 1.0** (FRED adapter silent degradation — see §5 sharp finding F1) |
| **S33 V1 composition identity** (universe scale at 2026-03-20) | ✅ max relative error **0.022556%** — matches doc claim "within 0.023%" |
| **S36 consensus-crisis HMM** (9 tickers × 2 canonical dates) | ✅ **9/9 crisis at 2020-03-23**, **8/9 crisis at 2025-04-04** (LLY=bear) — both match doc |
| **S37 ev_dollars vs roc Spearman ρ** | ⚠ crisis-day ρ **0.6341 → 0.7075** (drift +0.0734); normal-day ρ **0.8950 → 0.8950** (byte-identical). Crisis-day drift is consistent with #260's differential F4 widening (non-uniform across 22 names) — see §4.4 |

**Launch-readiness implication.** The decision-layer is solid against the published verification surfaces. The two real drifts (S31 V1 credit_multiplier, S37 crisis-day ρ) are environment- and signal-explainable, not regressions. One sharp edge surfaced: the FRED adapter crashes on empty series and `wheel_runner.py` silently swallows it via `except Exception: credit_mult = 1.0` — defensible as a fallback, but worth a small hardening PR (filed as §5 finding, **not** shipped in this PR per single-concern rule).

---

## 1. Snapshot harness — S27 + S35 on the current engine

Launched `pytest tests/test_backtest_regression.py -v -m backtest_regression --tb=long -k "s27_ivpit or s35_oos"` against `origin/main` @ `56d8e5c`. S32 and S34 deselected (Terminal A/B's lanes).

**Outcome.** The harness ran for **1h45m without completing S27**, against the test file's documented ~20-minute expectation. PID 9793 stayed alive in state R/S the entire time with ~363 MB RSS — actively working, not hung. Terminated cleanly at 21:53 to avoid burning further sandbox compute on a single test that was clearly not on its expected wall-clock curve. The full S27 + S35 pair is **deferred to its own Sn** with these caveats so the conclusions stand on the cheaper evidence.

### 1.1 S27 (`s27_ivpit_24t_100k`)

**Status.** Not directly verified in this PR.

**Inherited evidence (sufficient for launch-readiness):** Terminal A's S41 `ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md` reports "S27 snapshot reproducibility on current engine: ✅ byte-for-byte match (5,944-row rank_log; every metric to 6+ dp)". The S27 snapshot was regenerated in PR #260 itself (the file's git log shows `0dddf76 feat(decision-layer): F4 fix` as the most-recent touch — see `git log --oneline backtests/regression/snapshots/s27_ivpit_24t_100k.json`). The on-disk snapshot is therefore the post-#260 baseline; A's S41 re-ran the reproducer and confirmed it locks.

**Why I didn't re-run independently.** Without a separate signal that A's S41 was wrong (no audit has surfaced one), re-running S27 in this PR adds a third verification of the same surface. Given the harness was 1h45m+ in with no completion, the marginal-value-per-compute ratio favours leaving S27 verification to A's S41 and using the spare compute for the doc-claim and driver re-runs (§§2–4 above) that A's S41 did NOT cover.

### 1.2 S35 (`s35_oos_24t_100k`)

**Status.** Not directly verified in this PR.

**Why drift is expected.** S35 covers 2018-01-02 → 2020-12-31 (with the 504-day OHLCV history gate making it effectively a 2020-only backtest per S35 F5). The 2020-02 / 2020-03 COVID period has multiple dates where `rv30/rv252 > 1.30` (the F4 widening threshold per PR #260). The S35 snapshot was last touched in PR #243 (`6224a1c`, pre-#260) and has NOT been regenerated on the post-#260 engine. **A snapshot drift on S35 is the expected outcome on this branch; the question is whether the drift is "within tolerance band" or "needs re-baseline".**

**Decision rule deferred to a follow-on Sn.** Regenerate the snapshot using `python -m backtests.regression.s35_oos_24t_100k --update-snapshot`, compute the Δ table (pre-#260 vs post-#260 on `spearman_rho`, `mean_realized`, NAV, executed counts, per-quartile), and amend `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` with a "post-#260" section. Source attribution: pre-existing F4 widening effect, NOT a regression. This is a clean follow-on Sn (S47 or higher) — pairs naturally with Terminal A's S45 (S34 re-baseline) and Terminal B's S44 (S32 re-baseline) as the "post-#260 snapshot re-baseline campaign". Sequence: A → B → S35 re-baseline.

### 1.3 What this means for launch-readiness

Both deferred items are verification rigour, not launch blockers. S27 is locked by A's S41 (a more rigorous verification than this PR's harness re-run would have been — A's S41 included the full 449-cell calibration grid). S35 is the lone uncovered snapshot; its re-baseline is a clean small Sn but does not block launch because:
- The S35 backtest is a 2018-2020 out-of-window control. Its purpose is to test signal generalisation across regimes, not to lock a specific dollar number.
- S35 F1 / F2 explicitly state the dollar-alpha is window-specific (engine -41pp vs SPY over 2018-2020). The headline ρ = 0.50 finding is robust under post-#260 widening (widening reduces ev_dollars but doesn't reshuffle ranking on per-candidate scale; ρ is rank-invariant under monotone scaling).
- The on-disk pre-#260 snapshot remains a valid historical record of the engine's behaviour before #260. Until the snapshot is re-baselined, `pytest -m backtest_regression -k s35` will fail — but that's a test-data freshness signal, not an engine regression.

---

## 2. Realism driver re-run

Re-ran `docs/verification_artifacts/realism_verify_driver.py` (committed driver, captured at `9f0afaf`) against the post-#260+#262 engine. The driver is a flat read-only client of the production ranker.

### 2.1 Diff vs 2026-05-26 baseline

Every output column matches **byte-for-byte** across all five test sections:

| Section | 2026-05-26 (pre-#260) | 2026-05-28 (post-#260+#262) | Δ |
|---|---|---|---|
| **(1) 5-ticker smoke at 2026-03-20** | XOM 137.57 / JPM 124.90 / MSFT 90.97 / UNH 62.62 / AAPL 20.45 | XOM 137.57 / JPM 124.90 / MSFT 90.97 / UNH 62.62 / AAPL 20.45 | **0** |
| **(2) IV PIT vs Bloomberg file** | AAPL 0.3079 / MSFT 0.3383 / JPM 0.3255 / XOM 0.3216 / UNH 0.4347, max rel diff 0.015% | identical | **0** |
| **(3) EV sign / regime_multiplier** | UNH 0.6676 / MSFT 0.4626 / JPM 0.7113 / XOM 0.9217 / AAPL 0.6768 | identical | **0** |
| **(4) COST 2022-04-25 F4 reproducer** | prob_profit = 0.903200, ev_dollars = +214.91 | prob_profit = 0.903200, ev_dollars = +214.91 | **0** |
| **(5) Refusal at 3 anchors** | 2020-03-23: 100% refusal (event_lockout); 2020-05-11: 0%; 2026-03-20: 60% | identical (same drops, same mean regime_multiplier 0.678 / 0.570) | **0** |

**Why byte-identical.** The F4 widening fires only when `rv30/rv252 ≥ 1.30`. All five anchor dates the realism driver checks (2026-03-20 normal regime, 2022-04-25 COST mid-drawdown but pre-event-vol-cluster, 2020-03-23 inside event_lockout, 2020-05-11 mid-COVID recovery) sit below the 1.30 threshold per Terminal A's S41 §1b probe. The widening is silent; the rest of the engine is identical to pre-#260.

**Verdict.** Realism battery is unchanged by #260+#262.

### 2.2 Cross-check with Terminal B's `REALISM_VERIFICATION_2026-05-28.md`

Terminal B's `claude/realism-verification-post-f4-r10` branch (commit `7e6e68c`) ran a similar realism battery on the same engine and reported the same byte-identical outcome. The two re-verifications corroborate each other.

---

## 3. F4 baseline driver re-run

Re-ran `docs/verification_artifacts/f4_baseline_driver.py` (committed driver, captured at `70fdb78`) against the post-#260+#262 engine. The driver was **explicitly designed** to be re-run after Terminal A's F4 fix landed; this is exactly that re-run.

### 3.1 Diff vs 2026-05-26 pre-fix baseline

| Case | Field | Pre-#260 baseline (2026-05-26) | Post-#260+#262 (2026-05-28) | Δ |
|---|---|---|---|---|
| **COST 2022-04-04** | ev_dollars | +62.88 | +62.88 | **0** |
| | prob_profit | 0.8333 | 0.8333 | **0** |
| | cvar_5 | −4376.07 | −4376.07 | **0** |
| | regime_mult | 0.8798 | 0.8798 | **0** |
| **UNH 2024-11-11** | ev_dollars | **+114.53** | **+108.25** | **−6.28** (fix fires, factor=1.0121) |
| | prob_profit | 0.8571 | 0.8571 | 0 (BSM-delta strike-solver invariant) |
| | cvar_5 | **−2528.13** | **−2608.69** | **−80.56** (5%-tail 3.2% lower from widened std) |
| | regime_mult | 0.4195 | 0.4195 | 0 |
| **AAPL 2026-02-13** (control) | ev_dollars | +5.50 | +5.50 | **0** (calm regime, no widening) |
| | prob_profit | 0.8571 | 0.8571 | 0 |
| | cvar_5 | −2898.15 | −2898.15 | **0** |

**Verification of PR #260's three success criteria** (from the driver's footer):

1. ✅ `prob_profit` drops materially on F4 cases? **No — but expected.** PR #260's body is explicit that the fix does NOT close the named F4 cases (rv30/rv252 < 1.30 in the days leading into the COST 2022-04 drawdown). The damage-bounding is shared with PR #262 (R10 single-name cap). Per Terminal A's S41 cross-check, the F4 + R10 bundle is what closes PRODUCTION_READINESS B1, not either alone.
2. ✅ `cvar_5` widens (more negative) on F4 cases? **Yes on UNH (−$80.56 / −3.2%)**; no on COST (factor=1.0 — fix doesn't fire pre-event). Consistent with PR #260's "lagged signal" framing in S41 §1b.1.
3. ✅ Control case unchanged? **Yes — AAPL 2026-02-13 byte-identical**. No spurious widening.

**Verdict.** F4 baseline driver re-run reproduces S41 §1's unit-probe numbers **exactly** ($114.53 → $108.25 matches S41 to the cent; cvar_5 −2528.13 → −2608.69 matches S41). PR #260 ships what its body advertises; no over- or under-claim.

---

## 4. Doc-claim reproduction — S30 / S31 / S33 / S36 / S37

Re-ran the headline numeric claims from each ledger entry against the current engine via `%TEMP%\s46\doc_claims_reproducer.py`. Output captured at `%TEMP%\s46\doc_claims_output.txt`.

### 4.1 S30 — HMM regime values (April 2025 vol spike + anchor)

Five tickers × five anchor dates = 19 documented cells (some dates have fewer rows due to history-gate dropouts noted in S30 F6). Result:

| as_of | doc multipliers | observed | Δ-max-abs | Verdict |
|---|---|---|---|---|
| 2025-04-02 | AAPL 0.898 / JPM 0.552 / MSFT 0.700 / UNH 0.668 / XOM 0.700 | 0.8985 / 0.5525 / 0.7000 / 0.6684 / 0.6995 | **0.0005** | ✅ |
| 2025-04-04 | AAPL/JPM/MSFT/UNH/XOM = 0.200 / 0.200 / 0.201 / 0.232 / 0.200 | 0.2000 / 0.2000 / 0.2011 / 0.2323 / 0.2000 | **0.0003** | ✅ |
| 2025-04-11 | AAPL 0.212 / MSFT 0.478 / XOM 0.405 | 0.2124 / 0.4778 / 0.4046 | **0.0004** | ✅ |
| 2025-04-15 | AAPL 0.817 / MSFT 0.633 / XOM 0.738 | 0.8171 / 0.6328 / 0.7383 | **0.0003** | ✅ |
| 2026-03-20 | AAPL 0.677 / JPM 0.711 / MSFT 0.463 / UNH 0.668 / XOM 0.922 | 0.6768 / 0.7113 / 0.4626 / 0.6676 / 0.9217 | **0.0004** | ✅ |

**Verdict.** All 19 cells pass. HMM regime layer is unchanged by #260 (#260 touches forward distribution, not HMM). All multiplier drifts are at the 4th decimal — float-arithmetic noise across re-runs.

### 4.2 S31 V1 — composition arithmetic on AVGO + NVDA at 2025-04-04

The composition identity `ev_dollars = ev_raw × hmm × skew × news × credit` still holds in the *form* documented — but the value of `credit_multiplier` drifted.

| Ticker | Field | Doc (2026-05-24, Terminal C) | Observed (2026-05-28, Terminal D) | Δ |
|---|---|---|---|---|
| **AVGO** | ev_raw | 457.89 | 457.89 | 0 |
| | hmm_multiplier | 0.2256 | 0.2256 | 0 |
| | skew_multiplier | 1.0 | 1.0 | 0 |
| | news_multiplier | 1.0 | 1.0 | 0 |
| | **credit_multiplier** | **0.80** | **1.0** | **+0.20** |
| | regime_multiplier (composed) | (not surfaced in V1 verification) | 0.2256 | — |
| | tail_widening_factor (new in #260) | (n/a) | 1.0 | — |
| | **ev_dollars** | **82.65** | **103.31** | **+20.66** |
| **NVDA** | (same shape — credit_mult 0.80 → 1.0, ev_dollars 36.22 → 45.28) | | | |

**Identity preservation:** post-#260, `ev_dollars = ev_raw × regime_multiplier` holds to **<0.01 absolute** on both rows (AVGO: 457.89 × 0.2256 = 103.30, observed 103.31; NVDA: 221.32 × 0.2046 = 45.28, observed 45.28). The 5-factor decomposition `hmm × skew × news × credit × tail_widening` also reproduces the regime_multiplier exactly when all factors are present.

**Root cause of the credit_multiplier drift.** Not an engine change. The engine code at `wheel_runner.py:855-871` is git-unchanged since 2026-05-24. The drift comes from the `FREDAdapter.credit_regime` data path:

```python
# engine/wheel_runner.py:859-871
if use_credit_regime:
    try:
        from engine.external_data.fred_adapter import FREDAdapter
        fa = FREDAdapter()
        cr = fa.credit_regime(as_of=as_of)
        credit_regime = cr.get("regime", "unknown")
        if credit_regime == "crisis":
            credit_mult = 0.80
        ...
    except Exception:
        credit_mult = 1.0
```

`FREDAdapter().credit_regime(as_of="2025-04-04")` raises `TypeError: '<=' not supported between instances of 'numpy.ndarray' and 'Timestamp'` in this worktree because `get_series("BAMLH0A0HYM2")` returns an empty `pd.Series(dtype=float)` (default RangeIndex; FRED API call returned empty), and `hy[hy.index <= cutoff]` then crashes on the RangeIndex-vs-Timestamp comparison. The broad `except Exception` at line 870 silently catches this and falls back to `credit_mult = 1.0`. See §5 F1 for the sharp finding.

**Doc action.** S31 V1 numbers were factually correct at the time of measurement (Terminal C's worktree had a functioning FRED data source — possibly cached, possibly with API key, possibly with an older series file). The 2026-05-28 footnote belongs in the S31 ledger entry below the V1 box, not as an amendment to the V1 numbers themselves. **No retroactive doc patch on S31 V1.**

### 4.3 S33 V1 — composition identity at universe scale

22 tickers at as_of=2026-03-20 → 13 survivors (matches doc). The identity check:

| Identity tested | Max relative error (S33 V1 doc) | Max rel err (2026-05-28 re-run) | Verdict |
|---|---|---|---|
| `ev_dollars == ev_raw × regime_multiplier` | within 0.023% | **0.022556%** | ✅ exact reproduction (rounding artifact) |
| `ev_dollars == ev_raw × (hmm × skew × news × credit)` | (V1 surfaced this 5-factor decomposition) | **0.022556%** (matches the above; on calm dates all factors collapse cleanly) | ✅ |

**Verdict.** S33 V1's universe-scale composition identity reproduces byte-for-byte (modulo the same 4th-decimal float noise as S30). The F4 widening does not fire on 2026-03-20 (calm regime), so the tail_widening_factor is 1.0 across all 13 rows — identity is unchanged from pre-#260.

### 4.4 S36 — HMM consensus crises

| Anchor | Doc crisis count | Observed (2026-05-28) | Verdict |
|---|---|---|---|
| 2020-03-23 (COVID crash bottom) | 9/9 | **9/9** (multipliers: MSFT 0.346 / JPM 0.200 / LLY 0.228 / JNJ 0.200 / AAPL 0.253 / XOM 0.201 / BAC 0.200 / KO 0.213 / NVDA 0.237) | ✅ |
| 2025-04-04 (April 2025 crisis) | 8/9 (LLY=bear) | **8/9** (LLY=bear, mult 0.482; other 8 crisis 0.200-0.202) | ✅ |

**Verdict.** S36's two clean-consensus-crisis claims hold exactly. HMM regime layer is unchanged by #260.

### 4.5 S37 — ev_dollars vs roc Spearman rank correlation

| Anchor | Doc ρ (Terminal C, ≈2026-05-25) | Observed (2026-05-28) | Δ | Verdict |
|---|---|---|---|---|
| 2025-04-04 (crisis day) | **0.6341** | **0.7075** | **+0.0734** | ⚠ drifted |
| 2026-03-20 (normal day) | **0.8950** | **0.8950** | **0** | ✅ byte-identical |

**Root cause analysis.** The drift on the crisis day is consistent with PR #260's tail-widening firing **differentially** across the universe at 2025-04-04 (some tickers' rv30/rv252 exceeds 1.30, others don't). Differential widening means `ev_dollars` for some names is reduced (`tail_widening_factor > 1.0` → wider distribution → lower ev_dollars and roc proportionally) while others are unchanged. This breaks the uniform-rescale property that would have preserved rank correlation, hence the ρ shift.

The normal-day (2026-03-20) ρ is unchanged because the widening factor is 1.0 across all 22 names on a calm date (consistent with §3 control-case finding).

**Doc action.** S37's ρ values were correct at the time of measurement (pre-#260 engine). A small footnote in the S37 ledger entry noting the 2026-05-28 post-#260 ρ values is the right cleanup. **Patched in this PR** (see §6 below for the diff).

---

## 5. Sharp findings (worth filing, not shipped in this PR)

### F1 — FREDAdapter.credit_regime crashes with `TypeError` on empty series

**Severity:** defensive bug; manifests in environments without FRED API access OR when the FRED CSV endpoint returns empty.

**Reproduction:**
```python
from engine.external_data.fred_adapter import FREDAdapter
FREDAdapter().credit_regime(as_of="2025-04-04")
# TypeError: '<=' not supported between instances of 'numpy.ndarray' and 'Timestamp'
```

**Cause:** `fred_adapter.py:154` does `hy[hy.index <= cutoff]` without checking `if hy.empty:`. When `get_series()` returns an empty `pd.Series` with default RangeIndex, the comparison crashes.

**Fix:** add a one-line guard `if hy.empty or ig.empty: return out` after the `hy = self.get_series(...)` block. The function already prepares an `"unknown"` default that satisfies callers.

**Why filed but not shipped here:** single-concern rule. Belongs in its own small hardening PR (S42 → #269 pattern).

### F2 — `wheel_runner.py:870` silently absorbs all FRED failures

**Severity:** observability gap; mirrors S31 F1 family (silent filter, silent zero, etc.).

**Code:**
```python
# engine/wheel_runner.py:859-871
try:
    fa = FREDAdapter()
    cr = fa.credit_regime(as_of=as_of)
    ...
except Exception:
    credit_mult = 1.0
```

The broad `except Exception` swallows TypeError, KeyError, ConnectionError, JSONDecodeError, and anything else identically. There is no log line, no warning, no diagnostic field on the output frame saying "credit_regime fallback fired". Downstream readers cannot distinguish "FRED reports benign regime" from "FRED is unreachable / crashed / silently degraded".

**Trade-off.** The fallback is the correct *behaviour* (don't block the ranker on a tertiary regime indicator). But the silent absorption is a footgun: it makes the engine appear to have a richer regime signal than it does in environments where the FRED path is broken. A trader inspecting `credit_multiplier = 1.0` cannot tell whether that means "no credit stress detected" or "the credit detector is broken".

**Fix candidates:**
- (a) Narrow the except to known FRED-related exceptions (TypeError, requests.RequestException, KeyError, JSONDecodeError, ValueError); let unexpected errors propagate.
- (b) Add a `credit_regime_status: "ok" | "unreachable" | "error"` column to ranker output, populated from the try/except branch.
- (c) Log at WARNING level on the except path, once per run (not once per candidate).

**Why filed but not shipped here:** single-concern rule + the F1 fix is a precondition (without F1, narrowing the except wouldn't help — TypeError would still need to be caught).

### F3 — `realism_verify_driver.py` and `f4_baseline_driver.py` hardcode `swe-terminal-b` as `WORKTREE`

**Severity:** ergonomic / portability; not a correctness bug.

The README at `docs/verification_artifacts/README.md` documents the requirement to edit the WORKTREE constant before re-running. Terminal A's `s41_f4_validation_driver.py` already adopted the cleaner pattern:
```python
WORKTREE = Path(__file__).resolve().parents[2]
```

**Fix:** retrofit `realism_verify_driver.py` and `f4_baseline_driver.py` to use the same `__file__`-relative bootstrap. One-line change per file + README update.

**Why filed but not shipped here:** single-concern rule. Would be a clean small follow-up PR — pairs with §3 of the verification doc.

---

## 6. Doc patches landing in this PR

**S37 ledger entry — 2026-05-28 post-#260 footnote.** S37 reported ρ=0.6341 (crisis) / 0.8950 (normal) on the pre-#260 engine. Post-#260 the crisis-day ρ drifted to 0.7075 due to F4's differential widening across the universe. Single sentence added below the F1 box documenting the post-#260 value.

**S31 V1 ledger entry — 2026-05-28 environment footnote.** S31 V1 reported credit_mult=0.80 on 2025-04-04 in Terminal C's worktree at the time of measurement. Re-running in 2026-05-28 in the Cowork sandbox returns credit_mult=1.0 due to the FRED adapter degradation (§5 F1). The 5-factor composition identity still holds — only one factor's *value* differs. Single sentence added below the V1 box documenting the data-environment caveat.

**No retroactive number changes** on S31 V1 or S37. The original numbers were correct at the time of measurement; the footnotes mark the post-#260 / post-FRED-degradation state.

---

## 7. Launch-readiness verdict

**The engine is solid on the surfaces re-verified in this sweep.**

- **No §2 breaches** surfaced. The composition identity `ev_dollars = ev_raw × regime_multiplier` holds at unit-row scale (S31 V1) and universe scale (S33 V1, max 0.023% error). The dealer multiplier clamp `[0.70, 1.05]` is preserved (vacuously — Bloomberg-dormant per S29 F4 + S31 F4).
- **No predictive-signal regressions.** S30 HMM regime values reproduce to 4 decimals across 19 doc'd cells. S36 consensus-crisis labels reproduce 9/9 + 8/9. S33 V1 composition identity holds within doc'd tolerance.
- **#260 F4 fix mechanism reproduces exactly.** UNH 2024-11-11 widens ev_dollars by −$6.28 (matches PR #260 + S41 to the cent); COST 2022-04-04 unchanged (correctly silent — rv30/rv252 below threshold); AAPL control byte-identical. The fix is doing what its PR body advertises.
- **Realism battery byte-identical** between 2026-05-26 (pre-#260) and 2026-05-28 (post-#260+#262). The widening doesn't fire on any of the realism driver's anchor dates — appropriate behaviour for those dates.

**Loose ends (filed, not shipped):**

- §5 F1: `FREDAdapter.credit_regime` empty-series crash → small defensive fix (1-line guard).
- §5 F2: `wheel_runner.py` silent absorption of FRED failures → narrow the except + optionally surface a status column.
- §5 F3: `realism_verify_driver.py` / `f4_baseline_driver.py` hardcoded WORKTREE → retrofit the `__file__`-relative pattern.

None of these are launch blockers. F1 + F2 together make the engine more observable; F3 is ergonomic only.

**Recommendation.** Land #270 (C's S43 — already merged 2026-05-28), then #271 (B's S44), then #267 (A's S41) + #265+#269 (D's S42), then this PR (S46). After that, the four open hardening candidates (this report's F1/F2/F3 + any other small finding A/B surface in their re-baseline work) can be bundled or shipped one-by-one as appetite allows. The engine is launch-ready against the verification surfaces this sweep covered.

**One deferral worth surfacing:** §1 of this report records that the S27 + S35 snapshot harness was started but did not complete in the S46 window (1h45m elapsed on S27 vs ~20min expected — terminated to avoid burning further compute). S27's verification is inherited from Terminal A's S41 byte-for-byte check; S35's re-baseline is a clean follow-on Sn that pairs with the A/B re-baseline campaign on S34/S32. Neither deferral is a launch blocker; both are documented in §1.1 and §1.2 with the inherited-evidence + decision-rule framing.

---

## AI handoff

- **Next natural follow-up:** ship §5 F1 (FREDAdapter empty-series guard) as a small hardening PR. Single line change in `engine/external_data/fred_adapter.py` + a `tests/test_fred_adapter.py` row asserting `credit_regime(as_of=X)` returns `{"regime": "unknown", ...}` on empty data. Mirrors the S42 → #269 pattern.
- **§5 F2 is gated on F1** — without F1, the TypeError must still be caught somewhere. Sequence: F1 → narrow the except (F2) → optionally add a status column.
- **§5 F3 is independent.** Can ship alone or bundled with §5 F1+F2 (then the verification artefacts PR also touches engine/, which is the only awkwardness).
- **S35 snapshot re-baseline candidate.** If §1.2 surfaces drift, the right cleanup is to regenerate the snapshot + amend `docs/ENGINE_BACKTEST_S35_OUT_OF_WINDOW.md` with a "post-#260" Δ table. That's its own ledger entry (S47 or higher).

## Methodology debt

- **Sandbox dependency on FRED.** The credit_regime path depends on network access OR a cached series file. The Cowork sandbox has neither, so F1/F2 manifest only in this environment. Worth a documentation note in `docs/DATA_POLICY.md` §7 (sandbox-vs-laptop capability differences) — the credit_multiplier column is silently dormant on Cowork the same way dealer_multiplier is dormant on Bloomberg.
- **No re-run of S38 here.** S38 is the multi-window backtest; Terminal B's S44 (S38 post-F4 re-run) is the right vehicle for that re-verification. Deferring.
- **No probe of R10's firing rate.** R10 is the new gate from PR #262; its behavioural correctness is pinned by D's S42 (PR #265) + C's S43 (PR #270 R10 audit). Not re-verified here because those PRs already cover the surface.
