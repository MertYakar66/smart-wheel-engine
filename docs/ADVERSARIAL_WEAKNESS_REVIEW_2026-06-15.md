# Adversarial Weakness Review — 2026-06-15

**Reviewer:** Terminal X (senior quant / adversarial systems review).
**Mode:** Read-only investigation. No code modified; findings only.
**Baseline:** branch `claude/suggest-rolls-defensive-surfacing`, HEAD `e351bf5`
(110 commits ahead of `origin/main`). 821 tracked files.
**Method:** nine review dimensions worked in parallel, each returning
file:line-backed findings. The three highest-severity items (D-2 stale spot,
HMM crisis label, dealer-clamp application site) were independently reproduced
against source by the lead reviewer before publication.

**Marking convention.** *Proven* = verified directly against source/data.
*Attested* = a doc/test/number claims it; not independently re-run this pass.
*Unverified* = could not determine this session (see the final section).

> This is an adversarial review: it documents how the system fails, misleads,
> or underperforms. It is deliberately not a health check. The core §2 plumbing
> is sound and is recorded as such where verified — but the bulk of this doc is
> the exposure that the §2 policing does not reach.

---

## Resolution log (post-review fixes)

The review itself is a point-in-time snapshot; fixes shipped after it are
tracked here. All on branch `claude/weakness-review-fixes` (working tree).

| Date | Finding | Resolution |
|---|---|---|
| 2026-06-15 | **D-2** live `as_of=None` stale-spot (hardening #1) | **FIXED** — `spot_date` provenance on every row + warn-once + opt-in `refuse_stale_live`; `as_of` path byte-identical (no baseline shift). Worklog `d2-live-spot-staleness`; tests `test_pit_leaks.py::TestLiveSpotStaleness`. Verified on real data: now warns "live spot is 87 days stale" + `spot_date=2026-03-20`. |
| 2026-06-15 | Dealer-clamp single-point defense (Dim-1) | **FIXED** — `DEALER_MULT_FLOOR/CEIL` + re-assert at apply site; byte-identical. |
| 2026-06-15 | `distribution_source` provenance mislabel (Dim-1 F1) | **FIXED** — row reports engine's real source, not stale cascade "none". |
| 2026-06-15 | HMM silent non-convergence (Dim-2) | **FIXED (audit flag)** — `hmm_converged` on the row; surfaced that live fits return `converged=False` at `n_iter=20`. Multiplier unchanged (relabel is a separate baseline-shifting effort). |
| 2026-06-15 | Tolerance too loose (Dim-5 F2) | **FIXED** — `rel=0.05`→`1e-9`. |
| 2026-06-15 | R11 absent from contract + nav docs (Dim-8 B1/B2) | **FIXED** — R1–R11 across CLAUDE.md §2 + 5 nav docs; R11 cases added to the named pin `test_dossier_invariant.py`. |
| 2026-06-15 | Trio coverage gate (Dim-5 F1) | **FIXED** — per-file floors in `ci.yml` (dossier 88 / ev_engine 90 / wheel_runner 54). |
| 2026-06-15 | D-3 D11 puller swallow (Dim-3) | **CORRECTED — overstated.** Per-ticker connector is a documented throughput tradeoff; contamination protection holds. Pullers not changed. |
| 2026-06-15 | `har_rv` lag-alignment suspicion (Could-not-verify) | **REFUTED — alignment correct.** Traced `forward_distribution.py:260-291`: predictors (daily/weekly/monthly RV) are all at day *t−1*, target at *t* (textbook HAR-RV, Corsi 2009); lengths match (`n−23`); trailing rolling means + PIT cutoff ⇒ no look-ahead. No off-by-one. |

### Pre-existing branch failure — diagnosed (NOT introduced by this work)

`tests/test_f4_rv_widening.py::TestF4CasesRanker::test_calm_regime_5_ticker_smoke_preserves_main_baseline`
is **RED on this branch independent of every fix here** (confirmed by stashing
all engine edits — fails identically). Root cause, precisely: the test asserts
`len(df) == 5` but calls `rank_candidates_by_ev` with **no `as_of`** (despite its
docstring "smoke **at 2026-03-20**"), so it runs at wall-clock *now*. With
`use_event_gate=True` (default) + a static earnings calendar, JPM/XOM/UNH get
**event-locked-out** when their next earnings fall inside the 35-DTE window — at
today's clock JPM's 2026-07-14 earnings do exactly that → 4 rows. **It is NOT
spot-staleness and tomorrow's data pull will NOT fix it** (at a pinned
`as_of=2026-03-20` only 2 rows survive — JPM/XOM/UNH lock out on their April
earnings). The test is wall-clock-non-deterministic; the proper fix (owner's
call) is `use_event_gate=False` + a pinned `as_of`, or relax the `==5`
assertion to its real intent (widening factor == 1.0 on whatever rows survive).
Left unchanged — it pins F4 behavior I don't own. *(Corrects my earlier
"downstream of stale data, may self-resolve" note, which was wrong.)*

---

## Executive summary

1. **No validated dollar-alpha over a passive benchmark, and the headline ρ is
   misread.** Spearman ρ ∈ [0.19, 0.55] measures *bomb-detection across the full
   EV range* (it includes the 30–45 % of rows with EV ≤ 0). Within the traded
   EV>0 region, `ev_dollars` has ≈0 rank-correlation with realized P&L (I1). The
   engine loses to passive by **−41pp to −104pp** in every multi-year window
   tested; 64–83 % of even its positive NAV is incidental equity beta on
   assignments, not put-selection skill (S43 §4). Honestly documented — but the
   ρ line invites overestimation.

2. **A live `as_of=None` run today silently prices off an 87-day-stale spot
   (CRITICAL, proven).** The 30-day staleness gate is wrapped in
   `if as_of is not None:` (`wheel_runner.py:976`); `spot` is then set from the
   last close unconditionally (`:1015`). All trade-relevant Bloomberg series end
   2026-03-20; today is 2026-06-15. The exact "no silent spot substitution"
   violation the gate was built to prevent, reachable on the default provider
   with no warning.

3. **The HMM "crisis" label is assigned by sorted position, decoupled from the
   fitted regime (HIGH, proven).** `regime_hmm.py:349-350` always labels the
   lowest-scoring of 4 states "crisis" (→ 0.2 multiplier), regardless of whether
   the data contains a crisis. Mechanism behind the documented "crisis fired on
   98 % of 2022-2024 dates." Live on the put ranker.

4. **`prob_profit` is over-confident exactly where it matters
   (HIGH, attested).** When the engine says 96 % profitable, realized is ~70 %
   (−27pp); structural across 10/10 backtest configs, worst in crises. The
   POT-GPD tail machinery that could fix it exists in `tail_risk.py` but is **not
   wired into the `prob_profit` path.**

5. **The committed CI regression harness validates synthetic-premium ranking
   stability, not realistic P&L (HIGH, proven/attested).** Premiums are
   BSM-synthetic off zero-skew Bloomberg IV; realized P&L is held-to-expiry
   intrinsic; friction is a flat 8 % spread that moves a 5-year NAV ~15bp. Dollar
   outcomes are **harness-dependent**: identical inputs gave 305 vs 516 executed
   puts and a ±$103k NAV swing across harness versions (S43 §9). No dollar
   backtest number is reproducible without pinning the exact harness.

6. **A live §2 reviewer rule (R11) is absent from the structural contract and
   every nav doc (HIGH, proven).** R11 fires in `candidate_dossier.py:535-561`
   (shipped #306/D23) but `CLAUDE.md:32`, `docs/REPO_MAP.md`, `PROJECT_STATE.md`,
   `MODULE_INDEX.md`, `README.md`, `TESTING.md` all still say "R1–R10." The
   invariant test those docs name as the reviewer pin (`test_dossier_invariant.py`)
   does **not** exercise R11.

7. **Survivorship bias is structural in the live universe path (HIGH, proven).**
   `get_universe()` (`data_connector.py:703`) unions current-survivor panels;
   every delisted name probed (SIVB, FRC, SBNY, ATVI, PXD…) is absent. A PIT
   membership table (`sp500_index_membership.csv`) and a survivorship-audit
   function exist but the production `WheelRunner` path uses neither.

8. **The invariants the project polices most heavily actually hold.** No
   tradeable path bypasses `EVEngine.evaluate`; R1–R11 are strictly
   downgrade-only (AST-introspection tripwire tests); the dealer multiplier is
   sign-preserving, `ev_dollars`-only, clamped to [0.70, 1.05]. The
   decision-layer *test suite* is genuinely strong. The weaknesses are in
   **calibration, data freshness, validation honesty, and doc drift — not the §2
   plumbing.**

---

## The central validity question

**Is there valid out-of-sample evidence that high-EV candidates outperform a
simpler benchmark? Partially — and only for the narrow claim the project now
actually makes.**

- **YES, valid OOS: top-K-by-EV beats random draw from the same pool.** I6-B
  (n=730, 73 monthly cohorts, real Bloomberg outcomes): monthly top-10 returns
  +$166 to +$206 mean vs −$24/−$26 for random/all-population. Mechanism is *tail
  avoidance*, not fine ordering. The benchmark beaten is "random from the pool,"
  **not a passive index.**
- **NO, not validated: dollar-alpha over a passive index.** S35 −41pp vs SPY;
  S43 −51 to −104pp vs equal-weight universe across 4 rolling windows, never
  beating it. The earlier "+27pp" is both $100k- and 2022-2024-window-specific.
- **NO: `ev_dollars` does not predict realized dollar P&L** (Spearman ≈ −0.002
  within the traded region; I1). It is a tail-aware ranking score, not a
  forecast — which PROJECT_STATE now states correctly.

**Contamination weakening even the positive result:** HMM/POT-GPD/F4 parameters
were tuned "with full 2018-2026 visibility" (S35 Caveat 2), so the "OOS" windows
are not truly OOS in the calibration layer. F4 RV-widening was selected on the
same S27 ρ it was then validated against. The one properly-held-out result
(prob_profit recalibration, train 2020-23 / test 2024-26) **fails
leave-one-crisis-out** (I9): the regime where it matters.

**Headline:** a validated *ranker and crisis-refusal mechanism*, not a validated
*alpha generator*. The framing in PROJECT_STATE is honest; the risk is a reader
who stops at the ρ line.

---

## Findings by dimension

Severity: **critical** (wrong outputs, real-money risk) · **high** (fragile,
degraded, or unverified) · **medium** (meaningful, not immediate) · **low**
(hygiene).

### Dimension 1 — Decision layer correctness

Hard invariants hold: every tradeable path routes through `EVEngine.evaluate`;
R1–R11 are strictly downgrade-only; the dealer multiplier cannot flip a sign. No
critical break. Residuals:

| Sev | Location | Finding | Status |
|---|---|---|---|
| medium | `ev_engine.py:477-485` | EV omits expected exit-leg cost (`exit_commission+exit_slippage` computed but never subtracted). Every EV biased ~$1–4/contract high; flips marginal trades to "proceed." Deferred (D19). | proven, prior/open |
| medium | `forward_distribution.py:334-340` | Calendar-vs-trading-day horizon mismatch inflates the forward distribution ~46 %; affects every `prob_profit`/`prob_assignment`/`cvar_5`/`ev_dollars`. Deferred (D21). | proven, prior/open |
| medium | `wheel_runner.py:1701` vs `ev_engine.py:606-612` | `distribution_source` provenance mislabel — IV-lognormal fallback path is written to the row/token as `"none"`, not `"lognormal_fallback"`. Audit trail records a provenance that doesn't match the computation. Rare (full cascade failure). | proven, new |
| low | `ev_engine.py:523-524` | Dealer clamp is single-point-defended — no `np.clip(…,0.70,1.05)` re-assert at the application site. Holds today only by construction; a future regime branch >1.05 silently breaches §2. | proven, new |
| low | `ev_engine.py:621` | Public `evaluate` has no upper IV clamp (ranker caps at 5; `evaluate` only does `max(iv,1e-4)`). Caller passing `iv=50` gets garbage-but-finite EV, no exception. No in-repo caller does this. | proven, new |

### Dimension 2 — Quant model weaknesses

| Sev | Location | Finding | Status |
|---|---|---|---|
| high | `regime_hmm.py:349-350` | HMM crisis label decoupled from fit (sorted-position labels). Lowest of 4 states always "crisis" → 0.2 multiplier in calm tape. Live on put ranker. | proven, new |
| high | `ev_engine.py:448` + `tail_risk.py:85` | GPD small-exceedance dead zone — EVT runs only at n≥200 (~10 exceedances at 95th pct), but `min_exceedances=15` returns non-converged below 15. EVT silently no-ops in the 200–300-scenario regime it was added to rescue; `cvar_99_evt` stays NaN. The 200-gate and 15-floor are mutually inconsistent. | proven, new |
| high | `ev_engine.py:448-454` | D21 ~46 % horizon over-dispersion flows into `fit_gpd_tail(-pnls)`, contaminating `cvar_99_evt`/`tail_xi`/`heavy_tail` (which gates position down-sizing). Not just a central-tendency bias. | proven, new |
| medium | `regime_hmm.py:167-207`, `wheel_runner.py:1348-1351` | HMM silent non-convergence — `converged=False` after `n_iter=20` is computed then discarded; a garbage posterior drives a real multiplier with no audit flag. Only an exception yields neutral 1.0. | proven, new |
| medium | `tail_risk.select_threshold` | POT threshold hardcoded at 95th pct, never varied; zero sensitivity analysis. GPD ξ is notoriously threshold-sensitive. Model card calls it "data-driven"; it is a fixed quantile. | proven, new |
| medium | `tail_risk.py:219-220` | ξ≥1 CVaR sentinel `var*3.0` — arbitrary uncited finite multiplier on an infinite-mean tail; under-states CVaR by an unknown amount with no distinguishing flag. | proven, new |
| medium | `tail_risk.py:204-206` | GPD shallow-confidence fallback returns `VaR=u, CVaR=u+beta`, independent of α, for `confidence≤0.95` — a quietly meaningless number. Live default (0.99) avoids it; footgun. | proven, new |
| medium | `dealer_positioning.py:103,263,470-471` | Dealer "always long-calls-short-puts" hardcoded for every ticker/date. Names where dealers are short calls get a sign-flipped GEX/regime → multiplier moves the wrong way. Magnitude clamped; direction can be wrong. | proven, new |
| medium | `option_pricer.py:531,546,515` | Volga/Ultima silently carry the vega ÷100 scaling (100× smaller than their docstring formula); Vanna is *not* ÷100 while Vega is. Latent (no live consumer reads volga/ultima/speed/color; vanna is observability-only). | proven, latent |
| low | `forward_distribution.py:203` | Block bootstrap wraps circularly (`% n_rets`) — can stitch a 2020-recovery return onto a 2024 day inside one "autocorrelation-preserving" block. Defensible (standard stationary bootstrap wraps) but partially violates the structure the docstring sells. | proven, low |

**Verified clean (Dim 2):** forward-distribution look-ahead — all four samplers
filter `df.index <= cutoff` before any return math; PIT clean. BSM Greek closed
forms match Hull term-by-term. Theta (per-year) and vega (per-vol-point) unit
contracts consistent at consumer call sites. Numerical guards (S≤0, T≤0, σ floor,
IV-solver clamp) block reachable log(0)/div-0/sqrt(neg).

### Dimension 3 — Data quality and integrity

| Sev | Location | Finding | Status |
|---|---|---|---|
| critical | `wheel_runner.py:976,1015` | **D-2** stale-spot on live `as_of=None` — staleness gate skipped, spot = 87-day-stale last close, no warning. | proven, new |
| high | `data_connector.py:703` | **D-1** survivorship bias structural in `get_universe()`; PIT membership table + audit fn exist but production `WheelRunner` uses neither. 504-day gate bounds one direction only (rejects newcomers; never adds back the dead). | proven, confirms prior |
| medium | `data_connector.py:639` | **D-6** `get_fundamentals` is a dateless snapshot consumed in historical `as_of` contexts. IV PIT leak was fixed; dividend_yield (BSM strike solve), GICS sector (R9), beta still leak today's snapshot into backtests. Historical PIT source exists, not wired in. No regression guard. | proven leak; magnitude attested |
| medium | `pull_theta_option_history.py:419-425` + 4 siblings | **D-3** D11 PerEndpointFailure contract silently defeated in 5 of 8 pullers (`except Exception: debug; continue`, `get_failures()` never drained). A failure on a healthy Terminal is indistinguishable from a true empty. | proven, new |
| low | `data/bloomberg/` | **D-4** two tracked CSVs are empty stubs (`sp500_corporate_actions.csv` 0 bytes, `sp500_iv_history.csv` 0 rows). Empty corp-actions ⇒ split/spin adjustments have no source in the bloomberg path. | proven |
| low | `treasury_yields.csv` | **D-5** cross-series staleness mismatch — treasury ends 2026-05-05 (41d) vs equity/vol 2026-03-20 (87d); inconsistent PIT layers in a live run. | proven |

**Verified sound (Dim 3):** ranker IV *is* PIT-correct (`_resolve_pit_atm_iv`,
end_date-filtered) — the S23-F3 fix is real and load-bearing. OHLCV PIT cutoff
applied. Live ranker IV comes from `vol_iv_full.csv` (503/503), not the dormant
28-ticker iv_surface — no silent flat-IV fabrication reaches EV. Treasury
%→decimal sound (D20).

### Dimension 4 — Backtest and validation

See *The central validity question* above. Additional:

| Sev | Location | Finding | Status |
|---|---|---|---|
| low (critical if misused) | `src/backtest/wheel_backtest.py` | Non-§2, fabricated-premium, zero-cost heuristic backtester. Header says so, but it is importable and `run_backtest()` yields a clean-looking curve that is pure fiction. Must never be cited as engine performance. | proven |
| medium | `ml/wheel_model.py` | ML entry model OOS integrity **unverified** — `src/backtest` header flags a "purging-gap"; whether a genuine purged+embargoed holdout exists was not confirmed. | attested/open |

### Dimension 5 — Test suite weaknesses

| Sev | Location | Finding | Status |
|---|---|---|---|
| high | `ci.yml`, `pyproject.toml` | Coverage gate diluted ~12× — single aggregate `--cov-fail-under=80` over ~63k LOC; the decision-layer trio (~4,953 LOC, ~8 %) has no per-module floor and could fall to ~40 % without tripping the gate. | proven |
| low | `test_audit_invariants.py:305-312` | `test_regime_multiplier_scales_ev` tolerance `rel=0.05` is ~1000× too loose — the two trades share a deterministic seed so the relationship is exact to float; a 5 % non-linearity regression would pass. | proven |
| low | `test_dossier_r9_r10_audit.py:1008` | Degenerate candidate (strike=0) reaches `proceed`; pinned as "correct." §2-safe, but the reviewer has no degenerate-input→downgrade rule. | proven |
| info | `test_f4_tail_risk_gap.py:273-292` | F4 xfail (`strict=False`) self-disarms on fix, but asserts a real behavioral value (not a signature) so it does not false-green on a no-op stub. Confirm it tightens to `strict=True` when F4 lands. | proven |

**Verified clean / stronger than expected (Dim 5):** the three zero-collection
print-script "test" files are **gone** (deleted in real commits). Decision-layer
invariant tests use real objects with near-zero patching.
`test_dossier_downgrade_property.py` is a genuine AST-introspection anti-vacuity
tripwire (a future R12 without severity protection trips CI for free). Token-gate
fire-time recheck and dealer clamp boundaries are pinned at EVResult level. No
tests weakened/deleted improperly in recent history.

### Dimension 6 — Architecture and integration

| Sev | Location | Finding | Status |
|---|---|---|---|
| high | `financial_news/models.py` + `schema.py` | v1/v2 split is **live**, not just duplicate defs — `Article`/`Story`/`Category` consumed by disjoint module sets; a v1 object in a v2 processor won't share contract → silent attribute/isinstance mismatch. Off EV path. | proven, confirms prior |
| high | `MODULE_INDEX.md:129-152` | Falsely claims `engine/__init__.py` doesn't re-export the decision layer; source (`__init__.py:21-23`) does. Code fine (ROADMAP A3); the authoritative map is wrong → agent could redo shipped work. | proven, new (doc) |
| medium | `strangle_timing.py:31`, `tv_signals.py:48`, `engine_api.py:1362` | `src/` is "deprecated" (D2) but not removable — 3 live EV-path imports of `src.features.technical` + `data/quality.py` (chain-quality gate). Removing `src/` today breaks the timing strategy and the gate. | proven |
| low/medium | `engine_api.py:1013-1014` | Silent EV-ranker swallow in the committee handler (`except Exception: ev_row=None` → synthetic-BSM trade). Non-authoritative (no §2 breach) but masks engine failure; inconsistent with the tradeable handler (`:664-667`) which re-raises HTTP 500. | proven, new |
| low | `mcp_client.py:324-352` | `_classify` error strings still `TODO(live-verify)` — substring matches never confirmed against a live server. Downgrade-only; loss of audit precision, not a §2 breach. | confirmed open |

**Verified clean/resolved (Dim 6):** §2 holds across every `engine_api.py` route;
heuristic endpoints labeled non-authoritative; advisor `filter_approved` is
downgrade-only with zero live EV-path callers; news ingest severed.
`backtests/__init__.py` ImportError **resolved** (`WheelBacktester` names match).
Connector empty/None defaults are dropped by the ranker, not fabricated into
candidates.

### Dimension 7 — Multi-agent operational

| Sev | Location | Finding | Status |
|---|---|---|---|
| high | `PROJECT_STATE.md` | 135 commits / 15 days stale ("Last updated 2026-05-31"). The whole D18→D23 arc, S47 work, and heavy-verify-2026-06-09 are absent. Still pins durable structural facts (R1-R10 authority table, "audit-viii / ~2,500 tests") that have changed. | proven |
| medium | `scripts/check_lane_claim.py` | D-number/Sn allocation "at merge" is convention, enforced nowhere — the gate fires only on trio file edits and checks for a PR-body block; explicitly does not verify mutual exclusion vs other open PRs. DECISIONS/PROJECT_STATE/MODULE_INDEX/FILE_MANIFEST are shared mutable state with only convention-level collision control. | proven |
| medium | working tree | Shared-working-tree risk is live now — the primary clone is on a feature branch (110 ahead) with modified tracked files + 12 untracked paths, violating its own reserved-clean-clone rule (PARALLEL_SESSIONS §7). | proven |
| low | `PARALLEL_SESSIONS.md` | Conventions documented but unenforced — worklog-per-task, "edit only owns / read reads," "fetch+rebase before push" are advisory with no gate. | attested |

### Dimension 8 — Documentation as a failure surface

| Sev | Location | Finding | Status |
|---|---|---|---|
| high | `CLAUDE.md:32` + REPO_MAP + PROJECT_STATE + MODULE_INDEX + README + TESTING | **B1/B2** R11 absent from §2 contract and every nav doc; `test_dossier_invariant.py` (named pin) doesn't cover R11 (grep `R11|elevated_vol_top_bin|vix_level` → 0 matches). R11 is pinned only in unlinked `test_r11_elevated_vol.py`. Recurrence of `canonical-doc-rcount-drift` one rule later — the "canonical count: R1-R10" label is now itself wrong. | proven |
| medium | `docs/REPO_MAP.md:18` | Routes DECISIONS as "D1…D21"; actual max is D23 — omits D22 and the very D23 that introduced R11. | proven |
| medium | `docs/SESSION_HANDOFF.md` | Self-declared superseded 2026-05-18 snapshot still in the doc tree (claims "1734 tests," "D1-D11"). Banner-flagged, but a fresh agent reads stale state first. Should be archived. | proven |
| low | `MODULE_INDEX.md` | `engine/__init__` note reads like an open TODO for already-shipped work (same root as the Dim-6 doc finding). | attested |

### Dimension 9 — Repo structure and navigability

| Sev | Location | Finding | Status |
|---|---|---|---|
| medium | `FILE_MANIFEST.md` | Documents untracked files (`studies/premium_correction/pilot.py`, `tests/test_premium_correction_pilot.py`) → `check_manifest_coverage.py` mismatch; the manifest is ahead of the index it mirrors. | proven |
| medium | repo root / `docs/` | Untracked clutter a fresh agent reads on orientation — `HEAVY_VERIFY_FINDINGS_2026-06-09.md` and `_scan_tests.txt` (a 117-line grep dump) are **not** gitignored; five untracked `docs/*.md` sit alongside canonical docs with no draft signal. | proven |
| low | `docs/REPO_MAP.md:71` | Pins "111 flat test_*.py"; actual is 114 — in a doc that elsewhere preaches not to pin counts. | proven |

**C4 — can coverage gaps be read from suite structure alone? No.** The R11 gap
is invisible from navigation — the docs name `test_dossier_invariant` as the
reviewer pin; only reading file contents reveals R11 lives in a separate unlinked
file. A flat 114-file `tests/` dir with no per-rule index means rule→test mapping
requires grep, not navigation.

---

## What I could not verify

- **D-2 blast radius** — proved the gate blind spot and unconditional stale-spot
  assignment; did not execute the live cockpit/engine_api path to confirm no
  other layer injects a fresh spot first (default bloomberg connector has no
  live-spot override).
- **ML entry model holdout** (`ml/wheel_model.py`) — not read this pass; the
  "purging-gap" is attested by the `src/backtest` header. **Open.**
- **Whether R11 VIX>25, MIN_PROCEED_EV=10, R9=25 %, R10=10 % were validated
  OOS** — no doc found tuning them on held-out data; appear to be round-number
  defaults.
- **Every dollar figure in S35/S43/S38/I2** — attested from docs, not re-run.
  The ρ values are CI-locked; the dollar NAVs are doc-only and harness-dependent.
- **HMM EM-local-optimum ranking impact / current crisis-label firing rate** —
  mechanism verified unchanged; effect on candidate ordering needs a two-date
  backtest not run here.
- **`har_rv` regressor lag alignment** (`forward_distribution.py:266-274`) —
  traced as correct, not numerically verified; a one-row off-by-one would bias
  the vol forecast. Worth a unit-test probe.
- **Live CI behavior** of `check_manifest_coverage.py` / `check_lane_claim.py` on
  this branch — inferred from documented contract + proven tracked/untracked
  mismatch, not an executed run.
- **Whether any live provider currently feeds a non-None `market_structure`** —
  if not, the dealer module (and its F1-adjacent concerns) may be dormant on the
  default bloomberg path.

---

## Risk map

| Component | Assessment | Single biggest risk | Evidence basis |
|---|---|---|---|
| `EVEngine.evaluate` core / §2 plumbing | Trustworthy | Exit-cost & D21 biases nudge marginal trades to "take" | Routing/downgrade/clamp verified (proven) |
| Forward distribution / PIT | Fragile | D21 ~46 % over-dispersion contaminates EV + tail; look-ahead clean | `forward_distribution.py:334` (proven) |
| Tail risk (POT-GPD) | Fragile | 200-gate vs 15-exceedance dead zone; not wired into prob_profit | `ev_engine.py:448`, `tail_risk.py:85` (proven) |
| Regime HMM | Fragile | Crisis label decoupled from fit; silent non-convergence | `regime_hmm.py:349` (proven) |
| Dealer positioning | Fragile | Hardcoded one-sided assumption can invert sign | `dealer_positioning.py:470` (proven); live-feed status unverified |
| Copula | Unverified (dormant) | df=5 hardcoded, correlation never estimated; safe only because no live caller | `portfolio_copula.py` (proven dormant) |
| Data freshness / spot | Fragile | 87-day-stale spot on live `as_of=None` | `wheel_runner.py:976,1015` (proven) |
| Universe / survivorship | Fragile | Current-membership backtests inflate results | `data_connector.py:703` (proven) |
| Validation / backtests | Unverified (for $-alpha) | Synthetic premiums + harness-dependent dollars + in-sample tuning | I1/I6/S43 (attested), `_common.py` (proven) |
| prob_profit calibration | Fragile | −27pp top-bin over-confidence, worst in crises | 10/10 configs (attested) |
| Decision-layer tests | Trustworthy | Aggregate coverage gate can't protect the trio's % | `ci.yml`, real-object tests (proven) |
| `engine_api.py` routing | Trustworthy | Committee handler masks EV failures as synthetic | All routes enumerated (proven) |
| news_pipeline / financial_news | Fragile (off EV path) | Live v1/v2 type split | Disjoint import sets (proven) |
| Docs / nav contract | Fragile | R11 missing from §2 contract + nav docs + named pin | Grep across docs (proven) |
| Multi-agent coordination | Fragile | Allocation is convention; primary tree dirty now | PARALLEL_SESSIONS + git status (proven) |

---

## Hardening priorities

Ranked by leverage. Each names the specific weakness it addresses and why it
outranks what follows. **Not implemented — documentation only.**

1. **Refuse or loudly flag a stale spot when `as_of is None`** (D-2). Extend the
   staleness gate to fire against `now()` when `as_of` is None, or refuse when
   `now − last_close > tolerance`. The only *live, silent, operator-facing* path
   to a wrong-priced trade today.
2. **Wire POT-GPD tail extension into `prob_profit`; fix the 200-vs-15
   small-sample inconsistency in the same pass** (Dim 2). The engine's
   highest-conviction trades are its most over-confident; the fix machinery
   already exists in `tail_risk.py`.
3. **Fix or quarantine the HMM crisis label** (`regime_hmm.py:349`). Map labels
   to fitted regime statistics, not sorted position; check `converged` before
   use. Live on the put ranker, systematically mis-shrinks EV in calm tape.
4. **Wire `WheelRunner` to the existing PIT membership table (D-1) and add
   `as_of` to `get_fundamentals` (D-6).** Both PIT sources exist on disk; the
   production path doesn't use them. Corrupts validation rather than live trades,
   hence below #1–#3.
5. **Pin the backtest harness and add real-fill friction to the committed
   regression suite.** CI-locked numbers validate synthetic-premium ranking;
   dollar outcomes swing ±$103k on identical inputs. Promote the I2 real-fill
   methodology in or mark every dollar figure harness-dependent.
6. **Propagate R11/D23 through CLAUDE.md §2, REPO_MAP, PROJECT_STATE,
   MODULE_INDEX, README, TESTING, and add an R11 case to the named invariant
   test** (B1/B2). One mechanical sweep closes the highest-leverage doc finding.
7. **Make the D11 PerEndpointFailure contract uniform across all 8 Theta
   pullers** (D-3). `PerEndpointFailure`-aware branch + `get_failures()` sidecar
   drain, so a lossy pull is distinguishable from a clean one.
8. **Add a second, gated coverage run scoped to the decision-layer trio**
   (`--cov=engine/ev_engine --cov=engine/wheel_runner --cov=engine/candidate_dossier
   --cov-fail-under=90`). The aggregate gate cannot protect the 8 % of LOC that
   matters most.
9. **Fix the FILE_MANIFEST/git mismatch and degitignore-or-remove root clutter**
   (C1/C2). Cheap; the manifest mismatch will red the build.
10. **Make the dealer one-sided assumption data-driven or explicitly bound its
    directional error** (`dealer_positioning.py:470`). Magnitude is clamped and
    the module may be dormant on the default provider, but the *direction* can
    still be wrong — flag before any provider feeds a live `market_structure`.

---

*Net: the system polices its core invariant rigorously and successfully — those
held under direct verification. The real exposure is everywhere the policing does
not reach: a silently stale live spot, a structurally mislabeled regime, an
over-confident probability with the fix un-wired, survivorship and PIT leaks in
the data path, a validation story whose dollar numbers are synthetic and
non-reproducible, and a structural contract that no longer lists one of its own
live rules.*
