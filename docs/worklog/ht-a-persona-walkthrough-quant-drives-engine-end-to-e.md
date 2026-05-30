---
id: ht-a
title: Persona walkthrough — quant drives engine end-to-end
kind: verification
status: in-flight
terminal: A
pr:
decisions: []
date: 2026-05-30
headline: Bloomberg-default path is HMM-overlay-only; 316 positive-EV survivors trimmed silently by top_n; P25=P50=P75 collapses; §2 invariants observed upheld (D16 + D17 + R1/R1a + reviewer-never-upgrades).
surface:
  - docs/HEAVY_PERSONA_WALKTHROUGH.md
  - docs/verification_artifacts/persona_walkthrough_driver.py
  - docs/verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt
---

## Goal

HT-A asked for an end-to-end walkthrough of the production engine
through the eyes of a professional quant trader, at `as_of=2026-03-20`
(freshest Bloomberg date). Concretely: scan SP500 → rank →
build dossiers with the EnginePhaseReviewer → issue EV-authority
tokens → wire `WheelTracker.consume_ranker_row`, against four
realistic operator asks ("rank me 20 names", "why was X filtered",
"size this within a $250k book", "what's the downside if assigned").
Document *what surfaces well*, *where the operator is left
guessing*, *any silent filtering*, and *how the §2 EV-authority
path behaves under realistic use*.

Read-only on `engine/`. Any defect → finding, not a fix; the Major
Session triages into a fix-card next cycle.

## What we tried

1. **Read the operator surface end-to-end.**
   `engine/wheel_runner.py:725` rank_candidates_by_ev (full schema
   + drops list); `engine/wheel_runner.py:1901` select_book;
   `engine/candidate_dossier.py:130` EnginePhaseReviewer
   (R1-R10); `engine/wheel_tracker.py:346,603,1629`
   issue_ev_authority_token / consume_ranker_row /
   portfolio_context_snapshot;
   `engine/portfolio_risk_gates.py` PortfolioContext + the
   five-gate library.
2. **Wrote a single-file driver** at
   `docs/verification_artifacts/persona_walkthrough_driver.py`
   that flat-prints stdout in the same pattern as the existing
   `realism_verify_driver.py` / `f4_baseline_driver.py`. Four
   sections (Asks 1-4) + a §2-invariants negative-control battery
   + run footer.
3. **First run produced a Windows cp1252 encoding crash** at the
   §2 section: the em-dashes / `§` / single `✓` glyph survived
   the visible stdout but a Bash `>` redirect into a file
   crashed on `✓` mid-script. The first 250 lines (Asks 1-4)
   still landed because they didn't hit the `✓` glyph.
4. **Fixed the encoding by reconfiguring stdout to UTF-8 at the
   top of the driver** (`sys.stdout.reconfigure(encoding="utf-8")`)
   and also swapped the `✓` for ASCII "OK" for defence-in-depth.
   Re-ran end-to-end; 286 lines of clean stdout; the §2 section
   completes with all five negative-control checks landing.
5. **Wrote the findings doc** `docs/HEAVY_PERSONA_WALKTHROUGH.md`
   citing the captured raw output by line number for every
   quantitative claim. 9 findings (F-A1 through F-A9); 7
   SURFACE-class operator-surface gaps + 2 §2-positive
   confirmations of the EV-authority invariant.

## What worked

- **Single-file driver pattern.** Following the existing
  realism / F4 driver template made the captured raw output
  grep-friendly and the findings doc citable by line.
- **`.attrs['drops_summary']` is the engine's strongest single
  operator surface.** Every dropped candidate carries gate +
  reason; the by-gate roll-up answers "why isn't name X here"
  in one attribute lookup. The persona doc highlights this in
  §1.3 + §2.
- **D16 + D17 audit log shape is excellent.** Every
  `issue_ev_authority_token` / `_consume_ev_authority_token`
  / `_evaluate_d17_hard_blocks` decision is recorded with the
  decision pivot (post_open_sector_pct, sector_limit, nav,
  nav_source, narrative). Three sector_cap_breach rejections
  (REGN/Healthcare/27.1%, FICO/Unknown/39.7%, FIX/Unknown/48.5%)
  fired with clean narratives.
- **§2 invariants all observed upheld.** D16 leg 1, D16 leg 2,
  R1, R1a, "reviewer never upgrades on a perfect chart" — all
  five negative-control cases land as designed. The
  PerfectChartProvider helper class is a useful pattern for
  future R1/R2 audits.
- **HMM disambiguation fields (S33 F4 closer) work in the
  walkthrough.** The persona reading `hmm_regime=normal`
  alongside `hmm_realized_vol_252d_ann=53.5%` and
  `hmm_realized_return_252d_ann=+133.7%` for FIX gets the right
  picture (high-vol + high-trend ⇒ "normal" label is
  defensible). The 252d stats are the single best
  context-disambiguation surface on the diagnostic row.

## What didn't

- **`pnl_p25 == pnl_p50 == pnl_p75` for every survivor.**
  The "headline distribution spread" collapses at 35 DTE on
  empirical NOS because the non-overlapping sample size is
  too small to spread three percentiles apart. The fields are
  populated (not NaN) and rounded to 2dp — so the operator
  reads `IQR = $0.00` with no warning. Filed as Finding F-A7.
- **The 316 silently-trimmed positive-EV survivors.** The
  rank's output answers "top 20 by ev_per_day" exactly; it does
  NOT answer "how many positive-EV candidates exist today".
  503 universe − 167 drops − 20 shown = 316 unaccounted. Cited
  the gap inline rather than masking it. Filed as Finding F-A1.
- **`select_book(ranking=top_20_frame)` silently uses the
  truncated frame.** Bypasses the docstring's "whole feasible
  pool" promise that fires only when `ranking=None`. Filed as
  Finding F-A2.
- **Dossier R7-R10 unreachable on Bloomberg-default path.** R2
  (chart_context_missing) returns first because no chart
  provider is attached. The portfolio_context_snapshot the
  driver wired up was correctly built — 0 held positions
  because the seeds were rejected at the tracker layer — but
  the dossier reviewer never invokes check_var /
  check_stress_scenario / check_sector_cap / check_single_name_cap.
  Filed as Finding F-A6.
- **`sector="Unknown"` for 11 of 20 survivors makes R9 fire on
  a missing-data label.** Two of three observed R9 rejections
  fired against "Unknown" rather than a real GICS sector.
  Operationally misleading. Filed as Finding F-A3.
- **Four overlays are structurally 1.0 on Bloomberg-default**
  (news, credit, dealer, skew); the fifth (tail-widening) fired
  on only 2 of 20 names. The diagnostic columns advertise a
  five-axis regime overlay, but in practice HMM dominates.
  Filed as Finding F-A4.
- **EVT fields are dataclass defaults, not assessments**, at
  35 DTE. `cvar_99_evt=NaN`, `tail_xi=NaN`, `heavy_tail=False`
  because the EVT fit doesn't run at the sample size produced
  by empirical NOS over 504d / 35d horizon. Filed as Finding
  F-A5.

### Dead end (encoding crash)

The first driver run crashed mid-§2-section because Windows'
default cp1252 stdout encoding chokes on `✓` ("✓") when
redirected to a file. The crash left a partial file (sections
0-4 + part of §2). Initial impulse was to drop the Unicode glyph
and re-run — instead, the durable fix is the
`sys.stdout.reconfigure(encoding="utf-8")` bootstrap at the top
of the driver, so future drivers in this directory don't have
to police every glyph. The `✓ → OK` swap stayed for
defence-in-depth.

## How we fixed it

- Reconfigured stdout to UTF-8 at the top of the driver
  (`sys.stdout.reconfigure(encoding="utf-8")` guarded by
  `hasattr(sys.stdout, "reconfigure")` for older Python
  fallback). Swapped the lone `✓` glyph for ASCII "OK".
- The findings document records the encoding behaviour in the
  reproducibility section so the next agent re-running this
  driver on Windows isn't caught by the same trap.

## Evidence

- **Driver:** `docs/verification_artifacts/persona_walkthrough_driver.py`
  (713 lines, ruff-clean, deterministic, sys.path-bootstrapped).
- **Raw output:** `docs/verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt`
  (286 lines, captured on `main @ 56c671d` at `as_of=2026-03-20`).
- **Findings doc:** `docs/HEAVY_PERSONA_WALKTHROUGH.md` —
  9 findings (§6), 5 §2-invariant traces (§5), Asks 1-4 cited
  by raw-output line.

### Operator-visible numbers cited in the doc

- Universe: 503 tickers. Survivors shown: 20. Drops (per
  `.attrs['drops_summary']`): 167 = 87 ev_threshold + 68 event +
  11 history + 1 premium. Silently trimmed: 316.
- Top survivor FIX: spot $1360.79, 1212 put at $47.19, EV
  $2547.56 (ev_per_day $146.97), prob_profit 0.9714,
  prob_assignment 0.0286, regime_multiplier 0.621 (≡
  hmm_multiplier).
- FIX distribution: P25 = P50 = P75 = $4647.29, cvar_5
  = $4102.46, IQR = $0.00.
- select_book over the top-20 frame: 6 of 20 picked, total
  collateral $242,050, total EV $2960.92, utilization 96.82%,
  method `exact_knapsack`.
- Tracker strict-mode walkthrough: 3 D17 sector_cap_breach
  rejections — REGN at Healthcare 27.1%, FICO at Unknown
  39.7%, FIX at Unknown 48.5% (all 25% sector limit).
- Dossier verdict distribution: 20/20 `review` with reason
  `chart_context_missing` (R2 fires first because no chart
  provider is attached).
- §2 invariant checks: 5/5 upheld (D16 leg 1, D16 leg 2,
  R1, R1a, reviewer-never-upgrades-on-perfect-chart).

### Commands

```bash
# Self-contained re-run (UTF-8 stdout enforced from inside driver):
"/c/Users/merty/AppData/Local/Programs/Python/Python312/python.exe" \
    docs/verification_artifacts/persona_walkthrough_driver.py \
    > docs/verification_artifacts/persona_walkthrough_$(date +%Y-%m-%d)_raw_output.txt
```

Wall time on the dev box: 5-15 minutes (full SP500 scan +
dossier batch + tracker walkthrough + §2 invariants).

## Unresolved / handoff

- **9 findings sit in `docs/HEAVY_PERSONA_WALKTHROUGH.md` §6.**
  Major Session is the triage authority — read severity + the
  "suggested fix shape" line under each finding. The two §2
  findings (F-A8, F-A9) are positive confirmations, not bugs.
- **Open Bloomberg-default path question.** On this connector,
  four of five non-HMM overlays are structurally absent
  (Finding F-A4). The engine's diagnostic schema advertises
  them; the operator UI / API surface might want a
  `live_overlay_count` or similar to make the "single-axis HMM
  overlay" reality explicit. Out of scope for HT-A.
- **The 316 silently-trimmed survivors (F-A1) interact with
  F-A2.** If we want `select_book(ranking=df)` to fit against
  the full pool, the simplest fix is to widen `top_n` whenever
  a `ranking` is passed — but that breaks the existing
  "ranking-as-display" semantics. Genuine design question for
  the Major Session.
- **`pnl_p25 == pnl_p50 == pnl_p75` (F-A7) is a structural
  small-sample artefact at 35 DTE NOS.** A future fix might
  fall back to block-bootstrap percentiles in this regime, or
  flag the collapse with a `pnl_percentile_status` field.
- **HT-B / HT-C / HT-D context.** The companion heavy-verify
  cards (PIT realism vs actuals, news-severance + calibration
  re-verify, R10 strict-mode at scale) may interact with HT-A
  findings — particularly F-A4 (HT-C's "is D18 a no-op" question
  is partly answered by F-A4's news_multiplier=1.0 across all
  20 survivors here) and F-A7 (HT-B's calibration probe should
  observe the same percentile collapse). Cross-reference at
  cycle close.

[[edit-cache-rebase-trap]] — relevant to the encoding-crash
recovery: I re-ran the entire driver rather than salvaging the
first crash's partial output. Determinism plus a single-file
driver makes the re-run cheap.

[[bash-env-var-expansion-trap]] — the redirect target uses Bash
`$(date)` substitution; if running from PowerShell the user must
adjust to `$(Get-Date -Format yyyy-MM-dd)` or hard-code the date.
