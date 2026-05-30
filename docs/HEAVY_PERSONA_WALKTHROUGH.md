# Heavy persona walkthrough — a professional quant uses the Smart Wheel Engine end-to-end

**Card:** HT-A (heavy-verify cycle, 2026-05-30) ·
**Branch:** `claude/heavy-persona-walkthrough` ·
**Engine SHA:** `main @ 56c671d` (post-#249/#260/#262/#287/#288) ·
**As-of date:** `2026-03-20` (freshest Bloomberg date per SessionStart hook) ·
**Driver:** [`verification_artifacts/persona_walkthrough_driver.py`](verification_artifacts/persona_walkthrough_driver.py) ·
**Raw output:** [`verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt`](verification_artifacts/persona_walkthrough_2026-05-30_raw_output.txt)

> **What this document is.** A read-only walkthrough of the engine through
> the eyes of a professional quant trader. The driver scripts four
> realistic operator asks against the production engine, captures stdout
> verbatim, and this doc summarises *what surfaces well*, *where the
> operator is left guessing*, *any silent filtering*, and *how the §2
> EV-authority path behaves under realistic use*. **No engine code was
> modified.** Anything that looks like a bug is logged as a *Finding*
> for the Major Session to triage into a fix-card next cycle — see
> §6.

---

## 0. Scope & method

The driver runs end-to-end on the Bloomberg-CSV path (the default
`MarketDataConnector` per `CLAUDE.md` §4) at `as_of=2026-03-20`:

1. `WheelRunner.rank_candidates_by_ev` over the full SP500 universe
   (35 DTE, 25-delta, 1 contract, top 20, positive-EV only,
   `include_diagnostic_fields=True`).
2. `WheelRunner.select_book` to fit a $250k account under a 25%/name
   concentration cap (§2-safe knapsack post-processor — never calls
   `EVEngine`).
3. `WheelTracker` constructed strict (`require_ev_authority=True`,
   connector wired, `min_nav_for_trading=1000`) and seeded with two
   held puts so `PortfolioContext` has real positions for R7-R10 to
   score against.
4. `build_dossiers` over the top 20 with a deliberately-null chart
   provider and the live `PortfolioContext` attached, so R2 vs
   R5/R7-R10 outcomes are observable.
5. `consume_ranker_row` for the top survivor (the canonical §2 wire).
6. Anchor candidate downside drilldown — distribution percentiles,
   CVaR_5, EVT tail diagnostics, regime + dealer + skew + credit +
   news multipliers, breakeven, ROC.
7. Negative-control battery: D16 leg 1 refuses non-positive EV at
   issuance; D16 leg 2 rejects stale EV at consume; dossier R1
   blocks negative EV; dossier R1a blocks non-finite EV; a *perfect*
   chart attached to a negative-EV row still produces
   `verdict='blocked'`.

The persona ("Q") is a senior quant trader who has used the
engine before but who today is checking *what the engine actually
shows me* before placing real trades.

> **Read this doc together with the raw output.** Every quantitative
> claim in §1-§4 cites a line you can find in
> `persona_walkthrough_2026-05-30_raw_output.txt`. If a number is not
> in the raw output, it is not in this doc. The driver does not pick
> the candidates: the engine does.

---

## 1. Ask 1 — "rank me 20 names"

_To be filled in from the raw output: row counts, universe coverage,
which sectors dominate top-20, which fields the persona finds
immediately useful, which feel buried._

### 1.1 What the engine surfaces well

_Concrete columns the persona reads in seconds:_

### 1.2 Where the operator is left guessing

_Columns where the persona can't tell the story without diving into
source or the docs:_

### 1.3 Silent filtering — drops_summary census

_The`.attrs['drops_summary']` payload — per-gate counts, samples per
gate, share of universe gated out, which gate would surprise the
persona most:_

---

## 2. Ask 2 — "why was X filtered"

_The trader picks a few names they expected to see — does the engine
explain itself?_

### 2.1 Drops are visible *only* on the universe-wide rank's `.attrs`

### 2.2 Per-ticker re-rank as the explainer fallback

### 2.3 The operator-typo case

_What happens when the trader asks about a name not in the universe._

---

## 3. Ask 3 — "size this within a $250k book"

### 3.1 `select_book` — the §2-safe knapsack

_Account size, concentration cap, names chosen, total collateral, total
EV, utilization. What the persona can read off `book.attrs`._

### 3.2 Tracker in strict mode — opening seeds + the top survivor

_Seeded held puts, NAV mark-to-market, what the audit log shows._

### 3.3 Dossier verdicts on the top 20 with `PortfolioContext` attached

_Verdict distribution (proceed / review / skip / blocked) +
verdict_reason histogram. R2 (chart-missing) vs R5
(ev_below_proceed_threshold) vs R7-R10 (portfolio gates) — which
fires most often._

### 3.4 The full §2 path — `consume_ranker_row` and the audit log

_Token issued / consumed lines, current_ev_dollars threading, the
full reject taxonomy if any rejects fire._

---

## 4. Ask 4 — "what's the downside if I get assigned"

_Anchor candidate's distribution percentiles, EVT diagnostics, regime
+ dealer context, capital efficiency cross-checks._

### 4.1 Distribution shape — P25 / P50 / P75 / CVaR_5

### 4.2 Heavy-tail diagnostics — EVT cvar_99, tail_xi, heavy_tail flag

### 4.3 Regime + dealer overlays — what scaled the final EV

### 4.4 Assignment economics — basis, breakeven, ROC cross-check

---

## 5. §2 invariants observed in production code paths

A standalone trace of CLAUDE.md §2 guarantees, run against the engine
on the same as_of:

### 5.1 D16 leg 1 — `issue_ev_authority_token` refuses `ev_dollars ≤ 0`

### 5.2 D16 leg 2 — `open_short_put` consume rejects stale EV

### 5.3 Dossier R1 — negative EV → `verdict='blocked'`, reason=`negative_ev`

### 5.4 Dossier R1a — non-finite EV → `verdict='blocked'`, reason=`ev_non_finite`

### 5.5 Reviewer never upgrades — a perfect chart on a negative-EV row still produces `'blocked'`

---

## 6. Findings

> **Severity scale** (loose):
> **SURFACE** = the engine has the data but the operator can't see it
> without diving into docs/source. **GAP** = the operator can't make
> a defensible decision with what the engine surfaces today. **§2**
> = an invariant CLAUDE.md §2 names is observable as upheld
> (positive finding) or violable (would be a hard bug — none expected
> in this read-only walkthrough, but logged distinctly so the
> Major Session can audit).

_Each finding has Severity, What, Where, Why-it-matters,
Suggested-fix-shape (NOT a fix). All fixes are for the Major Session
to triage into a fix-card next cycle._

| # | Severity | Title | Pointer |
|---|---|---|---|
| F-A1 | _to fill_ | _to fill_ | _to fill_ |

---

## 7. Reproducibility

```bash
# From any worktree (the driver bootstraps sys.path to this worktree's
# absolute path — edit the WORKTREE constant at the top of the driver
# if running outside swe-terminal-a):
"/c/Users/merty/AppData/Local/Programs/Python/Python312/python.exe" \
    docs/verification_artifacts/persona_walkthrough_driver.py \
    > docs/verification_artifacts/persona_walkthrough_$(date +%Y-%m-%d)_raw_output.txt
```

Output is deterministic given the same Bloomberg CSVs + the same
`as_of`. The HMM regime cache (`WheelRunner._hmm_regime_cache`) is
per-process so a fresh run recomputes; the regime label is itself
deterministic because `GaussianHMM(..., random_state=42)`.

---

## 8. References

- `CLAUDE.md` §2 — the EV-authority invariant.
- `PROJECT_STATE.md` §1 — authoritative entry points (rank, dossier,
  tracker, api).
- `DECISIONS.md` D16 — EV-authority token (issuance + consume
  predicate).
- `DECISIONS.md` D17 — portfolio-risk gates (R7-R10 + tracker
  hard-blocks).
- `engine/wheel_runner.py:725` — `rank_candidates_by_ev`.
- `engine/wheel_runner.py:1901` — `select_book`.
- `engine/candidate_dossier.py:130` — `EnginePhaseReviewer`.
- `engine/candidate_dossier.py:535` — `build_dossiers`.
- `engine/wheel_tracker.py:346` — `issue_ev_authority_token`.
- `engine/wheel_tracker.py:603` — `consume_ranker_row`.
- `engine/wheel_tracker.py:1629` — `portfolio_context_snapshot`.
- `docs/verification_artifacts/README.md` — driver placement +
  re-run conventions.
- Companion verification drivers in this directory:
  `realism_verify_driver.py`, `f4_baseline_driver.py`,
  `s41_f4_validation_driver.py`.
