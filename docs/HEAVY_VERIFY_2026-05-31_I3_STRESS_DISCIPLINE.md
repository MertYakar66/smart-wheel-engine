# Heavy-Verify Campaign 2026-05-31 — I3: Does it behave like a disciplined trader under stress?

**Investigation:** risk-gate binding, concentration, compounding downgrades,
earnings lockout (incl. PIT-correctness), and crash-window tail behavior.
**Drivers:** `docs/verification_artifacts/campaign_2026-05-31/i3_*.py`.
**Raw:** `raw_output/i3_*_RAW.txt`. **Status:** observe-and-document; `engine/` not modified
(`git diff origin/main -- engine/` empty). Independently spot-verified by the lead
(gate default + the crash-procyclicality realization both reproduced).

---

## VERDICT (where the engine is disciplined, where it isn't)

> **Mixed — sound rules, but two material gaps. SOUND: the negative-EV hard-block
> and the downgrade-only review contract are airtight (no rescue path; see also I4).
> GAP 1 (dormant discipline): the portfolio risk caps (sector 25%, single-name 10%,
> portfolio-delta, Kelly) are correctly built and unit-tested but DISABLED on the
> default path every backtest and the ranker use (`require_ev_authority=False`);
> they would bind in ~half of the months scanned if armed. GAP 2 (procyclical at
> crash entry, the most serious behavioral finding): the engine's empirical forward
> distribution lags the regime, so at the 2020 crash entry it labelled 89% of
> candidates positive-EV — and those trades realized −$1,305/contract at 82%
> assignment. It recommended selling puts into the buzzsaw. It self-corrects within
> ~4 weeks once realized crash vol enters the trailing window.**

Confidence: **high** (A, C, E — code-cited + realized-outcome-tested); **medium-high**
(D — PIT look-ahead is real but conservative in direction).

---

## A — Risk caps are dormant on the default path (CONFIRMED, high value)

* `engine/wheel_tracker.py:241` — `require_ev_authority: bool = False` (default).
  The D16 token guard (`:351`) and the D17 hard-block call
  (`_evaluate_d17_hard_blocks`) both sit behind `if self.require_ev_authority:`.
  **With the default, the four hard-blocks (sector 25% NAV, single-name 10% NAV,
  portfolio-delta, Kelly) never run.** Independently confirmed by the lead.
* `engine/wheel_runner.py`: `RiskManager` is not referenced; `rank_candidates_by_ev`
  and `select_book` never set `require_ev_authority`. The portfolio risk gate is
  **unwired into the ranker and book selection.**
* R7 (VaR) / R8 (stress) fire only in the dossier reviewer when a `PortfolioContext`
  with `returns_data` is attached — and **no upstream caller assembles
  `returns_data`**, so R7's covariance/historical path is structurally dead on the
  default path (R7 emits `VaR check skipped`).
* **Quantified:** scanning all 73 non-empty monthly snapshots, the default book
  (top-N by `ev_dollars`, one cash-secured contract/name, $1M) breaches the dormant
  caps in **174 month/book combos** — worst clean (multi-name) sector concentration
  **Healthcare 47% NAV (2024-12), Consumer Cyclical 44% (2025-04)**, well past the
  25% R9 cap. (Single-name breaches are partly cash-securing mechanics on
  ultra-high-priced names — NVR/AZO — flagged, not over-claimed.)

> **Implication:** any statement that "the engine respects 10%/25% concentration
> caps" is **unsupported** unless the tracker was explicitly built with
> `require_ev_authority=True`. No ranker path does. The discipline exists on paper.

## B — Concentration-cap asymmetry (CONFIRMED)

* `select_book.max_weight_per_name` (`wheel_runner.py:2028`) defaults `None` (off);
  even when set it only drops a single too-expensive name (0/1 knapsack, no
  aggregation across names).
* **Gate asymmetry (reproduces S24):** `check_sector_cap`
  (`portfolio_risk_gates.py:343`) sums **option notional only**;
  `check_portfolio_delta` (`:642`) includes **assigned-stock delta**. A book of
  3,000 assigned AAPL shares ($450k, 45% NAV) reports **0.0% sector** (passes) but
  **fails** the delta cap. The same position is invisible to one gate, fully counted
  by the other.
* **Design tension:** the delta cap is $300/$100k NAV = **$3,000** for a $1M account;
  a single 1,000-share assigned lot (~$150k delta) is 50× the cap. For a *wheel*,
  where assignment is the expected path, this cap (when armed) would refuse
  essentially any post-assignment book — a senior trader should know it before
  enabling strict mode.

## C — Compounding downgrades (CONFIRMED clean)

* Downgrade-only holds: negative EV, +inf, NaN all → `blocked` even with a perfect
  chart and benign dealer regime. **No rescue path** (also independently shown in I4).
* Rules are **first-match by source order R1→R10, not by severity.** A candidate that
  is both chart-missing (R2→review) and spot-mismatched (R3→skip) returns the
  *milder* `review` (R2 fires first), not `skip`. Both are non-tradeable, so no §2
  breach — but the surfaced reason is the less-conservative one.

## D — Earnings lockout + PIT look-ahead (REAL, conservative direction)

* The **2022-07-01 candidate collapse** (≈442→57 universe-wide) is attributed:
  on a 25-name subset, gate-ON dropped **20 of 24** to the `event` gate in July vs
  1 in June — i.e. almost entirely **Q2 earnings clustering**, not the 504-day gate
  or staleness.
* **PIT look-ahead (real finding):** the gate reads `conn.get_next_earnings(ticker,
  as_of)` (`data_connector.py:396`), which filters a **static CSV of realized
  `announcement_date`s** to `> as_of` (110 rows dated into 2028). It is a fixed
  calendar of *realized* dates, not an as-of-known *scheduled* calendar — so for any
  earnings whose realized date differed from what was scheduled at `as_of`, the gate
  decides with hindsight.
* **Severity bounded, direction conservative:** the ±5-day buffer means only
  earnings within ~`as_of+40d` can bite, and the gate can only **remove** candidates
  (never rescue) — so the look-ahead **cannot inflate backtest returns**; it makes
  the lockout slightly more accurate than was knowable. But per-month candidate
  counts and *which* names survive are not strictly PIT.

## E — Procyclical at crash entry (MATERIAL, independently verified)

On **2020-03-02** (≈3 weeks before the March-23 bottom) the engine flagged **89% of
candidates positive-EV** with a *shallower* median `cvar_5` than calm 2021 — the
opposite of crash discipline. Adversarially investigated:

* The shallow cvar_5 is partly scale (lower crash strikes → lower collateral) **and**
  a lagging forward distribution: `distribution_source` is **100%
  `empirical_overlapping`**, drawn from a trailing window still dominated by the 2019
  bull, so downside is under-modeled while IV-inflated premiums pull EV positive.
* The HMM **correctly** labelled 63% "crisis" and de-rated `regime_multiplier` to
  median **0.31×** — but that was **not enough to flip the EV sign**.
* **Decisive realized-outcome test** (lead-verified, independently reproduced):

  | entry date | positive-EV | realized mean/contract | win % | assign % | worst |
  |---|---|---|---|---|---|
  | 2020-03-02 (pre-bottom) | 89% (335/378) | **−$1,305** | 23% | 82% | −$83,039 (NVR) |
  | 2020-04-01 (recovery) | 100% (63/63) | +$302 | 100% | 0% | +$27 |
  | 2021-06-01 (calm) | 9% (over-cautious) | +$221 | 92% | 8% | −$1,123 |

* **The engine recommended selling puts into the crash and they got run over.** It
  self-corrects within ~4 weeks once realized crash vol enters the trailing window,
  and is *over*-cautious in calm tape. The danger is concentrated at the
  **regime-transition entry**, where the empirical distribution lags hardest.

> **This is the same root cause as I1's crisis-regime −26pp top-bin over-confidence.
> Two independent investigations converge:** the empirical forward distribution
> under-models the tail exactly at crisis onset, so the engine is simultaneously
> most over-confident (I1) and most procyclical (I3-E) when it matters most.

## Caveats
Single-name breach magnitudes in A are partly cash-securing mechanics on
ultra-high-priced names (the clean A finding is the multi-name *sector* breaches).
E's dollar figures are frictionless/synthetic-premium (the *direction* is robust
across 335 names; the magnitude would soften slightly net of fills). D's subset is
25 names; A's scan is all 73 non-empty months.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i3_gate_dormancy.py
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i3_crash_tail_behavior.py
# ... i3_cap_asymmetry.py, i3_compounding_downgrades.py, i3_earnings_lockout_pit.py
```
All raw output under `raw_output/i3_*_RAW.txt`.
