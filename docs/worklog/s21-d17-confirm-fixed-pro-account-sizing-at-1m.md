---
id: S21
title: D17 confirm-fixed + pro-account sizing at $1M
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Close the S15 loop. S15 (PR #148) found the portfolio-
aggregation gap: `RiskManager` / `SectorExposureManager` /
`HierarchicalRiskParity` / `StressTester` / Kelly helpers shipped in
`engine/risk_manager.py` + `engine/stress_testing.py` but **were not
imported by any decision-layer file** — positions could open with no
NAV-level sector cap, no portfolio-delta cap, no Kelly check.

Between S15 and S21 the institutional risk layer landed in two
phases:

- **#163 (D17 Phase 2)** wired three **hard-block gates** into
  `engine/wheel_tracker.py:_evaluate_d17_hard_blocks` —
  `check_sector_cap`, `check_portfolio_delta`, `check_kelly_size`
  from the new `engine/portfolio_risk_gates.py`. Fires only in
  strict mode (`require_ev_authority=True`), after the D16 token
  consume.
- **#165 (D17 Phase 3)** wired two **soft-warn rules** into
  `engine/candidate_dossier.py:EnginePhaseReviewer.review` — R7
  (`check_var`, default 5% NAV / 30-day / 95% conf.) and R8
  (`check_stress_scenario` C4 vol spike + `check_dealer_regime`).
  Fires only when a `PortfolioContext` is attached and the current
  verdict is `proceed`. Downgrade-only.

S21 verifies the new gates actually block what S15 said they
should block, characterises a pro-scale book at $1M, and exercises
R7/R8 with a constructed `PortfolioContext`.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, 25-ticker universe across 7 GICS sectors (same
set as S17). Three prongs:

- **Prong A** — sector-cap trip demo. `WheelTracker(initial_capital=
  150_000, require_ev_authority=True)`. Attempt CAT first
  (Industrials, strike $625.5, single-put collateral $62,550 =
  **41.7% of $150k NAV**). Expect `check_sector_cap` to reject —
  default `_DEFAULT_MAX_SECTOR_PCT=0.25`. Then attempt smaller
  positive-EV tickers (PG / HD / CVX) to see what else fires.
- **Prong B** — pro-account sizing at $1M. `initial_capital=
  1_000_000`, strict mode. Sort all 9 positive-EV candidates by
  `ev_dollars` desc, attempt to open sequentially with the full
  token-issue + token-consume flow. Record which gate fires per
  rejected attempt.
- **Bonus** — R7/R8 exercise. Build a `PortfolioContext` from the
  final Prong B book; attach to a `CandidateDossier` for the
  highest-EV candidate (CAT). Run `EnginePhaseReviewer.review` with
  and without the context; observe the verdict delta. Also call
  `check_var` and `check_stress_scenario` directly to capture
  their numerical results.

All paths use a synthetic `ChartContext` (visible_price=spot) so R2
and R3 stay silent and the gate path is the only thing exercised.
Strict mode uses `tracker.issue_ev_authority_token(ev_row)` →
`tracker.open_short_put(..., ev_authority_token=token,
current_ev_dollars=ev_dollars, prob_profit=prob_profit)`. Throwaway
driver under `%TEMP%\s21_d17\` (system temp, not in repo). Same
S18/S19/S20 pattern.

**Path.** D17 hard-blocks at `engine/wheel_tracker.py:1579-1719`:
gate sequence is `nav < min_nav_for_trading` (pre-gate) → sector
cap → portfolio delta cap → Kelly size cap. **First failing gate
short-circuits** and writes a `reject` entry into
`_ev_authority_log` with `action="reject", reason=..., ticker=...,
nav=..., nav_source=...`. R7/R8 soft-warns at
`engine/candidate_dossier.py:282-381` (post-#165): only fire when
`portfolio_context is not None and verdict == "proceed"`; both
downgrade-only.

**Status.** Done. **Verdict: D17 hard-blocks confirm-fixed (sector
cap works exactly as S15 predicted) + one binding-constraint
finding (default portfolio-delta cap is the dominant gate at all
NAV scales tested).**

**Findings:**

- **Prong A — `check_sector_cap` correctly blocks CAT at $150k.** Live
  output from the driver:

  ```
  CAT open attempt: opened=False, reject=sector_cap_breach
  reject_details: {
      'current_ev_dollars': 290.26, 'nav': 150000.0,
      'nav_source': 'live_mark_to_market',
      'sector': 'Industrials',
      'sector_pct': 0.417,
      'sector_limit': 0.25,
      'narrative': "Sector 'Industrials' would be 41.7%
                    (limit: 25.0%). Current positions: []"
  }
  ```

  Exactly the structural gap S15 named, mechanically closed. The
  audit entry shape (`action=reject, reason=sector_cap_breach,
  sector_pct, sector_limit, narrative`) matches the D17 audit-log
  schema pinned by `tests/test_ev_authority_log_schema.py`. The
  `nav_source='live_mark_to_market'` confirms the NAV was computed
  via `_compute_live_nav` (D17's "live NAV under gate decisions"
  principle), not against `initial_capital`. **S15's
  `SectorExposureManager`-orphaned finding is closed for sector
  exposure on the tracker hard-block path.**

- **Prong A unexpected — portfolio-delta cap is the *dominant*
  binding constraint at $150k.** After CAT failed sector, the next
  three positive-EV candidates (PG / HD / CVX, all in non-Industrials
  sectors) all failed at **gate 2**, not gate 3:

  ```
  PG  -> portfolio_delta_breach  post_open_delta=$3,382  cap=$450
  HD  -> portfolio_delta_breach  post_open_delta=$7,540  cap=$450
  CVX -> portfolio_delta_breach  post_open_delta=$4,698  cap=$450
  ```

  `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` × ($150k / $100k) =
  **$450 dollar-delta cap**. A single short put on a non-megacap
  ticker carries $3-8k of long-delta-dollars (per `RiskManager.
  calculate_portfolio_greeks`'s `delta * contracts * 100 * spot`
  computation at `engine/risk_manager.py:359`). **At $150k NAV the
  default delta cap admits zero single-put positions.** Prong A
  final book: 0 positions. **Logged** — not a bug (the gate works
  exactly as designed), but a calibration question: the default
  `_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` at
  `engine/portfolio_risk_gates.py:59` may be too tight for the
  retail wheel use case.

- **Prong B — at $1M, only 2/9 positive-EV candidates open.** Same
  binding constraint, just shifted: delta cap = 300 × ($1M / $100k)
  = **$3,000**.

  | Ticker | EV $ | Strike | post-open Δ$ | Outcome |
  |---|---:|---:|---:|---|
  | CAT | +290.26 | 625.50 | 16,507 | BLOCKED (delta) |
  | CVX | +117.73 | 192.00 | 4,698 | BLOCKED (delta) |
  | MRK | +85.09 | 107.00 | (≤3000) | **OPENED** |
  | HON | +18.66 | 209.00 | 5,279 | BLOCKED (delta) |
  | PG | +18.51 | 138.00 | 3,382 | BLOCKED (delta) |
  | HD | +18.37 | 303.00 | 7,540 | BLOCKED (delta) |
  | KO | +15.29 | 71.50 | (≤3000) | **OPENED** |
  | ORCL | +12.20 | 136.00 | 3,650 | BLOCKED (delta) |
  | GOOGL | +1.59 | 283.00 | 7,382 | BLOCKED (delta) |

  Final Prong B book: **2 positions (MRK + KO)**, cash $1,000,237,
  BP $982,387. **All 7 blocks were `portfolio_delta_breach`** — not
  one sector, Kelly, or NAV-floor block in this universe at $1M.
  Compared to S17's 7-position book at $1M without D17, **D17 in
  strict mode caps the realistic book at low-strike tickers**
  (MRK $107, KO $71.5 are the lowest in the universe). **Logged.**

- **Gate ordering is sector → delta → Kelly; the first failing gate
  short-circuits.** Per `_evaluate_d17_hard_blocks` at
  `engine/wheel_tracker.py:1665-1700`. In Prong B, the delta cap
  fires first on every blocked candidate (the gate body returns the
  audit entry and the function returns); we never see the Kelly
  gate or sector gate fire downstream. **For the Kelly gate to
  exercise in a future Sn, the delta cap would have to be raised or
  the candidate set restricted to ones that pass delta.** **Logged.**

- **R7 (VaR) properly skips on missing data — Q3 semantics work.**
  Constructed `PortfolioContext` from the Prong B book (MRK + KO,
  2 positions, $999,964 NAV). With no `correlation_matrix` and no
  `returns_data` passed, `check_var` returns
  `passed=True, reason='missing_data',
  details={'var_check': 'skipped', 'skip_reason':
  'no_correlation_matrix_or_returns_data'}`. The reviewer at
  `engine/candidate_dossier.py:300-309` correctly logs
  `R7: VaR check skipped (no_correlation_matrix_or_returns_data)`
  and does NOT downgrade. **Matches the documented "soft-warns
  don't fire on absent evidence" rule (Q3 of the #154 C4 design
  checkpoint, per `engine/portfolio_risk_gates.py:92-98`).**
  **Logged as a positive.**

- **R8 (stress + dealer-regime) didn't fire on the 2-position
  book.** Direct `check_stress_scenario` call yielded
  `passed=True, scenario='C4 Vol Spike',
  portfolio_pnl_dollars=-$5,583, drawdown_pct=0.56%,
  drawdown_limit_pct=0.08`. Under the C4 standard scenario (-10%
  spot + 30% IV per `engine/portfolio_risk_gates.py:71-77`), the
  2-position book takes a $5,583 hit — **0.56% of $1M NAV, far
  below the 8% cap**. To exercise R8 we'd need a larger / more
  concentrated book, or test directly against `check_dealer_regime`
  with a `short_gamma_amplifying` regime label. Out of scope here.
  **Logged.**

- **Verdict-delta with vs without `PortfolioContext` — no
  downgrade observed.** For the highest-EV candidate (CAT,
  ev_dollars=290.26) with a synthetic chart that passes R3
  (`visible_price=spot=656.77`), the reviewer returned
  `proceed / ev_above_threshold` in BOTH cases. With context
  attached, the extra note `R7: VaR check skipped` was appended;
  no verdict change. **R7/R8 only downgrade when the gates fire on
  real data; with the missing-data skip path, the candidate
  proceeds.** **Logged as a positive** — the soft-warn contract is
  downgrade-only and silent-on-skip, which is exactly what S19
  said `_sanitize_nans` does for the response path: "don't claim
  what you can't prove."

- **§2 verified across D17.** Every Prong A and Prong B attempt
  routed through `EVEngine.evaluate` upstream (via
  `WheelRunner.rank_candidates_by_ev` → token issue → token consume
  → D17 gates). The D16 + D17 stack is "EV-authority *and* portfolio-
  risk authority" — a candidate can pass EV and still be blocked
  by D17 (CAT at $150k: EV +$290.26 → blocked by sector cap; CAT at
  $1M: EV +$290.26 → blocked by delta cap). **The §2 invariant —
  no tradeable verdict without EVEngine.evaluate — extends naturally
  into D17 (no tradeable position without passing portfolio-risk
  gates too). No bypass observed.** **Logged.**

- **`_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0` calibration.** The
  cap at `engine/portfolio_risk_gates.py:59` (= 300 dollar-delta
  per $100k NAV) is the binding constraint at both $150k and $1M
  on this universe. Per `check_portfolio_delta` at
  `engine/portfolio_risk_gates.py` (and per the D17 design comment
  on #113), this is **deliberately conservative** — a wheel book
  with 5 unbalanced single puts could carry $20-40k delta-dollars
  unhedged. But the cap of $3,000 at $1M means a book is naturally
  capped at low-strike tickers; **a $1M pro running default D17
  strict mode is structurally biased toward $50-$200 strike names**
  (MRK / KO / WMT / PFE / KO-class). **Logged** as a calibration
  data-point for the eventual D17 default-tuning discussion. Not a
  bug, not a fix in this Sn.

**Verdict.**

- **D17 hard-block sector cap: CONFIRM-FIXED.** The structural gap
  S15 named (no sector-exposure enforcement at position-open) is
  mechanically closed by #163. CAT at $150k correctly blocked with
  `sector_pct=0.417, sector_limit=0.25` and a structured audit
  entry. The five-finding orphaned-`SectorExposureManager` thread
  from S15 is **closed for the tracker hard-block path**. (The
  dossier soft-warn path R7/R8 also exists post-#165; R7 properly
  skipped on missing data in this run, R8 didn't trigger on the
  small book.)
- **D17 portfolio-delta cap: WORKS BUT TIGHT.** At default
  `300/$100k NAV`, admits only low-strike single puts. At $150k the
  cap is $450 — no single put passes. At $1M the cap is $3,000 —
  2 / 9 positive-EV candidates pass. **Logged as a calibration
  data-point**, not a fix-up.
- **D17 Kelly gate: NOT EXERCISED.** Gate ordering puts Kelly third;
  delta cap fired first on every blocked attempt. A future Sn
  could either raise the delta cap or pick candidates that pass
  delta to exercise Kelly.
- **R7 / R8 dossier soft-warns: BEHAVE AS DOCUMENTED.** R7 properly
  skips on missing correlation/returns data (Q3); R8 stress
  scenario benign on a small book. To fully exercise either,
  follow-up Sn needs a richer `PortfolioContext` (real returns,
  larger book).

**AI handoff.**

- **Highest-leverage observation from this run: the D17 portfolio-
  delta default may be too tight for the wheel use case.** Three
  options the user / Terminal A could weigh, each with a literal
  proposed diff at `engine/portfolio_risk_gates.py:59`:

  ```python
  # engine/portfolio_risk_gates.py:59 (current)
  _DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0

  # option 1 (loosen by 3x):
  _DEFAULT_DELTA_CAP_PER_100K_NAV = 1000.0  # admits CAT at $1M

  # option 2 (loosen by 10x for wheel-specific use):
  _DEFAULT_DELTA_CAP_PER_100K_NAV = 3000.0  # ~16k delta at $1M

  # option 3 (leave default, document as conservative):
  # — add a doctest example showing "default admits 2 positions at $1M"
  ```

  This is a **design decision, not a bug**. Numbers above just
  contextualise the calibration question. Out of scope for this Sn.

- **The delta-cap-as-binding-constraint finding makes the Kelly
  gate (gate 3) effectively unreachable** at default settings,
  because gate 2 short-circuits first. If a future Sn wants to
  exercise Kelly (`check_kelly_size` at
  `engine/portfolio_risk_gates.py:384`), the test driver should
  either raise the delta cap per-call (the gate functions accept
  override kwargs) or filter to low-delta candidates that pass
  gate 2.

- **R7 / R8 follow-up Sn would need a real `PortfolioContext`** —
  build `returns_data` from connector OHLCV (compute daily log
  returns over a 252-day window) and `correlation_matrix` via
  `RiskManager.calculate_correlation_matrix` (or equivalent).
  Out of scope here; flagged for a future Sn.

- **The audit-log schema for `reject` entries is structured** —
  `engine/wheel_tracker.py:_evaluate_d17_hard_blocks` writes
  `action='reject', reason=<gate>, ticker, nav, nav_source,
  current_ev_dollars, sector, sector_pct, sector_limit,
  narrative` (sector path; other gates have parallel detail
  bags). **This is the structured-drops finding S16 asked for** —
  on the reject path, the audit log IS structured with discrete
  fields. (`.attrs["drops"]` on the ranker output still uses free-
  text `reason`; the post-#163 reject log is the structured
  alternative on the tracker side.) **Logged as a positive — S16's
  AI-handoff fix-up #1 is partially closed.**

- **S15 closure update.** S15 named six aspects of orphaned risk
  layer; D17 has closed three:
  - ✅ `SectorExposureManager` — wired via `check_sector_cap`.
  - ✅ Portfolio Greeks (`calculate_portfolio_greeks`) — used by
    `check_portfolio_delta`.
  - ✅ VaR (`calculate_var`) — wired via R7 / `check_var`.
  - ⚠ Kelly (`calculate_kelly_fraction`) — wired but not yet
    exercised in any test (delta gate short-circuits first).
  - ⚠ Stress testing (`StressTester`) — wired via R8 /
    `check_stress_scenario` but doesn't fire on small books.
  - ❌ HRP (`HierarchicalRiskParity`) — still orphaned (no consumer
    in the decision-layer; only `tests/test_advanced_quant.py`).
  **S15-mark-2 future Sn could be a re-run with a larger Prong B
  book to exercise Kelly + R8, plus a fresh HRP-orphan check.**

- **Ruled out per the prompt:** any decision-layer code change
  (S21 found surfaces, did not fix), Theta provider (sandbox-
  blocked), advisor committee (not in scope), tuning the gate
  defaults (design discussion, not a usage test's remit).

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every Prong A and Prong B candidate routed through `EVEngine.evaluate` upstream; D17 hard-blocks fire AFTER the EV check.
  - Qualitative verdict: match — both prongs reproduce the documented behavior on `main`:
    - **Prong A** ($150k strict mode, CAT @ $625.50): `action=reject reason=sector_cap_breach` exactly as the original entry's audit-log narrative.
    - **Prong B** ($1M strict mode, 9 positive-EV candidates): **2 opened, 7 rejected with `portfolio_delta_breach`** — identical pattern to the original (delta cap = $3,000 at $1M binds before sector or Kelly).
    - The five `portfolio_delta_breach` tickers observed (CAT, GOOGL, CVX, PG, HON) overlap the original's set (CAT, CVX, HON, PG, HD, GOOGL, ORCL).
  - Numerical drift > 5%: not applicable — the cap-binding-constraint result is deterministic and ev_dollars/Δ$ map nearly 1:1 (CAT ev=+290 → +445 from PR #179 doesn't change the delta-breach decision because delta-dollars dominate at $1M).
  - Notes: `check_var` and `check_stress_scenario` signatures evolved post-S21 — both now require an explicit `candidate_option: dict` argument (the candidate being assessed). The R7 / R8 reviewer-integration is unchanged — they still fire only when `PortfolioContext is not None and verdict == "proceed"`.
