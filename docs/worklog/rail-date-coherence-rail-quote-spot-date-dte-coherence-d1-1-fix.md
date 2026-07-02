---
id: rail-date-coherence
title: Rail quote/spot date + DTE coherence (D1-1 fix)
kind: fix
status: in-flight
terminal: X
pr:
decisions: []
date: 2026-07-01
headline: >-
  The real-premium rail no longer pairs a quote from one market day with a
  spot from another: quote date must equal the spot bar date, quote DTE must
  match the modeled horizon (±10d), and as_of=None queries the market as of
  the spot bar. Closes the live 4-18x EV inflation (adversarial review D1-1).
surface: [engine/wheel_runner.py, engine/data_connector.py]
---

## Goal

Close finding **D1-1** of the 2026-07-01 adversarial weakness review
(Session-X-verified, CONFIRMED critical): the real-premium rail (#435) paired
option quotes from the larder's frontier (2026-06-17) with spots from the
older OHLCV frontier (2026-06-04), booking the 13-day market move as phantom
`edge_vs_fair`. Live `as_of=None` EV inflated 4-18× (MSFT $1,465.51 vs $82.89
rail-off; `prob_profit` 0.886→0.943 into the R11 band; UNH's negative EV
dragged −$77.36 → −$11.21 toward a rescue no reviewer can catch — the
corruption enters upstream of `ev_raw`). Exposed surfaces: everything that
omits `as_of` — `scripts/orchestrate.py`, `/api/committee`, `/api/tv/scan`,
`/api/candidates` default, the CLAUDE.md §4 smoke.

## What we tried

Three candidate designs, evaluated on paper against the live defect plus the
weekend/holiday and stale-universe cases before writing code (the fix had to
be refuse-only and byte-identical where the rail is absent):

1. wall-clock staleness bound only (connector-side);
2. quote-DTE-vs-modeled-DTE check only;
3. exact quote-date == spot-bar-date match + DTE tolerance + as_of-of-the-
   spot-bar querying (shipped).

## What worked

Design 3. Both guards are needed — each catches a case the other cannot:

- **date match** kills frontier skew (quote 06-17 vs spot 06-04) but passes
  the same-session-but-both-stale case, where the coherent quote's time value
  spans `(today + dte_target) − frontier` days, not `dte_target` days;
- **DTE tolerance** (|quote dte − modeled dte| ≤ 10 = 7d expiry snap + 3d
  weekend spot-bar lag) kills that second case but alone would admit up to
  ~7d of frontier skew.

`as_of=None` now resolves to the spot bar's date before querying the
connector, so the candidate quote is the coherent one rather than the larder
frontier; `get_option_premium_chain`'s own `as_of=None` branch is additionally
bounded by `max_staleness_days` vs the wall clock (the AB-4 public footgun for
non-ranker callers).

## What didn't

- **Date tolerance > 0** between quote and spot: any admitted gap books real
  market movement as edge; the larder is dense daily EOD so exact match costs
  almost nothing (rejections fall back to synthetic-BSM, which is what those
  rows used before #435).
- **DTE-only** or **wall-clock-only** designs: see above — each leaves a
  live admission path (up to expiry-tol frontier skew; fresh-larder/stale-
  OHLCV skew respectively).

## How we fixed it

`_resolve_real_premium` gains `spot_date` / `dte_target` / `dte_tol_days`
keyword guards (None = skipped, for legacy stubs; all four ranker call sites —
puts, covered-call, strangle put+call — supply both). The spot-bar date is
captured next to each ranker's `spot = ohlcv["close"].iloc[-1]`. Refuse-only
by construction: every new branch can only turn `market_mid` into
`synthetic_bsm`, never the reverse; rail-absent behavior is byte-identical.

## Evidence

- 5-ticker smoke, this box (larder frontier 06-17, OHLCV frontier 06-04):
  `as_of=None` now all `synthetic_bsm` — MSFT $82.89, UNH −$77.36, matching
  the rail-off control **to the cent**; `as_of='2026-06-04'`
  (frontier-consistent) still serves `market_mid` (MSFT $76.50); historical
  `as_of='2023-06-15'` still serves `market_mid`. No over-refusal.
- `tests/test_real_premium_wiring.py` + `tests/test_option_premium_accessor.py`
  44/44 green: 9 new unit pins (`TestQuoteSpotCoherence`), 2 e2e pins of the
  two live defect shapes (`TestRankerQuoteSpotCoherenceE2E` — prior-session
  quote at dated as_of; skewed larder frontier at as_of=None), 1 connector
  wall-clock pin.
- `test_f4_rv_widening.py`: 21/21 green under the CI condition
  (`SWE_OPTION_PREMIUM_DIR` → empty). The one with-rail local failure
  ($5.35 AAPL pin) is **pre-existing at origin/main** (stash A/B proven,
  identical failure) — that is D4-2 (regression-lock rail neutralization),
  the next campaign item, not this diff.

## Unresolved / handoff

- Universe-wide wall-clock staleness (D1-2/D3-2 — frontier vs today is
  invisible at runtime) is deliberately NOT fixed here; next PR (the rescued
  2026-06-15 stash design, branch `claude/rescue-2026-06-15-fixes`, is input).
- Regression-harness rail neutralization + larder fingerprint (D4-2/D5a-1)
  after that.
- Docs that describe the old `as_of=None` = "latest larder snapshot" accessor
  semantics should be reconciled when the campaign's doc pass lands.
