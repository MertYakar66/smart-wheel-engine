---
id: S30
title: HMM regime-multiplier realism (April 2025 vol spike)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Realism check on the HMM regime layer — does the
engine's `hmm_multiplier` (output of the 4-state Gaussian HMM at
`engine/regime_hmm.py:76`, wired into `rank_candidates_by_ev` at
`engine/wheel_runner.py:1143-1179`) actually shift in the right
direction at a known regime transition, and is the magnitude
realistic? CLAUDE.md §2 documents the multiplier range as
`[0.0, 1.25]` (per-state weights: `crisis: 0.2, bear: 0.5,
normal: 1.0, bull_quiet: 1.25`). S30 tests both downward and
upward transitions against the April 2025 broad-market vol
spike — a real-world event with measurable spot, IV, and
realized-vol moves.

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
five-name watchlist (AAPL, MSFT, JPM, XOM, UNH) across two
transition windows + one calm-period anchor:

| Window | Pre-date | Post-date | Transition (survey-identified on AAPL) |
|---|---|---|---|
| T1 — downturn | 2025-04-02 (Wed) | 2025-04-04 (Fri) | bear → crisis |
| T2 — recovery | 2025-04-11 (Fri) | 2025-04-15 (Tue) | crisis → bear |
| Anchor | — | 2026-03-20 (cutoff) | for current-regime comparison |

Survey used AAPL log-returns 2018-2026 fit to a 4-state Gaussian
HMM (n_iter=20, random_state=42 — same as the live ranker config
at `wheel_runner.py:1165`) and Viterbi-decoded the state path
to find the transitions inside the data window.

Driver under `%TEMP%\s29\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]]. 25-delta / 35-DTE
short puts. `use_event_gate=False` to isolate the HMM signal
from earnings-window noise. Read-only on decision layer.

**Path.** `WheelRunner.rank_candidates_by_ev` at
`engine/wheel_runner.py:631`. HMM fitting at lines 1143-1179
(`tail = log_rets[-504:]`, cached by `(ticker, hash(fingerprint))`).
Per-state weights at `regime_hmm.py:265-280`
(`position_multiplier`). Final regime multiplier composed at
`wheel_runner.py:1300`
(`combined_regime_mult = hmm_regime_mult * skew_mult * news_mult
* credit_mult`), passed into `ShortOptionTrade.regime_multiplier`,
consumed by `EVEngine.evaluate`.

**Status.** Done. **Verdict: the engine's HMM is the
best-behaved of the four knowledge surfaces in this campaign.
It captured the April 2025 vol spike correctly across ALL FIVE
names simultaneously (broad-market crisis recognition), respects
its documented multiplier range, and the downstream EV behaviour
is realistic — the multiplier cut is balanced against the
simultaneous IV-spike-driven premium increase, not a blind
kill-switch.**

**Findings:**

- **(F1 — major positive) The HMM correctly recognised the April
  2025 broad-market crisis on every name.** Pre-vs-post the
  2025-04-03 transition:

  ```
  ticker   mult_pre   regime_pre   ->  mult_post  regime_post   d_mult   d_spot     d_iv
  AAPL        0.898       normal   ->      0.200       crisis   -0.698  -15.85%  +73%
  JPM         0.552         bear   ->      0.200       crisis   -0.352  -14.46%  +76%
  MSFT        0.700       normal   ->      0.201       crisis   -0.499   -5.84%  +39%
  UNH         0.668         bear   ->      0.232       crisis   -0.436   +0.35%  +25%
  XOM         0.700       crisis   ->      0.200       crisis   -0.499  -12.07%  +89%
  ```

  Per-ticker independent HMM fits converged on `crisis` for
  all five names within two trading days. The realised spot
  moves (-5% to -16%) and IV spikes (+25% to +89%) on four of
  five names are consistent with a real broad-market crisis
  episode; UNH was a partial exception (spot held, IV moved
  less) but the HMM still moved to crisis on UNH's own
  returns. **Logged as a positive — per-ticker HMM converging
  on the same regime label across a watchlist IS the right
  signal for a macro event, even without an explicit
  cross-asset aggregator.**

- **(F2 — positive) Reverse transition (recovery) captured on
  2025-04-15.** Pre-vs-post the 2025-04-14 transition:

  ```
  ticker   mult_pre   regime_pre   ->  mult_post  regime_post   d_mult
  AAPL        0.212       crisis   ->      0.817       normal   +0.605
  MSFT        0.478         bear   ->      0.633         bear   +0.155
  XOM         0.405       crisis   ->      0.738       normal   +0.334
  ```

  AAPL recovers crisis → normal (multiplier rises 4×). XOM same
  shape. MSFT stays bear but moves up the bear-confidence axis
  (0.48 → 0.63). The HMM is **responsive to recovery, not just
  downturns** — it does not get permanently stuck in crisis
  after a vol spike. JPM and UNH dropped out of the result set
  on one of the two dates in T2 (history-gate or chain-quality
  cutoff during the high-vol window — F6 below). **Logged as
  a positive — the HMM mean-reverts correctly.**

- **(F3 — positive) HMM multipliers respect the documented
  `[0.20, 1.25]` envelope.** Every `hmm_multiplier` value
  observed in the matrix sits inside the bound. The crisis
  weight maps to exactly 0.200 (the hard floor) for pure-state
  posteriors; mixed states yield values like 0.201, 0.212,
  0.232 (posterior-weighted average across the 4-state
  weights). The maximum observed was XOM at 0.9217 on
  2026-03-20 (`bull_quiet` posterior weighted into the average,
  sub-1.25 because the posterior is not 100% bull_quiet).
  **Logged.**

- **(F4 — design positive) EV impact is balanced, not blind.**
  On T1 the HMM cut multiplier 0.5-0.9 → 0.2, but `ev_dollars`
  did not simply collapse. Same window:

  ```
  ticker    ev_pre   ev_post   d_ev      Driver
  AAPL      +32.52   +60.95   +28.43    IV spike +73% → premium 3.05 -> 4.60 outweighed HMM cut
  JPM     +108.33    +75.40   -32.93    HMM cut dominated
  MSFT    +195.91    +90.39  -105.52    HMM cut dominated
  UNH     +520.19   +177.81  -342.38    HMM cut + premium up less than HMM impact
  XOM      -12.58    +20.19   +32.77    IV spike +89% rescued an underwater candidate
  ```

  The engine is doing balanced multi-signal math — IV spike
  raises the synthetic premium (and `ev_dollars` is positive in
  the integrand) while the HMM multiplier cuts the final scaling.
  Whichever wins on a given name depends on which moved more.
  **This matches the intent of the dealer-multiplier asymmetric
  clamp (CLAUDE.md §2) — the HMM is a *de-emphasis* lever, not
  a *veto*.** A real wheel trader does not want a regime
  indicator to fully suppress trades; the engine's behaviour
  here is realistic. **Logged as a positive.**

- **(F5 — observability gap) No cross-asset coherence
  signal.** The HMM is fit per-ticker independently. When
  5/5 names go to `crisis` simultaneously (T1), that is a
  macro-event signal worth surfacing distinctly. Currently
  the trader sees five separate "this name's HMM says
  crisis" rows in the ranker output, not a single
  "macro regime appears to be in crisis" header. A
  `macro_regime` aggregator column (or board-level
  signal in the dossier) would close this. **Logged as a
  follow-on observability finding** (same family as S22 F1,
  S28 F2, S29 F5 — the campaign's observability theme).

- **(F6 — data dropout, minor) UNH and JPM dropped from
  some result sets during T2.** UNH disappears on the
  2025-04-15 post-row; JPM disappears on the 2025-04-11
  pre-row. Most likely cause: history-gate or chain-quality
  gate triggered by the unusual return distribution from
  the 2025-04-04 crisis day. Not a HMM bug but a data-flow
  observation — the HMM had data to fit on the surviving
  rows; the question is why the upstream filters dropped
  these specific cells. **Logged.**

- **(Anchor) Current data-cutoff regime is mildly defensive.**
  At `as_of=2026-03-20`:

  ```
  ticker      hmm_regime   hmm_multiplier
  AAPL        bear         0.677
  JPM         bear         0.711
  MSFT        bear         0.463
  UNH         bear         0.668
  XOM         bull_quiet   0.922
  ```

  Four of five megacaps in bear regime; energy (XOM) the
  outlier in `bull_quiet`. Consistent with a recent
  modest-vol environment, not a crisis. **Logged as an
  anchor.**

- **§2 verified.** `rank_candidates_by_ev` routes every
  candidate through `EVEngine.evaluate`. The HMM
  multiplier is multiplicative and cannot rescue a
  negative-EV trade (XOM in T1 went from -$12.58 to +$20.19
  not because of HMM but because of the IV-spike-driven
  premium increase; the HMM actually pulled the multiplier
  *down*). **Logged as a positive.**

**Realism Check.**

| Aspect | Engine | Real-market behaviour | Verdict |
|---|---|---|---|
| AAPL HMM regime at 2025-04-04 (post-vol-spike) | `crisis` (mult 0.200) | -15.85% spot, +73% IV spike — clear crisis | ✓ Aligned |
| 5/5 watchlist names at `crisis` on 2025-04-04 | All five converge to `crisis` simultaneously | Broad-market sell-off; macro event signature | ✓ Aligned (per-ticker convergence is the correct macro signal) |
| AAPL recovery to `normal` at 2025-04-15 | mult 0.817 (`normal`) | AAPL +2.0% recovery, IV partially crushed | ✓ Aligned |
| MSFT lag-recovery (stays bear) | mult 0.633 (`bear`) | MSFT only -0.7% net recovery, IV still elevated | ✓ Aligned (cautious mean-reversion) |
| Anchor regime at 2026-03-20 | 4/5 bear, XOM bull_quiet | Recent regime is mildly defensive | ✓ Aligned (consistent with current data) |
| HMM multiplier respect `[0.20, 1.25]` envelope | Observed range 0.200 - 0.922 across matrix | Documented bound | ✓ Aligned |
| EV-impact direction on T1 | Mixed: AAPL/XOM up, JPM/MSFT/UNH down | Trader expects regime cut but vol spike raises premium | ✓ Aligned (balanced multi-signal) |
| Cross-asset coherence signal | None — each name reads independently | Trader watching 5/5 crisis would call macro event | ⚠ Observability gap (F5) |

**Verdict.**

- **The HMM is the best-behaved knowledge surface in this
  campaign.** It captured a real broad-market crisis on
  every name in the watchlist simultaneously, respects its
  documented multiplier bounds, mean-reverts correctly on
  recovery, and behaves realistically as a *de-emphasis*
  signal (not a veto) when composed with IV spikes.

- **No new bug surfaced in the HMM logic itself.** F1-F4
  are all positives. F5 is a *missing* feature (cross-asset
  aggregator) rather than a logic error. F6 is upstream
  data-flow, not HMM.

- **The HMM is the realism counter-example to S29's skew
  finding.** Skew is dormant on Bloomberg because the data
  isn't there; HMM is alive because it consumes OHLCV
  log-returns — a column that *is* populated cleanly across
  503 tickers × 8+ years. The lesson generalises: engine
  surfaces that consume well-supported data columns work as
  built; surfaces that need chain-level data don't, on
  Bloomberg.

**AI handoff.**

- **Fix #1 (observability, follow-on to F5):** add a
  `macro_regime` row at the top of the ranker output (or a
  `macro_regime_unanimous` boolean in the dossier metadata).
  Compute as the modal `hmm_regime` across the result set
  when the agreement rate exceeds a threshold (e.g. 4/5 of
  ranked names share a regime). Surfaces the broad-market
  signal that's currently latent in five separate rows.

- **Fix #2 (observability, related):** add a
  `hmm_state_posterior` column with the full 4-state
  posterior (or just the top-2 probabilities) so the trader
  can distinguish "0.21 because I'm 95% crisis" from "0.32
  because I'm 50% crisis + 50% bear" — currently both
  serialize to similar multiplier values but mean different
  things downstream.

- **Fix #3 (data-flow, follow-on to F6):** investigate why
  UNH and JPM dropped from the T2 result sets. If it's the
  history gate firing on tail-window length (the 504-day
  tail at the high-vol windows might shift the cache key
  enough to gate them), the fix is at `wheel_runner.py`
  history-gate level. If it's chain-quality gate, that's
  separate (and not a real concern given Bloomberg has no
  chain anyway). Small Sn (~1 page).

- **HMM-cross-asset Sn:** the natural follow-on to S30 is a
  basket Sn that rank-orders 50 S&P 500 names on the same
  watershed dates (e.g. 2025-04-04) and reports the fraction
  in each HMM regime. If 40+/50 go to crisis simultaneously,
  that confirms the macro signal interpretation. If it's
  noisier (15/50 crisis, 20/50 bear, 15/50 other), the
  per-ticker convergence in S30 was a small-sample artifact
  and the F5 macro-aggregator design would need a smarter
  threshold.

- **Dealer positioning (S31 candidate, deferred this cycle):**
  the dealer-regime path is similar to skew in that it needs
  per-strike gamma data Bloomberg doesn't have. After a Theta
  replay, a small Sn could test R6 / R8 contract fire on a
  synthetic `dealer_regime_by_ticker={"AAPL": "short_gamma_amplifying"}`
  via `WheelTracker.portfolio_context_snapshot` and
  `build_dossiers(portfolio_context=ctx)` (the #174 wire). Pair
  with skew on the same Theta data.

**Methodology debt.**

- **Single watershed event (April 2025 vol spike).** The
  T1 finding ("all 5 names converge to crisis simultaneously")
  is from one transition. To make the broader claim ("the
  per-ticker HMM converges on macro events"), basket the test
  across a half-dozen other historical vol-spike dates (e.g.
  COVID 2020-03, August 2024 yen-carry unwind, October 2023,
  September 2022 CPI shock) and verify the same shape. If 4
  of 6 events show 4/5 convergence, the macro-signal
  interpretation is strong; if it's 2 of 6, F5's macro
  aggregator needs a per-event regime classifier.

- **Cache not exercised.** The HMM cache (keyed by
  `(ticker, hash(tail-fingerprint))`) means re-running the
  same as_of for the same ticker hits the cache and skips
  re-fit. The 2-as_of test in S30 always re-fits (the tail
  fingerprint differs between adjacent days). A
  cache-stress Sn would help confirm "the cache is invalidated
  correctly when new bars arrive" but isn't a realism
  question.

- **No comparison to a heuristic baseline.** A pro trader's
  mental model for regime might be "VIX > 30 → crisis, VIX
  < 15 → bull_quiet, in between → normal/bear." The
  engine's HMM agrees on the T1 dates (VIX clearly spiked
  to >30 in April 2025) but a side-by-side comparison
  across a basket would quantify "HMM is X% concordant with
  VIX-threshold regime" — a useful sanity check.
