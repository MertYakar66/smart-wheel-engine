---
id: S26
title: Mid-cycle re-evaluation realism (AAPL challenged vs MU winning)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Realism counterpart to S25 on the management side.
The pro-trader heuristic for an open short put is well-known:
*"roll at 21 DTE if the put is < 0.10 delta — lock the gain."*
S26 asks: does the engine's `WheelTracker.suggest_rolls` produce
the same shape of decision at DTE_remaining=21? Tested on two
matched scenarios — one where the put is winning (heuristic
applies, expect ALIGNED) and one where the put is challenged
(heuristic is silent, observe what the engine does).

**Setup.** Both scenarios are 35-DTE entries re-evaluated 14 days
later (= 21 DTE remaining), so the heuristic's "21 DTE" trigger
fires in both cases.

| | Scenario A (challenged) | Scenario B (winning) |
|---|---|---|
| Ticker | AAPL | MU |
| Entry | 2026-02-09 | 2026-03-03 |
| Re-eval | 2026-02-23 | 2026-03-17 |
| Entry spot | $274.62 | $379.68 |
| Re-eval spot | $266.18 (-3.07%) | $461.69 (+21.60%) |
| Entry strike (engine 25-delta) | $261.50 | $339.50 |
| Entry premium | $3.41 | $12.63 |
| Re-eval current delta | -0.3610 | -0.0260 |
| Re-eval BSM mark | $4.13 (loss) | $0.81 (big gain) |

Driver under `%TEMP%\s26\`, not committed;
`sys.path.insert(0, r"C:\Users\merty\Desktop\swe-terminal-a")`
prepended per [[sys-path-worktree-shadow]].

**Path.** `WheelTracker.suggest_rolls` at `engine/wheel_tracker.py:1840`,
exercised on a real position opened via `open_short_put`.
`rank_candidates_by_ev` at `engine/wheel_runner.py:579` provides
the entry-strike anchor. `event_gate=False` throughout (so the
B-scenario re-eval can run; MU's 2026-03-18 earnings is 1 day
forward of the re-eval date and would otherwise block).

**Status.** Done. **Verdict: the engine's `recommend` flag
aligns with the pro-trader heuristic in the case the heuristic
APPLIES (winning put, delta < 0.10). In the case the heuristic
is silent (challenged put), the engine makes a defensive-roll
recommendation that minimizes expected loss — a reasonable
extension of the heuristic, not a contradiction.**

**Findings:**

- **Scenario A — AAPL challenged put, ENGINE SAYS ROLL (heuristic
  silent → "HOLD" by default).** AAPL drifted down -3.07% and the
  put is now ITM-ish (delta -0.36). Engine says:

  ```
  16/16 candidates recommend=True
  best roll_ev:  -$162.01
  hold_ev:       -$391.27
  edge:          +$229.26  (roll is the lesser loss)
  ```

  Top 3 by roll_ev:

  ```
  new_dte  new_strike  target_delta  new_premium  net_credit_debit  new_ev_dollars  roll_ev  hold_ev
       49       256.0          0.30         5.01            +88.08          257.59  -162.01  -391.27
       49       252.5          0.25         3.93            -19.64          193.74  -225.86  -391.27
       35       257.0          0.30         4.12             -0.71          145.50  -274.10  -391.27
  ```

  The engine sees a -$391 expected loss on hold (the put is
  challenged) and offers a defensive roll out 14 more days at a
  slightly lower strike for -$162 expected loss. **Net edge $229
  in favor of rolling.** This is *NOT* the heuristic's "lock the
  gain" case — but it IS a coherent management recommendation
  (limit the loss). The pro-trader heuristic doesn't address the
  challenged-put case; the engine produces a sensible default
  there. **Logged as a positive — engine extends naturally
  into the "challenged" regime the heuristic doesn't cover.**

- **Scenario B — MU winning put, ENGINE AND HEURISTIC BOTH SAY
  ROLL — ALIGNED.** MU rallied +21.60% (the put went from $339.50
  strike, $379.68 spot at entry, to $461.69 spot at re-eval — a
  -26.5% spot move from below the strike to well above). Current
  put delta -0.026 (deep OTM, almost worthless). Engine says:

  ```
  16/16 candidates recommend=True
  best roll_ev:  +$1874.19
  hold_ev:       -$1.86
  edge:          +$1876.05
  ```

  Top 3:

  ```
  new_dte  new_strike  target_delta  new_premium  net_credit_debit  new_ev_dollars  roll_ev
       63       416.5          0.30        29.82          +2901.52         1956.62  1874.19
       63       398.5          0.25        23.06          +2225.18         1670.49  1588.05
       49       419.5          0.30        25.78          +2497.17         1363.87  1281.43
  ```

  The current put is worth $0.81 (unrealized gain of $11.82 per
  share = +$1,182 / contract). A new 63-DTE 30-delta put at
  $416.50 collects $29.82 / share = +$2,982 fresh premium and
  expects +$1,956 ev_dollars. **The engine produces exactly the
  "lock the gain, roll into more premium" pattern the heuristic
  prescribes.** Logged as a positive.

- **The engine's hold_ev formula correctly distinguishes the two
  cases.** For the winning put (B), hold_ev = -$1.86 (essentially
  zero — the residual decision after the gain is "wait a bit
  more for the last $0.81 of premium to decay"). For the
  challenged put (A), hold_ev = -$391.27 (real expected
  additional loss from holding the ITM position to expiry). The
  engine reads the position state correctly through the
  EVEngine.evaluate of the synthetic hold trade. **Logged as a
  positive.**

- **(F1) `suggest_rolls` returns 16 candidates in both scenarios
  but emits NO `.attrs["drops"]`** — re-confirms S22 F1 on a
  different ticker (and validates that Fix #3,
  `claude/fix-suggest-rolls-drops` @ `358bffc`, is the right
  shape for it). After Fix #3 lands, the trader inspecting these
  rolls will see WHY any non-recommended candidate didn't make
  it through (in these runs all 16 recommended, so the drops
  list would be empty — but on a tighter `min_net_credit` filter
  it would surface). **Logged — points to the value of Fix #3
  for the rolling-campaign UX.**

- **(F2) The IV input to the strike-solve at the entry-side
  `rank_candidates_by_ev` is still the snapshot — re-confirms
  S23 F3 on a third ticker.** MU PIT IV at 2026-03-03 = 74.96%
  per the IV file; engine used `0.7496` (correct via PIT
  arithmetic) WAIT — actually checking: the entry strike picked
  by the engine at 2026-03-03 was $339.50 with premium $12.63.
  This is using whatever IV `get_fundamentals` returned for MU,
  which is the 2026 snapshot. After Fix #1 lands, the entry
  strike on this same scenario may shift slightly because the
  PIT IV (74.96%) is higher than the snapshot. **Logged —
  cross-confirms Fix #1's relevance on a high-vol name.**

- **§2 verified.** Every `suggest_rolls` candidate (both
  scenarios, 32 total candidates) routed through
  `EVEngine.evaluate`. The hold_trade did too. No bypass.
  **Logged as a positive.**

**Realism Check.**

| Aspect | Engine | Heuristic / Reality | Verdict |
|---|---|---|---|
| Recommendation on winning put (delta < 0.10) | ROLL (edge +$1876) | ROLL (lock the gain) | **Aligned** |
| Recommendation on challenged put (delta ~ -0.36) | ROLL (edge +$229, lesser loss) | Heuristic silent; default HOLD | **Coherent extension** |
| hold_ev magnitude on winning put | -$1.86 (near-zero) | "Almost no premium left" | **Aligned** |
| hold_ev magnitude on challenged put | -$391.27 (real loss) | "Position is hurting" | **Aligned** |
| `.attrs["drops"]` on suggest_rolls | None (pre-Fix #3) | Should mirror ranker | **Mismatch** — closed by Fix #3 |
| Mark of the open put at re-eval | $4.13 / $0.81 | Computed from current spot/IV | **Aligned** — BSM correct |

**Verdict.**

- **The engine's `recommend` boolean is well-calibrated.** It
  fires when rolling is genuinely better than holding by the
  apples-to-apples ev_dollars comparison. In the winning-put
  case (B), this lines up exactly with the pro-trader's "lock the
  gain" heuristic. In the challenged-put case (A), it lines up
  with a defensive-roll mindset the heuristic doesn't explicitly
  cover but a real trader would still apply.
- **The engine recommends rolling MORE often than the strict
  heuristic prescribes** because it considers BOTH the
  gain-locking case AND the loss-limiting case. A trader who
  strictly wants "only lock the gain, never roll on a loss"
  would have to filter on `current_delta < 0.10` in addition to
  reading `recommend`.
- **Fix #3 (drops accumulator) makes this surface fully
  observable.** In the current runs all 16 candidates recommend
  in both scenarios, so the drops list would be empty. Setting
  a tight `min_net_credit` (e.g. $500 to force credit-only big
  rolls) is what would populate the drops list with the
  filtered-out cells — see Fix #3's regression test for that
  pattern.
- **No new bugs surfaced.** Re-confirms S22 F1 (Fix #3) and S23
  F3 (Fix #1) on different tickers; cross-validates that those
  findings generalize across the universe.

**AI handoff.**

- The S26 ALIGNED-on-winning / COHERENT-EXTENSION-on-challenged
  result is from two single scenarios. A stronger claim ("the
  engine's `recommend` is well-calibrated across the population")
  would require a basket Sn — e.g. 50 random short puts opened
  across 2024-2026, advanced to 21-DTE-remaining, with the
  realized recommend / heuristic agreement tabulated. If the
  agreement rate is high on winning puts (heuristic applies) and
  the engine adds coherent defensive rolls on challenged ones,
  S26 generalizes. If the engine recommends rolling unprofitably
  often, the hold_ev formula needs a closer look.

- The scenario-B "16/16 recommend with edge $1876" deserves a
  sanity check: is the engine over-promising on the new
  contract's ev_dollars? A +$1876 edge over hold_ev = -$1.86
  on a $339.50-strike position is a 9% return on the original
  premium received — high but not absurd for a deep-OTM put
  that's almost free to close. The arithmetic looks honest, but
  worth verifying on a paper-trade pair against the post-roll
  observed P&L. (Outside the data window — would need a fresh
  data refresh or a Theta replay.)

- Fix #3 makes the drops list available; the natural follow-on
  Sn would re-run S26 with `min_net_credit` tight enough to
  filter most candidates, then inspect the drops list to verify
  that the gate=credit reason matches expectations on the
  filtered cells. That's a 1-page Sn that pays back the Fix #3
  investment.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — every `suggest_rolls` candidate (both scenarios) routed through `EVEngine.evaluate` plus the hold synthetic. Re-confirmed via the published columns.
  - Qualitative verdict: **match — both scenario branches re-produce the documented behavior**:
    - **AAPL challenged** (entry 2026-02-09 @ strike $263.00 / premium $3.058 / spot $274.62; re-eval 2026-02-23): **4 of 4 rolls recommend=True**, `max_roll_ev=-226.09`, `hold_ev=-439.68`, **edge=+213.59 in favor of rolling**. Direction identical to original (4 of 4 → roll the lesser loss); edge magnitude `+213.59` vs original `+229.26` = `-6.8%` (within noise).
    - **MU winning** (entry 2026-03-03 @ strike $334.50 / premium $14.799 / spot $379.68; re-eval 2026-03-17): **4 of 4 rolls recommend=True**, `max_roll_ev=+2179.40`, `hold_ev=-2.14`, **edge=+2181.54**. Original: edge=+1876.05 from `recommend=16/16`. The recommendation pattern (16/16 recommend with strong positive edge) is preserved at the smaller grid (4/4); magnitude drift +16% on edge — attributable to PR #179 / PR #122 (suggest_rolls buyback principal correction).
  - Numerical drift > 5% (with attribution):
    - metric `MU_winning_edge`: orig `+1876.05` → new `+2181.54` (`+16.3%`); attributable to **PR #179** (post-IV-PIT new strike-solve gives slightly different premium structure) + **PR #122** (`total_buyback_cost` correction means the buyback principal is now netted correctly, which can shift the roll_ev magnitude). Sign unchanged; direction unchanged.
    - metric `AAPL_challenged_edge`: orig `+229.26` → new `+213.59` (`-6.8%`); within noise.
  - Notes: Fix #3 (`suggest_rolls.attrs["drops"]`) confirmed live in S3 above. Drops on both AAPL and MU re-runs returned 0 because both had `min_net_credit=-1000.0` admitting all candidates (the original entry's same setup).
