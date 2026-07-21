# Heavy-Verify Campaign 2026-05-31 — I7 (Wave 3): roll/management economics

**Investigation:** does the engine's `WheelTracker.suggest_rolls` recommendation beat
holding a challenged short put to assignment? (The wheel is a *management* strategy;
this is the first look at the management layer.)
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i7_roll_economics.py`
(reuses `campaign_lib.py` + the shared snapshots). **Raw:** `raw_output/i7_roll_economics_RAW.txt`.
**Status:** observe-and-document; `engine/` not modified. Sampled 1-in-6 challenged
puts (`I7_STRIDE=6`) for tractability; the population scan is unbiased across time.

---

## VERDICT

> **When the engine offers a credit roll, rolling a challenged short put beats holding
> to assignment by ~+$195/contract (95% CI [+53, +345], 87% win rate, t=2.62) — it
> avoids terminal assignment ~80% of the time and is positive in every regime
> (strongest in the 2022 bear and 2020 crash). The management layer adds value. BUT
> read the caveats: the engine offers a credit roll only ~26% of the time (its
> default discipline is NOT to roll for a debit), the `recommend` flag is not
> discriminative, and the roll's edge is partly a HORIZON artifact (rolling extends
> duration, so some of the gain is just extra time-to-recover in a net-rising market).**

Confidence: **medium-high** — n=168 paired, CI excludes 0, robust across regimes;
tempered by the horizon-mismatch confound (below) and BSM-synthetic buyback costs.

This is the campaign's first **positive operational finding about position management**
(complements I6-B's "the ranking adds selection value").

---

## Method

* **Population:** from the 74 monthly snapshots, every entered 25-delta/35-DTE short
  put is checked at its roll date (entry + 28 days, i.e. ~7-DTE). A put is
  **challenged** if the underlying is at/below the strike (ITM) then. 636 challenged
  puts (1-in-6 sample); `suggest_rolls` errored on 0.
* **Two arms, both measuring forward P&L FROM the roll moment** (the entry premium is
  sunk and excluded in both — a fair comparison):
  * **HOLD** = take the original put to its expiry = `−intrinsic_old × 100`.
  * **ROLL** = `(new_premium − new_intrinsic_at_new_expiry) × 100 − buyback_cost(old)`,
    where the new strike/DTE/premium come straight from `suggest_rolls` (which routes
    through `EVEngine.evaluate` — §2 preserved). Buyback of the old put is BSM at the
    roll moment.
* **Default arm** uses the engine's real discipline (`min_net_credit=0` → only
  credit rolls offered). A **permissive arm** (`min_net_credit=−∞`) forces a rescue
  roll on every challenged put to isolate the roll *mechanics* from the *discipline*.

## Headline — HOLD vs ROLL (paired, where a credit roll was offered)

| metric | value |
|---|---|
| n paired | 168 |
| mean HOLD realized | **−$328.84** |
| mean ROLL realized | **−$133.53** |
| mean (ROLL − HOLD) | **+$195.30**, 95% CI [+$52.56, +$344.75] |
| median (ROLL − HOLD) | +$81.85 |
| roll-wins rate | 86.9% |
| paired t | 2.62 |

The CI excludes zero → rolling, when offered, is a real improvement, not noise.

## The engine's discipline — it usually declines to roll

| of 636 challenged puts | count | share |
|---|---|---|
| engine offered ≥1 credit roll | 168 | 26.4% |
| engine offered NO credit roll (default min_net_credit=0) | 468 | **73.6%** |

**73.6% of the time the engine's discipline is to NOT roll for a credit** — it won't
pay a debit to "rescue" a loser by default. That is arguably correct behavior (chasing
debit rolls is how wheels blow up), but it means the +$195 benefit applies only to the
quarter of cases where a credit roll exists; in the other three-quarters you take
assignment (or override the default). The **permissive arm** (forcing a rescue roll on
634/636) still shows +$205/contract (CI [+128, +284], 65% win) — so the mechanics add
value even when forced — but 17.4% of rescue rolls end ITM *again* (deferral).

## Assignment / deferral (adversarial: does roll reduce risk or just defer it?)

Of 168 rolled puts: 52.4% would have been assigned at the old expiry; the roll's new
put *also* ends ITM 19.6% of the time (deferred, not avoided) → **roll avoided terminal
assignment in 80.4%**. Mean buyback cost paid at the roll moment: $333. So rolling
genuinely reduces assignment most of the time, but ~1-in-5 just kicks the can.

## Is the `recommend` flag predictive?

| flag | n | mean(ROLL−HOLD) | win% |
|---|---|---|---|
| recommend=True | 129 | +$188.38 (CI [+10, +373]) | 87.6% |
| recommend=False | 39 | +$218.20 (CI [+77, +420]) | 84.6% |

**No** — both groups benefit similarly (recommend=False slightly more, overlapping CIs).
The flag does not discriminate *when* a roll helps; the credit-roll *availability* is
the real signal.

## By regime and moneyness

| regime | n | mean(ROLL−HOLD) | win% |
|---|---|---|---|
| crash_2020 | 17 | +$136.6 (CI excl. 0) | 94.1% |
| bull_2021 | 15 | +$96.9 (CI spans 0) | 86.7% |
| bear_2022 | 44 | +$185.5 (CI excl. 0) | 100.0% |
| recovery_2023_2024 | 53 | +$133.6 (CI spans 0) | 77.4% |
| recent_2025 | 29 | +$414.2 (CI spans 0, high-notional) | 79.3% |

Positive everywhere; statistically clean in the bear and crash (where management matters
most). By moneyness: ATM (0-1% ITM) +$230 and shallow-ITM (1-5%) +$141 — both CIs
exclude 0; shallower challenges benefit more.

## Adversarial caveats (what would weaken this)

* **HORIZON MISMATCH (the main confound).** HOLD is measured to the *old* expiry;
  ROLL is measured to the *new, later* expiry (21-63 DTE beyond the roll). So ROLL has
  more time for the underlying to recover, and 2020-2026 was net-rising — part of the
  edge is simply "stay in longer in an up-market." The regime cut partly addresses this
  (bear_2022 roll still +$185, win 100% — the bear had sharp rallies the rolled puts
  rode), but a sustained one-way downtrend would punish duration extension. Treat the
  +$195 as an upper-ish bound, strongest where premium re-collection (not drift) drives it.
* **BSM-synthetic buyback cost** (no real Theta quote for the old put at the roll
  moment) — a modeling approximation, noted in the convention.
* **Sample:** 1-in-6 stride (168 paired). Directionally robust; the regime cells with
  CIs spanning 0 (bull, recovery, recent) are under-powered individually.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg I7_STRIDE=6 python -u docs/verification_artifacts/campaign_2026-05-31/i7_roll_economics.py
# I7_STRIDE=1 runs the full challenged population (slow: ~1h).
```
All numbers in `raw_output/i7_roll_economics_RAW.txt`.
