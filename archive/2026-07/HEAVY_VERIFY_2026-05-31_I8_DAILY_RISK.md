# Heavy-Verify Campaign 2026-05-31 — I8 (Wave 3): daily-marked risk — a correction to I2

**Investigation:** rebuild the I2 wheel NAV path with **daily** marks to get the true
intra-month max-drawdown and daily-return Sharpe (I2 marked NAV only monthly).
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i8_daily_risk.py` (imports
I2's exact `Sim`/`Position` bookkeeping). **Raw:** `raw_output/i8_daily_risk_RAW.txt`,
`raw_output/i8_daily_risk_risk_slip1.0_mf.json`. **Status:** observe-and-document;
`engine/` not modified. Lead-verified (monthly reproduction matches I2 exactly).

---

## VERDICT — corrects I2's drawdown magnitudes

> **I2's central thesis holds in DIRECTION (the wheel is defensive — lower drawdown
> than buy-and-hold in every regime), but its drawdown MAGNITUDES were monthly-mark
> lower bounds that understated true intra-month path risk by 1.6×–8×. Size the wheel
> against a ~−20% intra-crash drawdown, NOT the −2.6% monthly figure. And the "¼ the
> drawdown vs passive" framing was partly an artifact of comparing the wheel's
> *monthly* DD to the index's *continuous* DD; apples-to-apples (both daily) the wheel
> takes ~0.4–0.6× the index drawdown — a real defensive edge, just smaller than the
> monthly framing implied.**

Confidence: **high.** The driver reuses I2's exact lifecycle (validated: monthly
reproduction = −6.60 / −2.59 / −0.84 and returns 4.49 / 9.08 / 40.2 match I2's table),
inserting a non-mutating daily valuation between snapshots.

---

## True daily vs I2 monthly max-drawdown

| regime | I2 monthly DD | **TRUE daily DD** | understatement | daily trough |
|---|---|---|---|---|
| crash_2020 | −2.6% | **−20.56%** | **7.97× (−18.0pp)** | 2020-03-23 (COVID bottom) |
| bear_2022 | −6.6% | **−10.31%** | 1.56× (−3.7pp) | 2022-03-07 |
| recovery_2023_2024 | −0.8% | **−4.03%** | 4.80× (−3.2pp) | 2023-10-27 |

**The smoking gun is 2020.** I2's monthly mark caught the book on 2020-04-01 — *after*
the V-recovery had begun — logging only −2.6%. The daily path shows the book fell
**−20.56%** (peak 2020-02-24 → trough on the exact COVID bottom), an assignment cascade
into the crash that recovered before the next monthly snapshot. Monthly marking hid
~18pp of realized intra-month pain. Worst single day **−9.14%**, worst 5-day **−15.99%**.
This is the same procyclicality I3-E found (the engine sold puts into the crash); I8
shows what it did to the *book*, not just per-contract.

## Wheel vs index drawdown — apples-to-apples (both marked daily)

| regime | wheel daily DD | cap-index daily DD | ratio |
|---|---|---|---|
| crash_2020 | −20.56% | −33.83% (2020-03-23) | **0.61×** |
| bear_2022 | −10.31% | −27.41% (2022-10-14) | **0.38×** |
| recovery_2023_2024 | −4.03% | −10.07% | 0.40× |

The index DDs match reality (COVID −33.8%, 2022 bear −27.4% — actual S&P 500). The
wheel's defensive edge is **real (~0.4–0.6× the index drawdown)** but **not the ¼
implied by I2's monthly framing.** I2's INDEX/doc claim is corrected accordingly.

## Sharpe/Sortino collapse under daily marks

Monthly marking smooths intra-month vol, inflating Sharpe 2–2.5× in stress regimes:

| regime | daily Sharpe | monthly Sharpe (I2-style) |
|---|---|---|
| crash_2020 | 0.70 | 1.78 |
| bear_2022 | 0.27 | 0.53 |
| recovery_2023_2024 | 2.71 (daily) | 3.66 |

Daily Sortino ≈ daily Sharpe. The honest risk-adjusted number is the daily one.

## Caveats (direction stated)
* **Option MTM is intrinsic-only** (same as I2), so daily == monthly exactly at
  snapshot dates. A short option's true buyback = intrinsic + time value ≥ intrinsic,
  so intrinsic-only marks NAV slightly **high** → the true daily drawdowns are if
  anything marginally **deeper** than reported (these are conservative lower bounds).
* Inherited from I2: survivorship-biased (up) benchmark, price-only index DD,
  ~30–45% modeled fills.

## Bottom line for a risk officer
The wheel is defensive, but **size it against the daily numbers: a −20% intra-crash
drawdown, not −2.6%.** The monthly backtest framing materially flattered the path risk.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i8_daily_risk.py
```
All numbers in `raw_output/i8_daily_risk_RAW.txt`.
