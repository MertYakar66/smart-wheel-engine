# Heavy-Verify Campaign 2026-05-31 — I5: Re-verify prior claims on the current engine

**Investigation:** do three load-bearing prior findings still hold on the post-#294
engine? Plus the net state of PR #294 on the EV-authority path.
**Driver:** `docs/verification_artifacts/campaign_2026-05-31/i5_prior_claims.py`.
**Raw:** `raw_output/i5_prior_claims_RAW.txt`. **Status:** observe-and-document; `engine/`
not modified.

---

## VERDICT

> **All three prior claims still hold on the current engine; the recent code-review
> campaign (PR #294) did NOT change the EV-authority decision path.** (A) MU covered
> call is still negative-EV despite a fat premium — confirmed with an updated number.
> (B) The ranker IV is point-in-time and materially affects EV — the fix is live;
> forcing the old stale-snapshot fallback moves EV by up to 2.6×. (C) The Bloomberg
> IV file still has zero skew — `put_iv == call_iv` for 100% of populated rows.

Confidence: **high** (each reproduced with its own code on real data).

---

## #294 net state on the EV path (grounding for "re-verify on the current engine")

The heavy-cycle findings (HT-B/D) were computed pre-#294. I confirmed with my own
`git diff 75eda2e..origin/main -- engine/ev_engine.py` that **the only EV-core
change is comment text** (the deferred-D19 note): `ev_raw = mean(pnls)` (:383),
`prob_profit = mean(pnls>0)` (:393), and `ev_dollars = ev_raw × regime_mult` (:538)
are byte-identical. D19 (exit costs into ev_dollars) and D21 (trading-bar horizon)
were **reverted** by `0cf88d1`; the CSP-downside term (744b196) lives in
`payoff_engine.py recommend_strikes` (a display surface), and the committee
p_otm/p_profit fix (e6dcd46) lives in `engine_api.py` (advisor surface). **So the
honest answer to "did the changes invalidate prior findings": no — they did not
touch the decision path.** Prior calibration findings are expected to reproduce
structurally (I1 confirms), and the value of the re-run is larger-n + Brier/ECE +
the real-spread P&L, not drift detection.

## Claim A — MU covered call ~25-delta: NEGATIVE EV despite a fat premium (CONFIRMED)

Re-run on the current engine (event gate off, since MU has 1-day-out earnings):

| as_of | spot | call strike | premium | iv | **ev_dollars** | prob_profit | cvar_5 |
|---|---|---|---|---|---|---|---|
| 2026-03-17 | 461.69 | 548 | $13.34 | 0.694 | **−$811.91** | 0.771 | −$19,961 |
| 2026-03-19 | 444.27 | 521 | $12.17 | 0.652 | **−$790.68** | 0.857 | −$19,568 |

Even collecting ~$1,334 of premium, EV is −$812 because the probability-weighted
left tail (assignment loss, cvar_5 ≈ −$20k) dominates. **`ev_dollars` is
probability-weighted P&L including tails, not premium-if-all-goes-well.** S25's
original figure was −$1,058 *pre*-PIT-IV-fix; the current −$812 reflects the PIT IV
(0.694 vs the old 0.649) and a slightly different 25-delta strike (548 vs 541) —
the claim holds, the number moved exactly as the PIT-IV fix predicts.

## Claim B — ranker IV is point-in-time (MECHANISM CONFIRMED LIVE)

A/B at `as_of=2022-06-01`: rank normally (PIT IV via `_resolve_pit_atm_iv`) vs
monkeypatching `connector.get_iv_history → None` to force the **pre-d26a8d6 stale
snapshot fallback**:

| ticker | iv (PIT) | iv (snapshot) | ev (PIT) | ev (snapshot) | sign flip |
|---|---|---|---|---|---|
| AAPL | 0.333 | 0.262 | −27.0 | −79.2 | no |
| MSFT | 0.304 | 0.266 | +155.6 | +97.8 | no |
| JPM | 0.300 | 0.322 | −13.4 | −0.7 | no |
| XOM | 0.321 | 0.319 | −73.0 | −74.1 | no |
| UNH | 0.287 | **0.432** | +153.4 | **+405.1** | no |

The PIT-IV resolution is **live** (current call sites `wheel_runner.py:1006/2300/2837`)
and differs materially from the stale snapshot (UNH IV 0.287 vs 0.432 → EV $153 vs
$405, a 2.6× swing). **No sign flips at this date** for these 5 names — the
originally-documented AAPL/MSFT flip was at a different as_of (near the 2026 snapshot
date, where the gap is largest). Honest nuance: the fix is active and materially
moves EV magnitudes; whether it flips a *sign* is date/name-specific. The claim's
substance (IV must be PIT or EV is wrong) holds.

## Claim C — Bloomberg IV file has no skew (CONFIRMED)

`data/bloomberg/sp500_vol_iv_full.csv`: of 1,361,615 rows, 1,353,901 have both
`hist_put_imp_vol` and `hist_call_imp_vol` populated, and **100.0000%** of those are
**exactly equal** (max |put−call| = 0.0). Skew is structurally dormant on Bloomberg
for two independent reasons: no asymmetry in the data **and** the skew overlay needs
a chain (`conn.get_options()`) the connector doesn't expose. Any engine output
involving put/call skew (`skew_multiplier`, `put_skew`) is a 1.0/empty pass-through.

## Reproduce
```bash
SWE_DATA_PROVIDER=bloomberg python docs/verification_artifacts/campaign_2026-05-31/i5_prior_claims.py
```
All numbers in `raw_output/i5_prior_claims_RAW.txt`.
