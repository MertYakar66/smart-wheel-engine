# Heavy-verify — Full-wheel realized P&L: does the engine stay realistic through assignment→covered-call→recovery? (Mac terminal)

**Campaign:** #450 (W7, capstone of #436) · **Branch:** `claude/mac-w7-full-wheel-realized` · **Started:** 2026-06-28
**Terminal:** Mac (fresh clone, no real-premium rail) · **Mode:** validation / **measurement-only**
**Closes:** the put-leg-only caveat shared by W3 / W5 / W6 (#448) / the skew study (#449) / the rail audit (#447)
**Data path / provider:** `SWE_DATA_PROVIDER` unset → **bloomberg** (committed CSVs under `data/bloomberg/`); connector = `MarketDataConnector`.

> The decision trio (`ev_engine` / `wheel_runner` / `candidate_dossier`) is **never edited** and
> `EVEngine.evaluate` is **never bypassed**: the engine selects every short put
> (`rank_candidates_by_ev`) AND every covered call (`rank_covered_calls_by_ev`); this driver only
> does P&L *accounting* on the engine's own choices (CLAUDE.md §2). The machine-readable result is
> persisted to `docs/verification_artifacts/full_wheel_2026-06-28/w7_full_wheel.json` **before** any
> pretty-print. `PYTHONIOENCODING=utf-8` for all runs.

---

## §1 Why this is the capstone

Every measurement this campaign produced is **put-leg-only**: held-to-expiry CSP P&L,
`premium − max(0, strike − spot_at_expiry)`. That shares one caveat — it ignores the rest of the
wheel. The real strategy is a *cycle*: short cash-secured put → **if assigned, hold the shares** →
**sell a covered call** → called-away / roll / keep holding through drawdown-and-recovery. The
biggest realism question left is therefore: **does the engine stay realistic and net-positive across
the FULL wheel cycle, or does the assignment→covered-call→recovery leg change the verdict?**

The R11 dollar-impact study already warned this matters: put-leg "averted loss" numbers largely
*disappear* once assignments wheel into covered-call recovery
(`docs/verification_artifacts/r11_dollar_impact_2026-06-01/`). And I7 roll economics
(`docs/HEAVY_VERIFY_2026-05-31_I7_ROLL_ECONOMICS.md`) measured rolling beats holding **+$195/contract
(CSP-leg-only)** — a reference, not a full-wheel number. W7 closes the gap.

**Three questions (#450):**
1. Does full-wheel realized P&L stay positive, vs (a) put-leg-only, (b) capital-matched
   buy-and-hold of the same name/horizon, (c) the engine's ex-ante EV (`ev_dollars`)?
2. What does assignment actually cost — assignment rate by VIX-at-entry, covered-call cycles to
   resolution, and the realized recovery-leg economics?
3. Does the engine's EV ranking survive the leg it never directly scored —
   Spearman(`ev_dollars`, full_cycle_realized)?

---

## §2 Method — per-cycle accounting on engine-selected legs

**Driver:** `scripts/audit_full_wheel.py` → JSON `w7_full_wheel.json`. Rail-independent, read-only on
`engine/`; the trio is imported and *called*, never modified.

1. **Selection routes through the engine (§2 invariant), both legs positive-EV.** Each cycle's short
   put is the engine's own `rank_candidates_by_ev(as_of, universe_limit=None, top_n=40,
   min_ev_dollars=0)` — the tradeable positive-EV book. On assignment the covered call is the
   engine's own `rank_covered_calls_by_ev(ticker, shares_held=100, as_of=…, min_ev_dollars=0)` top
   pick by `ev_per_day` (both route through `EVEngine.evaluate`). The CC leg uses the **same
   positive-EV gate** as the put leg — a negative-EV covered call is an R1 block, and the simulator
   **never overrides it** (it holds the shares uncovered and retries next cycle), so no R1 block is
   bypassed. No heuristic CC selection; no EV rescue. **`ev_dollars` is put-leg-only ex-ante EV** —
   the recovery leg is neither scored nor forecast by the engine, so full-cycle-vs-EV comparisons
   *quantify the recovery leg's impact*, they do not measure "engine conservatism".
2. **Cycle accounting** (canonical `backtests/regression/_common` helpers, full friction):
   - Put **OTM** at expiry → keep premium; cycle done. (This number equals the W6 put-leg measure.)
   - Put **ITM** → **assigned**. The canonical put leg already marks the shares
     strike → `spot_at_put_expiry` (it carries the `−(strike−spot)·100` intrinsic), so the **stock
     leg continues from `spot_at_put_expiry`, not from the strike** — using the strike would
     double-count the put intrinsic. *(This was the one merge-blocking bug the Windows terminal's
     independent verification caught on the first PR push; see the correction note in §3. Equivalent
     to WheelTracker's basis=strike WITH a premium-only put leg; here the put leg carries the
     intrinsic, so the basis is the post-mark spot.)*
     Then repeatedly sell the engine's covered call and hold to its expiry: `spot > cc_strike` →
     **called away** (stock sold at `cc_strike`), cycle resolved; else keep the CC premium and keep
     holding (re-wheel), bounded by `CAP_CC_CYCLES=8` and the pre-splice frontier.
   - Residual shares at the cap / frontier → **marked to market** (`open_mark`; reported, never
     dropped — dropping would be survivorship). Results are shown **both** including and excluding
     open marks.
   - P&L uses `_forward_replay_realized_pnl` + `_spot_on_or_after` + the `friction_*` helpers
     (imported, not reimplemented). **Buy-and-hold** comparison = capital-matched: notional
     `strike×100`, shares `notional/spot_entry`, held entry→resolution.
3. **Grid:** monthly first-business-day 2020-01…2025-12, full universe, top_n=40 positive-EV puts
   per date. All cycle resolutions capped at the **pre-splice frontier 2026-03-20** (#439) — no
   split-scale contamination. Cycles bucketed by **VIX-at-entry** (calm ≤15 / elevated 15-25 /
   crisis >25), by the brief's named windows (2020-2024, 2021-2025), and by the genuinely **disjoint**
   regime halves (2020-2022 vs 2023-2025 — the independence check from W6).
4. **Reuse, not re-derive:** cites I7 roll economics + the R11 dollar-impact caveat; reuses the
   canonical helpers and the WheelTracker assignment/called-away conventions.
5. **Stats honesty:** cluster-bootstrap (over as_of) mean CIs + Wilson win-rate CIs; cycles are NOT
   i.i.d. (recurring names, overlapping cycles) — flagged. No SUPPORTED at n<~30. Assignment-rate
   and open-mark fractions reported per bucket so survivorship is explicit. Spearman uses a
   permutation p-value.

---

## §3 Results

> **Correction note.** The first PR push carried an **assignment-leg double-count** (the canonical
> put leg already marks the shares strike→`spot_at_put_expiry`, but the stock leg used `basis=strike`,
> re-counting the put intrinsic on every assigned cycle). The **Windows terminal's independent
> verification caught it** (a value-assert against hand-computed ground truth — `strike=100, prem=3,
> spot@expiry=90, cc=105, called-away@110` should net **+$1,000**, the bug returned 0). My own 4-lens
> adversarial review verified the formula *structure* but missed the double-mark — exactly the failure
> mode independent verification exists to catch. **Fixed** (`basis=spot_at_put_expiry`), a value-assert
> test added, and the full 2,009-cycle grid **re-run**. The fix raised assigned-cycle values most in
> stressed/deep-assignment regimes, so — as the reviewer predicted — the headline strengthened and the
> §3.4 recovery-edge regime *signs* reversed to positive. All numbers below are post-fix.

**2,009 full-wheel cycles** over 71 monthly entries, full universe, **0 forward-spot skips**
(survivorship nil; the ranker's 504-day-history + chain-quality gates pre-filter to names with forward
data). All P&L per cycle (≈ per contract / 100 shares), full friction. Resolutions: `put_otm` 1,582
(78.7%), `open_mark` 350 (17.4%), `called_away` 77 (3.8%).

### §3.1 Headline economics (calm/elevated/crisis pooled by window)

| Cut | n | full-cycle **mean** | mean CI95 | **median** | **win-rate** | resolved-only / incl-open-mark | put-leg-only | buy-and-hold |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| **pooled 2020-2025** | 2009 | **+$651.01** | [180, 1195] ✅excl 0 | **+$268.17** | **90.7%** | +$314 / +$651 | +$59.05 | +$925.48 |
| disjoint 2020-2022 (COVID+bear) | 1127 | +$533.09 | [−47, 1057] | +$277.01 | 89.6% | +$305 / +$533 | −$20.03 | +$589.90 |
| disjoint 2023-2025 (low-vol) | 882 | +$801.69 | [113, 1868] ✅excl 0 | +$262.11 | 92.1% | +$324 / +$802 | +$160.10 | +$1354.29 |

The full-cycle **mean is positive on every cut**, and on the **pooled** set (and 3 of the 5 cuts) the
cluster-bootstrap **CI excludes 0** — a stronger result than the put-leg-only campaign measures. The
2020-2022 half's mean CI still straddles 0 (the COVID/bear crash tail), but its **median is +$277** and
its win-rate 89.6%. The dollar mean is heavy-tailed, so the **median + win-rate remain the robust
central measures**.

### §3.2 The short-volatility profile (foregrounded)

Distribution of full-cycle P&L (pooled): p05 **−$2,054**, p25 +$128, median +$268, p75 +$559, p95
+$4,424, **min −$75,025 / max +$116,901**. The **90.7% of cycles that win average +$1,285; the 9.3%
that lose average −$5,529** — a frequent-small-win / rare-large-loss payoff. By resolution:

| Resolution | n (%) | mean | median | win-rate |
|---|---:|---:|---:|---:|
| `put_otm` (premium kept) | 1582 (78.7%) | +$443 | +$264 | 100% |
| `open_mark` (held to frontier) | 350 (17.4%) | **+$2,249** | +$932 | 59.4% |
| `called_away` | 77 (3.8%) | **−$2,342** | −$328 | 41.6% |

This profile is a dealbreaker for some traders and the whole point for others — **selection is
trader-specific risk tolerance, not an engine defect.**

### §3.3 Assignment economics — the engine *holds*, it doesn't wheel

Assignment rate **21.2%** (427 of 2009). The striking finding: **of the 427 assignments, 58% sell
ZERO covered calls** (mean **0.66 CC/cycle**). With both legs held to the engine's positive-EV gate
(§2), the engine **blocks most covered calls on freshly-assigned, depressed names as negative-EV** —
implicitly advising *hold for recovery* rather than cap it. That instinct is **strongly vindicated**:
positions **held uncovered** (`open_mark`, mean **+$2,249**) vastly outperform the cases where a CC was
sold and **called away** (mean **−$2,342**, 41.6% win, n=77 — selling an OTM call whose strike sits
below the assignment basis crystallizes a capped loss). So the realistic engine-driven wheel is mostly
*short-put income → on assignment, hold the stock → recover*, **not** covered-call wheeling.

### §3.4 The recovery edge is POSITIVE in every regime (magnitude regime-dependent)

Full-cycle minus put-leg-only (the value the recovery leg adds) by regime:

| Regime | recovery edge (full − put-leg) |
|---|---:|
| crisis-entry (VIX>25) | **+$813** |
| elevated 2023-2025 (bull) | +$730 |
| elevated (15-25) pooled | +$518 |
| calm (≤15) | +$222 |
| elevated 2020-2022 (2022 bear) | **+$188** |

The recovery leg is **net-additive in every regime, including the 2022 bear (+$188) and calm (+$222)**.
Its **magnitude** is strongly regime-dependent (larger when the market subsequently rises — crisis/bull
~+$730-813), but its **sign does not flip negative**. This **refines** the R11 dollar-impact caveat
rather than simply confirming it: the recovery leg's *value scales with post-assignment market
direction*, but over 2020-2025 it stayed positive — it did not turn net-costly even in the bear.
*(Pre-fix, the double-count made these edges read negative in calm/bear, which would have spuriously
"confirmed" R11; the corrected signs are positive — see the §3 correction note.)*

### §3.5 By VIX-band — crisis-entry is the best regime, calm-entry the weakest (but positive)

| Band | n | full-cycle mean | median | win | assignment rate | Spearman(ev,full) |
|---|---:|---:|---:|---:|---:|---:|
| calm (≤15) | 154 | **+$231.98** | +$206 | 85.1% | 0.273 | 0.339 |
| elevated (15-25) | 1197 | +$557.09 | +$252 | 88.5% | 0.251 | 0.428 |
| crisis (>25) | 658 | **+$919.94** | +$320 | 96.1% | 0.129 | 0.618 |

**Crisis-entry is the best full-wheel regime** (fat premiums + the *lowest* assignment rate, 13% + the
strongest recovery); **calm-entry is the weakest** (thin premiums + the *highest* assignment rate, 27%
+ the smallest recovery edge) — but **calm is now net-positive (+$232)**, not negative. The ranking
authority strengthens monotonically with stress (calm 0.34 → elevated 0.43 → crisis 0.62).

### §3.6 The wheel still underperforms buy-and-hold (gap narrowed)

A capital-matched buy-and-hold of the same names over the same horizons returns **+$925/cycle vs the
wheel's +$651** (and +$1,354 vs +$802 in the 2023-2025 bull). **The wheel remains an income /
capped-upside strategy, not a return-maximizer** — it gives up upside for premium and a higher
win-rate. (The gap narrowed sharply vs the pre-fix figure because the recovery leg, correctly counted,
recaptures more of the assigned-name upside.) A reader must not infer from "net-positive" that the
wheel beats simply holding the stock in a net-rising market; it does not.

### §3.7 EV-ranking authority — moderate, not authoritative

Spearman(`ev_dollars`, full_cycle_realized) = **0.492, p=0.0005** (n=2009): the engine's put-leg EV
**rank-orders full-cycle dollars** — the ranking *survives the recovery leg it never scored*. But
**rho²≈0.24**: only ~24% of full-cycle variance is explained; the other ~76% is post-assignment market
direction the engine cannot forecast. The permutation p is **optimistic under clustering** (recurring
names, overlapping cycles), so the honest read is a **moderate** signal, not authoritative. It
strengthens with stress (calm 0.34 → elevated 0.43 → crisis 0.62).

### §3.8 Honesty / robustness

- **§2 respected both legs.** Put and covered call both gated at `min_ev_dollars=0`; the simulator
  never overrides an R1 negative-EV block (it holds uncovered and retries). `ev_dollars` is *put-leg
  only* — comparing full-cycle realized to it *quantifies the recovery leg*, it does not show "engine
  conservatism" (the fair same-leg check: ex-ante put-EV +$105 vs realized put-leg +$59 → the engine
  **over-estimated the put leg by +$46**; the recovery leg then added +$592).
- **Open marks 17.4% (unrealized).** Reported, not dropped. They are net-*positive* here (+$2,249 mean
  — uncapped recoveries), so excluding them is conservative; resolved-only (+$314) and incl-open-mark
  (+$651) bracket the truth, and the win-rate/median (which count open marks by their mark) carry the
  central claim.
- **Frontier cap.** Cycles cap at the pre-splice frontier 2026-03-20 (#439); `CAP_CC_CYCLES=8`
  (≤280 days). Late-window / unrecovered names are frozen mid-flight and open-marked.
- **Net-rising-market confound.** 2020-2025 rose net; the recovery edge's magnitude reflects that.
  §3.4 shows it shrinks (but stays positive) in the 2022 bear — do not extrapolate the magnitude to a
  sustained bear, where it could in principle turn negative.

---

## §4 Verdict

> **ENGINE STAYS REALISTIC THROUGH THE FULL WHEEL — modal economics viable, dollar mean now CI-positive
> at the pooled level, short-vol tail, ranking holds at moderate strength. Closes the put-leg-only
> caveat for the #436 campaign. No engine rule warranted.**

Driving the **real strategy end-to-end** — every short put and every covered call selected by the
engine (`EVEngine.evaluate`, §2 intact, both legs positive-EV) — the full wheel is **net-positive**:
**90.7% of cycles win**, **median +$268/cycle**, mean positive on every window cut and **CI-positive on
the pooled set** (+$651, CI [180, 1195]) and 3 of 5 cuts. The engine's EV ranking **survives the
recovery leg it never scored** (Spearman 0.49, moderate). The put-leg-only measure the rest of the
campaign relied on is therefore **validated as a conservative floor** — the full cycle realizes much
more in aggregate (+$592 recovery edge).

Four caveats a one-line "net-positive" would hide:
1. **Short-vol tail.** ~9% of cycles lose ~$5.5k each (worst −$75k); the dollar mean is heavy-tailed
   (the 2020-2022 cut's mean CI still straddles 0). Win-rate + median are the robust central measures.
2. **The recovery edge's magnitude is regime-dependent** but its **sign stays positive in every
   regime** here (+$188 even in the 2022 bear) — this **refines** the R11 caveat (value scales with
   post-assignment direction) rather than confirming a net-cost; over a *sustained* bear it could in
   principle reverse, which 2020-2025 cannot test.
3. **The engine manages assignments by holding, not wheeling** (58% of assignments sell no covered
   call); held-uncovered (+$2,249) vastly beats sold-and-called-away (−$2,342). The engine's blocking
   of marginal covered calls is **vindicated**.
4. **The wheel underperforms buy-and-hold** (+$651 vs +$925) — income / capped-upside, not a
   return-maximizer.

**No engine rule.** The engine selects realistically and ranks with moderate authority; every regime —
including calm and the 2022 bear — is now **net-positive at the full-cycle level**, so there is no
systematic net-cost to gate. Calm-entry is the weakest regime but **positive (+$232)**; there is no
SUPPORTED net-cost anywhere (the campaign's no-SUPPORTED-below-30 / R11b discipline). **For the Windows
terminal (measurement, not a spec):** the magnitude regime-dependence of the recovery leg is worth
watching in a future sustained-bear sample; W7 does not support any rule today, and **no trio change is
implied by this campaign**.

---

## §5 Reproduction

```bash
# Full run (~40 min single process, or parallel collect→merge ~12 min):
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py            # inline
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py --dates "<chunk>" --collect-to recs0.jsonl
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py --analyze-from "recs*.jsonl"
# smoke:
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py --quick --limit 25 --top-n 15

# Methodology pins (fast) — includes the value-assert that pins assigned-cycle dollars:
pytest tests/test_w7_full_wheel.py -v
```

Artifact: `docs/verification_artifacts/full_wheel_2026-06-28/w7_full_wheel.json`.
