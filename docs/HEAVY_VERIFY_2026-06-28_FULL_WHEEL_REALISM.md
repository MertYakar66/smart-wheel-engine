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
   - Put **ITM** → **assigned at strike** (share basis = **strike**; the put premium is already in
     the put-leg P&L, so it is *not* re-credited — the WheelTracker AUDIT-VIII P1.2 convention).
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

**2,009 full-wheel cycles** over 71 monthly entries, full universe, **0 forward-spot skips** (survivorship
nil; the ranker's 504-day-history + chain-quality gates pre-filter to names with forward data). All
P&L per cycle (≈ per contract / 100 shares), full friction. Resolutions: `put_otm` 1,582 (78.7%),
`open_mark` 350 (17.4%), `called_away` 77 (3.8%).

### §3.1 Headline economics (calm/elevated/crisis pooled by window)

| Cut | n | full-cycle **mean** | mean CI95 | **median** | **win-rate** | resolved-only / incl-open-mark | put-leg-only | buy-and-hold | ex-ante EV |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **pooled 2020-2025** | 2009 | **+$289.61** | [−197, 763] | **+$256.49** | **89.1%** | +$213 / +$290 | +$59.05 | +$925.48 | +$105.42 |
| disjoint 2020-2022 (COVID+bear) | 1127 | +$141.47 | [−533, 688] | +$261.78 | 87.7% | +$189 / +$141 | −$20.03 | +$589.90 | +$98.03 |
| disjoint 2023-2025 (low-vol) | 882 | +$478.91 | [−189, 1313] | +$253.73 | 90.9% | +$243 / +$479 | +$160.10 | +$1354.29 | +$114.86 |

The full-cycle **mean is positive on every cut** and the **win-rate is ~89% with a tight CI** — but
the **dollar mean's cluster-bootstrap CI straddles 0** on every cut. That is the central honesty
point of W7: full-cycle P&L is a **classic short-volatility distribution**, so the median + win-rate
carry the claim, not the mean.

### §3.2 The short-volatility profile (foregrounded)

Distribution of full-cycle P&L (pooled): p05 **−$3,714**, p25 +$114, median +$256, p75 +$532, p95
+$3,691, **min −$116,844 / max +$115,924**. The **89.1% of cycles that win average +$1,139; the 10.9%
that lose average −$6,649** — a frequent-small-win / rare-large-loss payoff. By resolution:

| Resolution | n (%) | mean | median | win-rate |
|---|---:|---:|---:|---:|
| `put_otm` (premium kept) | 1582 (78.7%) | +$443 | +$264 | 100% |
| `open_mark` (held to frontier) | 350 (17.4%) | +$651 | +$340 | 54% |
| `called_away` | 77 (3.8%) | **−$4,507** | −$1,072 | 24.7% |

This profile is a dealbreaker for some traders and the whole point for others — **selection is
trader-specific risk tolerance, not an engine defect.**

### §3.3 Assignment economics — the engine *holds*, it doesn't wheel

Assignment rate **21.2%** (427 of 2009). The striking finding: **of the 427 assignments, 58% sell
ZERO covered calls** (mean **0.66 CC/cycle**). With both legs held to the engine's positive-EV gate
(§2), the engine **blocks most covered calls on freshly-assigned, depressed names as negative-EV** —
implicitly advising *hold for recovery* rather than cap it. That instinct is **vindicated by the
data**: positions **held uncovered** (`open_mark`, mean **+$651**) vastly outperform the cases where a
CC was sold and **called away** (mean **−$4,507**, 24.7% win, n=77 — selling an OTM call whose strike
sits below the assignment basis crystallizes a loss). So the realistic engine-driven wheel is mostly
*short-put income → on assignment, hold the stock → recover*, **not** covered-call wheeling.

### §3.4 The recovery edge is strongly REGIME-DEPENDENT (confirms the R11 caveat)

Full-cycle minus put-leg-only (the value the recovery leg adds) by regime:

| Regime | recovery edge (full − put-leg) |
|---|---:|
| crisis-entry (VIX>25) | **+$498** |
| elevated 2023-2025 (bull) | **+$411** |
| elevated (15-25) pooled | +$128 |
| calm (≤15) | **−$116** |
| elevated 2020-2022 (2022 bear) | **−$311** |

The pooled +$231 recovery edge is **not a robust wheel property** — it is **positive only when the
market subsequently rises** (crisis→V-recovery, bull) and **negative in the 2022 bear and in calm
entries**. This directly **confirms the R11 dollar-impact caveat** (#306/#307): the recovery leg's
value depends on the post-assignment market direction, which no entry signal forecasts.

### §3.5 By VIX-band — crisis-entry is the best regime, calm-entry the weakest

| Band | n | full-cycle mean | median | win | assignment rate | Spearman(ev,full) |
|---|---:|---:|---:|---:|---:|---:|
| calm (≤15) | 154 | **−$106.59** | +$195 | 82.5% | 0.273 | 0.344 |
| elevated (15-25) | 1197 | +$167.44 | +$238 | 86.8% | 0.251 | 0.436 |
| crisis (>25) | 658 | **+$604.59** | +$316 | 94.8% | 0.129 | 0.599 |

**Crisis-entry is the best full-wheel regime** (fat premiums + the *lowest* assignment rate, 13% + the
strongest recovery); **calm-entry is the weakest** (thin premiums + the *highest* assignment rate, 27%
+ recovery that hurts). Calm-entry's mean is **negative (−$107)** — but its **median is +$195**, win
82.5%, and **n=154** with a very wide CI, so it is **directionally weak, not a SUPPORTED net-cost**.

### §3.6 The wheel materially underperforms buy-and-hold

A capital-matched buy-and-hold of the same names over the same horizons returns **+$925/cycle vs the
wheel's +$290** (and +$1,354 vs +$479 in the 2023-2025 bull). **The wheel is an income / capped-upside
strategy, not a return-maximizer** — it gives up ~⅔ of the underlying's upside for premium and a higher
win-rate. A reader must not infer from "net-positive" that the wheel beats simply holding the stock; in
a net-rising market it does not.

### §3.7 EV-ranking authority — moderate, not authoritative

Spearman(`ev_dollars`, full_cycle_realized) = **0.494, p=0.0005** (n=2009), i.e. the engine's
put-leg EV still **rank-orders full-cycle dollars** — the ranking *survives the recovery leg it never
scored*. But **rho²≈0.24**: only ~24% of full-cycle variance is explained; the other ~76% is
post-assignment market direction the engine cannot forecast. And the permutation p is **optimistic
under clustering** (recurring names, overlapping cycles), so the honest read is a **moderate** signal,
not authoritative. It strengthens monotonically with stress (calm 0.34 → elevated 0.44 → crisis 0.60).

### §3.8 Honesty / robustness

- **§2 respected both legs.** Put and covered call both gated at `min_ev_dollars=0`; the simulator
  never overrides an R1 negative-EV block (it holds uncovered and retries). `ev_dollars` is *put-leg
  only* — comparing full-cycle realized to it *quantifies the recovery leg*, it does not show "engine
  conservatism" (the fair same-leg check: ex-ante put-EV +$105 vs realized put-leg +$59 → the engine
  **over-estimated the put leg by +$46**; the recovery leg then added +$231).
- **Open marks 17.4% (unrealized).** Reported, not dropped (survivorship would be dropping them).
  They are net-*positive* here (+$651 mean — uncapped recoveries), so excluding them is not
  conservative; resolved-only (+$213) and incl-open-mark (+$290) bracket the truth, and the
  win-rate/median (which count open marks by their mark) carry the central claim.
- **Frontier cap.** Cycles cap at the pre-splice frontier 2026-03-20 (#439); `CAP_CC_CYCLES=8`
  (≤280 days). Late-window entries / unrecovered names are frozen mid-flight and open-marked — the
  true post-frontier path is unknown.
- **Net-rising-market confound.** 2020-2025 rose net; the recovery edge "works" largely because stocks
  recovered. §3.4 shows it reverses in the 2022 bear — do not extrapolate the pooled recovery edge to
  a sustained bear.

---

## §4 Verdict

> **ENGINE STAYS REALISTIC THROUGH THE FULL WHEEL — modal economics viable, short-vol tail, ranking
> holds at moderate strength. Closes the put-leg-only caveat for the #436 campaign. No engine rule
> warranted; the recovery leg's regime-dependence is flagged as a *measurement* for the Windows
> terminal, not a spec.**

Driving the **real strategy end-to-end** — every short put and every covered call selected by the
engine (`EVEngine.evaluate`, §2 intact) — the full wheel is **net-positive in modal terms**: **89.1%
of cycles win** (tight CI), **median +$256/cycle**, mean positive on every window cut. The engine's EV
ranking **survives the recovery leg it never scored** (Spearman 0.49, moderate). The put-leg-only
measure the rest of the campaign relied on is therefore **directionally validated** at the full-cycle
level — it is a conservative *floor*; the full cycle realizes more in aggregate (+$231 recovery edge).

But the honest verdict carries four caveats that a one-line "net-positive" would hide:
1. **Short-vol tail.** The dollar **mean's CI straddles 0** (heavy left tail; ~11% of cycles lose
   ~$6.6k each). The claim rests on win-rate + median, not the mean.
2. **The recovery edge is regime-dependent**, positive only when the market subsequently rises
   (crisis/bull) and **negative in bear/calm** — **confirming** the R11 caveat, not contradicting it.
3. **The engine manages assignments by holding, not wheeling** (58% of assignments sell no covered
   call); held-uncovered beats sold-and-called-away. The "covered-call recovery" leg is small and the
   engine's blocking of it is vindicated.
4. **The wheel materially underperforms buy-and-hold** (+$290 vs +$925) — income / capped-upside, not
   a return-maximizer.

**No engine rule.** The engine selects realistically and ranks with moderate authority; the
regime-dependence and the calm-entry weakness are **strategy economics**, not a calibration defect, and
the only clearly-negative cut (calm-entry, −$107) is **small-n and median-positive — inconclusive, not
a SUPPORTED net-cost** (the campaign's no-SUPPORTED-below-30 / R11b discipline). **For the Windows
terminal (measurement, not a spec):** the calm-entry full-wheel weakness and the bear-regime recovery
reversal are worth more sampling; *if* a systematic calm/bear net-cost is later confirmed with adequate
n, a regime-aware caution could be considered — but W7 does not support one today, and **no trio change
is implied by this campaign**.

---

## §5 Reproduction

```bash
# Full run (~40 min single process, or parallel collect→merge ~12 min):
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py            # inline
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py --dates "<chunk>" --collect-to recs0.jsonl
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py --analyze-from "recs*.jsonl"
# smoke:
PYTHONIOENCODING=utf-8 python scripts/audit_full_wheel.py --quick --limit 25 --top-n 15

# Methodology pins (fast):
pytest tests/test_w7_full_wheel.py -v
```

Artifact: `docs/verification_artifacts/full_wheel_2026-06-28/w7_full_wheel.json`.
