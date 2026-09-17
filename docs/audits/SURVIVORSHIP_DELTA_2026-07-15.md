# Survivorship-Delta Backtest — how much of the pinned edge is survivor bias?

**Audit ID:** SURVIVORSHIP_DELTA_2026-07-15
**Author:** RESEARCH-1 (read-only research terminal)
**CMD:** #494 · CMD 1 → RESEARCH-1
**Status:** DONE · analysis-only (no engine / decision-layer changes)
**Data provider:** `bloomberg` · connector `MarketDataConnector(deep_history=True)`

---

## 0. TL;DR verdict

The engine's **ranking skill survives** a survivorship-correct universe, but its
**backtested dollar returns do not.** Adding the three S&P 500 catastrophes a real
put-seller would have been assigned (SVB, First Republic, Signature) to the pinned
100-name survivor universe — same engine, same `backtests/survivorship.py` driver on
both arms — moves the results as follows:

| Lens | Arm A (survivor) | Arm B (survivorship-aware) | Δ | Survivorship illusion |
|---|---|---|---|---|
| **Ranking edge** — mean realized P&L / ranked candidate-day | **$78.57** | **$60.26** | **−$18.31** | **23.3% of the edge is illusory** (76.7% survives) |
| **Portfolio** — final NAV profit on $1M | **+$263,695 (26.4%)** | **+$105,696 (10.6%)** | **−$157,999** | **59.9% of the profit is illusory** (40.1% survives) |
| Ranker ordering — Spearman ρ | 0.348 | 0.338 | −0.011 | ~robust |
| Hit rate | 78.8% | 78.1% | −0.6 pp | ~robust |
| Ex-ante `ev_mean` | 36.39 | 36.41 | +0.02 | **unchanged — the EV model never saw it coming** |

**Headline number (CONFIRMED):** on a real-dollar basis, **~60% of the pinned
strategy's profit is a survivorship illusion**; on the ranking-quality basis the repo
uses to pin "edge," **~23%** is. The ranker still orders candidates well (ρ, hit-rate
barely move); what collapses is the *P&L*, because the engine repeatedly sold puts into
three names that were bleeding toward zero and its EV model priced them as *attractive*
the whole way down.

---

## 1. What was tested, and why it is a fair test

The pinned regression drivers (`backtests/regression/_common.py`) take a **fixed**
ticker list and never consult index membership, so they are survivor-biased by
construction: only names that survived to today can ever be selected. The 2022 S&P 500
contained ~505 names; ~74 later left the index, including the 2023 banking failures
**SVB (SIVBQ)**, **First Republic (FRCB)**, and **Signature (SBNY)** — names a real
cash-secured-put seller would have been **assigned** as they collapsed.

**Both arms run on the identical `backtests/survivorship.py` driver and the identical
engine** (`WheelRunner.rank_candidates_by_ev` → `EVEngine.evaluate`; no bypass, no
decision-layer edit). The only difference is the universe:

| | Arm A — survivor status quo | Arm B — survivorship-aware |
|---|---|---|
| Universe | `UNIVERSE_100` (the S34 100-name set; all survivors) | `UNIVERSE_100 + [SIVBQ, FRCB, SBNY]` |
| Everything else | identical | identical |

Config (matches **S34 `$1M`/100-name**, `backtests/regression/s34_universe_100t_1m.py`):
`capital=$1,000,000`, `2022-01-03 → 2024-12-31` (782 business days), `seed=42`,
`top_n=15`, `max_new_per_day=3`, `dte_target=35`, `delta_target=0.25`, `contracts=1`,
`friction_level="none"` (matches the pinned *frictionless* edge the CMD references).

Because both arms share a driver, the **A−B delta cancels** the known ~1% pinned-snapshot
`ev_mean` drift in this environment (CMD note) — confirmed: `ev_mean` is 36.39 vs 36.41.

### A methodological trap that was avoided
`ConsolidatedBloombergLoader.get_universe_as_of` returns an **alphabetically sorted**
list, and ~5 leading entries are numeric placeholder codes (`9903115D`, …). The naive
`max_universe=100` cap (`names[:100]`) would therefore slice off dead placeholder codes
and **exclude the catastrophes entirely** (their sorted indices are 415 / 200 / 409) —
producing a degenerate Arm B with a spuriously ~0 delta. Arm B here instead uses an
explicit fixed universe so the catastrophes are guaranteed present during their trading
life and correctly stale-skipped after delisting.

---

## 2. Data gate — CONFIRMED before any run

| Check | Result |
|---|---|
| `deep/sp500_ohlcv__delisted.csv.gz` | 2.38M rows, 1,015 tickers, 1990→2026. Real crash paths: SVB $340→halt 2023-03-10; FRC $115→$13.7→seized 2023-05-01; SBNY $275→halt 2023-03-10 |
| `deep/sp500_vol_iv__delisted.csv.gz` | IV present for all three **through** delisting (SIVBQ→2023-03-30, FRCB→2023-06-30, SBNY→2023-03-30) |
| PIT membership | `get_universe_as_of` = ~503–506 names for 2022–24, **includes** SIVBQ/FRCB/SBNY on 2023 dates |
| Deep connector pricing | prices the names 2022→2023-Q1, **stale-skips** them post-delisting (0 candidates by 2024-01 — no phantom trades) |
| `tests/test_survivorship_harness.py` | **5 passed** (`SWE_DEEP_TEST_DATA` set): Lehman/WaMu PIT inclusion; `terminal_spot(LEHMQ)` returns last close ~3.65 flagged delisted, not `None` |

> ⚠ The **non-deep** `sp500_vol_iv_full.csv` does *not* contain the delisted names —
> only the `deep_history=True` connector path resolves them. A non-deep probe
> misleadingly reports "no IV," which would (wrongly) look like a BLOCKED data gap.

---

## 3. Full delta table (CONFIRMED — live engine output)

```
metric                   Arm A (survivor)   Arm B (surv-aware)          Δ (B−A)
final_nav                  1,263,695.20        1,105,696.34        -157,998.87
spearman_rho                      0.3484              0.3377             -0.0108
hit_rate                          0.7877              0.7813             -0.0064
mean_realized                    78.5694             60.2641            -18.3053
ev_mean                          36.3880             36.4078             +0.0198
row_count (ranked)               10,914              11,088                +174
executed_trades                     148                 155                  +7
put_assignments                      52                  59                  +7
open_at_end                          53                  59                  +6
iv_mean                          0.3007              0.3106             +0.0099
```

Runtime: Arm A 7,008s / Arm B ~7,000s (≈9.0 s/day) on this box.

---

## 4. Tail decomposition — per-name realized P&L on the catastrophes (Arm B)

Forward-replay realized P&L summed across every ranked candidate-day (the same measure
that drives `mean_realized`). Arm A has **0** catastrophe rows (sanity-checked) — by
construction it never sees them.

| Name | | Cand-days | Settled @ delisting price | Σ realized | Mean / day | Worst |
|---|---|---:|---:|---:|---:|---:|
| SIVBQ | SVB Financial | 135 | 1 | **−$99,020** | −$733 | −$11,123 |
| FRCB | First Republic | 157 | 24 | **−$67,218** | −$428 | −$9,905 |
| SBNY | Signature Bank | 0 | — | **$0** | — | — |
| **Total** | | | | **−$166,238** | | |

Catastrophe mean realized/candidate-day = **−$569** vs rest-of-universe **+$77**.
Cross-check: portfolio NAV Δ (−$157,999) ≈ tail Σ (−$166,238); the residual is
base-trade displacement (Arm B opens 7 more trades as the bank puts compete for the
`max_new_per_day=3 / top_n=15` slots).

### 4a. The damage is a slow bleed, not the headline delisting jump (important)
Realized P&L on the three names, by quarter:

| Quarter | Cand-days | Σ realized |
|---|---:|---:|
| 2022 Q1 | 48 | +$5,748 |
| 2022 Q2 | 64 | −$2,788 |
| 2022 Q3 | 55 | **−$48,173** |
| 2022 Q4 | 47 | **−$88,200** |
| 2023 Q1 | 59 | −$22,910 |
| 2023 Q2 | 19 | −$9,915 |

**~82% of the loss lands in 2022 H2 — SVB's rate-driven decline — not the March-2023
collapse.** Only 25 of 292 catastrophe positions actually settled at a *delisting*
price; the rest settled at ordinary market prices as the stocks fell (e.g. an SVB put
struck $340.5 on 2022-10-04 settling at $218.42; an FRC put struck $115 on 2023-02-23
settling at $13.69). The survivorship illusion is therefore **mostly the pre-delisting
bleed a fixed survivor list silently excludes**, with the delisting endpoint as the
final chapter — not a single terminal shock.

### 4b. Signature (SBNY) never traded — and why it matters
The ranker produced **zero** candidates for SBNY on every date tested (2022-03 →
2023-02). SBNY has OHLCV/IV only from 2021-10, so by 2022 it lacks the history the
forward-distribution path requires and is self-excluded. This is a **data-history
threshold, not risk skill** — had Signature carried the ~5y history SVB did, the tail
would be larger. The measured −$166k is thus a *floor* on the survivorship cost.

---

## 5. Why the EV model didn't catch it (mechanism)

`ev_mean` is **unchanged** (36.39 → 36.41) and `iv_mean` rises (0.301 → 0.311): the
engine saw the declining banks' **elevated implied vol as attractive premium** and rated
those puts *positive-EV* the entire way down. The downgrade reviewers that could have
intervened did not bind per-name in this path (R11 elevated-vol size-down is gated on a
market-wide `vix_level` > 25, not a single-name signal). So the ranker kept surfacing
bank puts as top-bin candidates while their realized outcomes were catastrophic — the
definition of a survivorship illusion: **the ex-ante distribution looked normal; only
the realized, survivorship-complete path reveals the loss.**

This is a *research observation about backtest interpretation*, not a proposed engine
change. The HARD INVARIANT and decision layer are untouched.

---

## 6. Verdict (answering the CMD)

- **Fraction of the pinned edge that is survivorship illusion:**
  - **~23%** on the ranking-edge metric (`mean_realized`: $78.57 → $60.26).
  - **~60%** on real portfolio profit (final NAV: +$263,695 → +$105,696).
- **What survives:** the *ordering* quality (Spearman ρ 0.348 → 0.338; hit-rate 78.8% →
  78.1%) — the ranker is not merely a survivorship artifact.
- **What does not:** the *return*. A survivor-biased 100-name wheel shows +26.4% over
  2022–24; the survivorship-complete version shows +10.6%. Selling cash-secured puts on
  a broad index without modeling assignment on the tail names overstates realized profit
  by ~1.6× here.

---

## 7. Provenance, scope, and honesty labels

- **CONFIRMED:** every number in §3–§6 is live output of the current engine on both
  arms; the loss positions were spot-checked against real 2022–23 price history
  (SVB $340→$210s through 2022; FRC $115→$13.69 in Mar-2023). NAV Δ and tail Σ
  independently corroborate.
- **CANNOT-VERIFY:** the ultimate provenance of the `bloomberg` CSVs as *licensed*
  Bloomberg data. The delisted price/IV trajectories match real market history to the
  dollar, so they are at minimum a faithful representation — but this audit cannot prove
  the source pedigree.
- **Not compared 1:1 to the S34 snapshot:** `survivorship.py` is a puts-only,
  forward-replay-settled driver; S34's `run_backtest_multi_friction` wheels into covered
  calls. Arm A's $1.264M ≠ the S34 pinned $1.345M for that reason (a ~6% *driver*
  difference, not survivorship). The A−B delta is valid because both arms use the same
  driver.
- **Scope:** three catastrophes added to a 100-name survivor base at `friction="none"`.
  Other 2022–24 S&P departures were premium acquisitions (no put-seller crash), so the
  crash-delisting tail is these three banks; SBNY under-contributes (§4b), so the effect
  is a lower bound. Full-index (~500-name) PIT both-arms would sharpen the estimate at
  ~5× the runtime.

---

## 8. Reproduce

```bash
export SWE_DATA_PROVIDER=bloomberg SWE_DEEP_TEST_DATA="$PWD/data/bloomberg"
python -m pytest tests/test_survivorship_harness.py -v          # 5 passed
# Arm A: run_survivorship_backtest(tickers=UNIVERSE_100, capital=1e6,
#   start="2022-01-03", end="2024-12-31", friction_level="none",
#   top_n=15, max_new_per_day=3, dte_target=35, delta_target=0.25, contracts=1)
# Arm B: same, tickers=list(UNIVERSE_100)+["SIVBQ","FRCB","SBNY"]
```

*Generated by RESEARCH-1 for #494 · CMD 1. Read-only; docs-only commit.*
