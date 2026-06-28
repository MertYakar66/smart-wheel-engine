# Heavy-verify — Data-wiring accuracy + engine output-realism reliability (Mac terminal)

**Campaign:** #436 · **Branch:** `claude/mac-data-wiring-reliability` · **Started:** 2026-06-27
**Terminal:** Mac (fresh clone, no real-premium rail) · **Mode:** validation-only (measure & report; no engine-behaviour edits)
**Data path / provider:** `SWE_DATA_PROVIDER` unset → **bloomberg** (committed CSVs under `data/bloomberg/`); connector = `MarketDataConnector`.

> Every claim below carries the data path and, where it is a statistical estimate,
> a confidence interval. Per-check JSON is persisted under
> `docs/verification_artifacts/data_wiring_2026-06-27/` **before** any pretty-print,
> so a console-encoding crash never loses compute. `PYTHONIOENCODING=utf-8` set for all runs.

---

## §1 Bring-up (green baseline)

| Step | Result |
|---|---|
| Python | 3.12.2 (repo targets 3.12) in `.venv` |
| Deps | `requirements.txt` + dev extras (`hypothesis`, `requests-mock`, `pytest-asyncio`) — the test extras live in `pyproject [project.optional-dependencies].dev`, not `requirements.txt` |
| Provider | `SWE_DATA_PROVIDER` **unset → bloomberg** (logged; silent selection is a known bug source). `type(WheelRunner().connector).__name__` = `MarketDataConnector` ✓ |
| 5-ticker smoke (CLAUDE.md §4.2) | healthy — `ev_dollars` / `iv` / `premium` all non-null. **4/5 rows** returned (JPM filtered out — investigated in W2). |
| Green baseline | `pytest tests/ -m "not backtest_regression"` (the per-PR CI lane, `.github/workflows/ci.yml`) = **3275 passed, 33 skipped, 17 xfailed, 0 failed** (349 s). |

**Note on the 4 backtest-regression failures.** A bare `pytest tests/` reports 4 failures
(`test_backtest_regression.py::…[s27/s32/s34/s35]`). These are the slow (S34 ~3.5 h) snapshot
reproducers that the CI lane **deselects** via `-m "not backtest_regression"` and that fail in this
sandbox on a missing optional `typer` dependency — **not a code regression**. The authoritative
green baseline is the CI-lane count above. Every PR in this campaign keeps that lane green.

---

## §W1 — Data-wiring accuracy audit

**Driver:** `scripts/audit_data_wiring.py` → JSON: `w1_ohlcv.json`, `w1_vol_iv.json`,
`w1_treasury.json`, `w1_generic_sources.json`, `w1_summary.json`.

### Per-source pass/fail table

| Source | Connector accessor | Rows / coverage | Monotonic | NaN / sentinel | Units | Verdict |
|---|---|---|---|---|---|---|
| OHLCV | `get_ohlcv` | 2117 rows/ticker, 2018-01-02→2026-06-04 | ✅ per ticker | 0 NaN/non-positive close (sample); OHLC invariant `high≥max(o,c,l)` holds 0 violations | split-adjusted $ | ⚠️ **PASS except split-scale defect (BKNG, CVNA)** |
| vol/IV | `get_iv_history` | 1,037,278 raw rows; 2,062,702 served IV cells | n/a | raw carries 17 sub-3 + 5,910 NaN cells; **served band 100% clean** — every served IV ∈ (3.0, 10000] | PERCENT | ✅ PASS |
| Treasury / RFR | `get_risk_free_rate` | rate_3m 8,455 rows, **1994-01-03→2026-06-05** | ✅ | none | decimal (÷100, D20) | ✅ PASS |
| Dividends | `get_dividends` | AAPL 91 rows | ✅ ex_date | 1 negative `dividend_amount` (AAPL) — flagged below | $ | ✅ PASS (minor) |
| Corp actions | `get_corporate_actions` | AAPL 6 disruptive | ✅ eff_date | — | ratio/amount | ✅ PASS |
| Fundamentals | `get_fundamentals` | snapshot dict, 13 keys | n/a | — | IV PERCENT (`implied_vol_atm`=25.57), beta raw | ✅ PASS |
| Credit risk | `get_credit_risk` | dict: `sp_rating`, `altman_z_score`, `interest_coverage_ratio` | n/a | — | — | ✅ PASS |
| VIX | `get_vix` / `get_vix_regime` | 9,200 rows, 1990-01-02→2026-06-04, range [9.14, 82.69] | ✅ | — | index points | ✅ PASS |

### Defect list (copy-pasteable reproductions)

#### D-W1-1 — OHLCV split-scale discontinuity at the 2026-03-23 data splice (BKNG ÷25, CVNA ÷4.7) — **CONFIRMED**

The OHLCV monolith is a splice of an older split-adjusted pull (history) and a recent slice
(≥ 2026-03-23). Names that split **between** the two pulls carry the split adjustment **only on the
recent slice**, producing a one-day scale break at 2026-03-23 plus an early-application window where
served prices are post-split-scaled but the real market was still pre-split.

```python
from engine.data_connector import MarketDataConnector
c = MarketDataConnector()
b = c.get_ohlcv("BKNG")["close"]
print(b.loc["2026-03-20"], "->", b.loc["2026-03-23"])   # 4286.81 -> 175.87  (ratio 0.041 ≈ 1/25)
# BKNG 25:1 split EFFECTIVE 2026-04-06, but applied 2026-03-23 (~2 weeks early)
v = c.get_ohlcv("CVNA")["close"]
print(v.loc["2026-03-20"], "->", v.loc["2026-03-23"])   # 283.28 -> 59.92  (ratio 0.212 ≈ 1/4.7)
# CVNA 5:1 split EFFECTIVE 2026-05-08, applied 2026-03-23 (~6.5 weeks early)
```

- **Pre-2026-03-23** served prices are the *real* point-in-time (unadjusted) market prices.
- **From 2026-03-23** they are post-split-scaled.
- **Effect:** (a) an internal discontinuity — a fake **−96%** (BKNG) / **−79%** (CVNA) single-day
  return on 2026-03-23 that poisons any realized-vol / return / tail computation crossing that date;
  (b) a window `[2026-03-23, eff_date)` where served spot is **25× / 4.7× too low** vs the true
  PIT market, so a backtest with `as_of` in that window mis-scales spot → strike → premium → `ev_dollars`.
- **Reach:** only at historical `as_of` ∈ the affected window or backtests crossing 2026-03-23. The
  default ranker uses the latest frontier (2026-06-04), where the post-split scale is *correct*, so
  **current/live rankings are unaffected**.
- **Fix class:** **data regeneration** (re-pull OHLCV with consistent full-history adjustment, or apply
  the adjustment only from the effective date) — *not* a decision-trio fix and not a connector-behaviour
  edit this campaign is permitted to make. Filed as a GitHub issue for the rail/data owner. Pinned by a
  strict-xfail test in `tests/test_w1_data_wiring.py`.

#### D-W1-1b — NFLX ~10× scale — **REFUTED as a corruption**

NFLX 2018-01-02 served close = **$20.107** (real unadjusted ≈ $201.07). This is **not** a per-ticker
corruption: the **entire** NFLX series is uniformly back-adjusted ÷10 for the 2025-11-17 10:1 split
(no internal discontinuity — a universe sweep finds **zero** >2× daily jumps in NFLX). This is standard
Bloomberg split-adjusted convention and is internally consistent: strikes, premiums and spot all scale
together, so `prob_profit` / IV / returns are unaffected and only the *absolute* dollar magnitude of a
pre-2025-11-17 NFLX backtest is ÷10 — the same property every split in the dataset exhibits (e.g. AAPL
pre-2020 is ÷4). **Not a defect.**

#### Real-move discontinuities (NOT defects, recorded for completeness)

The universe sweep (511 tickers) found **6** names with a >2× single-day close move. Two are the split
artifacts above; the other four are genuine market events:

| Ticker | Date | Move | Cause |
|---|---|---|---|
| GL (Globe Life) | 2024-04-11 | −53% | Fuzzy Panda/Viceroy short-seller report |
| OXY | 2020-03-09 | −53% | Saudi-Russia oil price-war crash |
| TRGP | 2020-03-09 | −53% | same oil crash |
| PCG (PG&E) | 2019-01-14 | −52% | wildfire-liability bankruptcy announcement |

#### D-W1-2 — one negative AAPL `dividend_amount` row — **minor, flagged**

`get_dividends("AAPL")` has 1 row with a negative `dividend_amount`. Low-severity (dividends feed the
ex-div lockout / carry, not a hard EV gate), recorded for the data owner; not pinned.

### Verified properties (pinned green in `tests/test_w1_data_wiring.py`)

- Served IV band is always `(3.0, 10000]` (the `_clean_vol_iv_inplace` gate) on a sampled set — raw
  sub-3 / sentinel cells never leak to a consumer.
- The `134217.7` deep-IV sentinel and the 10000-floor: **0** occurrences in the committed monolith
  (the sentinel lives only in the deep/delisted gz panels, absent on this clone), and the served band
  is clean regardless.
- OHLCV post-rename invariant `high ≥ max(open, close, low)` holds with 0 violations on the sample.
- Treasury `rate_3m` coverage spans the full feasible OHLCV window (starts 1994 ≤ 2018).

---

<!-- W2–W5 sections appended by their respective PRs -->
