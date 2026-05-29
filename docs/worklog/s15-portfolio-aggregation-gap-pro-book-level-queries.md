---
id: S15
title: Portfolio-aggregation gap (pro book-level queries)
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** A sophisticated trader running a multi-position book asks
book-level questions — net Greeks (Δ/Γ/V/Θ), CSP collateral as % of
NAV, sector / single-ticker concentration, theta decay per day, VaR.
The campaign covered per-trade EV across 14 scenarios; this one asks
what `WheelTracker` (the only stateful book object on the live decision
path) can answer to a pro user, and grades each query
existing / reconstructable / structurally-missing / **unwired layer
exists**. The §2 question: do any of these queries open an EV-bypass
surface, or are they purely observability?

**Setup.** `SWE_DATA_PROVIDER=bloomberg`, `MarketDataConnector`,
`as_of=2026-03-20`, query date `now=2026-03-25`, 7-position book on a
$1M account: 3 SHORT_PUT (AAPL/XOM/CAT @ 95% spot, 35 DTE), 2
STOCK_OWNED (JPM/PG via put assignment at strike), 2 COVERED_CALL
(MSFT/UNH; short call @ 105% spot, 30 DTE). Spots and IVs from
`connector.get_ohlcv` + `get_iv_history`; premiums priced BSM-fair
(`norm.cdf` from `K * exp(-rT) * N(-d2) - S * N(-d1)`) to avoid the
synthetic-edge confound. Each pro query asked of `WheelTracker` + the
live `engine/*` surface and graded against the *unwired* portfolio
layer (`engine/risk_manager.py`, `engine/stress_testing.py`).

**Path.** `WheelTracker.positions: dict[str, WheelPosition]` is the
only book state the live decision path exposes. `WheelTracker` itself
has `cash`, `available_buying_power()`, `mark_to_market(now, prices,
current_ivs=...)`, `get_performance_summary()`, and the P4-shipped
`to_dict`/`from_dict`/`save`/`load` — and *nothing else portfolio-
aggregated*. The richer layer (`engine/risk_manager.py`'s `RiskManager`
+ `SectorExposureManager` + `HierarchicalRiskParity`, plus
`engine/stress_testing.py`'s Greeks ladders / scenario matrices) is
**not imported by any decision-layer file**.

**Status.** Done. §2 holds — the queries are observability only;
nothing in the path proposes a trade, so no EV-bypass surface opens.
The headline finding is **not** "the engine can't answer pro book-
level questions". It is "**the engine has the answers in an unwired
second layer**". `engine/risk_manager.py` already implements portfolio
Greeks, VaR (parametric / historical / covariance / Monte-Carlo with
optional jump diffusion), sector exposure, sector limits, HRP, Kelly
sizing; `engine/stress_testing.py` already implements full Greeks
ladders, scenario matrices, extreme scenarios. **None of it is
imported by `engine/wheel_runner.py`, `engine/wheel_tracker.py`,
`engine/ev_engine.py`, `engine/candidate_dossier.py`, or
`engine_api.py`** (`grep -r RiskManager engine/`,
`grep -r SectorExposure engine/` — verified). Its only live caller is
`dashboard/quant_dashboard.py`, which is **deprecated** per
`PROJECT_STATE.md` §4. A pro running a real wheel book through
`WheelTracker` is using the engine's *retail* surface; the
institutional layer beside it is orphaned.

**Findings:**

- **Numbers from the run** (7 positions, $1M, 2026-03-25). Cash
  $902,919, available BP $800,178, **CSP collateral reserved
  $102,741**, NAV (mark-to-market) $1,006,696, **collateral / NAV
  10.21%**, total notional $211,948 (21.1% NAV). Net **Δ +398.69,
  Γ -5.60, V -$211 per 1% vol, Θ +$104.80/day** (hand-derived from
  `option_pricer.black_scholes_all_greeks` per leg + stock-delta).
  Sector breakdown by notional: **Industrials 30.1% (CAT),
  Information Technology 29.1% (AAPL+MSFT), Financials 13.5% (JPM),
  Health Care 13.2% (UNH), Energy 7.2% (XOM), Consumer Staples 6.8%
  (PG)**. **Single-ticker > 5% NAV: CAT at 6.35%** ($63,899 notional
  / NAV $1,006,696). None of these aggregates were produced by a
  method on `WheelTracker` — all required hand-rolled assembly.

- **Q1–Q4 — net Greeks (Δ/Γ/V/Θ): UNWIRED LAYER EXISTS.**
  `engine/risk_manager.py:RiskManager.calculate_portfolio_greeks(
  positions, spot_prices) → PortfolioGreeks` returns delta, gamma,
  theta (per-day after the `/365` at line 354), vega, rho,
  delta_dollars, gamma_dollars via
  `engine/option_pricer.py:black_scholes_all_greeks`. Three reasons it
  doesn't help `WheelTracker` users today:
  (1) **schema mismatch** — `RiskManager` expects
  `[{symbol, option_type, strike, dte, iv, contracts, is_short, ...}, ...]`,
  while `WheelTracker.positions` is `dict[str, WheelPosition]` with
  `put_strike` / `put_entry_iv` / `put_expiration_date` keyed off
  `state`; no adapter exists in the engine.
  (2) **options-only** — the loop requires `option_type` and would
  crash/skip `STOCK_OWNED` positions; **stock-delta (`shares × 1`) for
  `STOCK_OWNED` / `COVERED_CALL` legs has to be added by the caller** —
  in the test book that's +200 delta out of the +399 total, so a naive
  pipe-through would understate long delta by ~50%.
  (3) **no live caller** — `grep RiskManager` across `engine/` and
  `engine_api.py` returns 0 hits in `wheel_runner.py`,
  `wheel_tracker.py`, `ev_engine.py`, `candidate_dossier.py`,
  `engine_api.py`; the only callers are `dashboard/quant_dashboard.py`
  (deprecated), `tests/test_risk_manager.py` /
  `tests/test_advanced_quant.py`, and the `engine/__init__.py`
  re-exports. **Logged.**

- **Q5 — cash / BP / NAV / collateral-NAV: PARTIAL EXISTING.**
  `WheelTracker.cash`, `WheelTracker.available_buying_power()`
  (cash-secured definition, post the P4 fix), and
  `WheelTracker.mark_to_market(now, prices, current_ivs=...)` cover
  cash, deployable BP, and NAV. CSP-collateral-reserved is
  `cash - available_buying_power` (one line); collateral / NAV is
  `(cash - available_buying_power) / mark_to_market(...)` (two lines).
  Both ratios — the ones a pro literally watches — are
  **reconstructable but not surfaced**. No
  `WheelTracker.collateral_to_nav()` / `WheelTracker.notional_to_nav()`.
  **Logged.**

- **Q6 — sector concentration: UNWIRED LAYER EXISTS + structural data
  gap.** `engine/risk_manager.py:SectorExposureManager` implements
  `calculate_sector_exposures(positions, portfolio_value)`,
  `get_sector_violations(positions, portfolio_value)`,
  `check_sector_limit(symbol, proposed_notional, positions,
  portfolio_value)`, and `suggest_diversification(...)`, with a
  **25%-per-sector default cap** that the test book's **30.1%
  Industrials** would breach. Three reasons unreachable from
  `WheelTracker`:
  (1) it takes the same option-dict shape as `calculate_portfolio_greeks`
  (same adapter gap as Q1–Q4);
  (2) **`WheelPosition` has no `sector` field** — nothing on the tracker
  carries sector at all, and `ev_row` from `rank_candidates_by_ev`
  carries none either;
  (3) `SectorExposureManager.__init__(sector_map=..., max_sector_pct=0.25)`
  requires an injected sector dict — `DEFAULT_SECTOR_MAP` is the
  fallback, but the canonical sector source in this repo is
  `data/bloomberg/sp500_fundamentals.csv:gics_sector_name`, whose
  ticker column is **Bloomberg-suffix format (`'AAPL UW Equity'`)** and
  requires `str.split()[0]` to normalize — no engine code does this
  normalization. `MarketDataConnector` exposes `get_fundamentals(ticker)`
  but no `get_sector(ticker)` shorthand. **Logged.**

- **Q7 — single-ticker concentration: RECONSTRUCTABLE; no enforcement.**
  `RiskLimits` (in `risk_manager.py`) carries `max_per_ticker_pct` as
  policy config, but `WheelTracker.open_short_put` /
  `open_covered_call` don't reference it — there's no concentration
  enforcement on entry. In the test book, **CAT lands at 6.35% NAV**
  with no flag raised. A pro running a "no single ticker > 5%" rule
  has to enforce it externally. **Logged.**

- **Q8 — total notional: RECONSTRUCTABLE.** Sum over
  `positions.values()` of `put_strike * 100` (SHORT_PUT) or
  `stock_shares * spot` (STOCK_OWNED / COVERED_CALL). No
  `WheelTracker.total_notional()`. **Logged.**

- **Q9 — theta decay $/day on the book: UNWIRED LAYER EXISTS.** Same
  path as Q1–Q4 — `PortfolioGreeks.theta` is per-day after the
  `/365` conversion at `risk_manager.py:354`. Net theta in the test
  book: **+$104.80/day**, +0.0104%/day on $1M NAV (positive — the
  book is net short premium, which is the wheel's design). **Logged.**

- **(bonus) VaR / CVaR + stress tests: UNWIRED LAYER EXISTS.**
  `RiskManager.calculate_var(...)` cascades parametric → historical
  (`>30 days returns`) → covariance (correlation matrix + per-asset
  vol) → Monte-Carlo (Cholesky + optional jump diffusion).
  `engine/stress_testing.py` exposes `greeks_stress_ladder` (P&L
  decomposition across spot moves), `greeks_scenario_matrix`, and
  `extreme_greeks_scenarios`. Same wiring gap — no live caller from
  the decision / tracking path. **Logged.**

- **`option_pricer.black_scholes_all_greeks` and per-Greek helpers
  (`black_scholes_delta` / `gamma` / `vega` / `theta`) are wired and
  in use** by `engine/dealer_positioning.py`,
  `engine/stress_testing.py`, and `engine/risk_manager.py` itself.
  The *per-leg* primitive is healthy; only the *book-level* assembly
  is unreachable from `WheelTracker`. **Logged.**

- **`WheelPosition` has no `contract_count` field — the tracker
  assumes one contract per position.** Per
  `WheelTracker.available_buying_power`'s own docstring: "The tracker
  is one contract — 100 shares — per position (`WheelPosition` has no
  contract-count field)". A pro book holding 5-contract positions
  can't represent that in the current schema without opening five
  separate positions per ticker (which the per-ticker dedupe in
  `open_short_put` would reject). Out of scope for this Sn's
  questions, but anyone wiring multi-contract aggregation has to add
  `contracts` first. **Logged.**

**§2 verdict.** Holds. None of the queries propose a trade, set a
verdict, or alter `EVEngine.evaluate`'s input — they all read the book
and emit observability. The unwired-layer finding is not an EV-bypass
surface; it's a missing pro-grade observability surface on top of an
existing tracker. R1–R6, the audit-invariant tests, and the
EV-authority gate are unaffected.

**AI handoff.**

- A natural follow-up — *not yet claimed* — is a small adapter:
  `WheelTracker.portfolio_greeks(spot_prices, now=None,
  risk_free_rate=0.04, current_ivs=None) → PortfolioGreeks` that (1)
  builds `RiskManager`-shape position dicts from `self.positions`
  (state-aware: `SHORT_PUT` → 1 put-leg dict; `COVERED_CALL` → 1
  call-leg dict; nothing for the stock leg here), (2) calls
  `RiskManager.calculate_portfolio_greeks`, (3) **adds stock-delta**
  (`shares × 1`) post-call for `STOCK_OWNED` and `COVERED_CALL`
  positions — `RiskManager` itself won't, and the test confirmed the
  stock leg dominates long-delta in a covered-call book.

- The sector wiring needs two pieces: (a) a `sector` field on
  `WheelPosition` populated at `open_short_put` time via either a
  `sector_resolver: Callable[[str], str]` callback or a connector
  method `get_sector(ticker)`; (b) a
  `WheelTracker.sector_exposure(spot_prices)` wrapping
  `SectorExposureManager.calculate_sector_exposures` using the per-
  position notional logic this Sn worked out (collateral for
  `SHORT_PUT`, `shares × spot` for `STOCK_OWNED` / `COVERED_CALL`).
  A `MarketDataConnector.get_sector(ticker)` helper would centralize
  the Bloomberg-suffix strip.

- The Q5 ratios (collateral / NAV, notional / NAV) are a five-line
  obvious wrapper on `WheelTracker`; if any of the above lands they
  should ride along.

- The deprecated `dashboard/quant_dashboard.py` *is* a live caller of
  `RiskManager` — a parallel usage test could check whether the
  primary Next.js dashboard (`dashboard/src/`) exposes portfolio
  Greeks / sector exposure / VaR via the API, or whether removing the
  CLI dashboard would silently kill those views. **Candidate for a
  future Sn.**

- A pro-account sizing test ($1M+ vs S4's $50k retail) would naturally
  re-encounter the concentration gap; the two questions are entwined
  and a single follow-up can serve both.

- **Re-verified 2026-05-26** by Terminal A (engine SHA `8a17b0b`).
  - §2 invariant: holds — observability layer only; no EV-bypass surface introduced.
  - Qualitative verdict: **partial — most of S15's "unwired" surfaces are now WIRED via D17.** Re-grepping the decision-layer sources on `main`: `PortfolioContext` is referenced in `engine/candidate_dossier.py` and `engine/wheel_tracker.py`; `check_sector_cap` is called in `engine/wheel_tracker.py`. The `take_snapshot` builder maps `WheelTracker.positions` (state-aware) into the option+stock-leg dicts `PortfolioContext` expects — closing the "schema mismatch" finding the original S15 named. **S21 confirms three of S15's six orphan surfaces are now mechanically wired** (`SectorExposureManager` via R-D17 hard-block; portfolio Greeks via `check_portfolio_delta`; VaR via R7); Kelly (gate 3) and HRP remain orphan in production.
  - Numerical drift > 5%: not applicable — S15 enumerated capability gaps, not numerical baselines.
  - Notes: `dashboard/quant_dashboard.py` is still the only live caller of `RiskManager` outside test surfaces (deprecated per PROJECT_STATE.md §4). HRP (`HierarchicalRiskParity`) still orphan. The S15 closure update list at the end of S21 enumerates the per-aspect status — see S21 re-verify below for the full state.

---
