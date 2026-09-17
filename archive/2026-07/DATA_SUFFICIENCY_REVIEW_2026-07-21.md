# Data-Sufficiency Review — 2026-07-21

_Answers the operator question: **"Is the data we feed the engine enough — do we
need more data, more history, more datasets, wider ranges?"** Everything below
was verified from source (file:line), not from docs — the docs drift and two of
them are corrected herein. Method: 6 parallel source tracers + 3 adversarial
refuters + a completeness critic + 6 gap-fillers over `engine/`, `data/`,
`backtests/`, and the on-disk CSVs (frontier 2026-07-02, ~536 MB under
`data/bloomberg/`)._

---

## 0. The one-paragraph answer

**You do not need more datasets, and you do not need more history — with two
exceptions: a routine frontier refresh (19 days stale) and a targeted backfill
of ~18 truncated tickers (a pull *bug*, not a depth problem — WMT is currently
silently excluded from the universe).** What actually starves the engine is not
acquisition: on the live Bloomberg path, `ev_raw` is driven by exactly five
inputs (OHLCV, the composite ATM IV, the 3-month treasury rate, dividend
yield, and the earnings/corp-action lockouts), every candidate's premium is
synthetic BSM (`edge_vs_fair ≡ 0`, spread cost model synthetic ±5%), and the
capabilities built to fix that — real premiums, skew, dealer GEX — are starved
by **one local data-production job** (run `scripts/produce_option_premiums.py`
against the Theta larder on the laptop; the ranker wiring already landed in
#435/#463/#465) and **two wiring jobs** (skew off the already-pulled 16-year
5×5 IV surface; a stored-chain adapter for GEX). Most incremental Bloomberg
pulls would land on the 23-of-26 broad_pull panels and ~11 monoliths that have
**zero consumers**. And several EV-quality ceilings are pure code, immune to
any pull: the hardcoded 5-year forward-distribution window, the unreachable
EVT gate (with a live inversion that penalizes exactly the wrong names), the
D21 horizon inflation, D19 exit-cost omission, and the R7 VaR gate that can
never fire.

---

## 1. History & range — enough per method? (YES, with one real bug)

| Method | Window it actually uses | More history helps? |
|---|---|---|
| Forward distribution (`forward_distribution.py`) | **5.0y hardcoded** — `lookback_years=5.0` (`:73`) is not exposed by `best_available_forward_distribution` (`:315-323`); no call site can widen it | **No (live).** Disk already holds 8.5y (2018-01-02→2026-07-02); ~3.5y is already discarded by the trim (`:117-119`). |
| HMM regime (`wheel_runner.py:1992-1993`) | last **504** log-returns (`[-504:]`), refit inline, memoized | **No.** Older bars are structurally discarded. (Note: the 504/200 constants live in `wheel_runner`, not `regime_hmm.py`.) |
| EVT / POT-GPD tail (`ev_engine.py:505`) | needs `len(pnls) >= 200`; non-overlapping tier yields **exactly 35** scenarios at dte 35 (41 at 30, 27 at 45) | **No — count-bound, not years-bound.** Even uncapped 32y ≈ 229 scenarios, and `fit_gpd_tail` needs 15 exceedances at the 95th pct ⇒ effective floor ≈ **300** (`tail_risk.py:82-127`). Unreachable on the IID tier by construction. |
| IV-rank / percentile (`data_connector.py:620-653`) | 252d lookback | **No.** 8y on disk vs 1y needed. Not on the EV ranker path anyway. |
| VRP accessor (`data_connector.py:655-672`) | last row IV − `volatility_30d` | No history dependence. |

**The EVT inversion (live defect, not a data gap).** Names with 504–700 bars
fall to the `empirical_overlapping` tier (469–665 *autocorrelated*
pseudo-scenarios), which **does** clear the 200 gate — so the GPD fit and the
0.5× heavy-tail EV penalty (`ev_engine.py:567-568`) fire **only** for
short-history names on the least-IID samples, and **never** for full-history
names (35 IID scenarios). Live members of that band today: GEV(568),
SOLV(569), LIN(664), DASH(693), VLTO(693), COO(694). Block-bootstrap and
HAR-RV tiers are arithmetically unreachable above the 504-bar gate (reachable
sliver: horizons 41–59 with 100–119 bars — below the gate always).

**The real history problem is per-name truncation, a pull bug.** 18/515
normalized tickers sit below the 504-bar gate (`wheel_runner.py:1615-1623`,
drops recorded only in `df.attrs` — silent). Root causes (verified against
`sp500_index_membership.csv`):

- **Exchange-transfer truncation** — OHLCV starts at the NYSE↔Nasdaq transfer
  date: WMT (**141 bars**, from 2025-12-09), KMB (274), CPB (450), DPZ (375),
  PLTR (399), LIN (664), COO (694), ROP (751).
- **Ticker-rename split** — BK→BNY: `BK UN` ends 2026-03-20, `BNY UN` starts
  2026-03-23 with 71 bars; `normalize_ticker` merges exchange suffixes, not
  renames, so **the company is invisible under both tickers** (BNY fails the
  history gate, BK the 30-day staleness gate).
- **Index-add without lookback** — BNY, CASY, COHR, LITE, SATS, VEEV, VRT all
  start exactly 2026-03-23.

Fix: **backfill rows into the same monolith** (Terminal, trivial) — no code
change, no `SWE_DEEP_HISTORY`. This also pushes the 504–700-band names onto
the IID tier, defusing the EVT inversion for them.

**Deep-1994 / delisted archive**: EV-inert live (the 5y trim reaches only
2021-07). It matters for backtests only: as_of < 2023-01-02 windows are
truncated today (all of the S27/S32/S34 2022 leg; S35 2018-2020 can't rank
before 2020), and the delisted slices feed only the survivorship harness.
Its durable store already exists (git buffer branch
`deep-history/bloomberg-raw`); flipping it on is a declared re-baseline event
(`data_connector.py:194-196`). Nothing to pull.

---

## 2. What we pull that is EV-critical vs. dormant

**Live-EV (moves `ev_raw`/`ev_dollars`)** — `sp500_ohlcv.csv` (spot, forward
distribution, HMM, rv-widening), `sp500_vol_iv_full.csv` (the single composite
IV = mean(`hist_put_imp_vol`,`hist_call_imp_vol`) → strike solve, synthetic
premium, BSM fair, lognormal fallback — **one number moves all four**),
`treasury_yields.csv` (`rate_3m` only), `sp500_fundamentals.csv` (+
`broad_pull/dividend_pit` PIT override) for q / IV-fallback / R9 sector,
`sp500_dividends.csv` (CC early-exercise only).

**Live-lockout (remove-only)** — `sp500_earnings.csv` (thin: ~37/503 forward)
+ `broad_pull/per_name/sp500_snapshot_bdp.csv` overlay (closes it to ~90%;
PIT-gated, fails open with 45d alarm), `sp500_corporate_actions.csv`.
`broad_pull/macro_calendar` is wired but **default OFF**
(`use_macro_event_gate=False` in all four signatures — deliberate: whole-window
lockout on monthly prints would empty a 21–63 DTE book).

**Live-reviewer** — `vix_term_structure.csv` → `get_vix_regime` → R11
`vix_level` (and strangle timing). *Not* `sp500_vol_indices.csv`.

**Live but network-fetched, not pulled** — the credit multiplier
(0.80/0.92/1.0) is a **live FRED HTTP fetch** (BAMLH0A0HYM2,
`wheel_runner.py:1443-1457`), not `sp500_credit_risk.csv`. In the sandbox
FRED is proxy-blocked ⇒ always 1.0 with `credit_regime='unknown'` (3-way
ambiguous); on a dated as_of with FRED unreachable, `credit_regime()` raises a
TypeError that the broad `except` masks to 1.0; and the S27/S32/S34/S35 locks
neither pin nor neutralize FRED ⇒ **the committed baselines are
environment-dependent** (~12–16% of window days are stressed/crisis on the
on-disk HY-OAS proxy). The offline fix sits unconsumed on disk:
`broad_pull/macro_vol/credit_spreads.csv` (2004→2026-06-16).

**Dormant / dead (no consumer — do not refresh, do not re-pull):**
23 of 26 broad_pull panels (iv_surface, vol_term_rv, returns_micro,
options_sentiment, beta_shares, estimates_fwd/m, fundamentals_q/ext_q,
valuation_m, short_interest, macro_releases, all of macro_rates/*, all of
macro_vol/*); monoliths `sp500_liquidity.csv` (accessor has **zero** callers),
`sp500_macro.csv`, `sp500_vix_full.csv`, `sp500_analyst.csv`,
`sp500_vol_dvd.csv` (consolidated-loader dead-reads), `sp500_sector_etfs.csv`,
`sp500_institutional.csv`, `sp500_iv_snapshot_today.csv`,
`sp500_historical_fundamentals.csv`, `sp500_earnings_yf.csv` (deliberately
not consulted), `sp500_fundamentals_yf.csv`, `sp500_short_interest.csv.xlsx`.
`sp500_index_membership.csv` is backtest-only (survivorship universe).
`sp500_credit_risk.csv` is advisory-only (`analyze_ticker`/memos).

---

## 3. Built-but-starved — the real gaps, and what each actually needs

The headline "one per-strike-chain gap starves edge/VRP, skew, SVI, and GEX at
once" survives adversarial review **only as geography, not as prescription**:
the four capabilities need **three different unblockers**, and two of the four
are overstated.

1. **Real premiums / `edge_vs_fair` — DATA-PRODUCTION only (laptop, no
   Terminal, no code).** The full rail is wired flag-free end-to-end:
   `_resolve_real_premium` (`wheel_runner.py:236-364`, all three rankers) ⇄
   `get_option_premium*` (`data_connector.py:1107-1291`) ⇄
   `scripts/produce_option_premiums.py` (#435, hardened #463, regression-
   neutralized #465). Missing: the parquets. Action: on the laptop, run
   `python scripts/produce_option_premiums.py --tickers all --workers 4`
   against the Theta larder (`data_processed/theta/option_history/`, ~390M
   rows, 154 names, 2016→2026-06-17). **Coupling:** the rail's date-coherence
   guards demand quote-date == spot-bar date and refuse >7d-stale chains at
   live as_of — so live activation needs the **monolith frontier AND the
   larder frontier current together** (larder ends 2026-06-17 ⇒ also re-run
   `pull_theta_option_history.py` forward). What it revives: `edge_vs_fair`
   and the real spread/cost model. What it does **not** revive: skew, GEX,
   chain OI (the rail feeds only the premium swap; `chain_df` stays None).
2. **Skew multiplier — WIRING only.** `skew_slope` needs three IVs
   (`skew_dynamics.py:151-174`); the 16-year 5×5 surface
   (`broad_pull/iv_surface`, iv_{30…365d}_{90…110}, 2010→2026-07-02, loader
   exists, zero consumers) is sufficient raw material. Today `chain_df` comes
   only from `get_options`/`get_option_chain` — methods `MarketDataConnector`
   doesn't have — so `skew_mult=1.0` uniformly (`wheel_runner.py:2040-2150`).
   A small surface-fed adapter makes it live. EV-moving ⇒ ceremony-tier +
   re-baseline. The Bloomberg {80,120} wings pull stays deferred until this
   and (1) land (per the runbook, correctly).
3. **Dealer GEX / R6 — WIRING + one derived column.** The analyzer needs
   `{strike, option_type, open_interest, implied_vol}` and **recomputes all
   greeks internally via BSM** (`dealer_positioning.py:319,485-544`) — vendor
   greeks are NOT the blocker. Per-strike OI is already in the rail schema
   (`OPTION_PREMIUM_COLUMNS`); per-strike IV is derivable (BSM back-solve of
   mid, or surface interpolation). A `get_option_chain(ticker, as_of)` on
   `MarketDataConnector` serving rail-strikes + OI + derived IV would light
   up skew and GEX simultaneously (wheel_runner probes exactly that method
   name). Alternatively, live `ThetaConnector.get_option_chain` already
   returns iv/delta/OI on the laptop with Terminal up.
4. **SVI surface — moot until an architecture decision.** Zero decision-layer
   consumers (grep over the trio is empty); D9 keeps it off the EV path by
   design. Wiring it anywhere is a D9 reversal — decide the consumer first.
   (Data routes both exist: Theta per-strike larder, or
   `create_empirical_surface` off `vol_term_rv`.)
5. **PIT fundamentals — WIRING (the data largely exists).**
   `get_fundamentals(as_of)` applies PIT to **dividend_yield only**; the rest
   is a dateless snapshot. The advisor committee inherits structural lookahead
   via `analyze_ticker`'s dateless call (`wheel_runner.py:981`). Meanwhile a
   dated EV-field panel sits **staged but unintegrated**
   (`staging/fundamentals_pit/sp500_fundamentals_pit.csv`, monthly
   2015→2026-05, 503 names), plus four more dated panels on disk unconsumed
   (`historical_fundamentals`, `fundamentals_q`, `fundamentals_ext_q`,
   `valuation_m`). The W28 *dated ratings* pull is the only genuinely missing
   piece.
6. **Nelson–Siegel term structure — WIRING.** The map's premise was wrong:
   `vol_iv`'s 30/60/90/260d columns are **realized** vol, but
   `broad_pull/per_name/vol_term_rv.csv.gz` already carries
   `atm_iv_{30,60,90,180,365,730}d` daily — six IV tenors, loader present,
   zero consumers. Only 7/14d short-end tenors exist nowhere (tier-blocked).
7. **R7–R10 portfolio gates — WIRING + one bug.** R7 (VaR) is **structurally
   unfireable**: no production `PortfolioContext` builder populates
   `returns_data`/`correlation_matrix` (both computable from OHLCV on disk),
   and `check_var` hard-skips without them (`portfolio_risk_gates.py:870-878`).
   R8–R10 run on the cockpit flow but the held-book dict is keyed
   `"ticker"` while the gates read `"symbol"`/`"is_short"` ⇒ the held book
   contributes **zero** to stress/sector/single-name aggregation (real NAV +
   candidate-only notional are what's actually evaluated).

**Overstatement corrections to the old headline:** (a) first-order ATM VRP
already flows into `ev_raw` — the premium is priced at implied vol while the
P&L distribution is realized returns; what's missing is the *market-mid vs
fair* refinement and per-strike/skew-level VRP. (b) News multiplier is a
severed stub returning 1.0 **by design** (D18) — no data fixes it.

---

## 4. Missing-and-pullable — the Terminal list, by ROI

1. **Frontier bump 2026-07-02 → today** (all four live monoliths are 19d
   stale; the connector's own CRITICAL warning prescribes it, and freshness
   is the binding constraint on the premium rail too). **Corrected
   procedure** (the runbook §1 is stale — pins are 2026-06-04, not
   2026-03-20, and `pull_vol_iv.py` **does** exist since #472/14b21cb):
   `SWE_PULL_MODE=forward SWE_PULL_END=<today> python scripts/pull_{ohlcv,vol_iv,liquidity}.py`
   — `_bbg_panel.py` is merge-only (no regression risk); the one footgun is
   `pull_treasury_yields.py`, which **full-overwrites** with default end
   2026-06-05. Then the §1d re-baseline tail (4 frontier pins + 4 regression
   snapshots).
2. **Targeted OHLCV backfill of the 18 truncated names** (§1 above; includes
   WMT/PLTR/KMB/CPB/DPZ + the BK→BNY rename merge + the 2026-03-23 adds).
   Cheap, lands on existing code, un-drops household names, defuses the EVT
   inversion for the 504–700 band.
3. **Dated PIT fundamentals + ratings history (W28)** — the docs' own only
   substantive net-new pull; pairs with integrating the staged PIT panel.
4. **`sp500_earnings.csv` forward backfill — marginal** (snapshot_bdp overlay
   already covers ~90% of forward dates); keep the overlay's `asof` fresh
   instead.
5. **Deliberately NOT worth pulling:** total-return/dividend-adjusted OHLCV
   (**harmful** — raw price returns are the correct physical measure for
   terminal-price option P&L; TR-adjustment would upward-bias puts), extra IV
   tenors or deeper IV history (nothing reads them; term structure already on
   disk unconsumed), `sp500_vol_dvd.csv` refresh (dead-read), anything in
   §2's dormant list, VIX futures (already on disk twice), macro panels.

---

## 5. Blocked at the university tier — the ceiling pulls cannot fix

- **Bloomberg per-strike option chains** — OMON is manual screen-only; no
  BDS/BQL mass pull (5 independent in-repo authorities; the 2026-06-17
  Terminal session produced none). Additional subtlety: even if a chain CSV
  appeared, `_split_adjust_option_premium` assumes **raw Theta strikes** —
  Bloomberg's split-adjusted data would be double-adjusted. **Theta is the
  only chain path.**
- **Theta at this tier**: greeks/IV *history* 404 (back-solve required),
  corp-actions 404, no realtime stock quotes.
- **Bucket F (all-NaN at pull)**: borrow rate + SI %-of-float, CDS spreads,
  rating watch/outlook/change-date, ESG, DRSK PD, NFCI, 7/14d IV tenors,
  `BEST_PERIOD_END_DT`.
- **Authority caveat**: source the block list from
  `staging/BROAD_PULL_MANIFEST.md` (tested-and-failed census), **not**
  `DATA_POLICY.md` §2 — the "VIX futures UX1-UX7 tier-blocked on both
  providers" row is provably wrong (pulled twice via plain BDH; puller
  `staging/macro_vol/pull_vix_futures.py` is committed and repeatable).
  Treat the doubtful items (skew-grid legs, delta-bucket 25Δ IV, OIS leg,
  Moody's fields) as test-first, not asserted.

---

## 6. "We lack the data" vs. "we have it and nothing consumes it"

| Actually lacking (must acquire/produce) | Have it — nothing consumes it (wiring) |
|---|---|
| Option-premium parquets (produce from larder, laptop) | 5×5 IV surface → skew (16y on disk) |
| Fresh frontier ≥ 2026-07-21 (Terminal) | `vol_term_rv` ATM-IV term → NS/empirical surface |
| Truncated-name OHLCV backfill (Terminal) | `credit_spreads.csv` → offline credit multiplier |
| Larder forward extension past 2026-06-17 (Theta) | `macro_calendar` gate (wired, default OFF — needs day-of semantics, not whole-window) |
| Dated ratings history (W28, Terminal) | staged PIT fundamentals panel + 4 dated fundamentals panels |
| Per-strike IV (derive: BSM back-solve of rail mids) | `short_interest`, `options_sentiment`, estimates, macro_rates/macro_vol panels |
| — | OHLCV → R7 returns/correlation (gate unfireable today) |

**Docs corrections recorded here** (both load-bearing for planning): (i) there
is **no** `data/data_manifest.json` / `scripts/fetch_data.py` Drive-migration
mechanism in this repo — the deep archive's durable store is the git buffer
branch `deep-history/bloomberg-raw`; Drive holds a partial 2-of-12 mirror.
(ii) `BLOOMBERG_TERMINAL_NEXT_SESSION.md` §1 predates #472: the frontier bump
it describes as "next" already executed to 2026-07-02 on the lab Terminal, and
its "no vol_iv producer" claim is stale.

---

## 7. The honest ceiling — what no pull fixes (code/design)

- **Hardcoded 5y forward window** — widening it is the only way more history
  could ever matter live, and even then EVT stays unreachable (§1).
- **D21** — calendar DTE consumed as trading bars: every candidate's horizon
  ~46% too long; known, deferred to a re-baseline.
- **D19** — exit commission+slippage computed but never subtracted from
  `ev_raw`; EV mildly overstated by design deferral.
- **EVT gate design** — dead letter on the IID tier; fires on the non-IID
  tier (the inversion, §1).
- **R7 unfireable / R8-R10 held-book shape mismatch** (§3.7).
- **Credit multiplier** — network-dependent, silent-failing, TypeError on the
  dated offline path, un-pinned in the regression locks (§2).
- **News multiplier** — severed stub (D18), returns 1.0 by contract.

---

## 8. Priority queue (what to actually do, in order)

| # | Action | Kind | Where |
|---|---|---|---|
| 1 | Frontier bump via `SWE_PULL_MODE=forward SWE_PULL_END=<today>` (+ §1d re-baseline tail; avoid `pull_treasury_yields.py` overwrite footgun) | Pull | Terminal |
| 2 | Backfill the 18 truncated names into the monolith (WMT et al.; merge BK→BNY) | Pull (bug fix) | Terminal |
| 3 | Extend the Theta larder forward past 2026-06-17, then `produce_option_premiums.py --tickers all` | Produce | Laptop |
| 4 | Wire skew off the 5×5 surface (or a rail-backed `get_option_chain` that lights skew + GEX together) | Wiring (ceremony + re-baseline) | Code |
| 5 | Wire `credit_spreads.csv` as the credit-regime source (offline-deterministic, lockable; fixes the TypeError path and the environment-dependent baselines) | Wiring | Code |
| 6 | Integrate the staged PIT-fundamentals panel; thread `as_of` through `analyze_ticker`; W28 dated-ratings pull when at the Terminal anyway | Wiring + small pull | Code + Terminal |
| 7 | Fix R7 context builders (returns/correlation from OHLCV) and the R8-R10 held-book key mismatch | Wiring (bug fix) | Code |
| 8 | Deep-1994 restore from the buffer branch **only** when the next backtest re-baseline is scheduled | Restore | Laptop |

Not on the queue by intent: total-return series, extra IV tenors, vol_dvd
refresh, any dormant-panel refresh, VIX futures, SVI wiring (blocked on a D9
decision), macro-gate enablement (needs day-of semantics first).
