# DATA ACQUISITION ROADMAP — Bloomberg pull-broadly catalog + gap/fix plan

_Compiled from a 6-category specialist sweep + the prior gap analysis; field mnemonics to be confirmed via Bloomberg FLDS._

**Compiled: 2026-06-09** · field mnemonics are best-effort — confirm each via Bloomberg `FLDS` before scripting a pull.

---

## 1. Mandate, invariant, and honest limits

**Pull-broadly mandate.** The Bloomberg terminal is FREE for this project and storage is effectively UNLIMITED. The owner's instruction is explicit: pull **broadly**. Any dataset with a plausible incremental benefit to a decision module — even a "nice-to-have" that only sharpens a diagnostic — is worth listing and, eventually, pulling. This document is the catalog: the prior critical-gap analysis (what blocks correctness *today*) plus an expansive 6-category specialist sweep (everything we could conceivably want).

**The §2 invariant — every input obeys it.** No tradeable candidate bypasses `EVEngine.evaluate`. Every dataset below enters the engine in exactly one of three §2-safe roles, and **nothing in this document can rescue a negative-EV trade**:

- **`evaluate-input-correctness`** — a correctness input *consumed inside* `EVEngine.evaluate` (risk-free rate, dividend yield, IV/skew, beta/correlation, realized vol). Fixing or enriching it changes the *value* of `ev_raw` because the engine was always supposed to use the right number; it never adds a rescue path.
- **`remove-only-gate`** — an event/lockout input to `event_gate.py` (earnings, FOMC/CPI/NFP, splits, special dividends, M&A). It can only *remove* a candidate from the tradeable set (lockout); it never makes a blocked candidate tradeable.
- **`downgrade-only` / `advisory-sizing`** — a reviewer or advisor signal (news/ESG sentiment, analyst revisions, credit ratings, short interest, dealer GEX, the advisor committee). It can downgrade a verdict (proceed → review/skip) or scale `ev_dollars` *down* via the dealer multiplier clamped to `[0.70, 1.05]` — it never touches `ev_raw` and never upgrades.

If a dataset's listed role is `advisory-sizing` or `downgrade-only`, wire it as a chained provider participant or a downgrade-only reviewer (per CLAUDE.md §2). Do not introduce a code path that converts a non-tradeable candidate into a tradeable one without a fresh `EVEngine.evaluate` call.

**Honest Bloomberg limits.** Pull broadly, but two structural ceilings cap what Bloomberg can give us at this tier:

- **No per-strike option chains.** `OMON` is a manual, screen-only export — there is no `BDS`/`BQL` mass-pull of the full chain. Per-strike open interest and the OI-weighted gamma profile that `dealer_positioning.py` needs **must come from Theta** (we already hold **~390M** EOD-chain rows 2016–2026 across the `option_history*` trees). **Caveat:** Theta's EOD history carries OHLC + bid/ask + **OI only — no greeks/IV** (those routes are 404/not-entitled, see `docs/THETA_ENTITLEMENT_RETEST_2026-06-17.md`); the GEX greeks must be **back-solved** from those chains. Anything in the tables below tagged `use-theta` / `have-already` reflects this.
- **The IV file is ATM-only.** `sp500_vol_iv_full.csv` carries ATM approximations (`hist_put_imp_vol`, `hist_call_imp_vol`), and on Bloomberg `put_iv == call_iv` for ~100% of rows — there is **no skew** in the IV file. True moneyness-by-strike IV (the dormant skew surface) comes from the `xxDAY_IMPVOL_yy.y%MNY_DF` field family via `BDH`, or from Theta per-strike chains. ATM term structure is robust; smile is not.

---

## 2. Tier 0 — critical gaps (do first)

These are the items the prior gap analysis flagged as load-bearing for *correctness today*. The PULL items are net-new data the engine is silently defaulting around; the FIX items are the highest-ROI zero-pull wiring. All are §2-safe.

| # | Item | Type | Engine consequence today | §2 role | Feasibility |
|---|---|---|---|---|---|
| T0-1 | **Deep 5×5 IV-surface archive → connector** | FIX (no pull) | Dormant skew. The gitignored deep moneyness×tenor archive exists; wiring it reactivates `skew_dynamics.py` with zero pull. **Highest ROI.** | evaluate-input-correctness | have-already |
| T0-2 | **Moneyness-IV-by-strike (re-activate skew)** | PULL P1 | Skew dormant on Bloomberg (ATM-only IV file). `BDH 30DAY_IMPVOL_75/90/110/125%MNY_DF` across tenors. | evaluate-input-correctness | BDH |
| T0-3 | **Dated dividend-yield (PIT)** | PULL P1 | `EQY_DVD_YLD_12M` lookahead: `get_fundamentals` has no `as_of` (`engine/data_connector.py`); BSM `q` uses a 2026 snapshot in historical backtests. | evaluate-input-correctness | BDH |
| T0-4 | **PIT dividend lookahead fix** | FIX (no pull) | Same defect as T0-3 from the code side — add `as_of` to `get_fundamentals` so the dated series is selected point-in-time. | evaluate-input-correctness | have-already |
| T0-5 | **Macro-event calendar (FOMC/CPI/NFP/PCE)** | PULL P1 | FOMC is currently **hardcoded**; CPI/NFP/PCE invisible to the gate. `BQL eco_release_dt` (query exists, §1 of `bloomberg_bql_pulls.md`). | remove-only-gate | BQL |
| T0-6 | **Deep UST curve ≥2018** | PULL P1 | `treasury_yields.csv` starts 2021-05; pre-2021 falls back to a hardcoded ~5% risk-free. `BDH USGG3M/6M/2YR/5YR/10YR`. | evaluate-input-correctness | BDH |
| T0-7 | **Corporate actions (restore, then re-pull)** | PULL P2 / FIX | `sp500_corporate_actions.csv` is **2 bytes (empty)** → split/M&A gate is inert. A populated copy exists off-main — **prefer restore** over re-pull; `BQL cac_*` (§2) backfills the rest. | remove-only-gate | BQL / have-already |
| T0-8 | **Theta per-strike OI → `DealerPositioningAnalyzer`** | FIX (no pull) | Dealer multiplier inert at 1.0. We already hold Theta chains; wire OI+greeks into the GEX calc. | advisory-sizing | use-theta |
| T0-9 | **Re-pull regressed dividends (CTRA/BK/LW/PAYC)** | FIX/PULL P2 | `sp500_dividends.csv` source regression: only 427/503 coverage. Data-layer re-pull. | evaluate-input-correctness | BDS |
| T0-10 | **Land D19 exit-cost + D21 DTE→trading-bars + recalibration** | FIX (no pull) | Two deferred decision-layer changes that must ship *together* with a mandatory S27/S32/S34/S35 re-baseline. | evaluate-input-correctness | have-already |
| T0-11 | **Pre-2018 OHLCV backfill from deep archive** | PULL P3 / FIX | OHLCV starts 2018-01-02; the 504-day survivorship gate rejects candidates until ~2020. Split-adj-vs-raw caveat; mandatory S27/S32/S34/S35 re-baseline. | evaluate-input-correctness | have-already |
| T0-12 | **Short-interest / borrow** | PULL P3 | Advisory sizing only — **no engine consumer yet** (`risk_manager.py` thresholds are hardcoded). `BQL eqy_short_interest` (§3). | advisory-sizing | BQL |
| T0-13 | **Implied-correlation `COR1M`** | PULL P3 | Portfolio copula correlation-regime input. `BQL` (§7). | evaluate-input-correctness | BDH |
| T0-14 | **VVIX / SKEW** | PULL P3 | **Already held** in `vol_indices.parquet` — verify completeness, no fresh pull needed. | advisory-sizing | have-already |

> **Honest non-feasibility (Tier 0 footnote):** per-strike chain OI for dealer GEX is **NOT feasible on Bloomberg** at this tier (OMON manual only). Source it from **Theta** (T0-8).

---

## 3. Volatility & options-derived

ATM term structure and the vol-index family are cheap, high-confidence `BDH`/`BQL` pulls; the smile (moneyness grid, put/call skew) is the highest-value net-new but bumps the ATM-only ceiling — confirm put-vs-call field existence via FLDS or fall back to Theta/OMON. Items already partially on disk are flagged in the Engine-use / notes column.

| Dataset | Bloomberg vehicle | Engine use | Benefit | Feasibility | §2 role |
|---|---|---|---|---|---|
| ATM IV term structure (7d/14d/30d/60d/90d/180d/365d/730d) | `xxDAY_IMPVOL_100.0%MNY_DF` per tenor | `option_pricer`, `skew_dynamics`, `regime_detector`, `forward_distribution` — true term structure for SVI/Nelson-Siegel; replaces static 30d/60d. Have 30d/60d already. | P1-high | BDH | evaluate-input-correctness |
| Moneyness IV grid (25d/10d put, 10d/25d call) × tenors | `30DAY_IMPVOL_{75/90/110/125}%MNY_DF`, repeat 60/90/180d | `skew_dynamics` — delta risk-reversals (25d put−25d call) and butterflies (10d−ATM); completes the skeleton. Smile family: `{50…140}%MNY_DF`. | P1-high | BDH | evaluate-input-correctness |
| Realized vol (HV) at 10/20/30/60/90/120/180/260d | `nDAY_HV` family | `regime_detector`, `tail_risk`, `forward_distribution` — IV−RV vol-risk-premium, regime bins, vol-cone backfill; 260d feeds the RV bootstrap. Have 30/60/90/260d. | P1-high | BDH | evaluate-input-correctness |
| IV Rank / IV Percentile | `IMPVOL_RANK` (BDP) or derive 252d rolling pct of ATM IV | `regime_detector` — percentile-driven position sizing; advisor (Buffett value × vol). Theta `get_iv_rank` exists as cross-check. | P1-high | BDP | advisory-sizing |
| VIX family (VIX, VVIX, SKEW, VXN, RVX, VXEEM, OVX, GVZ) | `… Index` `PX_LAST` (BQL §5) | `regime_detector`, `tail_risk`, `dealer_positioning` — SKEW tail regimes, VVIX vol-of-vol, sector/EM/oil/gold breadth. `sp500_vix_full.csv` holds 2015-01-02 → 2026-03-20 (16,955 rows) → optionally extend back to 2011. | P1-high | BDH | advisory-sizing |
| VIX futures term structure (UX1–UX7) | `UX1…UX7 Index` `PX_LAST` (BQL §4) | `regime_detector` — contango/backwardation; UX1 vs UX3/UX6 spread. `vix_term_structure.csv` lacks the futures curve today. | P1-high | BDH | advisory-sizing |
| Put-call IV skew by expiry (1m/3m/6m, ATM/10d/25d) | `xxDAY_IMPVOL_*%MNY_DF` put vs call legs **(verify via FLDS)**; else OMON weekly snapshots | `skew_dynamics` — skew level + momentum; steepening put skew → sell less premium. | P1-high | manual-OMON | evaluate-input-correctness |
| Liquidity / per-strike OI / bid-ask / greeks | Theta `/v3/option/snapshot/quote` (Bloomberg has no per-strike) | `wheel_runner`, `option_pricer`, `dealer_positioning` — strike liquidity, slippage, OI-weighted gamma/pinning. We hold ~390M Theta EOD-chain rows (OHLC+bid/ask+OI; greeks back-solved). | P1-high | use-theta | evaluate-input-correctness |
| Vol-of-vol (VVIX or smile-curvature std) | `VVIX Index` `PX_LAST`; or `std(IV across moneyness)` | `regime_detector`, `dealer_positioning` — VVIX>20 ⇒ unstable surface ⇒ size down GEX multiplier. | P2-medium | BDH | advisory-sizing |
| Historical vol cones (5/25/50/75/95 pct of RV) | Client-side rolling quantiles of the `nDAY_HV` series | `regime_detector`, `forward_distribution` — regime-relative RV percentile; outlier flag vs cone. Store as `vol_cones.csv`. | P2-medium | manual-OMON | advisory-sizing |
| Single-name IV indices (AAPLV/TSLAV/NVDAV…, GVIX) | `…V Index` `PX_LAST` **(verify via FLDS)** | `regime_detector`, `skew_dynamics` — high-confidence mega-cap regime signal; diagnostic, not load-bearing. | P3-nice-to-have | BDH | advisory-sizing |

---

## 4. Equity pricing & microstructure

Daily-bar enrichments (total-return, beta, ADV, shares-out, spreads) are cheap `BDH`/`BDP` and feed cost-model/sizing/copula correctness. Anything needing per-strike chains or tick/intraday tape routes to Theta or is computed client-side.

| Dataset | Bloomberg vehicle | Engine use | Benefit | Feasibility | §2 role |
|---|---|---|---|---|---|
| Total-return series (dividend-adjusted price) | `TOT_RETURN_INDEX_NET_DVDS` **(verify via FLDS)**; else `price × (1+div_yield)` | `forward_distribution` (log-return sampling, HAR-RV), `tail_risk` (POT-GPD) — removes artificial ex-div jumps vs price-only OHLCV. | P1-high | BDH | evaluate-input-correctness |
| Bid-ask spread history (EOD) | `BID`, `ASK` (BDP); `LAST_BID/LAST_ASK` **(verify via FLDS)** | `EVEngine` cost model (slippage), liquidity gating, `cost_model`. `sp500_liquidity.csv` lacks bid-ask. | P1-high | BDH | evaluate-input-correctness |
| Average daily volume (20/60/90d) | `PX_VOLUME` rolling, or `EQY_AVG_VOL_*D` **(verify via FLDS)** | Liquidity gating, position sizing (`max_adv_pct`). Have `avg_vol_30d`. | P1-high | BDH | evaluate-input-correctness |
| Turnover ratio history | `TURNOVER`/`EQY_TURNOVER` or `(vol×price)/mcap` **(verify via FLDS)** | Liquidity gating, market-impact scaling. Have `turnover` (verify definition). | P1-high | BDH | evaluate-input-correctness |
| Shares outstanding (PIT) | `EQY_SH_OUT` (BDH, Per=D/M) | `EVEngine`/`portfolio_risk_gates` single-name notional cap; must update on splits. | P1-high | BDH | evaluate-input-correctness |
| Beta (raw & adjusted; 1Y/2Y/5Y) | `BETA_RAW_OVERRIDABLE`, `BETA_ADJUSTED_OVERRIDABLE` (BDH) | `portfolio_copula` covariance, `portfolio_intelligence`, risk-gate beta filter. Have snapshot only; need history. | P1-high | BDH | evaluate-input-correctness |
| Per-name rolling correlation to SPX (20/60/252d) | SPX-member `px_last` (BQL §7) → rolling corr; or `COR1M/3M/6M` | `portfolio_copula` (Student-t CVaR), diversification score, regime corr-filter. Compute from OHLCV we already have. | P1-high | BQL | evaluate-input-correctness |
| Ex-dividend adjustment factor | `EXD_ADJ_FACTOR`; `DVD_HIST_ALL` (BDS) | `forward_distribution` ex-div isolation, `option_pricer` `q`, short-call assignment. Have `sp500_dividends.csv` ex-dates. | P1-high | BDH | evaluate-input-correctness |
| Split/spin-off adjustment (PIT share count) | `SPLIT_ADJ_FACTOR`; `CORP_ACT_EFFECTIVE_DT` (BDS) / BQL §2 | `forward_distribution` jump-removal, sizing recalibration. `sp500_corporate_actions.csv` empty. | P1-high | BDH | evaluate-input-correctness |
| Per-strike OI (& back-solved IV) | OMON (manual) or **Theta EOD chains 2016–2026 (have)** | `dealer_positioning` GEX/walls, `skew_dynamics` by-delta, `option_pricer` OI-weighted smile. *(Theta history = OI+OHLC; IV/greeks back-solved or from 2026 snapshots.)* | P1-high | have-already | evaluate-input-correctness |
| IV skew slope & term structure (ATM, 25d/10d) | `30/60/90DAY_IMPVOL_{90/100/110}%MNY_DF` (EXTRACTION_GUIDE §5) | `skew_dynamics` momentum, `option_pricer` vanna/charm/volga, `tail_risk`. ATM approximations in `sp500_vol_iv_full.csv`. | P1-high | BDH | evaluate-input-correctness |
| Realized vol (5/20/60/252d) | `5/20/60/252DAY_HV` (BDH) or local std | `forward_distribution` HAR-RV, `tail_risk` POT-GPD scaling, regime bins. | P1-high | BDH | evaluate-input-correctness |
| Market-maker inventory (gamma-weighted OI) | Theta per-strike OI+delta → `GEX=−Σδᵢ·OIᵢ`; or BQL `dealer_gex_total` (tier) | `dealer_positioning` gamma-flip forecast, `skew_dynamics`. **Have Theta; wire it.** | P1-high | use-theta | advisory-sizing |
| VWAP history (EOD) | `VWAP` (BDH) | Liquidity gating, `signal_context` timing, `cost_model` vs benchmark. Not critical for mid-market EV. | P2-medium | BDH | advisory-sizing |
| Effective spread (time-weighted) | `EFFECTIVE_SPREAD` **(verify via FLDS)**; or EOD (ask−bid)/mid | `cost_model` realized slippage, liquidity gating. Quoted spread first (cheaper). | P2-medium | BDH | advisory-sizing |
| Amihud illiquidity λ | None — compute `|ret|/$vol` from BDH(PX_LAST,PX_VOLUME) | Liquidity gating threshold, `cost_model`, copula adjustment. | P2-medium | manual-OMON | advisory-sizing |
| Leverage effect (return↔RV corr; realized skew/kurtosis) | None — compute from daily returns | `skew_dynamics` smirk forecast, `tail_risk` hazard, regime detection. | P2-medium | manual-OMON | advisory-sizing |
| Cross-asset vol correlation (stock↔index↔VIX) | BQL VIX/VXN/SPX → local rolling corr; `CORRVOL`/`VOLSWAP` **(verify via FLDS)** | regime HMM panic/crowding flag, `portfolio_intelligence`, `signal_context`. | P2-medium | BQL | advisory-sizing |
| High-resolution OHLCV (1m/5m/15m/1h) | BDH intraday if tier allows; else Theta | `dealer_positioning` intraday gamma, overnight-gap risk, block-trade impact. Tier-restricted; nice-to-have. **(verify via FLDS)** | P2-medium | BDH | advisory-sizing |
| Intraday return distribution (overnight vs intraday skew/kurt) | Theta intraday; or daily open-close gap proxy | `event_gate` overnight-jump risk, `dealer_positioning` resets, regime day/night split. | P2-medium | use-theta | advisory-sizing |
| Block-trade frequency & order-flow imbalance | Bloomberg OMON Trades tab (no mass-pull); Theta `OMON_IMBALANCE` equiv | `dealer_positioning` gamma-flip timing, order-flow analytics. Theta is the better source. **(verify via FLDS)** | P2-medium | use-theta | advisory-sizing |
| Price-impact coefficients (Almgren-Chriss λ) | None — estimate `λ` from fills or sector averages | `cost_model` slippage scaling, liquidity gating, size cap. | P3-nice-to-have | manual-OMON | advisory-sizing |
| Float % | `EQY_FLOAT_PCT` (BDP) | Liquidity gating (low-float = higher cost), dealer-flow effective shares. Static snapshot. | P3-nice-to-have | BDP | advisory-sizing |
| Tick size / lot size | `PX_TICK_SIZE` **(verify via FLDS)** | `cost_model` rounding slippage, quote rounding. US equity ≈ $0.01; rarely binding. | P3-nice-to-have | BDP | advisory-sizing |
| Microstructure noise ratio (HF vs LF vol) | None — compute from intraday bars | `signal_context` entry gate, smile calibration noise. Needs minute bars. | P3-nice-to-have | manual-OMON | advisory-sizing |

---

## 5. Fundamentals & estimates

Financial statements and estimates are mostly `BDH`/`BQL` and feed the **advisor committee** (Buffett/Munger quality) — almost all `advisory-sizing`. PIT discipline matters: pull `BEST_PERIOD_END_DT` / `BEST_FISPD_SHEET_DT` alongside every statement field to timestamp filing vs period-end and prevent lookahead. GICS and dividend-yield are correctness inputs the engine consumes directly.

| Dataset | Bloomberg vehicle | Engine use | Benefit | Feasibility | §2 role |
|---|---|---|---|---|---|
| Income statement — annual & quarterly (PIT) | `IS_REVENUE/OPERATING_INCOME/NET_INCOME/EBITDA/EPS…`, `BEST_PERIOD_END_DT` | Buffett/Munger quality, earnings-drift, sector scoring. Replaces hardcoded Buffett list. | P1-high | BDH | advisory-sizing |
| Balance sheet — annual & quarterly (PIT) | `BS_TOT_ASSET/CASH/LONG_TERM_DEBT/TOT_EQUITY/GOODWILL…` | Financial strength, leverage, capital-allocation critique. | P1-high | BDH | advisory-sizing |
| Cash-flow statement — annual & quarterly (PIT) | `CF_OPERATING/CAPEX/FREE_CASH_FLOW/INVESTING/FINANCING` | FCF yield, accruals quality (NI−FCF), drift. `FCF_YIELD` in snapshot. | P1-high | BDH | advisory-sizing |
| Earnings surprise history (actual vs est) | `IS_EPS`, `BEST_EPS_MEDIAN`, `EARN_EST_EPS_SURPRISE_PCT` | `earnings_drift` confidence scaling, EV CI calibration, event-buffer width. | P1-high | BDH | advisory-sizing |
| Forward consensus estimates (EPS/Rev/EBITDA 1–3y) | `BEST_EPS/REVENUE/EBITDA/TARGET_PRICE/ANALYST_RATING` | Buffett intrinsic-value anchor, revision-shift signal, quality verdict. PIT as-of estimates. | P1-high | BDH | advisory-sizing |
| Valuation multiples (P/E, P/B, EV/EBITDA, P/S, PEG) — history | `BEST_PE_RATIO`, `PB_RATIO`, `EV_EBITDA_RATIO`, `PS_RATIO`, `PEG_RATIO` | Margin-of-safety vs median, value-trap inversion, compression/expansion. Have PE snapshot. | P1-high | BDH | advisory-sizing |
| ROE / ROIC / ROA (+5y avg) | `RETURN_COM_EQY`, `ROIC`, `ROA`, `BEST_ROE_5YR_AVG` | Buffett moat (ROIC>10%), Munger capital-allocation, trend degradation. Have trailing ROE. | P1-high | BDH | advisory-sizing |
| Capex & FCF metrics | `CF_CAPITAL_EXPENDITURE`, `FCF_YIELD`, `FCF_CONVERSION_RATIO`, `CAPEX_AS_PCT_SALES` | "Real earnings," capital intensity, FCF stress. FCF/NI<0.8 = accrual risk. | P1-high | BDH | advisory-sizing |
| EPS — actual/estimate/forward (history & forecast) | `IS_EPS`, `BEST_EPS`, `BEST_EPS_GROWTH`, `BEST_EPS_REVISIONS_UD_3M` | `earnings_drift` trend, P/E anchor, `event_gate` lockout. `sp500_earnings.csv` exists. | P1-high | BDH | evaluate-input-correctness |
| GICS sector / industry group / industry / sub-industry | `GICS_SECTOR_NAME`, `…INDUSTRY_GROUP/INDUSTRY/SUB_INDUSTRY_NAME` | `portfolio_risk_gates` R9 sector cap normalization, copula cross-name corr. Have sector+group only. | P1-high | BDP | evaluate-input-correctness |
| Analyst estimate revisions & dispersion | `BEST_EPS_20D_REVISION`, `BEST_EPS_STD_DEV`, `EARN_REVISIONS_UP/DOWN_1YR_PERCENT` | Revision momentum (monotonic on put-seller win rate); dispersion = consensus weakness. | P2-medium | BDH | advisory-sizing |
| Profitability margins (gross/op/net) — Q&A | `GROSSMARGIN`, `OPERATING_MARGIN`, `NET_MARGIN`, `EBITDA_MARGIN` | Pricing-power/moat, cost-structure inversion, margin-compression flag. | P2-medium | BDH | advisory-sizing |
| Leverage & coverage | `INTEREST_COVERAGE`, `DEBT_TO_EBITDA`, `NET_DEBT_TO_EQUITY`, `LEVERAGE_RATIO` | Munger stress test (cover<2x distress; D/EBITDA>4x covenant), credit-risk downgrade. | P2-medium | BDH | advisory-sizing |
| Dividend yield / payout / coverage — history | `EQY_DVD_YLD_IND`, `BEST_DPS`, `DVD_PAYOUT_RATIO`, `FCF_PAYOUT_RATIO` | `option_pricer` time-varying `q`, yield-trap filter, ex-div lockout width. | P2-medium | BDH | evaluate-input-correctness |
| Book value & tangible BV | `BS_TOT_EQUITY`, `BS_TANGIBLE_BOOK_VALUE`, `BOOK_VALUE_PER_SHARE` | P/B×ROE compounder screen; asset-quality inflection (goodwill writedowns). | P2-medium | BDH | advisory-sizing |
| Sales/revenue growth (YoY, 5Y CAGR) | `REVENUE_GROWTH_1Y/5Y`, `SALES_GROWTH_FORECAST` | Growth sustainability, PEG, revenue stress; accel/decel = regime shift. | P2-medium | BDH | advisory-sizing |
| R&D spend & R&D/sales | `RESEARCH_AND_DEVELOPMENT_EXPENSE`, `RND_OVER_SALES`, `RND_GROWTH` | Moat durability (tech/pharma R&D>10%); falling R&D% = erosion. | P2-medium | BDH | advisory-sizing |
| Net-margin trend (3/5Y stability, CV) | `NET_MARGIN`, `BEST_NET_MARGIN_5Y_AVG` | Moat stability (low CV = quality), predictability for drift. | P2-medium | BDH | advisory-sizing |
| Accruals / earnings-quality score | `ACCRUALS_QUALITY_SCORE`, `BENEISH_M_SCORE` **(verify via FLDS)**; else compute | `earnings_drift` confidence, "real" earnings. High accruals = red flag. | P2-medium | BQL | advisory-sizing |
| Piotroski F / Altman Z / Beneish M | `PIOTROSKI_SCORE`, `ALTMAN_ZSCORE`, `BENEISH_MSCORE` **(verify via FLDS)**; else compute | Quality screen (F>7), distress early-warning (Z>2.99), manipulation flag. | P2-medium | BQL | advisory-sizing |
| Debt maturity schedule / refinancing risk | `DEBT_DUE_IN_1YR/2_5YR/AFTER_5YR`, `WEIGHTED_AVG_MATURITY` | Munger refinancing-cliff stress, `tail_risk` credit scenario. | P2-medium | BQL | advisory-sizing |
| Consensus forecast accuracy / bias | `BEST_EPS_ERROR_1Y/_PCT`, `BEST_FORECAST_BIAS` **(verify via FLDS)**; else compute | `earnings_drift` bias correction, advisor confidence, event-buffer tuning. | P2-medium | BQL | evaluate-input-correctness |
| Dividend safety / sustainability score | `DIVIDEND_SAFETY_SCORE` **(verify via FLDS)**; else payout×coverage×stability | Dividend-cut risk gate, suspension scenario, ex-div lockout width. | P2-medium | BQL | advisory-sizing |
| Asset quality (intangibles/goodwill/WC, DSO/DIO/DPO) | `BS_INTANGIBLE_ASSET/GOODWILL`, `WORKING_CAPITAL`, `DAYS_*_OUTSTANDING` | Impairment risk, M&A integration risk, WC cycles for cyclicals. **(verify via FLDS)** | P3-nice-to-have | BDH | advisory-sizing |
| Sector relative valuation (rel P/E, sector avg) | Compute `BEST_PE_RATIO` by `GICS_SECTOR`; `CONSENSUS_PE_SECTOR` **(verify via FLDS)** | Peer-relative cheapness, value-trap flag, sector rotation. | P3-nice-to-have | BQL | advisory-sizing |
| Valuation percentiles vs history (5/10Y rank) | `PE/PB/EV_EBITDA_PERCENTILE_5Y` **(verify via FLDS)**; else rolling quantiles | Cheap/expensive regime context, false-cheapness trap. | P3-nice-to-have | BQL | advisory-sizing |
| Share buybacks / dilution | `BUYBACK_AUTHORIZATION`, `SHARE_REPURCHASE_ACTIVITY`, `SHARES_BASIC_GROWTH` | Capital-allocation quality; EPS-from-buyback-only = red flag. **(verify via FLDS)** | P3-nice-to-have | BQL | advisory-sizing |
| Insider ownership & transactions | `INSIDERS_OWN_PCT`, `INSIDER_BUYING/SELLING_ACTIVITY` **(verify via FLDS)** | Skin-in-game, mass insider selling = red flag. Needs Form-4 parsing. | P3-nice-to-have | BQL | advisory-sizing |
| Implied cost of equity (from target price) | `BEST_TARGET_PRICE/_LOW/_HIGH` → compute IRR/Gordon | Intrinsic-value calibration, required-return gate. **(verify via FLDS)** | P3-nice-to-have | BQL | advisory-sizing |
| Organic vs acquisition growth | `ORGANIC_REVENUE_GROWTH`, `ACQUIRED_REVENUE_CONTRIBUTION` **(verify via FLDS)** | Organic-growth quality, M&A integration/bubble risk. Often IR/NLP only. | P3-nice-to-have | manual-OMON | advisory-sizing |
| Customer concentration | `TOP_CUSTOMER_REVENUE_PCT` **(verify via FLDS)**; else 10-K parse | Single-customer >30% risk; "what if top customer leaves?" | P3-nice-to-have | manual-OMON | advisory-sizing |
| Revenue by geography | `REVENUE_DOMESTIC/INTERNATIONAL_PCT` **(verify via FLDS)**; else 10-K segments | FX/geopolitical exposure, `tail_risk` FX scenario. | P3-nice-to-have | manual-OMON | advisory-sizing |

---

## 6. Credit, ownership & positioning

CDS, ratings, short interest, ownership, and dealer GEX are mostly `downgrade-only` / `advisory-sizing` — they soften EV or trigger the existing R6/R8/R9/R10 soft-warns, never rescue. Corporate actions are a `remove-only-gate`. Per-name beta and index membership are partly already on disk.

| Dataset | Bloomberg vehicle | Engine use | Benefit | Feasibility | §2 role |
|---|---|---|---|---|---|
| CDS spreads (1y/5y/10y) | `CDS_SPREAD_5Y` for members(SPX); BDH for history | `tail_risk` CVaR tail-dependence, R8 stress trigger (>200bps name / 50bps portfolio shock). Reprices faster than Altman-Z. | P1-high | BDH | downgrade-only |
| Credit ratings (S&P/Moody's/Fitch) + watch + outlook | `RTG_SP_LT_LC_ISSUER_CREDIT`, `RATING_WATCH`, `RATING_OUTLOOK` | Risk-regime tag, Buffett quality, HMM transitions; junk-grade → 0.85× ev_dollars (no hard refusal). Have `RTG_SP…` in `sp500_credit_risk.csv`. | P1-high | BDP | downgrade-only |
| Short-interest ratio & days-to-cover | `eqy_short_interest`, `eqy_short_interest_pct_of_float` (BQL §3) | `risk_manager` crowded-short, R10 single-name squeeze soft-warn (>25% float + >5% NAV). No `sp500_short_interest.csv` on disk (only a `.csv.xlsx` workbook). | P1-high | BQL | advisory-sizing |
| Institutional / insider / float % | `EQY_INST_PCT_SH_OUT`, `EQY_INSIDER_HOLDINGS_PCT`, `EQY_FREE_FLOAT_PCT` | R9/R10 concentration context, skin-in-game signal, roll-feasibility (float<60% illiquid). `sp500_institutional.csv` snapshot (~23KB). | P1-high | BDP | advisory-sizing |
| Earnings surprise & revision momentum | `best_eps`, `earn_est_eps_surprise_pct` (BQL §6) historical | `forward_distribution` regime conditioning, R5-style event gate (<10d + >2.5σ → review). Have `sp500_analyst.csv`/`earnings.csv`. | P1-high | BQL | downgrade-only |
| Corporate actions (splits/special divs/M&A) | `cac_announcement_date`, `cac_type`, `cac_ratio` (BQL §2) | `event_gate` early-assignment lockout, `binomial_tree` assignment fee, deal-arb. **File empty (2 bytes) — HIGH refill.** | P1-high | BQL | remove-only-gate |
| Per-name beta & correlation to SPX | `BETA` (BDP/BDH); or compute from OHLCV | `portfolio_copula` tail-dependence, idiosyncratic/systematic split, Kelly scale. Cheap to compute locally. | P1-high | BDP | evaluate-input-correctness |
| Bond-implied default probability (DRSK) | `DRSK_DERIVED_PROB_DEFAULT` (BDP); else CDS→PD transform | `tail_risk` CVaR conditioning, Taleb tail; R8 soft-warn if portfolio PD>0.5% + short-gamma. High value in credit cycles. | P2-medium | BDP | advisory-sizing |
| Index membership history & PIT weights | `idx_weight_date`, `idx_weight` (BQL); `INDEX_ADDITION_DATE` | `event_gate` add/remove dynamics (R5-style review <20d), composition gating, HMM rebalance flows. `sp500_index_membership.csv` (981KB) has history. | P2-medium | BQL | downgrade-only |
| Dividend aristocrat status & payout history | `DVD_HIST_ALL` (BDS); compute 5Y+ streak | Buffett/Munger quality; downgrade CC return if payout>80%; bias dividend-growers to lower-delta puts. Have `sp500_dividends.csv` (3.9MB). | P2-medium | BDP | advisory-sizing |
| Analyst ratings evolution & target drift | `best_analyst_rating`, `best_target_price`, `tot_analyst_rec` (BQL) | R8 soft-warn (downgrade <5d → 0.90×), regime input. Have snapshot (34KB) only; need history. | P2-medium | BQL | advisory-sizing |
| Crowded-short regime flag (portfolio-aggregate) | Aggregate `eqy_short_interest_pct_of_float` (BQL), rolling mean/std | `risk_manager` Kelly adjustment; R10 soft-warn (portfolio avg>12% + name>20% + 10% NAV). Tail-dependence amplifier in crises. | P2-medium | BQL | downgrade-only |
| Repurchase activity (buyback announcements) | `SHARE_BUYBACK_AUTHORIZATION_DATE`; `EQY_SH_OUT` YoY decline | Buffett/Munger capital-allocation signal, implied-support filter. Low real-time value for 20–60DTE. | P2-medium | BDP | advisory-sizing |
| Dealer GEX per-strike (SPX/SPY/QQQ) | `dealer_gex_total`, `dealer_gex_profile` (BQL §9, tier) | R6 soft-warn (short-gamma + strike at put wall → review). `dealer_positioning.py` already infers from Theta OI. Low incremental value. | P2-medium | BQL | downgrade-only |
| Sector rotation flows / ETF constituent weights | `weight` for members(XLF/XLK/…) (BQL §10) | R9 soft-warn (rebalance <15d + weight cut → 0.95×), regime rotation. Have `sp500_sector_etfs.csv` (1.5MB). | P2-medium | BQL | downgrade-only |
| Options flow (call/put ratio, flow direction) | `call_put_ratio`, `options_notional_flow` (BQL) **(verify via FLDS)** | Simons stat-arb advisory, `skew_dynamics` regime. Theta tape is superior; BQL is 15–30m delayed. | P2-medium | BQL | advisory-sizing |
| Borrow rate & availability | `EQUITY_SHORT_BORROW_RATE_NET`, `BORROW_COST` (BDP) | Ex-div assignment-delta flag, Buffett market-quality. Snapshot-only; advisory. | P3-nice-to-have | BDP | advisory-sizing |
| Idiosyncratic vs systematic variance | None — regress returns vs SPY, residual variance | Taleb tail (idiosyncratic = diversifiable), copula tail-dependence weighting. Compute locally. | P3-nice-to-have | use-theta | advisory-sizing |
| 13F holdings (top owners, concentration) | No BQL — SEC EDGAR 13F parse / DES portal | Co-investor presence, copula correlation bias. Quarterly 45-day lag; low decision value. | P3-nice-to-have | manual-OMON | advisory-sizing |
| Supply-chain relationships (SPLC) | SCAP screen / Cap IQ (no BQL) | `tail_risk` copula degradation in supply shocks, Taleb. Very low value for 20–60DTE. | P3-nice-to-have | manual-OMON | advisory-sizing |
| Regulatory / litigation risk | RISK screen / EDGAR (no field); use news pipeline | Taleb tail, sector contagion. `edgar_adapter.py` + `impact_scorer.py` already capture this. | P3-nice-to-have | manual-OMON | advisory-sizing |
| Management turnover (CEO/CFO changes) | No field — 8-K via `edgar_adapter.py` / news classifier | Buffett/Munger quality, leadership-void regime. Capture via news pipeline. | P3-nice-to-have | manual-OMON | advisory-sizing |

---

## 7. Macro, rates, FX & cross-asset

Rates curves and the economic-release calendar are correctness inputs (risk-free rate, lockout gate); vol/credit/correlation indices are mostly advisory regime signals. UST/SOFR curves and the macro calendar overlap Tier 0 — listed here for completeness with the full tenor/series breadth.

| Dataset | Bloomberg vehicle | Engine use | Benefit | Feasibility | §2 role |
|---|---|---|---|---|---|
| UST curve history (par/zero 1m–30y) | `BDH USGG{tenor} Index PX_LAST` 2015–2026 | Risk-free input to BSM (hardcoded 5% pre-2021); term-structure discounting, HAR-RV. `treasury_yields.csv` starts 2021-05. | P1-high | BDH | evaluate-input-correctness |
| OIS/SOFR term structure (1m–30y) | `BDH USSOC{tenor} Index` 2018+ | Post-LIBOR overnight risk-free for pricing/HAR-RV; CVaR discounting under stress. | P1-high | BDH | evaluate-input-correctness |
| Economic release calendar (FOMC/CPI/PCE/NFP/ISM/GDP/Retail/PPI) | `BQL eco_release_dt/event/actual/survey/importance` (§1) | `event_gate` macro lockout w/ per-event importance — fixes hardcoded FOMC + silent macro-miss. | P1-high | BQL | remove-only-gate |
| Fed funds rate decisions & path | `FDTR Index`, `ZQ` fed-funds futures (BDH) | `regime_multiplier` rate-regime, BSM risk-free calibration, FOMC guidance. | P1-high | BDH | evaluate-input-correctness |
| Volatility indices (SKEW/VVIX/VXN/MOVE/OVX/GVZ/RVX) | `BDH … Index PX_LAST` (partly BQL §5) | `regime_detector` tail/vol-of-vol/sector regimes (SKEW<100 crash signal). Partly in `vol_indices.csv`. | P1-high | BDH | advisory-sizing |
| IV term structure & calendar dynamics (30/60/90/120d ATM) | `BDH 30/60/90/120DAY_IMPVOL_100.0%MNY_DF` | `regime_detector`/`monte_carlo` contango vs backwardation. Have 30/60d (EXTRACTION_GUIDE §5). | P1-high | BDH | evaluate-input-correctness |
| Vol-of-vol (VVIX + realized) | `VVIX Index` (BQL §5); realized from VIX returns | `regime_detector` vol-of-vol classification; VVIX>30 = vega convexity. | P1-high | BDH | evaluate-input-correctness |
| SKEW tail-risk index | `SKEW Index` (BQL §5) | `regime_detector` tail regime + Taleb; cross-check `tail_risk` POT-GPD. Already on disk. | P1-high | BDH | evaluate-input-correctness |
| Corporate actions (splits/special divs/M&A) per SPX | `BQL cac_*` (§2) | `event_gate` lockout; `sp500_corporate_actions.csv` empty. | P1-high | BQL | remove-only-gate |
| Real yields & TIPS / inflation swaps | `USGG5REAL Index`, `USSW2/5/10 Index` (BDH) | `regime_detector` inflation-expectation regime; portfolio drift/CVaR vs real-yield shocks. | P2-medium | BDH | advisory-sizing |
| Fed funds futures implied path | `BDH ZQ{contract} Index` | `regime_multiplier` terminal-rate expectation; macro event-buffer calibration. **(verify ticker suffix via FLDS)** | P2-medium | BDH | advisory-sizing |
| Financial conditions (NFCI/MOVE/CVIX) | `BDH NFCI/MOVE/CVIX Index` | `regime_detector` tightening/easing; rates-vol degrades long-dated options. | P2-medium | BDH | advisory-sizing |
| Credit spreads (IG/HY OAS) | `BDH LUACOAS/LUACTRUU/LF98OAS Index` | Credit-regime tail context; equity-credit turn → regime downgrade. | P2-medium | BDH | advisory-sizing |
| Yield-curve slope (2s10s, 5s30s, 3m10y) | Compute from `USGG{tenor}` | Recession-warning inversion flag → lower notional caps; macro-event importance escalation. | P2-medium | BDH | advisory-sizing |
| Sector & factor ETF indices (XLK…XLRE) | `BDH {TICKER} US Equity OHLCV` | `portfolio_risk_gates` sector cap, defensive-sector regime flag, cross-sector CVaR. Consolidate into `sp500_sector_etfs.csv`. | P2-medium | BDH | advisory-sizing |
| VIX futures full term (UX1–UX7) | `BDH UX1…UX7 Index` (BQL §4) | Contango/backwardation regime, dealer GEX shifts, HAR-RV front-curve. Fill `vix_term_structure.csv`. | P2-medium | BDH | advisory-sizing |
| SPX realized correlation (COR1M/3M/6M) | `BDH COR1M/3M/6M Index` (BQL §7) | `portfolio_copula` Student-t CVaR correlation regime; corr→0.9 contagion. | P2-medium | BDH | evaluate-input-correctness |
| Stock-specific short interest & borrow (per name) | `BQL eqy_short_interest/_pct_of_float/borrow_rate_net` (§3) | `risk_manager` crowded-short, `portfolio_risk_gates` single-name caps. Monthly sufficient. | P2-medium | BQL | remove-only-gate |
| Dealer gamma/vanna (SPX level) | `BQL dealer_gex_total/dex_total/vanna` (§9, tier) | `dealer_positioning` ground-truth GEX, gamma-flip → `regime_multiplier`. Else option-chain approximation (wired). | P2-medium | BQL | advisory-sizing |
| Market breadth (A/D, %>50MA, put/call) | `AD Index`, `PUC Index`, `SPXA50R Index` **(verify via FLDS)** | Regime confirmation/divergence; breadth deterioration leads VIX. Low signal-to-noise. | P3-nice-to-have | BDH | advisory-sizing |
| FX majors (DXY/EURUSD/GBPUSD/JPYUSD) | `BDH … Curncy OHLC` | Cross-asset tail context (USD↔EM stress, flight-to-safety). | P3-nice-to-have | BDH | advisory-sizing |
| Commodities (WTI/Gold/Copper/Natgas) | `BDH CL1/GC1/HG1/NG1 Comdty` | Inflation-regime signal, energy-sector (XLE) tail correlation. | P3-nice-to-have | BDH | advisory-sizing |
| Global vol indices (V2X/VHSI/VNKY/VKOSPI) | `BDH … Index` (BQL §11) | Cross-asset/global regime; contagion ahead of US. US-focused strategy can skip. | P3-nice-to-have | BDH | advisory-sizing |
| HY credit derivatives (CDX IG/HY, iTraxx) | `BDH CDXIG/CDXHY US5Y Index` | Credit-regime confirmation vs OAS; reflationary signal. | P3-nice-to-have | BDH | advisory-sizing |
| SPX valuation regimes (P/E, CAPE/Shiller) | `SHILLER Index` **(verify via FLDS)**; else aggregate constituents | Portfolio-drift regime (value vs growth); CAPE>25 warning. | P3-nice-to-have | BDH | advisory-sizing |
| Treasury term premium | `CRX10Y Index` **(verify via FLDS)**; else nominal−real | Macro regime, HAR-RV drift calibration. Limited history. | P3-nice-to-have | BDH | advisory-sizing |
| Issuer CDS 5y (per name) | `BQL cds_spread_5y` (§8, tier) | Complements Altman-Z; faster repricing on misses; single-name caps. | P3-nice-to-have | BQL | advisory-sizing |
| Analyst revisions stream (per name) | `BQL best_eps/target_price/rating` (§6) | `earnings_drift` revision momentum, upgrade/downgrade filter on earnings dates. | P3-nice-to-have | BQL | advisory-sizing |

---

## 8. Sentiment, ESG, news & events

These are the canonical `downgrade-only` / `remove-only-gate` layer: the D18 transparency surface (news/social sentiment) never touches EV; corporate-action and macro calendars are lockout gates; analyst/ESG/insider signals are advisory. Several event datasets restate Tier 0 / §6 / §7 items from the news-and-events lens.

| Dataset | Bloomberg vehicle | Engine use | Benefit | Feasibility | §2 role |
|---|---|---|---|---|---|
| Corporate actions: stock splits (announce/effective) | `CAC_ANNOUNCEMENT_DATE`, `CAC_EFFECTIVE_DATE`, `CAC_TYPE`, `CAC_RATIO` (BQL §2) | `event_gate` 3-day split lockout (early assignment on short calls). **File empty (2 bytes).** | P1-high | BQL | remove-only-gate |
| Corporate actions: special dividends | `DVD_ANNOUNCE_DT`, `DVD_EX_DT`, `DVD_RECORD_DT`, `DVD_AMOUNT`, `DVD_TYPE` (BDS) | `event_gate` dividend-buffer lockout; `from_bloomberg_calendar()` ready. Short-put dividend-capture avoidance. | P1-high | BDS | remove-only-gate |
| Earnings announcement dates & times (BMO/AMC) | `EARNING_ANNOUNCEMENT_TIMING`, `…_CONFIRMED` | `event_gate` 5-day earnings lockout; confirmation flag reduces timing uncertainty. Have date+time in `sp500_earnings.csv`. | P1-high | BDP | remove-only-gate |
| Macro economic release calendar | `ECO_RELEASE_DT/EVENT/IMPORTANCE/COUNTRY` (BQL §1) | `event_gate` 1-day macro lockout (FOMC/CPI/NFP/GDP/PMI/claims) via `from_bloomberg_calendar()`. | P1-high | BQL | remove-only-gate |
| Fed funds rate decisions & guidance | `FDTR Index`, `FOMC_DECISION_DT`, `FED_RATE_DECISION` | `regime_multiplier`, BSM risk-free calibration vs `treasury_yields.csv`. | P1-high | BDH | evaluate-input-correctness |
| IV term structure & calendar spread (90/120d add) | `30/60/90/120DAY_IMPVOL_100.0%MNY_DF` | `regime_detector`/`monte_carlo` contango/backwardation; strangle timing. On disk via EXTRACTION_GUIDE §5. | P1-high | BDH | evaluate-input-correctness |
| Vol-of-vol (VVIX + realized) | `VVIX Index` (BQL §5); compute realized from VIX returns | `regime_detector` vol-of-vol; VVIX>30 vega convexity → strangle vol-buy. | P1-high | BDH | evaluate-input-correctness |
| SKEW index | `SKEW Index` (BQL §5) | `regime_detector` tail regime + Taleb; SKEW>120 hedging, <100 complacency. Feeds `tail_risk`/`monte_carlo`. | P1-high | BDH | evaluate-input-correctness |
| Corporate actions: spinoffs/split-offs/M&A | `CAC_TYPE` in (Spinoff/Split-off/Merger/Acquisition), dates, ratio (BQL §2) | `event_gate` + position tracking; M&A jump risk, spinoff tail gamma. May need deal-tracking tier; EDGAR 8-K fallback. | P2-medium | BQL | remove-only-gate |
| Earnings surprise & guidance flags | `EARN_EST_EPS_SURPRISE_PCT`, `GUIDANCE_RAISED/LOWERED_FLAG`, `IS_EPS`, `BEST_EPS_MEDIAN` | `earnings_drift`/`monte_carlo` beat-miss model; guidance = leading deterioration. Have IS_EPS+BEST_EPS_MEDIAN. | P2-medium | BDH | evaluate-input-correctness |
| EPS revision history (current vs month-ago) | `BEST_EPS_MEDIAN`, `EPS_REVISION_CNT_UP/DN` | EV risk-gate fundamental signal; EPS cuts precede misses (IV crush). | P2-medium | BDP | evaluate-input-correctness |
| Analyst consensus & recommendation distribution | `BEST_ANALYST_RATING`, `TOT_ANALYST_REC`, `REC_{BUY/HOLD/SELL}_CNT` | Buffett consensus-strength/crowding (all-buys = bubble). Have rating+count; add distribution. | P2-medium | BDP | advisory-sizing |
| Analyst target price & upside | `BEST_TARGET_PRICE`, `BEST_TARGET_UP_DN_PCT`, `TOT_TARGET_PRICE_MEDIAN/HIGH/LOW` | Buffett valuation anchor; revision momentum. Have snapshot; want BDH time-series. | P2-medium | BDP | advisory-sizing |
| Analyst rating changes & momentum | `REC_RATING_CHG_DT/PREV/NEW` (BDS) **(verify via FLDS)** | D18 transparency; rating cascades = crowding. Event-driven; BDS likely needed. | P2-medium | BDS | downgrade-only |
| Recommendation distribution heatmap (time-series) | `REC_BUY/HOLD/SELL_CNT` (BDH quarterly) | Buffett/Munger crowding; compute rating entropy for consensus level. | P2-medium | BDH | advisory-sizing |
| Short interest & borrow rate | `EQY_SHORT_INTEREST(_PCT_OF_FLOAT)`, `EQUITY_SHORT_BORROW_RATE_NET` (BQL §3) | `risk_manager` short-put sizing; crowded short + low borrow = danger, high borrow = gamma-flip risk. | P2-medium | BQL | advisory-sizing |
| Institutional ownership & activist positioning | `INST_OWN_PCT`, `ACTIVIST_FLAG/STAKE_PCT` **(verify via FLDS)** | Crowding (rotation-vulnerable), activist tail risk. Activist needs manual monitoring. | P2-medium | BDP | advisory-sizing |
| Macro surprise index (Citi/Bloomberg) | `CESIUSD Index`, `BESIUSD Index` | `regime_detector` macro gauge (>0 better-than-expected); IV term-structure expectation. Clean daily signal. | P2-medium | BDH | evaluate-input-correctness |
| CPI / inflation print calendar | `CPI YOY/MOM Index`, `CPURNSA Index`, `ECO_*` | `regime_multiplier` inflation-shock; macro lockout + post-release vol shock. | P2-medium | BDH | advisory-sizing |
| Unemployment / NFP calendar | `NFP TCH Index`, `NMPCHNG Index`, `ECO_*` | `regime_multiplier`; NFP surprise vol spikes + gamma convexity. | P2-medium | BDH | advisory-sizing |
| Debt/credit ratings + changes | `SP_RATING`, `MOODYS_RATING`, `SR_UNSECURED_RATING`, `RATING_CHANGE_DT/DIR` | `risk_manager` credit-quality gate; downgrade <6mo → do-not-sell-puts. Complements CDS in `sp500_credit_risk.csv`. | P2-medium | BDP | remove-only-gate |
| Options market structure (P/C ratio, OI skew) | `OPT_PUT_CALL_OI/VOL_RATIO`, `OPT_SKEW`, `OPT_SKEW_SLOPE_25D/10D` **(verify via FLDS)** | `skew_dynamics` dealer positioning/tail appetite. Cross-validate vs Theta-computed greeks. | P2-medium | BDP | evaluate-input-correctness |
| Dealer GEX/DEX per symbol | `DEALER_GEX_TOTAL`, `DEALER_DEX_TOTAL`, `DEALER_GEX_PROFILE` (BQL §9, tier) | `dealer_positioning`; GEX<0 dealers short gamma → crash risk. Else Theta-chain approximation. | P2-medium | BQL | advisory-sizing |
| Cross-asset correlation regimes | `COR1M/3M/6M Index` (BDH §7); or local Spearman/Kendall | `portfolio_copula` Student-t CVaR; corr→1 breaks diversification (2008 behavior). | P2-medium | BDH | evaluate-input-correctness |
| Sector rotation flows (ETF weight Δ MoM) | `weight` for members(XLF…XLRE) (BQL §10, frq=M) | `portfolio_risk_gates` sector cap; over/underweight vs SPX → cap tightening. | P2-medium | BQL | advisory-sizing |
| FCF growth & cash conversion | `CF_FREE_CASH_FLOW`, `CF_OPERATIONS/INVESTING/FINANCING_CASH_FLOW`, `CF_FREE_CASH_FLOW_MARGIN` | Buffett/Munger quality; FCF>NI = quality, FCF↓ while EPS↑ = red flag. Add to `sp500_fundamentals.csv`. | P2-medium | BDH | advisory-sizing |
| News sentiment & article volume | `NEWS_SENTIMENT`, `NEWS_CATEGORY`, `NEWS_VOLUME_PCT_CHANGE_252D` | D18 transparency lens (`news_sentiment.py`); dashboard + row dict, **does NOT influence EV**. Taleb input. | P3-nice-to-have | BDH | downgrade-only |
| Social sentiment & trending | `SOCIAL_SENTIMENT`, `SOCIAL_VOLUME`, `SOCIAL_HEAT_IND` **(verify via FLDS)** | D18; retail-crowding on short puts. Simons advisory. May need Refinitiv. | P3-nice-to-have | BDH | downgrade-only |
| ESG scores & ratings | `ESG_SCORE`, `ESG_RANK_SECTOR/REGION`, `ESG_PERCENTILE` **(verify via FLDS)** | Munger regulatory-risk lens; flags ESG-tail names ahead of activist/regulatory shocks. Not an EV multiplier. | P3-nice-to-have | BDP | downgrade-only |
| ESG controversy flags & incidents | `ESG_CONTROVERSY(_SCORE)`, `CONTROVERSIES/LITIGATION/REGULATORY_FLAG` (BDS) **(verify via FLDS)** | Taleb tail scanner; pending litigation/probes. Incident detail → OMON/EDGAR. | P3-nice-to-have | BDS | downgrade-only |
| Bankruptcy & restructuring flags | `BANKRUPTCY_FLAG/CHAPTER/FILING_DT`, `RESTRUCTURING_FLAG` (BDS) **(verify via FLDS)** | Taleb downgrade (avoid wheel on distressed). Likely needs EDGAR/CDS-widening proxy. | P3-nice-to-have | BDS | downgrade-only |
| Insider trading (Form 4, 10b5-1, blackout) | `INSIDER_PURCHASE/SELLING_RATIO`, `INSIDERS_OWN_PCT` (BDS) **(verify via FLDS)** | Taleb scanner (insider selling before miss). Needs Form-4 parse / EDGAR. | P3-nice-to-have | BDS | downgrade-only |
| Capital allocation (buybacks, dividend initiations) | `SHP_REPURCHASE_*`, `DVD_INITIATION_DT`, `DVD_INCREASE_CNT` (BDS) **(verify via FLDS)** | Buffett/Munger capital-allocation quality. Buyback announcements via 8-K/EDGAR. | P3-nice-to-have | BDS | advisory-sizing |

---

## 9. Implementation / wiring work (no pull needed)

These ship value with **zero new Bloomberg pull** — they are code-and-restore tasks. All §2-safe. (Tier 0 surfaces the highest-ROI subset; this is the complete list.)

| # | Work item | Module(s) | What it unblocks | §2 role |
|---|---|---|---|---|
| W-1 | **Wire the deep 5×5 IV-surface archive into the connector** | `engine/data_connector.py`, `engine/skew_dynamics.py` | Zero-pull skew reactivation — the gitignored deep moneyness×tenor archive is read directly. **Highest ROI.** | evaluate-input-correctness |
| W-2 | **Fix PIT dividend lookahead** | `engine/data_connector.py` (`get_fundamentals`, no `as_of` today) | BSM dividend yield `q` selected point-in-time in historical backtests instead of a 2026 snapshot. | evaluate-input-correctness |
| W-3 | **Wire Theta per-strike OI into `DealerPositioningAnalyzer`** | `engine/dealer_positioning.py` | Dealer multiplier currently inert at 1.0 → real OI-weighted GEX from the ~390M Theta EOD-chain rows we already hold (greeks back-solved). | advisory-sizing |
| W-4 | **Land D19 exit-cost + D21 DTE→trading-bars + recalibration together** | `engine/ev_engine.py`, `engine/wheel_runner.py` | Two deferred decision-layer changes; ship as one unit with the mandatory S27/S32/S34/S35 re-baseline (cross-cutting invariant). | evaluate-input-correctness |
| W-5 | **Restore corporate-actions from the off-main branch** | `data/bloomberg/sp500_corporate_actions.csv`, `engine/event_gate.py` | Split/special-div/M&A gate is inert (file is 2 bytes). A populated copy exists off-main — prefer restore over re-pull. | remove-only-gate |
| W-6 | **Pre-2018 OHLCV backfill from the deep archive** | `data/bloomberg/sp500_ohlcv.csv` (+ deep) | OHLCV starts 2018-01-02; the 504-day survivorship gate blocks candidates until ~2020. Split-adj-vs-raw caveat; mandatory S27/S32/S34/S35 re-baseline. | evaluate-input-correctness |

> **Re-baseline reminder:** W-4 and W-6 (and any deep-archive merge) trigger the mandatory `S27/S32/S34/S35` regression re-baseline — `python -m backtests.regression.<id> --update-snapshot`. The fingerprint pins OHLCV only, so IV/treasury changes are a latent blind spot; re-run the determinism gate before pinning.

---

## 10. Quick-start: pullable today

The cheapest, highest-impact pulls whose **BQL/BDH queries already exist** in `scripts/bloomberg_bql_pulls.md` — copy/paste into Excel, save to `data/bloomberg/`, wire a `load_*` method in `data/consolidated_loader.py`, re-run `scripts/feature_smoke_test.py`. Start here on the next terminal session.

| Pull | Query (in `bloomberg_bql_pulls.md`) | Target file | Why first | §2 role |
|---|---|---|---|---|
| **Treasury / UST curve** | §6-style BDH `USGG3M/6M/2YR/5YR/10YR Index PX_LAST` | `treasury_yields.csv` | Kills the pre-2021 hardcoded ~5% risk-free default — pure correctness, cheapest high-impact. | evaluate-input-correctness |
| **Macro-event calendar** | §1 `BQL eco_release_dt/event/importance/country` | `sp500_macro_calendar.csv` | Activates the dormant `event_gate.py` macro path (FOMC hardcoded today). | remove-only-gate |
| Corporate actions | §2 `BQL cac_*` (or restore off-main first) | `sp500_corporate_actions.csv` | Split/M&A gate is inert (2-byte file). | remove-only-gate |
| Short interest & borrow | §3 `BQL eqy_short_interest…` | `sp500_short_interest.csv` | Crowded-short sizing input (no `.csv` on disk today — only a `.csv.xlsx` workbook). | advisory-sizing |
| VIX futures UX1–UX7 | §4 `BQL px_last UX1…UX7` | `vix_futures_curve.csv` | Contango/backwardation regime; one tidy pull. | advisory-sizing |
| SKEW / vol-of-vol family | §5 `BQL px_last SKEW/VVIX/VXN/RVX/OVX/GVZ…` | `vol_indices.csv` | Tail + vol-of-vol regime; `sp500_vix_full.csv` holds 2015→2026 (16,955 rows). | advisory-sizing |
| Realised correlation | §7 `BQL px_last COR1M/3M/6M` | `spx_correlation.csv` | First-order for portfolio CVaR; cheap. | evaluate-input-correctness |
| Analyst revisions stream | §6 `BQL best_eps/target_price/rating` | `sp500_analyst_revisions.csv` | Revision momentum (monotonic on put-seller win rate); snapshot today. | advisory-sizing |

> **FLDS discipline.** Every field mnemonic in this document — both the quick-start queries and the full catalog — must be confirmed via Bloomberg **FLDS** before the pull is treated as authoritative. Silent mnemonic drift (e.g., a moneyness-family suffix that does not split put vs call, or an `EQY_AVG_VOL_*D` field that does not exist at this tier) is a recurring failure mode. Pull, then verify the returned column shape against the loader's expected column map; treat `field_confidence: low` items as unconfirmed until FLDS says otherwise.

---

## 11. Legend

**Feasibility codes**

| Code | Meaning |
|---|---|
| `BDH` | Bloomberg `=BDH(...)` historical time-series pull (Excel). Cheapest path for daily/quarterly series. |
| `BDP` | Bloomberg `=BDP(...)` current-snapshot pull. Single point-in-time value per field. |
| `BDS` | Bloomberg `=BDS(...)` bulk-data set (multi-row per security, e.g. dividend/rating/action histories). |
| `BQL` | Bloomberg Query Language `=BQL.QUERY("...")` — universe-wide structured pulls (members(), dates=range()). |
| `manual-OMON` | No mass-pull field; screen-only manual export (e.g. OMON option chains, Trades tab) or client-side computation from already-pulled series. |
| `use-theta` | Not feasible on Bloomberg at this tier — source from **Theta** (per-strike chains, tick/intraday tape). |
| `have-already` | Data (or a populated copy) already exists on disk / off-main / in Theta; the task is wiring or restore, not a fresh pull. |

**Benefit tiers**

| Tier | Meaning |
|---|---|
| `P1-high` | Load-bearing for correctness or a primary signal — do first. Typically `evaluate-input-correctness` or a live `remove-only-gate`. |
| `P2-medium` | Materially sharpens a regime/risk/advisor signal; clear engine consumer exists or is one wiring step away. |
| `P3-nice-to-have` | Incremental/diagnostic value. Worth listing under the pull-broadly mandate (free terminal, unlimited storage); pull opportunistically. |

**§2 role codes** (see §1 for the full contract)

| Role | Meaning |
|---|---|
| `evaluate-input-correctness` | Correctness input consumed *inside* `EVEngine.evaluate`; changes the value of `ev_raw`, never adds a rescue path. |
| `remove-only-gate` | Event/lockout input to `event_gate.py`; can only remove a candidate (lockout), never make a blocked one tradeable. |
| `downgrade-only` | Reviewer signal; downgrades a verdict (proceed → review/skip), never upgrades. |
| `advisory-sizing` | Advisor/sizing signal; scales `ev_dollars` *down* via the `[0.70, 1.05]` dealer multiplier or biases sizing — never touches `ev_raw`, never rescues. |