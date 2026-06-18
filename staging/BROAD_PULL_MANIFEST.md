# Broad Bloomberg pull — running manifest (lab session 2026-06-17, resumed 2026-06-18)

Branch `claude/bloomberg-broad-pull-2026-06-17` off main @ 83eacdd. Per
`docs/BLOOMBERG_PULL_LIST.md` + `docs/DATA_ACQUISITION_ROADMAP.md`. **Bloomberg pulls only —
staged, monoliths + trio byte-untouched, held for review. No integration / engine / §9 wiring.**
Method: FLDS-verify each mnemonic via a recent slice → reusable puller (xbbg 1.3.0 narwhals tidy
→ pivot) → validate (coverage + sane bands + overlap-to-cent where a vintage exists) → stage →
commit+push per dataset.

## Resume log — session 2026-06-18 (fresh lab box, no local state)
Box `IITS-I108-09`, user `mertmert`. Home profile re-provisioned → last night's clone + venv gone;
**banked work survived because it was pushed to this branch** (HEAD 746b4ab, verified). Rebuilt to
"ready to pull" per `docs/FRESH_LAB_BOX_SETUP.md`:
- Cloned single-branch (daybot branch excluded by construction — WHEEL-only rail held).
- venv `C:\Users\mertmert\smart-wheel-engine\.venv` (gitignored): **blpapi 3.26.5.1, xbbg 1.3.0,
  pandas 2.3.3, numpy 2.3.5, narwhals 2.22.1** (pinned to this box's proven base-conda versions;
  `PYTHONUTF8=1` for narwhals repr).
- **Decisive API gate GREEN:** `blp.bdh('AAPL US Equity',…)` → real entitled values, narwhals tidy
  `['ticker','date','field','value']`. **Latest available bar = 2026-06-18** (AAPL 297.37).
- OHLCV rotation re-confirmed on committed bytes: `open==max` & `low==min` frac=1.000 → refresh
  fragments MUST replicate the rotated FIELD_MAP.
- **Blocked on:** GitHub push auth — GCM configured but no stored credential; push needs the
  user's one-line browser-OAuth (headless can't complete it). Metered pull awaits user go-ahead
  per `docs/FRESH_LAB_BOX_SETUP.md` STEP 4 ("tell me you're ready and wait").

## Step 0 — census vs real committed bytes (DATA_INVENTORY is STALE; drifts recorded)
| file | DATA_INVENTORY / roadmap claim | ACTUAL committed bytes (main) |
|---|---|---|
| daily ohlcv/vol_iv/liquidity | end 2026-03-20 | **end 2026-06-04**; latest available bar now **2026-06-18** (gate-confirmed) |
| treasury_yields.csv | "starts 2021-05 — backfill" (T0-6) | **1994-01-03 → 2026-06-05**, full curve; rate_1m from 2001-07-31 (CMT inception). T0-6 backfill largely unnecessary — verify tenor completeness instead |
| sp500_corporate_actions.csv | "empty 2-byte stub" (T0-7) | **52,442 rows, max 2026-06-05** — already populated. Verify completeness, do NOT restore |
| deep/ archive (5×5 IV surface, deep OHLCV) | "on disk 2005→2026 (gitignored)" | **ABSENT in this fresh clone** (gitignored → never cloned). T0-2 moneyness grid pulled fresh here (✅ banked) |
| vix_term_structure.csv | — | cols `vix/vix_3m/vix_6m` (constant-maturity), end 2026-06-04; **lacks UX1–UX7 futures** |
| dividends / earnings | — | forward-dated (ex_date→2027-03-12; earnings→2028-01-19) |
| fundamentals / credit_risk | — | dateless snapshots (the #354 lookahead) |

Currency-refresh delta is therefore **2026-06-05 → 06-18** (latest bar = today, gate-confirmed),
not from 03-20.

## Datasets pulled (running)
| dataset | file (staging/) | fields | range · freq | rows | validation | committed |
|---|---|---|---|---|---|---|
| Vol-index complex | `macro_vol/sp500_vol_indices.csv` | PX_LAST: VIX/VVIX/SKEW/VXN/RVX/OVX/GVZ/MOVE/VXEEM/CVIX | 2004→2026-06-17 · D | 5847 | sane bands (VIX 9.1–82.7, MOVE 36–265); 2008/COVID spikes present | ✅ |
| Implied correlation | `macro_vol/spx_correlation.csv` | PX_LAST: COR1M/3M/6M | 2006→2026-06-17 · D | 5146 | corr-index 3–96 | ✅ |
| Credit OAS (IG/HY) | `macro_vol/credit_spreads.csv` | PX_LAST: LUACOAS, LF98OAS | 2004→2026-06-17 · D | 5647 | IG 0.71–6.18%, HY 2.33–19.71% | ✅ |
| **T0-2 moneyness IV skew surface** | `iv_surface/sp500_iv_surface.csv.gz` | 5×5: `{30DAY,60DAY,3MTH,6MTH,12MTH}_IMPVOL_{90,95,100,105,110}.0%MNY_DF` | 2010→2026-06-17 · D | **1,944,699** (509 names) | 96–98% grid coverage; skew put-rich (AAPL 30d 90%=28.7>ATM 24.8), upward term structure; **100%MNY col = current ATM IV** | ✅ |
| **T0-5 macro-event calendar** | `macro_calendar/sp500_macro_calendar.csv` | bds `ECO_FUTURE_RELEASE_DATE_LIST` (date+time) + bdp NAME/COUNTRY, 11 events (FOMC/CPI/coreCPI/PCE/NFP/unemp/claims/GDP/ISM-mfg/ISM-svc/retail) | 2025→2027-12 · E | 352 sched | release times 08:30/10:00/14:00; forward through 2027 | ✅ |
| T0-5 macro release history | `macro_calendar/sp500_macro_releases.csv` | bdh PX_LAST, same 11 events | 2015→2026-06-17 · E | 4724 prints | actuals | ✅ |
| **A · ohlcv refresh** | `currency_refresh/sp500_ohlcv__2026-06-05_2026-06-18.csv` | rotated O/H/L/C/V | 06-05→06-18 · D | 5075 (508 nm) | rotation gate 1.0000; KLAC 10:1 split seam flagged (see VALIDATION.md) | ✅ |
| **A · liquidity refresh** | `currency_refresh/sp500_liquidity__2026-06-05_2026-06-18.csv` | avg_vol_30d/turnover/shares_out | 06-05→06-18 · D | 5080 (508 nm) | overlap-to-cent (ex-KLAC + benign vol finalization <2%) | ✅ |
| **A · vix_term refresh** | `currency_refresh/vix_term_structure__2026-06-05_2026-06-18.csv` | vix/vix_3m/vix_6m | 06-05→06-18 · D | 10 | overlap 06-04 Δ=0 exact; contango held | ✅ |
| **B · VIX futures UX1–UX7** | `macro_vol/vix_futures_curve.csv` | PX_LAST: UX1..UX7 Index | 2006→2026-06-18 · D | 5150 | bands ux1 9.6–72.6 (2008/2020 spikes); contango 82% of days; upward tail | ✅ |
| **B · T0-12 short interest** | `short_interest/sp500_short_interest.csv` | `SHORT_INTEREST` (shares) + `SHORT_INT_RATIO` (days-to-cover); biweekly | 2015→2026-05-29 · biweekly | 134,035 (509 nm) | SI median 7.4M sh; DTC median 2.92; pct-of-float + borrow **entitlement-blocked** (all-NaN) → bucket F | ✅ |
| **B · T0-3 dividend-yield PIT** | `dividend_pit/sp500_dividend_yield_pit.csv` | EQY_DVD_YLD_12M / _IND / DVD_SH_12M; monthly | 2010→2026-05-29 · M | 72,461 (421 nm) | #354 gap = "no dividend" not missing (89/90 non-payers confirmed; BK 1-name anomaly). Fixes lookahead | ✅ |
| **D · OIS/SOFR curve** | `macro_rates/ois_sofr_curve.csv` | USSO 1m–30y + SOFRRATE + USOSFR 1–10y | 2001→2026-06-18 · D | 6393 | 0–5.7%; SOFR from 2018 | ✅ |
| **D · real yields/TIPS** | `macro_rates/real_yields.csv` | USGGT 2/5/10/30y + USSWIT infl-swap 2/5/10 | 2000→2026-06-18 · D | 6900 | real −3..7.4%; infl-swap 0.8–4.8% | ✅ |
| **D · fed funds** | `macro_rates/fed_funds.csv` | FDTR + FF1 | 2000→2026-06-18 · D | 6850 | target 0.25–6.5% | ✅ |
| **D · macro surprise** | `macro_rates/macro_surprise.csv` | CESIUSD + CESIG10 (Citi; BESIUSD not entitled) | 2003→2026-06-18 · D | 6044 | −145..271 | ✅ |
| **D · FX majors** | `macro_rates/fx.csv` | DXY/EURUSD/USDJPY/GBPUSD | 2000→2026-06-18 · D | 6904 | correct hist ranges | ✅ |
| **D · commodities** | `macro_rates/commodities.csv` | CL1/GC1/HG1/NG1 | 2000→2026-06-18 · D | 6652 | WTI −37.6 (Apr-2020) → 145 captured | ✅ |
| **D · global vol + CDX** | `macro_rates/global_vol.csv` | V2X/VHSI/VNKY/VKOSPI + CDX IG/HY (IBOXUMAE/HYSE) | 2000→2026-06-18 · D | 6880 | vol 10–104; IG 44–152bp HY 269–871bp | ✅ |
| **D · sector/factor ETFs** | `macro_rates/sector_factor_etfs_ohlcv.csv` | OHLCV, 15 ETFs (natural map) | 1998→2026-06-18 · D | 94,646 | high==max 1.0; XLRE 2015 XLC 2018 inception ✓ | ✅ |
| **C · total-return + EOD bid/ask** | `per_name/returns_micro.csv` | TOT_RETURN_INDEX_NET_DVDS, PX_BID, PX_ASK | 2010→2026-06-18 · D | 1,874,882 (511 nm) | 70MB; tot_return 100%, bid/ask 99.5%; px_bid>px_ask 0.0000% | ✅ |
| **C · ATM IV term + realized vol** | `per_name/vol_term_rv.csv.gz` | atm_iv_{30,60,90,180,365,730}d + rv_{10..260}d | 2010→2026-06-18 · D | 1,963,364 (510 nm) | 55MB gz (raw 183MB gitignored); cov 95–99%; 2y IV 79%; outliers flagged | ✅ |
| **C · beta + shares-out** | `per_name/beta_shares.csv` | BETA_RAW_OVERRIDABLE + EQY_SH_OUT | 2010→2026-05-29 · M | 93,605 (510 nm) | beta median 1.004; adjusted-beta not entitled; outliers flagged | ✅ |
| **C · PIT fundamentals (Q)** | `per_name/fundamentals_q.csv` | revenue/oper_inc/net_income/ebitda/eps/assets/liab/fcf/cfo/roe/nd_ebitda/gross_margin | 2010→2026-05-31 · Q | 31,479 (511 nm) | 79–100% cov; period-end dated (filing-lag PIT not captured) | ✅ |
| **C · estimates + analyst (M)** | `per_name/estimates_m.csv` | best_eps/sales/ebitda/target/pe + best_rating + analyst_count | 2010→2026-05-29 · M | 92,680 (511 nm) | target/rating/count 99–100%; est levels ~21% (FPERIOD override re-pull queued) | ✅ |
| **C · ratings/GICS/ownership snapshot** | `per_name/sp500_snapshot_bdp.csv` | RTG_SP/MOODY/FITCH + full GICS + inst/float + next-earnings | 2026-06-18 snapshot | 511 nm | SP 89%/Moody 75%/Fitch 65%; GICS+ownership 100%; **watch/outlook all-NaN** (not entitled) | ✅ |
| **C · valuation multiples (M)** | `per_name/valuation_m.csv` | px_to_book/ev_to_ebitda/px_to_sales/pe/peg | 2010→2026-05-29 · M | 89,079 | medians P/B 3.3, EV/EBITDA 13.6, P/E 20.8; peg 21%; outliers flagged | ✅ |
| **C · profitability/leverage/payout/growth (Q)** | `per_name/fundamentals_ext_q.csv` | roic/oper&net&ebitda margin/debt_to_equity/int_coverage/dvd_payout/sales_growth/trail_fcf | 2010→2026-05-31 · Q | 31,470 | ROIC 9.5%, margins/leverage sane | ✅ |
| **C · forward estimates (FPERIOD)** | `per_name/estimates_fwd.csv` | best_eps/sales/ebitda × 1BF + 2BF | 2010→2026-05-29 · M | 93,169 | 1BF eps/sales 100% — fixes the est-level coverage gap | ✅ |
| **C · options structure + news sentiment** | `per_name/options_sentiment.csv` | pc_oi_ratio(derived), pc_vol_ratio, oi_call, oi_put, news_sent | 2010→2026-06-18 · D | 1,998,083 | 88MB; pc_oi derived oi_put/oi_call; pc_vol & news_sent outliers flagged for winsorization | ✅ |

Omitted: `NFCI Index` (BlpRequestError — not entitled).

**T0-2 field-family lessons (FLDS by non-null VALUE count, not field echo):** the populated equity
surface is the documented **5×5** — long tenors use **MTH** naming (`90/180/365DAY` return all-NaN;
use `3MTH/6MTH/12MTH`), and only moneyness **{90,95,100,105,110}** populate (wings `{80,120}` empty).
**Storage:** raw CSV is 318 MB (float IV) → exceeds GitHub's 100 MB limit; committed **gzipped**
(round IV to 2 dp → 93 MB `.gz`), raw `.csv` gitignored. Future large pulls follow the same pattern.

---

## Remaining catalog — reconciled & disposition-tracked (resume point)
Legend: `[ ]` to pull · `[~]` verify-only (have-already) · `[x]` done/not-needed · `⛔` can't-pull/out-of-scope.
All mnemonics FLDS-verify by non-null VALUE count on a recent slice before the full pull. Per-name =
Universe A (~503 SPX members from `INDX_MWEIGHT`). Long pulls: recent-first, resumable, commit-per-chunk.

**Step-0 reconciliation (fixes the stale double-listing):** T0-2 skew surface and T0-5 macro
calendar are **DONE** (✅ above) — removed from remaining. `vol_iv` ATM refresh **not needed** —
current ATM IV rides the skew surface's `100%MNY_DF` column (06-17).

### A · Currency refresh 06-05 → 06-18 (staged fragments; monoliths byte-untouched) — ✅ DONE
- [x] `ohlcv` tail — rotated FIELD_MAP + gate (1.0000); **KLAC 10:1 split seam flagged** · `currency_refresh/`
- [x] `liquidity` tail — `avg_vol_30d / turnover / shares_out` (EOD `PX_BID/PX_ASK` → bucket C item) · `currency_refresh/`
- [x] `vix_term_structure` tail — `vix / vix_3m / vix_6m` (overlap Δ=0) · `currency_refresh/`
- [x] `vol_iv` — N/A (current via skew-surface 100%MNY col)

### B · Quick single-series pulls
- [x] **VIX futures UX1–UX7** — `macro_vol/vix_futures_curve.csv`, 5150 rows 2006→06-18, contango 82% ✅
- [x] **T0-12 short interest** — `short_interest/sp500_short_interest.csv`: `SHORT_INTEREST`+`SHORT_INT_RATIO` 134k rows ✅. pct-of-float + borrow **not entitled** (→ bucket F)
- [x] **T0-3 #354 dividend PIT** — `dividend_pit/sp500_dividend_yield_pit.csv` (72,461 rows, 421 nm, dated monthly). **Investigation: gap is "no dividend" not "missing"** (89/90 non-payers confirmed via DVD_HIST; BK 1-name field anomaly flagged). Fixes the lookahead. ✅

### C · P1 per-name catalog (the bulk — recent-first, resumable, commit-per-chunk)
- [x] ATM IV term + realized vol — `per_name/vol_term_rv.csv.gz` (1.96M rows) ✅
- [x] total-return series — `per_name/returns_micro.csv` ✅
- [x] EOD bid/ask history — `per_name/returns_micro.csv` (PX_BID/PX_ASK) ✅
- [x] beta + shares-out — `per_name/beta_shares.csv` (BETA_RAW_OVERRIDABLE + EQY_SH_OUT) ✅
- [x] PIT financial statements — `per_name/fundamentals_q.csv` (12 entitled IS_/BS_/CF_/ratio fields; period-end dated) ✅
- [x] estimates + analyst — `per_name/estimates_m.csv` (best_eps/sales/ebitda/target/pe/rating/count) ✅; est-levels re-pull w/ FPERIOD override queued
- [x] valuation / profitability / leverage / FCF / growth — `per_name/valuation_m.csv` (px_to_book/ev_ebitda/px_to_sales/pe/peg) + `per_name/fundamentals_ext_q.csv` (roic/margins/debt_to_equity/int_coverage/dvd_payout/sales_growth/trail_fcf) ✅
- [x] options market structure — `per_name/options_sentiment.csv` (put/call OI & vol ratios, total call/put OI) + news sentiment ✅ (see below)
- [x] forward estimates (FPERIOD 1BF/2BF) — `per_name/estimates_fwd.csv` ✅ (fixes the ~21% level coverage)
- [x] credit ratings (SP/Moody/Fitch) — `per_name/sp500_snapshot_bdp.csv` ✅ (watch/outlook all-NaN → bucket F)
- [x] GICS full + institutional/float — `per_name/sp500_snapshot_bdp.csv` ✅
- [x] earnings timing — `EXPECTED_REPORT_DT` in snapshot ✅ (surprise computable from IS_EPS vs BEST_EPS downstream)
- ⛔ CDS spreads — `CDS_SPREAD_5Y/1Y/10Y` all-NaN, not entitled → bucket F

### D · P2/P3 macro & cross-asset single-series (roadmap §7–§8) — ✅ DONE (8 files in `macro_rates/`)
- [x] sector/factor ETFs OHLCV — `sector_factor_etfs_ohlcv.csv` (15 ETFs, 1998→06-18)
- [x] OIS/SOFR — `ois_sofr_curve.csv` (USSO + SOFRRATE + USOSFR)
- [x] real yields/TIPS — `real_yields.csv` (USGGT + USSWIT)
- [x] fed funds path — `fed_funds.csv` (FDTR + FF1; ZQ not entitled)
- [x] macro surprise — `macro_surprise.csv` (CESIUSD/CESIG10; BESIUSD not entitled)
- [x] FX majors — `fx.csv`
- [x] commodities — `commodities.csv`
- [x] global vol + CDX — `global_vol.csv` (V2X/VHSI/VNKY/VKOSPI + IBOXUMAE/IBOXHYSE)
- [x] news sentiment — `NEWS_SENTIMENT_DAILY_AVG` pulled in `per_name/options_sentiment.csv` ✅
- ⛔ ESG scores/controversies — `ESG_DISCLOSURE_SCORE`/E/S/G/`SUSTAINALYTICS_RANK` all-NaN, not entitled → bucket F

### E · Verify-only (Step 0 found present) — ✅ DONE
- [x] corp-actions **verified complete** — KLAC 10:1 split IS present (announced 05-07, effective 06-12); also CRWD 4:1, DD. 52,442 rows, do NOT restore. (Corrected the bucket-A worry.)
- [x] treasury tenor completeness verified (1994+ full curve, 3m–30y ~8,455 rows, 1m from 2001, sofr from 2018) **+ tail refresh** `currency_refresh/treasury_yields__2026-06-06_2026-06-18.csv` (overlap 06-05 Δ=0 exact)
- [x] index-membership history verified present (72,696 rows w/ weights)

### F · Can't-pull / out-of-scope
- ⛔ `NFCI Index` — not entitled (BlpRequestError)
- ⛔ short-interest **pct-of-float** (`EQY_SHORT_INTEREST_PCT_OF_FLOAT`/`SI_PERCENT_FLOAT`/all variants) & **borrow rate** (`EQUITY_SHORT_BORROW_RATE_NET`/`COST_OF_BORROW`/`GC_RATE`/…) — all-NaN, no SLB entitlement at this tier (FLDS-by-value-count confirmed). `SHORT_INTEREST`+`SHORT_INT_RATIO` pulled instead; pct-of-shares-out derivable later from `EQY_SH_OUT`.
- ⛔ per-strike OI / greeks / smile, intraday/tick tape — `use-theta` (OMON not mass-pullable)
- ⛔ 13F / insider Form-4 / supply-chain / litigation, vol cones, Amihud λ — manual-OMON / EDGAR / client-side compute
- ⛔ §9 wiring & FIX items (T0-1/4/8/10/11, W-1…W-6) — code work, **out of scope** (pulls-only rail)

**Done = this list reaches zero** (every `[ ]` → ✅ pulled, `[~]` → verified, or ⛔ marked can't-pull).
