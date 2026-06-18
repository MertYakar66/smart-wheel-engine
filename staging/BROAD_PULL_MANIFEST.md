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

### A · Currency refresh 06-05 → 06-18 (staged fragments; monoliths byte-untouched)
- [ ] `ohlcv` tail — ROTATED FIELD_MAP `{PX_HIGH→open,PX_LAST→high,PX_LOW→low,PX_OPEN→close,PX_VOLUME→volume}` + per-row `open==max & low==min` gate · BDH
- [ ] `liquidity` tail — `avg_vol_30d / turnover / shares_out` (+ add EOD `PX_BID/PX_ASK`) · BDH
- [ ] `vix_term_structure` tail — `vix / vix_3m / vix_6m` · BDH
- [x] `vol_iv` — N/A (current via skew-surface 100%MNY col)

### B · Quick single-series pulls
- [ ] **VIX futures UX1–UX7** — `PX_LAST` `UX1…UX7 Index` → `vix_futures_curve.csv` (+ extend vix_term_structure) · BDH
- [ ] **T0-12 short interest** — `EQY_SHORT_INTEREST`, `EQY_SHORT_INTEREST_PCT_OF_FLOAT`, `EQUITY_SHORT_BORROW_RATE_NET` · BDH
- [ ] **T0-3 #354 dividend PIT** — investigate the 69% carry coverage: confirm the 31% is "no dividend" not "missing"; dated `EQY_DVD_YLD_12M`/`EQY_DVD_YLD_IND` to fill if missing · BDH

### C · P1 per-name catalog (the bulk — recent-first, resumable, commit-per-chunk)
- [ ] ATM IV term structure — `7/14/30/60/90/180/365/730DAY_IMPVOL_100.0%MNY_DF` · BDH
- [ ] realized-vol family — `10/20/30/60/90/120/180/260DAY_HV` · BDH
- [ ] total-return series — `TOT_RETURN_INDEX_NET_DVDS` · BDH
- [ ] EOD bid/ask history — `PX_BID`/`PX_ASK` · BDH
- [ ] beta history — `BETA_RAW_OVERRIDABLE`/`BETA_ADJUSTED_OVERRIDABLE` (1Y/2Y/5Y) · BDH
- [ ] shares-out PIT — `EQY_SH_OUT` (monthly) · BDH
- [ ] PIT financial statements — `IS_/BS_/CF_` line items + `BEST_PERIOD_END_DT`, `BEST_FISPD_SHEET_DT` · BDH
- [ ] estimates — `BEST_EPS/REVENUE/EBITDA/TARGET_PRICE` + revisions/dispersion · BDH
- [ ] valuation / profitability / leverage / FCF / growth families · BDH
- [ ] credit ratings + watch + outlook — `RTG_SP_LT_LC_ISSUER_CREDIT`, `RATING_WATCH`, `RATING_OUTLOOK` · BDP
- [ ] CDS spreads — `CDS_SPREAD_5Y` (1Y/10Y) · BDH
- [ ] GICS full + institutional/insider/float — `GICS_*`, `EQY_INST_PCT_SH_OUT`, etc. · BDP
- [ ] analyst history — `BEST_ANALYST_RATING`/`BEST_TARGET_PRICE`/`REC_*_CNT` · BDH/BDS
- [ ] earnings surprise + special-div + earnings timing — `EARN_EST_EPS_SURPRISE_PCT`, `DVD_*`, `EARNING_ANNOUNCEMENT_TIMING` · BDH/BDS

### D · P2/P3 macro & cross-asset single-series (roadmap §7–§8)
- [ ] sector/factor ETFs OHLCV (extend `sp500_sector_etfs.csv`) · BDH
- [ ] OIS/SOFR — `USSO{tenor} Index` · BDH
- [ ] real yields/TIPS — `USGGT{tenor}`/`USSWIT{tenor}` · BDH
- [ ] fed funds path — `FDTR Index`, `ZQ{contract}` · BDH
- [ ] macro surprise — `CESIUSD`/`BESIUSD Index` · BDH
- [ ] FX majors — `DXY`, `EURUSD`/`USDJPY`/`GBPUSD Curncy` · BDH
- [ ] commodities — `CL1/GC1/HG1/NG1 Comdty` · BDH
- [ ] global vol + CDX — `V2X/VHSI/VNKY/VKOSPI Index`, `CDXIG/CDXHY US5Y` · BDH
- [ ] ESG scores/controversies, news/social sentiment — FLDS-verify entitlement first · BDP/BDH

### E · Verify-only (Step 0 found present — no pull)
- [~] corp-actions completeness (52,442 rows → 2026-06-05; do NOT restore)
- [~] treasury tenor completeness (1994+ full curve; rate_1m from 2001-07)
- [~] index-membership history present (verify/extend)

### F · Can't-pull / out-of-scope
- ⛔ `NFCI Index` — not entitled (BlpRequestError)
- ⛔ per-strike OI / greeks / smile, intraday/tick tape — `use-theta` (OMON not mass-pullable)
- ⛔ 13F / insider Form-4 / supply-chain / litigation, vol cones, Amihud λ — manual-OMON / EDGAR / client-side compute
- ⛔ §9 wiring & FIX items (T0-1/4/8/10/11, W-1…W-6) — code work, **out of scope** (pulls-only rail)

**Done = this list reaches zero** (every `[ ]` → ✅ pulled, `[~]` → verified, or ⛔ marked can't-pull).
