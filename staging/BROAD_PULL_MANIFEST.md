# Broad Bloomberg pull — running manifest (lab session 2026-06-17)

Branch `claude/bloomberg-broad-pull-2026-06-17` off main @ 83eacdd. Per
`docs/BLOOMBERG_PULL_LIST.md` + `docs/DATA_ACQUISITION_ROADMAP.md`. **Bloomberg pulls only —
staged, monoliths + trio byte-untouched, held for review. No integration / engine / §9 wiring.**
Method: FLDS-verify each mnemonic via a recent slice → reusable puller (xbbg 1.3.0 narwhals tidy
→ pivot) → validate (coverage + sane bands + overlap-to-cent where a vintage exists) → stage →
commit+push per dataset.

## Step 0 — census vs real committed bytes (DATA_INVENTORY is STALE; drifts recorded)
| file | DATA_INVENTORY / roadmap claim | ACTUAL committed bytes (main) |
|---|---|---|
| daily ohlcv/vol_iv/liquidity | end 2026-03-20 | **end 2026-06-04** (already ~2.5 mo more current) |
| treasury_yields.csv | "starts 2021-05 — backfill" (T0-6) | **1994-01-03 → 2026-06-05**, full curve; rate_1m from 2001-07-31 (CMT inception). T0-6 backfill largely unnecessary — verify tenor completeness instead |
| sp500_corporate_actions.csv | "empty 2-byte stub" (T0-7) | **52,442 rows, max 2026-06-05** — already populated. Verify completeness, do NOT restore |
| deep/ archive (5×5 IV surface, deep OHLCV) | "on disk 2005→2026 (gitignored)" | **ABSENT in this fresh clone** (gitignored → never cloned). T0-2 moneyness grid must be pulled fresh here, not "extended" |
| vix_term_structure.csv | — | end 2026-06-04 (lacks UX1–UX7 futures) |
| dividends / earnings | — | forward-dated (ex_date→2027-03-12; earnings→2028-01-19) |
| fundamentals / credit_risk | — | dateless snapshots (the #354 lookahead) |

Currency-refresh delta is therefore **2026-06-05 → 06-17** (~9 trading days), not from 03-20.

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

## Remaining catalog (priority order)
- **Currency refresh** (frontier 06-05→06-17): ohlcv + liquidity + vol_iv + vix_term_structure recent tail — fragments to extend monoliths.
- **T0-2 moneyness IV skew surface** (HIGH; deep archive absent here → fresh): `30/60/90/180DAY_IMPVOL_{80,90,95,100,105,110,120}%MNY_DF` × Universe A. Large; FLDS the field family first.
- **VIX futures UX1–UX7** (extend vix_term_structure).
- **T0-5 macro-event calendar** (BQL eco_release_dt/event/importance) — needs roadmap BQL spec.
- **T0-12 short interest** (EQY_SHORT_INTEREST + pct_of_float, borrow rate).
- **P1 per-name**: ATM IV term structure, realized-vol family, total-return, EOD bid/ask, beta history, shares-out PIT, PIT financial statements + estimates, ratings/watch/outlook, CDS spreads.
- Verify: corp-actions completeness (already populated); treasury tenor completeness 1994+.
- **P2/P3** as time allows (sector/factor ETFs, OIS/SOFR, real yields, FX, commodities, ESG, analyst history).
