# Verification card — Bloomberg refresh-runbook audit (for a memoryless terminal)

**You are a fresh Claude Code terminal with repo access and NO prior context.**
Your job: independently verify that the refreshed Bloomberg runbook
`docs/BLOOMBERG_TERMINAL_NEXT_SESSION.md` is **accurate against the current
codebase**, so an operator can trust it at a Bloomberg lab terminal. A wrong
field mnemonic, path, or line number would waste a lab session — so be
adversarial and check every falsifiable claim.

**Rules:** READ-ONLY. Commit nothing, edit nothing. You have no Bloomberg
Terminal, so do **not** run any `pull_*.py` — anything that needs live
Bloomberg entitlement (does a field resolve? does a pull return rows?) is
`UNVERIFIABLE-NEEDS-TERMINAL`, not a guess. Report a structured
**PASS / FAIL / UNVERIFIABLE** per check with one-line evidence (a `file:line`
or command output), then an overall verdict.

---

## Preflight
```bash
git fetch origin claude/smart-wheel-engine-overview-0txgye
git checkout claude/smart-wheel-engine-overview-0txgye && git pull --ff-only origin claude/smart-wheel-engine-overview-0txgye
sed -n '1,14p' docs/BLOOMBERG_TERMINAL_NEXT_SESSION.md      # expect header "Refreshed 2026-07-21"
```
Read `docs/BLOOMBERG_TERMINAL_NEXT_SESSION.md` in full before checking — every
claim you verify is a quote from it.

---

## Checks

**C1 — Served-file contract (the "Served? Yes/No" labels).**
The runbook labels each file "Served?" = **read by `MarketDataConnector` on the EV
path** — which is a SUPERSET of `_FILES` membership: the 10 `_FILES` monoliths PLUS
files the connector reads via `BroadPullLoader` (e.g. `snapshot_bdp`,
`dividend_yield_pit`). Do NOT treat `_FILES` membership as the whole definition of
"served."
```bash
grep -n "_FILES" engine/data_connector.py         # the 10-key dict
grep -n "BroadPullLoader\|snapshot_bdp\|dividend_yield_pit" engine/data_connector.py  # the extra served path
```
PASS iff: (a) `_FILES` contains exactly the 10 keys `ohlcv, vol_iv, dividends,
earnings, treasury, vix, fundamentals, credit_risk, liquidity, corporate_actions`,
all labeled served in the runbook; (b) `snapshot_bdp` and `dividend_yield_pit` are
labeled served because the connector reads them via `BroadPullLoader` (they are
**not** in `_FILES` — that is correct, not a defect); (c) `sp500_macro`,
`sp500_index_membership`, `sp500_sector_etfs`, `sp500_short_interest`,
`sp500_vol_dvd` are neither in `_FILES` nor read on the EV path and are labeled
off-path. Flag only a genuine label↔code disagreement, not "not in `_FILES`."

**C2 — The stale-end-date trap AND the `SWE_PULL_MODE=both` backfill trap.**
Runbook §Priority-1 says the `_bbg_panel` pullers ship `end_date="2026-06-04"`
hardcoded, that you must set `SWE_PULL_END`, AND that a routine append **must** set
`SWE_PULL_MODE=forward` because the default `mode=both` also backfills the backward
gap to the 1994 floor.
```bash
grep -n 'end_date' scripts/pull_ohlcv.py scripts/pull_vol_iv.py scripts/pull_historical_fundamentals.py
grep -n 'SWE_PULL_MODE\|SWE_PULL_END\|floor\|forward\|backfill\|both' scripts/_bbg_panel.py
# and confirm the runbook actually sets the override:
grep -n 'SWE_PULL_MODE=forward' docs/BLOOMBERG_TERMINAL_NEXT_SESSION.md
```
PASS iff: (a) `pull_ohlcv.py:67` / `pull_vol_iv.py:53` carry `end_date="2026-06-04"`
and `pull_historical_fundamentals.py:29` carries `"2026-03-20"`; (b)
`_bbg_panel.py` defaults `SWE_PULL_MODE` to `both` and `plan_windows` emits a
backward window to the 1994 floor when mode is `both`/`backfill`; **and (c) every
runbook command for a `_bbg_panel` script (ohlcv, vol_iv, liquidity, macro,
iv_surface) sets `SWE_PULL_MODE=forward`.** FAIL if any such command omits the
`forward` override or the runbook still claims a bare/`SWE_PULL_END`-only run
"appends nothing" (it would backfill 1994→2018). _(This was the DISCREPANCY the
2026-07-21 audit found and the runbook was corrected for — re-confirm it stayed
fixed.)_

**C3 — Field mnemonics match what each script requests.**
```bash
grep -n 'PX_OPEN\|PX_LAST\|PX_VOLUME' scripts/pull_ohlcv.py
grep -n 'HIST_PUT_IMP_VOL\|HIST_CALL_IMP_VOL\|VOLATILITY_260D\|Fill' scripts/pull_vol_iv.py
grep -n 'VOLUME_AVG_30D\|TURNOVER\|EQY_SH_OUT' scripts/pull_liquidity.py
grep -n 'VIX\|VIX3M\|VIX6M\|PX_LAST' scripts/pull_vix_term_structure.py
grep -n 'EXPECTED_REPORT_DT\|next_earnings\|GICS' scripts/pull_snapshot_bdp.py
grep -n 'EQY_DVD_YLD_12M\|VOLATILITY_30D\|GICS_SECTOR\|30day_impvol\|30DAY_IMPVOL' scripts/pull_snapshots.py
```
PASS iff the fields the runbook lists for each file are the fields that script
actually pulls (no missing/renamed mnemonic).

**C4 — OHLCV rotation.** Runbook claims OHLCV is stored **rotated**
(`open←PX_HIGH, high←PX_LAST, close←PX_OPEN`) and the connector inverse-renames
around `data_connector.py:517-524`.
```bash
sed -n '505,530p' engine/data_connector.py
grep -n 'rotate\|PX_HIGH\|PX_OPEN\|rename' scripts/pull_ohlcv.py | head
```
PASS iff the connector's rename map and the script's rotation are mutually
consistent with the claim (i.e. the layout round-trips to correct OHLC).

**C5 — Post-pull frontier pins.** Runbook §7-C lists 5 stale-clone pins holding
`2026-07-02` / `2026-07-03` at specific `file:line`.
```bash
grep -n 'EXPECTED_FRONTIER\|EXPECTED_EARNINGS_CALENDAR_ASOF\|FRONTIER' \
  tests/test_preflight_environment.py tests/test_data_connector.py \
  tests/test_data_to_engine.py tests/test_data_integrity_bloomberg.py
grep -n 'EXPECTED_FRONTIER' engine/data_connector.py
```
PASS iff the 5 constants exist at ~the stated lines with `2026-07-02`/`-03` values.

**C6 — Re-baseline snapshots.** Runbook §7-E says **exactly** `s27/s32/s34/s35`
carry `--update-snapshot`, and there is **no** `param_oos_regime_*` module.
```bash
ls backtests/regression/*.py
grep -rl 'update-snapshot\|update_snapshot' backtests/regression/*.py
ls backtests/regression/param_oos_regime_* 2>&1        # expect: No such file
```
PASS iff only those four modules match and the `param_oos_regime_*` glob is empty.

**C7 — `sp500_vol_dvd.csv` safe to leave stale.** Runbook §8 says it is NOT read
by the served connector (only the non-served `ConsolidatedBloombergLoader`).
```bash
grep -rn 'vol_dvd' engine/data_connector.py data/consolidated_loader.py
```
PASS iff absent from `data_connector.py` and present only in `consolidated_loader.py`.

**C8 — Theta-only classification.** Runbook §6 says real option premiums,
per-strike chains/greeks/OI, the `vol_indices.parquet`, and VIX futures are
Theta-only — not xbbg-pullable at a Bloomberg seat. Spot-check:
```bash
grep -n 'theta\|xbbg\|yfinance\|blp' scripts/pull_vol_indices.py | head
```
PASS iff `pull_vol_indices.py` is Theta/yfinance (not `xbbg`), corroborating
"not a Bloomberg pull." (Whether the Theta larder itself exists locally is
`UNVERIFIABLE` here — it's gitignored.)

**C9 — Earlier work also landed (the 9 audit-fix merges).**
```bash
git log origin/main --oneline -14 | grep -E '#49[5-9]|#50[0-3]'
git ls-remote --heads origin 'claude/fix-*'            # expect: empty
```
PASS iff all nine PRs (#495–#503) appear as squash commits on `origin/main` and
no `claude/fix-*` remote branch remains.

---

## Report format
One block per check:
```
C1 Served-file contract .......... PASS   evidence: _FILES @ data_connector.py:NNN lists ohlcv/vol_iv/…; vol_dvd absent
C2 Stale-end-date trap ........... FAIL   evidence: pull_vol_iv.py end_date is at line 48, not 53 (runbook says 53)
...
```
End with **VERDICT: RUNBOOK-TRUSTWORTHY** (all PASS or only UNVERIFIABLE-NEEDS-TERMINAL)
or **VERDICT: DISCREPANCIES** followed by the numbered list of every FAIL with the
exact correction needed. Do not soften a FAIL into a PASS because it "seems
minor" — the operator decides.
