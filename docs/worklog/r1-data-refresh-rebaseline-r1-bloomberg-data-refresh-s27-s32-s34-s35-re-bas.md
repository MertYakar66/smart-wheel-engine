---
id: r1-data-refresh-rebaseline
title: R1 Bloomberg data refresh + S27/S32/S34/S35 re-baseline
kind: backtest
status: completed
terminal:
pr: 338
decisions: []
date: 2026-06-06
headline: R1 = data-only refresh (Option B, 16 monoliths; sp500_dividends.csv held at main to avoid a source regression) + S27/S32/S34/S35 re-baseline, deep-read OFF; in-window EV-path movers are treasury_yields.csv (dominant; corrects wrong/missing main rates) + sp500_fundamentals.csv eqy_dvd_yld_12m (BSM dividend_yield, small/pervasive), plus S34's UNIVERSE_100 swap; ohlcv + vol_iv byte-identical in-window; trio byte-identical.
surface:
  - "In-window EV-path movers (adversarially verified, NOT treasury-only): treasury_yields.csv (dominant; 2018-20 0->784 rows; restated 2022-24 values CORRECT vs a known main bug, e.g. 2022-04-01 rate_2y 1.25->2.46%) AND sp500_fundamentals.csv eqy_dvd_yld_12m (BSM dividend_yield via get_fundamentals; dateless snapshot; small but pervasive, AAPL 0.42->0.34%). ohlcv + vol_iv in-window slices byte-identical (vol_iv -324k all pre-2018, ohlcv +26k all 2026). Other 15 refreshed files do NOT feed the EV ranking path."
  - "S34 baseline PROVISIONAL: data refresh drifted the connector universe (BNY/CASY in, CMG/CMI out) -> UNIVERSE_100 regenerated + S34 re-run (test_universes_match_connector enforces the derivation). BUT BNY is a re-ticker of BK (BK ohlcv stops 2026-03-20, BNY starts 2026-03-23 = BoNY Mellon rebrand) and CASY is also recent-only; both have 0 in-window 2022-24 rows while displaced CMG/CMI have full in-window data -> S34 effectively backtests 98 real names. Correct per current derivation; flag provisional. Data-layer follow-up: fix BK<->BNY continuity (same thread as BK dividends drop) then re-baseline S34 — tracked in issue #339 (seam audit confirms BNY+CASY are the ONLY two recent-only names; S34 = 98 real names). UNIVERSE_24 unaffected (S27/S32/S35 valid)."
  - "Refresh also invalidated two recent-date test pins: test_aapl_control (AAPL 2026-02-13 ev 5.50->5.27) and test_calm_regime smoke (no-as_of date drift) — both re-pinned in this PR."
  - "Data-quality (dividends HELD AT MAIN): the refresh source's sp500_dividends.csv dropped entire 2018-24 ex-div history for CTRA/BK/LW/PAYC (BK in UNIVERSE_100), so it was NOT refreshed (held at main -> BK 28 rows restored). OFF the rank_candidates_by_ev path so snapshots unaffected + §2 intact; dividends now ~2.5mo stale vs the rest. Data-layer follow-up: re-pull then refresh."
---

## Goal
Execute R1 from the data-layer activation roadmap: a **data-only** merge of the
refreshed Bloomberg monoliths into `main` (Option B — refresh only the 24 files
already tracked; NOT the 9 new files / 89 MB bid_ask / `.xlsx`), then re-baseline
the S27/S32/S34/S35 regression snapshots against `main`'s engine + the refreshed
data. Deep-read stays **OFF** (monolith-only); deltas must be cleanly
attributable to data. Decisions (operator): F-1 deep-read flip OUT of R1; F-2 R1
decoupled from R0b; F-3 Option-B checkout; UNIVERSE_100 drift -> regenerate + re-run S34;
sp500_dividends.csv held at main (refresh source dropped 2018-24 history, off the EV path).

## What we tried / How we fixed it
1. Landed the three preconditions to `main` (squash): #332 (docs roadmap), #333
   (R0a credit dead-read), #334 (R5 fingerprint pins vol_iv + treasury sha). `main`
   at `3dd4109`. R5 must precede the re-baseline so regenerated snapshots emit the
   new shas.
2. Worktree `claude/r1-data-refresh-rebaseline` off `main`. Option-B refresh:
   `git checkout origin/data/bloomberg-refresh-2026-06-02 -- <each of the 24 tracked
   files>` -> 17 changed initially; `sp500_dividends.csv` then reverted (held at main)
   after adversarial QA found a source regression (Data-quality) -> **16 committed**.
   0 new files, `.xlsx` untouched (pre-existing on main). `check_manifest_coverage.py`
   clean (0 uncovered).
6. Adversarial pre-PR verification (read-only, 3 agents) REFUTED the initial
   "treasury-only" attribution -> corrected to treasury + fundamentals dividend_yield
   (+ S34 universe swap); confirmed §2/scope/universe/test-repins; surfaced the
   dividends regression (held at main).
3. Re-baseline via the canonical drivers with `--update-snapshot`,
   `SWE_DEEP_HISTORY` unset (deep OFF) + provider=bloomberg (default).
4. UNIVERSE_100 drift (BNY/CASY in, CMG/CMI out, from new constituents in the
   refreshed data) regenerated in `backtests/regression/universes.py`; S34 re-run
   with the new universe. UNIVERSE_24 unaffected.
5. Re-pinned the two recent-date test expectations the refresh moved:
   `test_aapl_control` (ev 5.50->5.27) and the `test_calm_regime` smoke (pinned
   `as_of=2026-03-20`, assert no-widening on survivors rather than the
   never-achievable len==5).

## Evidence
Data shas (pinned-main -> refreshed): ohlcv `c3d5443->7a3e77a4`, vol_iv
`a64b747->aab4eada`, treasury `a76a3ef->48a20a88`.

**In-window attribution (adversarially verified — NOT treasury-only):** the EV/ranking
path (`rank_candidates_by_ev` + `rank_covered_calls_by_ev` -> `EVEngine.evaluate`) reads
ohlcv (spot), vol_iv (IV), treasury (risk-free via `get_risk_free_rate`), and the BSM
`dividend_yield` from `get_fundamentals` (`eqy_dvd_yld_12m`). Of the originally-changed
files, only treasury + fundamentals feed the path AND change in-window:
- ohlcv 2022-2024 + 2018-2020 slices: byte-identical (content-hash) main vs refresh.
- vol_iv 2022-2024 + 2018-2020 slices: byte-identical. The -324,337 vol_iv row drop
  is entirely pre-2018 (main pre-2018=350,441 -> refresh 0); +26k ohlcv growth is
  entirely 2026. Neither touches a window. (The changed fundamentals IV columns are
  dead in-window — PIT IV resolves from vol_iv 10/10.)
- **treasury (dominant mover):** DIFFERS in-window (2022-24: 753->782 rows; 2018-20:
  **0 -> 784 rows**, main had no pre-2021-05 curve -> 5% NaN-fallback). Restated 2022-24
  values are a **correction of a known main bug** (2022-04-01 rate_2y 1.25% -> 2.46%,
  the real yield). Largest effect on S35.
- **fundamentals `eqy_dvd_yld_12m` (secondary mover, small/pervasive):** dateless
  snapshot -> same changed value applied to every backtest day, both windows; changed
  for ~all names (AAPL 0.42->0.34%, UNH 3.18->2.21%). 22/24 UNIVERSE_24 names carry a
  non-zero BSM dividend_yield after the /100 + >0.30 guard; dividend-only sensitivity
  ~UNH strike +$0.16 (non-zero, small vs treasury).
- The other refreshed files do NOT feed the EV path (liquidity uncalled; credit/vix
  off-path or FRED; 9 files have no connector accessor).
- **`sp500_dividends.csv` — held at main, but NOT inert (mechanism corrected):** it is
  off the *short-put* ranker (`rank_candidates_by_ev`; puts never consult ex-div) and does
  NOT feed the BSM `dividend_yield` (that's `eqy_dvd_yld_12m` from fundamentals). But it
  **is** read by the *covered-call* ranker: `rank_covered_calls_by_ev -> get_next_dividend`
  feeds `EVEngine`'s ex-div early-assignment expected-loss input, which changes which CC is
  selected -> the CC-leg realized cash (`final_nav`/`final_cash`). So dividends move the CC
  leg in every book; short-put `ev_mean`/entry-count stay dividend-independent. Held at main
  in both old and new here, so dividends contribute **zero** to the old->new delta above —
  but a future re-pull WILL shift all four `final_nav`s (see the controlled-swap proof under
  Unresolved), which is why the re-pull re-baselines all four, not just S34. §2 untouched
  (no `EVEngine.evaluate` bypass).

**Data-quality (dividends HELD AT MAIN):** the refresh source's `sp500_dividends.csv`
dropped the entire 2018-2024 ex-dividend history for CTRA/BK/LW/PAYC (BK in UNIVERSE_100),
verified via `get_dividends` (BK -> 0 rows; not migrated to corporate_actions).
**Decision (operator): hold `sp500_dividends.csv` at main** — reverted, not refreshed, so
history is preserved (BK -> 28 rows restored); committed refresh is therefore **16** files,
not 17. It is OFF the short-put path (`rank_candidates_by_ev`) but ON the covered-call path
(`get_next_dividend` ex-div early-assignment) — so the snapshots are baselined against
*main* dividends; holding at main keeps them consistent (had we shipped refreshed dividends,
all four `final_nav`s would shift — proven below). §2 intact; dividends are now ~2.5mo stale
vs the rest. Data-layer follow-up: re-pull dividends, then **re-baseline all four** snapshots.

**Snapshot deltas (old -> new):**
| Snapshot | spearman_rho | final_nav | trades | iv_mean |
|---|---|---|---|---|
| S27 (2022-24 $100k) | 0.1855 -> 0.1833 | 112,311 -> 113,382 | 43 -> 45 | +-0.0000 |
| S32 (2022-24 $1M fric) | 0.1837 -> 0.1819 | 1,073,819 -> 1,072,269 | 105 -> 102 | +-0.0000 |
| S34 (2022-24 $1M 100t, new UNIVERSE_100; PROVISIONAL) | 0.3222 -> 0.3152 | 1,308,573 -> 1,287,703 | 303 -> 295 | +-0.0000 |
| S35 (2018-20 $100k OOS) | 0.4904 -> 0.5120 | 115,830 -> 112,605 | 40 -> 33 | -0.0011 |

(final_nav values are the committed main-div snapshots, exact: S27 113,381.96 / S32
1,072,269.50 / S34 1,287,703.49 / S35 112,604.58. These differ from the *earlier*
refreshed-dividends snapshots by exactly the CC-leg ex-div amounts — S32 +$219.81,
S34 +$5.33, S27/S35 sub-tolerance — see the dividend-mechanism note below.)

iv_mean is flat across all (vol_iv in-window identical; tiny shifts are
executed-set composition). Movement = treasury (dominant) + fundamentals dividend_yield
(small, pervasive). S35 moves most (its 2018-20 risk-free curve went from the
NaN-fallback to real rates). S34's delta additionally reflects the BNY/CASY<->CMG/CMI
universe swap — and is **PROVISIONAL**: BNY is a BK re-ticker (BK ohlcv stops 2026-03-20,
BNY starts 2026-03-23) and CASY is recent-only (both 0 in-window 2022-24 rows), while the
displaced CMG/CMI have full in-window data (753 rows each). So S34 effectively backtests 98
real names; clears after the BK<->BNY data fix + an S34 re-baseline (handoff).

**Trio byte-identical** to pre-R1 main (git hash-object): `ev_engine.py`
`e991c111`, `wheel_runner.py` `07dd45c4`, `candidate_dossier.py` `6b724001`. Zero
`engine/` changes in R1.

**Full suite** (`pytest -m "not backtest_regression"`, refreshed data, after fixes):
2817 passed; only 3x `test_theta_connector` remain (local Theta-server 472 — confirmed
fail on main too, environment, NOT R1). The initial run's other 3 fails were R1-caused
and resolved: `test_universes_match_connector` (UNIVERSE_100 regen), `test_aapl_control`
+ `test_calm_regime` (re-pinned). `test_universes_match_connector` is a fast (non-marker)
test on main -> the UNIVERSE_100==connector[:100] derivation is enforced going forward.

## Resolved before merge
- **Regression-marker determinism re-run — CLEAN 4/4** (`test_backtest_matches_snapshot`,
  the reproducibility proof the fast-CI suite skips). All four reproduce the committed
  snapshots: S27/S32/S35 (1 passed each) + S34 (1 passed, 1:58 h). Determinism was in fact
  proven *twice* — two fully independent regen runs (a planned one + a stray concurrent one)
  produced byte-identical values (S32 `811672.686459997`, S34 `391286.1946799898`), and the
  markers reproduce them a third time. No skips/errors.
- **Dividend-mechanism root-cause (controlled-swap proof).** An initial S32/S34 marker
  mismatch traced to snapshots that had been generated with *refreshed* dividends while the
  committed tree holds *main* dividends. Controlled swap: refreshed dividends -> S32 reproduces
  the (stale) value; main dividends -> S32 `811672.69` (+$219.81), S34 `391286.19` (+$5.33),
  S27/S35 sub-tolerance. Mechanism = `get_next_dividend` -> CC ex-div early-assignment EV
  input -> CC selection -> CC-leg realized cash (above). Fixed by regenerating all four
  against committed main dividends.
- **Fingerprint extended to the full connector set (`_common.py` only).** Added
  `connector_data_sha256()` (SHA-256 of all 9 connector input CSVs, incl. dividends) to the
  snapshot fingerprint, closing the blind-spot that let the dividends drift go unrecorded
  (the legacy fingerprint pinned only ohlcv/vol_iv/treasury). Harness-only change; trio still
  byte-identical. Enforcement (a guard test that fails fast on drift) is deferred to issue
  #340 with the placebo-RNG cleanup.

## Unresolved / handoff
- **f4 smoke fix** is carried in R1 (`test_calm_regime` as_of pin); the standalone
  `claude/f4-smoke-pin-asof` branch is being CLOSED (operator decision — R1 carries it).
- **Data-layer follow-ups — tracked in GitHub issue #339** (whole thread): (a) re-pull
  `sp500_dividends.csv` (held at main; refresh source dropped 2018-24 ex-div history for
  BK/CTRA/LW/PAYC), then refresh; (b) fix the BoNY-Mellon BK<->BNY re-ticker (BK ohlcv
  stops 2026-03-20, BNY starts 2026-03-23 — restore continuity so BNY isn't a separate
  phantom in the connector universe); (c) backfill CASY history; then **re-derive
  UNIVERSE_100 + re-baseline S34** to clear its provisional flag. Seam audit (read-only,
  all 100 UNIVERSE_100 names vs the 2026-03-23 seam): **BNY + CASY are the ONLY two
  recent-only members** (both first-ohlcv 2026-03-23, 0 rows in BOTH the 2022-24 and
  2018-20 windows) -> S34 backtests **98 real names exactly** (not an undercount). Same
  data thread as the BK dividends drop.
- **Deep-read flip-on** (R2) stays a SEPARATE reviewed step (deep gz not on main).
  **R0b** (sector source / R9) deferred — when it lands, first check whether the
  regression harness even arms R9 before assuming a re-baseline.
- The 9 new refresh files (incl. 89 MB unsharded `sp500_bid_ask.csv`) + `.xlsx`
  drop ride with R4/R7, not R1.
