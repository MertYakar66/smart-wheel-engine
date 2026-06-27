# WIRING CAMPAIGN — turning the banked broad-pull data into engine capability

_Compiled 2026-06-22 · **PLAN ONLY — docs, no code.** Dependency-ordered execution
plan for wiring the ~25 newly-pulled Bloomberg datasets (31 staged files) into the engine. The wiring
itself is §2-sensitive, EV-moving, and re-baseline-coupled — it runs later, under
human supervision. This document makes that session fast and clean; it changes no
engine code, touches no decision trio, and runs no re-baseline._

**Ground truth (read these first, in this order):**
- `staging/BROAD_PULL_MANIFEST.md` *(on branch `claude/bloomberg-broad-pull-2026-06-17`)* — what was actually pulled (the banked bytes).
- `docs/DATA_ACQUISITION_ROADMAP.md` — the catalog: every dataset → engine consumer → §2 role → benefit; §9 = the no-pull wiring work; §10 = quick-start.
- `docs/NEXT_DATA_SESSION_RUNBOOK.md` — the three (E) trio/risk-gate fixes (#372 / #369 / #378) **and the re-baseline governing rule**.
- `docs/DATA_TEST_AUDIT_2026-06-09.md` — capability corrections C1–C4 + the W14–W37 register (the skew/IV-gate/credit-path facts this plan depends on).
- `docs/DATA_INVENTORY.md` — the census (regenerated alongside this doc; the staged datasets are its §6).
- `CLAUDE.md` §2 — the invariant every step below obeys.

> **What "banked" means here — UPDATED 2026-06-26 (Phase 0B has landed).** The net-new
> datasets are now **on `main`** at **`data/bloomberg/broad_pull/<bucket>/`** (27 files,
> **byte-identical** to the `staging/<bucket>/` paths cited in the rows below), merged via
> **PR #417** alongside `data/broad_pull_loaders.py` (`BroadPullLoader` — a read-only,
> **dormant** loader; imported only by its tests, **not yet wired into the connector
> accessors** the engine reads). So **read `data/bloomberg/broad_pull/…` wherever a row
> below says `staging/…`**.
>
> **Two exceptions still branch-only** (`claude/bloomberg-broad-pull-2026-06-17` under
> `staging/`, **not on `main`**): the **Phase 0A `currency_refresh/` tails** (OHLCV /
> liquidity / vix-term / treasury, 2026-06-05→06-18) and **`staging/BROAD_PULL_MANIFEST.md`**.
> Per-dataset bytes/ranges/counts are in `docs/DATA_INVENTORY.md` §6 (byte-verified
> 2026-06-22). The data is now visible; **nothing below is *wired* yet** — this is the order
> in which to wire it (Phase 0B = loaders, substantially landed; the EV move is each
> later step that connects a loader to its consumer).

---

## 1. The governing rule — one re-baseline, everything EV-moving lands before it

The slow backtest snapshots **S27 / S32 / S34 / S35** must be re-pinned (~4 h total)
whenever the data or the EV output changes. The runbook's standing rule
(`NEXT_DATA_SESSION_RUNBOOK.md`, "Execution order at a glance"):

> **Everything that moves the frontier or the EV output must land BEFORE the
> 4-snapshot re-pin** — so the single re-baseline captures the data change, the
> (E)-fix EV impact, and every wiring step's `ev_raw` shift in one pass. **Do not pay
> the ~4 h re-baseline tax twice.**

So the campaign is ordered as: **make the data visible → land the held trio fixes →
wire the prize (skew) → wire the rest → re-baseline once.** Anything tagged
**re-baseline-coupled** below is, by definition, something the single Phase-R re-pin
must absorb.

### §2 contract — the three roles every input takes (from `DATA_ACQUISITION_ROADMAP.md` §1)

Nothing in this campaign can rescue a negative-EV trade. Every dataset enters in one
of these §2-safe roles:

| §2 role | What it may do | What it may **never** do |
|---|---|---|
| `evaluate-input-correctness` | change the *value* of `ev_raw` (the engine was always meant to use the right number) | add a rescue path |
| `remove-only-gate` | remove a candidate from the tradeable set (lockout) | make a blocked candidate tradeable |
| `downgrade-only` | downgrade a verdict (proceed → review/skip) | upgrade a verdict |
| `advisory-sizing` | scale `ev_dollars` *down* via the dealer multiplier `[0.70, 1.05]`, or bias sizing | touch `ev_raw`, rescue, or upgrade |

### Ceremony legend (the review weight each step carries)

| Flag | Meaning | Applies to |
|---|---|---|
| **CEREMONY** | full **§2 lane-claim ceremony** + §2 panel + **held for review** + the matching characterization test flipped (not deleted). **No autonomous land.** | decision-trio (`ev_engine` / `wheel_runner` / `candidate_dossier`) + risk-gate (`risk_manager` / `portfolio_risk_gates`) + any new/changed downgrade reviewer |
| **PANEL** | **§2 panel + held for review** — connector/serving or event-gate change that moves EV inputs but adds **no verdict logic** (the #369 precedent) | connector accessors feeding `EVEngine.evaluate`; `event_gate` remove-only wiring |
| **PLAIN** | normal branch + PR review — additive integration or display-only surface, **no EV impact** | new connector loaders nothing consumes yet; D18 transparency (news) |

> **All EV-moving steps are held for review** regardless of flag. "Re-baseline-coupled"
> means the step changes `EVEngine.evaluate` / ranker output on an S27/S32/S34/S35 path
> (`ev_raw`; a lockout that fires in the backtest; `ev_dollars` via dealer multiplier or
> a reviewer that runs in the ranked path; or a risk-gate verdict on the portfolio-context
> snapshot S34). Display-only, advisor-committee-only, and not-yet-consumed plumbing are
> **not** re-baseline-coupled.

> **Role label vs mechanism.** The `§2 role` column below carries the **roadmap's** coarse
> per-dataset label (roadmap §3–§8), which is occasionally looser than the precise lane: e.g.
> VIX→R11 and beta→R7/R8/R10 are *downgrade reviewers* (the `downgrade-only` lane) though the
> roadmap tags them `advisory-sizing`/`evaluate-input-correctness`; the credit reviewer scales
> `ev_dollars` (the `advisory-sizing` lane) though the roadmap tags ratings `downgrade-only`.
> **The `Engine consumer` + `Ceremony` columns are authoritative for how the step is gated**;
> the role label is kept roadmap-faithful for traceability.

---

## 2. Execution order at a glance (strict)

| Phase | What | Re-baseline-coupled? | Ceremony |
|---|---|---|---|
| **0** | **Integration — staged data → connector-readable.** Fold the currency refresh into the monoliths; add `load_*` accessors for every net-new file. *Nothing wires until the engine can see it.* | Only **0A** (frontier refresh) | 0A: PANEL · 0B: PLAIN |
| **1** | **The three (E) trio/risk-gate fixes** — **#372** (R9→GICS) → **#369** (IV-fallback clean) → **#378** (IV-staleness + rate-fallback). Held, §2-reviewed, one PR each. | Yes (all three) | **CEREMONY** ×3 |
| **2** | **Wire the skew surface (the prize)** → connector + `skew_dynamics` + `option_pricer`. | Yes | PANEL |
| **3** | **Wire the rest by consumer** — event gate · regime/tail/copula · forward-distribution · cost model · risk gates · advisors · credit reviewer. | Mixed (per row) | Mixed (per row) |
| **R** | **The single re-baseline** — re-pin S27/S32/S34/S35, bump `EXPECTED_FRONTIER`, re-pick W16/W30, flip xfails, clear S34 provisional, §2 panel. **Operator-merge, no auto-update.** | — (this *is* the re-pin) | CEREMONY (operator) |

> Dependency notes that fix the order:
> - **0A moves the frontier** (`2026-06-04 → 2026-06-18`) so it cascades into Phase R's
>   `EXPECTED_FRONTIER` bump + W16/W30 re-pick.
> - **#378 must land before 0A's spot-frontier bump** (a sequencing exception to the
>   "data-before-trio" default). 0A advances OHLCV/spot to 06-18, but the legacy ATM-IV
>   monolith (`sp500_vol_iv_full.csv`) has **no in-repo producer** (runbook line 132) and
>   stays at ~06-04 — so 0A *itself* opens the ~10-trading-day IV↔spot staleness gap that
>   #378 guards (audit W36). Either land #378 ahead of 0A, **or** re-pin the served ATM IV
>   in the same step as 0A so spot and IV frontiers advance together. #378 must also land
>   before **Phase 2 (skew)** and **Phase 3C** *consume* the new IV-term/surface data.
> - **#372 reads main `fundamentals.gics_sector_name`** (W19/#373 already served — **no
>   `snapshot_bdp` prerequisite**), with a counted-`'Unknown'` fallback. The staged
>   `snapshot_bdp` GICS is a single **2026-06-18** as-of; if used at all it must be
>   held-flat / back-versioned to avoid baking a lookahead into #372's re-baseline-coupled
>   path.

Row format for every wiring target below:
**dataset · engine consumer (module/file) · §2 role · EV-moving? → re-baseline-coupled · banked at · roadmap/§9 ref.**

---

## Phase 0 — Integration: staged data → connector-readable (prerequisite)

The staged files live under `staging/<bucket>/` in shapes the connector does not yet
read. Phase 0 folds the refresh into the existing monoliths and adds a `load_*` method
(per `DATA_ACQUISITION_ROADMAP.md` §10: "wire a `load_*` method in
`data/consolidated_loader.py`") for each net-new file. **Integration alone is not
EV-moving** — the EV shift happens when a loader is *connected to a consumer* (Phases
1–3). The one exception is **0A**, which overwrites series the engine already reads.

### 0A — Frontier refresh (overwrites consumed monoliths → EV-moving)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| OHLCV tail `2026-06-05→06-18` (5,075 rows, 508 nm; **rotated** O/H/L/C, gate 1.0000; **KLAC 10:1 split seam flagged**) | `data_connector.get_ohlcv` → `forward_distribution` cascade + HMM regime | evaluate-input-correctness | **Yes → coupled (moves frontier)** | `staging/currency_refresh/sp500_ohlcv__2026-06-05_2026-06-18.csv` | roadmap §A, runbook Phase 1A/A3 |
| Liquidity tail (5,080 rows, 508 nm; `avg_vol_30d/turnover/shares_out`) | `get_liquidity` → liquidity gating, sizing | evaluate-input-correctness | **Yes → coupled** | `staging/currency_refresh/sp500_liquidity__2026-06-05_2026-06-18.csv` | roadmap §4 |
| VIX-term tail (10 rows; `vix/vix_3m/vix_6m`; overlap 06-04 Δ=0) | `get_*` vix term consumers | advisory-sizing | No (regime context) | `staging/currency_refresh/vix_term_structure__2026-06-05_2026-06-18.csv` | roadmap §A |
| Treasury tail (9 rows `06-08→06-18`; full tenor + sofr; overlap Δ=0) | `get_risk_free_rate` → BSM `r` | evaluate-input-correctness | **Yes → coupled** | `staging/currency_refresh/treasury_yields__2026-06-06_2026-06-18.csv` | roadmap §7, runbook A3 |

> **0A is mechanical but EV-moving:** de-dup the overlap to the cent, then the frontier
> bump must flow into Phase R (`EXPECTED_FRONTIER` + the data-test `FRONTIER` constants;
> W16/W30 re-pick). Hold KLAC's 10:1 split seam for the seam audit (see manifest
> `currency_refresh/VALIDATION.md`). Flag: **PANEL** (serving change, no verdict logic).
>
> **⚠️ Staleness coupling — 0A creates the gap #378 guards.** 0A advances OHLCV/spot to
> 06-18 but does **not** refresh the legacy ATM-IV monolith (`sp500_vol_iv_full.csv`, no
> in-repo producer), so the moment 0A lands the served IV is ~10 trading days staler than
> spot. **Land #378 (Phase 1) before 0A, or re-pin the served ATM IV in the same step as
> 0A.** Until the legacy ATM IV is refreshed (or superseded by Phase 3C's `vol_term_rv`),
> the spot/IV frontiers diverge — pin the 06-04→06-18 overlap.

### 0B — Net-new connector loaders (plumbing → not EV-moving until wired)

> **Status 2026-06-26: substantially LANDED (PR #417).** `data/broad_pull_loaders.py`
> (`BroadPullLoader`) now reads every net-new file below from `data/bloomberg/broad_pull/`
> (gz-aware, winsorization-aware, lazy per-ticker series). It is **dormant** — imported only
> by `tests/test_broad_pull_loaders.py` + `tests/test_broad_pull_wiring_xfail.py`, **0
> callers in `engine/`**, and **not yet wired into `MarketDataConnector`'s `get_*` accessors**.
> So the remaining 0B gap is exposing these loaders through the connector path each Phase-2/3
> consumer reads; the rows below stand as the per-file consumer map.

Each net-new staged file needs a connector loader before any Phase-2/3 step can consume
it. Adding the loader is **PLAIN** and **not re-baseline-coupled** (nothing reads it yet);
the EV move is attributed to the wiring step that connects it. The files:
`iv_surface/sp500_iv_surface.csv.gz`, `macro_calendar/{sp500_macro_calendar,sp500_macro_releases}.csv`,
`macro_vol/{sp500_vol_indices,spx_correlation,credit_spreads,vix_futures_curve}.csv`,
`macro_rates/{ois_sofr_curve,real_yields,fed_funds,macro_surprise,fx,commodities,global_vol,sector_factor_etfs_ohlcv}.csv`,
`per_name/{returns_micro,vol_term_rv.csv.gz,beta_shares,fundamentals_q,fundamentals_ext_q,estimates_m,estimates_fwd,valuation_m,options_sentiment,sp500_snapshot_bdp}.csv`,
`dividend_pit/sp500_dividend_yield_pit.csv`, `short_interest/sp500_short_interest.csv`.

> **Storage gotcha** (manifest): two files are committed **gzipped** (`iv_surface` 96.8 MB
> `.gz`, `vol_term_rv` 58.7 MB `.gz`) because the raw CSVs exceed GitHub's 100 MB limit;
> the loader must read `.gz` directly. **Winsorization flags** (manifest): `options_sentiment`
> `pc_vol`/`news_sent` and several per-name level series carry outliers flagged for
> winsorization — clamp at load, not silently.

---

## Phase 1 — The three (E) trio/risk-gate fixes (CEREMONY each, held)

Decision-layer / risk-gate code changes (not data). They change EV output, so they land
**after** the data is visible (Phase 0) and **before** the re-pin (Phase R). Order is
**#372 → #369 → #378** (runbook Phase 2). Each: separate lane-claimed trio PR, **held for
review**, §2 panel the operator verifies, downgrade-only contract intact, no
`EVEngine.evaluate` bypass, the matching characterization test **flipped** (not deleted).

| # | Fix | Engine consumer | §2 role | EV-moving? → re-baseline | Banked / source | Ref |
|---|---|---|---|---|---|---|
| **#372** (HIGH) | R9 sector cap reads real `gics_sector_name` (today: hardcoded `DEFAULT_SECTOR_MAP`, **132/511** mapped → 379 collapse to `'Unknown'`, silently weakening the cap) | `risk_manager.py` (`SectorExposureManager`/`check_sector_cap` `:1755`) + `wheel_runner` sector col `:1777` | downgrade-only (R9 soft-warn); GICS data is evaluate-input-correctness | **Yes → coupled** | main `fundamentals.gics_sector_name` (W19/#373 **already served — no new loader**); counted-`'Unknown'` fallback. *(Do not source from the 2026-06-18 `snapshot_bdp` snapshot unless held-flat — lookahead.)* | roadmap §5 GICS, audit W17/C2, runbook 2A |
| **#369** | #363 IV gate also cleans the **fundamentals-fallback** IV path (today `_clean_vol_iv_inplace` runs only for `key=='vol_iv'`; the fallback at `wheel_runner.py:1082-1111` is normalised only by an inline `if iv>3.0: iv/=100` heuristic) | `data_connector.py` (#363 gate) + `wheel_runner` fallback IV `:1082`/CC `:2418` | evaluate-input-correctness | **Yes → coupled** | engine fix (no new data) | audit W27, runbook 2B |
| **#378** | IV-staleness gate on `_resolve_pit_atm_iv` (mirror the spot path's 30-day rule) + rate-fallback divergence (`data_integration.get_current_risk_free_rate` silent `0.05` vs connector NaN-on-missing) | `wheel_runner.py:153-202` + `engine/data_integration.py:323` | evaluate-input-correctness | **Yes → coupled** | engine fix; **couples to** staged IV-term refresh + treasury tail | audit W36/W37, runbook 2C |

> **Why #378 early — and why it must precede 0A:** it is latent on *today's* `main` (max
> IV↔OHLCV gap = 1 day) but **bites under a staggered refresh**. Two distinct triggers: (1)
> **Phase 0A** advances spot to 06-18 while the un-producer'd legacy ATM-IV file stays
> ~06-04 → an immediate ~10-day IV↔spot gap; (2) **Phase 2/3C** later *consume* the staged
> `vol_term_rv` / skew IV. So land #378 **before 0A's frontier bump** (or re-pin ATM IV with
> 0A), and certainly before Phase 2/3C consume the new IV. The earlier "data-before-trio"
> default is overridden here by the staleness coupling.

---

## Phase 2 — Wire the skew surface (the prize) (PANEL, held)

The dormant `skew_dynamics.py` reactivates the moment a real moneyness×tenor surface is
connector-served. The staged **fresh** surface supersedes the §9 **W-1** dependency on
the gitignored deep archive: W-1 said "wire the gitignored deep 5×5 archive"; we now have
a **committed, byte-present** surface to 2026-06-17 instead.

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| **Moneyness IV skew surface** — 5 tenor × 5 moneyness (`iv_{30,60,90,180,365}d_{90,95,100,105,110}`), **1,944,699 rows, 509 nm, 2010-01-04 → 2026-06-17** | `data_connector.py` (new surface accessor) → `engine/skew_dynamics.py` (risk-reversals 25Δ put−call, butterflies) + `engine/option_pricer.py` (moneyness-aware IV; vanna/charm/volga) | evaluate-input-correctness | **Yes → coupled** | `staging/iv_surface/sp500_iv_surface.csv.gz` | roadmap T0-1/T0-2, §9 **W-1**, §3 moneyness grid, §4 IV-skew slope |

> **Honest limits to carry into the wiring:** (1) the surface is the documented **5×5**
> (`{90,95,100,105,110}` moneyness; wings `{80,120}` are empty; long tenors use MTH naming
> upstream) — `skew_dynamics` must not assume 7-point wings. (2) The `100%MNY` column **is**
> the current ATM IV (manifest), so it reconciles with the ATM term structure (Phase 3C) —
> pin the overlap. (3) **`edge_vs_fair` stays structurally 0** on the Bloomberg path (audit
> C4 / W28): a sharper surface improves BSM IV inputs but does **not** revive the VRP signal,
> which still needs a market-mid option-premium producer. Do not let a
> coverage agent treat skew wiring as turning VRP live.
>
> **Update — the market-mid producer rail + ranker wiring have both landed.**
> `scripts/produce_option_premiums.py` + `MarketDataConnector.get_option_premium*` serve the
> real EOD `mid` from the Theta larder (`data_processed/option_premium/`, gitignored; coverage
> 2016→2026-06), split-adjusted on load to the engine frame. `wheel_runner`'s three rankers now
> swap `ShortOptionTrade.premium`/`bid`/`ask` from synthetic-BSM to the served mid via
> `_resolve_real_premium` — so `edge_vs_fair` / VRP is **live where the rail is present**.
> **No re-baseline:** the rail is gitignored, so CI/regression fall back to synthetic
> (byte-identical) — committed snapshots stay synthetic and the real-premium path is local-only,
> like all Theta-dependent behaviour. The **double-count concern is moot**: the production
> forward distribution is empirical (realized returns), not IV-scaled, so skew premium is not
> counted twice. Independent of fair-value skew (Phase 2) — the two compose but neither blocks
> the other. The synthetic-path C4/W28 invariant stays green (the wiring touches the caller, not
> `ev_engine`).

---

## Phase 3 — Wire the rest, by consumer (mixed ceremony, held)

### 3A — Event-gate inputs → event gate (remove-only-gate)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Ceremony | Banked at | Ref |
|---|---|---|---|---|---|---|
| Macro-event calendar — **11 events** (FOMC/CPI/coreCPI/PCE/NFP/unemp/claims/GDP/ISM-mfg/ISM-svc/retail), **352 scheduled rows, fwd through 2027-12** + release history (4,724 actuals, 2015→2026-06-17) | `engine/event_gate.py` (`from_bloomberg_calendar`) → **first gate** in `EVEngine.evaluate` | remove-only-gate | **Yes → coupled** | PANEL | `staging/macro_calendar/{sp500_macro_calendar,sp500_macro_releases}.csv` | roadmap T0-5, §7, §8, §10 |
| **Corporate actions** (splits/M&A/rights) — **already on `main`** (52,442 rows, **not staged**); only the wiring remains | `engine/event_gate.py` split / early-assignment lockout | remove-only-gate | **Yes → coupled** (fires lockouts in the backtest) | PANEL | `data/bloomberg/sp500_corporate_actions.csv` (on `main`) | roadmap T0-7, §9 **W-5** |

> Macro calendar fixes the **hardcoded FOMC** path and the silent CPI/NFP/PCE miss
> (`build_default_event_gate` registers only fomc/cpi/nfp today — the staged calendar adds
> the other 8). The **corp-actions** row is the §9 **W-5** wiring: the *data* is already on
> `main` (no restore/pull — the stale inventory's "2-byte stub" claim is wrong), but the
> `event_gate` consumption is EV-moving and **must be scheduled here, before Phase R** — left
> unplaced it would force a second re-baseline. Both feed `evaluate`'s first short-circuit, so
> the operator may elevate either **PANEL** to a full panel.

### 3B — Vol indices / correlation / OAS / cross-asset → regime detector · tail risk · copula

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Ceremony | Banked at | Ref |
|---|---|---|---|---|---|---|
| Vol-index complex — VIX/VVIX/SKEW/VXN/RVX/OVX/GVZ/MOVE/VXEEM/CVIX (5,847 rows, 2004→) | `regime_detector` + `tail_risk` + **`candidate_dossier` R11** (VIX>25 & high-conf → downgrade) | advisory-sizing *(R11 mechanism = downgrade-only)* | **Yes → coupled** (VIX feeds R11, audit C3/W35) | **CEREMONY** (R11 reviewer) | `staging/macro_vol/sp500_vol_indices.csv` | roadmap §3 VIX family, §7 |
| Implied correlation — COR1M/3M/6M (5,146 rows, 2006→) | `engine/portfolio_copula.py` (Student-t CVaR correlation regime → R7/R8) | evaluate-input-correctness *(copula → R7/R8 = downgrade)* | **Yes¹ → coupled¹** (R7/R8 on **S34** portfolio-context only) | **CEREMONY** (risk-gate) | `staging/macro_vol/spx_correlation.csv` | roadmap §7 COR |
| Credit OAS — IG (`LUACOAS`) / HY (`LF98OAS`) (5,647 rows, 2004→2026-06-16) | `regime_detector` credit regime; ranker `credit_mult` (today FRED HY-OAS, audit C1) | advisory-sizing | **Yes² → coupled²** (only if it replaces/augments the ranker `credit_mult`) | **CEREMONY** (ranker `credit_mult`) | `staging/macro_vol/credit_spreads.csv` | roadmap §7 credit spreads |
| VIX futures UX1–UX7 (5,150 rows, 2006→; contango 82%) | `regime_detector` contango/backwardation | advisory-sizing | No (regime context; the EV regime mult is the per-ticker OHLCV HMM, not VIX futures — C3) | PANEL/PLAIN | `staging/macro_vol/vix_futures_curve.csv` | roadmap §3/§7 UX1-7 |
| OIS/SOFR curve · real yields/TIPS · fed funds · macro surprise (Citi) · FX · commodities · global vol+CDX | `regime_detector` rate/inflation/cross-asset regime; OIS/SOFR optionally BSM discount curve | advisory-sizing (OIS/SOFR optionally evaluate-input-correctness) | No (regime context) — **Yes only if** OIS/SOFR wired as the discount curve | PANEL/PLAIN (PANEL if OIS/SOFR→discount) | `staging/macro_rates/{ois_sofr_curve,real_yields,fed_funds,macro_surprise,fx,commodities,global_vol}.csv` | roadmap §7 |
| Sector/factor ETFs OHLCV — 15 ETFs (94,646 rows, 1998→) | `portfolio_risk_gates` R9 sector context + `regime_detector` defensive-sector | downgrade-only *(per the R9-gate consumer, roadmap §6 line 168; not the §7 advisory-sizing row)* | **Yes³ → coupled³** (only if wired into R9) | **CEREMONY** (R9 risk-gate) — PLAIN if regime-display only | `staging/macro_rates/sector_factor_etfs_ohlcv.csv` | roadmap §7 (augments `sp500_sector_etfs.csv`) |

> **VIX file disambiguation (resolves the 0A↔3B asymmetry):** R11 reads the VIX *level* via
> `get_vix_regime`. If that accessor reads `macro_vol/sp500_vol_indices.csv` (this 3B row),
> the coupling is here. **If it reads a series the Phase-0A refresh overwrites**
> (`vix_term_structure.csv` / `sp500_vix_full.csv`), then **0A's VIX-term row is also
> R11-coupled** and should be re-flagged from "No" to coupled. **Confirm `get_vix_regime`'s
> source file at wiring time** and flag 0A accordingly. Superscripts ¹²³ mark EV-impact that
> is **conditional** on the consumer actually being wired into the EV path / S34 portfolio-context.

### 3C — ATM IV term · realized vol · total return → forward distribution (evaluate-input-correctness)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| ATM IV term + realized vol — `atm_iv_{30,60,90,180,365,730}d` + `rv_{10,20,30,60,90,120,180,260}d` (**1,963,364 rows, 510 nm, 2010→06-18**) | `forward_distribution.py` (HAR-RV, RV bootstrap) + `regime_detector` + `tail_risk` + `option_pricer` term structure + `skew_dynamics` | evaluate-input-correctness | **Yes → coupled** | `staging/per_name/vol_term_rv.csv.gz` | roadmap §3 ATM term + realized vol, §4 |
| Total-return index (`tot_return`, div-adjusted) | `forward_distribution` (log-return sampling — removes artificial ex-div jumps) + `tail_risk` (POT-GPD) | evaluate-input-correctness | **Yes → coupled** | `staging/per_name/returns_micro.csv` | roadmap §4 total-return |

> **Depends on #378** (Phase 1) landing first. Note the staleness it guards is **created at
> Phase 0A** (spot → 06-18 while legacy ATM IV stays ~06-04), not here — 3C is where that gap
> finally *closes*, because serving `vol_term_rv` supersedes the un-refreshed legacy ATM IV.
> Until 3C lands, the spot/IV frontiers diverge (see the 0A staleness note). Flag **PANEL**
> (serving feeding `evaluate`).

### 3D — EOD bid/ask → cost model (evaluate-input-correctness)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| EOD bid/ask (`px_bid`, `px_ask`; **px_bid>px_ask = 0.0000%**) — same file as 3C total-return (1,874,882 rows, 511 nm, 2010→06-18) | `engine/cost_model.py` + `EVEngine` cost model (slippage), liquidity gating | evaluate-input-correctness | **Yes → coupled** | `staging/per_name/returns_micro.csv` | roadmap §4 bid-ask spread |

> First real EOD bid/ask history — `sp500_liquidity.csv` never carried it. Replaces a modelled
> spread with an observed one inside the cost model → moves net EV. Flag **PANEL**.

### 3E — Beta + shares-out → copula + single-name cap (evaluate-input-correctness)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| Beta (`beta_raw`) + shares-out (93,605 rows, 510 nm, 2010→, monthly) | `engine/portfolio_copula.py` covariance + risk-gate beta filter; `portfolio_risk_gates` single-name notional cap (**R10**) via `shares_out` | evaluate-input-correctness | **Yes⁴ → coupled⁴** (copula → R7/R8 and R10 on **S34** portfolio-context) | `staging/per_name/beta_shares.csv` | roadmap §4 beta, §6 beta/correlation |

> Touches risk gates → **CEREMONY**. Today only a beta snapshot exists; this is the history.
> Adjusted-beta was not entitled (raw only) — note in the copula calibration.

### 3F — Fundamentals & estimates → advisor committee (advisory-sizing)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| PIT financials (Q) · profitability/leverage/payout/growth (Q) · estimates+analyst (M) · forward estimates (1BF/2BF) · valuation multiples (M) | `advisors/` (Buffett/Munger/Simons committee) + `earnings_drift` | advisory-sizing (downgrade-only) | **No⁵** (advisors are advisory; not in the S27/S32/S34/S35 ranked path) | `staging/per_name/{fundamentals_q,fundamentals_ext_q,estimates_m,estimates_fwd,valuation_m}.csv` | roadmap §5 |

> **PIT caveat:** these are **period-end dated** (filing-lag PIT not captured) — the advisor
> must read them as-of period-end, accepting the lag. Flag **PLAIN** by default — the advisor
> committee is advisory-only and stays out of the ranked S27/S32/S34/S35 path (CLAUDE.md §1),
> so this is not re-baseline-coupled (the "No⁵" status). **Escalate to PANEL only if** the
> committee is wired into the ranked backtest path (then superscript ⁵ → coupled). *(The
> EV-moving slice of "fundamentals" is the dividend-yield carry — see 3G — not these statement
> fields.)*

### 3G — Dividend-yield PIT → BSM carry `q` (evaluate-input-correctness) — the #354 fix

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| Dated dividend-yield (`dvd_yld_12m/_ind/dvd_sh_12m`, **72,461 rows, 421 nm, 2010→2026-05-29, monthly, DATED**) | `data_connector.get_fundamentals` (**add `as_of`**) → `wheel_runner.py:1129` → `ev_engine` BSM `q` | evaluate-input-correctness | **Yes → coupled** (fixes the lookahead — historical `q` stops reading the 2026 snapshot) | `staging/dividend_pit/sp500_dividend_yield_pit.csv` | roadmap T0-3/T0-4, §6 dividend yield, §9 **W-2**, runbook Phase 5 #354 |

> This is the **#354 PIT** unlock: the staged panel is now **dated**, which is exactly what
> #354 needed. Wiring it = the separate trio PR that adds `as_of` to `get_fundamentals` /
> `get_credit_risk` and threads it from `wheel_runner` → **CEREMONY**. The W2 PIT `xfail`
> (`test_fundamentals_credit_are_point_in_time`) flips when this lands.

### 3H — Ratings / GICS / ownership / earnings-timing snapshot → multiple consumers

| Slice of `sp500_snapshot_bdp.csv` (511 rows, **as-of 2026-06-18**) | Engine consumer | §2 role | EV-moving? → re-baseline | Ceremony | Ref |
|---|---|---|---|---|---|
| `rtg_sp/rtg_moody/rtg_fitch` → **new credit reviewer** (junk-grade → **0.85× `ev_dollars`**, no hard refusal; final multiplier set in the §2 panel) | `candidate_dossier.py` reviewer (today credit CSV is **off the EV path**, audit C1) | downgrade-only *(mechanism = `ev_dollars` scaling)* | **Yes → coupled** (wiring builds a live reviewer) | **CEREMONY** | roadmap §6 ratings, §8, audit C1 |
| `gics_sector`/group/industry/sub | informs **#372** R9 — but **#372 sources GICS from main `fundamentals`**, not this snapshot (lookahead) | downgrade-only | Yes → coupled (via #372) | (see #372) | roadmap §5 GICS |
| `inst_pct`/`free_float_pct`/`float_shares` → R9/R10 concentration context | `portfolio_risk_gates` | advisory-sizing | No (context only) | PLAIN | roadmap §6 institutional/float |
| `next_earnings_dt` (`EXPECTED_REPORT_DT`) → earnings lockout refresh | `event_gate` (earnings) | remove-only-gate | **Yes → coupled** | PANEL | roadmap §8 earnings timing |

> **Snapshot caveat (load-bearing):** this is a **single as-of (2026-06-18)**, not history —
> ratings and GICS carry **current** values, so feeding them into a historical backtest is a
> **lookahead**. The credit reviewer must hold the rating flat / back-version it; **#372 must
> take GICS from main `fundamentals.gics_sector_name`** (see Phase 1), not this snapshot.
> Entitlement-blocked: `RATING_WATCH`/outlook all-NaN (manifest bucket F). Credit-reviewer
> wiring touches `candidate_dossier` → **CEREMONY**; the `0.85×` is the roadmap §6 figure.

### 3I — Options structure + news sentiment (mixed: advisory + display-only)

| Slice of `per_name/options_sentiment.csv` (1,998,083 rows, 511 nm, 2010→06-18, **102.3 MB**) | Engine consumer | §2 role | EV-moving? → re-baseline | Ceremony | Ref |
|---|---|---|---|---|---|
| `pc_oi_ratio`/`pc_vol_ratio`/`oi_call`/`oi_put` → dealer/skew advisory | `dealer_positioning` / `skew_dynamics` | advisory-sizing | **Yes⁶ → coupled⁶** (only if wired into the dealer multiplier `[0.70,1.05]`) | **CEREMONY** (dealer mult) — PLAIN if not wired | roadmap §6 options flow |
| `news_sent` → **D18 transparency** | `news_sentiment.py` (dashboard + row dict) | downgrade-only *(display-only in practice)* | **No — display-only, "does NOT influence EV"** (roadmap §8) | PLAIN | roadmap §8 news sentiment |

> Theta per-strike OI is the superior GEX source (roadmap §9 **W-3**, Theta-sourced — **not**
> in the broad-pull); these BBG P/C ratios are a coarser advisory cross-check. **Winsorize**
> `pc_vol`/`news_sent` at load (manifest outlier flags). The dealer-mult touch → **CEREMONY**;
> the news D18 surface → **PLAIN**.

### 3J — Short interest → risk_manager crowded-short / R10 (advisory-sizing)

| Dataset | Engine consumer | §2 role | EV-moving? → re-baseline | Banked at | Ref |
|---|---|---|---|---|---|
| Short interest + days-to-cover (`short_interest`, `short_int_ratio`; **134,035 rows, 509 nm, 2015-01-15→2026-05-29, biweekly**) | `risk_manager.py` crowded-short + **R10** single-name squeeze soft-warn (**no engine consumer exists today** — wiring builds one) | advisory-sizing | **Yes → coupled** (a new soft-warn changes verdicts on S34) | `staging/short_interest/sp500_short_interest.csv` | roadmap T0-12, §6 short interest |

> **Entitlement gap (manifest bucket F):** `pct_of_float` + borrow rate are **all-NaN** (no SLB
> entitlement); only raw SI + DTC are usable. `pct_of_shares_out` is derivable later from
> `EQY_SH_OUT` (Phase 3E `shares_out`). Building a new soft-warn touches `risk_manager` →
> **CEREMONY**.

---

## Phase R — The single re-baseline (operator-merge, CEREMONY)

Runs **once**, after every EV-moving step above has landed. Captures, in one pass: the
Phase-0A data change, the latent **#363** `ev_mean` re-pricing, the **Phase-1 (E)** fixes,
and **every Phase-2/3 `ev_raw` shift**. Procedure (runbook Phase 3/4):

1. **Expect the #340 fingerprint guard to go red first** — `test_snapshot_data_fingerprint_matches_current`. This is the signal, not a bug (it hashes raw bytes → catches Phase-0A; the serving-logic re-pricings surface as `ev_mean`-only marker drift, caught at unit speed by the W14 served-output test).
2. **Re-pin all four snapshots** (~4 h) — `python -m backtests.regression.{s27_ivpit_24t_100k, s32_friction_24t_1m, s34_universe_100t_1m, s35_oos_24t_100k} --update-snapshot`. Capture via the full-metric diff harness; per-snapshot attribution line (frontier Δ / #363 `ev_mean` / which (E)/wiring step). Tag RA-style worklog, **not** a D-number.
3. **Confirm the markers** (~2.5 h) — `pytest tests/test_backtest_regression.py -m backtest_regression` (byte-identical to the regenerated snapshots). Gate other runs with `-m "not backtest_regression"` (bare `pytest tests/` hangs on S34, #367).
4. **Bump `EXPECTED_FRONTIER`** (`tests/test_preflight_environment.py`) **and the data-test `FRONTIER`** (`tests/test_data_to_engine.py:39` + `test_data_integrity_bloomberg.py`) to `2026-06-18` — Phase 0A moved the frontier.
5. **Re-pick W16/W30** JPM earnings-window names *only if the frontier aged JPM out* (probe `get_next_earnings` within ~40 d of the new frontier); **re-pin the full-universe 480/31 split**.
6. **Flip the xfail trackers** as each lands (#354 W2 PIT on the 3G `as_of`; #355 blue-chip if backfilled; #372's W17 characterization).
7. **Clear S34's ⚠️ provisional flag** and run the **§2 review panel** on the connector/universe change *and on each (E) fix*. **Operator-merge, no auto-update of snapshots.**

---

## 3. Map to the roadmap §9 wiring list (what the broad-pull changes)

| §9 | Item | Status after broad-pull |
|---|---|---|
| **W-1** | Wire deep 5×5 IV-surface archive → connector | **Superseded** — Phase 2 wires the *fresh committed* surface (`staging/iv_surface/`), no gitignored-archive dependency. |
| **W-2** | Fix PIT dividend lookahead | **Data ready** — Phase 3G wires the dated `dividend_pit` panel + `as_of` (the #354 unlock). |
| **W-3** | Wire Theta per-strike OI → `DealerPositioningAnalyzer` | **Out of broad-pull** (Theta-sourced). The BBG P/C ratios (3I) are a coarse cross-check only. |
| **W-4** | Land D19 exit-cost + D21 DTE→bars + recal | **Independent** decision change, **out of scope** for this data-wiring campaign. It is re-baseline-coupled too: to "ride the same Phase-R re-baseline" it must be given its **own pre-Phase-R CEREMONY landing** alongside the (E) trio; otherwise defer it to a **separate** re-baseline (do not assume it rides Phase R implicitly). |
| **W-5** | Wire corporate-actions into the event gate | **Data already present** on `main` (**52,442 rows** — not the 2-byte stub the stale inventory claimed); no restore/pull. The `event_gate` wiring is **scheduled as a Phase 3A row** above (EV-moving → re-baseline-coupled). |
| **W-6** | Pre-2018 OHLCV backfill from deep archive | **Out of broad-pull** (deep archive, gitignored). Separate effort; also re-baseline-coupled. |

---

## 4. Out of scope for this document (deliberately)

This is a plan. It does **not**: write or touch any engine/connector/wiring code; edit
the decision trio (`ev_engine.py` / `wheel_runner.py` / `candidate_dossier.py`) or any
risk gate; run or re-pin the re-baseline; move, integrate, or edit any data file; or
modify the (E) trio fixes. Every step above is described, not implemented. Execution
waits for the supervised session per `docs/NEXT_DATA_SESSION_RUNBOOK.md`.

## 5. Cross-references

- `staging/BROAD_PULL_MANIFEST.md` — banked bytes (per-dataset rows/ranges/caveats; entitlement bucket F).
- `docs/DATA_ACQUISITION_ROADMAP.md` — §1 (the §2 roles), §3–§8 (consumer mappings), §9 (W-1…W-6), §10 (quick-start), §11 (legend).
- `docs/NEXT_DATA_SESSION_RUNBOOK.md` — Phase 2 (the (E) trio) + Phase 3/4 (re-baseline) + the governing rule.
- `docs/DATA_TEST_AUDIT_2026-06-09.md` — C1 (credit off EV path), C2 (R9 map), C3 (VIX→R11), C4 (`edge_vs_fair`≡0); W17/W27/W36/W37.
- `docs/DATA_INVENTORY.md` §6 — the staged-data census (byte-verified input map for every row above).
- `CLAUDE.md` §2 — the invariant.
