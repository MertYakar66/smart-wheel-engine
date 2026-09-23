# Data Policy

How market data, derived features, credentials, and analyst
deliverables flow through this project — where the data lives (the
operator's desktop, never git — DECISIONS D31), what is regenerable,
and what must never leave the desktop.

This file is the operational counterpart to the data-layer entry
in `CLAUDE.md`. For the verified per-file census of what is actually
on disk, see `DATA_INVENTORY.md`. (The old partitioned-Parquet schema
design was aspirational, never matched on-disk reality, and is
archived at `archive/2026-06/DATA_SPECIFICATION.md`.) The full
provider capability matrix lives in §2 below.

---

## 1. Three data tiers — all under one data root

| Tier | Lives at (relative to the data root) | In git? | Source / regen path |
|---|---|---|---|
| **Raw** | `data/bloomberg/*.csv` (served monoliths), `data/bloomberg/broad_pull/`, `data/bloomberg/deep/`, `data_raw/` (yfinance pulls, constituents, day-bot ticks) | **no** (D31; the served files stay tracked on `main` only until the step-5 untracking PR lands) | Bloomberg Terminal exports (frozen — the Terminal is no longer available), yfinance pulls, the day-bot |
| **Processed** | `data_processed/theta/**`, `data_processed/vol_indices*.parquet`, `data_processed/option_premium/`, `data_processed/ibkr/`, `data_processed/sim/` | **no** | `scripts/pull_all.py` on a Theta-up machine; each rail regenerates its own store |
| **Derived** | `data/features/<group>/ticker=<X>/` | **no** | `scripts/backfill_features.py` |

**The data root.** Every path above is relative to the data root: the
folder `SWE_DATA_ROOT` names, or — when the variable is unset — the
current working directory, which is the repository folder when you run
from it. That unset case is exactly the behaviour every caller had
before, so nothing changes for a machine that keeps its data inside the
checkout. `engine/paths.py` is the single resolver (`paths.resolve`,
`paths.bloomberg_dir()`, `paths.processed_dir()`, …); the narrower
overrides that already existed — `SWE_DATA_PROCESSED_DIR`,
`SWE_IBKR_DATA_DIR`, `SWE_OPTION_PREMIUM_DIR`, `SWE_SIM_DATA_DIR` —
keep winning inside their scope. Section 6 describes the desktop set-up.

**The manifest.** `data/DATA_MANIFEST.json` lists every dataset the
engine owns — path, byte size, sha256, dataset group and the git object
it was first taken from (144 files, 1.74 GB: generated 2026-09-18 from
the git objects themselves, completed 2026-09-23 with the feature sidecars
and the `data_archive/` branch archive — `docs/DATA_INVENTORY.md` §C.2). It is the point-in-time record of what the
engine ran on — the audit value the data commits used to carry, without
the bytes. `python scripts/data_manifest.py check --root <root>` proves
a root complete; `materialize` fills one from git; `build` regenerates
the manifest after a refresh (and carries the ledger forward).

---

## 2. Provider selection — `SWE_DATA_PROVIDER`

| Value | Connector class | Where read |
|---|---|---|
| `bloomberg` (default if unset) | `MarketDataConnector` | `engine/data_connector.py` |
| `theta` | `ThetaConnector` | `engine/theta_connector.py` |

Read by `WheelRunner.connector` (lazy-load property in
`engine/wheel_runner.py`) and by `scripts/diagnose_candidates.py`'s
provider resolution. The `.claude/settings.json` SessionStart hook
prints a warning when the variable is unset and defaults to
`bloomberg`.

### Capability matrix

| Capability | `bloomberg` (CSV panels under the data root) | `theta` (live Terminal) |
|---|---|---|
| Historical OHLCV | ✅ `data/bloomberg/sp500_ohlcv.csv` | ✅ stock EOD |
| IV history | ✅ `sp500_vol_iv_full.csv` (ATM) | ⚠ snapshot only — v3 IV/greeks **history is 404/not-entitled**; the `theta/iv_history` series we hold is `source=bloomberg` |
| Liquidity | ✅ `sp500_liquidity.csv` | ⚠ derived |
| Fundamentals | ✅ `sp500_fundamentals*.csv` | ❌ (not in v3) |
| Option chains (live) | ❌ | ✅ requires Terminal @ `127.0.0.1:25503` |
| First-order greeks | ❌ | ✅ live snapshot only (no greeks **history** — 404) |
| VIX / SKEW EOD | ✅ | ✅ (EOD only — snapshots blocked) |
| VIX futures (UX1–UX8) | ❌ | ❌ (tier-blocked) |
| Corporate actions | ✅ | ❌ (not in v3) |

Read this matrix before assuming a feature works on both providers.
Per-strike greeks/IV exist on disk only as the 2026 `theta/chains` + `iv_surface`
snapshots; the `theta/iv_history` series is ATM-only with `source=bloomberg` (not
Theta-native), so there is **no** Theta greeks/IV history time series.

---

## 3. What never enters git

**Market data — none of it (D31, 2026-09-18).** Not the Bloomberg
panels, not the deep slices, not the day-bot ticks, not the feature
shards, not the Theta corpus, and not any future collection. The data
lives under the data root on the operator's desktop; Google Drive is
the backup (delayed, not a source of truth). The only data-shaped files
git holds are `data/DATA_MANIFEST.json` (hashes, not bytes) and the small
synthetic fixtures under `tests/fixtures/`. A refresh therefore ends
with `python scripts/data_manifest.py build --root <root>` and a commit
of the manifest, never of the data. (Status: the served monoliths and
the small `data_raw/` / `data/features` samples are still tracked on
`main` until the step-5 untracking PR lands — see `DATA_INVENTORY.md`
§A for the live status of each step.)

**Credentials and installed software:**

| Path | Why |
|---|---|
| `Theta/` (entire dir) | Installed software (~239MB jars) + creds.txt (Theta API token) + config.toml. See `DECISIONS.md` D6. |
| `**/ThetaTerminal*.jar` | Defence-in-depth: catches the jar even outside `Theta/`. |
| `*/creds.txt` | Catches credential files in any subdir. |
| `.env`, `.envrc` | Standard env-secret convention. |
| `secrets/`, `credentials/`, `*.credentials`, `*_credentials.json`, `service_account*.json` | Standard credential file conventions. |

**Data stores and large regenerable artefacts (all under the data root):**

| Path | Regen via | Size |
|---|---|---|
| `data/bloomberg/`, `data_raw/` | not regenerable — the Terminal is gone; `materialize` from git until step 5, then the desktop root + Drive | 536 MB + 365 MB deep + 491 MB ticks |
| `data_processed/` | `scripts/pull_all.py` | many GB across theta sub-dirs |
| `data/features/**/ticker=*/` | `scripts/backfill_features.py` | ~1.2 GB |
| `dashboard/node_modules/` | `npm install` | ~hundreds of MB |
| `dashboard/.next/` | `npm run build` or `npm run dev` | tens of MB |

**Generated output that accumulates:**

| Path | Origin |
|---|---|
| `tradingview/screenshots/` | Chart screenshots for the filesystem / Playwright chart providers (`engine/tradingview_bridge.py`); the analyst workspace that produced other deliverables here was removed 2026-09-17 (D30) |
| `__pycache__/`, `*.pyc` | Python bytecode |
| `.pytest_cache/`, `.hypothesis/` | Test machinery |
| `.DS_Store` | macOS Finder metadata |

If you find any of the above tracked in git, untrack with
`git rm --cached <path>` rather than deleting from disk — the local
file may still be needed.

---

## 4. Point-in-time discipline

The engine treats every input as point-in-time. Two consequences:

1. **No look-ahead.** A backtest on date `T` may not consume data
   that became known after `T`. `tests/test_point_in_time.py` pins
   the no-lookahead invariant; consult it before adding a new data
   source that has any backfill or revision history.
2. **Provenance over freshness.** When a value is reported (e.g.
   "AAPL EPS 2024:Q4 = 1.46"), the date column on that row is the
   date the value was *first known*, not the as-of date of the
   period. yfinance / Bloomberg occasionally back-revise; revisions
   show as a new row with a later announcement date, not as an
   in-place edit.

The Bloomberg CSVs you'll find in `data/bloomberg/` are wide-format
panels keyed by `(ticker, period, announcement_date)`. The
`announcement_date` column is the PIT key.

---

## 5. Refresh procedures

| Refresh | Command | Frequency | Notes |
|---|---|---|---|
| Theta full pull | `python scripts/pull_all.py` | daily on laptop | requires Terminal up; ~8 hours wall clock; see `THETA_USAGE.md` §20 |
| yfinance fundamentals + earnings | `python scripts/pull_fundamentals_yf.py`, `pull_earnings_yf.py` | weekly | refreshes the CSVs under the data root; rebuild + commit the manifest afterwards |
| Treasury yields | `python scripts/pull_treasury_yields_yf.py` | weekly | refreshes `treasury_yields.csv` |
| Feature shards | `python scripts/backfill_features.py` | when feature def changes | regenerates `data/features/`; AAPL stays as sample |
| Bloomberg Terminal exports | Excel macros in `scripts/bloomberg_*.bas` | monthly | manual; produces fresh wide-format panels |
| Theta capability probe | `python scripts/probe_theta_capabilities.py` | when tier coverage changes | regenerates `data_processed/theta_capabilities.json` (the tier map). Run if the file is absent or after a Theta v3 plan change. |

The yfinance refreshes mutate the CSVs under the data root in place.
**Policy (D31, 2026-09-18 — supersedes the ROADMAP C1 "keep tracking"
policy of 2026-05-30):** a refresh is never a data commit. The
point-in-time "what data did we run on?" audit trail is the manifest:
after any refresh run `python scripts/data_manifest.py build --root
<root>` and commit `data/DATA_MANIFEST.json` (hashes and sizes, a few
hundred KB) — the ledger says exactly which bytes the engine ran on,
and `check` proves any copy against it.

> **The `*_yf.csv` files are currently UNCONSUMED parallel files.** The
> connector (`engine/data_connector.py`) reads `sp500_fundamentals.csv`
> and `sp500_earnings.csv` (the Bloomberg files) — NOT their `_yf`
> counterparts. Running `pull_fundamentals_yf.py` / `pull_earnings_yf.py`
> refreshes the parallel files but does not change engine behaviour until
> a merge/consume step is wired (not yet done). **Forward earnings dates**
> are instead served by the broad-pull snapshot overlay (below): the
> `_yf` earnings file was deliberately passed over for that role — no
> knowledge-date stamp, ~70 % forward coverage vs the snapshot's 100 %,
> and its 18-year history diverges from the Bloomberg record inside
> pinned backtest windows (a naive union rewrites history).

> **⚠ Earnings-calendar overlay — refresh + bump on every broad-pull
> snapshot re-pull.** `get_next_earnings` / `get_recent_earnings` overlay
> `broad_pull/per_name/sp500_snapshot_bdp.csv::next_earnings_dt` (PIT-gated
> on its `asof` column) to feed the live earnings lockout — this is the fix
> for the D3-1 collapse (the Bloomberg earnings file carries forward dates
> for only ~39/511 names). The overlay **fails OPEN as it ages**: its dates
> fall behind the wall clock and simply stop registering, so the lockout
> silently decays back toward ~8 % coverage roughly one quarter after the
> snapshot date (the current 2026-06-18 snapshot covers announcements
> through 2026-09-25). On **every broad-pull snapshot refresh**, bump
> `EXPECTED_EARNINGS_CALENDAR_ASOF` in `tests/test_preflight_environment.py`
> in the same commit; before any live `as_of=None` use, run the opt-in
> age check: `SWE_LIVE_PREFLIGHT=1 pytest tests/test_earnings_calendar_overlay.py`.

> **⚠ Producer census (refreshed 2026-07-08, post the #477 xbbg-puller
> salvage — closes the D28 parked item).** Of the **10 monolith files** in
> `engine/data_connector.py::_FILES` (plus the two `broad_pull/` panels it
> reads outside `_FILES` — the PIT dividend-yield panel and the #464
> `sp500_snapshot_bdp.csv` earnings overlay, both fingerprint-pinned since
> #465), **9 now have a runnable in-repo producer**; every `xbbg` puller
> needs a logged-in Bloomberg Terminal:
>
> - `sp500_ohlcv.csv` / `sp500_liquidity.csv` — `pull_ohlcv.py` /
>   `pull_liquidity.py` (⚠ legacy hardcoded `end_date="2026-03-20"`;
>   edit before running or the frontier *regresses*).
> - `treasury_yields.csv` — `pull_treasury_yields_yf.py` (yfinance,
>   no Terminal needed).
> - `sp500_vol_iv_full.csv` — `pull_vol_iv.py` (salvaged; pins
>   `end_date="2026-06-04"`, the current frontier).
> - `sp500_dividends.csv` — `pull_dividends.py`;
>   `vix_term_structure.csv` — `pull_vix_term_structure.py`;
>   `sp500_corporate_actions.csv` — `pull_corporate_actions.py` (the
>   manual BQL recipe at `scripts/bloomberg_bql_pulls.md` §2 remains the
>   documented historical alternative; Theta's corp-actions endpoints
>   404 at this tier).
> - `sp500_credit_risk.csv` / `sp500_fundamentals.csv` —
>   `pull_snapshots.py` (snapshot-mode BDP fields).
>
> The one file still **without** a producer is `sp500_earnings.csv`
> (the BDS earnings backfill — deferred; queue tracked in
> `docs/NEXT_DATA_SESSION_RUNBOOK.md`). Pre-salvage per-file
> investigation (historical):
> [`bloomberg_refresh_runbook.md`](bloomberg_refresh_runbook.md).

> **⚠ Preflight frontier guard — bump on every OHLCV refresh.** On **every
> OHLCV refresh** (any change that moves `sp500_ohlcv.csv`'s most-recent bar),
> **bump `EXPECTED_FRONTIER` in `tests/test_preflight_environment.py` in the
> same commit.** That constant pins the date the bundled OHLCV is expected to
> reach; the preflight guard (`test_bundled_ohlcv_reaches_expected_frontier`)
> fails loud on a tree ending earlier — the stale-clone / wrong-tree class of
> mistake. Keep the two in lockstep: refresh-without-bump lets the guard rot
> (it passes on stale data); bump-without-refresh makes it false-fail.

> **⚠ Runtime frontier staleness (D1-2/D3-2, 2026-07-03).** The pin above
> catches stale TREES; a current tree with old data is caught at RUNTIME:
> `get_data_frontier` warns once per connector when the frontier is > 7
> days behind the wall clock, the three rankers attach a structured
> `attrs["staleness"]` (surfaced on `/api/candidates` and as
> `frontier_age_days` on `/api/status`), and a live deployment can arm
> **`SWE_REFUSE_STALE_LIVE=1`** (or `refuse_stale_live=True`) to
> hard-refuse `as_of=None` ranks on a stale frontier — default OFF
> (warn-and-rank; a default refuse would blank the book, the #462 lesson).
> Dated backtests are untouched (wall-clock reads gate on `as_of=None`).

---

## 6. The data root — desktop home, first fill, backup

The operator's main Windows desktop is the home of every dataset (D31).
The data root is one folder, preferably **outside** the repository
folder (a checkout must be disposable; the data is not), holding the
three conventional trees:

```
D:\swe-data\
  data\bloomberg\            served monoliths, broad_pull\, deep\
  data\features\             feature shards
  data_raw\                  yfinance pulls, constituents, bloomberg\ticks\
  data_processed\            theta\, option_premium\, ibkr\, sim\, vol_indices*.parquet, …
  data_archive\              what only the non-main branches held (read by no code);
                             git\ holds the full-history bundle of every branch
```

`SWE_DATA_ROOT` names it. Unset, the engine reads the same trees relative
to the current working directory — the repository folder when you run
from it — so a desktop that keeps its data inside the checkout keeps
working unchanged; the root is what lets the checkout become disposable.

**First fill, in this order (each step verified before the next):**

```powershell
# 1. the non-main branches hold the deep slices, the day-bot ticks and the archive rows
git fetch origin deep-history/bloomberg-raw claude/daybot-bloomberg-pull backup/drive-tier-c-2026-07-22 data/drive-migration
# 2. create every manifest file that is missing from the root, byte-verified;
#    nothing that already exists is overwritten (a differing file is reported)
python scripts/data_manifest.py materialize --root D:\swe-data
# 3. prove the root: expect "<N> ok, 0 missing, 0 mismatched" — N = the manifest's rows (144 on 2026-09-23)
python scripts/data_manifest.py check --root D:\swe-data
python scripts/data_manifest.py census --root D:\swe-data
# 4. move (or copy) the local-only stores under the root — data_processed\theta,
#    data_processed\option_premium, data\features shards, data_processed\ibkr —
#    or leave them where they are and point SWE_DATA_PROCESSED_DIR /
#    SWE_OPTION_PREMIUM_DIR / SWE_IBKR_DATA_DIR at them.
# 5. make the root known to every new shell, then run the OPERATING_MODEL §9.4 smoke
[Environment]::SetEnvironmentVariable("SWE_DATA_ROOT", "D:\swe-data", "User")
```

Only after step 5 pull the commit that untracks the data: `git pull` of an
untracking commit removes the *tracked* copies from the working tree (git
sees a deletion), which is harmless once the root outside the checkout is
verified and in use. The step-5 PR is held until the operator reports the
`check` line from step 3.

**Backup.** Google Drive holds a verified copy of the served monoliths and
the deep slices and a copy of the local-only stores (`DATA_INVENTORY.md`
§C); the day-bot ticks have no Drive copy yet and must get one before their
branch is deleted. Drive is a backup, never a source of truth: the engine
reads the desktop root, and `check` — not a folder listing — is what
"the copy is complete" means.

**Cadence.** `check --size-only` takes seconds and belongs in any
bring-up; a full `check` after every refresh; `build` + a manifest commit
after every refresh (§5).

**Environments without a data root** (CI, Cowork sandboxes, a fresh
clone) run the unit lane only: tests that need the data carry the
`requires_data` marker and skip visibly (`TESTING.md`); the §9.4 smoke
cannot run there. Nothing fetches data from Drive automatically.

---

## 7. Sandbox-vs-laptop capability

| Operation | Cowork sandbox | Laptop with Terminal |
|---|---|---|
| `WheelRunner.rank_candidates_by_ev` (5 explicit tickers) | ✓ ~2s when a data root is present; ✗ otherwise (D31: sandboxes hold no data) | ✓ |
| Data-backed tests (`requires_data` marker) | ✗ skip visibly without a data root | ✓ |
| `scripts/diagnose_candidates.py` (full universe) | ✗ exceeds 45s timeout | ✓ ~3 min |
| Theta chains / Greeks / IV surfaces | ✗ no Terminal | ✓ |
| `pip install -r requirements.txt` (full) | ✗ exceeds 45s timeout — batch in 3 chunks | ✓ |
| Parquet reads | ✗ until `pip install pyarrow --break-system-packages` | ✓ |
| Live MCP-driven TradingView analysis | ✗ no GUI | ✓ |

The SessionStart hook validates the laptop-side prerequisites on
every fresh Claude session. In Cowork it warns about the missing
provider and falls back to the bloomberg-CSV path.

**Sandbox operational notes:**

- **`pyarrow` is not pre-installed in a fresh Cowork bash sandbox.**
  Any parquet read dies with `Unable to find a usable engine` until
  you run `pip install pyarrow --break-system-packages`.
- **`pip install -r requirements.txt` exceeds the 45 s bash timeout.**
  Batch the installs: `scipy` alone, then
  `statsmodels arch scikit-learn`, then `yfinance pydantic`. The
  SessionStart hook can handle the batching when configured.
- **The 45 s rule applies to any single bash call** — if it can't
  finish in 45 s, chunk it and pass state through workspace files.
  Bash calls don't share state across invocations, so `nohup &` does
  not persist.

---

## 8. Adding a new data source — checklist

Before introducing a new data source (new connector, new external
adapter, new RSS feed):

1. Decide which **provider tier** it belongs to (raw / processed /
   derived). Update §1.
2. Decide whether it goes in git (raw, small, PIT) or stays out
   (processed, large, regenerable).
3. If it has credentials, add the credential file to `.gitignore`
   **before** the first pull.
4. If it touches the EV path, register it as a downgrade-only
   reviewer per `DECISIONS.md` D1 — never as a direct ranker input.
5. Add a `pull_*.py` under `scripts/` and wire it into
   `scripts/pull_all.py` so daily refreshes pick it up.
6. Add a row to the capability matrix in §2 above and to §1 of
   this file.
