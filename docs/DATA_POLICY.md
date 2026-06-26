# Data Policy

How market data, derived features, credentials, and analyst
deliverables flow through this repo — what is tracked, what is
ignored, what is regenerable, and what must never leave the laptop.

This file is the operational counterpart to the data-layer entry
in `CLAUDE.md`. For the **disk-verified inventory** of every dataset we
actually hold (counts, date ranges, per-dataset columns + dtypes) see
`docs/DATA_INVENTORY.md`; for the forward-looking schema design see
`DATA_SPECIFICATION.md`. The full provider capability matrix lives in §2 below.

---

## 1. Three data tiers

| Tier | Lives at | Tracked in git? | Regen path |
|---|---|---|---|
| **Raw** | `data/bloomberg/*.csv`, `data_raw/` | mostly **yes** (committed CSVs) | yfinance pulls + Bloomberg Terminal exports |
| **Processed** | `data_processed/theta/**`, `data_processed/vol_indices*.parquet` | **no** (gitignored) | `scripts/pull_all.py` on a Theta-up laptop |
| **Derived** | `data/features/<group>/ticker=<X>/` | **AAPL only** as a sample | `scripts/backfill_features.py` |

The committed Bloomberg CSVs are the **fallback** for any environment
that can't reach Theta Terminal (Cowork sandbox, CI, a fresh laptop
without the Terminal up). They are also point-in-time records of what
the engine ran on, which has audit value.

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

| Capability | `bloomberg` (CSVs in git) | `theta` (live Terminal) |
|---|---|---|
| Historical OHLCV | ✅ `data/bloomberg/sp500_ohlcv.csv` | ✅ stock EOD |
| IV history | ✅ `sp500_vol_iv_full.csv` (ATM) | ⚠ snapshot only — v3 IV/greeks **history is 404/not-entitled**; the `theta/iv_history` series we hold is `source=bloomberg` |
| Liquidity | ✅ `sp500_liquidity.csv` | ⚠ derived |
| Fundamentals | ✅ `sp500_fundamentals*.csv` | ❌ (not in v3) |
| Option chains (live) | ❌ | ✅ requires Terminal @ `127.0.0.1:25503` |
| First-order greeks | ❌ | ✅ live snapshot only (no greeks **history** — 404) |
| VIX / SKEW EOD | ✅ | ✅ (EOD only — snapshots blocked) |
| VIX futures (UX1–UX8) | ❌ | ❌ (tier-blocked) |
| Corporate actions | ❌ `sp500_corporate_actions.csv` is an **empty 2-byte stub** (dividends held separately in `sp500_dividends.csv`) | ❌ (not in v3) |

Read this matrix before assuming a feature works on both providers.
Per-strike greeks/IV exist on disk only as the 2026 `theta/chains` + `iv_surface`
snapshots and the back-solved `iv_history`; there is **no** Theta greeks/IV time series.

---

## 3. What never enters git

**Credentials and installed software:**

| Path | Why |
|---|---|
| `Theta/` (entire dir) | Installed software (~239MB jars) + creds.txt (Theta API token) + config.toml. See `DECISIONS.md` D6. |
| `**/ThetaTerminal*.jar` | Defence-in-depth: catches the jar even outside `Theta/`. |
| `*/creds.txt` | Catches credential files in any subdir. |
| `.env`, `.envrc` | Standard env-secret convention. |
| `secrets/`, `credentials/`, `*.credentials`, `*_credentials.json`, `service_account*.json` | Standard credential file conventions. |

**Large regenerable artefacts:**

| Path | Regen via | Size |
|---|---|---|
| `data_processed/` | `scripts/pull_all.py` | many GB across theta sub-dirs |
| `data/features/**/ticker=*/` (except AAPL) | `scripts/backfill_features.py` | ~1.2 GB |
| `dashboard/node_modules/` | `npm install` | ~hundreds of MB |
| `dashboard/.next/` | `npm run build` or `npm run dev` | tens of MB |
| `tradingview/tradingview-mcp-jackson/` (nested repo + node_modules) | clone the repo + `npm install` | ~26 MB node_modules |

**Generated output that accumulates:**

| Path | Origin |
|---|---|
| `tradingview/research/*.docx` | Analyst workflow deliverables |
| `tradingview/research/*.pdf` | Analyst workflow deliverables |
| `tradingview/models/*.xlsx`, `tradingview/models/*.csv` | Analyst workflow deliverables |
| `tradingview/screenshots/` | TradingView MCP screenshots |
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
| Theta full pull | `python scripts/pull_all.py` | daily on laptop | requires Terminal up; ~8 hours wall clock; see `THETA_PULL_SESSION_NOTES.md` |
| yfinance fundamentals + earnings | `python scripts/pull_fundamentals_yf.py`, `pull_earnings_yf.py` | weekly | refreshes the committed CSVs |
| Treasury yields | `python scripts/pull_treasury_yields_yf.py` | weekly | refreshes `treasury_yields.csv` |
| Feature shards | `python scripts/backfill_features.py` | when feature def changes | regenerates `data/features/`; AAPL stays as sample |
| Bloomberg Terminal exports | Excel macros in `scripts/bloomberg_*.bas` | monthly | manual; produces fresh wide-format panels |
| Theta capability probe | `python scripts/probe_theta_capabilities.py` | when tier coverage changes | regenerates `data_processed/theta_capabilities.json` (the tier map). Run if the file is absent or after a Theta v3 plan change. |

The yfinance refreshes mutate tracked CSVs in place. **Policy (ROADMAP C1,
decided 2026-05-30): keep tracking — treat each refresh as a data commit.**
The point-in-time "what data did we run on?" audit trail outweighs the
commit-per-refresh history noise, so `sp500_earnings_yf.csv`,
`sp500_fundamentals_yf.csv`, and `treasury_yields.csv` stay tracked.

---

## 6. Drive-mount caveats

The repo lives on a Google Drive mount. Two operational rules apply:

1. **Drive is an eventually-consistent mirror, not a source of
   truth.** A branch the laptop hasn't checked out won't appear on
   Drive. Always resolve via `git fetch origin && git checkout`,
   not `ls`. To read a newer revision without checking it out, use
   `git show origin/<branch>:<path>`.
2. **Drive denies `unlink` on existing tracked files.** `git pull`
   may fetch refs but fail to update the worktree with
   `error: unable to unlink old '<file>': Operation not permitted`.
   Workaround: when this hits, fetch + read via
   `git show origin/<branch>:<path>` and apply manually, or do the
   pull on the laptop (where unlink works) and let Drive sync.

These rules are surfaced again in `docs/LAPTOP_SETUP.md`; the
data-refresh use case hits them constantly so the canonical version
lives here.

---

## 7. Sandbox-vs-laptop capability

| Operation | Cowork sandbox | Laptop with Terminal |
|---|---|---|
| `WheelRunner.rank_candidates_by_ev` (5 explicit tickers) | ✓ ~2s | ✓ |
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
