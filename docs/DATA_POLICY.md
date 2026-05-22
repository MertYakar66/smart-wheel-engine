# Data Policy

How market data, derived features, credentials, and analyst
deliverables flow through this repo — what is tracked, what is
ignored, what is regenerable, and what must never leave the laptop.

This file is the operational counterpart to `CLAUDE.md` §3
(provider matrix) and `DATA_SPECIFICATION.md` (schemas).

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

Read in `engine/wheel_runner.py:130` and
`scripts/diagnose_candidates.py:59`. The `.claude/settings.json`
SessionStart hook prints a warning when the variable is unset and
defaults to `bloomberg`.

The full per-capability matrix (chains / IV history / fundamentals /
corporate actions / VIX / etc.) lives in `CLAUDE.md` §3 — read it
before assuming a feature works on both providers.

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

The yfinance refreshes mutate tracked CSVs in place. Treat each
refresh as a data commit (see `ROADMAP.md` C1 for the open question
on whether to gitignore them instead).

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

`CLAUDE.md` §3 has the canonical version of these rules; this file
duplicates them because data refresh sessions hit them constantly.

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
6. Add a row to `CLAUDE.md` §3 capability matrix and to §1 of this
   file.
