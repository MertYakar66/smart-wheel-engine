# CLAUDE.md — Session orientation for Smart Wheel Engine

You are working on a probabilistic expected-value (EV) decision engine for
wheel strategies (short cash-secured puts → covered calls) on S&P 500 names.
This file is the single source of orientation for a fresh Claude session.
Read it in full before acting.

For machine/environment rehydration (cloning, Theta Terminal, regenerating
local data), see `LAPTOP_SETUP.md`. This file is about **how the product
thinks**, not how to install it.

---

## 1. Mental model — four layers

1. **Data layer** (`data/`, `data_processed/`, `scripts/pull_*.py`)
   Point-in-time OHLCV, option chains, IV, fundamentals, macro, news.
   Two providers; selected by the `SWE_DATA_PROVIDER` env var (default
   `bloomberg`). See §3.

2. **Quant layer** (`engine/`)
   Black-Scholes-Merton pricing + Greeks to 3rd order, empirical forward
   distributions (non-overlapping → block bootstrap → HAR-RV cascade),
   POT-GPD tail risk, 4-state Gaussian HMM regime detection, Nelson-Siegel
   skew, Student-t copula portfolio CVaR, SVI surface fitting, dealer
   GEX/walls/gamma-flip positioning.

3. **Decision layer** — the authoritative ranker.
   - `engine/ev_engine.py` — `EVEngine.evaluate(...)` at line 234. Runs
     event lockout → forward distribution → cost model → regime &
     dealer multipliers → returns `EVResult`.
   - `engine/wheel_runner.py` — `WheelRunner.rank_candidates_by_ev(...)`
     at line 445. The one ranker every tradeable path routes through.
   - `engine/candidate_dossier.py` — `EnginePhaseReviewer` (line 113)
     applies the 6 downgrade rules (R1–R6).

4. **Interface layer**
   - `engine_api.py` — HTTP API on `:8787` (stdlib `ThreadingHTTPServer`).
   - `dashboard/` — Next.js dashboard (node_modules gitignored).
   - `engine/tradingview_bridge.py` — `FilesystemChartProvider`,
     `PlaywrightChartProvider`, `ChainedChartProvider`. Chart is a
     sanity check, not a decider.
   - `advisors/` — Buffett/Munger/Simons/Taleb committee. Advisory only.
   - Local Ollama for memo/summary generation (72B / 32B models).

---

## 2. The hard invariant — never break this

**No tradeable candidate bypasses `EVEngine.evaluate`.**

Chart providers, news sentiment, the advisor committee, the dealer
positioning overlay, and the TradingView bridge can all **downgrade** a
verdict (negative → blocked, proceed → review, proceed → skip). None of
them can **rescue** a negative-EV trade. The dealer multiplier is clamped
to `[0.70, 1.05]` and only scales the final `ev_dollars` — it never
touches `ev_raw`.

If you add a new input (new data source, new advisor, a TradingView MCP
chart feed), wire it as a participant in a chained provider or as a
downgrade-only reviewer. Do not introduce a code path that converts a
non-tradeable candidate into a tradeable one without a fresh
`EVEngine.evaluate` call.

The `EnginePhaseReviewer` rules, for reference:
- R1: negative EV → blocked (hard stop)
- R2: chart missing → review
- R3: spot mismatch > 2% → skip
- R4: phase contradiction → skip
- R5: EV above threshold → proceed
- R6: short-gamma regime + strike at/above put wall → downgrade

---

## 3. Data-provider matrix

`SWE_DATA_PROVIDER` env var, read in `engine/wheel_runner.py:130` and
`scripts/diagnose_candidates.py:59`. Default: `bloomberg`.

| Capability | `bloomberg` (CSVs in git) | `theta` (live Terminal) |
|---|---|---|
| Historical OHLCV | ✅ `data/bloomberg/sp500_ohlcv.csv` | ✅ stock EOD |
| IV history | ✅ `sp500_vol_iv_full.csv` | ✅ |
| Liquidity | ✅ `sp500_liquidity.csv` | ⚠ derived |
| Fundamentals | ✅ `sp500_fundamentals*.csv` | ❌ (not in v3) |
| Option chains (live) | ❌ | ✅ requires Terminal @ `127.0.0.1:25503` |
| First-order greeks | ❌ | ✅ |
| VIX / SKEW EOD | ✅ | ✅ (EOD only — snapshots blocked) |
| VIX futures (UX1–UX8) | ❌ | ❌ (tier-blocked) |
| Corporate actions | ✅ | ❌ (not in v3) |

**In a sandbox without Terminal access** (e.g. Claude Cowork, CI):
- Run with `SWE_DATA_PROVIDER=bloomberg`. Everything that doesn't need
  a live chain snapshot works.
- Backfilled Theta pulls under `data_processed/theta/**` can be read
  directly when present — they're gitignored and must be rehydrated
  via `scripts/pull_all.py` on a machine with Terminal, then copied in.
  Current Drive snapshot (2026-04-23 manifest): `chains/` 495/503,
  `stocks_eod/` + `iv_history/` 493/503 (full coverage); `iv_surface/`
  **28/503** — mega-caps + sector ETFs only (AAPL, MSFT, SPY, QQQ, the
  XL* sectors, plus 20 others — full list in the directory);
  `index_options_{chains,surfaces}/` cover SPX/SPXW/NDX/RUT/DJX/XSP;
  `vix_family.parquet` ~12y. The manifest's 29 persistent failures are
  the tier ceiling, not missing data.
- **`iv_surface/` coverage is 5.6% of the universe — and currently
  unused by any live decision path.** The SVI tools in
  `engine/volatility_surface.py` (`VolatilitySurfaceBuilder`,
  `create_empirical_surface`, etc.) are exported but have zero
  non-test callers as of 2026-04-25; `get_iv_surface()` on either
  connector returns an empty DataFrame on missing data, not a
  flat-IV stub. Before wiring SVI surfaces into a feature or the
  decision path, decide on the missing-data contract: fail loudly
  on the 475 uncovered tickers, or use a clearly-named fallback
  (`flat_iv_fallback`, not silent). Audit pass: 2026-04-25, no
  live silent-fallback paths found.
- `data_processed/theta_capabilities.json` records the tier map; if
  absent, regenerate with `scripts/probe_theta_capabilities.py`.
- **Drive mounts are eventually-consistent mirrors of the laptop's
  currently-checked-out branch, not a source of truth for the repo.**
  Always resolve via `git fetch origin && git checkout <branch>`, not
  `ls` on the Drive path — a branch the laptop hasn't checked out
  won't appear.
- Drive mounts also deny `unlink` on existing tracked files, so
  `git pull` fetches refs but can't update the worktree
  (`error: unable to unlink old '<file>': Operation not permitted`).
  To read a newer revision without checking it out, use
  `git show origin/<branch>:<path>`. That's the workaround for
  cross-branch reads from Cowork.
- `pyarrow` is listed in `requirements.txt` but is **not**
  pre-installed in a fresh Cowork bash sandbox. Any parquet read
  dies with `Unable to find a usable engine` until you run
  `pip install pyarrow --break-system-packages`. The SessionStart
  hook (step 4) should assert `import pyarrow` alongside the provider
  log.
- `pip install -r requirements.txt` exceeds the 45s bash timeout in a
  fresh Cowork sandbox. Batch the installs: `scipy` alone, then
  `statsmodels arch scikit-learn`, then `yfinance pydantic` — or let
  the SessionStart hook (step 4) handle batching. Same rule applies
  to any single bash call: if it can't finish in 45s, chunk it and
  pass state through workspace files (bash calls don't share state
  across invocations, so `nohup &` doesn't persist).

**Always log which provider was actually selected** when starting a run.
Silent provider selection is a recurring bug source.

---

## 4. What this product does NOT do

Do not propose features in these categories without explicit consent —
they are out of scope by design:

- **Tick-level order flow / microstructure.** Theta v3 does not expose
  realtime stock quotes at this tier. Do not build features that assume
  sub-minute data.
- **Auto-execution.** The engine produces ranked candidates and memos.
  No broker wiring, no OMS, no order routing.
- **Non-US equities or non-S&P 500 names.** Universe is the S&P 500
  constituent list at `data_raw/sp500_constituents_current.csv`.
- **Non-wheel strategies.** Short puts + covered calls + strangles
  (timing-gated) only. Do not generalise to verticals, calendars,
  butterflies without a full EV-path redesign.
- **Overriding EV.** See §2.

---

## 5. File & test conventions

- **Tests** live in `tests/` (52 files, 170+ tests). Run with
  `pytest tests/ -v` or targeted: `pytest tests/test_ev_engine_upgrades.py -v`.
- **Scripts** are in `scripts/`. Pullers are `pull_*.py`; orchestration
  is `pull_all.py`; diagnostics are `diagnose_*.py` and
  `feature_smoke_test.py`.
- **Docs** in `docs/`. `ARCHITECTURE.md`, `DATA_SPECIFICATION.md`,
  `GOVERNANCE.md`, `GREEKS_UNIT_CONTRACT.md`, `MODEL_CARDS.md`.
  Unit conventions for Greeks are canonical — always consult
  `GREEKS_UNIT_CONTRACT.md` before changing Greek code.
- **Feature store**: `data/features/<group>/ticker=<X>/{data.parquet,
  metadata.json, stats.json}`. Groups: `dynamics`, `events`, `labels`,
  `options_features`, `regime`, `technical`, `vol_edge`, `volatility`.
  Only AAPL is committed as a sample; others regenerate via
  `scripts/backfill_features.py`.
- **Do not commit**: `.env`, `data_processed/theta/**`,
  `data_processed/vol_indices*.parquet`, non-AAPL feature shards,
  `__pycache__/`, `dashboard/node_modules/`. See `.gitignore`.

---

## 6. Working defaults for a fresh session

Before making non-trivial changes:

1. Read this file and `LAPTOP_SETUP.md`.
2. Confirm the provider. Default is wired in
   `engine/wheel_runner.py:130` as
   `os.environ.get("SWE_DATA_PROVIDER", "bloomberg").lower()` —
   unset → `bloomberg`. Verify empirically:
   `python -c "from engine.wheel_runner import WheelRunner;
   r=WheelRunner(); print(type(r.connector).__name__)"` should
   print `MarketDataConnector` in a Cowork sandbox.
3. Sanity-check the data layer. **Do not** run
   `scripts/diagnose_candidates.py` with the default full-universe
   (tickers=None) in Cowork — it takes ~3 min and exceeds the 45s
   bash timeout. Instead, pass an explicit 5-ticker list:

   ```python
   from engine.wheel_runner import WheelRunner
   df = WheelRunner().rank_candidates_by_ev(
       tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
       top_n=10, min_ev_dollars=-1e9,
       include_diagnostic_fields=True,
   )
   ```

   Runs in ~2s. If it returns 5 rows with non-null `ev_dollars`,
   `iv`, and `premium`, the Bloomberg CSVs + connector + EV engine
   path is healthy. On a laptop with the Theta Terminal up, run the
   full `scripts/diagnose_candidates.py` — that's the full-universe
   smoke test.
4. For any change touching the decision layer (`ev_engine.py`,
   `wheel_runner.py`, `candidate_dossier.py`): run the full test suite
   (`pytest tests/ -v`), not just the targeted file. The invariants are
   cross-cutting.

## 7. Known integration points (for planned work)

- **TradingView MCP → chart provider**: the integration seam is
  `engine/tradingview_bridge.py`. A live MCP feed should join the
  `ChainedChartProvider`, not replace `EVEngine`. Respect §2.
- **TradingView webhook ingest**: `POST /api/tv/webhook` in
  `engine_api.py`. Alerts land in a ring buffer; `/api/tv/ranked` and
  `/api/tv/dossier` are the read paths.
- **Dealer positioning overlay**: `engine/dealer_positioning.py` →
  `MarketStructure` → consumed by `EVEngine.evaluate` as an optional
  `market_structure` kwarg.

---

## 8. When in doubt

- Don't edit `main` directly. Work on a feature branch.
- Don't add a new data source or advisor without asking how it wires
  into §2.
- Don't generate a `CLAUDE.md` update, a new doc, or a plan document
  unless asked. Work from conversation context.
- If a recent commit looks surprising, read the commit message — the
  last ~20 commits are tier-aware Theta plumbing and are worth
  understanding before touching that area.
