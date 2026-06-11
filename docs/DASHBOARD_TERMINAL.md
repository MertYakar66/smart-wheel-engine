# DASHBOARD_TERMINAL.md — Runbook for the **Dashboard** terminal

> **ACTIVATION.** If the operator says **"You are responsible for the Dashboard"**
> (or names this terminal `dashboard`), then **you are the Dashboard terminal**.
> Read this file top to bottom, confirm the live rig is up (§2), and then operate
> per the **Command vocabulary** (§1).
>
> **Everything in this role is READ-ONLY / observational** (CLAUDE.md §2/§3):
> never place or modify an order, never run an IBKR write/order tool or method,
> never commit real account data. You do **not** touch the decision-layer trio
> (`ev_engine` / `wheel_runner` / `candidate_dossier`).

---

## 0. What this terminal is for

This terminal **owns the live IBKR portfolio dashboard** — the read-only
performance viewer at **`/portfolio`** (plus the shared-design **`/cockpit`** and
**`/terminal`** surfaces). Its single job: **keep the dashboard's data fresh and
correct on command**, by pulling the operator's real IBKR account through three
read-only channels and regenerating the viewer's data files.

**Other terminals:** leave the dashboard, the `/api/portfolio/*` surface, the
IBKR data pipeline, and `data_processed/ibkr/` to this terminal. If you need
portfolio data, read the engine's `/api/portfolio/*` endpoints — don't pull IBKR
yourself.

---

## 1. Command vocabulary (operator word → action)

| Operator types | You do | Section |
|---|---|---|
| **update** | **FULL refresh** — snapshot (connector) + ledger (Flex) + equity-curve point + verify | §4 |
| **update prices** | Snapshot only (fast; NAV / positions / marks / margin) | §4.1 |
| **update trades** *or* **update ledger** | Flex ledger rebuild only (Realized / Premium / Win-Rate) | §4.2 |
| **status** | Report what the engine currently *serves* — no pull | §6 |
| **show** *or* **screenshot** | Headless-Chrome capture of `/portfolio` (and on request `/cockpit`, `/terminal`) | §6 |
| **probe** | Re-run the read-only IBKR capability probes (connector + Gateway) | §7 |

After any `update`, **report**: NAV, day's notable movers, the four KPIs, the
margin cushion (flag if excess liquidity is thin), `source=live`, and the
snapshot `as_of`.

---

## 2. The live rig (where everything runs)

| Piece | Where | Notes |
|---|---|---|
| **Engine API** | `http://localhost:8811` | worktree **`swe-main`**, `python engine_api.py`. Env: `SWE_API_PORT=8811`, `SWE_IBKR_DATA_DIR=<primary>\data_processed\ibkr`, `SWE_DATA_PROVIDER=bloomberg`. **Re-reads the data files on every request** — so DATA refreshes need **no restart**. Only a CODE change needs a restart (and only with operator OK). |
| **Dashboard** | `http://localhost:3030` | worktree **`swe-view`**, Next.js dev. Proxies `/api/portfolio/*` → `:8811`. **Hot-reloads** on file edits. |
| **Data dir** (`$DATA`) | `C:\Users\merty\Desktop\smart-wheel-engine\data_processed\ibkr` | **gitignored** = the engine's source of truth. This is what `SWE_IBKR_DATA_DIR` points at. |
| **Python** | `C:\Users\merty\AppData\Local\Programs\Python\Python312\python.exe` | has `ib_insync`, `pandas`. The SessionStart "Python not found" is a Store-stub red herring. |

**`$DATA` contents:**
- `portfolio_snapshot.json` — live NAV / cash / margin / positions / marks / FX (the **snapshot**).
- `wheel_ledger.json` — closed wheel cycles → **Realized P&L / Premium / Win-Rate**.
- `portfolio_history.json` — monthly equity curve + a live daily tail → **Total Return**.
- `flex_credentials.json` — Flex token + query id (**gitignored secret**, §3.3).
- `_backup/` — timestamped backups. **Always back up before overwriting.**

**Health check on activation:**
```bash
netstat -ano | grep -E ":8811|:3030"          # both LISTENING?
curl -s http://localhost:8811/api/portfolio/summary   # netLiq + source:"live"?
```
If a server is down, ask the operator before (re)starting it — and never kill a
non-python process on those ports. Engine restart command is in §8.

---

## 3. The three IBKR data sources (all READ-ONLY)

### 3.1 claude.ai IBKR **connector** (cloud MCP) — primary for the snapshot
Cloud OAuth; **works at agent-time only** (absent in headless/cron runs). No
local app needed. Load tools via ToolSearch `select:mcp__claude_ai_Interactive_Brokers_IBKR__...`.

**Read tools (use freely):**
- `get_account_summary` · `get_account_balances` · `get_account_positions` → the snapshot inputs.
- `get_account_trades(period=TODAY..YEAR_TO_DATE / *_QUARTER)` → executions.
  ⚠️ **Option trades carry only the UNDERLYING symbol — no strike/right/expiry —
  and the feed MISSES expiries.** So connector trades **cannot** rebuild the wheel
  ledger; use **Flex** (§3.3) for Realized/Premium/Win-Rate.
- `get_price_history` (OHLCV, up to 5Y) · `get_price_snapshot` (quote + IV) ·
  `search_contracts` · `get_account_orders` · `get_order_instructions`.

**NEVER call** `create_order_instruction` / `delete_order_instruction` (the only
write tools). Large responses (e.g. YTD trades ≈ 350 KB) are saved to a file —
process them with Python, don't paste.

### 3.2 IBKR **Pro API** (local **IB Gateway**, `ib_insync`, port **4001**) — market data
- The Gateway **auto-logs-out daily**; the operator must re-login (credentials +
  2FA). When logged out, `:4001` is **not** listening.
- `ib_insync` **high-level `connect()` HANGS** on the account-data sync
  (`reqAccountUpdates` / `reqPnl` time out on this Gateway) → account pulls
  (positions / portfolio / accountSummary) hang. **For account data, prefer the
  connector (§3.1) or the Flex statement.**
- **Market-data WORKAROUND — low-level connect bypasses the hang:**
  ```python
  from ib_insync import IB, Stock
  ib = IB(); ib.client.connect("127.0.0.1", 4001, clientId=39); ib.client.startApi(); ib.sleep(3)
  # then these all work read-only:
  ib.reqContractDetails(c) · ib.reqHistoricalData(...) · ib.reqSecDefOptParams(...)  # full option chains
  ib.reqMktData(c,"",True,False) · ib.reqMatchingSymbols(q) · ib.reqNewsProviders() · ib.reqScannerParameters()
  ```
- **NEVER** `placeOrder` / `cancelOrder`.

### 3.3 IBKR **Flex Web Service** — the ONLY accurate ledger source
The wheel's income is mostly realized **at option expiry, which generates no
trade** — so the connector + Pro-API *executions* both miss it. The **Flex
Activity statement** includes expiries (`notes="Ep"`) and assignments
(`notes="A"`) with `fifoPnlRealized`. **The trade ledger MUST come from Flex.**

- **Credentials:** `$DATA\flex_credentials.json` (gitignored). Fields:
  - `token` — rotates, **IP-locked**, current window 2026-06-09 → 2026-07-07.
  - `query_id` = **`1537765`** ("SWE Wheel Ledger" — Activity Flex Query,
    **Trades** section, **Last 365 days**, **XML**).
  - `account` = `U17853958`.
  - **The token is a SECRET** — never print it in logs/committed files, never echo it.
- **Endpoints (v3):**
  1. **SendRequest** → reference code:
     `https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest?t={token}&q={query_id}&v=3`
     → returns `<ReferenceCode>` + a `<Url>` (host `gdcdyn`).
  2. **GetStatement** → the XML (retry if "statement generation in progress"):
     `{Url}?t={token}&q={ReferenceCode}&v=3`
- **Rebuild:** convert the XML `<Trade>` attributes → the CSV columns
  `scripts/ibkr_flex_ledger.py` expects, then run it with `--as-of` = today. It
  reconstructs cycles (FIFO + assignment/expiry detection), writes
  `wheel_ledger.json`, and refreshes the `premium` series in
  `portfolio_history.json`. **It also needs the ACAT PDF extract** at
  `C:\Users\merty\AppData\Local\Temp\ibkr_inception_extract.txt` (stock
  cost-basis seed). If the query/token errors:
  - `1014 Query is invalid` → token OK, query id wrong.
  - `1012 Token invalid/expired` or IP error → operator regenerates the token.

---

## 4. The **update** pipeline

> **Preferred (one command):** save the three connector JSONs, then run the
> helper. **Manual fallback** is spelled out under each sub-step.
>
> ```
> 1) Call the 3 read-only connector tools, save raw JSON to %TEMP%\swe_dash\:
>      get_account_summary   -> summary.json
>      get_account_balances  -> balances.json
>      get_account_positions -> positions.json
> 2) python scripts/dashboard_refresh.py all --inputs %TEMP%\swe_dash
> ```
> The helper does snapshot + ledger + curve + verify and prints the result. It
> reads `$DATA\flex_credentials.json` for Flex, backs up before overwriting, and
> never writes the token anywhere.

### 4.0 Always back up first
```bash
cp $DATA/portfolio_snapshot.json  $DATA/_backup/portfolio_snapshot.<ts>.json
cp $DATA/wheel_ledger.json        $DATA/_backup/wheel_ledger.<ts>.json
cp $DATA/portfolio_history.json   $DATA/_backup/portfolio_history.<ts>.json
```

### 4.1 Snapshot (connector)  — `update prices`
1. Call `get_account_summary` / `get_account_balances` / `get_account_positions`; save raw JSON.
2. Regenerate (note `--no-day-change`):
   ```bash
   python scripts/ibkr_live_snapshot.py \
     --summary summary.json --balances balances.json --positions positions.json \
     --out $DATA/portfolio_snapshot.json --no-day-change
   ```
   `--no-day-change` keeps `day_change_*` **NULL** — gated until reconciled
   against the IBKR app's own Day P&L ("don't ship an unverified headline").
3. **Sync the equity-curve live point** — update/append today's NAV into
   `portfolio_history.json` (label `"<Mon> <day>"`, `port`=NAV; keep `spy`,
   `premium`). This makes Total Return reflect the live NAV. (The helper does
   this automatically.)

### 4.2 Ledger (Flex) — `update trades` / `update ledger`
1. **SendRequest** (creds from `$DATA\flex_credentials.json`) → `ReferenceCode`.
2. **GetStatement** → XML (retry on "in progress").
3. XML → CSV (12 cols: `DateTime, AssetClass, Symbol, Strike, Expiry, Put/Call,
   Quantity, TradePrice, Proceeds, IBCommission, CurrencyPrimary,
   Open/CloseIndicator`; map from `dateTime, assetCategory, symbol, strike,
   expiry, putCall, quantity, tradePrice, proceeds, ibCommission,
   ibCommissionCurrency, openCloseIndicator`). Pass it as CSV-A + a header-only
   CSV-B (the importer dedups B after A's last timestamp).
4. Run with today's `AS_OF`:
   ```bash
   python scripts/ibkr_flex_ledger.py A.csv B.csv --out $DATA --as-of <YYYYMMDD>
   ```
5. **Validate**: realized YTD should land in the ~$24k region; compare to the
   pre-refresh backup; if it diverges wildly, STOP and report (don't ship).

### 4.3 Verify
```bash
curl -s :8811/api/portfolio/summary    # netLiq, realizedYtd, premium30d, winRate, source:"live"
curl -s :8811/api/portfolio/returns    # Total Return YTD reflects live NAV
curl -s :8811/api/portfolio/positions  # legs == 14 (or current count)
```

### 4.4 Clean up the temp account-data files (keep `_backup/`).

---

## 5. What the data feeds (dashboard surfaces)

- **KPI cards** — NetLiq + Unrealized P&L (snapshot, **live**); Total Return
  (history curve, live tail); **Realized / Premium / Win-Rate (Flex ledger)**.
- **Holdings** — flat **14-leg** view (`build_positions_flat`): one row per
  stock / option leg (`shares` / `short_put` / `short_call` / `long_call`),
  grouped by underlying.
- **Risk Radar** — single-name (R10, 10% NAV) + sector (R9, 25% NAV)
  concentration + margin-health gauge (from the snapshot).
- **Provenance** — a slice is "live" unless its file `source` is
  fixture/demo/mock. The `/portfolio` caveat line states which KPIs are live vs
  Flex-import-sourced.

Engine functions (in `engine/ibkr_portfolio_adapter.py`, read-only, outside the
trio): `account_summary` · `build_holdings_view` (wheel-aggregated) ·
`build_positions_flat` (flat legs) · `returns_view` (Total Return, keys off
`points[-1].port`) · `equity_view` · `income_view` (Realized=YTD `net_pnl`,
Premium=30-day `put_premium+call_premium`, Win-Rate via `WheelTracker`) ·
`risk_view`. The HTTP handler is `engine_api.py::_handle_portfolio_view`.

---

## 6. status / show
- **status** — `curl :8811/api/portfolio/{summary,income,returns,positions}` and
  report NAV, the four KPIs, margin cushion, `source`, `asOf`. No pull.
- **show** — headless capture (Chrome at `C:\Program Files\Google\Chrome\Application\chrome.exe`):
  ```bash
  chrome --headless=new --disable-gpu --hide-scrollbars --no-sandbox \
    --user-data-dir=<tmp> --window-size=1440,1320 --virtual-time-budget=10000 \
    --screenshot=<out.png> http://localhost:3030/portfolio
  ```
  Then Read the PNG. (Full-page = tall window; narrow window = single-column.)

## 7. probe
Re-run the read-only capability checks: the connector tools (§3.1) and the
low-level Gateway probe (§3.2). Report what's reachable (the Gateway is often
logged out; the connector is the reliable path).

---

## 8. Guardrails — DO / DON'T

- **READ-ONLY**, always. No order tools (`create_order_instruction`,
  `delete_order_instruction`) / methods (`placeOrder`, `cancelOrder`).
- **Never commit account data.** `data_processed/ibkr/` is gitignored; keep it so.
- **`day_change_*` stays NULL** until reconciled vs the IBKR app.
- **Don't restart servers** without operator OK. DATA refreshes are re-read
  live (no restart). Only a CODE change to the engine needs a restart:
  ```powershell
  # stop the python on :8811 (verify it's python first), then:
  $env:SWE_API_PORT='8811'; $env:SWE_IBKR_DATA_DIR='C:\Users\merty\Desktop\smart-wheel-engine\data_processed\ibkr'; $env:SWE_DATA_PROVIDER='bloomberg'
  Start-Process 'C:\Users\merty\AppData\Local\Programs\Python\Python312\python.exe' -ArgumentList 'engine_api.py' -WorkingDirectory 'C:\Users\merty\Desktop\swe-main' -WindowStyle Hidden
  ```
- **Flex token is secret** — never commit/print/echo it.
- **Branch + PR for code changes; never commit to `main`.** Data changes aren't
  commits (gitignored).

## 9. Current state (as of 2026-06-10 — update this line when it drifts)
- The flat-leg holdings view + design-unify + provenance-caveat edits **landed
  on `main`** via PR #391 (`aec90be`). The old swe-main/swe-view rig (:8811 /
  :3030) is retired; the live rig now runs from the **`swe-dash` worktree**
  (engine `:8787` + Next dev `:3000`), with `SWE_IBKR_DATA_DIR` pointed at the
  primary clone's `data_processed/ibkr` (the canonical artifact store).
- Branch `claude/dashboard-pro-overhaul` (2026-06-10, this worktree) carries
  the professional overhaul: **date-anchored period returns** (YTD was
  misstated ~10pp by in-window anchoring; pct/usd now share one anchor),
  income/margin/engine-gates surfaces on `/portfolio`, frontier-derived
  cockpit `as_of` + authoritative dossier verdicts, a de-fabricated terminal
  (no placeholder quotes/Greeks/agent panel), and news ingestion actually
  scheduled via `instrumentation.ts`.
- Account NAV ~$152k (recovering from the April drawdown); margin cushion is
  STILL thin (excess liquidity ~$9k at 1.76× leverage) — always surface margin
  health after an update.

## 10. Gotchas (hard-won)
- **Connector trades miss option expiries + lack contract detail** → Flex only
  for the ledger (§3.3).
- **Gateway**: daily logout; high-level `connect` hangs → low-level connect for
  market data (§3.2).
- **Windows/Bash**: backslash paths get mangled in the Bash tool — use
  PowerShell for path-y git/worktree ops; the harness resets tool cwd to the
  primary clone between calls (pass absolute paths).
- **Connector big responses** save to a file → process with Python.
- **MU-style strangle label**: the flat-leg view shows each leg's own name
  correctly; the *aggregated* `build_holdings_view` labels a row by its first
  leg (cosmetic only — unused by the flat view).
- **Re-run safety**: every regenerate backs up to `_backup/` first; the Flex
  rebuild validates realized-YTD against the prior value before it's trusted.
