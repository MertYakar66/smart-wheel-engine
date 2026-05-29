# TradingView Integration — Parent Guide

The repo uses TradingView in **two distinct roles**. They share the
`tradingview/` folder for ergonomics but have different contracts.
Anyone wiring up new TradingView code or analyst workflows should
read this file first to know which surface they're touching.

If you came looking for a deeper design contract for the MCP-driven
chart provider, jump straight to
[`TRADINGVIEW_MCP_INTEGRATION.md`](TRADINGVIEW_MCP_INTEGRATION.md).

---

## Role 1 — Engine-side bridge (production)

**Purpose:** Pine indicator on a TradingView chart fires a webhook
into the engine; engine enriches the alert with `EVEngine` and
returns a proceed/review/skip verdict.

**Files:**

| File | Purpose |
|---|---|
| `engine/tradingview_bridge.py` | `FilesystemChartProvider`, `PlaywrightChartProvider`, `ChainedChartProvider`. Future home of `MCPChartProvider` (see roadmap). |
| `engine/tv_signals.py` | Engine-side parity re-check of the Pine signal logic. Pin the constants in the Pine file. |
| `engine/chart_context.py` | `ChartContext` dataclass + `ChartContextProvider` Protocol. |
| `tradingview/smart_wheel_signals.pine` | Pine v5 indicator. Mirrors `engine/tv_signals.py`. |
| `tradingview/alert_payload_schema.json` | JSON Schema for the webhook body. |
| `tradingview/README.md` | Hands-on setup (install Pine, point alert at webhook). |

**Decision contract:** the bridge is a **downgrade-only reviewer**
on the EV path (`DECISIONS.md` D1, `D5`). Pine signal can downgrade
a verdict but can never rescue a negative-EV trade. Pine ↔ engine
parity is enforced by
`tests/test_tv_signals.py::test_pine_parity_constants`.

**Webhook flow:**

```
TradingView alert fires
        │
        ▼
POST /api/tv/webhook  (JSON payload matches alert_payload_schema.json)
        │
        ▼
engine_api.EngineAPIHandler._handle_tv_webhook
        │
        ├─► TVAlert.parse           (validate schema)
        ├─► compute_tv_signal       (parity re-check)
        ├─► WheelRunner.analyze_ticker  (wheel score, events, IV rank)
        ├─► EVEngine.evaluate       (the only ranker)
        │
        ▼
verdict ∈ {proceed, review, skip}
```

**Polling-only mode** (no tunnel, no webhook): hit
`GET /api/tv/signal?ticker=<T>` or `GET /api/tv/scan?limit=25`. Same
`TVSignal` struct. A 15-min cron is enough for daily workflow.

**Optional shared secret:** set `TV_WEBHOOK_SECRET` and add
`"secret":"<value>"` to the Pine alert message. The engine rejects
mismatched secrets with HTTP 401.

---

## Role 2 — Analyst workspace (research-only)

**Purpose:** Mert's one-person financial analyst function. Claude
drives a TradingView Desktop instance via Chrome DevTools Protocol
(CDP) on port 9222 using the `tradingview-mcp-jackson` MCP server.
Outputs are filed deliverables: research notes, models, Pine source.
**No autonomous execution. Mert is always the executor.**

**Files:**

| File | Purpose |
|---|---|
| `tradingview/CLAUDE.md` | Workspace contract for Claude when acting as analyst. Pre-flight checklist, conventions, gaps. |
| `tradingview/OVERVIEW.md` | The operating-overview narrative (what the function is, mandate, deliverable standards). |
| `tradingview/launch-tradingview-cdp.sh` | Launches TradingView Desktop with `--remote-debugging-port=9222`. Recovery path if CDP drops. |
| `tradingview/research/` | `.docx` research notes, `YYYY-MM-DD-<title>.docx`. Contents gitignored; `.gitkeep` preserves the dir. |
| `tradingview/models/` | `.xlsx` models, backtests, PnL sheets. Contents gitignored; `.gitkeep` preserves the dir. |
| `tradingview/pine/` | `<strategy-name>.pine` Pine sources from analyst work. `.gitkeep` preserves the dir. |
| `tradingview/tradingview-mcp-jackson/` | **Vendored MCP server (separate git repo + node_modules).** Gitignored. Clone-it-yourself. |

**The MCP server** exposes ~70 tools across chart control, market
data, Pine Script, strategy tester, watchlist, alerts, and bar
replay. Full tool catalog and decision tree live at
`tradingview/tradingview-mcp-jackson/CLAUDE.md` (vendored, not in this
repo). Output-size guidance: prefer `summary: true` on
`data_get_ohlcv`, `study_filter` on Pine tools, and screenshots over
large data dumps.

**Pre-flight checklist** (run silently at session start, per
`tradingview/CLAUDE.md`):

1. `tv_health_check` — confirm MCP ↔ TradingView connection
2. Read `tradingview-mcp-jackson/rules.json` for current watchlist
3. `chart_get_state` — confirm symbol/timeframe before any analysis

**Conventions:**

- Save research notes to `research/` as `YYYY-MM-DD-<title>.docx`
- Save models to `models/` as `YYYY-MM-DD-<title>.xlsx`
- Save Pine sources to `pine/` as `<strategy-name>.pine`
- Cite source + timestamp on every chart reading:
  *"SPY 1D, last close 502.13 @ 2026-04-25 close (TradingView)"*
- Never mock or fabricate data — fix the MCP connection if it's down

**Coverage gap (acknowledged):** TradingView covers price action,
technicals, Pine, and backtests. It does NOT cover fundamentals,
analyst estimates, earnings transcripts, SEC filings, real-time
newswires, options chain Greeks/IV surfaces, futures term structure,
or macro releases. Planned next-round additions to fill these gaps:
**FactSet** (fundamentals + estimates), **Aiera** (transcripts +
filings + events), **MT Newswires** (real-time news), **LSEG**
(cross-asset pricing including FX/futures/options analytics).

---

## Setup (laptop bring-up)

### One-time

1. **Install TradingView Desktop.**
   - macOS: download from tradingview.com, install to `/Applications`.
   - Windows: install from the Microsoft Store (the official Windows
     distribution). The resulting MSIX install lives at
     `C:\Program Files\WindowsApps\TradingView.Desktop_*_x64__*\TradingView.exe`,
     **not** `%LOCALAPPDATA%\TradingView\`. The PS1 launcher below uses
     `Get-AppxPackage -Name TradingView.Desktop` to locate it, since the
     WindowsApps ACL blocks `Get-ChildItem` listing and `where TradingView.exe`
     returns nothing for Store apps.
2. **Clone the MCP server** (gitignored, manual install):
   ```bash
   cd tradingview/
   git clone https://github.com/LewisWJackson/tradingview-mcp-jackson.git
   cd tradingview-mcp-jackson && npm install
   ```
3. **Wire the MCP server into Claude Code** (user scope, persists across
   sessions on this machine):
   ```bash
   # macOS / Linux
   claude mcp add --scope user tradingview -- node ~/path/to/tradingview-mcp-jackson/src/server.js

   # Windows (PowerShell)
   claude mcp add --scope user tradingview -- node C:\Users\<you>\Desktop\smart-wheel-engine\tradingview\tradingview-mcp-jackson\src\server.js
   ```
   Verify with `claude mcp list` — the `tradingview` row should show
   `✓ Connected`. **Restart Claude Code** after adding; MCP tools only
   load at startup.
4. **Pin the Pine indicator constants** to match
   `engine/tv_signals.py`. The `test_pine_parity_constants` test
   enforces it; mismatches break the suite.

### Every session

1. If TradingView isn't already up with CDP:
   - macOS: `bash tradingview/launch-tradingview-cdp.sh`
   - Windows: `powershell -ExecutionPolicy Bypass -File tradingview\launch-tradingview-cdp.ps1`

   **Never relaunch TradingView from the dock, taskbar, or Start menu** —
   those launch paths drop the `--remote-debugging-port=9222` flag and
   CDP attach silently fails.
2. Verify `http://localhost:9222/json` returns chart info.
3. In Cowork sessions, the `tv_health_check` MCP tool is the
   one-liner to confirm.

### Windows gotchas

- `tv_launch` (the MCP's auto-detect tool) does NOT find Microsoft Store
  installs — it searches `%LOCALAPPDATA%\TradingView\` and
  `%PROGRAMFILES%\TradingView\` only, and `where TradingView.exe` returns
  nothing for MSIX. Use the PS1 launcher instead.
- The WindowsApps folder is ACL-locked: `BUILTIN\Users` has
  `ReadAndExecute` (so direct spawn works) but cannot list the directory.
  The launcher relies on `Get-AppxPackage` for path resolution.
- After a reboot or any Start-menu relaunch of TradingView, CDP is gone —
  re-run the PS1 launcher.

---

## When to touch which role

| You want to… | Touch |
|---|---|
| Change the Pine indicator that drives webhooks | `tradingview/smart_wheel_signals.pine` + `engine/tv_signals.py` (parity) |
| Change the webhook enrichment / verdict path | `engine_api.py` `_handle_tv_webhook`, `_enrich_alert` |
| Add a new chart provider (MCP, screenshot service, etc.) | `engine/tradingview_bridge.py` — extend `ChainedChartProvider` |
| Add a new analyst workflow / output convention | `tradingview/CLAUDE.md` + `tradingview/OVERVIEW.md` |
| Update the MCP tool catalogue or decision tree | the vendored repo at `tradingview/tradingview-mcp-jackson/CLAUDE.md` (not here) |
| Test the bridge end-to-end | `pytest tests/test_tv_signals.py tests/test_tv_api.py tests/test_tv_dossier.py` |

---

## Open work

See `ROADMAP.md` Track A:

- **A1** `MCPChartProvider` implementation (design contract is
  locked in `TRADINGVIEW_MCP_INTEGRATION.md`).

---

## Cross-references

- `TRADINGVIEW_MCP_INTEGRATION.md` — MCP chart provider design
  contract (the M1 scope: 5 MCP tools, no quiet substitution on
  failure)
- `tradingview/README.md` — hands-on setup for Pine indicator + alert
- `tradingview/CLAUDE.md` — analyst workspace contract
- `tradingview/OVERVIEW.md` — analyst function operating overview
- `DECISIONS.md` D5 — why both surfaces share a folder
- `MODULE_INDEX.md` — `tradingview/` and engine bridge entries
