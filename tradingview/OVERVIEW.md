# Operating Overview — Mert's Financial Analyst Function

*Last updated: 2026-04-25*

## Purpose

Stand up a one-person financial analyst function for Mert, powered by Claude with live access to TradingView Desktop. The objective is to compress what would otherwise take a junior analyst (or several) — chart reading, technical screening, Pine Script development, backtesting, written research — into prompt-driven workflows that produce filed, dated deliverables Mert can act on or hand off.

This is **research and decision support**. No autonomous execution, no order routing, no broker integration. Outputs inform Mert's own trading decisions; he is always the executor.

## Operating Model

Mert writes prompts in Cowork (this chat surface). Claude executes against the live TradingView MCP and any other connected tools, then files deliverables into Mert's workspace folder at `~/Desktop/TradingView/`. Mert does not touch TradingView Desktop directly — the chart, watchlist, indicators, alerts, and Pine Script are all driven through Claude.

The TradingView Desktop app must be running with Chrome DevTools Protocol enabled on port 9222. A helper script (`launch-tradingview-cdp.sh`) handles this; if connection drops, that script is the recovery path. The Mac never needs to be touched beyond keeping TradingView open.

## Coverage Mandate

- **Asset classes:** US equities and ETFs, FX, futures and commodities, options
- **Style:** Swing (days to weeks) and position (months to years). No intraday or scalping.
- **Risk posture:** All recommendations must reference `tradingview-mcp-jackson/rules.json` — Mert's watchlist, bias criteria, entry/exit rules, risk parameters, and off-limits list. The rules file is the operating constitution; Claude does not improvise around it.

## Deliverable Standards

| Type | Folder | Format |
|---|---|---|
| Research notes | `research/` | `.docx`, named `YYYY-MM-DD-<title>.docx` |
| Models, backtests, PnL | `models/` | `.xlsx`, named `YYYY-MM-DD-<title>.xlsx` |
| Pine Script source | `pine/` | `.pine`, named `<strategy-name>.pine` |
| Charts / screenshots | `screenshots/` (auto) | TradingView MCP outputs |

Every chart reading cites the symbol, timeframe, and timestamp (e.g., "SPY 1D, last close 713.94 @ 2026-04-25, TradingView"). Numbers are pulled live — no fabrication, no estimation. If the MCP is down, Claude fixes the connection before producing analysis.

## Tooling Stack

**TradingView MCP (Jackson CDP fork)** — ~70 tools spanning chart control, market data (OHLCV, quotes, study values, depth), Pine Script (compile, static analysis, save, run), Strategy Tester read access (results, trades, equity curve), watchlist, native alerts, bar replay with paper trading, multi-pane layouts, and screenshots. Registered globally for both Cowork and Claude Code so either surface works.

**Document skills** — `docx` for research notes, `xlsx` for models and backtests, `pdf` for filings extraction, `pptx` for IC-style decks. These produce institutional-quality deliverables, not raw text dumps.

**Memory** — Persistent across sessions; tracks Mert's preferences, mandate evolution, and lessons from prior work.

**Workspace** — `~/Desktop/TradingView/` is the single source of truth. `CLAUDE.md` boots every session with the analyst mandate. `rules.json` carries the live trading rules. Output folders are pre-created.

## Standard Session Flow

1. Mert sends a prompt (idea, question, deliverable request)
2. Claude runs pre-flight: `tv_health_check` → read `rules.json` → `chart_get_state`
3. Claude executes the analysis using TradingView MCP + relevant skills
4. Deliverable is filed to the appropriate folder with a dated filename, then linked back in chat
5. Claude reports concisely — analyst tone, sourced numbers, no padding

## Known Gaps

TradingView covers price action, technicals, Pine Script, and backtests. It does **not** cover:

- Company fundamentals (revenue, margins, FCF, balance sheet)
- Analyst estimates and consensus
- Earnings transcripts and SEC filings
- Real-time newswires
- Options chain greeks and IV surfaces
- Futures term structure
- Macro releases (CPI, NFP, FOMC)

When a question requires those, Claude states so explicitly rather than improvising. Planned next-round additions to fill these gaps: **FactSet** (fundamentals + estimates), **Aiera** (transcripts + filings + events), **MT Newswires** (real-time news), **LSEG** (cross-asset pricing including FX/futures/options analytics).

## Boundaries

- Research only. Outputs are decision support, never autonomous execution signals.
- Live data only — no mocking, no fabrication. Connection failures are fixed, not worked around.
- Rules-driven — `rules.json` is referenced before any recommendation. Setups outside its parameters are flagged, not produced silently.
- Mert remains the sole executor of any trade.
