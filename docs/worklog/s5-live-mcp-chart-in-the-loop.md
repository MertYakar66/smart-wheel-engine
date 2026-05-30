---
id: S5
title: Live MCP chart in the loop
kind: usage
status: completed
migrated_from: docs/USAGE_TEST_LEDGER.md
---

**Purpose.** Exercise the live TradingView MCP chart
(`MCPChartProvider` inside a `ChainedChartProvider`) through the real
dossier flow — confirm a live chart reaches verdicts an offline
session cannot, and confirm §2: the chart can only downgrade, never
rescue.

**Setup.** Live infra, 2026-05-21: TradingView Desktop build
`3.1.0.7818` (Electron 38.2.2) with CDP on `localhost:9222`; the
tradingview-mcp `tv` CLI on PATH and connected; `engine/mcp_client.py`
`MCPCLIClient` driving `MCPChartProvider`. `SWE_DATA_PROVIDER=bloomberg`,
`SWE_USE_MCP_CHART=1`. Real candidates from
`WheelRunner.rank_candidates_by_ev`; dossiers via `build_dossiers` +
`EnginePhaseReviewer` (R1–R6).

**Status.** Done. The live MCP chart was exercised end-to-end through
the dossier flow. §2 holds — no violation, no bug, no decision-layer
or `tests/` change. One real integration caveat logged (R3 vs stale
data).

**Findings:**

- **The live chart is genuinely in the loop.** `MCPChartProvider.fetch`
  captured real charts on demand — AAPL `visible_price=304.99`, XOM
  `visible_price=155.29`, each with a screenshot, `source="mcp"`,
  `is_ok=True`. Roughly 12 s per `capture_screenshot` round-trip —
  confirms `TRADINGVIEW_MCP_INTEGRATION.md` §8 q3: MCP is rightly
  opt-in via `SWE_USE_MCP_CHART`, not default-on.
- **Before/after, real candidate (XOM, `ev_dollars=+134.46`).**
  Offline (`FilesystemChartProvider`, no cached screenshot) ->
  `review` / `chart_context_missing` (R2). Live MCP chart ->
  `skip` / `spot_price_mismatch` (R3). The live chart takes the
  candidate off the R2 "no chart" hold and onto a real verdict.
- **§2 holds — verified on real data.** The real AAPL candidate
  (`ev_dollars=-39.05`) stayed `blocked` (R1) with a genuine OK live
  MCP chart attached (`is_ok=True`, real screenshot, live price). R1
  returns at `candidate_dossier.py:158` *before the chart is even
  examined* — the live chart structurally cannot rescue a negative-EV
  candidate or bypass `EVEngine.evaluate`. A controlled negative-EV
  row (`ev_dollars = -50`) + a real live chart -> also `blocked`.
- **`proceed` stays R5's EV decision, not the chart's.** With the real
  live chart attached to a controlled row: `ev_dollars >=
  min_proceed_ev` (10) and `visible_price` within R3's 2 % of the
  engine spot -> `proceed` (R5); a 10 %-off spot -> `skip` (R3). The
  chart's only role is to *not* downgrade — it never upgrades.
- **R3 vs stale data — integration caveat, not a bug.** The live
  chart's `visible_price` is real-time; the engine's `spot` under
  `SWE_DATA_PROVIDER=bloomberg` is a stale EOD-CSV value (XOM 155.29
  live vs 161.22 CSV, ~3.7 %; AAPL 304.99 vs 247.64, ~23 %). Both
  exceed R3's 2 % tolerance, so a live chart *systematically* `skip`s
  real Bloomberg candidates. R3 works as designed — it catches a spot
  disagreement — but the stale side is the engine's data. The live
  MCP chart is `proceed`-useful only paired with a current-spot
  provider (`theta`, or freshly refreshed Bloomberg — cf. S6).
  §2-safe: `skip` is a downgrade. **Logged.**
- **PIT discipline holds.** `MCPChartProvider.fetch(as_of=...)`
  short-circuits to `error="pit_violation"` with no capture; inside a
  `ChainedChartProvider` it falls through to `FilesystemChartProvider`
  — no live screenshot leaks into a historical / backtest review.
- **Fails closed.** A forced MCP error returns a `ChartContext`
  carrying a canonical `MCP_ERROR_MODES` value (`mcp_unavailable`,
  `browser_disconnected`), `is_ok=False` — never a fabricated
  screenshot. In a chain the error falls through to the next provider,
  and the reviewer then sees a missing chart -> R2 `review`.
- **Symbol resolution.** A bare ticker (`AAPL`, `XOM`) resolves
  correctly — TradingView picked `BATS:AAPL`; no `EXCHANGE:TICKER`
  prefix needed. The `mcp_client.py` `TODO(live-verify)`
  ambiguous-ticker fallback did not trip for these large-caps.

**§2 verdict:** holds — no bypass. The live MCP chart is a
downgrade-only `ChartContextProvider`: it cannot rescue a negative-EV
candidate (R1 fires first), cannot manufacture a `proceed` (that is
R5's EV check), and cannot bypass `EVEngine.evaluate`.

- **Re-verified 2026-05-26** by Terminal A — **SKIPPED (operator-gated)**. Per the task spec skip list: S5 requires live TradingView Desktop + CDP on `:9222` + `tradingview-mcp` CLI on PATH. The Cowork sandbox / fresh-checkout terminal has no live TradingView Desktop process. The MCPChartProvider plumbing (the `engine/mcp_client.py` + `engine/tradingview_bridge.py` seam, and the dossier-invariant test guard that auto-activates when MCP ships) is structurally still present on `main` per PROJECT_STATE.md §3 and unchanged since S5 ran. §2 status inherits from the original entry; no live verification this pass.
