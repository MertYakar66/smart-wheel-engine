# TradingView MCP Integration — Design Contract

This is a **design contract**, not an implementation plan. It pins the
constraints and decision boundaries the implementer must respect when
wiring the TradingView MCP into the dossier layer. Implementation
choices (file structure, exact API shapes, retry policy specifics) are
deferred to the implementer via §8 *Open questions*.

The companion contract test
`tests/test_dossier_invariant.py::test_mcp_provider_*` is import-guarded:
dormant until `engine.tradingview_bridge.MCPChartProvider` exists, then
auto-activates and pins the contract structurally.

---

## 1. Goal

Provide live, on-demand chart context to the dossier layer
(`engine/candidate_dossier.py`) by wiring the TradingView MCP as a
`ChartContextProvider`. The current dossier flow depends on
cron-captured screenshots (`FilesystemChartProvider`) or the slow
browser-driven `PlaywrightChartProvider`. MCP is the live preferred
source: faster than Playwright (no headless Chromium spin-up), fresher
than the cron cache.

## 2. Non-goals

The MCP integration explicitly does **not**:

- Replace `EVEngine.evaluate` or any other ranker. The MCP provides
  context, not decisions. See `CLAUDE.md` §2.
- Become an alternative OHLCV / fundamentals / IV backend. The
  `SWE_DATA_PROVIDER` matrix (`bloomberg` / `theta`) is the *data*
  axis; MCP is the *visual* axis.
- Auto-execute trades, place alerts as side effects, or push state
  back to TradingView during the dossier build.
- Run Pine scripts as part of the engine's decision path.

The MCP exposes ~70 tools across chart control, market data, Pine,
strategy testing, watchlists, alerts, and bar replay. Milestone 1 uses
**three** of them (§5). Resist the rest until M2 with deliberation.

## 3. Hard invariants

The integration must respect every rule below. The first three are
locked in by tests; violating them must fail CI.

| Rule | Source | Lock |
|---|---|---|
| EV is the only ranker | `CLAUDE.md` §2; `EVEngine.evaluate` | `tests/test_audit_invariants.py` |
| Any chart provider routes through `EnginePhaseReviewer` | `CLAUDE.md` §7 | `tests/test_dossier_invariant.py` |
| Reviewers can downgrade, never upgrade | `engine/candidate_dossier.py` | `TestEnginePhaseReviewer.test_negative_ev_blocked_even_with_perfect_chart` |
| **No quiet substitution on MCP failure** | this doc, §7 | `tests/test_dossier_invariant.py::test_mcp_provider_errored_context_routes_to_review` (pending) |

The fourth rule is new and specific to this integration: when the MCP
fails for any reason, the provider returns a `ChartContext` with the
`error` field populated. It does **not** return a fabricated
screenshot, a cached stale value, or a "fallback ChartContext" with
optimistic defaults. Failure is visible.

## 4. The seam

A new class `MCPChartProvider(ChartContextProvider)` lives in
`engine/tradingview_bridge.py`, alongside the existing
`FilesystemChartProvider` and `PlaywrightChartProvider`.

**Canonical `ChainedChartProvider` ordering** (also part of the
contract; see §8 question 3 for the latency-driven adjustment that may
override this default):

```
providers = [
    MCPChartProvider,          # 1. live, fast — preferred when up
    FilesystemChartProvider,   # 2. cron cache — instant, possibly stale
    PlaywrightChartProvider,   # 3. headless browser — slow, last-resort
]
```

Rationale: live first (fresh state for the R3 spot cross-check), cached
second (cron populates this nightly; fine for non-critical reviews),
browser last (Chromium spin-up is ~3–5s; useful only when both above
fail).

`ChainedChartProvider` already implements first-success-wins; nothing
in that orchestration changes. The MCP provider participates as one of
N, not as a replacement for the chain.

## 5. Milestone 1 scope

M1 wires three MCP tools — the minimum to attach a live chart context
to a dossier:

| MCP tool | Purpose | `ChartContext` field |
|---|---|---|
| `chart_set_symbol` | Ensure the chart shows the requested ticker | (state setup) |
| `chart_set_timeframe` | Set the requested timeframe | (state setup) |
| `chart_get_state` | Read current spot, symbol, timeframe | `visible_price` |
| `capture_screenshot` | Save the chart as PNG | `screenshot_path` |

That gives the dossier:

- A real-time spot for R3 (currently `FilesystemChartProvider` has a
  stale-screenshot blind spot here).
- A fresh PNG for the human review pane.

`visible_indicators` stays empty in M1. Phase classification (R4) is
still driven by the engine's own `phase` field; the MCP doesn't
provide a chart-derived phase yet.

## 6. Out of scope for M1

Tempting tools that stay deferred:

- `data_get_ohlcv` — collides with the `SWE_DATA_PROVIDER` matrix. A
  third OHLCV source needs its own design discussion.
- `data_get_indicator` / `data_get_study_values` — could populate
  `visible_indicators["phase"]` from a Pine indicator running on the
  chart. Doubles the R4 surface area and adds a Pine-runtime
  dependency. Defer to M2 with explicit calibration of which Pine
  indicator's phase output is canonical.
- `pine_*` (write/run/compile) — engine is the brain. Pine compute
  inside the dossier path violates §3.
- `alert_*` — already half-wired via `/api/tv/webhook` and
  `engine_api.py`'s ring buffer. Different integration entirely.
- `tab_*`, `layout_*`, `pane_*`, `watchlist_*`, `replay_*`, `draw_*` —
  trader-workflow tools, not engine-input tools.

When someone proposes wiring one of these in, the proposal must
explain why M1 scope is being broken.

## 7. Missing-data contract

Every MCP call can fail. When it does, `MCPChartProvider.fetch` returns
`ChartContext(error="<canonical_value>", screenshot_path=None, ...)`.

**Canonical failure-mode taxonomy.** The implementer does not invent
new error strings; if a new mode appears, it gets added to this table
first.

| `error` value | Meaning |
|---|---|
| `mcp_unavailable` | MCP daemon not running / CDP `:9222` unreachable |
| `symbol_not_found` | `chart_set_symbol` returned a "no such symbol" response |
| `screenshot_timeout` | `capture_screenshot` exceeded the per-call budget |
| `browser_disconnected` | Chromium session died mid-call |
| `stale_state` | `chart_get_state` returned data older than tolerance |
| `pit_violation` | `as_of` was set; live MCP must not be consulted (see §8 q4) |
| `unexpected_error` | Catch-all; must be paired with a logged stack trace |

**No retries inside `MCPChartProvider`.** Let `ChainedChartProvider`
fall through to the next backend. Retries are the chain's job, not the
individual provider's.

**Test seam requirement.** `MCPChartProvider` must expose a
deterministic error-injection mechanism for tests — for example a
`with_forced_error(error_value)` classmethod, an injectable MCP
client, or a subclass override. Without it, the contract test in §8
cannot run. Implementer's choice on the exact API.

## 8. Open questions

To be answered by the implementer; answers fold into §9 (decision log)
at implementation time.

1. **MCP client location.** Wrapper directly in
   `engine/tradingview_bridge.py`, or a new `engine/mcp_client.py` the
   bridge imports? Lean toward the latter if the wrapper grows past
   ~150 lines.

2. **Authentication / session lifecycle.** MCP runs locally via CDP on
   `:9222`. Document the connection model — what happens on auth
   failure, who owns the session, what the user must do to bring the
   MCP up. Reference whichever local-dev doc captures the equivalent
   for ThetaTerminal so conventions match.

3. **Latency budget and default ordering.** Measure
   `capture_screenshot` typical RTT. If >2s p95, M1 should ship with
   `MCPChartProvider` as opt-in (env var `SWE_USE_MCP_CHART=1`) rather
   than the default first provider — for batch dossier builds across
   the top 10 candidates, 20s of MCP round-trips beats 5s of cached
   `FilesystemChartProvider`. Latency dictates the §4 default
   ordering.

4. **PIT discipline.** When `as_of` is set (backtests, historical
   reviews), MCP must not be consulted at all — a live screenshot
   served back to a 2024-06 backtest is a look-ahead leak even if the
   screenshot is annotated "current." Proposed default:
   `MCPChartProvider.fetch` returns
   `ChartContext(error="pit_violation", ...)` immediately when
   `as_of is not None`. `ChainedChartProvider` then falls through to
   `FilesystemChartProvider` for cached historical screenshots. Pin
   this in the contract test, not just the doc.

5. **Symbol mapping.** Engine tickers are bare strings (`AAPL`,
   `MSFT`); TradingView uses `EXCHANGE:SYMBOL` form (`NASDAQ:AAPL`).
   `engine/tradingview_bridge.py` already has an `exchange_map` for
   the Playwright provider. Reuse it, or pass through to MCP's own
   symbol resolver?

## 9. Decision log

Empty. Populated as decisions in §8 are made.
