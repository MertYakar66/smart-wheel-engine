# TradingView Integration — Engine bridge

TradingView has one role in this repository: the **engine-side bridge**. A
Pine indicator on a TradingView chart fires a webhook into the engine; the
engine enriches the alert with `EVEngine` and returns a proceed / review /
skip verdict. Chart screenshots can be attached to a candidate dossier as a
sanity check through the chart providers.

> **Removed 2026-09-17 (`DECISIONS.md` D30, Track F):** the second role this
> guide used to describe — the MCP-driven analyst workspace (`tradingview/CLAUDE.md`,
> `tradingview/OVERVIEW.md`, the CDP launchers, the vendored
> `tradingview-mcp-jackson` server) — and the opt-in `MCPChartProvider` /
> `engine/mcp_client.py` / `SWE_USE_MCP_CHART` path. The design contract is
> preserved at `archive/2026-09/TRADINGVIEW_MCP_INTEGRATION.md`. In the same
> ruling, reviewer R2 (chart missing) became a note instead of a stop.

---

## Files

| File | Purpose |
|---|---|
| `engine/tradingview_bridge.py` | `FilesystemChartProvider`, `PlaywrightChartProvider`, `ChainedChartProvider`; `build_default_provider` chains filesystem first, Playwright as an optional last resort. |
| `engine/tv_signals.py` | Engine-side parity re-check of the Pine signal logic. Pin the constants in the Pine file. |
| `engine/chart_context.py` | `ChartContext` dataclass + `ChartContextProvider` Protocol. |
| `tradingview/smart_wheel_signals.pine` | Pine v5 indicator. Mirrors `engine/tv_signals.py`. |
| `tradingview/alert_payload_schema.json` | JSON Schema for the webhook body. |
| `tradingview/README.md` | Hands-on setup (install Pine, point alert at webhook). |

**Decision contract:** the bridge is a **downgrade-only reviewer**
on the EV path (`DECISIONS.md` D1, D5). Pine signal can downgrade
a verdict but can never rescue a negative-EV trade. Pine ↔ engine
parity is enforced by
`tests/test_tv_signals.py::test_pine_parity_constants` — pin the Pine
constants to `engine/tv_signals.py`; mismatches break the suite.

**Chart context in the dossier:** `EnginePhaseReviewer` consults the chart
only for R3 (spot mismatch → skip) and R4 (phase contradiction → skip,
dormant). Since D30 a missing or errored chart is recorded as a note
("chart context unavailable: …"), R3/R4 are skipped, and the ladder
continues to R5–R11. The chart is a sanity check, never a decider.

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

**Screenshots (optional):** drop a screenshot at
`<screenshots_dir>/<TICKER>/<TIMEFRAME>.png` (default `screenshots/`) and the
`FilesystemChartProvider` attaches it to the dossier; the dossier endpoints
accept `screenshots_dir` as a query parameter.

---

## When to touch what

| You want to… | Touch |
|---|---|
| Change the Pine indicator that drives webhooks | `tradingview/smart_wheel_signals.pine` + `engine/tv_signals.py` (parity) |
| Change the webhook enrichment / verdict path | `engine_api.py` `_handle_tv_webhook`, `_enrich_alert` |
| Add a new chart provider (screenshot service, etc.) | `engine/tradingview_bridge.py` — extend `ChainedChartProvider`; any new provider is an explicit ask (`OPERATING_MODEL.md` §9.6) |
| Test the bridge end-to-end | `pytest tests/test_tv_signals.py tests/test_tv_api.py tests/test_tv_dossier.py` |

---

## Cross-references

- `tradingview/README.md` — hands-on setup for Pine indicator + alert
- `DECISIONS.md` D5 (chart is a reviewer, not a decider), D30 (R2 note; MCP removal)
- `MODULE_INDEX.md` — `tradingview/` and engine bridge entries
- `archive/2026-09/TRADINGVIEW_MCP_INTEGRATION.md` — the retired MCP design contract
