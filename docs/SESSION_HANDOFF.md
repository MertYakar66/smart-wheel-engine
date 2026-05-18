# Session Handoff

**Written:** 2026-05-18, by an autonomous work batch against `main` @
`e2b5115`.

A snapshot of in-flight work and open state for the next session. This
is the narrow companion to `PROJECT_STATE.md` (current structural /
temporal state) and `ROADMAP.md` (scoped-but-not-done): it records what
*this* batch left in motion.

---

## 1. Branches pushed, not merged

This batch pushed feature branches only — **no PRs were opened** (the
operator opens PRs explicitly). Each branch is cut from `main` @
`e2b5115`; none has been merged, so `main` is unchanged.

| Branch | Contents | Commits |
|---|---|---|
| `claude/mcp-chart-provider` | TradingView MCP integration, Stage 1 + Stage 2 | `1e7465f`, `c4ac013` |
| `claude/fix-f3-theta-indices-test` | F3 — Windows-only theta-indices test fix | `1634c45` |
| `claude/doc-drift-f6-f11` | Doc-drift cleanup + this handoff doc | (this branch) |
| `claude/coverage-wheel-runner-data-connector` | Coverage backfill, test-only | see batch report |
| `fix/theta-connector-ohlcv-date-column` | Pre-existing PR #83 branch; lint investigation | see batch report |

## 2. TradingView MCP integration — where it stands

- **Stage 1** (offline contract skeleton — `MCPChartProvider`) and
  **Stage 2** (`MCPCLIClient`, the `tv`-CLI subprocess client) are
  implemented on `claude/mcp-chart-provider`. The two import-guarded
  contract tests in `tests/test_dossier_invariant.py::test_mcp_provider_*`
  are now active and pass.
- **Transport decision:** Option A — the `tv` CLI. Recorded in
  `DECISIONS.md` D12 and `docs/TRADINGVIEW_MCP_INTEGRATION.md` §9.
- **Stage 3 is NOT done and is gated.** Wiring `MCPCLIClient` as a
  provider default (behind `SWE_USE_MCP_CHART`) needs two human
  decisions still open — `docs/TRADINGVIEW_MCP_INTEGRATION.md` §8 q2
  (which machine hosts TradingView Desktop; session lifecycle) and q3
  (latency budget / default chain ordering). Do **not** wire MCP onto
  the live decision path until q2/q3 are answered.
- **Live verification pending:** `MCPCLIClient` is written against the
  *documented* tradingview-mcp CLI and has not run against a live
  server. Grep `engine/mcp_client.py` for `TODO(live-verify)` — each
  marker is a JSON field name or error string to confirm on a machine
  with TradingView Desktop + the server.
- **§2 invariant intact:** `MCPChartProvider` is a
  `ChartContextProvider`; it can only ever downgrade a verdict, never
  rescue a negative-EV trade.
- When `claude/mcp-chart-provider` merges, two docs go stale and must
  be updated at merge time: `MODULE_INDEX.md` ("Future home of
  `MCPChartProvider`") and `PROJECT_STATE.md` §3 ("nothing has shipped
  on the MCP class itself").

## 3. Found bug — logged, not fixed

F3 was a Windows-only test failure: `scripts/pull_theta_indices_history.py`
`main()` reassigned `sys.stdout` to a fresh `TextIOWrapper`, which
clobbered pytest's `capsys` on Windows. Fixed on
`claude/fix-f3-theta-indices-test` by switching to
`stream.reconfigure()`.

**The same `sys.stdout = io.TextIOWrapper(...)` reassignment
anti-pattern exists in 13 other scripts:** `backfill_features.py`,
`pull_fundamentals_yf.py`, `pull_vol_indices.py`, `pull_earnings_yf.py`,
`pull_treasury_yields_yf.py`, `pull_all.py`, `pull_theta_vix_futures.py`,
`probe_theta_capabilities.py`, `pull_theta_option_tape.py`,
`pull_theta_options_flow.py`, `pull_theta_iv_surface_history.py`,
`pull_theta_corp_actions.py`, `pull_news_sentiment.py`. These are
**latent** — no test currently invokes their `main()` under `capsys`,
so the full suite is green — but any future such test fails on Windows.
Recommended follow-up: a dedicated branch applying the same
`reconfigure()` fix across all 13.

## 4. Open backlog

- **PR #83** (`fix/theta-connector-ohlcv-date-column`) — open, "Lint &
  Type Check" job red. See the batch report for the diagnosis.
- **Coverage backfill** — `engine/wheel_runner.py` and
  `engine/data_connector.py` are under-covered; test-only branch in
  this batch.
- **F-findings backlog** — F1/F2/F4/F5 merged earlier (PRs #84-86); F3
  fixed this batch; the F6-F11 doc-drift pass this batch corrected the
  *verifiable* drift (stale line numbers, the 29→32 endpoint count, the
  71→72 test-file count, the `src/backtest/` omission in
  `PROJECT_STATE.md` §4, the D1–D10→D1–D11 count). F6-F9's original
  chat-report specifics were never committed; remaining drift should be
  handled case-by-case against live code.
- **ROADMAP** — `ROADMAP.md` Track A (A1 MCP — now Stage 1/2 done; A2
  iv_surface missing-data contract still `open question`), Track B (doc
  repair B1-B6), Track C, Track F (lint debt).
- **CLAUDE.md** is user-maintained — a proposed drift diff (`evaluate`
  line 234→237, `diagnose_candidates.py:60`→`:102`, R6 wording) is in
  the batch report, deliberately **not** committed.

## 5. Environment notes

- Dev box: Windows 11. Python:
  `C:\Users\merty\AppData\Local\Programs\Python\Python312\python.exe`
  (3.12.10). The SessionStart "Python was not found" line is a
  Microsoft-Store stub red herring — use the path above.
- No TradingView Desktop, no tradingview-mcp server, and no Theta
  Terminal on this box: anything needing a live Theta chain or a live
  MCP server cannot be verified here (run `SWE_DATA_PROVIDER=bloomberg`).
- Full test suite on `main`: 1734 tests, ~105s, green.
