---
id: mcp-classify-cdp-down
title: Classify the live CDP-down tv-CLI error as mcp_unavailable (C4 / S47 follow-up)
kind: fix
status: in-flight
terminal: UltraCode
pr:
decisions: []
date: 2026-06-01
headline: engine/mcp_client._classify mapped the real "CDP connection failed after N attempts: fetch failed" string (emitted by the tv CLI when TradingView Desktop / CDP is down) to unexpected_error instead of mcp_unavailable; added "connection failed" / "fetch failed" to the mcp_unavailable substring set, pinned it, and narrowed the TODO(live-verify) marker to the still-Desktop-gated symbol_not_found / browser_disconnected wordings.
surface:
  - engine/mcp_client.py
  - tests/test_mcp_client.py
---

## Goal
<!-- What we set out to do, and why. -->

Close the long-standing `_classify` `TODO(live-verify)` residual on the
MCP chart transport. `MCPCLIClient` shells out to the `tv` CLI; on any
failure it raises `MCPClientError` with a canonical mode from
`MCP_ERROR_MODES`, and `_classify` maps the CLI's stderr blob to that
mode. The per-mode substrings were written against *plausible* wording
(the upstream README doesn't document the CLI's error strings), so the
mapping had never been checked against a live failure.

## What we tried
<!-- Approaches, in the order we tried them. -->

1. Drove the actual installed `tv` CLI with TradingView Desktop **down**
   (CDP unreachable) to capture the real error string, then checked it
   against `_classify`'s substring sets.

## What worked

The CDP-down string is reproducible and specific:
`"CDP connection failed after 5 attempts: fetch failed"` (exit 2, same
string for `tv state` / `tv symbol` / `tv quote`). Adding
`"connection failed"` and `"fetch failed"` to the `mcp_unavailable`
substring set maps it to the correct downgrade-only mode, and a direct
`_classify` unit test pins it.

## What didn't
<!-- The dead ends + WHY. This is the part that saves the next agent. -->

- **`symbol_not_found` / `browser_disconnected` could NOT be
  live-verified this session** — those require TradingView Desktop to be
  *up* (a valid session that then rejects a bad symbol / drops the
  socket), and Desktop is not installed here (`tv_health_check` →
  "CDP connection failed", `tv_launch` → "TradingView not found on
  win32"). Their `TODO(live-verify)` marker is kept, narrowed to just
  those two modes.

## How we fixed it
<!-- The approach that shipped. -->

`engine/mcp_client.py`:
- `_classify`: add `"connection failed"` and `"fetch failed"` to the
  `mcp_unavailable` substring tuple. Before the fix the CDP-down string
  matched none of `econnrefused / connection refused / not running /
  unreachable / :9222 / not recognized` and fell through to
  `unexpected_error` — a less-specific (but still downgrade-only)
  audit-trail signal.
- Module + `_classify` docstrings: record that the `mcp_unavailable`
  wording is now live-verified (2026-06-01), and narrow the remaining
  `TODO(live-verify)` to `symbol_not_found` / `browser_disconnected`.

`tests/test_mcp_client.py`:
- New `test_classify_cdp_connection_failed_is_mcp_unavailable` mirroring
  the existing `test_classify_not_recognized_is_mcp_unavailable`.

No behaviour change beyond the classification label: this is a
downgrade-only transport (`MCPChartProvider` can only cause the dossier
reviewer to downgrade, never rescue a negative-EV trade), so it is off
the §2 trio and §2-neutral. The fix makes a real failure carry the
correct, more-specific downgrade reason.

## Evidence
<!-- Exact commands run, numbers, links to raw artifacts. -->

Reproduction (Python312, before the fix, in-process):

```
_classify("chart_set_symbol", "CDP connection failed after 5 attempts: fetch failed")
   -> error='unexpected_error'   (wrong)
_classify("chart_set_symbol", "connect ECONNREFUSED 127.0.0.1:9222")
   -> error='mcp_unavailable'    (control, correct)
```

After the fix: `pytest tests/test_mcp_client.py -q` -> **43 passed**
(incl. the new CDP-down case). `ruff check` + `ruff format --check`
clean on both files.

## Unresolved / handoff
<!-- What's still open; what the next agent should look at next. -->

- `symbol_not_found` / `browser_disconnected` wordings remain
  unverified — confirm on a machine with TradingView Desktop running and
  remove the residual `TODO(live-verify)` then.
- `"connection failed"` is intentionally broad (covers
  `"cdp connection failed"` and any `"... connection failed ..."`
  variant); in this CLI every call targets the local CDP, so a
  connection failure always means the MCP/CDP is unavailable. Revisit if
  a future `tv` build emits `"connection failed"` for a non-availability
  cause.
