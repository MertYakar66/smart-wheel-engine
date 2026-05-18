"""
Synchronous ``tv``-CLI client for the tradingview-mcp server.

Integration **Stage 2** — the concrete
:class:`~engine.tradingview_bridge.MCPChartClient`.

Transport
---------
The engine reaches the tradingview-mcp server by shelling out to its
``tv`` command-line interface, which emits JSON on stdout (transport
"Option A", recorded in ``DECISIONS.md`` D12 and
``docs/TRADINGVIEW_MCP_INTEGRATION.md`` §9). It does **not** speak the
MCP-over-stdio JSON-RPC protocol and does **not** drive Chrome DevTools
Protocol directly — ``ChartContextProvider.fetch`` is synchronous, and
both alternatives are a poor fit (async bridging / re-implementing the
server). One :meth:`MCPCLIClient.capture` performs four subprocess
calls, exactly once each:

    tv symbol <SYMBOL>      (chart_set_symbol)
    tv timeframe <TOKEN>    (chart_set_timeframe)
    tv state                (chart_get_state)    -> visible_price
    tv screenshot -r chart  (capture_screenshot) -> screenshot_path

Contract (docs/TRADINGVIEW_MCP_INTEGRATION.md §7)
-------------------------------------------------
* **No retries.** One subprocess call per step; the first failure
  aborts the capture. Retry / fallback is ``ChainedChartProvider``'s
  job, not this client's.
* **Canonical failures only.** Every failure raises
  :class:`~engine.tradingview_bridge.MCPClientError` with a value from
  ``MCP_ERROR_MODES``; the client never returns a half-populated
  :class:`~engine.tradingview_bridge.MCPCaptureResult`.
* **PIT discipline** is the *provider's* responsibility
  (:meth:`MCPChartProvider.fetch` refuses when ``as_of`` is set); this
  client is never handed an ``as_of`` and never consults one.

Live-verification status
------------------------
This client is written against the **documented** CLI surface of
github.com/tradesdontlie/tradingview-mcp. It has **not** been run
against a live server (this environment has no TradingView Desktop and
no MCP server). The upstream README names the ``tv`` subcommands but
does **not** publish the JSON schema of their output. Every place that
depends on an unconfirmed field name or error string is marked
``TODO(live-verify)`` and coded defensively — multiple candidate keys,
conservative fallback to ``unexpected_error``. Those markers must be
resolved on a machine with the server running before Stage 3 wires this
client onto the live decision path.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from .chart_context import Timeframe
from .tradingview_bridge import MCPCaptureResult, MCPClientError

logger = logging.getLogger(__name__)


# Engine ``Timeframe`` literal -> the token the ``tv timeframe`` command
# expects (TradingView's own interval tokens). Mirrors the interval
# mapping ``tradingview_bridge._DEFAULT_TF_INTERVAL`` uses for chart URLs.
_TF_TO_CLI: dict[str, str] = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1D": "D",
    "1W": "W",
    "1M": "M",
}

# TODO(live-verify): the tradingview-mcp README documents the ``tv``
# subcommands but not the JSON schema of their stdout. The key tuples
# below list plausible field names in priority order; confirm the real
# ones against a running server and prune.
_PRICE_KEYS: tuple[str, ...] = (
    "price",
    "last",
    "last_price",
    "lastPrice",
    "close",
    "spot",
)
_SYMBOL_KEYS: tuple[str, ...] = ("symbol", "ticker", "name")
_TIMEFRAME_KEYS: tuple[str, ...] = ("timeframe", "interval", "resolution", "tf")
_SCREENSHOT_PATH_KEYS: tuple[str, ...] = (
    "path",
    "file",
    "filepath",
    "filePath",
    "screenshot",
    "screenshot_path",
    "screenshotPath",
    "output",
    "image",
)
# Nested blocks ``tv state`` might tuck a price inside.
_PRICE_BLOCKS: tuple[str, ...] = ("quote", "last_bar", "lastBar", "ohlc", "data")


class MCPCLIClient:
    """:class:`~engine.tradingview_bridge.MCPChartClient` backed by the
    tradingview-mcp ``tv`` CLI.

    Parameters
    ----------
    cli_command:
        Argv prefix for the CLI. Defaults to ``["tv"]`` (the command on
        ``PATH`` after ``npm link``). Override for an un-linked clone,
        e.g. ``["node", "/path/to/tradingview-mcp/src/cli/index.js"]``.
    timeout_s:
        Per-subprocess wall-clock budget. Each of the four steps gets
        its own ``timeout_s``; there is no retry, so the worst-case
        capture latency is ``4 * timeout_s``.
    """

    def __init__(
        self,
        *,
        cli_command: list[str] | None = None,
        timeout_s: float = 15.0,
    ) -> None:
        self.cli_command: list[str] = list(cli_command) if cli_command else ["tv"]
        self.timeout_s = float(timeout_s)

    # -- public API ----------------------------------------------------
    def capture(self, ticker: str, timeframe: Timeframe) -> MCPCaptureResult:
        """Drive the four-call capture sequence for one ticker.

        Returns a fully-populated
        :class:`~engine.tradingview_bridge.MCPCaptureResult` on success.
        Raises :class:`~engine.tradingview_bridge.MCPClientError` (a
        canonical taxonomy value) on any failure — never a partial
        result.
        """
        symbol = ticker.upper()
        tf_token = _TF_TO_CLI.get(timeframe, "D")

        # 1 + 2: point the chart at the requested symbol / timeframe.
        # TODO(live-verify): the bare ticker is passed to `tv symbol`
        # (TradingView resolves the listing). If the resolver picks the
        # wrong venue for an ambiguous ticker, switch to EXCHANGE:TICKER.
        self._run(["symbol", symbol], step="chart_set_symbol")
        self._run(["timeframe", tf_token], step="chart_set_timeframe")

        # 3: read the chart state back for the live spot price.
        # TODO(live-verify): if a live screenshot races chart loading,
        # a settle delay or a `tv state` readiness poll belongs here.
        state = self._run(["state"], step="chart_get_state")
        visible_price = _extract_price(state)
        visible_symbol = _extract_str(state, _SYMBOL_KEYS)
        visible_timeframe = _extract_str(state, _TIMEFRAME_KEYS)

        # 4: capture the screenshot.
        shot = self._run(["screenshot", "-r", "chart"], step="capture_screenshot")
        path = _extract_str(shot, _SCREENSHOT_PATH_KEYS)
        if not path:
            # No quiet substitution: a screenshot call that yields no
            # path is a failure, not an OK result with a missing file.
            keys = sorted(shot) if isinstance(shot, dict) else type(shot).__name__
            raise MCPClientError(
                "unexpected_error",
                f"capture_screenshot returned no recognisable path field (got: {keys})",
            )

        return MCPCaptureResult(
            screenshot_path=Path(path),
            visible_price=visible_price,
            visible_symbol=visible_symbol,
            visible_timeframe=visible_timeframe,
        )

    # -- internals -----------------------------------------------------
    def _run(self, args: list[str], *, step: str) -> Any:
        """Run one ``tv`` subcommand; return its parsed JSON payload.

        Maps every failure path to a canonical
        :class:`~engine.tradingview_bridge.MCPClientError`. No retries —
        a single :func:`subprocess.run` call.
        """
        cmd = [*self.cli_command, *args]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise MCPClientError(
                "mcp_unavailable",
                f"tv CLI not found ({self.cli_command[0]!r}); "
                "is the tradingview-mcp server installed and linked?",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            # capture_screenshot has its own canonical timeout value; a
            # hang on any other step means the MCP / CDP is unresponsive.
            mode = "screenshot_timeout" if step == "capture_screenshot" else "mcp_unavailable"
            raise MCPClientError(mode, f"{step} exceeded the {self.timeout_s:g}s budget") from exc
        except OSError as exc:  # spawn failure that is not a missing binary
            raise MCPClientError("unexpected_error", f"{step}: {exc}") from exc

        if proc.returncode != 0:
            blob = _blob(proc.stdout, proc.stderr) or f"exit {proc.returncode}"
            raise _classify(step, blob)

        return self._parse_json(step, proc.stdout)

    def _parse_json(self, step: str, stdout: str) -> Any:
        """Parse a CLI payload; unwrap an envelope; surface error envelopes."""
        text = (stdout or "").strip()
        if not text:
            raise MCPClientError("unexpected_error", f"{step}: empty CLI output")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MCPClientError(
                "unexpected_error",
                f"{step}: CLI output was not JSON: {text[:200]}",
            ) from exc

        if isinstance(payload, dict):
            # TODO(live-verify): confirm whether the tv CLI signals
            # failure with {"ok": false} / an "error" field, or only via
            # the process exit code. Treating either as failure is safe.
            if payload.get("ok") is False or payload.get("error"):
                detail = payload.get("error") or payload.get("message") or payload
                raise _classify(step, str(detail))
            # TODO(live-verify): confirm whether results are wrapped in a
            # {"data": {...}} / {"result": {...}} envelope or returned
            # bare. Descend into an envelope when one is present.
            for key in ("data", "result", "payload"):
                inner = payload.get(key)
                if isinstance(inner, dict):
                    return inner
        return payload


# ---------------------------------------------------------------------
# Module-level helpers (importable for white-box unit tests)
# ---------------------------------------------------------------------
def _blob(stdout: str | None, stderr: str | None) -> str:
    """Join non-empty stdout/stderr into one lowercase-able diagnostic blob."""
    return " ".join(part.strip() for part in (stdout, stderr) if part and part.strip())


def _classify(step: str, blob: str) -> MCPClientError:
    """Map a CLI error blob to a canonical :class:`MCPClientError`.

    TODO(live-verify): the substring checks are written against
    *plausible* error wording — the upstream README does not document
    the CLI's error strings. Confirm against a live server and tighten.
    Anything unmatched falls through to ``unexpected_error`` — still a
    downgrade-only signal, just less specific in the audit trail.
    """
    low = blob.lower()
    if step == "chart_set_symbol" and any(
        s in low for s in ("no such symbol", "symbol not found", "unknown symbol", "invalid symbol")
    ):
        return MCPClientError("symbol_not_found", f"{step}: {blob[:200]}")
    if any(s in low for s in ("disconnect", "target closed", "websocket closed", "browser closed")):
        return MCPClientError("browser_disconnected", f"{step}: {blob[:200]}")
    if any(
        s in low
        for s in ("econnrefused", "connection refused", "not running", "unreachable", ":9222")
    ):
        return MCPClientError("mcp_unavailable", f"{step}: {blob[:200]}")
    return MCPClientError("unexpected_error", f"{step}: {blob[:200]}")


def _extract_price(state: Any) -> float | None:
    """Best-effort live spot price from a ``tv state`` payload."""
    if not isinstance(state, dict):
        return None
    for key in _PRICE_KEYS:
        if state.get(key) is not None:
            try:
                return float(state[key])
            except (TypeError, ValueError):
                continue
    for block in _PRICE_BLOCKS:
        sub = state.get(block)
        if isinstance(sub, dict):
            nested = _extract_price(sub)
            if nested is not None:
                return nested
    return None


def _extract_str(payload: Any, keys: tuple[str, ...]) -> str | None:
    """First non-empty string value among ``keys`` in a JSON object."""
    if not isinstance(payload, dict):
        return None
    for key in keys:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
