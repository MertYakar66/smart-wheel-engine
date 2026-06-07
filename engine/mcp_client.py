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
server). One :meth:`MCPCLIClient.capture` performs five subprocess
calls, exactly once each:

    tv symbol <SYMBOL>      (chart_set_symbol)
    tv timeframe <TOKEN>    (chart_set_timeframe)
    tv state                (chart_get_state)    -> symbol, timeframe
    tv quote <SYMBOL>       (chart_get_quote)    -> visible_price
    tv screenshot -r chart  (capture_screenshot) -> screenshot_path

Contract (docs/TRADINGVIEW_MCP_INTEGRATION.md §7)
-------------------------------------------------
* **No retries.** One subprocess call per step. A *mandatory* step's
  first failure aborts the capture and raises
  :class:`~engine.tradingview_bridge.MCPClientError` with a value from
  ``MCP_ERROR_MODES``. Retry / fallback is ``ChainedChartProvider``'s
  job, not this client's.
* **One named exception — the best-effort ``tv quote`` step.** Step 4
  (``tv quote``) supplies ``visible_price`` and is the *only*
  deliberate departure from "first failure aborts": a ``tv quote``
  failure is caught, logged, and ``visible_price`` is left ``None`` —
  the capture still returns a successful ``MCPCaptureResult`` with a
  valid screenshot. A chart with no spot price is still useful
  (``MCPChartProvider`` is downgrade-only). The other four steps abort
  on first failure.
* **No quiet substitution.** Apart from the ``tv quote`` exception
  above, the client never returns a half-populated
  :class:`~engine.tradingview_bridge.MCPCaptureResult` or masks a
  failure — every mandatory-step failure is raised.
* **PIT discipline** is the *provider's* responsibility
  (:meth:`MCPChartProvider.fetch` refuses when ``as_of`` is set); this
  client is never handed an ``as_of`` and never consults one.

Live-verification status
------------------------
Verified **2026-05-19** against a live ``tv`` CLI (the
``LewisWJackson/tradingview-mcp-jackson`` fork) driving TradingView
Desktop on Windows via CDP. Confirmed: the five-call sequence; the
``success`` status field; bare-ticker resolution (``AAPL`` →
``tv state`` reports ``BATS:AAPL``); ``tv state`` carries **no price**
(symbol / resolution / studies only) — the live spot is read by the
best-effort ``tv quote <SYMBOL>`` step, verified to return a flat
``{"success": true, ..., "last": <spot>}`` payload, so ``visible_price``
is populated from its ``last`` field; the screenshot path key is
``file_path``; payloads are flat (no envelope).
The ``mcp_unavailable`` error wording in :func:`_classify` is now
live-verified (CDP-down: ``"connection failed"`` / ``"fetch failed"``,
confirmed 2026-06-01 against the installed ``tv`` CLI). The
``symbol_not_found`` / ``browser_disconnected`` wordings still need a
live server with Desktop **up** — they keep the ``TODO(live-verify)``
marker.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
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

# Field-name candidates for parsing ``tv`` JSON. Verified 2026-05-19:
# ``tv state`` returns ``symbol`` + ``resolution`` (no price field),
# ``tv quote`` returns the live spot as ``last`` (flat payload), and
# ``tv screenshot`` returns ``file_path``. The remaining candidates are
# kept as defensive fallbacks against CLI-version drift.
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
    "file_path",
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
# Nested blocks a ``tv quote`` payload might tuck a price inside.
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
        Per-subprocess wall-clock budget. Each of the five steps gets
        its own ``timeout_s``; there is no retry, so the worst-case
        capture latency is ``5 * timeout_s``.
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
        """Drive the five-call capture sequence for one ticker.

        Returns an :class:`~engine.tradingview_bridge.MCPCaptureResult`
        on success; raises
        :class:`~engine.tradingview_bridge.MCPClientError` (a canonical
        taxonomy value) if any *mandatory* step fails.

        The ``tv quote`` step (step 4) is best-effort — the one
        deliberate exception to "first failure aborts". It supplies
        ``visible_price``; a ``tv quote`` failure is caught and logged,
        ``visible_price`` is left ``None``, and the capture still
        returns a successful result. A screenshot with no spot price is
        still useful, since ``MCPChartProvider`` is downgrade-only. The
        other four steps abort on first failure (the §7 no-retry
        contract). This exception is also recorded in the module
        docstring.
        """
        symbol = ticker.upper()
        tf_token = _TF_TO_CLI.get(timeframe, "D")

        # 1 + 2: point the chart at the requested symbol / timeframe.
        # Verified 2026-05-19: the bare ticker resolves — `tv symbol AAPL`
        # is accepted and `tv state` then reports the listing as
        # `BATS:AAPL`. `tv symbol` / `tv timeframe` also return a
        # `chart_ready` bool a future round could poll on.
        self._run(["symbol", symbol], step="chart_set_symbol")
        self._run(["timeframe", tf_token], step="chart_set_timeframe")

        # 3: read the chart state for symbol / timeframe. Verified
        # 2026-05-19: `tv state` carries NO price (symbol / resolution /
        # studies only) — the live spot is read separately by `tv quote`
        # in step 4 below.
        state = self._run(["state"], step="chart_get_state")
        visible_symbol = _extract_str(state, _SYMBOL_KEYS)
        visible_timeframe = _extract_str(state, _TIMEFRAME_KEYS)

        # 4: read the live spot via `tv quote` — BEST-EFFORT, and the
        # ONE deliberate exception to the "first failure aborts" rule
        # the other four steps follow. `tv quote` supplies `visible_price`
        # (the price `tv state` omits); a quote failure must NOT abort
        # the capture — a screenshot with no spot price is still useful
        # (MCPChartProvider is downgrade-only). So a quote failure is
        # caught and logged, `visible_price` is left None, and the
        # capture continues to the screenshot. See the module docstring.
        visible_price: float | None = None
        try:
            quote = self._run(["quote", symbol], step="chart_get_quote")
            visible_price = _extract_price(quote)
        except MCPClientError as exc:
            logger.warning(
                "tv quote failed for %s (%s); visible_price left None, capture continues",
                symbol,
                exc,
            )

        # 5: capture the screenshot.
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
    def _spawn_argv(self, args: list[str]) -> list[str]:
        """Build the full subprocess argv for one ``tv`` call.

        On Windows an ``npm link``-ed ``tv`` is a ``tv.cmd`` shim, and
        :func:`subprocess.run` (``shell=False`` → ``CreateProcess``)
        cannot launch a ``.cmd`` directly — it raises
        ``FileNotFoundError``. So a bare single-token ``cli_command`` is
        routed through ``cmd /c`` on Windows. A multi-token
        ``cli_command`` (e.g. ``["node", ".../cli.js"]``) already names a
        real executable and is passed through unchanged.
        """
        cmd = [*self.cli_command, *args]
        if sys.platform == "win32" and len(self.cli_command) == 1:
            return ["cmd", "/c", *cmd]
        return cmd

    def _run(self, args: list[str], *, step: str) -> Any:
        """Run one ``tv`` subcommand; return its parsed JSON payload.

        Maps every failure path to a canonical
        :class:`~engine.tradingview_bridge.MCPClientError`. No retries —
        a single :func:`subprocess.run` call.
        """
        cmd = self._spawn_argv(args)
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
            # Verified 2026-05-19: the tv CLI reports status via a
            # ``success`` boolean. A ``{"success": false}`` payload is a
            # failure even with no ``error`` string — without the
            # ``success`` check it would slip through silently. ``ok`` is
            # kept as a defensive fallback.
            if (
                payload.get("success") is False
                or payload.get("ok") is False
                or payload.get("error")
            ):
                detail = payload.get("error") or payload.get("message") or payload
                raise _classify(step, str(detail))
            # Verified 2026-05-19: tv payloads are flat — no
            # {"data": {...}} / {"result": {...}} envelope. The descent
            # below is kept as a defensive fallback for CLI-version drift.
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

    The ``mcp_unavailable`` wording is live-verified: against the
    installed ``tv`` CLI with TradingView Desktop down (CDP
    unreachable), ``tv state`` / ``tv symbol`` / ``tv quote`` all exit
    non-zero with ``"CDP connection failed after N attempts: fetch
    failed"`` — so ``"connection failed"`` / ``"fetch failed"`` join the
    ``mcp_unavailable`` set (2026-06-01; S47 / C4). Before that fix this
    string fell through to ``unexpected_error``.

    TODO(live-verify): the ``symbol_not_found`` and
    ``browser_disconnected`` wordings are still written against
    *plausible* strings — confirming them needs a live server with
    Desktop *up*. Anything unmatched falls through to
    ``unexpected_error`` — still a downgrade-only signal, just less
    specific in the audit trail.
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
        for s in (
            "econnrefused",
            "connection refused",
            "connection failed",
            "fetch failed",
            "not running",
            "unreachable",
            ":9222",
            "not recognized",
        )
    ):
        return MCPClientError("mcp_unavailable", f"{step}: {blob[:200]}")
    return MCPClientError("unexpected_error", f"{step}: {blob[:200]}")


def _extract_price(payload: Any) -> float | None:
    """Best-effort live spot price from a ``tv quote`` payload."""
    if not isinstance(payload, dict):
        return None
    for key in _PRICE_KEYS:
        if payload.get(key) is not None:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                continue
    for block in _PRICE_BLOCKS:
        sub = payload.get(block)
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
