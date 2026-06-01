"""
Tests for engine/mcp_client.py — the Stage 2 tv-CLI MCPChartClient.

Every test MOCKS subprocess; none touches a real tradingview-mcp server
or TradingView Desktop. The mock dispatches on the `tv` subcommand and
returns scripted CompletedProcess objects (or raises scripted
exceptions), so the five-call capture sequence, the canonical failure
mapping, the no-retry contract (and its one best-effort exception, the
`tv quote` step), and the defensive JSON parsing are all exercised
deterministically.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from engine.mcp_client import (
    MCPCLIClient,
    _blob,
    _classify,
    _extract_price,
    _extract_str,
)
from engine.tradingview_bridge import MCPCaptureResult, MCPChartProvider, MCPClientError


# ---------------------------------------------------------------------
# subprocess mock
# ---------------------------------------------------------------------
def _completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["tv"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# The tv subcommands MCPCLIClient invokes. The mock locates the
# subcommand anywhere in argv so dispatch is robust to a multi-token
# cli_command prefix (e.g. ["node", "/path/cli.js"]).
_TV_SUBCOMMANDS = ("symbol", "timeframe", "state", "quote", "screenshot")


def _argv_tail(call: list[str]) -> list[str]:
    """Drop the spawn prefix (``tv`` / ``node cli.js`` / ``cmd /c tv``)
    from a recorded argv — return from the first tv-subcommand token
    onward, so call-sequence assertions are platform-independent.
    """
    for i, tok in enumerate(call):
        if tok in _TV_SUBCOMMANDS:
            return call[i:]
    return call


class _FakeRun:
    """Stands in for subprocess.run.

    Dispatches on the tv subcommand — located anywhere in argv, so it is
    robust to a multi-token cli_command prefix — to a scripted response:
    a CompletedProcess to return, or an exception instance to raise.
    Records every call so tests can assert call count and ordering
    (the no-retry contract).
    """

    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = dict(responses)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):  # loose signature: stands in for subprocess.run
        self.calls.append(list(cmd))
        subcommand = next((tok for tok in cmd if tok in _TV_SUBCOMMANDS), "")
        resp = self._responses.get(subcommand)
        if resp is None:
            resp = _completed(0, "{}")
        if isinstance(resp, BaseException):
            raise resp
        return resp


def _happy(screenshot_path: str = "/tmp/shot.png", price: float = 201.5) -> dict[str, object]:
    """A scripted response set where all five steps succeed."""
    return {
        "symbol": _completed(0, '{"ok": true}'),
        "timeframe": _completed(0, '{"ok": true}'),
        "state": _completed(0, json.dumps({"symbol": "AAPL", "timeframe": "D"})),
        "quote": _completed(0, json.dumps({"success": True, "symbol": "AAPL", "last": price})),
        "screenshot": _completed(0, json.dumps({"path": str(screenshot_path)})),
    }


def _patch(monkeypatch, fake: _FakeRun) -> _FakeRun:
    monkeypatch.setattr("engine.mcp_client.subprocess.run", fake)
    return fake


# =====================================================================
# 1. Happy path — the five-call capture sequence
# =====================================================================
class TestMCPCLIClientHappyPath:
    def test_five_calls_in_order(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        result = MCPCLIClient().capture("aapl", "1D")

        assert [_argv_tail(c) for c in fake.calls] == [
            ["symbol", "AAPL"],
            ["timeframe", "D"],
            ["state"],
            ["quote", "AAPL"],
            ["screenshot", "-r", "chart"],
        ]
        assert isinstance(result, MCPCaptureResult)
        assert result.screenshot_path == Path("/tmp/shot.png")
        assert result.visible_price == 201.5
        assert result.visible_symbol == "AAPL"
        assert result.visible_timeframe == "D"

    def test_timeframe_token_mapping(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient().capture("AAPL", "1h")
        assert ["timeframe", "60"] in [_argv_tail(c) for c in fake.calls]

    def test_custom_cli_command_prefix(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient(cli_command=["node", "/x/cli.js"]).capture("AAPL", "1D")
        assert fake.calls[0] == ["node", "/x/cli.js", "symbol", "AAPL"]


# =====================================================================
# 2. Failure mapping — every path → a canonical MCP_ERROR_MODES value
# =====================================================================
class TestMCPCLIClientFailureMapping:
    def test_missing_tv_binary_is_mcp_unavailable(self, monkeypatch):
        _patch(monkeypatch, _FakeRun({"symbol": FileNotFoundError("tv")}))
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_screenshot_timeout_is_screenshot_timeout(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL"}'),
                    "screenshot": subprocess.TimeoutExpired(["tv", "screenshot"], 15.0),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "screenshot_timeout"

    def test_state_timeout_is_mcp_unavailable(self, monkeypatch):
        # A hang on a non-screenshot step means the MCP/CDP is unresponsive.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": subprocess.TimeoutExpired(["tv", "state"], 15.0),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_symbol_not_found(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun({"symbol": _completed(1, "", "Error: no such symbol ZZZZ")}),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("ZZZZ", "1D")
        assert ei.value.error == "symbol_not_found"

    def test_browser_disconnected(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(1, "", "WebSocket closed unexpectedly"),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "browser_disconnected"

    def test_connection_refused_is_mcp_unavailable(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun({"symbol": _completed(1, "", "connect ECONNREFUSED 127.0.0.1:9222")}),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_generic_nonzero_exit_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(2, "", "kaboom"),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_non_json_output_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, "this is not json"),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_empty_output_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, "   "),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_error_envelope_is_classified(self, monkeypatch):
        # Exit 0 but a {"ok": false} envelope still counts as failure.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"ok": false, "error": "ECONNREFUSED :9222"}'),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "mcp_unavailable"

    def test_screenshot_without_path_is_unexpected_error(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL"}'),
                    "screenshot": _completed(0, '{"ok": true}'),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_os_error_on_spawn_is_unexpected_error(self, monkeypatch):
        _patch(monkeypatch, _FakeRun({"symbol": OSError("Exec format error")}))
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"


# =====================================================================
# 3. No-retry contract
# =====================================================================
class TestMCPCLIClientNoRetries:
    def test_aborts_on_first_failure(self, monkeypatch):
        fake = _patch(monkeypatch, _FakeRun({"symbol": _completed(1, "", "kaboom")}))
        with pytest.raises(MCPClientError):
            MCPCLIClient().capture("AAPL", "1D")
        # symbol failed → timeframe/state/screenshot never attempted.
        assert [_argv_tail(c) for c in fake.calls] == [["symbol", "AAPL"]]

    def test_one_call_per_step_through_screenshot(self, monkeypatch):
        fake = _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL"}'),
                    "screenshot": _completed(1, "", "kaboom"),
                }
            ),
        )
        with pytest.raises(MCPClientError):
            MCPCLIClient().capture("AAPL", "1D")
        # Exactly five calls (quote included), no retry of the failed
        # screenshot step.
        assert len(fake.calls) == 5


# =====================================================================
# 4. Defensive JSON handling
# =====================================================================
class TestMCPCLIClientJSONHandling:
    def test_envelope_unwrapping(self, monkeypatch):
        # `_parse_json` unwraps a {"data": {...}} envelope; verify the
        # unwrapped price reaches visible_price via the quote step.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL"}'),
                    "quote": _completed(
                        0, '{"ok": true, "data": {"symbol": "AAPL", "price": 150.0}}'
                    ),
                    "screenshot": _completed(0, '{"path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price == 150.0

    def test_nested_price_block(self, monkeypatch):
        # `_extract_price` descends into a nested price block (here
        # `ohlc`) on the quote payload.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL"}'),
                    "quote": _completed(0, '{"symbol": "AAPL", "ohlc": {"last": 175.25}}'),
                    "screenshot": _completed(0, '{"path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price == 175.25

    def test_screenshot_alternate_path_key(self, monkeypatch):
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL"}'),
                    "screenshot": _completed(0, '{"filePath": "/tmp/alt.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.screenshot_path == Path("/tmp/alt.png")

    def test_quote_without_price_yields_none(self, monkeypatch):
        # Price is supplemental: a quote payload with no recognised
        # price field yields visible_price None — not a failure, as
        # long as the screenshot still succeeds.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL", "timeframe": "D"}'),
                    "quote": _completed(0, '{"symbol": "AAPL"}'),
                    "screenshot": _completed(0, '{"path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price is None
        assert result.screenshot_path == Path("/tmp/x.png")


# =====================================================================
# 5. Module-level helpers
# =====================================================================
class TestModuleHelpers:
    def test_extract_price_non_dict(self):
        assert _extract_price(None) is None
        assert _extract_price("nope") is None
        assert _extract_price([1, 2]) is None

    def test_extract_price_skips_non_numeric(self):
        assert _extract_price({"price": "N/A", "last": 12.0}) == 12.0

    def test_extract_price_first_key_wins(self):
        assert _extract_price({"price": 1.0, "last": 2.0}) == 1.0

    def test_extract_price_none_when_absent(self):
        assert _extract_price({"symbol": "AAPL"}) is None

    def test_extract_str(self):
        assert _extract_str({"symbol": "AAPL"}, ("symbol", "ticker")) == "AAPL"
        assert _extract_str({"ticker": "MSFT"}, ("symbol", "ticker")) == "MSFT"
        assert _extract_str({"symbol": "   "}, ("symbol",)) is None
        assert _extract_str("not a dict", ("symbol",)) is None

    def test_classify_symbol_not_found_only_for_symbol_step(self):
        assert _classify("chart_set_symbol", "no such symbol").error == "symbol_not_found"
        # The same wording on a different step is NOT a symbol error.
        assert _classify("chart_get_state", "no such symbol").error == "unexpected_error"

    def test_classify_fallthrough_is_unexpected_error(self):
        assert _classify("chart_get_state", "something weird").error == "unexpected_error"

    def test_blob(self):
        assert _blob("a", "b") == "a b"
        assert _blob("", "x") == "x"
        assert _blob(None, None) == ""


# =====================================================================
# 6. Composition with MCPChartProvider (Stage 1 + Stage 2)
# =====================================================================
class TestMCPCLIClientWithProvider:
    def test_provider_with_cli_client_success(self, monkeypatch, tmp_path):
        png = tmp_path / "AAPL_1D.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        _patch(monkeypatch, _FakeRun(_happy(screenshot_path=str(png), price=190.0)))

        provider = MCPChartProvider(client=MCPCLIClient())
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.is_ok() is True
        assert ctx.error == ""
        assert ctx.source == "mcp"
        assert ctx.visible_price == 190.0
        assert ctx.screenshot_path == png

    def test_provider_with_cli_client_failure_routes_to_errored_context(self, monkeypatch):
        # A client failure must surface as an errored ChartContext, not
        # an exception — the provider never raises.
        _patch(
            monkeypatch,
            _FakeRun({"symbol": _completed(1, "", "connect ECONNREFUSED :9222")}),
        )
        provider = MCPChartProvider(client=MCPCLIClient())
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.is_ok() is False
        assert ctx.error == "mcp_unavailable"

    def test_provider_pit_guard_skips_cli_client(self, monkeypatch):
        # PIT short-circuit sits in front of the real client too: with
        # as_of set, subprocess.run must never be invoked.
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        provider = MCPChartProvider(client=MCPCLIClient())
        ctx = provider.fetch("AAPL", "1D", as_of=datetime(2024, 6, 1))
        assert ctx.error == "pit_violation"
        assert fake.calls == []


# =====================================================================
# 7. Live-verify fixes (2026-05-19) — Windows invocation + verified shapes
# =====================================================================
class TestMCPCLIClientLiveVerifyFixes:
    def test_windows_wraps_bare_command_in_cmd_c(self, monkeypatch):
        # #1: a bare ["tv"] on Windows is routed through `cmd /c` —
        # subprocess cannot launch the tv.cmd npm shim directly.
        monkeypatch.setattr("engine.mcp_client.sys.platform", "win32")
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient().capture("AAPL", "1D")
        assert fake.calls[0] == ["cmd", "/c", "tv", "symbol", "AAPL"]
        assert _argv_tail(fake.calls[0]) == ["symbol", "AAPL"]

    def test_windows_multitoken_command_not_wrapped(self, monkeypatch):
        # A multi-token cli_command already names a real executable —
        # it is not wrapped, even on Windows.
        monkeypatch.setattr("engine.mcp_client.sys.platform", "win32")
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient(cli_command=["node", "/x/cli.js"]).capture("AAPL", "1D")
        assert fake.calls[0] == ["node", "/x/cli.js", "symbol", "AAPL"]

    def test_non_windows_bare_command_not_wrapped(self, monkeypatch):
        monkeypatch.setattr("engine.mcp_client.sys.platform", "linux")
        fake = _patch(monkeypatch, _FakeRun(_happy()))
        MCPCLIClient().capture("AAPL", "1D")
        assert fake.calls[0] == ["tv", "symbol", "AAPL"]

    def test_screenshot_file_path_key(self, monkeypatch):
        # #2: the real screenshot path key is `file_path`.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"success": true, "symbol": "X"}'),
                    "screenshot": _completed(0, '{"success": true, "file_path": "/tmp/f.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.screenshot_path == Path("/tmp/f.png")

    def test_success_false_is_a_failure(self, monkeypatch):
        # #4: {"success": false} with no "error" key must still fail —
        # the latent bug fixed this round.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"success": false}'),
                }
            ),
        )
        with pytest.raises(MCPClientError) as ei:
            MCPCLIClient().capture("AAPL", "1D")
        assert ei.value.error == "unexpected_error"

    def test_capture_against_verified_live_shapes(self, monkeypatch):
        # The exact JSON shapes captured from the live tv CLI (2026-05-19).
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(
                        0, '{"success": true, "symbol": "AAPL", "chart_ready": false}'
                    ),
                    "timeframe": _completed(
                        0, '{"success": true, "timeframe": "D", "chart_ready": true}'
                    ),
                    "state": _completed(
                        0,
                        json.dumps(
                            {
                                "success": True,
                                "symbol": "BATS:AAPL",
                                "resolution": "1D",
                                "chartType": 1,
                                "studies": [{"id": "PL6Puu", "name": "Bollinger Bands"}],
                            }
                        ),
                    ),
                    "quote": _completed(
                        0,
                        json.dumps(
                            {
                                "success": True,
                                "symbol": "AAPL",
                                "time": 1779111000,
                                "open": 300.24,
                                "high": 300.66,
                                "low": 294.91,
                                "close": 297.84,
                                "last": 297.84,
                                "volume": 34482959,
                                "description": "Apple Inc.",
                                "exchange": "Cboe One",
                                "type": "stock",
                            }
                        ),
                    ),
                    "screenshot": _completed(
                        0,
                        json.dumps(
                            {
                                "success": True,
                                "method": "cdp",
                                "file_path": "/tmp/tv_chart.png",
                                "region": "chart",
                                "size_bytes": 125200,
                            }
                        ),
                    ),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_symbol == "BATS:AAPL"
        assert result.visible_timeframe == "1D"
        assert result.visible_price == 297.84  # from the live `tv quote` payload
        assert result.screenshot_path == Path("/tmp/tv_chart.png")

    def test_classify_not_recognized_is_mcp_unavailable(self):
        # Windows: a missing `tv` via `cmd /c` exits non-zero with
        # "'tv' is not recognized as an internal or external command".
        err = _classify(
            "chart_set_symbol",
            "'tv' is not recognized as an internal or external command, "
            "operable program or batch file.",
        )
        assert err.error == "mcp_unavailable"

    def test_classify_cdp_connection_failed_is_mcp_unavailable(self):
        # Live-observed 2026-06-01: with TradingView Desktop down, the
        # `tv` CLI exits non-zero with "CDP connection failed after N
        # attempts: fetch failed". That is a connection failure to the
        # CDP/MCP, so it must classify as mcp_unavailable, not the
        # less-specific unexpected_error it fell through to before.
        err = _classify(
            "chart_set_symbol",
            "CDP connection failed after 5 attempts: fetch failed",
        )
        assert err.error == "mcp_unavailable"


# =====================================================================
# 8. tv quote — the best-effort 5th step (live spot price)
# =====================================================================
class TestMCPCLIClientQuoteStep:
    """`tv quote` (step 4) supplies visible_price and is the one
    deliberate exception to "first failure aborts": a quote failure is
    caught, visible_price stays None, and the capture still succeeds.
    """

    def test_quote_success_populates_visible_price(self, monkeypatch):
        # A successful `tv quote` populates visible_price from `last`
        # (the live spot field, verified 2026-05-19).
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL", "timeframe": "D"}'),
                    "quote": _completed(0, '{"success": true, "symbol": "AAPL", "last": 297.84}'),
                    "screenshot": _completed(0, '{"file_path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price == 297.84
        assert result.screenshot_path == Path("/tmp/x.png")

    def test_quote_nonzero_exit_leaves_price_none_and_capture_succeeds(self, monkeypatch):
        # A `tv quote` non-zero exit must NOT abort the capture: the
        # screenshot still lands and a successful result is returned.
        fake = _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL", "timeframe": "D"}'),
                    "quote": _completed(1, "", "Error: quote unavailable"),
                    "screenshot": _completed(0, '{"file_path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert isinstance(result, MCPCaptureResult)
        assert result.visible_price is None
        assert result.screenshot_path == Path("/tmp/x.png")
        # All five steps were still attempted, in order — the quote
        # failure did not short-circuit the screenshot.
        assert [_argv_tail(c) for c in fake.calls] == [
            ["symbol", "AAPL"],
            ["timeframe", "D"],
            ["state"],
            ["quote", "AAPL"],
            ["screenshot", "-r", "chart"],
        ]

    def test_quote_timeout_leaves_price_none_and_capture_succeeds(self, monkeypatch):
        # A `tv quote` timeout is swallowed exactly like a non-zero
        # exit — best-effort, never fatal.
        _patch(
            monkeypatch,
            _FakeRun(
                {
                    "symbol": _completed(0, "{}"),
                    "timeframe": _completed(0, "{}"),
                    "state": _completed(0, '{"symbol": "AAPL", "timeframe": "D"}'),
                    "quote": subprocess.TimeoutExpired(["tv", "quote"], 15.0),
                    "screenshot": _completed(0, '{"file_path": "/tmp/x.png"}'),
                }
            ),
        )
        result = MCPCLIClient().capture("AAPL", "1D")
        assert result.visible_price is None
        assert result.screenshot_path == Path("/tmp/x.png")
