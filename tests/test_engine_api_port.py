"""Unit tests for ``engine_api._resolve_port`` — closes D15 Unresolved.

The launch-readiness contract (`docs/LAUNCH_READINESS.md` §8) and the
per-terminal env loader (`scripts/setup-terminal.sh`) treat
``SWE_API_PORT`` as an authoritative override. This file pins:

- the default (8787) when the env is unset or empty;
- the override path (any in-range integer);
- the loud-failure path (malformed or out-of-range raises ``ValueError``
  rather than silently falling back — D11's general principle).

Tested through a pure function ``_resolve_port(env=…)`` rather than
through process boot so the suite stays fast and deterministic.
"""

from __future__ import annotations

import pytest

from engine_api import _DEFAULT_API_PORT, _resolve_port


class TestResolvePortDefault:
    def test_unset_env_returns_default(self):
        assert _resolve_port({}) == _DEFAULT_API_PORT

    def test_empty_string_returns_default(self):
        assert _resolve_port({"SWE_API_PORT": ""}) == _DEFAULT_API_PORT

    def test_whitespace_only_returns_default(self):
        assert _resolve_port({"SWE_API_PORT": "   "}) == _DEFAULT_API_PORT

    def test_default_constant_is_8787(self):
        """Pin the historical port value — single-instance setups
        without the env var must keep binding 8787."""
        assert _DEFAULT_API_PORT == 8787


class TestResolvePortOverride:
    def test_terminal_b_port(self):
        """setup-terminal.sh b yields SWE_API_PORT=8788."""
        assert _resolve_port({"SWE_API_PORT": "8788"}) == 8788

    def test_terminal_c_port(self):
        """setup-terminal.sh c yields SWE_API_PORT=8789."""
        assert _resolve_port({"SWE_API_PORT": "8789"}) == 8789

    def test_explicit_default_port(self):
        """Setting the env to the default still works."""
        assert _resolve_port({"SWE_API_PORT": "8787"}) == 8787

    def test_strips_surrounding_whitespace(self):
        assert _resolve_port({"SWE_API_PORT": "  9000 "}) == 9000

    def test_low_boundary(self):
        assert _resolve_port({"SWE_API_PORT": "1"}) == 1

    def test_high_boundary(self):
        assert _resolve_port({"SWE_API_PORT": "65535"}) == 65535


class TestResolvePortLoudFailure:
    """Bad values raise — silent fallback would defeat the "log which
    port was selected" hygiene (CLAUDE.md §4 has the same lesson for
    provider selection)."""

    def test_non_integer_raises(self):
        with pytest.raises(ValueError, match="base-10 integer"):
            _resolve_port({"SWE_API_PORT": "eight-thousand"})

    def test_float_raises(self):
        """A float value (8787.0) is a typo that must surface loudly,
        not get truncated."""
        with pytest.raises(ValueError, match="base-10 integer"):
            _resolve_port({"SWE_API_PORT": "8787.0"})

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_port({"SWE_API_PORT": "0"})

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_port({"SWE_API_PORT": "-1"})

    def test_above_max_port_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _resolve_port({"SWE_API_PORT": "65536"})


class TestResolvePortReadsProcessEnv:
    """When called with no argument, reads ``os.environ`` directly.
    Smoke-tests the live behaviour."""

    def test_unset_returns_default(self, monkeypatch):
        monkeypatch.delenv("SWE_API_PORT", raising=False)
        assert _resolve_port() == _DEFAULT_API_PORT

    def test_set_returns_override(self, monkeypatch):
        monkeypatch.setenv("SWE_API_PORT", "9999")
        assert _resolve_port() == 9999

    def test_set_to_default_is_no_op(self, monkeypatch):
        monkeypatch.setenv("SWE_API_PORT", "8787")
        assert _resolve_port() == 8787


class TestEngineHTTPServerListenQueueDepth:
    """Pin the kernel listen-queue depth for the production HTTP server.

    The S20 reliability arc measured 133 of 200 ``ConnectionRefusedError``
    at 16 concurrent connect attempts when the stdlib default
    ``request_queue_size = 5`` was in effect. A regression that silently
    reverts to the stdlib default would re-open that production-readiness
    gap without surfacing in any other test. This class makes the regression
    loud.
    """

    def test_engine_http_server_overrides_stdlib_default(self):
        """The engine API server must use a non-default listen-queue depth."""
        import socketserver

        from engine_api import _EngineHTTPServer

        assert _EngineHTTPServer.request_queue_size > socketserver.TCPServer.request_queue_size

    def test_listen_queue_depth_constant_is_128(self):
        """Pin the value to 128 — matches uvicorn / gunicorn defaults and
        the S20 AI-handoff recommendation."""
        from engine_api import _LISTEN_QUEUE_DEPTH

        assert _LISTEN_QUEUE_DEPTH == 128

    def test_engine_http_server_inherits_from_threading(self):
        """Belt-and-suspenders: the subclass must remain a
        ``ThreadingHTTPServer`` so per-request threading still applies
        (S20 G5 isolation property)."""
        from http.server import ThreadingHTTPServer

        from engine_api import _EngineHTTPServer

        assert issubclass(_EngineHTTPServer, ThreadingHTTPServer)

    def test_engine_http_server_class_attribute_matches_constant(self):
        """The subclass's ``request_queue_size`` must match
        ``_LISTEN_QUEUE_DEPTH`` — a guard against accidental
        de-synchronisation between the constant and the class attribute."""
        from engine_api import _LISTEN_QUEUE_DEPTH, _EngineHTTPServer

        assert _EngineHTTPServer.request_queue_size == _LISTEN_QUEUE_DEPTH
