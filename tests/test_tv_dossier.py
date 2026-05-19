"""
Tests for the TradingView visual-context layer (chart context, bridge,
dossier, and chart reviewer).

Architectural assertions locked in by these tests
-------------------------------------------------
1. The bridge is thin — it only captures URL + screenshot. It does not
   parse numbers out of the chart.
2. The chart reviewer cannot upgrade a negative-EV trade. Ever.
3. The dossier layer degrades gracefully when the bridge fails —
   candidates without charts still produce a valid dossier with
   ``verdict="review"`` and a missing-chart warning.
4. The filesystem provider detects stale screenshots and flags them
   via the ``error`` field.
5. The Playwright provider lazy-imports playwright and returns a
   clearly-flagged error ChartContext when playwright is unavailable,
   never crashing.
6. The engine-first Mode B workflow runs end-to-end: rank by EV →
   attach charts → produce dossiers sorted by the underlying EV
   ordering.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.candidate_dossier import (
    CandidateDossier,
    EnginePhaseReviewer,
    build_dossiers,
)
from engine.chart_context import ChartContext
from engine.tradingview_bridge import (
    ChainedChartProvider,
    FilesystemChartProvider,
    MCPCaptureResult,
    MCPChartProvider,
    MCPClientError,
    PlaywrightChartProvider,
    build_default_provider,
    build_tradingview_url,
)


# =========================================================================
# 1. build_tradingview_url
# =========================================================================
class TestTradingViewURL:
    def test_nasdaq_ticker_maps_to_nasdaq(self):
        url = build_tradingview_url("AAPL", "1D")
        assert "NASDAQ" in url
        assert "AAPL" in url
        assert "interval=D" in url

    def test_unknown_ticker_defaults_to_nyse(self):
        url = build_tradingview_url("XYZ", "1D")
        assert "NYSE" in url

    def test_timeframe_interval_mapping(self):
        assert "interval=60" in build_tradingview_url("AAPL", "1h")
        assert "interval=W" in build_tradingview_url("AAPL", "1W")
        assert "interval=D" in build_tradingview_url("AAPL", "1D")

    def test_custom_exchange_map_override(self):
        url = build_tradingview_url("AAPL", "1D", exchange_map={"AAPL": "AMEX"})
        assert "AMEX" in url


# =========================================================================
# 2. FilesystemChartProvider
# =========================================================================
class TestFilesystemProvider:
    def test_missing_screenshot_returns_error_context(self, tmp_path):
        provider = FilesystemChartProvider(base_dir=tmp_path)
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.error == "screenshot_not_found"
        assert ctx.is_ok() is False
        assert ctx.ticker == "AAPL"
        assert "NASDAQ" in ctx.browser_url

    def test_existing_screenshot_returned(self, tmp_path):
        # Create a fake AAPL/1D.png
        ticker_dir = tmp_path / "AAPL"
        ticker_dir.mkdir()
        img = ticker_dir / "1D.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        provider = FilesystemChartProvider(base_dir=tmp_path)
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.is_ok() is True
        assert ctx.screenshot_path == img
        assert ctx.source == "filesystem"

    def test_stale_screenshot_flagged(self, tmp_path):
        ticker_dir = tmp_path / "AAPL"
        ticker_dir.mkdir()
        img = ticker_dir / "1D.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        # Force mtime into the past
        import os

        old_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=10)
        os.utime(img, (old_time.timestamp(), old_time.timestamp()))

        provider = FilesystemChartProvider(
            base_dir=tmp_path,
            staleness_seconds=3600,  # 1h freshness
        )
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.error.startswith("stale_screenshot")
        assert ctx.is_ok() is False

    def test_ticker_is_uppercased(self, tmp_path):
        ticker_dir = tmp_path / "AAPL"
        ticker_dir.mkdir()
        img = ticker_dir / "1D.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        provider = FilesystemChartProvider(base_dir=tmp_path)
        ctx = provider.fetch("aapl", "1D")
        assert ctx.ticker == "AAPL"
        assert ctx.is_ok() is True


# =========================================================================
# 3. PlaywrightChartProvider graceful-fallback
# =========================================================================
class TestPlaywrightProviderGracefulFallback:
    def test_returns_error_context_when_playwright_missing(self, tmp_path, monkeypatch):
        # Force the playwright check to report not installed.
        provider = PlaywrightChartProvider(output_dir=tmp_path)

        def fake_check(self):
            return False

        monkeypatch.setattr(PlaywrightChartProvider, "_check_playwright", fake_check)

        ctx = provider.fetch("AAPL", "1D")
        assert ctx.error == "playwright_not_installed"
        assert ctx.is_ok() is False
        assert ctx.ticker == "AAPL"
        # Browser URL must still be produced for the audit trail
        assert "tradingview.com" in ctx.browser_url

    def test_never_raises_on_missing_playwright(self, tmp_path):
        provider = PlaywrightChartProvider(output_dir=tmp_path)
        # Should never raise even if playwright is not installed.
        ctx = provider.fetch("AAPL", "1D")
        assert isinstance(ctx, ChartContext)


# =========================================================================
# 4. ChainedChartProvider
# =========================================================================
class TestChainedChartProvider:
    def test_chain_returns_first_success(self, tmp_path):
        # First provider fails (empty dir), second provider has the file.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        real_dir = tmp_path / "real"
        (real_dir / "AAPL").mkdir(parents=True)
        (real_dir / "AAPL" / "1D.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        chain = ChainedChartProvider(
            [
                FilesystemChartProvider(base_dir=empty_dir),
                FilesystemChartProvider(base_dir=real_dir),
            ]
        )
        ctx = chain.fetch("AAPL", "1D")
        assert ctx.is_ok() is True
        assert "real" in str(ctx.screenshot_path)

    def test_chain_returns_last_error_when_all_fail(self, tmp_path):
        empty1 = tmp_path / "e1"
        empty2 = tmp_path / "e2"
        empty1.mkdir()
        empty2.mkdir()
        chain = ChainedChartProvider(
            [
                FilesystemChartProvider(base_dir=empty1),
                FilesystemChartProvider(base_dir=empty2),
            ]
        )
        ctx = chain.fetch("AAPL", "1D")
        assert ctx.is_ok() is False
        assert ctx.error != ""

    def test_empty_chain_returns_error(self):
        chain = ChainedChartProvider(providers=[])
        ctx = chain.fetch("AAPL", "1D")
        assert ctx.error == "no_providers_configured"


# =========================================================================
# 5. EnginePhaseReviewer — hard guardrails
# =========================================================================
class TestEnginePhaseReviewer:
    def _dossier(
        self,
        ev=50.0,
        spot=100.0,
        chart: ChartContext | None = None,
        phase="post_expansion",
    ) -> CandidateDossier:
        row = {
            "ticker": "AAPL",
            "ev_dollars": ev,
            "spot": spot,
            "phase": phase,
        }
        return CandidateDossier(ticker="AAPL", ev_row=row, chart_context=chart)

    def test_negative_ev_blocked_even_with_perfect_chart(self):
        """Hard guardrail: chart cannot upgrade a negative-EV trade."""
        chart = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime.now(UTC).replace(tzinfo=None),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.0,
            source="test",
        )
        dossier = self._dossier(ev=-50.0, chart=chart)
        reviewer = EnginePhaseReviewer()
        verdict, reason, _ = reviewer.review(dossier)
        assert verdict == "blocked"
        assert reason == "negative_ev"

    def test_missing_chart_degrades_to_review(self):
        reviewer = EnginePhaseReviewer()
        # Engine result with no chart context
        dossier = self._dossier(ev=50.0, chart=None)
        verdict, reason, _ = reviewer.review(dossier)
        assert verdict == "review"
        assert reason == "chart_context_missing"

    def test_errored_chart_degrades_to_review(self):
        reviewer = EnginePhaseReviewer()
        errored = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime.now(UTC).replace(tzinfo=None),
            screenshot_path=None,
            error="screenshot_not_found",
        )
        verdict, reason, _ = reviewer.review(self._dossier(ev=50.0, chart=errored))
        assert verdict == "review"
        assert reason == "chart_context_missing"

    def test_spot_price_mismatch_triggers_skip(self):
        chart = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime.now(UTC).replace(tzinfo=None),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=150.0,  # huge mismatch
            source="test",
        )
        dossier = self._dossier(ev=50.0, spot=100.0, chart=chart)
        reviewer = EnginePhaseReviewer(spot_tolerance_pct=0.02)
        verdict, reason, notes = reviewer.review(dossier)
        assert verdict == "skip"
        assert reason == "spot_price_mismatch"
        assert any("disagrees" in n for n in notes)

    def test_phase_contradiction_triggers_skip(self):
        chart = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime.now(UTC).replace(tzinfo=None),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.0,
            visible_indicators={"phase": "compression"},
            source="test",
        )
        dossier = self._dossier(ev=50.0, spot=100.0, chart=chart, phase="post_expansion")
        verdict, reason, _ = EnginePhaseReviewer().review(dossier)
        assert verdict == "skip"
        assert reason == "phase_contradiction"

    def test_clean_chart_above_threshold_proceeds(self):
        chart = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime.now(UTC).replace(tzinfo=None),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.05,  # within tolerance
            source="test",
        )
        dossier = self._dossier(ev=50.0, spot=100.0, chart=chart)
        verdict, reason, _ = EnginePhaseReviewer(min_proceed_ev=10.0).review(dossier)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"

    def test_low_ev_below_threshold_is_review(self):
        chart = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime.now(UTC).replace(tzinfo=None),
            screenshot_path=Path("/tmp/fake.png"),
            visible_price=100.0,
            source="test",
        )
        dossier = self._dossier(ev=5.0, spot=100.0, chart=chart)
        verdict, reason, _ = EnginePhaseReviewer(min_proceed_ev=10.0).review(dossier)
        assert verdict == "review"
        assert reason == "ev_below_proceed_threshold"


# =========================================================================
# 6. build_dossiers end-to-end
# =========================================================================
class TestBuildDossiers:
    def test_end_to_end_dossier_build(self, tmp_path):
        # Fake a ranked EV frame
        ev_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "spot": 187.4,
                    "strike": 180.0,
                    "premium": 2.15,
                    "ev_dollars": 25.0,
                    "prob_profit": 0.75,
                },
                {
                    "ticker": "MSFT",
                    "spot": 420.0,
                    "strike": 410.0,
                    "premium": 3.10,
                    "ev_dollars": -5.0,  # negative — must be blocked
                    "prob_profit": 0.55,
                },
            ]
        )
        # Create a screenshot for AAPL but NOT for MSFT
        aapl_dir = tmp_path / "AAPL"
        aapl_dir.mkdir()
        (aapl_dir / "1D.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        provider = FilesystemChartProvider(base_dir=tmp_path)
        dossiers = build_dossiers(
            ev_frame=ev_df,
            provider=provider,
            timeframe="1D",
            top_n=5,
        )

        assert len(dossiers) == 2

        aapl = next(d for d in dossiers if d.ticker == "AAPL")
        assert aapl.has_chart is True
        assert aapl.verdict in ("proceed", "review")
        assert aapl.ev_row["spot"] == 187.4

        msft = next(d for d in dossiers if d.ticker == "MSFT")
        # Even though MSFT has no chart, the NEGATIVE EV wins first —
        # the reviewer hits the negative_ev short-circuit before
        # checking chart context.
        assert msft.verdict == "blocked"
        assert msft.verdict_reason == "negative_ev"

    def test_empty_frame_returns_empty_dossiers(self):
        provider = FilesystemChartProvider(base_dir="/tmp")
        assert build_dossiers(ev_frame=pd.DataFrame(), provider=provider) == []

    def test_top_n_limits_chart_attachment(self, tmp_path):
        ev_df = pd.DataFrame(
            [{"ticker": f"T{i}", "spot": 100.0, "ev_dollars": 20.0} for i in range(10)]
        )
        provider = FilesystemChartProvider(base_dir=tmp_path)
        dossiers = build_dossiers(ev_frame=ev_df, provider=provider, top_n=3)
        assert len(dossiers) == 3


# =========================================================================
# 7. WheelRunner Mode B integration
# =========================================================================
class TestWheelRunnerDossierMode:
    def test_mode_b_runs_end_to_end(self, tmp_path):
        from engine.wheel_runner import WheelRunner

        # Build synthetic OHLCV
        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))

        class FakeConn:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": prices}, index=idx)

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.25,
                    "volatility_30d": 0.22,
                    "dividend_yield": 0.01,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["SYN1", "SYN2"]

        runner = WheelRunner()
        runner._connector = FakeConn()

        # Create a screenshot for SYN1 only
        (tmp_path / "SYN1").mkdir()
        (tmp_path / "SYN1" / "1D.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        provider = FilesystemChartProvider(base_dir=tmp_path)
        dossiers = runner.build_candidate_dossiers(
            tickers=["SYN1", "SYN2"],
            dte_target=30,
            delta_target=0.25,
            top_n=5,
            min_ev_dollars=-1e9,
            chart_provider=provider,
            chart_timeframe="1D",
        )
        # Both tickers should produce a dossier
        assert len(dossiers) >= 1
        # At least one should have a chart; SYN2 should not
        has_chart_map = {d.ticker: d.has_chart for d in dossiers}
        if "SYN1" in has_chart_map:
            assert has_chart_map["SYN1"] is True
        if "SYN2" in has_chart_map:
            assert has_chart_map["SYN2"] is False

        # Every dossier should have a structured verdict
        for d in dossiers:
            assert d.verdict in ("proceed", "review", "skip", "blocked")
            assert d.review_notes  # never empty — every review logs why

    def test_mode_b_sorted_by_ev_per_day(self, tmp_path):
        from engine.wheel_runner import WheelRunner

        rng = np.random.default_rng(0)
        n = 1200
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.010, n)))

        class FakeConn:
            def get_ohlcv(self, ticker):
                return pd.DataFrame({"close": prices}, index=idx)

            def get_fundamentals(self, ticker):
                return {
                    "implied_vol_atm": 0.25,
                    "volatility_30d": 0.22,
                    "dividend_yield": 0.01,
                }

            def get_risk_free_rate(self, as_of=None):
                return 0.05

            def get_next_earnings(self, ticker, as_of=None):
                return None

            def get_universe(self):
                return ["AAA", "BBB", "CCC"]

        runner = WheelRunner()
        runner._connector = FakeConn()

        provider = FilesystemChartProvider(base_dir=tmp_path)
        dossiers = runner.build_candidate_dossiers(
            tickers=["AAA", "BBB", "CCC"],
            top_n=3,
            min_ev_dollars=-1e9,
            chart_provider=provider,
        )
        # Extract ev_per_day from the rows — should be non-increasing
        ev_per_day = [d.ev_row.get("ev_per_day", 0) for d in dossiers]
        assert ev_per_day == sorted(ev_per_day, reverse=True)


# =========================================================================
# 8. ChartContext.to_dict JSON safety
# =========================================================================
class TestChartContextSerialisation:
    def test_to_dict_is_json_safe(self):
        import json

        ctx = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime(2026, 4, 14, 10, 30),
            screenshot_path=Path("/tmp/aapl.png"),
            visible_price=187.4,
            visible_indicators={"rsi": 54.3, "atr": 2.15},
            source="filesystem",
            browser_url="https://www.tradingview.com/chart/?symbol=NASDAQ%3AAAPL",
            notes="",
            error="",
        )
        d = ctx.to_dict()
        json.dumps(d)  # must not raise
        assert d["ok"] is True
        assert d["visible_price"] == 187.4
        # Accept either Unix or Windows path separator — pathlib.Path
        # normalises to the host OS style. The invariant we care about
        # is that the serialised path contains the source components,
        # not the slash direction.
        assert Path(d["screenshot_path"]) == Path("/tmp/aapl.png")

    def test_errored_context_serialisation(self):
        ctx = ChartContext(
            ticker="AAPL",
            timeframe="1D",
            captured_at=datetime(2026, 4, 14),
            error="screenshot_not_found",
        )
        d = ctx.to_dict()
        assert d["ok"] is False
        assert d["screenshot_path"] is None
        assert d["error"] == "screenshot_not_found"


# =========================================================================
# 9. build_default_provider convenience factory
# =========================================================================
class TestBuildDefaultProvider:
    def test_default_is_filesystem_only(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SWE_USE_MCP_CHART", raising=False)
        p = build_default_provider(screenshots_dir=tmp_path)
        assert isinstance(p, FilesystemChartProvider)

    def test_with_playwright_returns_chained(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SWE_USE_MCP_CHART", raising=False)
        p = build_default_provider(screenshots_dir=tmp_path, enable_playwright_fallback=True)
        assert isinstance(p, ChainedChartProvider)
        assert len(p.providers) == 2

    def test_mcp_opt_in_via_env(self, tmp_path, monkeypatch):
        # Stage 3 / contract §8 q3: SWE_USE_MCP_CHART opts MCP into the
        # chain — live MCP first, cached filesystem second (§4 order).
        monkeypatch.setenv("SWE_USE_MCP_CHART", "1")
        p = build_default_provider(screenshots_dir=tmp_path)
        assert isinstance(p, ChainedChartProvider)
        assert len(p.providers) == 2
        assert isinstance(p.providers[0], MCPChartProvider)
        assert isinstance(p.providers[1], FilesystemChartProvider)

    def test_mcp_explicit_enable_overrides_unset_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SWE_USE_MCP_CHART", raising=False)
        p = build_default_provider(screenshots_dir=tmp_path, enable_mcp=True)
        assert isinstance(p, ChainedChartProvider)
        assert isinstance(p.providers[0], MCPChartProvider)

    def test_mcp_explicit_disable_overrides_env(self, tmp_path, monkeypatch):
        # An explicit enable_mcp=False beats the environment variable.
        monkeypatch.setenv("SWE_USE_MCP_CHART", "1")
        p = build_default_provider(screenshots_dir=tmp_path, enable_mcp=False)
        assert isinstance(p, FilesystemChartProvider)

    def test_mcp_env_falsey_value_stays_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SWE_USE_MCP_CHART", "0")
        p = build_default_provider(screenshots_dir=tmp_path)
        assert isinstance(p, FilesystemChartProvider)

    def test_mcp_with_playwright_canonical_order(self, tmp_path, monkeypatch):
        # Full chain follows contract §4: MCP → filesystem → Playwright.
        monkeypatch.delenv("SWE_USE_MCP_CHART", raising=False)
        p = build_default_provider(
            screenshots_dir=tmp_path,
            enable_playwright_fallback=True,
            enable_mcp=True,
        )
        assert isinstance(p, ChainedChartProvider)
        assert len(p.providers) == 3
        assert isinstance(p.providers[0], MCPChartProvider)
        assert isinstance(p.providers[1], FilesystemChartProvider)
        assert isinstance(p.providers[2], PlaywrightChartProvider)


# =========================================================================
# 10. MCPChartProvider (integration Stage 1 — offline contract skeleton)
# =========================================================================
class _FakeMCPClient:
    """Test double for an MCPChartClient.

    Returns a fixed :class:`MCPCaptureResult`, or raises a fixed
    exception, and records every call so tests can assert call count
    (the contract forbids retries inside the provider).
    """

    def __init__(self, *, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, str]] = []

    def capture(self, ticker, timeframe):
        self.calls.append((ticker, timeframe))
        if self._exc is not None:
            raise self._exc
        return self._result


class TestMCPChartProvider:
    """Provider mechanics. The dossier-routing contract (errored context
    routes to review, PIT refusal) is pinned separately by the
    now-active tests/test_dossier_invariant.py::test_mcp_provider_* tests.
    """

    def test_default_constructor_is_constructible_and_unconfigured(self):
        # MCPChartProvider() with no args must work — the contract test
        # builds one. A live fetch honestly reports mcp_unavailable
        # because no tradingview-mcp server is wired in Stage 1.
        provider = MCPChartProvider()
        ctx = provider.fetch("AAPL", "1D")
        assert ctx.error == "mcp_unavailable"
        assert ctx.is_ok() is False
        assert ctx.source == "mcp"
        assert "tradingview.com" in ctx.browser_url

    def test_pit_violation_short_circuits_before_client(self):
        # as_of set → pit_violation, and the MCP client is NOT consulted
        # (a live screenshot in a backtest is a look-ahead leak).
        client = _FakeMCPClient(result=MCPCaptureResult(screenshot_path=Path("/unused.png")))
        provider = MCPChartProvider(client=client)
        ctx = provider.fetch("AAPL", "1D", as_of=datetime(2024, 6, 1))
        assert ctx.error == "pit_violation"
        assert ctx.screenshot_path is None
        assert ctx.is_ok() is False
        assert client.calls == []

    def test_with_forced_error_produces_errored_context(self):
        provider = MCPChartProvider.with_forced_error("screenshot_timeout")
        ctx = provider.fetch("msft", "1D")
        assert ctx.error == "screenshot_timeout"
        assert ctx.is_ok() is False
        assert ctx.ticker == "MSFT"

    def test_with_forced_error_rejects_noncanonical_value(self):
        # Contract §7: error strings are not invented ad hoc.
        with pytest.raises(ValueError):
            MCPChartProvider.with_forced_error("totally_made_up")

    def test_successful_capture_builds_ok_context(self, tmp_path):
        png = tmp_path / "AAPL_1D.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        client = _FakeMCPClient(result=MCPCaptureResult(screenshot_path=png, visible_price=201.5))
        ctx = MCPChartProvider(client=client).fetch("aapl", "1D")
        assert ctx.is_ok() is True
        assert ctx.error == ""
        assert ctx.screenshot_path == png
        assert ctx.visible_price == 201.5
        assert ctx.source == "mcp"
        assert ctx.ticker == "AAPL"

    def test_visible_indicators_empty_in_m1(self, tmp_path):
        # Contract §5: M1 does not populate a chart-derived phase.
        png = tmp_path / "x.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        client = _FakeMCPClient(result=MCPCaptureResult(screenshot_path=png))
        ctx = MCPChartProvider(client=client).fetch("AAPL", "1D")
        assert ctx.visible_indicators == {}

    def test_client_error_maps_to_taxonomy(self):
        client = _FakeMCPClient(exc=MCPClientError("browser_disconnected", "CDP session died"))
        ctx = MCPChartProvider(client=client).fetch("AAPL", "1D")
        assert ctx.error == "browser_disconnected"
        assert ctx.is_ok() is False

    def test_unexpected_exception_maps_to_unexpected_error(self):
        # A non-MCPClientError leaking out of the client must not crash
        # the provider — it maps to the canonical unexpected_error.
        client = _FakeMCPClient(exc=RuntimeError("boom"))
        ctx = MCPChartProvider(client=client).fetch("AAPL", "1D")
        assert ctx.error == "unexpected_error"
        assert ctx.is_ok() is False

    def test_missing_screenshot_file_is_not_a_quiet_pass(self, tmp_path):
        # Client claims success but the PNG is not on disk → the
        # provider surfaces unexpected_error, never an is_ok()=False
        # context with a blank error field.
        ghost = tmp_path / "never_written.png"
        client = _FakeMCPClient(result=MCPCaptureResult(screenshot_path=ghost))
        ctx = MCPChartProvider(client=client).fetch("AAPL", "1D")
        assert ctx.is_ok() is False
        assert ctx.error == "unexpected_error"

    def test_never_raises_on_client_failure(self):
        client = _FakeMCPClient(exc=RuntimeError("kaboom"))
        ctx = MCPChartProvider(client=client).fetch("AAPL", "1D")
        assert isinstance(ctx, ChartContext)

    def test_no_retries_one_client_call_per_fetch(self):
        # Contract §7: no retries inside the provider.
        client = _FakeMCPClient(exc=MCPClientError("mcp_unavailable"))
        MCPChartProvider(client=client).fetch("AAPL", "1D")
        assert len(client.calls) == 1


class TestMCPChartProviderInChain:
    """MCPChartProvider as one backend in a ChainedChartProvider."""

    def test_chain_falls_through_mcp_to_filesystem(self, tmp_path):
        # MCP errors → the chain falls through to a real filesystem
        # screenshot. Not a "quiet substitution": the MCP provider
        # returned an honest errored context and the chain legitimately
        # used the next backend (design contract §4).
        real_dir = tmp_path / "shots"
        (real_dir / "AAPL").mkdir(parents=True)
        (real_dir / "AAPL" / "1D.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        chain = ChainedChartProvider(
            [
                MCPChartProvider.with_forced_error("mcp_unavailable"),
                FilesystemChartProvider(base_dir=real_dir),
            ]
        )
        ctx = chain.fetch("AAPL", "1D")
        assert ctx.is_ok() is True
        assert ctx.source == "filesystem"

    def test_chain_surfaces_mcp_error_when_every_provider_fails(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        chain = ChainedChartProvider(
            [
                FilesystemChartProvider(base_dir=empty),
                MCPChartProvider.with_forced_error("mcp_unavailable"),
            ]
        )
        ctx = chain.fetch("AAPL", "1D")
        assert ctx.is_ok() is False
        assert ctx.error == "mcp_unavailable"
