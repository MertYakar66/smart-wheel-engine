"""
Invariant: any ChartContextProvider wired into the dossier layer flows
through a reviewer that can only downgrade, never upgrade.

Locked in to prevent future MCP-backed providers (e.g. the planned
TradingView MCP integration) from bypassing EnginePhaseReviewer's
R1-R6 rules. See CLAUDE.md §2 and §7.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.candidate_dossier import build_dossiers
from engine.chart_context import ChartContext, Timeframe


@dataclass
class _FakeMCPProvider:
    """Stand-in for a hypothetical MCP-backed ChartContextProvider.

    Returns a fixed ChartContext on fetch. The test's job is to verify
    that whatever comes out of this provider *still* flows through the
    default EnginePhaseReviewer — i.e. a novel provider cannot opt out
    of rule-based review by being novel.
    """

    chart_context: ChartContext

    def fetch(self, ticker: str, timeframe: Timeframe = "1D", *, as_of=None) -> ChartContext:
        return self.chart_context


def _chart(phase: str, spot: float = 100.0) -> ChartContext:
    return ChartContext(
        ticker="FAKE",
        timeframe="1D",
        captured_at=datetime(2026, 4, 25, 12, 0, 0),
        screenshot_path=Path("/tmp/fake.png"),
        visible_price=spot,
        visible_indicators={"phase": phase},
        source="test_fake_mcp_provider",
    )


def test_custom_provider_phase_contradiction_routes_through_reviewer():
    """R4: custom provider + positive EV + contradicting phase → skip/phase_contradiction.

    Proves build_dossiers always routes through the reviewer regardless
    of which ChartContextProvider implementation is supplied.
    """
    ev_df = pd.DataFrame([{
        "ticker": "FAKE",
        "spot": 100.0,
        "strike": 95.0,
        "premium": 2.0,
        "ev_dollars": 50.0,
        "phase": "post_expansion",
    }])
    provider = _FakeMCPProvider(chart_context=_chart(phase="compression", spot=100.0))
    dossiers = build_dossiers(ev_frame=ev_df, provider=provider, top_n=5)
    assert len(dossiers) == 1
    d = dossiers[0]
    assert d.verdict == "skip", (
        f"custom provider resolved to verdict={d.verdict!r} — reviewer was not applied"
    )
    assert d.verdict_reason == "phase_contradiction"


def test_custom_provider_pristine_chart_cannot_upgrade_negative_ev():
    """R1: custom provider + pristine chart + negative EV → blocked/negative_ev.

    Downgrade-only invariant: a flawless-looking ChartContext must not
    rescue a negative-EV trade. R1 short-circuits before the chart is
    even consulted.
    """
    ev_df = pd.DataFrame([{
        "ticker": "FAKE",
        "spot": 100.0,
        "strike": 95.0,
        "premium": 2.0,
        "ev_dollars": -25.0,
        "phase": "post_expansion",
    }])
    provider = _FakeMCPProvider(chart_context=_chart(phase="post_expansion", spot=100.0))
    dossiers = build_dossiers(ev_frame=ev_df, provider=provider, top_n=5)
    assert len(dossiers) == 1
    d = dossiers[0]
    assert d.verdict == "blocked", (
        f"pristine chart rescued negative-EV trade (verdict={d.verdict!r}) — R1 breached"
    )
    assert d.verdict_reason == "negative_ev"


# ----------------------------------------------------------------------
# MCP provider contract tests (pending implementation)
# ----------------------------------------------------------------------
# These tests are import-guarded — dormant until
# engine.tradingview_bridge.MCPChartProvider exists, then auto-activate
# and pin the contract structurally. See:
#   docs/TRADINGVIEW_MCP_INTEGRATION.md §3 (hard invariants),
#   §7 (missing-data contract), §8 q4 (PIT discipline).

import pytest

try:
    from engine.tradingview_bridge import MCPChartProvider  # noqa: F401
    _HAS_MCP_PROVIDER = True
except ImportError:
    _HAS_MCP_PROVIDER = False


@pytest.mark.skipif(
    not _HAS_MCP_PROVIDER,
    reason="MCPChartProvider not yet implemented — see docs/TRADINGVIEW_MCP_INTEGRATION.md",
)
def test_mcp_provider_errored_context_routes_to_review():
    """Contract: MCPChartProvider returning error='mcp_unavailable' flows
    through the dossier layer, hits R2, resolves to review/chart_context_missing.

    Pins 'no quiet substitution' (§3, §7).

    Requires the test seam from §7: MCPChartProvider must expose a
    deterministic way to force `fetch()` to return a ChartContext with
    a specific `error` value — e.g. `MCPChartProvider.with_forced_error(...)`,
    an injectable client, or a subclass.
    """
    from engine.tradingview_bridge import MCPChartProvider
    provider = MCPChartProvider.with_forced_error("mcp_unavailable")

    ev_df = pd.DataFrame([{
        "ticker": "FAKE",
        "spot": 100.0,
        "strike": 95.0,
        "premium": 2.0,
        "ev_dollars": 50.0,
        "phase": "post_expansion",
    }])
    dossiers = build_dossiers(ev_frame=ev_df, provider=provider, top_n=5)
    assert len(dossiers) == 1
    d = dossiers[0]
    assert d.verdict == "review", (
        f"errored MCP context did not route to review (verdict={d.verdict!r}) — "
        "no-quiet-substitution rule breached"
    )
    assert d.verdict_reason == "chart_context_missing"


@pytest.mark.skipif(
    not _HAS_MCP_PROVIDER,
    reason="MCPChartProvider not yet implemented — see docs/TRADINGVIEW_MCP_INTEGRATION.md",
)
def test_mcp_provider_pit_violation_when_as_of_set():
    """Contract: when as_of is set (PIT/backtest mode), MCPChartProvider
    must return ChartContext(error='pit_violation') — a live screenshot
    in a backtest is a look-ahead leak.

    Pins §8 question 4 (PIT discipline).
    """
    from datetime import datetime
    from engine.tradingview_bridge import MCPChartProvider
    provider = MCPChartProvider()  # live, no forced error

    ctx = provider.fetch("AAPL", "1D", as_of=datetime(2024, 6, 1))
    assert ctx.error == "pit_violation", (
        f"MCPChartProvider returned ctx.error={ctx.error!r} when as_of was set — "
        "should have refused the live call to avoid look-ahead leak"
    )
    assert ctx.screenshot_path is None
