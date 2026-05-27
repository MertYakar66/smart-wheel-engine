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
import pytest

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
    ev_df = pd.DataFrame(
        [
            {
                "ticker": "FAKE",
                "spot": 100.0,
                "strike": 95.0,
                "premium": 2.0,
                "ev_dollars": 50.0,
                "phase": "post_expansion",
            }
        ]
    )
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
    ev_df = pd.DataFrame(
        [
            {
                "ticker": "FAKE",
                "spot": 100.0,
                "strike": 95.0,
                "premium": 2.0,
                "ev_dollars": -25.0,
                "phase": "post_expansion",
            }
        ]
    )
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

    ev_df = pd.DataFrame(
        [
            {
                "ticker": "FAKE",
                "spot": 100.0,
                "strike": 95.0,
                "premium": 2.0,
                "ev_dollars": 50.0,
                "phase": "post_expansion",
            }
        ]
    )
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


# ======================================================================
# D17 dossier soft-warns — R7 (VaR) + R8 (stress + dealer regime)
# ======================================================================
class TestD17DossierSoftWarns:
    """R7 and R8 are downgrade-only soft-warns introduced in #154 C4.
    Each test attaches a PortfolioContext to the dossier and verifies
    the reviewer fires (or skips) the rule against the expected
    trigger conditions."""

    def _proceeding_dossier(self, ticker="TEST", strike=100.0, premium=2.0, ev_dollars=50.0):
        """Build a dossier in 'proceed' state via R5 (EV > threshold)
        with a clean chart (no R2/R3/R4/R6 trigger). Then attach
        portfolio_context per-test and re-run the reviewer."""
        from engine.candidate_dossier import CandidateDossier

        return CandidateDossier(
            ticker=ticker,
            ev_row={
                "ticker": ticker,
                "strike": strike,
                "premium": premium,
                "ev_dollars": ev_dollars,
                "iv": 0.25,
                "dte": 30,
                "spot": strike,
            },
            chart_context=ChartContext(
                ticker=ticker,
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=strike,
                visible_indicators={},  # no phase → R4 skipped
                source="test",
            ),
        )

    def test_no_portfolio_context_skips_r7_and_r8(self):
        """When portfolio_context is None, R7, R8, and R9 do not fire
        and the verdict from R5 (proceed at ev=50) is preserved."""
        from engine.candidate_dossier import EnginePhaseReviewer

        d = self._proceeding_dossier()
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        # No R7/R8/R9 notes in the trail.
        assert not any("R7" in n or "R8" in n or "R9" in n for n in notes)

    def test_r7_var_check_skips_when_no_correlation_or_returns(self):
        """PortfolioContext attached but no returns_data /
        correlation_matrix → check_var returns missing_data → R7
        records a skip note without downgrading."""
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier()
        d.portfolio_context = PortfolioContext(nav=100_000.0)  # no returns/corr
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        # R7 skipped (missing_data), R8 also passes (empty book), so
        # the verdict stays proceed.
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert any("R7: VaR check skipped" in n for n in notes)

    def test_r7_var_breach_downgrades_with_synthetic_returns(self):
        """Give check_var a returns_data input + a candidate that
        will produce a large VaR for the synthetic portfolio. R7
        fires."""
        import numpy as np

        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        idx = pd.date_range("2026-01-01", periods=120, freq="B")
        # Heavy-vol synthetic returns make VaR explode relative to NAV.
        returns = pd.DataFrame(
            {"portfolio": np.random.default_rng(7).normal(0, 0.08, 120)},
            index=idx,
        )
        d = self._proceeding_dossier(ticker="TEST", strike=100.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000.0,  # tiny NAV → any VaR > 5% NAV breaches easily
            spot_prices={"TEST": 100.0},
            returns_data=returns,
        )
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        # Heavy-vol returns at tiny NAV: R7 fires.
        assert verdict == "review"
        assert reason == "portfolio_var_breach"

    def test_r8_stress_breach_downgrades(self):
        """A large concentrated short-put position at small NAV will
        produce a stress drawdown > 8% under the C4 vol-spike → R8
        fires with reason='stress_breach'."""
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier(ticker="TEST", strike=100.0)
        # Pre-load a held short put on the same ticker so the
        # stress-scenario P&L is non-trivial. Tiny NAV makes any
        # drawdown breach the 8% cap.
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "TEST",
                    "option_type": "put",
                    "strike": 100.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"TEST": 100.0},
            nav=5_000.0,
        )
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "stress_breach"

    def test_r8_short_gamma_regime_downgrades(self):
        """If the candidate's underlying is in short_gamma_amplifying
        regime, R8 fires with reason='short_gamma_regime' even when
        stress passes."""
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000_000.0,  # huge NAV → stress doesn't breach
            spot_prices={"AAPL": 180.0},
            dealer_regime_by_ticker={"AAPL": "short_gamma_amplifying"},
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "short_gamma_regime"
        assert any("R8 (dealer)" in n for n in notes)

    def test_r8_passes_on_neutral_regime(self):
        """Neutral regime + clean stress → R8 doesn't fire; verdict
        stays proceed from R5."""
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000_000.0,
            spot_prices={"AAPL": 180.0},
            dealer_regime_by_ticker={"AAPL": "neutral"},
        )
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"

    def test_r7_r8_cannot_upgrade_negative_ev(self):
        """R1 still wins. Even with a PortfolioContext attached and
        gates that would pass, a negative-EV candidate stays
        blocked."""
        from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = CandidateDossier(
            ticker="BAD",
            ev_row={
                "ticker": "BAD",
                "strike": 100.0,
                "premium": 0.5,
                "ev_dollars": -25.0,  # R1 territory
                "iv": 0.25,
                "dte": 30,
                "spot": 100.0,
            },
            chart_context=ChartContext(
                ticker="BAD",
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=100.0,
                visible_indicators={},
                source="test",
            ),
        )
        d.portfolio_context = PortfolioContext(nav=10_000_000.0)
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        # R1 fires before R7/R8 even get a chance.
        assert verdict == "blocked"
        assert reason == "negative_ev"

    def test_r7_r8_cannot_upgrade_review_to_proceed(self):
        """Downgrade-only contract: a candidate already at 'review'
        (e.g., EV below threshold) doesn't get upgraded by R7/R8
        passing. R7/R8 only fire when verdict == 'proceed'."""
        from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = CandidateDossier(
            ticker="LOW",
            ev_row={
                "ticker": "LOW",
                "strike": 100.0,
                "premium": 0.5,
                "ev_dollars": 5.0,  # < min_proceed_ev=10 → 'review'
                "iv": 0.25,
                "dte": 30,
                "spot": 100.0,
            },
            chart_context=ChartContext(
                ticker="LOW",
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=100.0,
                visible_indicators={},
                source="test",
            ),
        )
        d.portfolio_context = PortfolioContext(nav=10_000_000.0)
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "ev_below_proceed_threshold"


class TestD17DossierR9SectorCap:
    """R9 (D17 B2 closure): sector_cap soft-warn on the dossier.

    Mirrors the R7/R8 tests above but exercises ``check_sector_cap``.
    Downgrade-only: an already-blocked or already-review verdict is
    never upgraded. Closes the documented integration test from
    ``docs/PRODUCTION_READINESS.md`` §6 B2.
    """

    def _proceeding_dossier(self, ticker="TEST", strike=100.0, premium=2.0, ev_dollars=50.0):
        from engine.candidate_dossier import CandidateDossier

        return CandidateDossier(
            ticker=ticker,
            ev_row={
                "ticker": ticker,
                "strike": strike,
                "premium": premium,
                "ev_dollars": ev_dollars,
                "iv": 0.25,
                "dte": 30,
                "spot": strike,
                "contracts": 1,
            },
            chart_context=ChartContext(
                ticker=ticker,
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=strike,
                visible_indicators={},
                source="test",
            ),
        )

    def test_r9_sector_cap_breach_downgrades_proceed_to_review(self):
        """A candidate whose proposed notional + held same-sector
        positions exceeds the sector cap (25% NAV default) gets
        downgraded to review with reason='sector_cap_breach'.

        AAPL is in Information Technology in DEFAULT_SECTOR_MAP.
        Pre-load a large existing AAPL short put so opening a second
        AAPL put pushes the IT sector exposure past 25% of a small NAV.
        """
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 180.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,  # one held → $18,000 notional
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 180.0},
            nav=50_000.0,  # post-open IT exposure: $36,000 / $50,000 = 72% > 25% → breach
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review", f"expected review, got {verdict} (notes={notes})"
        assert reason == "sector_cap_breach", f"expected sector_cap_breach, got {reason}"
        assert any("R9" in n for n in notes), f"expected R9 note, got: {notes}"

    def test_r9_passes_when_below_sector_cap(self):
        """Same setup but with a NAV large enough that the post-open
        sector exposure stays well under 25% → R9 doesn't fire,
        verdict stays proceed."""
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 180.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 180.0},
            nav=10_000_000.0,  # post-open IT exposure: $36k / $10M = 0.36% → well below cap
        )
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"

    def test_r9_skips_when_nav_is_zero(self):
        """Defensive: nav=0 or missing → R9 silently no-ops (matches
        Q3 missing-data semantics from R7/R8)."""
        from engine.candidate_dossier import EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = self._proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            spot_prices={"AAPL": 180.0},
            nav=0.0,  # nav=0 → skip
        )
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        # No NAV → no R9 fire, verdict stays proceed from R5.
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"

    def test_r9_cannot_upgrade_negative_ev(self):
        """R1 still wins: negative-EV candidate stays blocked even if
        R9 would have passed."""
        from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer
        from engine.portfolio_risk_gates import PortfolioContext

        d = CandidateDossier(
            ticker="AAPL",
            ev_row={
                "ticker": "AAPL",
                "strike": 180.0,
                "premium": 0.5,
                "ev_dollars": -25.0,
                "iv": 0.25,
                "dte": 30,
                "spot": 180.0,
                "contracts": 1,
            },
            chart_context=ChartContext(
                ticker="AAPL",
                timeframe="1D",
                captured_at=datetime(2026, 4, 25, 12, 0, 0),
                screenshot_path=Path("/tmp/fake.png"),
                visible_price=180.0,
                visible_indicators={},
                source="test",
            ),
        )
        d.portfolio_context = PortfolioContext(nav=10_000_000.0, spot_prices={"AAPL": 180.0})
        verdict, reason, _notes = EnginePhaseReviewer().review(d)
        assert verdict == "blocked"
        assert reason == "negative_ev"
