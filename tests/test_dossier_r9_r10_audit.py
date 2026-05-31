"""S42 — R9 + R10 reviewer audit (Terminal D).

The dossier downgrade ruleset expanded from R1-R8 to R1-R10 with:

* PR #255 — R9 sector_cap soft-warn (D17 B2 closure).
* PR #262 — R10 single-name exposure cap (F4 damage-bounding).

``tests/test_dossier_invariant.py`` already carries seven happy-path
spot checks across the two new rules. This file is the structural
audit that pins the **systemic** properties an external operator would
need to trust the two rules in production. Six probe families:

1. R9 sector_cap fires when it should (default cap, multi-position
   aggregation, unknown-sector fallback).
2. R10 single-name cap fires when it should (locked 10% default,
   put+call aggregation, long-skip).
3. Downgrade-only invariant — neither R9 nor R10 can rescue a
   verdict that R1 (negative / non-finite EV), R7 (VaR breach), or
   R8 (stress / dealer regime) already routed elsewhere.
4. Fail-closed on missing context — no spurious fires, no
   exceptions, when ``PortfolioContext`` is absent or malformed.
5. Cross-rule interaction (highest-value coverage) — when multiple
   soft-warns would fire, the rule order is R7 -> R8(stress) ->
   R8(dealer) -> R9 -> R10 by ``EnginePhaseReviewer.review`` code
   order. First firing rule short-circuits via ``return``; later
   rules' notes are absent from ``review_notes``.
6. Edge cases — cap-boundary semantics (strict ``>``; exact cap
   passes), zero proposed notional, malformed held positions.

Boundary semantics pinned by this audit (so a future tightening to
``>=`` trips a test):

* ``SectorExposureManager.check_sector_limit`` uses ``new_pct >
  max_sector_pct`` (strict). Exact 25% sector passes R9; anything
  strictly above fires.
* ``check_single_name_cap`` uses ``post_open_pct >
  max_single_name_pct`` (strict). Exact 10% single-name passes R10;
  anything strictly above fires.
* Dossier guard: R9 and R10 only run when ``nav > 0`` **and**
  ``proposed_notional > 0`` (``candidate_dossier.py`` lines 439, 474).
  Zero on either side silently skips.

Rule-order in ``EnginePhaseReviewer.review`` (for Family 5):
R1 -> R2 -> R3 -> R4 -> R5 (sets verdict) -> R6 -> R7 -> R8(stress)
-> R8(dealer) -> R9 -> R10. Each soft-warn returns early on first
firing; ``review_notes`` only contains notes from rules that
actually ran.

Reason-code strings are pinned (asserted as literals) so a future
refactor that renames a code (``"sector_cap_breach"`` ->
``"sector_breach"``) trips the audit before merging.

This file is read-only on ``engine/candidate_dossier.py`` and
``engine/portfolio_risk_gates.py``. No source edits.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.candidate_dossier import (
    MIN_PROCEED_EV_DOLLARS,
    CandidateDossier,
    EnginePhaseReviewer,
)
from engine.chart_context import ChartContext
from engine.portfolio_risk_gates import (
    _DEFAULT_MAX_SECTOR_PCT,
    _DEFAULT_MAX_SINGLE_NAME_PCT,
    PortfolioContext,
)


def _clean_chart(ticker: str, spot: float) -> ChartContext:
    """A ChartContext that passes R2/R3/R4 cleanly."""
    return ChartContext(
        ticker=ticker,
        timeframe="1D",
        captured_at=datetime(2026, 4, 25, 12, 0, 0),
        screenshot_path=Path("/tmp/fake.png"),
        visible_price=spot,
        visible_indicators={},
        source="s42_audit",
    )


def _proceeding_dossier(
    ticker: str = "AAPL",
    strike: float = 100.0,
    premium: float = 2.0,
    ev_dollars: float = 50.0,
    contracts: int = 1,
) -> CandidateDossier:
    """A dossier that R1-R6 leave in 'proceed' state.

    The reviewer's R5 sets ``verdict="proceed"`` because ``ev_dollars
    >= MIN_PROCEED_EV_DOLLARS``; R7/R8/R9/R10 then have a chance to
    downgrade. R2/R3/R4/R6 are silenced by the clean chart and the
    absence of ``market_structure``.
    """
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
            "contracts": contracts,
        },
        chart_context=_clean_chart(ticker, strike),
    )


# ======================================================================
# Family 1 — R9 sector_cap fires when it should
# ======================================================================
class TestS42F1_R9FiresCorrectly:
    """The R9 sector_cap soft-warn fires the verdict from 'proceed'
    to 'review' when opening the candidate would push the candidate's
    GICS sector over ``max_sector_pct * NAV`` (default 25% per
    ``_DEFAULT_MAX_SECTOR_PCT``)."""

    def test_r9_fires_at_threshold_exceeded_with_default_25pct_cap(self) -> None:
        """AAPL ($18k held) + AAPL ($18k proposed) at $50k NAV →
        Information Technology = 72% > 25% → R9 downgrades."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
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
            nav=50_000.0,
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"
        # Pin the literal R9 note prefix + sector name + cap pct (D17
        # locked default 25%) so a future refactor that changes the
        # wording trips the audit.
        r9_notes = [n for n in notes if "R9" in n]
        assert len(r9_notes) == 1
        assert "Information Technology" in r9_notes[0]
        assert "25.0%" in r9_notes[0]

    def test_r9_does_not_fire_when_post_open_below_cap(self) -> None:
        """Same held + proposed but NAV inflated → post-open IT
        exposure ~0.18% << 25%, R9 does not fire."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
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
            nav=20_000_000.0,
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R9" in n for n in notes)

    def test_r9_uses_dossier_locked_default_25pct_cap(self) -> None:
        """Pin the D17 locked default. The dossier does not surface
        a ``max_sector_pct`` override; ``check_sector_cap`` is called
        with the function default ``_DEFAULT_MAX_SECTOR_PCT = 0.25``.
        Tightening or loosening the constant must be a deliberate D-
        number decision, not a stealth edit."""
        assert _DEFAULT_MAX_SECTOR_PCT == 0.25
        # Construct a case that fires AT 25.001% so the verdict note
        # records the 25% limit string.
        d = _proceeding_dossier(ticker="AAPL", strike=125.10)  # $12,510 proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 125.10,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 125.10},
            nav=100_000.0,  # 2 * 12,510 / 100,000 = 25.02% > 25%
        )
        verdict, reason, _ = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"

    def test_r9_aggregates_multiple_held_same_sector_positions(self) -> None:
        """AAPL + MSFT are both Information Technology in
        ``DEFAULT_SECTOR_MAP``. Holding $15k AAPL + $15k MSFT and
        opening a $5k AAPL candidate sums all three into IT for the
        cap check: $35k / $100k NAV = 35% > 25%."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)  # $5k proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 150.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
                {
                    "symbol": "MSFT",
                    "option_type": "put",
                    "strike": 150.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
            ],
            spot_prices={"AAPL": 50.0, "MSFT": 150.0},
            nav=100_000.0,
        )
        verdict, reason, _ = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"

    def test_r9_unknown_sector_ticker_aggregates_under_unknown(self) -> None:
        """A ticker not in ``DEFAULT_SECTOR_MAP`` resolves to
        ``"Unknown"`` via ``SectorExposureManager.get_sector``.
        ``check_sector_cap`` does not crash; the cap rule applies to
        the Unknown bucket the same way as any other sector."""
        d = _proceeding_dossier(ticker="ZZZZ", strike=200.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "ZZZZ",
                    "option_type": "put",
                    "strike": 200.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"ZZZZ": 200.0},
            nav=50_000.0,  # 40k notional / 50k = 80% > 25%
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"
        r9_notes = [n for n in notes if "R9" in n]
        assert len(r9_notes) == 1
        assert "Unknown" in r9_notes[0]


# ======================================================================
# Family 2 — R10 single-name cap fires when it should
# ======================================================================
class TestS42F2_R10FiresCorrectly:
    """The R10 single-name cap soft-warn fires when opening the
    candidate would push the per-underlying SHORT option notional
    over ``max_single_name_pct * NAV`` (default 10% per
    ``_DEFAULT_MAX_SINGLE_NAME_PCT``). Sits beneath R9: a ticker
    concentrated as the dominant name in its sector can pass R9 at
    25% NAV but still trip R10 at 10% NAV."""

    def test_r10_locked_default_threshold_is_ten_percent_of_nav(self) -> None:
        """Pin the D17 / F4-damage-bounding default. Same rationale as
        F1.3 — changing this requires a D-number, not a stealth
        constant edit."""
        assert _DEFAULT_MAX_SINGLE_NAME_PCT == 0.10
        # 8% held + 5% proposed = 13% > 10% → R10 fires.
        d = _proceeding_dossier(ticker="BKNG", strike=50.0)  # $5k proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "BKNG",
                    "option_type": "put",
                    "strike": 80.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"BKNG": 100.0},
            nav=100_000.0,
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "single_name_breach"
        r10_notes = [n for n in notes if "R10" in n]
        assert len(r10_notes) == 1
        assert "BKNG" in r10_notes[0]
        assert "10.0%" in r10_notes[0]

    def test_r10_aggregates_multiple_holdings_on_same_ticker(self) -> None:
        """Two existing AAPL short puts at different strikes both
        contribute to the AAPL aggregate. $5k held #1 + $5k held #2 +
        $5k proposed = $15k / $100k NAV = 15% > 10% → R10 fires."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 50.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 50.0,
                    "dte": 45,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
            ],
            spot_prices={"AAPL": 100.0},
            nav=100_000.0,
        )
        # Held = $10k; proposed = $5k; nav = $100k → 15% > 10% →
        # R9 doesn't fire (15% < 25% Info Tech), R10 fires.
        verdict, reason, _ = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "single_name_breach"

    def test_r10_ignores_long_positions_on_same_ticker(self) -> None:
        """Per ``check_single_name_cap`` aggregation rule: only
        ``is_short=True`` positions contribute. A long position on the
        same ticker is skipped."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)  # $5k proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                # Long position — should be ignored.
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 200.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": False,  # long
                }
            ],
            spot_prices={"AAPL": 100.0},
            nav=100_000.0,
        )
        # Held short = $0 (long skipped); proposed = $5k; 5% < 10% → no R10.
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R10" in n for n in notes)

    def test_r10_aggregates_short_put_and_short_call_on_same_ticker(self) -> None:
        """Both put and call short legs contribute to the same-name
        aggregate (the gate sums short notional regardless of
        ``option_type``)."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)  # $5k proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 50.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
                {
                    "symbol": "AAPL",
                    "option_type": "call",  # short call
                    "strike": 60.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                },
            ],
            spot_prices={"AAPL": 100.0},
            nav=100_000.0,
        )
        # Held = $5k put + $6k call = $11k; proposed = $5k → $16k =
        # 16% > 10% → R10 fires. (Sector exposure is also 16% < 25%
        # → R9 does not fire first.)
        verdict, reason, _ = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "single_name_breach"

    def test_r10_different_ticker_holding_does_not_contribute(self) -> None:
        """MSFT held + AAPL candidate: R10 aggregates by SYMBOL, not
        by sector. $20k MSFT held doesn't push the AAPL single-name
        aggregate, only its own."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)  # $5k proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "MSFT",
                    "option_type": "put",
                    "strike": 200.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 100.0, "MSFT": 200.0},
            nav=100_000.0,  # AAPL single-name post-open: $5k = 5% < 10% → no R10
        )
        # IT-sector aggregate IS 25% post-open (20k held + 5k = 25k);
        # the boundary is strict > (see F6.1) so R9 does NOT fire at
        # exactly 25%. R10 also does not fire. Verdict stays proceed.
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R10" in n for n in notes)


# ======================================================================
# Family 3 — Downgrade-only invariant
# ======================================================================
class TestS42F3_DowngradeOnlyInvariant:
    """Neither R9 nor R10 can flip a verdict from a non-tradeable
    state to ``proceed``. Equally, when R7 or R8 already fired (and
    returned), R9 and R10 do not run at all — their notes must be
    absent from ``review_notes``."""

    def _negative_ev_with_sector_breach_setup(self, ev_dollars: float) -> CandidateDossier:
        """A candidate that would otherwise trip R9 (sector breach)
        but with a controllable ``ev_dollars`` to drive R1."""
        d = CandidateDossier(
            ticker="AAPL",
            ev_row={
                "ticker": "AAPL",
                "strike": 180.0,
                "premium": 2.0,
                "ev_dollars": ev_dollars,
                "iv": 0.25,
                "dte": 30,
                "spot": 180.0,
                "contracts": 1,
            },
            chart_context=_clean_chart("AAPL", 180.0),
            portfolio_context=PortfolioContext(
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
                nav=50_000.0,  # would trip R9 at 72% IT exposure
            ),
        )
        return d

    def test_r9_cannot_rescue_blocked_via_negative_ev_when_breach_present(self) -> None:
        """The candidate would trip R9 (sector breach), but R1 fires
        first because ``ev_dollars < 0``. Verdict stays ``blocked``."""
        d = self._negative_ev_with_sector_breach_setup(ev_dollars=-25.0)
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "blocked"
        assert reason == "negative_ev"
        # R9 never ran — no R9 note.
        assert not any("R9" in n for n in notes)

    def test_r9_cannot_rescue_blocked_via_non_finite_ev(self) -> None:
        """+inf EV → R1a fires (``ev_non_finite``), R9 never runs."""
        d = self._negative_ev_with_sector_breach_setup(ev_dollars=float("inf"))
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "blocked"
        assert reason == "ev_non_finite"
        assert not any("R9" in n for n in notes)

    def test_r10_cannot_rescue_blocked_via_non_finite_ev(self) -> None:
        """NaN EV → R1a fires before R10. Even with a held-AAPL +
        proposed-AAPL setup that would trip R10 at any positive EV,
        a NaN EV short-circuits to blocked."""
        d = CandidateDossier(
            ticker="AAPL",
            ev_row={
                "ticker": "AAPL",
                "strike": 50.0,
                "premium": 0.5,
                "ev_dollars": float("nan"),
                "iv": 0.25,
                "dte": 30,
                "spot": 100.0,
                "contracts": 1,
            },
            chart_context=_clean_chart("AAPL", 100.0),
            portfolio_context=PortfolioContext(
                held_option_positions=[
                    {
                        "symbol": "AAPL",
                        "option_type": "put",
                        "strike": 90.0,
                        "dte": 30,
                        "iv": 0.25,
                        "contracts": 1,
                        "is_short": True,
                    }
                ],
                spot_prices={"AAPL": 100.0},
                nav=100_000.0,  # would trip R10: 9k + 5k = 14% > 10%
            ),
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "blocked"
        assert reason == "ev_non_finite"
        assert not any("R10" in n for n in notes)

    def test_r9_does_not_run_after_r7_short_circuits(self) -> None:
        """R7 (VaR breach) fires first because the dossier code path
        evaluates R7 before R9. R9 never runs — confirm by absence of
        R9 note in ``review_notes``."""
        idx = pd.date_range("2026-01-01", periods=120, freq="B")
        # Heavy-vol synthetic returns + tiny NAV → R7 fires.
        returns = pd.DataFrame(
            {"portfolio": np.random.default_rng(7).normal(0, 0.08, 120)},
            index=idx,
        )
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000.0,  # tiny NAV → R7 VaR will breach 5%
            spot_prices={"AAPL": 180.0},
            returns_data=returns,
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
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "portfolio_var_breach"
        # R7 returned before R9 had a chance to run.
        assert not any("R9" in n for n in notes)
        assert not any("R10" in n for n in notes)

    def test_r10_does_not_run_after_r7_short_circuits(self) -> None:
        """Variant of the prior test focused on R10 rather than R9 —
        same short-circuit behaviour applies."""
        idx = pd.date_range("2026-01-01", periods=120, freq="B")
        returns = pd.DataFrame(
            {"portfolio": np.random.default_rng(7).normal(0, 0.08, 120)},
            index=idx,
        )
        # Use BKNG (Consumer Discretionary) + $5k proposed + $9k held
        # would trip R10 at 14% > 10% — but R7 fires first.
        d = _proceeding_dossier(ticker="BKNG", strike=50.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000.0,
            spot_prices={"BKNG": 100.0},
            returns_data=returns,
            held_option_positions=[
                {
                    "symbol": "BKNG",
                    "option_type": "put",
                    "strike": 90.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "portfolio_var_breach"
        assert not any("R10" in n for n in notes)

    def test_r9_r10_do_not_run_after_r8_short_circuits(self) -> None:
        """R8 (dealer regime short_gamma_amplifying) fires before R9
        and R10. Neither R9 nor R10 note appears."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000_000.0,  # huge NAV → stress passes
            spot_prices={"AAPL": 180.0},
            dealer_regime_by_ticker={"AAPL": "short_gamma_amplifying"},
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
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "short_gamma_regime"
        assert not any("R9" in n for n in notes)
        assert not any("R10" in n for n in notes)


# ======================================================================
# Family 4 — Fail-closed on missing context
# ======================================================================
class TestS42F4_FailClosedOnMissingContext:
    """Soft-warns must not fire on absent evidence (D11 / D17 Q3 +
    Q3-equivalent convention). Without a ``PortfolioContext``, both
    R9 and R10 silently skip. With a default-constructed context
    (nav=0, no holdings) the same skip semantics apply. Malformed
    held-position dicts (missing or wrong-typed fields) must not
    raise — the gate continues on bad rows."""

    def test_r9_and_r10_silent_with_no_portfolio_context(self) -> None:
        """``dossier.portfolio_context = None`` (the default) → both
        gates skipped silently. R7/R8 also dormant — verdict preserves
        R5's proceed."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        # portfolio_context is None by default (dataclass default).
        assert d.portfolio_context is None
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R7" in n or "R8" in n or "R9" in n or "R10" in n for n in notes)

    def test_r9_and_r10_silent_with_empty_default_portfolio_context(self) -> None:
        """``PortfolioContext()`` with default fields: nav=0, no
        holdings. Dossier guard ``if nav > 0 and proposed_notional > 0``
        keeps both gates dormant."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext()
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R9" in n or "R10" in n for n in notes)

    def test_r9_skips_held_position_missing_strike_gracefully(self) -> None:
        """A held position dict with ``strike`` missing is now
        silently skipped by ``SectorExposureManager.calculate_sector_exposures``
        — same defensive ``.get()`` pattern that
        ``check_single_name_cap`` already used. S42 Finding #1
        closed by this PR.

        Updated semantics: verdict stays at the R5 proceed result;
        no R9 note because the bad row contributed zero to the
        sector aggregate, leaving only the candidate's own notional
        ($18k / $50k = 36% > 25%) — wait, 36% trips R9 by itself.
        So R9 fires on the candidate-only aggregate. Refined
        assertion: verdict is review with sector_cap_breach, and
        R9 note mentions the candidate's sector (Information
        Technology) — NO crash."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    # 'strike' missing — gracefully skipped post-fix
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 180.0},
            nav=50_000.0,
        )
        # Without the malformed held row contributing, the candidate
        # alone is $18k / $50k = 36% of NAV — still > 25% R9 cap.
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"
        r9_notes = [n for n in notes if "R9" in n]
        assert len(r9_notes) == 1
        assert "Information Technology" in r9_notes[0]

    def test_r10_defensive_handling_now_reachable_after_filter_fix(self) -> None:
        """Once R8 stress no longer crashes on malformed held rows
        (S42 Findings #2 / #4 closed by the
        ``_filter_bsm_safe_positions`` filter in
        ``portfolio_risk_gates.py``), R10's pre-existing defensive
        try/except is reachable via the dossier path.

        This test uses an unknown-sector ticker (so R9 is the
        Unknown bucket) with a held position missing ``strike``.
        Post-fix path: R7 skips (no returns), R8 stress filters the
        malformed row out (no crash), R9 silently skips the same
        row (also filtered by Finding #1 fix), R10's defensive
        try/except in ``check_single_name_cap`` matches — but the
        row is already filtered upstream by R9 / R8, so R10 sees a
        clean held set + candidate. Net: no crash; verdict drops
        through to R10's pass branch."""
        d = _proceeding_dossier(ticker="XYZA", strike=200.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "XYZA",
                    # 'strike' missing — gracefully skipped post-fix
                    "option_type": "put",
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"XYZA": 200.0},
            nav=50_000.0,
        )
        # Candidate-only contribution: $20k / $50k = 40%. Unknown
        # sector bucket fires R9 at 40% > 25%.
        verdict, reason, _ = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"


# ======================================================================
# Family 5 — Cross-rule interaction (highest-value coverage)
# ======================================================================
class TestS42F5_CrossRuleInteraction:
    """When two or more soft-warns would fire on the same candidate,
    ``EnginePhaseReviewer.review`` evaluates them in code order
    (R7 → R8(stress) → R8(dealer) → R9 → R10) and short-circuits on
    the first firing rule via ``return``. The verdict_reason reflects
    that first firing rule; later rules' notes are absent from
    ``review_notes``.

    These tests pin the ordering — a future refactor that reorders
    the rules (or removes the short-circuit) must trip this audit.
    """

    def _heavy_vol_returns(self) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01", periods=120, freq="B")
        return pd.DataFrame(
            {"portfolio": np.random.default_rng(7).normal(0, 0.08, 120)},
            index=idx,
        )

    def test_r7_short_circuits_before_r9_when_both_would_fire(self) -> None:
        """Heavy-vol returns + small NAV + sector-concentrated holding
        → both R7 (VaR > 5%) and R9 (IT sector > 25%) would fire.
        R7 returns first."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000.0,  # tiny NAV → VaR breach + sector breach simultaneously
            spot_prices={"AAPL": 180.0},
            returns_data=self._heavy_vol_returns(),
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
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "portfolio_var_breach"
        assert any("R7" in n for n in notes)
        assert not any("R9" in n for n in notes)

    def test_r8_stress_short_circuits_before_r9_when_both_would_fire(self) -> None:
        """No VaR returns_data (R7 missing-data skip) + heavy
        concentrated short put on small NAV → R8 stress fires before
        R9. Sector-cap would have also fired but R8 wins."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=5_000.0,  # tiny → C4 stress -10%/+30% IV exceeds 8% drawdown
            spot_prices={"AAPL": 180.0},
            # No returns_data → R7 skips via missing_data
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
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        # R8 stress before R8 dealer before R9 / R10.
        assert reason == "stress_breach"
        assert any("R8 (stress)" in n for n in notes)
        assert not any("R9" in n for n in notes)
        assert not any("R10" in n for n in notes)

    def test_r8_dealer_short_circuits_before_r10_when_both_would_fire(self) -> None:
        """No returns_data, large enough NAV that stress passes, but
        ``dealer_regime_by_ticker`` flags the candidate as
        short_gamma_amplifying → R8 dealer fires. Single-name cap
        breach also present but R8 wins."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)  # $5k proposed
        d.portfolio_context = PortfolioContext(
            nav=100_000.0,
            spot_prices={"AAPL": 100.0},
            dealer_regime_by_ticker={"AAPL": "short_gamma_amplifying"},
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 90.0,  # $9k held → 14% > 10% → R10 would fire
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "short_gamma_regime"
        assert any("R8 (dealer)" in n for n in notes)
        assert not any("R10" in n for n in notes)

    def test_r9_short_circuits_before_r10_when_both_would_fire(self) -> None:
        """A position that breaches BOTH sector cap and single-name
        cap. R9 runs first (code order) and returns; R10 note is
        absent."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
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
            # Held = $18k AAPL; proposed = $18k AAPL.
            # Single-name: 36k / 100k = 36% > 10% → R10 would fire.
            # IT sector: 36k / 100k = 36% > 25% → R9 fires first.
            nav=100_000.0,
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "sector_cap_breach"
        assert any("R9" in n for n in notes)
        assert not any("R10" in n for n in notes)

    def test_all_four_soft_warns_trip_simultaneously_first_in_order_wins(self) -> None:
        """All four soft-warn RULES (R7, R8, R9, R10) have firing
        conditions met simultaneously, with R8 tripped on BOTH of its
        triggers (stress + dealer) — five total trigger conditions
        across four rules. ``EnginePhaseReviewer.review`` evaluates
        R7 first and short-circuits via ``return``. The verdict is
        ``review`` (not blocked, not proceed, not skip), the reason
        is the first firing rule's reason
        (``portfolio_var_breach``), and the ``review_notes`` contain
        only the R7 note — no R8 / R9 / R10 notes because those code
        paths never executed.

        This pins the rule ordering and the short-circuit contract.
        A future refactor that allowed multiple soft-warn notes to
        accumulate (e.g. a non-returning audit pass) would
        deliberately trip this test.
        """
        d = _proceeding_dossier(ticker="AAPL", strike=180.0)
        d.portfolio_context = PortfolioContext(
            nav=10_000.0,  # tiny NAV makes everything breach
            spot_prices={"AAPL": 180.0},
            returns_data=self._heavy_vol_returns(),  # R7 fires
            dealer_regime_by_ticker={"AAPL": "short_gamma_amplifying"},  # R8 dealer
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
            # Tiny NAV + concentrated AAPL: R8 stress would fire, R9
            # would fire (Info Tech > 25%), R10 would fire (single-
            # name > 10%). All four rules' conditions are met, with
            # R8 tripped on both triggers (5 total triggers).
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "review"
        assert reason == "portfolio_var_breach"
        # Only R7 ran; subsequent rules left no note.
        assert any("R7" in n for n in notes)
        assert not any("R8" in n for n in notes)
        assert not any("R9" in n for n in notes)
        assert not any("R10" in n for n in notes)


# ======================================================================
# Family 6 — Edge cases
# ======================================================================
class TestS42F6_EdgeCases:
    """Boundary semantics + odd inputs. The strict-``>`` cap behaviour
    is the highest-leverage pin here: a future change to ``>=`` would
    silently tighten every sector and single-name check by one
    boundary case."""

    def test_r9_passes_at_exact_25pct_sector_boundary(self) -> None:
        """``SectorExposureManager.check_sector_limit`` uses
        ``new_pct > max_sector_pct`` (strict). Exactly 25% IT
        exposure post-open → R9 PASSES (does not fire).

        Setup: AAPL $12.5k held + AAPL $12.5k proposed at $100k NAV
        = exactly $25k = 25.0% → strict > 25.0% is False → no fire.
        """
        d = _proceeding_dossier(ticker="AAPL", strike=125.0)  # $12,500 proposed
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": 125.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 125.0},
            nav=100_000.0,  # 2 * 12,500 / 100,000 = 25.000% exactly
        )
        # At exactly 25% R9 passes by strict-> semantics; R10 (single-
        # name) would be at 25% > 10% though, so R10 fires.
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        # R9 did NOT fire (exact 25% boundary passes). R10 DID fire
        # (25% > 10%). This confirms R9's strict-> boundary at 25%.
        assert reason == "single_name_breach"
        assert not any("R9" in n for n in notes)
        assert any("R10" in n for n in notes)

    def test_r10_passes_at_exact_10pct_single_name_boundary(self) -> None:
        """``check_single_name_cap`` uses ``post_open_pct >
        max_single_name_pct`` (strict). Exactly 10% single-name →
        R10 PASSES.

        Setup: BKNG $5k held + BKNG $5k proposed at $100k NAV =
        exactly 10.0%. BKNG is Consumer Discretionary so the IT
        sector cap isn't relevant; sector exposure is also 10.0% <
        25% R9 limit.
        """
        d = _proceeding_dossier(ticker="BKNG", strike=50.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "BKNG",
                    "option_type": "put",
                    "strike": 50.0,
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"BKNG": 100.0},
            nav=100_000.0,  # 2 * 5,000 / 100,000 = 10.000% exactly
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        # Exact 10% — strict > 10% is False — R10 does NOT fire.
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R10" in n for n in notes)

    def test_r8_stress_filters_candidate_strike_zero_gracefully(self) -> None:
        """A degenerate candidate (strike=0) used to crash R8 stress
        at the BSM pricing call before R9/R10 had a chance to skip
        it via their ``proposed_notional > 0`` guard. S42 Finding #2
        closed by the ``_filter_bsm_safe_positions`` filter in
        ``portfolio_risk_gates.py`` — R8 now silently drops the
        malformed candidate row before BSM sees it. R9/R10 then run
        normally; their existing guard catches proposed_notional=0
        and skips silently.

        Updated semantics: no crash; verdict stays at R5's proceed."""
        d = _proceeding_dossier(ticker="AAPL", strike=0.0, contracts=1)
        d.ev_row["spot"] = 0.0  # match chart spot so R3 doesn't fire
        d.portfolio_context = PortfolioContext(
            held_option_positions=[],
            spot_prices={"AAPL": 180.0},
            nav=10_000.0,
        )
        # No crash; soft-warns don't fire. R7 may leave a "VaR check
        # skipped (missing_data)" note because no returns_data was
        # supplied — that's a benign trace, not a firing.
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R9" in n or "R10" in n for n in notes)
        # The R8 stress + R8 dealer rules ran but passed (R8 stress
        # filtered the malformed candidate out; R8 dealer skipped on
        # missing dealer_regime_by_ticker).
        # No R8 firing note should appear.
        assert not any("R8 (stress):" in n or "R8 (dealer):" in n for n in notes)

    def test_r9_r10_path_honors_explicit_zero_contracts_after_or_fix(self) -> None:
        """S42 Finding #3 closed. The R9 and R10 paths in
        ``candidate_dossier.py`` no longer coerce an explicit
        ``contracts=0`` to 1 — the ``or 1`` truthy fallback has been
        removed so ``int(ev_row.get("contracts", 1))`` returns 0 when
        the key is explicitly 0, the missing-key default of 1
        remains intact (handled by the ``.get(..., 1)`` second
        argument).

        Updated semantics: with ``contracts=0``, proposed_notional=0
        triggers the existing ``if nav > 0 and proposed_notional > 0``
        guard around R9 and R10 — both gates skip silently. Verdict
        stays at the R5 proceed result. The held $18k AAPL alone is
        $18k / $100k = 18%, well below the 25% R9 sector cap and the
        10% R10 single-name cap when the candidate contributes 0.

        Trip condition: if a future refactor re-introduces the
        truthy coercion (or replaces it with another silent fallback),
        this test fires."""
        d = _proceeding_dossier(ticker="AAPL", strike=180.0, contracts=0)
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
            nav=100_000.0,
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        # contracts=0 honoured → proposed_notional=0 → R9/R10 skip.
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R9" in n or "R10" in n for n in notes)

    def test_r8_stress_filters_held_position_negative_strike_gracefully(self) -> None:
        """S42 Finding #4 closed. A held position with a negative
        strike (data corruption) used to crash R8 stress at the BSM
        pricing call. The ``_filter_bsm_safe_positions`` filter in
        ``portfolio_risk_gates.py`` now drops the malformed held row
        before BSM sees it. R9 and R10's defensive try/except in
        ``check_single_name_cap`` is also structurally reachable
        now (the cascading R8 crash no longer pre-empts R10).

        Updated semantics: held row dropped from the stress pool
        but the candidate is valid; R8 stress runs on a single-
        position pool (the candidate-only). At $5k proposed on $100k
        NAV the C4 stress drawdown stays below 8% so R8 stress
        passes. R9 sector exposure: held row contributes 0 (filtered
        from sector aggregation by Finding #1 fix), candidate
        contributes $5k = 5% < 25% — R9 passes. R10 single-name:
        held also filtered → 5% < 10% — R10 passes. Verdict
        proceeds."""
        d = _proceeding_dossier(ticker="AAPL", strike=50.0)
        d.portfolio_context = PortfolioContext(
            held_option_positions=[
                {
                    "symbol": "AAPL",
                    "option_type": "put",
                    "strike": -100.0,  # corruption — filtered post-fix
                    "dte": 30,
                    "iv": 0.25,
                    "contracts": 1,
                    "is_short": True,
                }
            ],
            spot_prices={"AAPL": 100.0},
            nav=100_000.0,
        )
        verdict, reason, notes = EnginePhaseReviewer().review(d)
        assert verdict == "proceed"
        assert reason == "ev_above_threshold"
        assert not any("R9" in n or "R10" in n for n in notes)


# ======================================================================
# Pin: the dossier reviewer's MIN_PROCEED_EV_DOLLARS hasn't drifted
# ======================================================================
def test_min_proceed_ev_dollars_default_pinned() -> None:
    """``MIN_PROCEED_EV_DOLLARS = 10.0`` is the canonical R5 threshold
    that this whole audit's `ev_dollars=50` setup assumes. If that
    constant drifts, the proceeding-dossier helper might silently
    fall below threshold and the audit would test R5's review path
    instead of the R7-R10 path. Pin it here so the audit trips on
    the upstream change."""
    assert MIN_PROCEED_EV_DOLLARS == 10.0


# ======================================================================
# Math invariants used in fixture construction
# ======================================================================
def test_fixture_math_for_r9_boundary_is_exact() -> None:
    """The F6.1 ``test_r9_passes_at_exact_25pct_sector_boundary``
    fixture relies on ``2 * 12500 / 100000 == 0.25`` being EXACT in
    float arithmetic. The same exact-rational property holds for
    the R10 boundary at 10% with ``2 * 5000 / 100000``. Verify both
    so a future fixture refactor that introduces a non-exact
    representation (e.g. 25% via ``75000 / 300000``) doesn't quietly
    break the boundary tests."""
    assert math.isclose(2 * 12500 / 100_000, 0.25, abs_tol=0.0)
    assert math.isclose(2 * 5000 / 100_000, 0.10, abs_tol=0.0)
