"""
Candidate dossier builder — combines EV output with chart context.

Role in the workflow
--------------------
This is **Mode B** from the TradingView strategic review:

    1. Engine ranks candidates first
       (``WheelRunner.rank_candidates_by_ev``)
    2. For the top N we go to TradingView via a
       :class:`~engine.chart_context.ChartContextProvider`
       and attach a screenshot to each candidate.
    3. A :class:`ChartReviewer` looks at the combined (EV + chart)
       package and produces a structured verdict:

         - ``proceed``   — engine and chart agree
         - ``review``    — engine is positive but chart is ambiguous,
                           human should take a second look
         - ``skip``      — chart strongly contradicts the engine
                           (e.g. fresh support violation)
         - ``blocked``   — EV was already negative; chart cannot upgrade

       **The reviewer can only downgrade a candidate, never upgrade.**
       This enforces the hard guardrail from the strategic review:
       charts never rescue a negative-EV trade.

The dossier is the final artifact the dashboard / trade ticket UI
consumes. It carries every structured number from the EV engine plus
a (possibly missing) screenshot path and a review verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import pandas as pd

from .chart_context import ChartContext, ChartContextProvider, Timeframe

Verdict = Literal["proceed", "review", "skip", "blocked"]

# Canonical proceed-threshold (in dollars) for EV-driven verdicts.
# Lives here so :class:`EnginePhaseReviewer`'s default and any external
# caller that mirrors the dossier-side ladder (e.g.
# ``engine_api.EngineAPIHandler._enrich_alert`` for ``/api/tv/webhook``,
# ``_handle_candidates`` for ``/api/candidates``) share the same number
# and cannot drift on threshold value. The two ladders are divergent
# **by design** in their overlay rules (chart agreement, ``prob_profit``
# floor) but the bare EV floor should be one number. Closes the
# threshold-drift half of C2 from ``docs/END_TO_END_REVIEW_2026_05_25.md``.
MIN_PROCEED_EV_DOLLARS: float = 10.0

# R11 (elevated-vol top-bin size-down) parameters — heavy-verify 2026-05-31 I11.
# The engine's high-confidence (top-bin) prob_profit is materially over-confident
# in crisis (I1: ~0.57 realized vs ~0.96 forecast) and that miss is neither
# forecastable (I9) nor cleanly detectable by a single transition signal (I10).
# I11's measured-robust response: size the top bin DOWN whenever VIX *level* is
# elevated. VIX>25 is the robust-not-optimal threshold — it survived leave-one-
# crisis-out in BOTH the 2020 crash and 2022 bear ({20,22.5,25} survive; >=27.5
# fails the 2022 fold). prob_profit>0.90 is the inherited top-bin cutoff (I1/I9/I10).
R11_TOP_BIN_PROB: float = 0.90
R11_VIX_THRESHOLD: float = 25.0


@dataclass
class CandidateDossier:
    """The combined EV + chart artifact the UI / trade ticket consumes."""

    ticker: str
    # Full EV row from rank_candidates_by_ev (strike, premium, ev_dollars,
    # cvar, prob_profit, etc.). We store the original dict so the
    # dashboard can render any column without the engine having to
    # pre-promote every field here.
    ev_row: dict[str, Any]
    chart_context: ChartContext | None = None
    # Optional aggregated dealer positioning (engine/dealer_positioning.py).
    # When present the EnginePhaseReviewer applies an extra rule (R6)
    # that can downgrade — never upgrade — a candidate based on the
    # market-structure regime relative to the trade strike.
    market_structure: Any = None
    # Optional portfolio-wide context for the D17 dossier soft-warns
    # (R7 = VaR; R8 = stress + dealer regime). When attached, the
    # reviewer can run check_var / check_stress_scenario /
    # check_dealer_regime against the held book and downgrade
    # proceed → review if a tail-risk gate fires. When absent, R7
    # and R8 do not fire — soft-warns should not fire on absent
    # evidence (Q3 of the #154 C4 design checkpoint). See
    # engine/portfolio_risk_gates.PortfolioContext.
    portfolio_context: Any = None
    # Optional market-wide VIX *level* on the candidate's as_of (NOT a ratio —
    # I10 showed ratios invert onset/recovery). When present the reviewer
    # applies R11: an elevated-vol size-down of the top bin (downgrade-only).
    # When absent (the default) R11 is a no-op — same missing-evidence semantics
    # as R6/R7. Typically threaded in by the ranker from the connector's
    # get_vix_regime(as_of). See heavy-verify 2026-05-31 I11.
    vix_level: float | None = None
    verdict: Verdict = "review"
    verdict_reason: str = ""
    review_notes: list[str] = field(default_factory=list)
    built_at: datetime = field(default_factory=lambda: datetime.now(UTC).replace(tzinfo=None))

    @property
    def ev_dollars(self) -> float:
        return float(self.ev_row.get("ev_dollars", 0.0) or 0.0)

    @property
    def has_chart(self) -> bool:
        return self.chart_context is not None and self.chart_context.is_ok()

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "ev_row": dict(self.ev_row),
            "chart_context": (self.chart_context.to_dict() if self.chart_context else None),
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "review_notes": list(self.review_notes),
            "has_chart": self.has_chart,
            "built_at": self.built_at.isoformat(),
        }


class ChartReviewer(Protocol):
    """Protocol for any implementation that scores a chart+EV pair.

    A chart reviewer is the only place in the system that is allowed
    to look at a screenshot and say "I don't like what I see".

    Implementations may be:

    * Rule-based — e.g. check the engine's regime banner vs chart
      phase for disagreement. This is the default :class:`EnginePhaseReviewer`.
    * LLM-powered — call Claude to describe what the chart shows and
      parse the description. Not implemented here to keep this module
      dependency-free, but the protocol is deliberately designed so
      you can drop a Claude-backed reviewer in later.
    """

    def review(self, dossier: CandidateDossier) -> tuple[Verdict, str, list[str]]: ...


# ----------------------------------------------------------------------
# Default rule-based reviewer
# ----------------------------------------------------------------------
class EnginePhaseReviewer:
    """Rule-based chart reviewer that compares engine regime to chart state.

    Rules (explicit and conservative):

    1. If ``ev_dollars`` is non-finite (``+inf`` / ``-inf`` / ``NaN``)
       OR strictly negative, the verdict is **blocked**. No chart
       review is allowed to override either case. This is the hard
       guardrail from the TradingView strategic review.

       The non-finite branch returns ``verdict_reason="ev_non_finite"``
       (distinct from ``"negative_ev"``) so the audit trail tells
       "engine produced an unparseable value — investigate the
       upstream computation" apart from "engine evaluated the trade
       as a loss". Closes C1 from
       ``docs/END_TO_END_REVIEW_2026_05_25.md``: without the
       non-finite block, ``+inf`` slid through both R1 (``+inf < 0``
       is False) and R5 (``+inf >= threshold`` is True) and was
       reported as ``"proceed"``; ``NaN`` silently degraded to
       ``"review"`` via R5's strict ``>=`` (``NaN >= threshold``
       is False), masking the real signal. No production code path
       injects non-finite ``ev_dollars`` today (S20 G3 confirms the
       network surface excludes ``ev_dollars`` from user-controllable
       fields), but the defense-in-depth gap is closed.

    2. If the chart context is missing or errored, the verdict is
       **review** — the trade can still go on, but a human should
       manually check before clicking. The ``verdict_reason`` says
       "chart_context_missing" so the UI can display the warning.

    3. If the chart context reports a ``visible_price`` that disagrees
       with the engine's spot by more than ``spot_tolerance_pct``, the
       verdict is **skip**. This catches stale screenshots that would
       otherwise pass through.

    4. *(Conditional — reserved.)* If the chart context's
       ``visible_indicators`` includes a ``phase`` field that disagrees
       with the engine's phase-based decision (e.g. chart says
       "compression", engine wants to sell), the verdict is **skip**.
       This rule is implemented and unit-tested
       (``tests/test_dossier_invariant.py``) but dormant in the
       production path: no current chart provider populates
       ``visible_indicators['phase']`` — it stays empty through M1 — and
       the ranker emits no ``phase`` field on ``ev_row``, so neither
       operand of the predicate is fed. R4 activates only when a
       phase-aware chart provider lands (see
       ``docs/TRADINGVIEW_INTEGRATION.md``). It is not a live downgrade
       today.

    5. Otherwise, if ``ev_dollars >= min_proceed_ev``, the verdict is
       **proceed**. Below that threshold it is **review** so the human
       opens the screenshot before firing.

    6. *(Conditional — audit V.)* If a `MarketStructure` is attached
       and the dealer regime is short-gamma amplifying with the
       candidate strike at or above the nearest put wall — OR the
       regime is near gamma flip — the verdict is **review**. Like
       R4, downgrade-only.

    7. *(Conditional — new in D17.)* If a `PortfolioContext` is
       attached and ``check_var`` reports portfolio VaR_95 (30-day
       horizon) above ``max_var_pct × NAV`` (default 5%), the
       verdict is **review** with ``verdict_reason="portfolio_var_breach"``.
       When the context is absent OR ``check_var`` skips for missing
       correlation/returns data, R7 does **not** fire — soft-warns
       don't fire on absent evidence (Q3 of the #154 C4 design
       checkpoint; matches D11's "no silent substitution" principle).

    8. *(Conditional — new in D17.)* One R8, two trigger conditions
       (mirrors R6). Either the C4 vol-spike scenario shows
       portfolio drawdown > 8% NAV (``check_stress_scenario`` fails)
       OR the candidate's underlying is in ``short_gamma_amplifying``
       regime (``check_dealer_regime`` fails). Distinct
       ``verdict_reason`` per trigger
       (``"stress_breach"`` / ``"short_gamma_regime"``) so the audit
       trail records which one. Like R6/R7, downgrade-only — never
       rescues a negative-EV trade (R1 already does that).

    9. *(Conditional — D17 B2 closure.)* If a `PortfolioContext` is
       attached and opening the candidate would push its GICS sector
       over ``max_sector_pct × NAV`` (default 25% per the D17 sector
       cap; see :func:`engine.portfolio_risk_gates.check_sector_cap`),
       the verdict is **review** with
       ``verdict_reason="sector_cap_breach"``. Soft-warn preview of
       the same gate the tracker applies as a HARD refusal at
       ``open_short_put`` time when ``require_ev_authority=True``
       (see :class:`engine.wheel_tracker.WheelTracker`). Like R7/R8,
       downgrade-only and only fires when context is attached —
       absent context keeps R9 a no-op (Q3 missing-data semantics).

    10. *(Conditional — F4 damage-bounding addition.)* If a
        `PortfolioContext` is attached and opening the candidate
        would push the SINGLE-NAME (per-underlying) short-option
        notional over ``max_single_name_pct × NAV`` (default 10% per
        :func:`engine.portfolio_risk_gates.check_single_name_cap`),
        the verdict is **review** with
        ``verdict_reason="single_name_breach"``. Sits BENEATH the
        sector cap (R9): a ticker concentrated as the only name in
        its sector could still pass R9 at 25% NAV; R10 catches it
        at 10% NAV first. Bounds idiosyncratic single-name drawdown
        damage (the F4 case style — see
        ``docs/F4_TAIL_RISK_DIAGNOSTIC.md`` §10) that no market-wide
        regime detector can predict. Downgrade-only; no-op when
        ``nav == 0`` or context is absent.

    11. *(Conditional — heavy-verify 2026-05-31 I11.)* If a
        market-wide ``vix_level`` is attached and the candidate is a
        high-confidence top-bin pick (``prob_profit > R11_TOP_BIN_PROB``,
        0.90) while ``vix_level > R11_VIX_THRESHOLD`` (25.0), the verdict
        is **review** with ``verdict_reason="elevated_vol_top_bin"``.
        Rationale: I1 found the top ``prob_profit`` bin is materially
        over-confident in the regime that *follows* an elevated-vol
        reading (~0.57 realized vs ~0.96 forecast in crisis), a miss
        that is neither forecastable (I9) nor cleanly detectable from a
        single onset signal (I10). I11 showed SIZING DOWN is favorably
        asymmetric in every well-powered crisis fold and the VIX>25 cut
        survives leave-one-crisis-out (θ≥27.5 fails the 2022 fold; 25 is
        the robust-not-optimal floor). Counterpart to R10: R10 bounds
        idiosyncratic single-name size, R11 bounds market-wide vol
        exposure on the over-confident top bin. Downgrade-only — never
        rescues a negative-EV trade (R1) and never upgrades; no-op when
        ``vix_level`` is absent (missing-evidence semantics, like
        R6–R10). The warning payload carries the candidate's OWN modeled
        tail (``cvar_5`` from ``ev_row``) — computed/regime-matched, not
        a hardcoded constant. See ``docs/HEAVY_VERIFY_2026-05-31_I11.md``.

    Notes:
      * The reviewer is pure — no I/O, no network, no LLM. It only
        consumes the already-captured dossier (plus the optional
        attached ``market_structure`` and ``portfolio_context``).
      * All decisions are logged as a list of review_notes strings so
        the audit trail can reconstruct exactly why the verdict
        landed where it did.
    """

    def __init__(
        self,
        min_proceed_ev: float = MIN_PROCEED_EV_DOLLARS,
        spot_tolerance_pct: float = 0.02,
    ) -> None:
        self.min_proceed_ev = min_proceed_ev
        self.spot_tolerance_pct = spot_tolerance_pct

    def review(self, dossier: CandidateDossier) -> tuple[Verdict, str, list[str]]:
        notes: list[str] = []
        ev = dossier.ev_dollars

        # Rule 1a: non-finite EV is blocked. +inf would otherwise slip
        # through both R1 (False: +inf < 0) and R5 (True: +inf >=
        # threshold) and be reported as "proceed"; NaN would silently
        # degrade to "review" via R5's strict >=. Distinct
        # verdict_reason ("ev_non_finite") so the audit trail tells
        # an unparseable engine value apart from an evaluated loss.
        # No production path injects non-finite ev_dollars today
        # (S20 G3 confirms; engine math is bounded by finite inputs)
        # but the defense-in-depth gap from RELIABILITY_ARC_REVIEW C1
        # is closed here. Chart cannot upgrade.
        if not math.isfinite(ev):
            notes.append(f"engine ev_dollars={ev!r} not finite - hard block (non-finite EV)")
            return "blocked", "ev_non_finite", notes

        # Rule 1: negative EV is blocked. Chart cannot save it.
        if ev < 0:
            notes.append(f"engine ev_dollars={ev:.2f} < 0 - chart cannot upgrade negative EV")
            return "blocked", "negative_ev", notes

        chart = dossier.chart_context
        if chart is None or not chart.is_ok():
            err = chart.error if chart is not None else "no_chart_provider"
            notes.append(f"chart context unavailable: {err}")
            return "review", "chart_context_missing", notes
        notes.append(f"chart captured from {chart.source} at {chart.captured_at}")

        # Rule 3: spot-vs-screenshot price disagreement.
        engine_spot = float(dossier.ev_row.get("spot", 0.0) or 0.0)
        if chart.visible_price is not None and engine_spot > 0:
            tol = engine_spot * self.spot_tolerance_pct
            diff = abs(chart.visible_price - engine_spot)
            if diff > tol:
                notes.append(
                    f"visible chart price {chart.visible_price:.2f} "
                    f"disagrees with engine spot {engine_spot:.2f} (|delta|={diff:.2f} > tol {tol:.2f})"
                )
                return "skip", "spot_price_mismatch", notes
            notes.append(
                f"chart price {chart.visible_price:.2f} agrees with engine spot {engine_spot:.2f}"
            )

        # Rule 4: phase disagreement between chart and engine.
        chart_phase = chart.visible_indicators.get("phase") if chart.visible_indicators else None
        engine_phase = dossier.ev_row.get("phase")
        if chart_phase and engine_phase and str(chart_phase) != str(engine_phase):
            # Specifically disagreement between post_expansion (engine
            # loves) and compression/expansion (bad).
            bad_phases = {"compression", "expansion"}
            if str(chart_phase) in bad_phases:
                notes.append(f"chart phase={chart_phase} contradicts engine phase={engine_phase}")
                return "skip", "phase_contradiction", notes
            notes.append(f"phase disagreement logged: chart={chart_phase} engine={engine_phase}")

        # Rule 5: EV threshold.
        if ev >= self.min_proceed_ev:
            verdict: Verdict = "proceed"
            reason = "ev_above_threshold"
            notes.append(f"ev_dollars={ev:.2f} >= {self.min_proceed_ev} - proceed")
        else:
            verdict = "review"
            reason = "ev_below_proceed_threshold"
            notes.append(f"ev_dollars={ev:.2f} < min_proceed {self.min_proceed_ev} - human review")

        # Rule 6: Dealer-positioning downgrade (audit V).
        # When an aggregated MarketStructure is attached AND the regime
        # is short-gamma amplifying AND the candidate strike is at or
        # above the nearest put wall, breach risk is materially higher
        # than the raw EV suggests. Downgrade proceed → review.
        # Hard guardrail: this rule NEVER upgrades. It can only shift
        # "proceed" to "review" (never touches "blocked" or "skip").
        ms = getattr(dossier, "market_structure", None)
        if ms is not None and verdict == "proceed":
            regime = getattr(ms, "regime", "")
            if regime == "short_gamma_amplifying":
                nearest_put = getattr(ms, "nearest_put_wall", None)
                strike = dossier.ev_row.get("strike")
                try:
                    strike_f = float(strike) if strike is not None else None
                except (TypeError, ValueError):
                    strike_f = None
                if (
                    nearest_put is not None
                    and strike_f is not None
                    and strike_f >= float(nearest_put.strike)
                ):
                    notes.append(
                        f"R6: short-gamma regime + strike {strike_f:.2f} "
                        f"at/above put wall {float(nearest_put.strike):.2f} "
                        "- breach risk amplified, downgrade to review"
                    )
                    return "review", "dealer_short_gamma_above_put_wall", notes
            elif regime == "near_flip":
                notes.append("R6: dealer regime near gamma flip - downgrade to review")
                return "review", "dealer_near_flip", notes

        # Rule 7: portfolio-level VaR (D17 soft-warn). Fires only if
        # the candidate currently has verdict == "proceed"; downgrade-
        # only, never upgrades. When no portfolio context is attached
        # OR check_var skips for missing data, R7 doesn't fire
        # (soft-warns don't fire on absent evidence — Q3).
        ctx = getattr(dossier, "portfolio_context", None)
        if ctx is not None and verdict == "proceed":
            from .portfolio_risk_gates import check_var

            candidate_dict = self._build_candidate_dict(dossier)
            var_result = check_var(
                held_option_positions=getattr(ctx, "held_option_positions", []),
                spot_prices=getattr(ctx, "spot_prices", {}),
                candidate_option=candidate_dict,
                nav=float(getattr(ctx, "nav", 0.0) or 0.0),
                returns_data=getattr(ctx, "returns_data", None),
                correlation_matrix=getattr(ctx, "correlation_matrix", None),
                volatilities=getattr(ctx, "volatilities", None),
            )
            if not var_result.passed:
                var_pct = var_result.details.get("var_pct", 0.0)
                limit_pct = var_result.details.get("var_limit_pct", 0.0)
                notes.append(
                    f"R7: portfolio VaR_95 {var_pct:.1%} exceeds {limit_pct:.1%} NAV "
                    "- downgrade to review"
                )
                return "review", "portfolio_var_breach", notes
            if var_result.reason == "missing_data":
                skip_reason = var_result.details.get("skip_reason", "missing_data")
                notes.append(f"R7: VaR check skipped ({skip_reason})")

        # Rule 8: portfolio stress + dealer regime (D17 soft-warn).
        # ONE rule with TWO trigger conditions per the Q1 design
        # decision: either the C4 vol-spike stress drawdown exceeds
        # the 8% NAV threshold OR the candidate's underlying is in
        # short_gamma_amplifying regime. Mirrors R6's "short-gamma +
        # put-wall OR dealer-flip" two-trigger pattern. Distinct
        # verdict_reason per trigger so the audit trail records
        # which one fired. Downgrade-only; never upgrades.
        if ctx is not None and verdict == "proceed":
            from .portfolio_risk_gates import check_dealer_regime, check_stress_scenario

            candidate_dict = self._build_candidate_dict(dossier)

            stress_result = check_stress_scenario(
                held_option_positions=getattr(ctx, "held_option_positions", []),
                spot_prices=getattr(ctx, "spot_prices", {}),
                candidate_option=candidate_dict,
                nav=float(getattr(ctx, "nav", 0.0) or 0.0),
            )
            if not stress_result.passed:
                drawdown = stress_result.details.get("drawdown_pct", 0.0)
                limit = stress_result.details.get("drawdown_limit_pct", 0.0)
                scenario = stress_result.details.get("scenario_name", "stress")
                notes.append(
                    f"R8 (stress): {scenario} drawdown {drawdown:.1%} exceeds "
                    f"{limit:.1%} NAV - downgrade to review"
                )
                return "review", "stress_breach", notes

            regime_result = check_dealer_regime(
                candidate_ticker=dossier.ticker,
                dealer_regime_by_ticker=getattr(ctx, "dealer_regime_by_ticker", None),
            )
            if not regime_result.passed:
                notes.append(
                    f"R8 (dealer): {dossier.ticker} in short_gamma_amplifying regime "
                    "- downgrade to review"
                )
                return "review", "short_gamma_regime", notes

        # Rule 9: sector cap (D17 soft-warn / B2 closure). Soft-warn
        # preview of the same gate the tracker applies as a HARD
        # refusal at open_short_put time when
        # require_ev_authority=True. Downgrade-only.
        if ctx is not None and verdict == "proceed":
            from .portfolio_risk_gates import check_sector_cap

            ev_row = dossier.ev_row
            # S42 Finding #3: drop the `or 1` truthy fallback on
            # contracts so an explicit ``contracts=0`` produces
            # proposed_notional=0 (and is caught by the guard below)
            # rather than being silently coerced to 1 contract.
            try:
                strike = float(ev_row.get("strike", 0) or 0)
                contracts = int(ev_row.get("contracts", 1))
            except (TypeError, ValueError):
                strike = 0.0
                contracts = 1
            proposed_notional = strike * 100.0 * contracts

            nav = float(getattr(ctx, "nav", 0.0) or 0.0)
            if nav > 0 and proposed_notional > 0:
                # #372: aggregate by the real GICS sector. ``ctx.sector_map``
                # (built from the connector by the tracker) covers held names;
                # merge the candidate's own GICS from its ranker row so the
                # gate buckets it correctly even when it is not yet held.
                # None → ``DEFAULT_SECTOR_MAP`` fallback (legacy behaviour).
                gics_map = dict(getattr(ctx, "sector_map", None) or {})
                cand_sector = ev_row.get("sector")
                if cand_sector:
                    gics_map.setdefault(dossier.ticker, cand_sector)
                sector_result = check_sector_cap(
                    symbol=dossier.ticker,
                    proposed_notional=proposed_notional,
                    held_option_positions=getattr(ctx, "held_option_positions", []),
                    nav=nav,
                    sector_map=gics_map or None,
                )
                if not sector_result.passed:
                    sector = sector_result.details.get("sector", "Unknown")
                    post_pct = sector_result.details.get("post_open_sector_pct", 0.0)
                    limit_pct = sector_result.details.get("sector_limit", 0.0)
                    notes.append(
                        f"R9: {sector} sector exposure would be {post_pct:.1%} "
                        f"(limit {limit_pct:.1%} NAV) — downgrade to review"
                    )
                    return "review", "sector_cap_breach", notes

        # Rule 10: single-name (per-underlying) exposure cap. Sits
        # BENEATH R9: even when sector cap is satisfied, a single
        # ticker concentrated as the dominant name in its sector
        # could still exceed the per-name floor. Bounds F4-style
        # idiosyncratic-drawdown damage. Downgrade-only.
        if ctx is not None and verdict == "proceed":
            from .portfolio_risk_gates import check_single_name_cap

            ev_row = dossier.ev_row
            # S42 Finding #3: same fix as R9 above — drop `or 1`.
            try:
                strike = float(ev_row.get("strike", 0) or 0)
                contracts = int(ev_row.get("contracts", 1))
            except (TypeError, ValueError):
                strike = 0.0
                contracts = 1
            proposed_notional = strike * 100.0 * contracts

            nav = float(getattr(ctx, "nav", 0.0) or 0.0)
            if nav > 0 and proposed_notional > 0:
                name_result = check_single_name_cap(
                    symbol=dossier.ticker,
                    proposed_notional=proposed_notional,
                    held_option_positions=getattr(ctx, "held_option_positions", []),
                    nav=nav,
                )
                if not name_result.passed:
                    post_pct = name_result.details.get("post_open_name_pct", 0.0)
                    limit_pct = name_result.details.get("name_limit_pct", 0.0)
                    notes.append(
                        f"R10: {dossier.ticker} single-name exposure would be "
                        f"{post_pct:.1%} (limit {limit_pct:.1%} NAV) — downgrade to review"
                    )
                    return "review", "single_name_breach", notes

        # Rule 11: elevated-vol top-bin size-down (heavy-verify 2026-05-31 I11).
        # When market-wide VIX *level* is elevated (> R11_VIX_THRESHOLD) AND this
        # is a high-confidence candidate (prob_profit > R11_TOP_BIN_PROB), the
        # engine's top-bin prob_profit is materially over-confident in the regime
        # that follows (I1: ~0.57 realized vs ~0.96 forecast in crisis) — a miss
        # that is neither forecastable (I9) nor cleanly detectable (I10). I11
        # showed the robust response is to SIZE DOWN regardless; the VIX>25 cut
        # survived leave-one-crisis-out (2020 +$86k / 2022 +$3.5k averted-vs-
        # forgone). Downgrade-only, never upgrades; no-op when vix_level is absent
        # (missing-evidence semantics, like R6-R10). The warning carries THIS
        # candidate's own modeled tail (cvar_5) — computed/regime-matched, not a
        # hardcoded constant, so it stays honest as data updates.
        vix_level = getattr(dossier, "vix_level", None)
        if vix_level is not None and verdict == "proceed":
            try:
                pp = float(dossier.ev_row.get("prob_profit", 0.0) or 0.0)
                vix_f = float(vix_level)
            except (TypeError, ValueError):
                pp, vix_f = 0.0, 0.0
            if pp > R11_TOP_BIN_PROB and vix_f > R11_VIX_THRESHOLD:
                cvar = dossier.ev_row.get("cvar_5")
                try:
                    cvar_s = f"${float(cvar):,.0f}" if cvar is not None else "n/a"
                except (TypeError, ValueError):
                    cvar_s = "n/a"
                notes.append(
                    f"R11: VIX={vix_f:.1f} > {R11_VIX_THRESHOLD} and prob_profit="
                    f"{pp:.3f} > {R11_TOP_BIN_PROB} — elevated-vol top bin. The crisis "
                    f"top bin historically realized ~0.57 vs the ~{pp:.0%} forecast "
                    f"(heavy-verify I1/I11); this candidate's modeled tail cvar_5="
                    f"{cvar_s}. Size down — downgrade to review."
                )
                return "review", "elevated_vol_top_bin", notes

        return verdict, reason, notes

    @staticmethod
    def _build_candidate_dict(dossier: CandidateDossier) -> dict:
        """Build the option position-dict shape upstream gate APIs
        expect, from a dossier's ev_row.

        Pure helper — no I/O, no state. Used by R7, R8, R9, and R10.
        """
        ev_row = dossier.ev_row
        # Default option_type to "put" because the wheel pipeline
        # ranks short puts; the column is rarely present on the row
        # itself.
        opt_type = str(ev_row.get("option_type", "put"))
        try:
            strike = float(ev_row.get("strike", 0) or 0)
        except (TypeError, ValueError):
            strike = 0.0
        try:
            dte = int(ev_row.get("dte", 0) or 0)
        except (TypeError, ValueError):
            dte = 0
        try:
            iv = float(ev_row.get("iv", 0) or 0)
        except (TypeError, ValueError):
            iv = 0.0
        return {
            "symbol": dossier.ticker,
            "option_type": opt_type,
            "strike": strike,
            "dte": dte,
            "iv": iv,
            "contracts": 1,
            "is_short": True,
        }


# ----------------------------------------------------------------------
# Dossier builder
# ----------------------------------------------------------------------
def build_dossiers(
    ev_frame: pd.DataFrame,
    provider: ChartContextProvider,
    reviewer: ChartReviewer | None = None,
    *,
    timeframe: Timeframe = "1D",
    top_n: int = 10,
    as_of: datetime | None = None,
    portfolio_context: Any = None,
    vix_level: float | None = None,
) -> list[CandidateDossier]:
    """Walk the ranked EV frame, attach a chart, run the reviewer.

    Args:
        ev_frame: Output of :meth:`WheelRunner.rank_candidates_by_ev`.
        provider: A :class:`ChartContextProvider` implementation.
        reviewer: Optional :class:`ChartReviewer`; defaults to
            :class:`EnginePhaseReviewer`.
        timeframe: Which TradingView timeframe to capture per candidate.
        top_n: Only attach charts to the top N candidates (charts are
            expensive; the long-tail candidates stay "ranked" but
            without visual context until the user clicks into them).
        as_of: PIT cutoff passed through to the provider for stale
            filesystem screenshots.
        portfolio_context: Optional
            :class:`engine.portfolio_risk_gates.PortfolioContext` to
            attach to every dossier built in this pass. When set, the
            reviewer's R7 (VaR) / R8 (stress + dealer regime)
            soft-warns fire live for each candidate; when ``None``
            (the default) the existing missing-data skip preserves
            today's behaviour. Typically constructed once per ranking
            pass via :meth:`engine.wheel_tracker.WheelTracker.portfolio_context_snapshot`.

    Returns:
        List of :class:`CandidateDossier`, one per (top-N) ticker.
    """
    reviewer = reviewer or EnginePhaseReviewer()
    dossiers: list[CandidateDossier] = []
    if ev_frame is None or len(ev_frame) == 0:
        return dossiers

    head = ev_frame.head(max(0, int(top_n)))

    for _, row in head.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue

        row_dict = row.to_dict()
        # Fetch chart context; a failed fetch is fine — dossier degrades.
        chart_ctx = provider.fetch(ticker, timeframe, as_of=as_of)
        dossier = CandidateDossier(
            ticker=ticker,
            ev_row=row_dict,
            chart_context=chart_ctx,
            portfolio_context=portfolio_context,
            vix_level=vix_level,
        )

        verdict, reason, notes = reviewer.review(dossier)
        dossier.verdict = verdict
        dossier.verdict_reason = reason
        dossier.review_notes = notes
        dossiers.append(dossier)

    return dossiers
