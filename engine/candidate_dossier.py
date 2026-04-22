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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

import pandas as pd

from .chart_context import ChartContext, ChartContextProvider, Timeframe


Verdict = Literal["proceed", "review", "skip", "blocked"]


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
    verdict: Verdict = "review"
    verdict_reason: str = ""
    review_notes: list[str] = field(default_factory=list)
    built_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

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
            "chart_context": (
                self.chart_context.to_dict() if self.chart_context else None
            ),
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

    def review(self, dossier: CandidateDossier) -> tuple[Verdict, str, list[str]]:
        ...


# ----------------------------------------------------------------------
# Default rule-based reviewer
# ----------------------------------------------------------------------
class EnginePhaseReviewer:
    """Rule-based chart reviewer that compares engine regime to chart state.

    Rules (explicit and conservative):

    1. If ``ev_dollars < 0`` the verdict is **blocked**. No chart review
       is allowed to override a negative-EV engine verdict. This is the
       hard guardrail from the TradingView strategic review.

    2. If the chart context is missing or errored, the verdict is
       **review** — the trade can still go on, but a human should
       manually check before clicking. The ``verdict_reason`` says
       "chart_context_missing" so the UI can display the warning.

    3. If the chart context reports a ``visible_price`` that disagrees
       with the engine's spot by more than ``spot_tolerance_pct``, the
       verdict is **skip**. This catches stale screenshots that would
       otherwise pass through.

    4. If the chart context's ``visible_indicators`` includes a
       ``phase`` field that disagrees with the engine's phase-based
       decision (e.g. chart says "compression", engine wants to sell),
       the verdict is **skip**.

    5. Otherwise, if ``ev_dollars >= min_proceed_ev``, the verdict is
       **proceed**. Below that threshold it is **review** so the human
       opens the screenshot before firing.

    Notes:
      * The reviewer is pure — no I/O, no network, no LLM. It only
        consumes the already-captured dossier.
      * All decisions are logged as a list of review_notes strings so
        the audit trail can reconstruct exactly why the verdict
        landed where it did.
    """

    def __init__(
        self,
        min_proceed_ev: float = 10.0,
        spot_tolerance_pct: float = 0.02,
    ) -> None:
        self.min_proceed_ev = min_proceed_ev
        self.spot_tolerance_pct = spot_tolerance_pct

    def review(
        self, dossier: CandidateDossier
    ) -> tuple[Verdict, str, list[str]]:
        notes: list[str] = []
        ev = dossier.ev_dollars

        # Rule 1: negative EV is blocked. Chart cannot save it.
        if ev < 0:
            notes.append(
                f"engine ev_dollars={ev:.2f} < 0 — chart cannot upgrade negative EV"
            )
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
                    f"disagrees with engine spot {engine_spot:.2f} (|Δ|={diff:.2f} > tol {tol:.2f})"
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
                notes.append(
                    f"chart phase={chart_phase} contradicts engine phase={engine_phase}"
                )
                return "skip", "phase_contradiction", notes
            notes.append(
                f"phase disagreement logged: chart={chart_phase} engine={engine_phase}"
            )

        # Rule 5: EV threshold.
        if ev >= self.min_proceed_ev:
            verdict: Verdict = "proceed"
            reason = "ev_above_threshold"
            notes.append(f"ev_dollars={ev:.2f} >= {self.min_proceed_ev} — proceed")
        else:
            verdict = "review"
            reason = "ev_below_proceed_threshold"
            notes.append(
                f"ev_dollars={ev:.2f} < min_proceed {self.min_proceed_ev} — human review"
            )

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
                        "— breach risk amplified, downgrade to review"
                    )
                    return "review", "dealer_short_gamma_above_put_wall", notes
            elif regime == "near_flip":
                notes.append("R6: dealer regime near gamma flip — downgrade to review")
                return "review", "dealer_near_flip", notes

        return verdict, reason, notes


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
        )

        verdict, reason, notes = reviewer.review(dossier)
        dossier.verdict = verdict
        dossier.verdict_reason = reason
        dossier.review_notes = notes
        dossiers.append(dossier)

    return dossiers
