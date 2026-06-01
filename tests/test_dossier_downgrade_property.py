"""§2 downgrade-only lattice property test for ``EnginePhaseReviewer.review``.

This is the mechanized form of the manual *"did any rule invert severity?"*
check that every decision-layer change in this repo currently relies on a human
to catch by eye (``CLAUDE.md`` §2). Instead of one example per rule, it states
the invariant ONCE as a property over the whole R1–R11 rule chain:

    A reviewer overlay can only HOLD or DOWNGRADE a verdict. It can never
    upgrade or rescue. Formally, on the severity lattice

        proceed < review < skip < blocked

    attaching any overlay (R6 short-gamma; R7–R10 portfolio context;
    R11 ``vix_level`` + ``prob_profit``) must satisfy

        severity(review(with_overlay)) >= severity(review(without_overlay))

    in EVERY cell of ``{overlay} x {ev_dollars}``.

The load-bearing piece is the ``META`` section. It introspects ``review``'s
source and proves, structurally:

  * exactly six downgrade-only overlay guards (``... and verdict == "proceed"``)
    exist — no more, no less;
  * no branch hard-returns ``"proceed"`` (so nothing can rescue a worse verdict);
  * every verdict literal returned is in the lattice; and
  * every overlay ``verdict_reason`` in the source is covered by a firing
    scenario in this file.

The consequence: a future rule (R12+) added WITHOUT severity protection — or
added without a firing scenario here — trips CI. That is what makes this test
protect reviewers that do not exist yet, for free.

Grounded against ``engine/candidate_dossier.py`` @ ``origin/main`` (R11 live via
#306/#307). If writing this test ever surfaces a *real* severity inversion in
``review`` it is a §2 finding, not a test bug — report it; do not paper over it.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer, Verdict
from engine.chart_context import ChartContext
from engine.portfolio_risk_gates import PortfolioContext

# ----------------------------------------------------------------------
# The severity lattice — pinned to the real Verdict Literal by
# test_severity_lattice_matches_verdict_literal below.
# ----------------------------------------------------------------------
# proceed (most tradeable) < review < skip < blocked (least tradeable).
# A reviewer overlay may only move a verdict UP this ladder (more
# restrictive) or hold it; it may never move it DOWN.
SEVERITY: dict[str, int] = {
    "proceed": 0,
    "review": 1,
    "skip": 2,
    "blocked": 3,
}


def _severity(verdict: str) -> int:
    assert verdict in SEVERITY, f"unknown verdict {verdict!r} — not in the severity lattice"
    return SEVERITY[verdict]


# The ev_dollars probe values the §2 card pins. Each lands in a distinct
# branch of review():
#   -50.0 -> R1   (negative)     -> blocked
#   nan   -> R1a  (non-finite)   -> blocked
#   +inf  -> R1a  (non-finite)   -> blocked
#   5.0   -> R5   (< MIN_PROCEED) -> review
#   50.0  -> R5   (>= threshold)  -> proceed  (the only cell where overlays fire)
#   None  -> CandidateDossier.ev_dollars coerces None -> 0.0 -> review
EV_VALUES: list[float | None] = [-50.0, float("nan"), float("inf"), 5.0, 50.0, None]


def _valid_chart(spot: float) -> ChartContext:
    """A clean, in-agreement chart so a proceed-eligible EV reaches R5
    ``proceed`` (no R2 missing / R3 spot-mismatch / R4 phase trigger)."""
    return ChartContext(
        ticker="FAKE",
        timeframe="1D",
        captured_at=datetime(2026, 4, 25, 12, 0, 0),
        screenshot_path=Path("/tmp/fake.png"),
        visible_price=spot,
        visible_indicators={},  # no 'phase' key -> R4 dormant
        source="test_downgrade_property",
    )


def _dossier(
    ticker: str,
    strike: float,
    ev_dollars: float | None,
    *,
    prob_profit: float | None = None,
    contracts: int = 1,
) -> CandidateDossier:
    """A proceed-eligible dossier (clean chart, spot == strike). The overlay
    attachments below are the ONLY thing that differs between the baseline and
    the with-overlay variant in each cell."""
    ev_row: dict[str, object] = {
        "ticker": ticker,
        "strike": strike,
        "premium": 2.0,
        "ev_dollars": ev_dollars,
        "iv": 0.25,
        "dte": 30,
        "spot": strike,
        "contracts": contracts,
        "cvar_5": -1234.0,  # R11 quotes this in its note; presence keeps it honest
    }
    if prob_profit is not None:
        ev_row["prob_profit"] = prob_profit
    return CandidateDossier(ticker=ticker, ev_row=ev_row, chart_context=_valid_chart(strike))


# R6 duck-typed stubs. review() reads only ``.regime`` and
# ``.nearest_put_wall.strike`` off ``market_structure`` (via getattr), so a stub
# exercises the exact R6 path without coupling this test to the real
# engine.dealer_positioning.MarketStructure constructor (which carries many
# unrelated required fields). The real GammaWall likewise exposes ``.strike``.
@dataclass
class _StubWall:
    strike: float


@dataclass
class _FiringMarketStructure:
    regime: str
    nearest_put_wall: _StubWall | None = None


# ----------------------------------------------------------------------
# Overlay firing scenarios — one per downgrade-only return branch in review().
# Each ``apply`` mutates a proceed-state dossier so that, at ev_dollars=50, the
# named overlay fires and downgrades proceed -> review. The R7–R10 recipes are
# lifted from the passing examples in tests/test_dossier_invariant.py so they
# stay in lockstep with what the engine actually treats as a breach; R6/R11
# derive directly from review()'s source.
# ----------------------------------------------------------------------
def _apply_r6_near_flip(d: CandidateDossier) -> None:
    d.market_structure = _FiringMarketStructure(regime="near_flip")


def _apply_r6_short_gamma(d: CandidateDossier) -> None:
    strike = float(d.ev_row["strike"])  # type: ignore[arg-type]
    # strike >= put-wall strike (equal qualifies via >=) -> R6 short-gamma fires.
    d.market_structure = _FiringMarketStructure(
        regime="short_gamma_amplifying",
        nearest_put_wall=_StubWall(strike=strike),
    )


def _apply_r7_var(d: CandidateDossier) -> None:
    idx = pd.date_range("2026-01-01", periods=120, freq="B")
    # Heavy-vol synthetic returns at tiny NAV make VaR_95 exceed 5% NAV.
    # Seeded (rng(7)) -> deterministic, matching test_dossier_invariant.
    returns = pd.DataFrame(
        {"portfolio": np.random.default_rng(7).normal(0, 0.08, 120)},
        index=idx,
    )
    d.portfolio_context = PortfolioContext(
        nav=10_000.0,
        spot_prices={d.ticker: 100.0},
        returns_data=returns,
    )


def _apply_r8_stress(d: CandidateDossier) -> None:
    d.portfolio_context = PortfolioContext(
        held_option_positions=[
            {
                "symbol": d.ticker,
                "option_type": "put",
                "strike": 100.0,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
            }
        ],
        spot_prices={d.ticker: 100.0},
        nav=5_000.0,  # tiny NAV -> vol-spike drawdown breaches the 8% cap
    )


def _apply_r8_dealer(d: CandidateDossier) -> None:
    d.portfolio_context = PortfolioContext(
        nav=10_000_000.0,  # huge NAV -> stress passes, dealer-regime trigger fires
        spot_prices={d.ticker: 180.0},
        dealer_regime_by_ticker={d.ticker: "short_gamma_amplifying"},
    )


def _apply_r9_sector(d: CandidateDossier) -> None:
    # AAPL is Info Tech; held $18k + proposed $18k = $36k / $50k NAV = 72% > 25%.
    d.portfolio_context = PortfolioContext(
        held_option_positions=[
            {
                "symbol": d.ticker,
                "option_type": "put",
                "strike": 180.0,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
            }
        ],
        spot_prices={d.ticker: 180.0},
        nav=50_000.0,
    )


def _apply_r10_single_name(d: CandidateDossier) -> None:
    # held $9k + proposed $5k = $14k / $100k = 14%: passes R9 (25%) but trips
    # R10 (10%) — the same config as test_r10_fires_when_r9_would_pass.
    d.portfolio_context = PortfolioContext(
        held_option_positions=[
            {
                "symbol": d.ticker,
                "option_type": "put",
                "strike": 90.0,
                "dte": 30,
                "iv": 0.25,
                "contracts": 1,
                "is_short": True,
            }
        ],
        spot_prices={d.ticker: 100.0},
        nav=100_000.0,
    )


def _apply_r11_vix(d: CandidateDossier) -> None:
    d.vix_level = 30.0  # > R11_VIX_THRESHOLD (25.0); ev_row carries prob_profit=0.95


@dataclass
class _OverlayScenario:
    """A single overlay's firing configuration."""

    rule: str  # human label, e.g. "R6:near_flip"
    reason: str  # the verdict_reason review() emits at the firing cell
    ticker: str
    strike: float
    apply: Callable[[CandidateDossier], None]
    prob_profit: float | None = field(default=None)

    def build(self, ev_dollars: float | None) -> tuple[CandidateDossier, CandidateDossier]:
        """(baseline, with_overlay) — identical except the overlay attachment."""
        base = _dossier(self.ticker, self.strike, ev_dollars, prob_profit=self.prob_profit)
        over = _dossier(self.ticker, self.strike, ev_dollars, prob_profit=self.prob_profit)
        self.apply(over)
        return base, over


OVERLAY_SCENARIOS: list[_OverlayScenario] = [
    _OverlayScenario("R6:near_flip", "dealer_near_flip", "FAKE", 100.0, _apply_r6_near_flip),
    _OverlayScenario(
        "R6:short_gamma", "dealer_short_gamma_above_put_wall", "FAKE", 100.0, _apply_r6_short_gamma
    ),
    _OverlayScenario("R7:var", "portfolio_var_breach", "TEST", 100.0, _apply_r7_var),
    _OverlayScenario("R8:stress", "stress_breach", "TEST", 100.0, _apply_r8_stress),
    _OverlayScenario("R8:dealer", "short_gamma_regime", "AAPL", 180.0, _apply_r8_dealer),
    _OverlayScenario("R9:sector", "sector_cap_breach", "AAPL", 180.0, _apply_r9_sector),
    _OverlayScenario("R10:single_name", "single_name_breach", "AAPL", 50.0, _apply_r10_single_name),
    _OverlayScenario(
        "R11:vix", "elevated_vol_top_bin", "AAPL", 100.0, _apply_r11_vix, prob_profit=0.95
    ),
]


# ----------------------------------------------------------------------
# The property — the §2 invariant as one statement instead of scattered
# per-rule examples.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("scenario", OVERLAY_SCENARIOS, ids=lambda s: s.rule)
@pytest.mark.parametrize("ev", EV_VALUES, ids=lambda e: f"ev={e}")
def test_overlay_never_reduces_severity(scenario: _OverlayScenario, ev: float | None) -> None:
    """For every (overlay, ev_dollars) cell: attaching the overlay holds or
    raises severity, never lowers it. This is CLAUDE.md §2 as one property."""
    reviewer = EnginePhaseReviewer()
    base, over = scenario.build(ev)
    v_base = reviewer.review(base)[0]
    v_over = reviewer.review(over)[0]
    assert _severity(v_over) >= _severity(v_base), (
        f"{scenario.rule} UPGRADED at ev={ev!r}: without={v_base!r} "
        f"(sev {_severity(v_base)}) -> with={v_over!r} (sev {_severity(v_over)}). "
        "An overlay reviewer must only hold or downgrade (CLAUDE.md §2)."
    )


@pytest.mark.parametrize("scenario", OVERLAY_SCENARIOS, ids=lambda s: s.rule)
def test_overlay_actually_downgrades_at_proceed(scenario: _OverlayScenario) -> None:
    """Teeth for the matrix above: at ev=50 the baseline reaches ``proceed`` and
    the overlay downgrades it to exactly ``review`` with the documented reason.

    Without this, a scenario that silently stopped firing (because the engine
    changed) would make ``test_overlay_never_reduces_severity`` vacuously true
    (with == without). This asserts the downgrade is real and strict."""
    reviewer = EnginePhaseReviewer()
    base, over = scenario.build(50.0)
    v_base, r_base, _ = reviewer.review(base)
    v_over, r_over, _ = reviewer.review(over)
    assert v_base == "proceed", f"{scenario.rule}: baseline not proceed ({v_base}/{r_base})"
    assert v_over == "review", f"{scenario.rule}: did not downgrade to review ({v_over}/{r_over})"
    assert _severity(v_over) > _severity(v_base)
    assert r_over == scenario.reason, (
        f"{scenario.rule}: fired with reason {r_over!r}, expected {scenario.reason!r}"
    )


def test_no_overlay_rescues_blocked() -> None:
    """The sharpest §2 cell, stated directly: no overlay can move a blocked
    verdict off ``blocked`` (R1/R1a short-circuit before any overlay runs).

    Checked two ways: (a) each overlay applied SINGLY; (b) all three
    *independent* attachment surfaces firing at once — ``market_structure``
    (R6) + ``portfolio_context`` (R7–R10 share this one field, so they are
    mutually exclusive by construction; only one can be attached at a time) +
    ``vix_level`` (R11). There is no single dossier on which all eight overlay
    branches can fire simultaneously — the three surfaces above are the true
    maximum, and even that cannot rescue a blocked verdict."""
    reviewer = EnginePhaseReviewer()
    for ev in (-50.0, float("nan"), float("inf")):
        # (a) each overlay applied singly to a blocked dossier
        for scenario in OVERLAY_SCENARIOS:
            d = _dossier(scenario.ticker, scenario.strike, ev, prob_profit=scenario.prob_profit)
            scenario.apply(d)
            assert reviewer.review(d)[0] == "blocked", (
                f"{scenario.rule} rescued a blocked trade at ev={ev!r} — §2 breach"
            )
        # (b) all three attachment surfaces firing at once
        d = _dossier("AAPL", 50.0, ev, prob_profit=0.95)
        _apply_r6_near_flip(d)  # market_structure surface
        _apply_r10_single_name(d)  # portfolio_context surface (one representative)
        _apply_r11_vix(d)  # vix_level surface
        assert reviewer.review(d)[0] == "blocked", (
            f"stacked overlays (R6 + portfolio_context + R11) rescued a blocked "
            f"trade at ev={ev!r} — §2 breach"
        )


# ----------------------------------------------------------------------
# Hypothesis: extend the ev_dollars axis to the full float domain for the
# cheap overlays (matches the property-test house style in
# tests/test_properties.py). The deterministic matrix above already covers
# the expensive R7/R9/R10 recompute paths exhaustively.
# ----------------------------------------------------------------------
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    _HYPOTHESIS = True
except ImportError:  # pragma: no cover - exercised only when hypothesis absent
    _HYPOTHESIS = False

_CHEAP_SCENARIOS = [
    s
    for s in OVERLAY_SCENARIOS
    if s.rule in {"R6:near_flip", "R6:short_gamma", "R8:dealer", "R11:vix"}
]


@pytest.mark.skipif(not _HYPOTHESIS, reason="hypothesis not installed")
@settings(max_examples=200, deadline=None)
@given(ev=st.floats(allow_nan=True, allow_infinity=True))
def test_overlay_never_reduces_severity_over_all_ev(ev: float) -> None:
    """severity(with) >= severity(without) for ANY ev_dollars (the whole real
    line, incl. nan/+-inf), across the cheap overlays."""
    reviewer = EnginePhaseReviewer()
    for scenario in _CHEAP_SCENARIOS:
        base, over = scenario.build(ev)
        v_base = reviewer.review(base)[0]
        v_over = reviewer.review(over)[0]
        assert _severity(v_over) >= _severity(v_base), (
            f"{scenario.rule} upgraded at ev={ev!r}: {v_base!r} -> {v_over!r}"
        )


# ======================================================================
# META — the load-bearing part. Structural assertions over review()'s
# source so a future rule added without severity protection trips CI.
# ======================================================================
# Number of downgrade-only overlay guards (`... and verdict == "proceed":`) in
# review(): R6, R7, R8, R9, R10, R11 — one guard each. If you ADD an overlay
# rule, add a firing scenario to OVERLAY_SCENARIOS above AND bump this; if you
# add a rule that does NOT gate on `verdict == "proceed"`, STOP — it can fire on
# a non-proceed verdict and may upgrade it, which breaks CLAUDE.md §2.
EXPECTED_OVERLAY_GUARDS = 6

# The one rule that returns "review" but is NOT a downgrade-only overlay: R2
# (chart missing). Everything else returning "review" is an overlay and must be
# covered by a firing scenario above.
_NON_OVERLAY_REVIEW_REASONS = {"chart_context_missing"}


def _review_tree() -> ast.AST:
    src = textwrap.dedent(inspect.getsource(EnginePhaseReviewer.review))
    return ast.parse(src)


def _returned_verdict_literals(tree: ast.AST) -> list[str]:
    """Every string constant returned in the FIRST slot of a ``return (...)``
    tuple in review()."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and node.value.elts:
            first = node.value.elts[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.append(first.value)
    return out


def _returned_verdict_reason_pairs(tree: ast.AST) -> list[tuple[str, str]]:
    """Every ``(verdict, reason)`` pair where BOTH are string constants."""
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            elts = node.value.elts
            if len(elts) >= 2 and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elts[:2]
            ):
                pairs.append((elts[0].value, elts[1].value))  # type: ignore[attr-defined]
    return pairs


def _count_proceed_guards(tree: ast.AST) -> int:
    """Count ``verdict == "proceed"`` comparisons — the downgrade-only overlay
    guards. The only `verdict == "proceed"` compares in review() are the six
    overlay gates."""
    n = 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "verdict"
        ):
            for op, comp in zip(node.ops, node.comparators, strict=False):
                if (
                    isinstance(op, ast.Eq)
                    and isinstance(comp, ast.Constant)
                    and comp.value == "proceed"
                ):
                    n += 1
    return n


def test_severity_lattice_matches_verdict_literal() -> None:
    """Pin the lattice to the real Verdict type. If a verdict string is
    added/removed without updating SEVERITY, the property tests would be
    unsound (severity() would KeyError or silently miss a verdict)."""
    literal_values = set(typing.get_args(Verdict))
    assert literal_values == set(SEVERITY), (
        f"Verdict Literal {literal_values} != severity lattice keys {set(SEVERITY)}. "
        "A verdict was added/removed — update SEVERITY and confirm its rank."
    )
    assert SEVERITY["proceed"] < SEVERITY["review"] < SEVERITY["skip"] < SEVERITY["blocked"]


def test_review_has_exactly_the_expected_overlay_guards() -> None:
    """Tripwire: review() has exactly EXPECTED_OVERLAY_GUARDS overlay gates.

    Adding (or removing) a `... and verdict == "proceed"` overlay guard trips
    this, forcing whoever changed review() to come back here, add a firing
    scenario to OVERLAY_SCENARIOS, and re-confirm the count — which guarantees
    the new rule is exercised by the downgrade-only property matrix."""
    n = _count_proceed_guards(_review_tree())
    assert n == EXPECTED_OVERLAY_GUARDS, (
        f"review() now has {n} `verdict == 'proceed'` overlay guards "
        f"(expected {EXPECTED_OVERLAY_GUARDS}). A downgrade-only overlay was "
        "added or removed. Add/remove its firing scenario in OVERLAY_SCENARIOS "
        "and update EXPECTED_OVERLAY_GUARDS. If a NEW rule does not gate on "
        "`verdict == 'proceed'`, it may fire on a non-proceed verdict and "
        "upgrade it — STOP, that breaks CLAUDE.md §2."
    )


def test_no_branch_hard_returns_proceed() -> None:
    """No-upgrade tripwire: ``proceed`` must reach the caller ONLY through the
    fall-through ``return verdict, reason, notes`` — never as a hard-returned
    literal. A ``return "proceed", ...`` anywhere is, by construction, a path
    that rescues a worse verdict, which §2 forbids."""
    literals = _returned_verdict_literals(_review_tree())
    assert "proceed" not in literals, (
        "review() hard-returns the literal verdict 'proceed' somewhere. The "
        "downgrade-only contract requires 'proceed' to flow only through the "
        "final fall-through return; a hard-returned 'proceed' is an upgrade "
        "path forbidden by CLAUDE.md §2."
    )


def test_all_returned_verdicts_are_in_the_lattice() -> None:
    """Every verdict literal review() can return is a known lattice verdict —
    so severity() can never KeyError at runtime and the property is total."""
    literals = set(_returned_verdict_literals(_review_tree()))
    unknown = literals - set(SEVERITY)
    assert not unknown, (
        f"review() returns verdict literal(s) {unknown} absent from the severity "
        "lattice. Add them to SEVERITY (with a rank) or fix the typo in review()."
    )


def test_every_overlay_reason_has_a_firing_scenario() -> None:
    """Coverage tripwire — the 'protect future reviewers for free' guarantee.

    Extract every reason returned alongside a ``"review"`` verdict in review(),
    drop the lone non-overlay one (R2 chart-missing), and assert the remainder
    is EXACTLY the set of reasons our firing scenarios exercise. A new overlay
    rule (R12+) that emits a new reason will widen the source set; if no
    scenario produces it, this fails — forcing the new rule into the
    downgrade-only matrix above."""
    pairs = _returned_verdict_reason_pairs(_review_tree())
    review_reasons = {reason for verdict, reason in pairs if verdict == "review"}
    overlay_reasons = review_reasons - _NON_OVERLAY_REVIEW_REASONS
    scenario_reasons = {s.reason for s in OVERLAY_SCENARIOS}
    assert overlay_reasons == scenario_reasons, (
        "Overlay verdict_reasons in review() are out of sync with the firing "
        f"scenarios.\n  in review() (overlay): {sorted(overlay_reasons)}\n"
        f"  covered by scenarios : {sorted(scenario_reasons)}\n"
        f"  uncovered (add a scenario!): {sorted(overlay_reasons - scenario_reasons)}\n"
        f"  stale (remove a scenario): {sorted(scenario_reasons - overlay_reasons)}"
    )


def test_every_overlay_return_is_a_downgrade_to_review() -> None:
    """Every overlay branch returns exactly ``"review"`` — the single safe
    downgrade target. An overlay that returned ``"proceed"`` (upgrade) or jumped
    to some other verdict via a new reason would be caught here."""
    pairs = _returned_verdict_reason_pairs(_review_tree())
    scenario_reasons = {s.reason for s in OVERLAY_SCENARIOS}
    for verdict, reason in pairs:
        if reason in scenario_reasons:
            assert verdict == "review", (
                f"overlay reason {reason!r} returns verdict {verdict!r}; overlays "
                "must downgrade to 'review' only (CLAUDE.md §2)."
            )
