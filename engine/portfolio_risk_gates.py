"""Portfolio-level risk gates — pure-function library shared by the
tracker hard-blocks and the dossier soft-warns (D17 / #154 C4).

S15 (PR #148) found that `engine/risk_manager.py` and
`engine/stress_testing.py` ship full machinery (portfolio Greeks,
parametric / historical / Monte-Carlo VaR, `SectorExposureManager`,
Kelly helpers, full stress ladders) but **none of it is imported by
the decision-layer trio** — positions open today with no NAV-level
sector cap, no portfolio-delta cap, no Kelly check. This module wires
that machinery in.

Single source of truth for both consumers:

- ``engine/wheel_tracker.py`` (Phase 2): three of these gates are
  hard-blocks at position-open time — ``check_sector_cap``,
  ``check_portfolio_delta``, ``check_kelly_size``. A failed gate
  refuses the position (audit-log reject).
- ``engine/candidate_dossier.py`` (Phase 3): two of these gates are
  soft-warns on the verdict — R7 = ``check_var``,
  R8 = ``check_stress_scenario`` + ``check_dealer_regime``. A failed
  gate downgrades ``proceed → review`` (never upgrades). R1 takes
  precedence as the hard-block for negative EV — see CLAUDE.md §2.

Pure-function design rationale (see #113 design comment, Q2/Q5):

- Tracker and dossier need identical predicates (same sector cap,
  same Kelly formula) — methods on ``WheelTracker`` would force the
  dossier reviewer to reach across modules awkwardly.
- Each gate function is independently unit-testable
  (``tests/test_portfolio_risk_gates.py``) with constructed position
  dicts; no fixture for the full tracker / dossier surface needed.
- Adding a sixth gate later is a new function + one import line,
  not surgery on the tracker.

All five locked defaults from #154 C4 are module-level constants
named ``_DEFAULT_*``. Overridable per-call but do **not** edit the
constants in this file — defaults are part of D17's contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as _date_cls
from typing import TYPE_CHECKING, Literal

from .risk_manager import (
    RiskManager,
    SectorExposureManager,
)
from .stress_testing import Scenario, ScenarioType, StressTester

if TYPE_CHECKING:
    from .wheel_tracker import WheelPosition


@dataclass
class PortfolioContext:
    """Portfolio-wide inputs for the D17 dossier soft-warns (R7 + R8).

    Attached to a :class:`CandidateDossier` so the reviewer can run
    the VaR and stress gates against the held book without re-reading
    the tracker. When absent, R7 and R8 do not fire — same
    "missing data → skip" semantics from Q3 of the design checkpoint
    (soft-warns should not fire on absent evidence).

    Caller (typically the dossier builder or the tracker, whichever
    has portfolio state in hand) constructs this once per ranking
    pass and attaches it to every dossier in the pass.

    Fields:
        held_option_positions: List of position dicts in the shape
            ``risk_manager.py``'s APIs expect — produced by
            :func:`take_snapshot`.
        spot_prices: Spot price per ticker for delta/stress math.
            Caller fetches.
        stock_holdings: ``(ticker, shares)`` pairs from
            :func:`take_snapshot`.
        nav: Net asset value (live or static per the operator's
            choice — see DECISIONS.md D17).
        dealer_regime_by_ticker: Optional ticker → regime map for the
            R8 short-gamma trigger. When None or candidate ticker
            missing, the regime branch of R8 skips
            (``check_dealer_regime`` returns ``missing_data``).
        returns_data: Optional historical returns for the
            ``check_var`` historical path.
        correlation_matrix: Optional correlation matrix for the
            ``check_var`` covariance path.
        volatilities: Optional per-ticker vol overrides for the
            covariance path.
    """

    held_option_positions: list[dict] = field(default_factory=list)
    spot_prices: dict[str, float] = field(default_factory=dict)
    stock_holdings: list[tuple[str, int]] = field(default_factory=list)
    nav: float = 0.0
    dealer_regime_by_ticker: dict[str, str] | None = None
    returns_data: object | None = None
    correlation_matrix: object | None = None
    volatilities: dict[str, float] | None = None


# ----------------------------------------------------------------------
# Locked defaults (D17). Override per-call; do not change the constants.
# ----------------------------------------------------------------------
_DEFAULT_MAX_SECTOR_PCT = 0.25
_DEFAULT_DELTA_CAP_PER_100K_NAV = 300.0
_DEFAULT_KELLY_FRACTION = 0.5
_DEFAULT_MAX_VAR_PCT = 0.05
_DEFAULT_VAR_CONFIDENCE = 0.95
_DEFAULT_VAR_HORIZON_DAYS = 30
_DEFAULT_MAX_STRESS_DRAWDOWN_PCT = 0.08
# Single-name (per-underlying) exposure cap. Aggregates SHORT option
# notional across all held positions on the same symbol; refuses /
# downgrades when adding the candidate would push the per-name total
# over ``max_pct × NAV``. Default 10% — a balanced wheel book can hold
# up to 10 names at full cap before maxing out. Closes the documented
# gap where ``check_sector_cap`` aggregates by GICS sector but a
# single ticker concentrated as the only name in its sector could
# still pass the sector check at 25% NAV — the per-name cap is the
# tighter per-underlying floor underneath the sector ceiling. F4
# damage-bounding mechanism (`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10
# noted the named cases as idiosyncratic single-name drawdowns).
_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10

# S31 Fix #5 — alt-cluster exposure cap. Complementary to
# check_sector_cap (GICS-strict): aggregates exposure across
# colloquial clusters that cross GICS sector boundaries. Closes the
# S31 F6 trader-mental-model gap where "tech" in trader-speak spans
# Information Technology (AAPL/MSFT/NVDA/AVGO) + Communication
# Services (META/GOOGL) + Consumer Discretionary (TSLA/AMZN) so the
# GICS sector cap doesn't aggregate them.
#
# Default cap is 0.40 (looser than the 0.25 sector cap because
# clusters are wider — a 25% mega-cap-growth cap would be too
# restrictive for institutional wheel books). Override per-call when
# tighter discipline is desired.
_DEFAULT_MAX_ALT_CLUSTER_PCT = 0.40

# Default cluster set. Membership is intentionally tight (rather than
# trying to enumerate every possible cluster a trader might care
# about) — operators with a different mental model pass a `clusters`
# kwarg with their own definitions. The starter set names the most
# common one ("the FAAMNG-T super-cluster") that recurs across
# institutional wheel-trader feedback.
_DEFAULT_ALT_CLUSTERS: dict = {
    "mega_cap_growth": frozenset({"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO"}),
}

# The C4 standard stress scenario — instantiated here (not in
# stress_testing.py's HYPOTHETICAL_SCENARIOS) to respect the prompt
# rule_out forbidding modifications to stress_testing.py. If a future
# task wants this scenario in the standard library, relocating it is
# a trivial follow-up — see #113 design comment.
_C4_VOL_SPIKE_SCENARIO = Scenario(
    name="C4 Vol Spike",
    scenario_type=ScenarioType.HYPOTHETICAL,
    description="D17 standard portfolio-risk stress: 10% spot drop + 30% IV spike",
    spot_change_pct=-0.10,
    iv_change_pct=0.30,
)


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------
@dataclass
class GateResult:
    """Outcome of a single gate check.

    - ``passed=True, reason=None`` — gate did not fire (the candidate
      survives this gate).
    - ``passed=False, reason="<one_of_the_known_reasons>"`` — gate
      fires. Tracker hard-blocks refuse the position; dossier
      soft-warns downgrade the verdict.
    - ``passed=True, reason="missing_data"`` — gate cannot evaluate
      (e.g. VaR with no correlation matrix or returns data, per Q3
      from the #113 design checkpoint). Treated as "no evidence to
      fire" so the candidate is not penalised for missing input data
      — matches D11's "no silent substitution" principle (don't
      claim what you can't prove). Caller is expected to surface the
      ``missing_data`` reason via ``details`` for the audit trail.

    ``details`` is a free-form dict that carries the field bag the
    audit-log entry shapes expect (e.g. ``sector_pct``,
    ``kelly_recommended_max``, ``portfolio_delta_dollars``). The
    schema-closure regression in
    ``tests/test_ev_authority_log_schema.py`` (Phase 2) pins the keys
    that flow from here into the tracker's audit log.
    """

    passed: bool
    reason: str | None = None
    details: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Position adapter — WheelPosition → upstream-API position dicts
# ----------------------------------------------------------------------
@dataclass
class PortfolioSnapshot:
    """Entry-time snapshot of the held portfolio.

    The upstream risk_manager / stress_testing APIs expect option
    position dicts (with strike, dte, iv, etc.); stock holdings are
    represented separately because those APIs are option-centric.
    """

    option_positions: list[dict] = field(default_factory=list)
    stock_holdings: list[tuple[str, int]] = field(default_factory=list)
    """Stock holdings as ``(ticker, shares)`` pairs."""


def take_snapshot(
    positions: dict[str, WheelPosition],
    *,
    today: _date_cls | None = None,
) -> PortfolioSnapshot:
    """Build a `PortfolioSnapshot` from `WheelTracker.positions`.

    Uses **entry-time IV** as the implied volatility input (the
    "entry-time snapshot" semantics from the C4 spec). When live
    Greeks are available (a future enhancement), the caller can
    enrich the snapshot post-hoc; the default path keeps the engine
    runnable in the Bloomberg-only sandbox where live option Greeks
    are not available.

    Per-state mapping:

    - ``SHORT_PUT`` → one option dict (short put leg).
    - ``STOCK_OWNED`` → one stock holding entry (no option leg —
      the short put was assigned).
    - ``COVERED_CALL`` → one option dict (short call leg) + one
      stock holding entry (the assigned shares).

    Positions with no live legs (e.g. fully closed but still in the
    ``positions`` dict by some caller-side mistake) are skipped
    silently.

    Args:
        positions: ``WheelTracker.positions``; a dict of ticker →
            ``WheelPosition``.
        today: Reference date for DTE computation. Defaults to
            ``date.today()``; injectable for deterministic tests.

    Returns:
        A `PortfolioSnapshot` with option_positions ready to pass to
        ``RiskManager.calculate_portfolio_greeks`` / ``calculate_var``
        / ``StressTester.run_scenario``, and stock_holdings for the
        portfolio-delta computation.
    """
    from .wheel_tracker import PositionState  # avoid circular import at module load

    if today is None:
        today = _date_cls.today()

    snapshot = PortfolioSnapshot()

    for ticker, pos in positions.items():
        if pos.state == PositionState.SHORT_PUT:
            if (
                pos.put_strike is not None
                and pos.put_entry_iv is not None
                and pos.put_expiration_date is not None
            ):
                dte = max(0, (pos.put_expiration_date - today).days)
                snapshot.option_positions.append(
                    {
                        "symbol": ticker,
                        "option_type": "put",
                        "strike": float(pos.put_strike),
                        "dte": dte,
                        "iv": float(pos.put_entry_iv),
                        "contracts": 1,
                        "is_short": True,
                    }
                )
        elif pos.state == PositionState.STOCK_OWNED:
            if pos.stock_shares > 0:
                snapshot.stock_holdings.append((ticker, int(pos.stock_shares)))
        elif pos.state == PositionState.COVERED_CALL:
            if pos.stock_shares > 0:
                snapshot.stock_holdings.append((ticker, int(pos.stock_shares)))
            if (
                pos.call_strike is not None
                and pos.call_entry_iv is not None
                and pos.call_expiration_date is not None
            ):
                dte = max(0, (pos.call_expiration_date - today).days)
                snapshot.option_positions.append(
                    {
                        "symbol": ticker,
                        "option_type": "call",
                        "strike": float(pos.call_strike),
                        "dte": dte,
                        "iv": float(pos.call_entry_iv),
                        "contracts": 1,
                        "is_short": True,
                    }
                )

    return snapshot


# ----------------------------------------------------------------------
# Gate 1: Sector cap (tracker hard-block)
# ----------------------------------------------------------------------
def check_sector_cap(
    symbol: str,
    proposed_notional: float,
    held_option_positions: list[dict],
    nav: float,
    *,
    max_sector_pct: float = _DEFAULT_MAX_SECTOR_PCT,
    sector_map: dict[str, str] | None = None,
) -> GateResult:
    """Refuse if opening the candidate would push the symbol's sector
    over ``max_sector_pct`` of NAV.

    Wraps ``SectorExposureManager.check_sector_limit``. Default
    sector map is ``DEFAULT_SECTOR_MAP`` from `risk_manager.py`.

    Args:
        symbol: The candidate's ticker.
        proposed_notional: Dollar notional of the proposed position
            (``strike * 100 * contracts`` for a short put).
        held_option_positions: The current option positions from
            ``take_snapshot(...).option_positions``. Stock holdings
            don't contribute to the option-side sector exposure.
        nav: Net asset value (live cash + mark-to-market). Caller
            decides whether to pass live or static NAV.
        max_sector_pct: Sector concentration cap; default 25% per
            D17.
        sector_map: Optional symbol → sector override; defaults to
            ``DEFAULT_SECTOR_MAP``.
    """
    manager = SectorExposureManager(
        sector_map=sector_map,
        max_sector_pct=max_sector_pct,
    )
    # SectorExposureManager.check_sector_limit expects positions with
    # symbol / strike / contracts; option_positions already match.
    is_allowed, reason_str = manager.check_sector_limit(
        symbol=symbol,
        proposed_notional=proposed_notional,
        positions=held_option_positions,
        portfolio_value=nav,
    )
    sector = manager.get_sector(symbol)

    # Compute the post-open exposure for the details bag.
    exposures = manager.calculate_sector_exposures(held_option_positions, nav)
    current_notional = exposures.get(sector).notional_exposure if sector in exposures else 0.0
    post_open_pct = (current_notional + proposed_notional) / nav if nav > 0 else 0.0

    if is_allowed:
        return GateResult(
            passed=True,
            reason=None,
            details={
                "sector": sector,
                "post_open_sector_pct": post_open_pct,
                "sector_limit": max_sector_pct,
            },
        )
    return GateResult(
        passed=False,
        reason="sector_cap_breach",
        details={
            "sector": sector,
            "post_open_sector_pct": post_open_pct,
            "sector_limit": max_sector_pct,
            "narrative": reason_str,
        },
    )


# ----------------------------------------------------------------------
# Gate 1c: Single-name exposure cap (F4 damage-bounding addition)
# ----------------------------------------------------------------------
def check_single_name_cap(
    symbol: str,
    proposed_notional: float,
    held_option_positions: list[dict],
    nav: float,
    *,
    max_single_name_pct: float = _DEFAULT_MAX_SINGLE_NAME_PCT,
) -> GateResult:
    """Refuse / downgrade if opening the candidate would push the
    SINGLE-NAME (per-underlying) short-option notional over
    ``max_single_name_pct × NAV``.

    Bounds the F4-style idiosyncratic-drawdown damage that no
    market-wide regime detector can predict (see
    ``docs/F4_TAIL_RISK_DIAGNOSTIC.md`` §10). Tighter per-underlying
    floor that sits beneath the GICS sector cap: a single ticker
    concentrated as the only name in its sector could still pass
    ``check_sector_cap`` at 25% NAV; this gate caps it at 10% NAV
    instead (default).

    Aggregation rule: sums the dollar notional of every HELD short
    option (put or call) whose ``symbol`` matches the candidate.
    Long positions, stock holdings, and other symbols are ignored.
    Notional is ``strike × 100 × contracts`` — the same convention
    ``check_sector_cap`` uses.

    Args:
        symbol: The candidate's ticker.
        proposed_notional: Dollar notional of the proposed position
            (``strike * 100 * contracts``).
        held_option_positions: Current option positions from
            ``take_snapshot(...).option_positions``. Each dict must
            carry ``symbol``, ``strike``, ``contracts``, and
            ``is_short``; non-short / non-matching rows are skipped.
        nav: Net asset value. Caller chooses live vs static.
        max_single_name_pct: Per-underlying cap; default 10% per
            ``_DEFAULT_MAX_SINGLE_NAME_PCT``.

    Returns:
        :class:`GateResult` with ``reason="single_name_breach"`` on
        failure. ``details`` carries ``symbol``, ``current_name_notional``,
        ``post_open_name_notional``, ``post_open_name_pct``, and
        ``name_limit_pct`` for the audit log.

    Missing-data behaviour: when ``nav <= 0`` the gate returns
    ``passed=True, reason="missing_data"`` (matches Q3 semantics from
    R7/R8/R9 — soft-warns don't fire on absent evidence; tracker hard-
    blocks treat this as "no NAV evidence, don't refuse"). Tracker
    callers are expected to gate on ``nav_exhausted`` separately.
    """
    sym = (symbol or "").upper()

    # Missing-data path: can't divide by zero NAV; can't enforce a %
    # cap without it. Tracker / dossier both treat this as no-fire.
    if nav <= 0:
        return GateResult(
            passed=True,
            reason="missing_data",
            details={
                "skip_reason": "nav_zero_or_negative",
                "symbol": sym,
                "name_limit_pct": max_single_name_pct,
            },
        )

    # Aggregate existing short option notional for the same symbol.
    current_name_notional = 0.0
    for pos in held_option_positions or []:
        try:
            if not bool(pos.get("is_short", False)):
                continue
            pos_sym = str(pos.get("symbol", "")).upper()
            if pos_sym != sym:
                continue
            strike = float(pos.get("strike", 0) or 0)
            contracts = int(pos.get("contracts", 0) or 0)
            current_name_notional += strike * 100.0 * contracts
        except (TypeError, ValueError):
            # Skip malformed rows; treat as zero contribution. Don't
            # crash the gate on bad inputs — caller has bigger problems.
            continue

    post_open_notional = current_name_notional + max(0.0, float(proposed_notional))
    post_open_pct = post_open_notional / nav

    details = {
        "symbol": sym,
        "current_name_notional": current_name_notional,
        "post_open_name_notional": post_open_notional,
        "post_open_name_pct": post_open_pct,
        "name_limit_pct": max_single_name_pct,
    }
    if post_open_pct > max_single_name_pct:
        return GateResult(
            passed=False,
            reason="single_name_breach",
            details=details,
        )
    return GateResult(passed=True, reason=None, details=details)


# ----------------------------------------------------------------------
# Gate 1b: Alt-cluster exposure cap (S31 Fix #5)
# ----------------------------------------------------------------------
def check_alt_cluster_cap(
    symbol: str,
    proposed_notional: float,
    held_option_positions: list[dict],
    nav: float,
    *,
    clusters: dict | None = None,
    max_cluster_pct: float = _DEFAULT_MAX_ALT_CLUSTER_PCT,
) -> GateResult:
    """Refuse if opening the candidate would push any alt-cluster the
    symbol belongs to over ``max_cluster_pct`` of NAV.

    **Complementary to** :func:`check_sector_cap` (GICS-strict).
    Closes the S31 F6 trader-mental-model gap where "tech"
    colloquially crosses three GICS sectors (Information Technology /
    Communication Services / Consumer Discretionary). The GICS
    sector_cap treats AAPL + TSLA as separate sectors; this gate lets
    a trader define their own clusters that aggregate them.

    Pass / fail semantics mirror :func:`check_sector_cap`:

    - ``passed=True`` if (a) the symbol is in no defined cluster, or
      (b) for every cluster the symbol belongs to, the post-open
      cluster exposure is below ``max_cluster_pct`` of NAV.
    - ``passed=False, reason="alt_cluster_cap_breach"`` if any
      cluster the symbol belongs to would exceed the cap.

    Args:
        symbol: The candidate's ticker.
        proposed_notional: Dollar notional of the proposed position
            (``strike * 100 * contracts`` for a short put).
        held_option_positions: Current option positions in the same
            shape ``take_snapshot(...).option_positions`` returns
            (each carrying ``symbol``, ``strike``, ``contracts``).
            Stock holdings are NOT aggregated here -- mirrors the
            sector-cap convention that stock-leg sizing is handled
            separately (no margin consumed).
        nav: Net asset value (live cash + mark-to-market). Caller
            decides whether to pass live or static NAV.
        clusters: Optional mapping ``{cluster_name: frozenset(symbols)}``
            overriding the default cluster definitions. The default
            (:data:`_DEFAULT_ALT_CLUSTERS`) ships one cluster
            (``"mega_cap_growth"``) -- operators with a richer mental
            model pass their own.
        max_cluster_pct: Per-cluster cap; default 0.40. Looser than
            the 0.25 GICS sector cap because clusters are wider.

    Returns:
        :class:`GateResult` with ``details`` carrying the cluster
        memberships and post-open percentages. Key shape mirrors
        :func:`check_sector_cap` post-S31 F7 fix (pass and fail use
        the same ``post_open_*`` keys -- no key asymmetry).
    """
    if clusters is None:
        clusters = _DEFAULT_ALT_CLUSTERS

    # Which clusters does this symbol belong to?
    memberships = sorted(name for name, members in clusters.items() if symbol in members)
    if not memberships:
        # Symbol is not in any defined cluster — the gate is silent
        # (matches the "missing data → skip" semantics from Q3 of the
        # #113 design checkpoint).
        return GateResult(
            passed=True,
            reason=None,
            details={
                "cluster_memberships": [],
                "max_post_open_cluster_pct": 0.0,
                "cluster_limit": max_cluster_pct,
            },
        )

    # For each cluster the symbol belongs to, sum existing notional
    # from held option positions whose symbol is in the same cluster.
    cluster_pcts: dict[str, float] = {}
    for cname in memberships:
        members = clusters[cname]
        current = sum(
            float(p.get("strike", 0.0)) * 100.0 * int(p.get("contracts", 1))
            for p in held_option_positions
            if p.get("symbol") in members
        )
        post_open = (current + proposed_notional) / nav if nav > 0 else 0.0
        cluster_pcts[cname] = post_open

    breaching = [(cname, pct) for cname, pct in cluster_pcts.items() if pct > max_cluster_pct]
    max_post_open = max(cluster_pcts.values()) if cluster_pcts else 0.0

    if breaching:
        worst_cluster, worst_pct = max(breaching, key=lambda kv: kv[1])
        narrative = (
            f"Alt-cluster '{worst_cluster}' would be {worst_pct:.1%} "
            f"of NAV (limit {max_cluster_pct:.1%}). "
            f"Cluster members held + proposed exceed the cap."
        )
        return GateResult(
            passed=False,
            reason="alt_cluster_cap_breach",
            details={
                "cluster_memberships": memberships,
                "breaching_cluster": worst_cluster,
                "post_open_cluster_pct": worst_pct,
                "max_post_open_cluster_pct": max_post_open,
                "cluster_limit": max_cluster_pct,
                "narrative": narrative,
            },
        )

    return GateResult(
        passed=True,
        reason=None,
        details={
            "cluster_memberships": memberships,
            "max_post_open_cluster_pct": max_post_open,
            "cluster_limit": max_cluster_pct,
        },
    )


# ----------------------------------------------------------------------
# Gate 2: Portfolio delta cap (tracker hard-block)
# ----------------------------------------------------------------------
def check_portfolio_delta(
    held_option_positions: list[dict],
    spot_prices: dict[str, float],
    candidate_option: dict,
    stock_holdings: list[tuple[str, int]],
    nav: float,
    *,
    delta_cap_per_100k_nav: float = _DEFAULT_DELTA_CAP_PER_100K_NAV,
    risk_free_rate: float = 0.05,
) -> GateResult:
    """Refuse if opening the candidate would push portfolio delta
    over the per-NAV cap.

    Delta cap is ``±delta_cap_per_100k_nav * (NAV / 100_000)``. Default
    ``±$300 / $100k NAV`` per D17.

    Portfolio delta = option-leg delta-dollars (from
    ``RiskManager.calculate_portfolio_greeks``) + stock-leg delta-
    dollars (``shares × spot`` for each holding).

    Args:
        held_option_positions: Current option positions.
        spot_prices: Spot price per ticker (caller resolves).
        candidate_option: Position dict for the candidate (same shape
            as held positions). Pass an empty dict ``{}`` if checking
            the existing portfolio without a new candidate.
        stock_holdings: ``(ticker, shares)`` pairs from
            ``take_snapshot(...).stock_holdings``.
        nav: Net asset value; the cap scales with this.
        delta_cap_per_100k_nav: Default 300.0 per D17.
        risk_free_rate: Used by Black-Scholes when computing the
            candidate's delta. Default 0.05.
    """
    rm = RiskManager(risk_free_rate=risk_free_rate)

    # Option-leg delta-dollars
    option_positions_with_candidate = list(held_option_positions)
    if candidate_option:
        option_positions_with_candidate.append(candidate_option)

    # The greeks API needs spot per symbol; fall back to strike if missing.
    full_spots = dict(spot_prices)
    for p in option_positions_with_candidate:
        full_spots.setdefault(p["symbol"], p["strike"])

    greeks = rm.calculate_portfolio_greeks(option_positions_with_candidate, full_spots)
    option_delta_dollars = greeks.delta_dollars

    # Stock-leg delta-dollars
    stock_delta_dollars = 0.0
    for ticker, shares in stock_holdings:
        spot = full_spots.get(ticker, 0.0)
        stock_delta_dollars += shares * spot

    portfolio_delta_dollars = option_delta_dollars + stock_delta_dollars
    cap_dollars = delta_cap_per_100k_nav * (nav / 100_000.0)

    if abs(portfolio_delta_dollars) <= cap_dollars:
        return GateResult(
            passed=True,
            reason=None,
            details={
                "portfolio_delta_dollars": portfolio_delta_dollars,
                "delta_cap_dollars": cap_dollars,
            },
        )

    # For the audit-log details, recompute the pre-candidate delta so
    # the operator can see how much the candidate moved the needle.
    pre_greeks = rm.calculate_portfolio_greeks(held_option_positions, full_spots)
    pre_option_delta = pre_greeks.delta_dollars
    pre_portfolio_delta = pre_option_delta + stock_delta_dollars

    return GateResult(
        passed=False,
        reason="portfolio_delta_breach",
        details={
            "current_portfolio_delta_dollars": pre_portfolio_delta,
            "post_open_delta_dollars": portfolio_delta_dollars,
            "delta_cap_dollars": cap_dollars,
        },
    )


# ----------------------------------------------------------------------
# Gate 3: Kelly size (tracker hard-block)
# ----------------------------------------------------------------------
def check_kelly_size(
    margin_required: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    nav: float,
    *,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
) -> GateResult:
    """Refuse if the candidate's margin requirement exceeds
    ``kelly_fraction × NAV`` — the half-Kelly per-trade sizing cap.

    **Implementation note (#154 C4 Phase 2 amendment):** an earlier
    sketch used the classical binary Kelly formula from
    ``calculate_kelly_fraction`` (``f* = (p*b - q) / b`` where
    ``b = avg_win / avg_loss``). For typical short puts, the
    binary form returns 0 — the loss-to-win ratio
    (``avg_loss = (strike − premium) × 100`` vs ``avg_win = premium × 100``)
    is so wide that ``f*`` goes negative for any realistic
    ``win_rate``. After the ``max(0, …)`` clamp, recommended
    exposure is $0 and the gate refuses every short put,
    regardless of edge. That makes the gate worse than useless
    (it just refuses indiscriminately, with no information value).

    The shipped form is the **per-trade NAV cap** interpretation of
    "half-Kelly": no single trade may consume more than
    ``kelly_fraction × NAV`` of margin (default 50% at half-Kelly).
    This is the practitioner's reading of half-Kelly as a sizing
    cap, captures the spirit of the D17 spec (limit single-position
    concentration), and actually fires useful refusals when an
    operator over-sizes one trade.

    ``win_rate`` / ``avg_win`` / ``avg_loss`` are kept in the
    signature for forward-compatibility with a future continuous-
    Kelly refinement (``f* = μ/σ²``) that would use a per-trade
    EV/variance estimate; they are not consumed by the current
    formula and the audit-log details bag does **not** carry them
    either (``common_details`` below ships only ``margin_required``,
    ``kelly_recommended_max``, and ``kelly_fraction``; the schema
    regression in ``tests/test_ev_authority_log_schema.py``'s
    ``_SHAPE_REJECT_KELLY`` pins exactly those three). When the
    continuous-Kelly refinement lands, both consumption and
    audit-log surfacing of these inputs should be added together.

    **Current-path reachability (#166 B3).** Under
    ``WheelTracker.open_short_put``'s single-contract emission today
    (``WheelPosition`` has no ``contracts`` field; ``available_buying_power``
    hardcodes 100 shares per ``SHORT_PUT``), the Kelly gate is
    structurally unreachable at any realistic NAV. One Reg-T short-put
    margin is typically $4-$13k for S&P names; the cap at default
    ``kelly_fraction=0.5`` is $50k at $100k NAV and scales linearly,
    so any NAV that leaves room for the (smaller, $300/$100k) delta
    cap to clear will also clear the Kelly cap, and below that the
    legacy ``self.cash < margin_required`` check refuses before D17
    is entered. The gate is therefore **preemptively reserved for a
    future multi-contract position path** where one trade can consume
    meaningful fractional NAV. Until that lands, no
    ``kelly_size_exceeded`` audit-log entry will be emitted from the
    tracker entry path; the function is exercised in tests via direct
    invocation and audit-log schema closure only. This is a
    conservative gate (it can only refuse, never rescue — see §2),
    so the reservation does not introduce risk.

    Args:
        margin_required: The Reg-T margin the trade would consume.
        win_rate: Probability of profit (forward-compat; unused).
        avg_win: Average win dollars (forward-compat; unused).
        avg_loss: Average loss dollars (forward-compat; unused).
        nav: Net asset value.
        kelly_fraction: Per-trade NAV cap as a fraction of NAV;
            default 0.5 (half-Kelly).
    """
    # NOTE: ``calculate_kelly_fraction`` (binary Kelly from
    # risk_manager.py) is intentionally NOT used here — see the
    # docstring above for the rationale. Kept available via the
    # import so a future continuous-Kelly refinement can drop in
    # without an import churn.
    _ = (win_rate, avg_win, avg_loss)  # forward-compat; unused today
    recommended_max = kelly_fraction * nav

    common_details = {
        "margin_required": margin_required,
        "kelly_recommended_max": recommended_max,
        "kelly_fraction": kelly_fraction,
    }

    if margin_required <= recommended_max:
        return GateResult(passed=True, reason=None, details=common_details)
    return GateResult(
        passed=False,
        reason="kelly_size_exceeded",
        details=common_details,
    )


# ----------------------------------------------------------------------
# Gate 4: VaR (dossier soft-warn → R7)
# ----------------------------------------------------------------------
def check_var(
    held_option_positions: list[dict],
    spot_prices: dict[str, float],
    candidate_option: dict,
    nav: float,
    *,
    max_var_pct: float = _DEFAULT_MAX_VAR_PCT,
    confidence: float = _DEFAULT_VAR_CONFIDENCE,
    horizon_days: int = _DEFAULT_VAR_HORIZON_DAYS,
    returns_data=None,
    correlation_matrix=None,
    volatilities: dict[str, float] | None = None,
    risk_free_rate: float = 0.05,
) -> GateResult:
    """Soft-warn (R7) if portfolio-level VaR_95 (30-day horizon)
    exceeds ``max_var_pct × NAV``.

    Per Q3 of the #113 design checkpoint: when neither
    ``correlation_matrix`` nor ``returns_data`` is provided, this
    gate **returns passed=True with reason="missing_data"** rather
    than silently falling through to the delta-normal approximation
    (D11's "no silent substitution"). The caller surfaces the
    missing_data reason via the audit / dossier note; R7 does not
    fire because there is no evidence to fire on.

    Args:
        held_option_positions, spot_prices, candidate_option: Same
            shape as ``check_portfolio_delta``.
        nav: Net asset value.
        max_var_pct: Default 0.05 (5% NAV) per D17.
        confidence: Default 0.95.
        horizon_days: Default 30.
        returns_data, correlation_matrix, volatilities: Optional
            inputs for the historical / covariance VaR paths in
            ``RiskManager.calculate_var``. If both correlation_matrix
            and returns_data are None, the gate skips (see above).
        risk_free_rate: Default 0.05.
    """
    if correlation_matrix is None and returns_data is None:
        return GateResult(
            passed=True,
            reason="missing_data",
            details={
                "var_check": "skipped",
                "skip_reason": "no_correlation_matrix_or_returns_data",
            },
        )

    rm = RiskManager(risk_free_rate=risk_free_rate)
    positions_with_candidate = list(held_option_positions)
    if candidate_option:
        positions_with_candidate.append(candidate_option)

    full_spots = dict(spot_prices)
    for p in positions_with_candidate:
        full_spots.setdefault(p["symbol"], p["strike"])

    var_dollars, cvar_dollars = rm.calculate_var(
        portfolio_value=nav,
        positions=positions_with_candidate,
        spot_prices=full_spots,
        returns_data=returns_data,
        correlation_matrix=correlation_matrix,
        volatilities=volatilities,
        confidence=confidence,
        horizon_days=horizon_days,
    )

    var_pct = var_dollars / nav if nav > 0 else 0.0
    threshold = max_var_pct

    if var_pct <= threshold:
        return GateResult(
            passed=True,
            reason=None,
            details={
                "var_dollars": var_dollars,
                "cvar_dollars": cvar_dollars,
                "var_pct": var_pct,
                "var_limit_pct": threshold,
                "confidence": confidence,
                "horizon_days": horizon_days,
            },
        )
    return GateResult(
        passed=False,
        reason="portfolio_var_breach",
        details={
            "var_dollars": var_dollars,
            "cvar_dollars": cvar_dollars,
            "var_pct": var_pct,
            "var_limit_pct": threshold,
            "confidence": confidence,
            "horizon_days": horizon_days,
        },
    )


# ----------------------------------------------------------------------
# Gate 5a: Stress scenario (dossier soft-warn — R8 trigger 1)
# ----------------------------------------------------------------------
def check_stress_scenario(
    held_option_positions: list[dict],
    spot_prices: dict[str, float],
    candidate_option: dict,
    nav: float,
    *,
    max_drawdown_pct: float = _DEFAULT_MAX_STRESS_DRAWDOWN_PCT,
    scenario: Scenario | None = None,
    risk_free_rate: float = 0.05,
) -> GateResult:
    """Soft-warn (R8 trigger 1) if the C4 vol-spike scenario shows
    portfolio drawdown > ``max_drawdown_pct × NAV``.

    Default scenario is ``_C4_VOL_SPIKE_SCENARIO`` (-10% spot +30%
    IV) per D17.

    Args:
        held_option_positions, spot_prices, candidate_option: Same
            shape as ``check_portfolio_delta``.
        nav: Net asset value.
        max_drawdown_pct: Default 0.08 (8% NAV) per D17.
        scenario: Override the default C4 vol-spike with an arbitrary
            ``Scenario``; useful for dossier-side scenario sweeps.
        risk_free_rate: Default 0.05.
    """
    if scenario is None:
        scenario = _C4_VOL_SPIKE_SCENARIO

    tester = StressTester(risk_free_rate=risk_free_rate)
    positions_with_candidate = list(held_option_positions)
    if candidate_option:
        positions_with_candidate.append(candidate_option)

    full_spots = dict(spot_prices)
    for p in positions_with_candidate:
        full_spots.setdefault(p["symbol"], p["strike"])

    if not positions_with_candidate:
        return GateResult(
            passed=True,
            reason=None,
            details={
                "scenario_name": scenario.name,
                "portfolio_pnl_dollars": 0.0,
                "drawdown_pct": 0.0,
                "drawdown_limit_pct": max_drawdown_pct,
                "note": "no_positions_to_stress",
            },
        )

    result = tester.run_scenario(
        scenario=scenario,
        positions=positions_with_candidate,
        spot_prices=full_spots,
        portfolio_value=nav,
    )

    # Drawdown is the negative P&L as a fraction of NAV.
    drawdown_pct = -result.portfolio_pnl / nav if nav > 0 and result.portfolio_pnl < 0 else 0.0

    if drawdown_pct <= max_drawdown_pct:
        return GateResult(
            passed=True,
            reason=None,
            details={
                "scenario_name": scenario.name,
                "portfolio_pnl_dollars": result.portfolio_pnl,
                "drawdown_pct": drawdown_pct,
                "drawdown_limit_pct": max_drawdown_pct,
            },
        )
    return GateResult(
        passed=False,
        reason="stress_breach",
        details={
            "scenario_name": scenario.name,
            "portfolio_pnl_dollars": result.portfolio_pnl,
            "drawdown_pct": drawdown_pct,
            "drawdown_limit_pct": max_drawdown_pct,
        },
    )


# ----------------------------------------------------------------------
# Gate 5b: Dealer regime (dossier soft-warn — R8 trigger 2)
# ----------------------------------------------------------------------
DealerRegime = Literal[
    "long_gamma_dampening",
    "short_gamma_amplifying",
    "near_flip",
    "neutral",
]


def check_dealer_regime(
    candidate_ticker: str,
    dealer_regime_by_ticker: dict[str, DealerRegime] | None,
) -> GateResult:
    """Soft-warn (R8 trigger 2) if the candidate's underlying is in
    ``short_gamma_amplifying`` regime per ``MarketStructure.regime``.

    Per Q1 of the #113 design checkpoint, R8 has two trigger
    conditions — stress drawdown (``check_stress_scenario``) and
    dealer regime (this function) — mirroring R6's "short-gamma +
    put-wall OR dealer-flip" pattern. The dossier reviewer fires R8
    if either gate returns ``passed=False``.

    When ``dealer_regime_by_ticker`` is None or the candidate's
    ticker is missing, this gate skips (``passed=True,
    reason="missing_data"``). Matches the Q3 D11 anti-pattern: don't
    fire R8 on absent evidence.

    Args:
        candidate_ticker: The ticker being evaluated.
        dealer_regime_by_ticker: Map of ticker →
            ``MarketStructure.regime``. Caller (dossier reviewer in
            Phase 3) obtains this from ``DealerPositioningAnalyzer``.
    """
    if dealer_regime_by_ticker is None or candidate_ticker not in dealer_regime_by_ticker:
        return GateResult(
            passed=True,
            reason="missing_data",
            details={
                "dealer_regime_check": "skipped",
                "skip_reason": "no_regime_data_for_ticker",
                "ticker": candidate_ticker,
            },
        )

    regime = dealer_regime_by_ticker[candidate_ticker]
    if regime == "short_gamma_amplifying":
        return GateResult(
            passed=False,
            reason="short_gamma_regime",
            details={"ticker": candidate_ticker, "dealer_regime": regime},
        )
    return GateResult(
        passed=True,
        reason=None,
        details={"ticker": candidate_ticker, "dealer_regime": regime},
    )
