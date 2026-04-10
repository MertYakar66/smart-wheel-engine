"""
Centralized policy-driven configuration for risk, signal, and advisor thresholds.

Replaces hardcoded defaults scattered throughout the engine with a single,
auditable source of truth.  All policy knobs live here; individual modules
should accept a ``TradingPolicyConfig`` (or the relevant sub-config) rather
than defining their own magic numbers.

Usage::

    from engine.policy_config import load_policy

    policy = load_policy()                       # built-in defaults
    policy = load_policy("prod_policy.json")     # override from file
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RiskPolicyConfig:
    """Risk-management policy knobs.

    Attributes:
        var_confidence: Confidence level for Value-at-Risk (e.g. 0.95 = 95%).
        max_drawdown_pct: Maximum tolerated portfolio drawdown (0-1).
        max_portfolio_delta: Net delta limit as fraction of portfolio value.
        max_daily_loss_pct: Daily loss circuit-breaker (0-1).
        max_var_pct: Maximum acceptable daily VaR as fraction of NAV.
        heuristic_fallback_enabled: When True, allow parametric (delta-gamma-
            vega) VaR even for concentrated books.  When False, concentrated
            books must use historical or Monte-Carlo VaR.
        concentrated_book_threshold: Number of positions at or below which a
            book is considered "concentrated" and heuristic VaR is blocked
            (unless *heuristic_fallback_enabled* is True).
    """

    var_confidence: float = 0.95
    max_drawdown_pct: float = 0.20
    max_portfolio_delta: float = 0.50
    max_daily_loss_pct: float = 0.03
    max_var_pct: float = 0.05
    heuristic_fallback_enabled: bool = False
    concentrated_book_threshold: int = 5


@dataclass
class SignalPolicyConfig:
    """Signal-generation policy knobs.

    Attributes:
        iv_rank_high: IV rank above which conditions are *favorable* for
            premium selling (0-1).
        iv_rank_low: IV rank below which conditions are *unfavorable* (0-1).
        trend_lookback_days: Number of trading days used for trend detection.
        profit_target_pct: Close position when this fraction of max profit is
            captured (0-1, e.g. 0.50 = 50%).
        stop_loss_multiplier: Stop when loss reaches this multiple of premium
            received (e.g. 2.0 = 200%).
        target_dte_min: Minimum acceptable DTE for new positions.
        target_dte_max: Maximum acceptable DTE for new positions.
        target_dte_ideal: Preferred DTE when multiple expirations qualify.
        earnings_buffer_days: Avoid new positions if earnings are within this
            many calendar days.
        fomc_buffer_days: Avoid new positions if FOMC is within this many days.
    """

    iv_rank_high: float = 0.50
    iv_rank_low: float = 0.20
    trend_lookback_days: int = 20
    profit_target_pct: float = 0.50
    stop_loss_multiplier: float = 2.0
    target_dte_min: int = 25
    target_dte_max: int = 45
    target_dte_ideal: int = 35
    earnings_buffer_days: int = 5
    fomc_buffer_days: int = 2


@dataclass
class AdvisorPolicyConfig:
    """Advisor-committee policy knobs.

    Attributes:
        committee_weights: Mapping of advisor name to vote weight.  Weights
            need not sum to 1 -- they are normalised at runtime.
        calibration_drift_threshold: Maximum acceptable drift (L1 norm on
            weight vector) before an automatic rebalance is triggered.
        rebalance_frequency_days: How often (calendar days) the committee
            weights are recalibrated, independent of drift.
    """

    committee_weights: dict[str, float] = field(default_factory=lambda: {
        "regime": 1.0,
        "volatility": 1.0,
        "event": 1.0,
        "technical": 1.0,
    })
    calibration_drift_threshold: float = 0.15
    rebalance_frequency_days: int = 30


@dataclass
class GreeksPolicyConfig:
    """Greeks unit-convention and decomposition policy.

    Keeping these in one place prevents the annual-vs-daily theta bugs and
    the per-1%-vs-per-100% vega confusion flagged in past audits.

    Attributes:
        theta_annual: If True the pricer returns *annualised* theta and
            consumers must divide by 365 for daily P&L.
        vega_per_vol_point: If True vega is quoted per 1 percentage-point
            (1 vol point) change in implied volatility.
        decomposition_residual_tolerance: Maximum acceptable unexplained
            fraction when decomposing P&L into Greek components (0-1,
            e.g. 0.10 = 10%).
    """

    theta_annual: bool = True
    vega_per_vol_point: bool = True
    decomposition_residual_tolerance: float = 0.10


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TradingPolicyConfig:
    """Root policy object aggregating all sub-policies.

    Instantiate with defaults::

        cfg = TradingPolicyConfig()

    Or load from JSON::

        cfg = load_policy("my_policy.json")
    """

    risk: RiskPolicyConfig = field(default_factory=RiskPolicyConfig)
    signal: SignalPolicyConfig = field(default_factory=SignalPolicyConfig)
    advisor: AdvisorPolicyConfig = field(default_factory=AdvisorPolicyConfig)
    greeks: GreeksPolicyConfig = field(default_factory=GreeksPolicyConfig)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_policy(path: str | None = None) -> TradingPolicyConfig:
    """Load a trading policy from a JSON file, or return built-in defaults.

    Args:
        path: Filesystem path to a JSON policy file.  When *None* (the
            default), a ``TradingPolicyConfig`` with all default values is
            returned.

    Returns:
        A fully-populated ``TradingPolicyConfig``.

    Raises:
        FileNotFoundError: If *path* is given but does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if path is None:
        config = TradingPolicyConfig()
    else:
        with open(path) as fh:
            raw: dict = json.load(fh)

        config = TradingPolicyConfig(
            risk=RiskPolicyConfig(**raw.get("risk", {})),
            signal=SignalPolicyConfig(**raw.get("signal", {})),
            advisor=AdvisorPolicyConfig(**raw.get("advisor", {})),
            greeks=GreeksPolicyConfig(**raw.get("greeks", {})),
        )

    # Fail fast on invalid policy — prevents silent misconfiguration
    errors = validate_policy(config)
    if errors:
        raise ValueError(
            f"Policy validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return config


def save_policy(config: TradingPolicyConfig, path: str) -> None:
    """Serialise a ``TradingPolicyConfig`` to a JSON file.

    Args:
        config: The policy to persist.
        path: Destination file path (will be created or overwritten).
    """
    with open(path, "w") as fh:
        json.dump(asdict(config), fh, indent=2, sort_keys=True)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_policy(config: TradingPolicyConfig) -> list[str]:
    """Check a policy for invalid or contradictory values.

    Returns:
        A list of human-readable error strings.  An empty list means the
        policy is valid.
    """
    errors: list[str] = []

    # -- Risk --
    r = config.risk
    if not (0.90 <= r.var_confidence <= 0.999):
        errors.append(
            f"risk.var_confidence={r.var_confidence} outside [0.90, 0.999]"
        )
    if not (0.0 < r.max_drawdown_pct < 1.0):
        errors.append(
            f"risk.max_drawdown_pct={r.max_drawdown_pct} must be in (0, 1)"
        )
    if not (0.0 < r.max_daily_loss_pct < r.max_drawdown_pct):
        errors.append(
            "risk.max_daily_loss_pct must be positive and less than "
            "risk.max_drawdown_pct"
        )
    if r.concentrated_book_threshold < 1:
        errors.append(
            "risk.concentrated_book_threshold must be >= 1"
        )
    if not (0.0 < r.max_var_pct < 1.0):
        errors.append(
            f"risk.max_var_pct={r.max_var_pct} must be in (0, 1)"
        )

    # -- Signal --
    s = config.signal
    if not (0.0 < s.iv_rank_low < s.iv_rank_high < 1.0):
        errors.append(
            "signal: need 0 < iv_rank_low < iv_rank_high < 1"
        )
    if s.trend_lookback_days < 5:
        errors.append(
            f"signal.trend_lookback_days={s.trend_lookback_days} too short "
            "(min 5)"
        )
    if not (0.0 < s.profit_target_pct < 1.0):
        errors.append(
            f"signal.profit_target_pct={s.profit_target_pct} must be in (0, 1)"
        )
    if s.stop_loss_multiplier <= 1.0:
        errors.append(
            f"signal.stop_loss_multiplier={s.stop_loss_multiplier} must be > 1"
        )
    if not (s.target_dte_min < s.target_dte_ideal < s.target_dte_max):
        errors.append(
            "signal: need target_dte_min < target_dte_ideal < target_dte_max"
        )
    if s.earnings_buffer_days < 0:
        errors.append("signal.earnings_buffer_days must be >= 0")

    # -- Advisor --
    a = config.advisor
    if not a.committee_weights:
        errors.append("advisor.committee_weights must not be empty")
    if any(w <= 0 for w in a.committee_weights.values()):
        errors.append("advisor.committee_weights must all be positive")
    if not (0.0 < a.calibration_drift_threshold < 1.0):
        errors.append(
            f"advisor.calibration_drift_threshold="
            f"{a.calibration_drift_threshold} must be in (0, 1)"
        )
    if a.rebalance_frequency_days < 1:
        errors.append("advisor.rebalance_frequency_days must be >= 1")

    # -- Greeks --
    g = config.greeks
    if not (0.0 < g.decomposition_residual_tolerance <= 1.0):
        errors.append(
            f"greeks.decomposition_residual_tolerance="
            f"{g.decomposition_residual_tolerance} must be in (0, 1]"
        )

    return errors
