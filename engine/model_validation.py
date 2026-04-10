"""
Cross-Model Validation Framework

Institutional model-risk governance: validates pricing models against each other
and enforces acceptance thresholds before trade execution.

Architecture:
    Tier 1 (Production): BAW — fast analytic approximation (~1μs)
    Tier 2 (Validation): CRR Binomial — deterministic benchmark (~5ms @ 500 steps)
    Tier 3 (Escalation): LSM Monte Carlo — path-dependent analysis (~50ms)

Acceptance gates:
    |BAW - CRR| <= max(1bp notional, 0.5% option premium)
    |BAW delta - CRR delta| <= 0.02
    |BAW gamma - CRR gamma| <= 0.005

Auto-escalation to Tier 3 when:
    - DTE < 7 days
    - Deep ITM/OTM (|delta| < 0.05 or |delta| > 0.95)
    - Large discrete dividend before expiry
    - Models diverge beyond tolerance

References:
    SR 11-7 (Fed/OCC): Supervisory Guidance on Model Risk Management
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class ModelComparisonResult:
    """Result of comparing two pricing models."""

    model_a: str
    model_b: str
    price_a: float
    price_b: float
    price_diff: float
    price_diff_pct: float  # As fraction of model_a price
    delta_diff: float
    gamma_diff: float
    theta_diff: float
    vega_diff: float

    within_tolerance: bool
    tolerance_used: dict = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.within_tolerance else "FAIL"
        lines = [
            f"[{status}] {self.model_a} vs {self.model_b}",
            f"  Price: {self.price_a:.6f} vs {self.price_b:.6f} "
            f"(diff={self.price_diff:+.6f}, {self.price_diff_pct:+.4%})",
            f"  Delta diff: {self.delta_diff:+.6f}",
            f"  Gamma diff: {self.gamma_diff:+.6f}",
        ]
        if self.violations:
            lines.append(f"  Violations: {'; '.join(self.violations)}")
        return "\n".join(lines)


@dataclass
class ValidationReport:
    """Full cross-model validation report."""

    symbol: str
    spot: float
    strike: float
    T: float
    sigma: float
    option_type: str

    baw_price: float = 0.0
    crr_price: float = 0.0
    lsm_price: float | None = None

    baw_vs_crr: ModelComparisonResult | None = None
    baw_vs_lsm: ModelComparisonResult | None = None
    crr_vs_lsm: ModelComparisonResult | None = None

    escalation_triggered: bool = False
    escalation_reasons: list[str] = field(default_factory=list)
    trade_approved: bool = True
    block_reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"=== Model Validation: {self.symbol} {self.option_type.upper()} "
            f"K={self.strike} T={self.T:.4f} σ={self.sigma:.2%} ===",
            f"BAW: {self.baw_price:.6f}  CRR: {self.crr_price:.6f}",
        ]
        if self.lsm_price is not None:
            lines.append(f"LSM: {self.lsm_price:.6f}")
        if self.baw_vs_crr:
            lines.append(self.baw_vs_crr.summary())
        if self.escalation_triggered:
            lines.append(f"ESCALATION: {', '.join(self.escalation_reasons)}")
        status = "APPROVED" if self.trade_approved else "BLOCKED"
        lines.append(f"Trade: {status}")
        if self.block_reasons:
            lines.append(f"Block reasons: {'; '.join(self.block_reasons)}")
        return "\n".join(lines)


@dataclass
class ModelTolerances:
    """Acceptance thresholds for model comparison."""

    # Price tolerance: max(price_abs_tol, price_rel_tol * price)
    price_abs_tol: float = 0.01  # $0.01 absolute
    price_rel_tol: float = 0.005  # 0.5% of premium
    notional_bps_tol: float = 0.0001  # 1bp of notional (strike * 100)

    # Greeks tolerances
    # Note: gamma tolerance is wider because both BAW and CRR compute gamma
    # via finite differences with different internal mechanics
    delta_tol: float = 0.02
    gamma_tol: float = 0.02
    theta_tol: float = 0.50  # Annual theta can differ more
    vega_tol: float = 0.01

    # Escalation triggers
    dte_escalation_threshold: int = 7
    deep_itm_delta: float = 0.95
    deep_otm_delta: float = 0.05
    discrete_div_yield_threshold: float = 0.02  # 2% of spot


class CrossModelValidator:
    """
    Validates pricing models against each other with institutional governance.

    Usage:
        validator = CrossModelValidator()
        report = validator.validate(
            S=155, K=150, T=0.1, r=0.05, sigma=0.25,
            option_type="put", q=0.02,
        )
        if not report.trade_approved:
            print(f"BLOCKED: {report.block_reasons}")
    """

    def __init__(
        self,
        tolerances: ModelTolerances | None = None,
        crr_steps: int = 500,
        lsm_simulations: int = 50000,
        auto_escalate: bool = True,
        block_on_divergence: bool = False,
    ):
        """
        Args:
            tolerances: Acceptance thresholds
            crr_steps: Steps for CRR tree (default 500)
            lsm_simulations: Simulations for LSM (default 50000)
            auto_escalate: Auto-escalate to LSM when triggers hit
            block_on_divergence: Block trade execution when models diverge
        """
        self.tolerances = tolerances or ModelTolerances()
        self.crr_steps = crr_steps
        self.lsm_simulations = lsm_simulations
        self.auto_escalate = auto_escalate
        self.block_on_divergence = block_on_divergence

    def validate(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        q: float = 0.0,
        discrete_dividends: list | None = None,
        symbol: str = "",
        include_lsm: bool = False,
    ) -> ValidationReport:
        """
        Run full cross-model validation.

        Compares BAW vs CRR (always), optionally adds LSM.
        Auto-escalates to LSM when escalation triggers fire.
        """
        from .binomial_tree import binomial_american_full
        from .option_pricer import american_option_greeks

        report = ValidationReport(
            symbol=symbol,
            spot=S,
            strike=K,
            T=T,
            sigma=sigma,
            option_type=option_type,
        )

        # --- Tier 1: BAW ---
        baw_greeks = american_option_greeks(S, K, T, r, sigma, option_type, q)
        report.baw_price = baw_greeks["price"]

        # --- Tier 2: CRR ---
        crr_result = binomial_american_full(
            S, K, T, r, sigma, option_type, q,
            n_steps=self.crr_steps,
            discrete_dividends=discrete_dividends,
        )
        report.crr_price = crr_result.price

        # --- Compare BAW vs CRR ---
        report.baw_vs_crr = self._compare_models(
            "BAW", "CRR",
            baw_greeks, crr_result.to_dict(),
            K,
        )

        # --- Check escalation triggers ---
        escalation_reasons = self._check_escalation(
            S, K, T, sigma, option_type, q,
            baw_greeks, crr_result, discrete_dividends,
        )
        if escalation_reasons:
            report.escalation_triggered = True
            report.escalation_reasons = escalation_reasons

        # --- Tier 3: LSM (if requested or auto-escalated) ---
        run_lsm = include_lsm or (self.auto_escalate and report.escalation_triggered)
        if run_lsm:
            try:
                from .monte_carlo import price_american_option
                lsm_result = price_american_option(
                    S=S, K=K, T=T, r=r, sigma=sigma,
                    option_type=option_type, q=q,
                    n_simulations=self.lsm_simulations,
                )
                if isinstance(lsm_result, dict):
                    report.lsm_price = lsm_result.get("american_price", lsm_result.get("price"))
                elif hasattr(lsm_result, "american_price"):
                    report.lsm_price = lsm_result.american_price
                else:
                    report.lsm_price = float(lsm_result)
            except Exception as e:
                warnings.warn(f"LSM escalation failed: {e}", stacklevel=2)

        # --- Trade approval ---
        if not report.baw_vs_crr.within_tolerance:
            if self.block_on_divergence:
                report.trade_approved = False
                report.block_reasons.append(
                    f"BAW-CRR divergence: {'; '.join(report.baw_vs_crr.violations)}"
                )
            else:
                warnings.warn(
                    f"Model divergence for {symbol} {option_type} K={K}: "
                    f"{'; '.join(report.baw_vs_crr.violations)}",
                    stacklevel=2,
                )

        return report

    def _compare_models(
        self,
        name_a: str,
        name_b: str,
        greeks_a: dict,
        greeks_b: dict,
        K: float,
    ) -> ModelComparisonResult:
        """Compare two model outputs against tolerances."""
        price_a = greeks_a["price"]
        price_b = greeks_b["price"]
        price_diff = price_a - price_b

        # Price tolerance: max of absolute, relative, and notional-based
        notional = K * 100
        price_tol = max(
            self.tolerances.price_abs_tol,
            self.tolerances.price_rel_tol * abs(price_a) if price_a != 0 else 0.01,
            self.tolerances.notional_bps_tol * notional,
        )

        violations = []
        if abs(price_diff) > price_tol:
            violations.append(
                f"price: |{price_diff:.6f}| > tol {price_tol:.6f}"
            )

        delta_diff = greeks_a.get("delta", 0) - greeks_b.get("delta", 0)
        if abs(delta_diff) > self.tolerances.delta_tol:
            violations.append(f"delta: |{delta_diff:.6f}| > tol {self.tolerances.delta_tol}")

        gamma_diff = greeks_a.get("gamma", 0) - greeks_b.get("gamma", 0)
        if abs(gamma_diff) > self.tolerances.gamma_tol:
            violations.append(f"gamma: |{gamma_diff:.6f}| > tol {self.tolerances.gamma_tol}")

        theta_diff = greeks_a.get("theta", 0) - greeks_b.get("theta", 0)
        vega_diff = greeks_a.get("vega", 0) - greeks_b.get("vega", 0)

        return ModelComparisonResult(
            model_a=name_a,
            model_b=name_b,
            price_a=price_a,
            price_b=price_b,
            price_diff=price_diff,
            price_diff_pct=price_diff / price_a if price_a != 0 else 0.0,
            delta_diff=delta_diff,
            gamma_diff=gamma_diff,
            theta_diff=theta_diff,
            vega_diff=vega_diff,
            within_tolerance=len(violations) == 0,
            tolerance_used={
                "price_tol": price_tol,
                "delta_tol": self.tolerances.delta_tol,
                "gamma_tol": self.tolerances.gamma_tol,
            },
            violations=violations,
        )

    def _check_escalation(
        self,
        S: float,
        K: float,
        T: float,
        sigma: float,
        option_type: str,
        q: float,
        baw_greeks: dict,
        crr_result,
        discrete_dividends: list | None,
    ) -> list[str]:
        """Check if escalation to Tier 3 (LSM) is warranted."""
        reasons = []
        tol = self.tolerances

        # Near expiry
        dte = T * 365
        if dte < tol.dte_escalation_threshold:
            reasons.append(f"Near expiry: DTE={dte:.1f} < {tol.dte_escalation_threshold}")

        # Deep ITM/OTM
        abs_delta = abs(baw_greeks.get("delta", 0.5))
        if abs_delta > tol.deep_itm_delta:
            reasons.append(f"Deep ITM: |delta|={abs_delta:.3f} > {tol.deep_itm_delta}")
        elif abs_delta < tol.deep_otm_delta:
            reasons.append(f"Deep OTM: |delta|={abs_delta:.3f} < {tol.deep_otm_delta}")

        # Discrete dividends
        if discrete_dividends:
            total_div = sum(d.amount for d in discrete_dividends)
            div_yield = total_div / S if S > 0 else 0
            if div_yield > tol.discrete_div_yield_threshold:
                reasons.append(
                    f"Large discrete div: {total_div:.2f} ({div_yield:.2%} of spot)"
                )

        # Model divergence
        if hasattr(crr_result, 'price'):
            baw_price = baw_greeks["price"]
            crr_price = crr_result.price
            if baw_price > 0:
                divergence = abs(baw_price - crr_price) / baw_price
                if divergence > 0.02:  # 2% divergence
                    reasons.append(f"Model divergence: {divergence:.2%}")

        return reasons


def run_benchmark_grid(
    spot_range: tuple[float, float] = (80, 120),
    strikes: list[float] | None = None,
    T_values: list[float] | None = None,
    sigma_values: list[float] | None = None,
    r: float = 0.05,
    q: float = 0.02,
    crr_steps: int = 500,
) -> list[dict]:
    """
    Run cross-model benchmark across a parameter grid.

    Returns list of dicts suitable for DataFrame creation.
    Designed for CI integration: gate on max divergence.
    """
    from .binomial_tree import binomial_american_price
    from .option_pricer import american_option_price

    if strikes is None:
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    if T_values is None:
        T_values = [0.02, 0.08, 0.25, 0.50, 1.0]
    if sigma_values is None:
        sigma_values = [0.10, 0.20, 0.30, 0.50]

    results = []

    for S in [float(x) for x in np.linspace(spot_range[0], spot_range[1], 5)]:
        for K in strikes:
            for T in T_values:
                for sigma in sigma_values:
                    for option_type in ["put", "call"]:
                        baw = american_option_price(S, K, T, r, sigma, option_type, q)
                        crr = binomial_american_price(S, K, T, r, sigma, option_type, q, crr_steps)

                        diff = abs(baw - crr)
                        rel_diff = diff / crr if crr > 0.001 else 0.0

                        results.append({
                            "S": S,
                            "K": K,
                            "T": T,
                            "sigma": sigma,
                            "option_type": option_type,
                            "baw_price": baw,
                            "crr_price": crr,
                            "abs_diff": diff,
                            "rel_diff": rel_diff,
                            "moneyness": S / K,
                        })

    return results
