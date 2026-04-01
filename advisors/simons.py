"""
Simons-Style Advisor Agent

Evaluates trades through the lens of:
- Statistical rigor and significance
- Signal validation and robustness
- Regime awareness
- Model limitations
- Data-driven decision making

"We're not trying to predict the future; we're trying to find statistically
significant patterns that have predictive value."
"""

from .base import BaseAdvisor
from .schema import (
    AdvisorInput,
    ConfidenceLevel,
    Judgment,
    RegimeType,
)


class SimonsAdvisor(BaseAdvisor):
    """
    Jim Simons / Renaissance-inspired quantitative advisor.

    Core principles:
    1. Trust data over narratives
    2. Statistical significance matters
    3. Models have regimes where they work and don't work
    4. Small edges compounded consistently beat big bets
    5. Know your model's limitations

    For options: validates statistical claims,
    checks robustness, and flags regime risks.
    """

    @property
    def name(self) -> str:
        return "Simons"

    @property
    def philosophy(self) -> str:
        return "Statistical rigor, signal validation, and regime awareness"

    @property
    def system_prompt(self) -> str:
        return """You are a quantitative investment advisor modeled after Jim Simons and Renaissance Technologies.

CORE BELIEFS:
- Markets are not random, but patterns are subtle and require rigorous methods to find
- A small edge, applied consistently with proper sizing, compounds to great returns
- Most "signals" are noise - statistical validation is non-negotiable
- The market changes; models that worked yesterday may not work today
- Never bet the firm on any single trade or strategy

EVALUATION FRAMEWORK:
1. Statistical Validity: Is the expected value statistically significant?
2. Sample Size: Is there enough data to trust the probability estimates?
3. Regime Fit: Does current market regime match where the model works?
4. Robustness: Does the signal persist across different time periods and conditions?
5. Model Limitations: What assumptions is the model making that might be wrong?

FOR OPTIONS EVALUATION:
- Are probability estimates from actual realized distributions or theoretical models?
- What's the standard error on the expected value?
- Has this strategy been profitable in similar market regimes historically?
- Is implied volatility pricing risk correctly, or is there systematic mispricing?
- What's the strategy's Sharpe ratio across multiple periods?

WHAT YOU REJECT:
- Narrative-driven trades without statistical backing
- Strategies that only work in backtests (overfit)
- Single data points or anecdotes as evidence
- Concentration that violates Kelly criterion
- Trading in regimes where the model has no edge

YOUR TONE:
- Precise and technical
- Data-focused
- Appropriately uncertain about model limitations
- Quantitative in all assessments"""

    def _analyze(self, input_data: AdvisorInput) -> dict:
        """Simons-style quantitative analysis."""
        trade = input_data.candidate_trade
        portfolio = input_data.portfolio
        market = input_data.market

        # Gather assessments
        position_size = self._assess_position_size(input_data)
        self._assess_concentration_risk(input_data)
        self._assess_iv_environment(input_data)
        prob = self._assess_probability_profile(input_data)

        key_reasons = []
        critical_questions = []
        hidden_risks = []
        would_approve_if = []
        would_reject_if = []

        # =====================================================================
        # STATISTICAL VALIDATION
        # =====================================================================

        # Expected value analysis
        ev = trade.expected_value

        # Standard error approximation (assuming binomial-like distribution)
        # For a single trade, SE is high
        implied_se = (prob["p_otm"] * (1 - prob["p_otm"])) ** 0.5 * 100
        ev_t_stat = ev / max(implied_se, 0.1)

        if ev > 0:
            if ev_t_stat > 2.0:
                key_reasons.append(
                    f"Expected value of {ev:.2f}% shows statistical significance (t-stat: {ev_t_stat:.2f})"
                )
            elif ev_t_stat > 1.0:
                key_reasons.append(
                    f"Expected value of {ev:.2f}% is positive but borderline significant (t-stat: {ev_t_stat:.2f})"
                )
            else:
                hidden_risks.append(
                    f"Expected value of {ev:.2f}% is within noise range (t-stat: {ev_t_stat:.2f})"
                )
        else:
            key_reasons.append(f"NEGATIVE expected value of {ev:.2f}% - no statistical edge exists")

        # Probability model validation
        critical_questions.append(
            f"Is the {prob['p_otm']:.0%} P(OTM) from realized distributions or Black-Scholes?"
        )

        if abs(trade.delta) > 0.30:
            hidden_risks.append(
                "Higher delta options have less reliable probability estimates - "
                "models break down near ATM"
            )

        # =====================================================================
        # REGIME ANALYSIS
        # =====================================================================

        regime = market.regime
        vix = market.vix
        vix_pct = market.vix_percentile

        # Define regime-strategy fit
        put_selling_regimes = {
            RegimeType.LOW_VOL: 0.7,  # OK but low premium
            RegimeType.NORMAL: 1.0,  # Ideal
            RegimeType.HIGH_VOL: 0.8,  # Good premium but risk
            RegimeType.CRISIS: 0.4,  # Dangerous
            RegimeType.TRENDING_UP: 0.9,  # Good for puts
            RegimeType.TRENDING_DOWN: 0.3,  # Bad for puts
        }

        regime_fit = put_selling_regimes.get(regime, 0.5)

        if regime_fit >= 0.8:
            key_reasons.append(
                f"Current regime ({regime.value}) has historically favorable conditions "
                f"for put selling (fit score: {regime_fit:.0%})"
            )
        elif regime_fit <= 0.5:
            hidden_risks.append(
                f"Current regime ({regime.value}) has poor historical fit for this strategy "
                f"(fit score: {regime_fit:.0%})"
            )
            critical_questions.append(
                "What's your evidence this strategy works in the current regime?"
            )

        # VIX analysis
        if vix_pct > 80:
            key_reasons.append(
                f"VIX at {vix:.1f} ({vix_pct:.0f}th percentile) - elevated premium opportunity "
                "BUT this often precedes larger moves"
            )
        elif vix_pct < 20:
            hidden_risks.append(
                f"VIX at {vix:.1f} ({vix_pct:.0f}th percentile) - compressed premium means poor risk/reward"
            )

        # =====================================================================
        # POSITION SIZING (KELLY CRITERION)
        # =====================================================================

        # Simplified Kelly: f* = (p * b - q) / b
        # where p = probability of win, q = 1-p, b = win/loss ratio
        p_win = prob["p_profit"]
        p_loss = 1 - p_win

        # Win amount = premium, Loss amount = strike - premium (for puts)
        win_amount = trade.premium * 100 * trade.contracts
        # Max loss (full assignment minus premium)
        max_loss = (trade.strike * 100 * trade.contracts) - win_amount

        if max_loss > 0:
            b_ratio = win_amount / max_loss
            kelly_fraction = (p_win * b_ratio - p_loss) / b_ratio

            # Actual fraction of portfolio
            actual_fraction = (trade.strike * 100 * trade.contracts) / portfolio.total_equity

            if kelly_fraction <= 0:
                hidden_risks.append(
                    "KELLY NEGATIVE: Mathematical expectation suggests no edge exists"
                )
            elif actual_fraction > kelly_fraction * 2:
                hidden_risks.append(
                    f"OVERSIZED: Position is {actual_fraction / kelly_fraction:.1f}x optimal Kelly fraction"
                )
            elif actual_fraction < kelly_fraction * 0.5:
                key_reasons.append(
                    f"Conservative sizing at {actual_fraction / kelly_fraction:.0%} of Kelly - reduces variance"
                )

        # =====================================================================
        # MODEL LIMITATIONS
        # =====================================================================

        hidden_risks.append(
            "Model assumes: log-normal returns, constant volatility, no jumps - "
            "all are violated in reality"
        )

        if trade.earnings_before_expiry:
            critical_questions.append(
                "Earnings create jump risk - is IV adequately pricing the gap risk?"
            )
            hidden_risks.append(
                "Jump diffusion models outperform Black-Scholes around earnings - "
                "standard probability estimates are unreliable"
            )

        # IV vs RV check
        critical_questions.append(
            "What's the historical IV-RV spread for this ticker? Is IV overpriced or underpriced?"
        )

        # =====================================================================
        # ROBUSTNESS CHECK
        # =====================================================================

        critical_questions.append(
            "Has this signal been profitable across multiple market regimes (2018, 2020, 2022)?"
        )

        if trade.dte < 21:
            hidden_risks.append(
                "Short DTE trades have higher variance in outcomes - requires larger sample size"
            )

        # =====================================================================
        # DETERMINE JUDGMENT
        # =====================================================================

        score = 0.0

        # EV contribution
        if ev > 2.0:
            score += 2.0
        elif ev > 1.0:
            score += 1.0
        elif ev > 0:
            score += 0.5
        else:
            score -= 2.0

        # Statistical significance
        if ev_t_stat > 2.0:
            score += 1.0
        elif ev_t_stat < 1.0:
            score -= 1.0

        # Regime fit
        score += (regime_fit - 0.5) * 2

        # Probability quality
        if prob["quality"] == "EXCELLENT":
            score += 1.0
        elif prob["quality"] == "GOOD":
            score += 0.5
        elif prob["quality"] == "POOR":
            score -= 1.5

        # Position sizing
        if "OVERSIZED" in str(hidden_risks):
            score -= 1.0
        if "Conservative sizing" in str(key_reasons):
            score += 0.5

        # Earnings risk
        if trade.earnings_before_expiry:
            score -= 0.5

        # Make judgment based on score
        if score >= 3.0:
            judgment = Judgment.STRONG_APPROVE
            judgment_summary = (
                f"Statistical edge confirmed (score: {score:.1f}). "
                "Probability estimates, regime fit, and sizing all pass validation."
            )
        elif score >= 1.5:
            judgment = Judgment.APPROVE
            judgment_summary = f"Positive expected value with acceptable statistical validity (score: {score:.1f})."
        elif score >= 0:
            judgment = Judgment.NEUTRAL
            judgment_summary = (
                f"Edge is marginal (score: {score:.1f}). "
                "Signal is not strong enough to confidently recommend."
            )
        elif score >= -1.5:
            judgment = Judgment.REJECT
            judgment_summary = (
                f"Insufficient statistical edge (score: {score:.1f}). "
                "Risk/reward does not meet quantitative thresholds."
            )
        else:
            judgment = Judgment.STRONG_REJECT
            judgment_summary = (
                f"No edge detected (score: {score:.1f}). "
                "This trade is indistinguishable from noise or has negative expectation."
            )

        # Build would_approve_if / would_reject_if
        would_approve_if.append("Expected value exceeds 2% with t-stat > 2.0")
        would_approve_if.append("Historical backtest shows profitability across 3+ regimes")
        would_approve_if.append("IV-RV spread confirms systematic overpricing of volatility")

        would_reject_if.append("Expected value within 1 standard error of zero")
        would_reject_if.append("Current regime has < 50% historical fit for strategy")
        would_reject_if.append("Position size exceeds 2x Kelly fraction")

        # Confidence
        if abs(score) > 2.5:
            confidence = ConfidenceLevel.HIGH
            confidence_explanation = "Strong signal in multiple quantitative dimensions"
        elif abs(score) > 1.0:
            confidence = ConfidenceLevel.MEDIUM
            confidence_explanation = "Signal present but not overwhelming - proceed with discipline"
        else:
            confidence = ConfidenceLevel.LOW
            confidence_explanation = (
                "Signal is weak and within noise range - any decision is essentially a coin flip"
            )

        # Ensure minimum key_reasons (validation requires at least 2)
        if len(key_reasons) < 2:
            key_reasons.append(
                f"Probability profile: {prob['quality']} (EV: {ev:.2f}%, P(OTM): {prob['p_otm']:.0%})"
            )
        if len(key_reasons) < 2:
            key_reasons.append(f"Position sizing assessment: {position_size}")

        return {
            "judgment": judgment,
            "judgment_summary": judgment_summary,
            "key_reasons": key_reasons[:5],
            "critical_questions": critical_questions[:4],
            "hidden_risks": hidden_risks[:4],
            "would_approve_if": would_approve_if[:3],
            "would_reject_if": would_reject_if[:3],
            "confidence": confidence,
            "confidence_explanation": confidence_explanation,
        }
