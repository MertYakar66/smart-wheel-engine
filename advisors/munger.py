"""
Munger-Style Advisor Agent

Evaluates trades through the lens of:
- Inversion (what kills this trade?)
- Cognitive bias detection
- Multidisciplinary mental models
- Second-order effects
- Intellectual honesty

"Invert, always invert."
"""

from .base import BaseAdvisor
from .schema import (
    AdvisorInput,
    ConfidenceLevel,
    Judgment,
)


class MungerAdvisor(BaseAdvisor):
    """
    Charlie Munger-inspired investment advisor.

    Core principles:
    1. Invert - figure out what would destroy the trade, then avoid it
    2. Recognize psychological biases that impair judgment
    3. Use multiple mental models from different disciplines
    4. Focus on what NOT to do
    5. Avoid cognitive overload and complexity

    For options: acts as the internal critic and
    intellectual opponent to every trade.
    """

    @property
    def name(self) -> str:
        return "Munger"

    @property
    def philosophy(self) -> str:
        return "Inversion, bias detection, and multidisciplinary thinking"

    @property
    def system_prompt(self) -> str:
        return """You are an investment advisor modeled after Charlie Munger's principles.

CORE BELIEFS:
- "All I want to know is where I'm going to die, so I'll never go there"
- Avoiding stupidity is easier than seeking brilliance
- The best way to get what you want is to deserve what you want
- Mental models from multiple disciplines compound understanding
- Most people are too confident in their own conclusions

EVALUATION FRAMEWORK (INVERSION):
1. What would make this trade a disaster?
2. What am I ignoring because I want this trade to work?
3. What's the second-order effect I'm not seeing?
4. What cognitive bias might be driving this decision?
5. If this trade fails, what will I wish I had considered?

COGNITIVE BIASES TO CHECK:
- Confirmation bias: Am I only seeing evidence that supports the trade?
- Recency bias: Am I overweighting recent market behavior?
- Overconfidence: Am I too certain about probability estimates?
- Social proof: Am I trading this because others are?
- Commitment bias: Am I holding because I don't want to admit error?
- Incentive bias: Is someone benefiting from me making this trade?
- Availability bias: Am I overweighting easily recalled information?

MENTAL MODELS TO APPLY:
- Second-order thinking: What happens after the obvious outcome?
- Margin of safety: What's the downside if I'm wrong?
- Opportunity cost: What else could this capital do?
- Incentive structures: Who benefits and how?
- Mean reversion: Is this situation extreme and likely to reverse?

YOUR ROLE:
- Internal critic and devil's advocate
- Identify what others are missing
- Challenge assumptions ruthlessly
- Prevent costly mistakes

YOUR TONE:
- Direct and unsparing
- Intellectually rigorous
- Skeptical by default
- Occasionally witty/sardonic"""

    # Common cognitive biases with descriptions
    BIASES = {
        "confirmation": "Seeking information that confirms existing beliefs",
        "recency": "Overweighting recent events in predictions",
        "overconfidence": "Excessive certainty in probability estimates",
        "anchoring": "Over-relying on first piece of information",
        "availability": "Judging probability by ease of recall",
        "loss_aversion": "Feeling losses more than equivalent gains",
        "sunk_cost": "Continuing because of past investment, not future value",
        "social_proof": "Following the crowd without independent analysis",
        "narrative": "Preferring stories over statistics",
        "hindsight": "Believing past events were predictable",
    }

    def _analyze(self, input_data: AdvisorInput) -> dict:
        """Munger-style inversion analysis."""
        trade = input_data.candidate_trade
        market = input_data.market

        # Gather assessments
        position_size = self._assess_position_size(input_data)
        concentration = self._assess_concentration_risk(input_data)
        earnings = self._assess_earnings_risk(input_data)
        iv_env = self._assess_iv_environment(input_data)
        prob = self._assess_probability_profile(input_data)

        key_reasons = []
        critical_questions = []
        hidden_risks = []
        would_approve_if = []
        would_reject_if = []

        detected_biases = []

        # =====================================================================
        # INVERSION: WHAT KILLS THIS TRADE?
        # =====================================================================

        # Scenario 1: Maximum adverse move
        max_loss_pct = (trade.strike / trade.underlying_price - 1) * 100
        key_reasons.append(
            f"INVERSION: If {trade.ticker} drops to ${trade.strike}, "
            f"you'll be assigned at a {abs(max_loss_pct):.1f}% premium to current price"
        )

        # Scenario 2: What if volatility is wrong?
        if iv_env["level"] in ["VERY_HIGH", "HIGH"]:
            hidden_risks.append(
                "Elevated IV suggests market expects movement - "
                "what do they know that you don't?"
            )
            detected_biases.append(("overconfidence",
                "You may be overconfident that IV will stay elevated or revert favorably"))

        # Scenario 3: Concentration death spiral
        if concentration["high_concentration"]:
            hidden_risks.append(
                f"DANGER: {concentration['sector']} concentration at {concentration['current_allocation']:.0f}%. "
                "A sector-wide event would compound losses across positions"
            )

        # =====================================================================
        # COGNITIVE BIAS DETECTION
        # =====================================================================

        # Check for recency bias
        if market.regime.value in ["trending_up", "low_volatility"]:
            detected_biases.append(("recency",
                "Recent calm market may be creating false confidence"))

        # Check for premium chasing (availability bias)
        if trade.premium > 10 and trade.delta < -0.25:
            detected_biases.append(("availability",
                "High premium is salient - are you properly weighing the assignment risk?"))

        # Check for social proof
        popular_tickers = {"NVDA", "TSLA", "AMD", "AAPL", "SPY", "QQQ"}
        if trade.ticker in popular_tickers:
            detected_biases.append(("social_proof",
                f"{trade.ticker} is heavily discussed - ensure this is YOUR analysis, not the crowd's"))

        # Check for overconfidence in probability
        if prob["p_otm"] > 0.70:
            detected_biases.append(("overconfidence",
                f"P(OTM) of {prob['p_otm']:.0%} sounds comfortable - "
                "but models are wrong most when they're most confident"))

        # Add bias warnings to key_reasons
        for bias_name, description in detected_biases[:3]:
            key_reasons.append(f"BIAS ALERT ({bias_name}): {description}")

        # =====================================================================
        # SECOND-ORDER THINKING
        # =====================================================================

        critical_questions.append(
            "SECOND-ORDER: If you get assigned, what happens to your portfolio beta and concentration?"
        )

        critical_questions.append(
            "INVERSION: What would make you look back on this trade as obviously stupid?"
        )

        if earnings["has_earnings"]:
            critical_questions.append(
                "The market knows earnings are coming - what edge do you have in pricing this risk?"
            )

        # Opportunity cost
        critical_questions.append(
            f"What else could ${trade.strike * 100 * trade.contracts:,.0f} of buying power do?"
        )

        # =====================================================================
        # MENTAL MODEL APPLICATION
        # =====================================================================

        # Mean reversion check
        if iv_env["iv_rank"] > 80:
            hidden_risks.append(
                "MEAN REVERSION: IV rank above 80 eventually reverts - "
                "timing this reversion is where premium sellers get hurt"
            )

        # Incentive check
        hidden_risks.append(
            "INCENTIVES: Brokers profit from activity. "
            "Is this trade genuinely better than doing nothing?"
        )

        # Margin of safety
        if trade.dte < 30 and trade.delta < -0.30:
            hidden_risks.append(
                "SHORT DTE + HIGHER DELTA = Limited time for recovery if things go wrong"
            )

        # =====================================================================
        # DETERMINE JUDGMENT
        # =====================================================================

        # Munger is naturally skeptical - starts from rejection
        red_flags = 0
        mitigating_factors = 0

        # Count red flags
        if len(detected_biases) >= 2:
            red_flags += 1
        if concentration["warning"]:
            red_flags += 2
        if earnings["has_earnings"] and trade.dte < 30:
            red_flags += 1
        if iv_env["level"] in ["VERY_HIGH", "HIGH"] and market.regime.value == "crisis":
            red_flags += 1  # Selling vol into crisis
        if prob["quality"] in ["POOR", "MARGINAL"]:
            red_flags += 1

        # Count mitigating factors
        if prob["quality"] in ["EXCELLENT", "GOOD"]:
            mitigating_factors += 1
        if trade.delta > -0.20:  # Low delta
            mitigating_factors += 1
        if position_size.startswith("SMALL"):
            mitigating_factors += 1
        if not concentration["high_concentration"]:
            mitigating_factors += 1

        # Make judgment
        if red_flags >= 4:
            judgment = Judgment.STRONG_REJECT
            judgment_summary = (
                "Multiple cognitive biases and structural risks present. "
                "This is exactly the kind of trade that looks good until it doesn't."
            )
        elif red_flags >= 2 and mitigating_factors < 2:
            judgment = Judgment.REJECT
            judgment_summary = (
                "Inverting this trade reveals more ways to lose than the premium justifies."
            )
        elif mitigating_factors >= 3 and red_flags <= 1:
            judgment = Judgment.APPROVE
            judgment_summary = (
                "Survived inversion analysis. Risks are acknowledged and sized appropriately."
            )
        elif mitigating_factors >= 4 and red_flags == 0:
            judgment = Judgment.STRONG_APPROVE
            judgment_summary = (
                "Rare trade that passes all filters. Proceed with discipline."
            )
        else:
            judgment = Judgment.NEUTRAL
            judgment_summary = (
                "Neither obviously good nor obviously bad. "
                "The absence of red flags isn't the same as the presence of edge."
            )

        # Build would_approve_if / would_reject_if
        would_approve_if.append("You can articulate exactly what would make this trade fail")
        would_approve_if.append("Position size is small enough that maximum loss doesn't matter")
        would_approve_if.append("You've identified and rejected at least 2 cognitive biases")

        would_reject_if.append("You can't explain this trade to a skeptic in 30 seconds")
        would_reject_if.append("The trade relies on something you can't verify continuing")
        would_reject_if.append("You're trading this because you saw someone else do it")

        # Confidence
        confidence = ConfidenceLevel.HIGH
        confidence_explanation = (
            "Inversion analysis is reliable - it finds what's wrong, not what's right. "
            "The more red flags, the higher my confidence in rejection."
        )

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
