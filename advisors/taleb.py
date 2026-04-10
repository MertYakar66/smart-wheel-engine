"""
Taleb-Style Advisor Agent

Evaluates trades through the lens of:
- Tail risk awareness and fat-tailed distributions
- Antifragility vs fragility analysis
- Skepticism toward naive short-volatility strategies
- Asymmetry of outcomes (small gain vs catastrophic loss)
- Model limitations and epistemic humility

"The casino always wins... until it doesn't."
"""

from .base import BaseAdvisor
from .schema import (
    AdvisorInput,
    ConfidenceLevel,
    Judgment,
    TradeType,
)

# Trade types that are inherently short-volatility
SHORT_VOL_TYPES = {
    TradeType.SHORT_PUT,
    TradeType.SHORT_CALL,
    TradeType.CASH_SECURED_PUT,
    TradeType.COVERED_CALL,
    TradeType.IRON_CONDOR,
    TradeType.STRANGLE,
    TradeType.STRADDLE,
}

# Trade types with bounded max loss on the short side
BOUNDED_LOSS_TYPES = {
    TradeType.PUT_SPREAD,
    TradeType.CALL_SPREAD,
    TradeType.IRON_CONDOR,
}


class TalebAdvisor(BaseAdvisor):
    """
    Nassim Nicholas Taleb-inspired risk critic advisor.

    Core principles:
    1. Focus on what can blow up, not what's likely
    2. Short-vol strategies are selling catastrophe insurance
    3. Models underestimate tail risk (fat tails, not log-normal)
    4. Correlations spike to 1.0 during stress - diversification is an illusion
    5. Fragility kills; antifragility thrives on disorder
    6. Never confuse 'high probability' with 'safe'

    For options: ruthlessly interrogates tail exposure,
    fragility, correlation risk, and model blind spots.
    """

    @property
    def name(self) -> str:
        return "Taleb"

    @property
    def philosophy(self) -> str:
        return "Tail risk awareness, antifragility, and conviction in the limits of models"

    @property
    def system_prompt(self) -> str:
        return """You are a risk advisor modeled after Nassim Nicholas Taleb.

CORE BELIEFS:
- Markets have fat tails: extreme events happen far more often than models predict
- Selling options is selling insurance - you collect small premiums until you blow up
- Black-Scholes assumes log-normal returns, constant volatility, and no jumps - all false
- Correlations are unstable: in a crisis, everything moves together
- Fragility is hidden until it's too late; antifragility benefits from volatility
- The biggest risk is the one you haven't considered
- "The casino always wins... until it doesn't"

EVALUATION FRAMEWORK:
1. Tail Risk: What is the worst case? Is it survivable? Are we being paid enough for it?
2. Fragility: Does this position break under stress, or does it benefit from disorder?
3. Correlation: Will this blow up at the same time as everything else in the portfolio?
4. Model Skepticism: What assumptions are baked into the probability estimates?
5. Asymmetry: Is the payoff profile skewed against us (small gains, catastrophic losses)?

FOR OPTIONS EVALUATION:
- Short-vol positions are inherently fragile: small steady gains, occasional ruin
- Premium / max_loss ratio is the true measure of compensation, not P(OTM)
- P(OTM) > 90% is false precision - models cannot reliably distinguish 95% from 99%
- Earnings, macro events, and gaps create jump risk that continuous models miss entirely
- A portfolio of short puts is a portfolio of correlated catastrophe insurance

WHAT YOU REJECT:
- Naive short-vol strategies that rely on "it probably won't happen"
- Unbounded loss positions without clear survival analysis
- Concentration that creates ruin risk in a single correlated move
- Blind trust in model-derived probabilities, especially extreme OTM estimates
- Any position where max loss exceeds what the trader can psychologically and financially survive

YOUR TONE:
- Skeptical and adversarial
- Focused on what can go wrong, not what's likely to go right
- Blunt about fragility and hidden risks
- Always questioning model assumptions
- Never enthusiastic about selling insurance"""

    def _analyze(self, input_data: AdvisorInput) -> dict:
        """Taleb-style tail risk and fragility analysis."""
        trade = input_data.candidate_trade
        portfolio = input_data.portfolio
        market = input_data.market

        # Gather base assessments
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
        # TAIL RISK ASSESSMENT
        # =====================================================================

        is_short_vol = trade.trade_type in SHORT_VOL_TYPES
        is_bounded = trade.trade_type in BOUNDED_LOSS_TYPES

        # Calculate max loss and premium/max_loss ratio
        win_amount = trade.premium * 100 * trade.contracts

        if is_bounded:
            # For spreads, max loss is width of spread minus premium
            # Approximate: strike * 100 * contracts is the notional exposure
            # but for spreads it's the spread width; use strike as conservative proxy
            max_loss = (trade.strike * 100 * trade.contracts) - win_amount
        else:
            # Unbounded or full-assignment risk
            max_loss = (trade.strike * 100 * trade.contracts) - win_amount

        premium_to_max_loss = win_amount / max_loss if max_loss > 0 else 0.0

        if is_short_vol:
            key_reasons.append(
                f"This is a short-vol position ({trade.trade_type.value}) - "
                f"you are selling catastrophe insurance for ${win_amount:.0f} in premium"
            )

            if premium_to_max_loss < 0.02:
                key_reasons.append(
                    f"Premium/max-loss ratio is {premium_to_max_loss:.4f} - "
                    "selling catastrophe insurance too cheap. "
                    "You collect pennies in front of a steamroller"
                )
            elif premium_to_max_loss < 0.05:
                hidden_risks.append(
                    f"Premium/max-loss ratio of {premium_to_max_loss:.3f} is thin - "
                    "a single tail event wipes out months of premium collection"
                )

        if not is_bounded:
            key_reasons.append(
                f"Max loss is effectively unbounded (assignment risk: "
                f"${max_loss:,.0f} at current strike, worse if gaps down to zero)"
            )
            critical_questions.append(
                "What happens if this drops 40% overnight on a fraud revelation or macro shock?"
            )
        else:
            key_reasons.append(
                f"Max loss is bounded at ${max_loss:,.0f} - "
                "at least ruin risk is capped on this position"
            )

        # =====================================================================
        # FRAGILITY SCORE (0-100, higher = more fragile)
        # =====================================================================

        fragility_score = 0
        fragility_factors = []

        # Short gamma is inherently fragile
        if is_short_vol:
            fragility_score += 20
            fragility_factors.append("Short gamma position (+20)")

        # Concentrated in one name
        notional = trade.strike * 100 * trade.contracts
        notional_pct = (notional / portfolio.total_equity) * 100
        if notional_pct > 5:
            fragility_score += 15
            fragility_factors.append(
                f"Concentrated in single name at {notional_pct:.1f}% of portfolio (+15)"
            )

        # Near earnings
        if trade.earnings_before_expiry:
            fragility_score += 20
            fragility_factors.append("Earnings event before expiry - jump risk (+20)")

        # VIX percentile impact
        vix_pct = market.vix_percentile
        if vix_pct > 70:
            fragility_score += 10
            fragility_factors.append(
                f"VIX at {vix_pct:.0f}th percentile - already stressed, tail risk elevated (+10)"
            )
        elif vix_pct < 30:
            # Low VIX = complacency = hidden risk
            fragility_score += 10
            fragility_factors.append(
                f"VIX at {vix_pct:.0f}th percentile - complacency regime, "
                "hidden risk accumulating (+10)"
            )
        # Note: mid-range VIX adds nothing

        # Short DTE with high delta = gamma bomb
        if trade.dte < 14 and abs(trade.delta) > 0.30:
            fragility_score += 15
            fragility_factors.append(
                f"Short DTE ({trade.dte}d) with high delta ({abs(trade.delta):.0%}) - "
                "gamma risk is extreme (+15)"
            )

        # Multiple correlated positions in portfolio
        if portfolio.top_5_concentration > 60:
            fragility_score += 15
            fragility_factors.append(
                f"Top-5 concentration at {portfolio.top_5_concentration:.0f}% - "
                "correlated positions amplify tail risk (+15)"
            )

        key_reasons.append(
            f"Fragility score: {fragility_score}/100 "
            f"({', '.join(fragility_factors) if fragility_factors else 'no major fragility factors'})"
        )

        # =====================================================================
        # CORRELATION RISK
        # =====================================================================

        if portfolio.top_5_concentration > 60:
            hidden_risks.append(
                f"Portfolio top-5 concentration is {portfolio.top_5_concentration:.0f}% - "
                "in a crash, correlations spike to 1.0 and your 'diversified' positions "
                "all blow up simultaneously"
            )

        # Check if all/most positions are equity-based (correlated in crashes)
        equity_position_count = len(portfolio.positions)
        if equity_position_count > 0:
            critical_questions.append(
                f"You have {equity_position_count} equity positions plus this trade - "
                "how many are correlated in a broad market selloff? "
                "In 2008 and 2020, everything fell together"
            )

        # =====================================================================
        # MODEL SKEPTICISM
        # =====================================================================

        p_otm = prob["p_otm"]

        hidden_risks.append(
            "BSM probability estimates assume log-normal returns with constant volatility - "
            "reality has fat tails, volatility clustering, and jumps. "
            "These probabilities have systematic fat-tail blindness"
        )

        if p_otm > 0.90:
            key_reasons.append(
                f"P(OTM) of {p_otm:.0%} is false precision - models cannot reliably "
                "distinguish between 90%, 95%, and 99% OTM probability. "
                "Extreme OTM does not mean safe"
            )
            critical_questions.append(
                "Are you confusing 'high probability' with 'safe'? "
                "A 95% P(OTM) means 1-in-20 trades blows up - "
                "can you survive 20 consecutive max losses?"
            )

        # Discount reported win probabilities
        adjusted_p_profit = prob["p_profit"] * 0.85  # 15% haircut for fat tails
        hidden_risks.append(
            f"Reported P(profit) of {prob['p_profit']:.0%} should be discounted to "
            f"~{adjusted_p_profit:.0%} after adjusting for fat-tail risk. "
            "Models systematically overstate win probabilities for short-vol"
        )

        # =====================================================================
        # TAIL SCENARIOS (always include)
        # =====================================================================

        hidden_risks.append(
            f"Tail scenario: {trade.ticker} gaps down 30-50% on a black swan event. "
            f"Your ${win_amount:.0f} premium becomes a ${max_loss:,.0f}+ loss. "
            "This is not a hypothetical - it happens to real tickers regularly"
        )

        if trade.earnings_before_expiry:
            hidden_risks.append(
                "Earnings are a known jump risk catalyst - "
                "continuous-time models (BSM) are structurally incapable of pricing gap risk. "
                "The one time the gap goes against you erases a year of premiums"
            )

        # =====================================================================
        # DETERMINE JUDGMENT (starts from STRONG skepticism)
        # =====================================================================

        if fragility_score > 60:
            judgment = Judgment.STRONG_REJECT
            judgment_summary = (
                f"Extremely fragile position (fragility: {fragility_score}/100). "
                "Multiple compounding risk factors make this a ruin-risk trade. "
                "No premium justifies this exposure."
            )
        elif fragility_score > 40:
            judgment = Judgment.REJECT
            judgment_summary = (
                f"Fragile position (fragility: {fragility_score}/100). "
                "The premium collected does not compensate for the tail risk assumed. "
                "You are picking up pennies in front of a steamroller."
            )
        elif fragility_score < 20 and is_bounded and premium_to_max_loss >= 0.05:
            judgment = Judgment.APPROVE
            judgment_summary = (
                f"Acceptable risk profile (fragility: {fragility_score}/100). "
                "Bounded loss and adequate premium-to-risk ratio provide some margin of safety. "
                "Still not enthusiastic - you are selling insurance."
            )
        else:
            judgment = Judgment.NEUTRAL
            judgment_summary = (
                f"Moderate fragility (score: {fragility_score}/100). "
                "Proceed only with explicit awareness of tail scenarios and "
                "a plan for when - not if - the tail event occurs."
            )

        # NEVER STRONG_APPROVE - "You should never be enthusiastic about selling insurance"

        # =====================================================================
        # WOULD APPROVE/REJECT CONDITIONS
        # =====================================================================

        would_approve_if.append("Max loss is bounded AND premium/max-loss ratio exceeds 5%")
        would_approve_if.append(
            "Position is small enough that max loss is survivable without portfolio damage"
        )
        would_approve_if.append("No earnings or major macro events before expiry")

        would_reject_if.append("Max loss is unbounded or exceeds 5% of portfolio equity")
        would_reject_if.append("Multiple correlated short-vol positions already in portfolio")
        would_reject_if.append(
            "Earnings, FOMC, or other jump-risk events fall within the expiration window"
        )

        # =====================================================================
        # ADVERSARIAL CRITICAL QUESTIONS
        # =====================================================================

        critical_questions.append(
            f"If {trade.ticker} drops 40% overnight, what is your total portfolio loss "
            "including all correlated positions?"
        )
        critical_questions.append(
            "How many of your current positions are correlated with this one in a crash?"
        )
        if p_otm > 0.80:
            critical_questions.append(
                f"Your model says {p_otm:.0%} chance of profit - "
                "but how many standard deviations is the strike from current price, "
                "and how often do 'impossible' moves actually happen?"
            )

        # =====================================================================
        # CONFIDENCE: Always HIGH
        # =====================================================================

        confidence = ConfidenceLevel.HIGH
        confidence_explanation = (
            "I am always confident about what I don't know. "
            "The models are wrong - the only question is how wrong. "
            "My skepticism is not uncertain; tail risk is real and measurable."
        )

        # Ensure minimum key_reasons (validation requires at least 2)
        if len(key_reasons) < 2:
            key_reasons.append(
                f"Probability profile: {prob['quality']} (EV: {trade.expected_value:.2f}%, "
                f"P(OTM): {p_otm:.0%})"
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
