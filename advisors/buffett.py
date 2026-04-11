"""
Buffett-Style Advisor Agent

Evaluates trades through the lens of:
- Business quality and competitive moats
- Long-term value creation
- Capital allocation discipline
- Margin of safety
- Owner mentality

"Price is what you pay, value is what you get."
"""

from .base import BaseAdvisor
from .schema import (
    AdvisorInput,
    ConfidenceLevel,
    Judgment,
)


class BuffettAdvisor(BaseAdvisor):
    """
    Warren Buffett-inspired investment advisor.

    Core principles:
    1. Only invest in businesses you understand
    2. Look for durable competitive advantages
    3. Insist on a margin of safety
    4. Think like an owner, not a trader
    5. Be fearful when others are greedy

    For options: evaluates whether the underlying
    business justifies the risk/reward profile.
    """

    @property
    def name(self) -> str:
        return "Buffett"

    @property
    def philosophy(self) -> str:
        return "Business quality and capital allocation discipline"

    @property
    def system_prompt(self) -> str:
        return """You are an investment advisor modeled after Warren Buffett's principles.

CORE BELIEFS:
- The stock market is a mechanism for transferring wealth from the impatient to the patient
- A wonderful business at a fair price beats a fair business at a wonderful price
- Risk comes from not knowing what you're doing
- Never invest in a business you cannot understand
- Time is the friend of the wonderful business, the enemy of the mediocre

EVALUATION FRAMEWORK:
1. Business Quality: Does this company have durable competitive advantages?
2. Management: Is leadership honest, capable, and shareholder-oriented?
3. Financial Strength: Strong balance sheet? Consistent earnings? High ROE?
4. Valuation: Is the price reasonable relative to intrinsic value?
5. Circle of Competence: Can we truly understand this business?

FOR OPTIONS EVALUATION:
- Would you be happy owning this stock if assigned?
- Is the premium adequate compensation for commitment of capital?
- Does this trade align with long-term ownership mentality?
- What's the downside if everything goes wrong?

WHAT YOU REJECT:
- Speculation disguised as investment
- Trades based purely on technical signals
- Overleveraged positions
- Concentration without conviction
- Premium chasing without quality assessment

YOUR TONE:
- Folksy but precise
- Patient and long-term oriented
- Focused on downside protection
- Skeptical of complexity"""

    def _analyze(self, input_data: AdvisorInput) -> dict:
        """Buffett-style analysis."""
        trade = input_data.candidate_trade
        market = input_data.market

        # Gather assessments
        self._assess_position_size(input_data)
        concentration = self._assess_concentration_risk(input_data)
        earnings = self._assess_earnings_risk(input_data)
        self._assess_iv_environment(input_data)
        prob = self._assess_probability_profile(input_data)

        # Build analysis
        key_reasons = []
        critical_questions = []
        hidden_risks = []
        would_approve_if = []
        would_reject_if = []

        # =====================================================================
        # BUSINESS QUALITY ASSESSMENT
        # =====================================================================
        quality_tickers = {"AAPL", "MSFT", "GOOGL", "JNJ", "PG", "KO", "BRK.B", "V", "MA", "UNH"}
        speculative_tickers = {"GME", "AMC", "BBBY", "RIVN", "LCID", "PLTR", "COIN"}

        is_quality = trade.ticker in quality_tickers
        is_speculative = trade.ticker in speculative_tickers

        if is_quality:
            key_reasons.append(
                f"{trade.ticker} represents a high-quality business with durable competitive advantages"
            )
        elif is_speculative:
            key_reasons.append(
                f"{trade.ticker} is highly speculative - business model unproven or facing structural challenges"
            )
            hidden_risks.append("Underlying business may not survive long-term")

        # =====================================================================
        # OWNERSHIP MENTALITY
        # =====================================================================
        # For short puts: would you want to own at strike price?
        if trade.trade_type.value in ["short_put", "cash_secured_put"]:
            strike_discount = (trade.underlying_price - trade.strike) / trade.underlying_price * 100

            if strike_discount > 10:
                key_reasons.append(
                    f"Strike price offers {strike_discount:.1f}% discount - decent margin of safety"
                )
            else:
                hidden_risks.append(
                    f"Strike only {strike_discount:.1f}% below current price - minimal margin of safety"
                )

            critical_questions.append(
                f"Would you be delighted to own {trade.ticker} at ${trade.strike} for the next 5 years?"
            )

        # =====================================================================
        # CAPITAL ALLOCATION
        # =====================================================================
        if concentration["high_concentration"]:
            hidden_risks.append(
                f"Already {concentration['current_allocation']:.0f}% in {concentration['sector']} - "
                "this trade increases concentration risk"
            )
            would_reject_if.append("Sector concentration exceeds 50%")

        # Premium as return on committed capital
        notional = trade.strike * 100 * trade.contracts
        premium_return = (trade.premium * 100 * trade.contracts) / notional * 100
        annualized_return = premium_return * (365 / trade.dte)

        if annualized_return > 20:
            key_reasons.append(
                f"Annualized premium return of {annualized_return:.1f}% is attractive"
            )
        elif annualized_return < 10:
            hidden_risks.append(
                f"Annualized return of {annualized_return:.1f}% may not justify capital commitment"
            )

        # =====================================================================
        # MARKET ENVIRONMENT
        # =====================================================================
        if market.regime.value == "crisis":
            key_reasons.append(
                "Market fear is elevated - historically a good time to be a seller of puts"
            )
            critical_questions.append(
                "Is this fear justified by fundamental deterioration, or is it temporary?"
            )
        elif market.regime.value == "high_volatility":
            key_reasons.append(
                "Elevated volatility means richer premiums - but ensure quality justifies risk"
            )

        # =====================================================================
        # EARNINGS RISK
        # =====================================================================
        if earnings["has_earnings"]:
            hidden_risks.append("Earnings event before expiration introduces binary risk")
            critical_questions.append(
                "Are you comfortable holding through earnings? What's the worst-case scenario?"
            )

        # =====================================================================
        # DETERMINE JUDGMENT
        # =====================================================================
        judgment = Judgment.NEUTRAL
        judgment_summary = ""

        negative_signals = 0
        positive_signals = 0

        # Count signals
        if is_speculative:
            negative_signals += 2
        if is_quality:
            positive_signals += 1

        if concentration["warning"]:
            negative_signals += 2
        elif concentration["high_concentration"]:
            negative_signals += 1

        if earnings["has_earnings"]:
            negative_signals += 1

        if annualized_return > 15:
            positive_signals += 1
        if annualized_return < 10:
            negative_signals += 1

        if prob["quality"] in ["EXCELLENT", "GOOD"]:
            positive_signals += 1
        elif prob["quality"] == "POOR":
            negative_signals += 2

        # Make judgment
        if negative_signals >= 4:
            judgment = Judgment.STRONG_REJECT
            judgment_summary = (
                "This trade does not meet our standards for business quality "
                "and capital allocation discipline."
            )
        elif negative_signals >= 2 and positive_signals < 2:
            judgment = Judgment.REJECT
            judgment_summary = "The risk/reward profile does not justify the commitment of capital."
        elif positive_signals >= 3 and negative_signals <= 1:
            judgment = Judgment.APPROVE
            judgment_summary = f"Quality business at reasonable terms. Would be pleased to own {trade.ticker} if assigned."
        elif positive_signals >= 4:
            judgment = Judgment.STRONG_APPROVE
            judgment_summary = (
                "Excellent opportunity combining business quality with attractive premium."
            )
        else:
            judgment = Judgment.NEUTRAL
            judgment_summary = (
                "Trade has merits but also concerns. Consider modifications or reduced size."
            )

        # Build would_approve_if / would_reject_if
        if judgment in [Judgment.REJECT, Judgment.STRONG_REJECT]:
            would_approve_if.append("Lower strike providing greater margin of safety")
            would_approve_if.append("Reduced position size to limit concentration")
        else:
            would_reject_if.append("Company fundamentals deteriorate")
            would_reject_if.append("Position grows beyond 5% of portfolio")

        # Confidence
        confidence = ConfidenceLevel.MEDIUM
        confidence_explanation = "Analysis based on business quality heuristics and position sizing"

        if is_quality or is_speculative:
            confidence = ConfidenceLevel.HIGH
            confidence_explanation = "Clear view on business quality informs judgment"

        # Ensure minimum key_reasons (spec: ≥2)
        if len(key_reasons) < 2:
            key_reasons.append(
                f"Probability profile: {prob['quality']} (EV: {trade.expected_value:.2f}%, "
                f"P(OTM): {prob['p_otm']:.0%})"
            )
        if len(key_reasons) < 2:
            notional = trade.strike * 100 * trade.contracts
            pct = notional / input_data.portfolio.total_equity * 100
            key_reasons.append(f"Position sizing: {pct:.1f}% of portfolio")

        # Ensure minimum critical_questions (spec: ≥1)
        if len(critical_questions) < 1:
            critical_questions.append(
                f"Would you be comfortable owning {trade.ticker} at ${trade.strike} "
                "for the long term if assigned?"
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
