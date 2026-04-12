"""
Committee Engine

Aggregates responses from all advisor agents into a unified
decision summary with consensus analysis.

This replicates institutional investment committee processes:
- Gather independent opinions
- Identify agreement and disagreement
- Synthesize into actionable recommendation
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from .base import BaseAdvisor
from .buffett import BuffettAdvisor
from .munger import MungerAdvisor
from .schema import (
    AdvisorInput,
    AdvisorResponse,
    ClosedTradeRecord,
    CommitteeOutput,
    ConfidenceLevel,
    Judgment,
    PortfolioReviewInput,
    PortfolioReviewOutput,
    PortfolioReviewResponse,
    PostMortemInput,
    PostMortemOutput,
    PostMortemResponse,
)
from .simons import SimonsAdvisor


class CommitteeEngine:
    """
    Investment committee that aggregates advisor opinions.

    Runs all advisors (optionally in parallel), then synthesizes
    their responses into a unified committee decision.
    """

    def __init__(self, parallel: bool = True):
        """
        Initialize committee with default advisors.

        Args:
            parallel: Run advisors in parallel for speed
        """
        self.parallel = parallel

        # Initialize all advisors
        self.advisors: list[BaseAdvisor] = [
            BuffettAdvisor(),
            MungerAdvisor(),
            SimonsAdvisor(),
        ]

        # Add Taleb advisor if available
        try:
            from .taleb import TalebAdvisor

            self.advisors.append(TalebAdvisor())
        except ImportError:
            pass  # Taleb advisor not yet installed

    def add_advisor(self, advisor: BaseAdvisor) -> None:
        """Add a custom advisor to the committee."""
        self.advisors.append(advisor)

    def evaluate(self, input_data: AdvisorInput) -> CommitteeOutput:
        """
        Run full committee evaluation on a trade.

        Args:
            input_data: Standardized trade input

        Returns:
            CommitteeOutput with aggregated analysis
        """
        start_time = time.time()

        # Run all advisors
        if self.parallel:
            responses = self._run_parallel(input_data)
        else:
            responses = self._run_sequential(input_data)

        # Aggregate responses
        output = self._aggregate_responses(input_data, responses)

        # Calculate total time
        output.total_processing_time_ms = (time.time() - start_time) * 1000

        return output

    def _run_sequential(self, input_data: AdvisorInput) -> list[AdvisorResponse]:
        """Run advisors sequentially."""
        responses = []
        for advisor in self.advisors:
            try:
                response = advisor.evaluate(input_data)
                responses.append(response)
            except Exception as e:
                # Create error response
                responses.append(self._create_error_response(advisor, str(e)))
        return responses

    def _run_parallel(self, input_data: AdvisorInput) -> list[AdvisorResponse]:
        """Run advisors in parallel using thread pool."""
        responses = []

        with ThreadPoolExecutor(max_workers=len(self.advisors)) as executor:
            future_to_advisor = {
                executor.submit(advisor.evaluate, input_data): advisor for advisor in self.advisors
            }

            for future in as_completed(future_to_advisor):
                advisor = future_to_advisor[future]
                try:
                    response = future.result(timeout=30)
                    responses.append(response)
                except Exception as e:
                    responses.append(self._create_error_response(advisor, str(e)))

        return responses

    def _create_error_response(self, advisor: BaseAdvisor, error: str) -> AdvisorResponse:
        """Create error response when advisor fails."""
        return AdvisorResponse(
            advisor_name=advisor.name,
            advisor_philosophy=advisor.philosophy,
            judgment=Judgment.NEUTRAL,
            judgment_summary=f"Analysis failed: {error}",
            key_reasons=["Advisor encountered an error during analysis"],
            critical_questions=["Review advisor logs for error details"],
            hidden_risks=["Unable to complete risk assessment"],
            would_approve_if=[],
            would_reject_if=[],
            confidence=ConfidenceLevel.VERY_LOW,
            confidence_explanation="Analysis incomplete due to error",
        )

    def _aggregate_responses(
        self, input_data: AdvisorInput, responses: list[AdvisorResponse]
    ) -> CommitteeOutput:
        """
        Aggregate individual advisor responses into committee decision.

        Analyzes:
        - Voting pattern (approve/reject/neutral)
        - Areas of agreement
        - Areas of disagreement
        - Unresolved risks
        - Required actions before trading
        """
        trade = input_data.candidate_trade

        # Count judgments
        approve_count = sum(
            1 for r in responses if r.judgment in [Judgment.APPROVE, Judgment.STRONG_APPROVE]
        )
        reject_count = sum(
            1 for r in responses if r.judgment in [Judgment.REJECT, Judgment.STRONG_REJECT]
        )
        neutral_count = sum(1 for r in responses if r.judgment == Judgment.NEUTRAL)

        total = len(responses)
        unanimous_approve = approve_count == total
        unanimous_reject = reject_count == total

        # Find areas of agreement and disagreement
        areas_of_agreement = self._find_agreement(responses)
        areas_of_disagreement = self._find_disagreement(responses)

        # Collect all unresolved risks
        unresolved_risks = self._collect_unresolved_risks(responses)

        # Determine committee judgment
        committee_judgment, committee_reasoning = self._determine_committee_judgment(
            responses, approve_count, reject_count, neutral_count
        )

        # Build required actions and modifications
        required_before_trade = self._build_required_actions(responses, committee_judgment)
        recommended_modifications = self._build_modifications(responses, committee_judgment)

        # Determine committee confidence
        committee_confidence = self._determine_confidence(responses, approve_count, reject_count)

        # Build trade summary
        trade_summary = (
            f"{trade.trade_type.value.upper()} on {trade.ticker}: "
            f"${trade.strike} strike, {trade.dte} DTE, "
            f"EV: {trade.expected_value:.2f}%, P(OTM): {trade.p_otm:.0%}"
        )

        return CommitteeOutput(
            request_id=input_data.request_id,
            trade_summary=trade_summary,
            advisor_responses=responses,
            unanimous_approve=unanimous_approve,
            unanimous_reject=unanimous_reject,
            approval_count=approve_count,
            rejection_count=reject_count,
            neutral_count=neutral_count,
            areas_of_agreement=areas_of_agreement,
            areas_of_disagreement=areas_of_disagreement,
            unresolved_risks=unresolved_risks,
            committee_judgment=committee_judgment,
            committee_reasoning=committee_reasoning,
            required_before_trade=required_before_trade,
            recommended_modifications=recommended_modifications,
            committee_confidence=committee_confidence,
            analysis_timestamp=datetime.utcnow(),
        )

    def _find_agreement(self, responses: list[AdvisorResponse]) -> list[str]:
        """Find areas where advisors agree."""
        agreements = []

        # Check if all agree on direction
        all_approve = all(
            r.judgment in [Judgment.APPROVE, Judgment.STRONG_APPROVE] for r in responses
        )
        all_reject = all(r.judgment in [Judgment.REJECT, Judgment.STRONG_REJECT] for r in responses)

        if all_approve:
            agreements.append("All advisors agree: APPROVE the trade")
        elif all_reject:
            agreements.append("All advisors agree: REJECT the trade")

        # Find common risks mentioned
        risk_counts = {}
        for r in responses:
            for risk in r.hidden_risks:
                # Simplify risk to key terms
                key_terms = self._extract_key_terms(risk)
                for term in key_terms:
                    risk_counts[term] = risk_counts.get(term, 0) + 1

        # Risks mentioned by majority
        majority_threshold = len(responses) / 2
        common_risks = [term for term, count in risk_counts.items() if count >= majority_threshold]

        if common_risks:
            agreements.append(f"Common risk themes: {', '.join(common_risks[:3])}")

        # Find common questions
        question_counts = {}
        for r in responses:
            for q in r.critical_questions:
                key_terms = self._extract_key_terms(q)
                for term in key_terms:
                    question_counts[term] = question_counts.get(term, 0) + 1

        common_questions = [
            term for term, count in question_counts.items() if count >= majority_threshold
        ]

        if common_questions:
            agreements.append(f"Shared concerns: {', '.join(common_questions[:3])}")

        return agreements[:5]

    def _find_disagreement(self, responses: list[AdvisorResponse]) -> list[str]:
        """Find areas where advisors disagree."""
        disagreements = []

        judgments = [r.judgment for r in responses]
        unique_judgments = set(judgments)

        if len(unique_judgments) > 1:
            # Describe the split
            for r in responses:
                if r.judgment in [Judgment.APPROVE, Judgment.STRONG_APPROVE]:
                    stance = "approves"
                elif r.judgment in [Judgment.REJECT, Judgment.STRONG_REJECT]:
                    stance = "rejects"
                else:
                    stance = "is neutral on"
                disagreements.append(f"{r.advisor_name} {stance} the trade")

        # Check for conflicting confidence levels
        confidences = [r.confidence for r in responses]
        if ConfidenceLevel.HIGH in confidences and ConfidenceLevel.LOW in confidences:
            disagreements.append("Confidence levels vary significantly across advisors")

        return disagreements[:5]

    def _collect_unresolved_risks(self, responses: list[AdvisorResponse]) -> list[str]:
        """Collect all unique unresolved risks."""
        all_risks = []
        seen_terms = set()

        for r in responses:
            for risk in r.hidden_risks:
                # Deduplicate by key terms
                key = tuple(sorted(self._extract_key_terms(risk)))
                if key not in seen_terms:
                    seen_terms.add(key)
                    all_risks.append(f"[{r.advisor_name}] {risk}")

        return all_risks[:6]

    def _determine_committee_judgment(
        self,
        responses: list[AdvisorResponse],
        approve_count: int,
        reject_count: int,
        neutral_count: int,
    ) -> tuple[Judgment, str]:
        """Determine final committee judgment."""
        total = len(responses)

        # Unanimous decisions
        if approve_count == total:
            return (
                Judgment.STRONG_APPROVE,
                "Unanimous approval from all advisors. Proceed with confidence.",
            )

        if reject_count == total:
            return (
                Judgment.STRONG_REJECT,
                "Unanimous rejection from all advisors. Do not proceed.",
            )

        # Majority decisions
        if approve_count > total / 2:
            return (
                Judgment.APPROVE,
                f"Majority approval ({approve_count}/{total}). Proceed with noted concerns.",
            )

        if reject_count > total / 2:
            return (
                Judgment.REJECT,
                f"Majority rejection ({reject_count}/{total}). Significant concerns exist.",
            )

        # Split decision
        return (
            Judgment.NEUTRAL,
            f"Split decision ({approve_count} approve, {reject_count} reject, "
            f"{neutral_count} neutral). Additional analysis recommended.",
        )

    def _determine_confidence(
        self, responses: list[AdvisorResponse], approve_count: int, reject_count: int
    ) -> ConfidenceLevel:
        """Determine committee confidence based on consensus."""
        total = len(responses)

        # Strong consensus = high confidence
        if approve_count == total or reject_count == total:
            # Check individual confidence levels
            avg_confidence = sum(self._confidence_to_num(r.confidence) for r in responses) / total
            if avg_confidence >= 4:
                return ConfidenceLevel.VERY_HIGH
            return ConfidenceLevel.HIGH

        # Majority = medium-high confidence
        if approve_count > total / 2 or reject_count > total / 2:
            return ConfidenceLevel.MEDIUM

        # Split = low confidence
        return ConfidenceLevel.LOW

    def _confidence_to_num(self, confidence: ConfidenceLevel) -> int:
        """Convert confidence level to numeric."""
        mapping = {
            ConfidenceLevel.VERY_LOW: 1,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.VERY_HIGH: 5,
        }
        return mapping.get(confidence, 3)

    def _build_required_actions(
        self, responses: list[AdvisorResponse], judgment: Judgment
    ) -> list[str]:
        """Build list of required actions before trading."""
        actions = []

        # If not unanimous approval, require answers to critical questions
        if judgment not in [Judgment.STRONG_APPROVE]:
            actions.append("Answer all critical questions raised by advisors")

        # Collect specific requirements from would_approve_if
        for r in responses:
            if r.judgment in [Judgment.REJECT, Judgment.STRONG_REJECT, Judgment.NEUTRAL]:
                for condition in r.would_approve_if[:1]:
                    actions.append(f"[{r.advisor_name}] {condition}")

        # Generic requirements
        if judgment == Judgment.NEUTRAL:
            actions.append("Gather additional data to break the tie")

        return actions[:5]

    def _build_modifications(
        self, responses: list[AdvisorResponse], judgment: Judgment
    ) -> list[str]:
        """Build list of recommended modifications to the trade."""
        modifications = []

        # Look for common modification themes
        for r in responses:
            for condition in r.would_approve_if:
                if "size" in condition.lower() or "smaller" in condition.lower():
                    modifications.append("Consider reducing position size")
                if "strike" in condition.lower() or "lower" in condition.lower():
                    modifications.append("Consider a lower strike price")
                if "dte" in condition.lower() or "time" in condition.lower():
                    modifications.append("Consider longer time to expiration")

        # Deduplicate
        modifications = list(dict.fromkeys(modifications))

        return modifications[:4]

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text for comparison."""
        # Simple keyword extraction
        keywords = {
            "concentration",
            "earnings",
            "volatility",
            "iv",
            "regime",
            "assignment",
            "bias",
            "risk",
            "premium",
            "delta",
            "gamma",
            "theta",
            "vega",
            "sector",
            "position",
            "sizing",
            "kelly",
            "statistical",
            "probability",
            "edge",
            "model",
            "validation",
        }

        text_lower = text.lower()
        found = [kw for kw in keywords if kw in text_lower]

        return found[:3]

    # =========================================================================
    # PORTFOLIO REVIEW MODE
    # =========================================================================

    def review_portfolio(self, review_input: PortfolioReviewInput) -> PortfolioReviewOutput:
        """
        Committee-level portfolio review.

        Each advisor reviews the full portfolio and provides:
        - Strengths and weaknesses
        - Blind spots the owner might be missing
        - Concentration and risk critiques
        - Actionable suggestions

        Args:
            review_input: Portfolio state with market context

        Returns:
            PortfolioReviewOutput with consensus analysis
        """
        start_time = time.time()
        reviews = []

        for advisor in self.advisors:
            try:
                review = self._advisor_portfolio_review(advisor, review_input)
                reviews.append(review)
            except Exception as e:
                reviews.append(
                    PortfolioReviewResponse(
                        advisor_name=advisor.name,
                        overall_assessment="error",
                        assessment_summary=f"Review failed: {e}",
                        strengths=[],
                        weaknesses=[],
                        blind_spots=[],
                        suggestions=[],
                        concentration_critique="Unable to assess",
                        risk_critique="Unable to assess",
                        strategy_critique="Unable to assess",
                        confidence=ConfidenceLevel.VERY_LOW,
                    )
                )

        # Aggregate
        consensus = self._aggregate_portfolio_reviews(reviews)
        consensus.request_id = review_input.request_id
        consensus.total_processing_time_ms = (time.time() - start_time) * 1000
        return consensus

    def _advisor_portfolio_review(
        self, advisor: BaseAdvisor, review_input: PortfolioReviewInput
    ) -> PortfolioReviewResponse:
        """Generate a portfolio review from a single advisor."""
        portfolio = review_input.portfolio
        market = review_input.market
        start = time.time()

        strengths = []
        weaknesses = []
        blind_spots = []
        suggestions = []

        # Diversification
        top5 = portfolio.top_5_concentration
        n_positions = portfolio.open_positions_count
        if top5 > 60:
            weaknesses.append(f"Top 5 concentration at {top5:.0f}% — high single-name risk")
            suggestions.append("Consider reducing position sizes in top holdings")
        elif top5 < 40 and n_positions >= 5:
            strengths.append(f"Good diversification: top 5 at {top5:.0f}%")

        # Cash position
        cash_pct = (
            (portfolio.cash_available / portfolio.total_equity * 100)
            if portfolio.total_equity > 0
            else 0
        )
        if cash_pct < 10:
            weaknesses.append(f"Cash at {cash_pct:.0f}% — limited flexibility for opportunities")
        elif cash_pct > 40:
            blind_spots.append(f"Cash at {cash_pct:.0f}% — potential cash drag on returns")
        else:
            strengths.append(f"Healthy cash position at {cash_pct:.0f}%")

        # Beta exposure
        if portfolio.portfolio_beta > 1.5:
            weaknesses.append(
                f"Portfolio beta {portfolio.portfolio_beta:.2f} — highly market-sensitive"
            )
        elif portfolio.portfolio_beta < 0.5:
            blind_spots.append(
                f"Low beta {portfolio.portfolio_beta:.2f} — may underperform in rallies"
            )

        # Drawdown
        if portfolio.max_drawdown_30d < -15:
            weaknesses.append(
                f"Recent 30d drawdown {portfolio.max_drawdown_30d:.1f}% — check risk controls"
            )
        elif portfolio.max_drawdown_30d > -5:
            strengths.append("Controlled recent drawdown")

        # VaR
        if portfolio.var_95 > 5:
            weaknesses.append(f"95% VaR at {portfolio.var_95:.1f}% — elevated daily risk")

        # Sector concentration
        max_sector = max(portfolio.sector_allocation.values()) if portfolio.sector_allocation else 0
        max_sector_name = (
            max(portfolio.sector_allocation, key=portfolio.sector_allocation.get)
            if portfolio.sector_allocation
            else "Unknown"
        )
        if max_sector > 40:
            weaknesses.append(f"Sector concentration: {max_sector_name} at {max_sector:.0f}%")
            suggestions.append(f"Reduce {max_sector_name} exposure below 30%")

        # Market regime awareness
        if market.regime.value in ("crisis", "high_volatility"):
            blind_spots.append(
                "Operating in elevated volatility — position sizes should be reduced"
            )

        # Margin usage
        margin_pct = (
            (portfolio.total_margin_used / portfolio.total_equity * 100)
            if portfolio.total_equity > 0
            else 0
        )
        if margin_pct > 50:
            weaknesses.append(f"Margin usage at {margin_pct:.0f}% — approaching dangerous levels")

        # Advisor-specific flavor
        if "Buffett" in advisor.name:
            blind_spots.append("Are all positions in businesses you would want to own outright?")
        elif "Munger" in advisor.name:
            blind_spots.append("What is the most obvious way this portfolio gets destroyed?")
        elif "Simons" in advisor.name:
            blind_spots.append("Is the edge statistically validated across multiple regimes?")
        elif "Taleb" in advisor.name:
            blind_spots.append("Is this portfolio robust to a 3-sigma correlation spike?")

        # Overall assessment
        n_weak = len(weaknesses)
        n_strong = len(strengths)
        if n_weak >= 4:
            overall = "fragile"
        elif n_weak >= 2 and n_strong < 2:
            overall = "concerning"
        elif n_strong >= 3 and n_weak <= 1:
            overall = "strong"
        else:
            overall = "healthy"

        return PortfolioReviewResponse(
            advisor_name=advisor.name,
            overall_assessment=overall,
            assessment_summary=f"{advisor.name} rates portfolio as {overall} with {n_strong} strengths and {n_weak} concerns.",
            strengths=strengths,
            weaknesses=weaknesses,
            blind_spots=blind_spots,
            suggestions=suggestions,
            concentration_critique=f"Top 5 at {top5:.0f}%, max sector {max_sector_name} at {max_sector:.0f}%",
            risk_critique=f"Beta {portfolio.portfolio_beta:.2f}, VaR {portfolio.var_95:.1f}%, DD {portfolio.max_drawdown_30d:.1f}%",
            strategy_critique=review_input.strategy_description
            or "No strategy description provided",
            confidence=ConfidenceLevel.HIGH,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _aggregate_portfolio_reviews(
        self, reviews: list[PortfolioReviewResponse]
    ) -> PortfolioReviewOutput:
        """Aggregate portfolio reviews into consensus."""
        # Consensus assessment
        assessments = [r.overall_assessment for r in reviews]
        if all(a in ("strong", "healthy") for a in assessments):
            consensus = "strong"
        elif any(a == "fragile" for a in assessments):
            consensus = "fragile"
        elif sum(1 for a in assessments if a in ("concerning", "fragile")) >= len(reviews) / 2:
            consensus = "concerning"
        else:
            consensus = "healthy"

        # Collect all items and find overlaps
        all_strengths = [s for r in reviews for s in r.strengths]
        all_weaknesses = [w for r in reviews for w in r.weaknesses]
        all_blind_spots = [b for r in reviews for b in r.blind_spots]
        all_suggestions = [s for r in reviews for s in r.suggestions]

        # Deduplicate
        def _dedup(items):
            seen = set()
            result = []
            for item in items:
                key = item[:40].lower()
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        return PortfolioReviewOutput(
            request_id="",
            advisor_reviews=reviews,
            consensus_assessment=consensus,
            consensus_strengths=_dedup(all_strengths)[:5],
            consensus_weaknesses=_dedup(all_weaknesses)[:5],
            critical_blind_spots=_dedup(all_blind_spots)[:5],
            priority_actions=_dedup(all_suggestions)[:5],
        )

    # =========================================================================
    # POST-MORTEM MODE
    # =========================================================================

    def post_mortem(self, pm_input: PostMortemInput) -> PostMortemOutput:
        """
        Committee-level post-mortem analysis of closed trades.

        Each advisor examines the trade history and identifies:
        - What was good process vs bad luck vs bad process
        - Lessons learned
        - Rules to add or change
        - Behavioral patterns

        Args:
            pm_input: Closed trades with portfolio/market context

        Returns:
            PostMortemOutput with consensus lessons
        """
        start_time = time.time()
        trades = pm_input.closed_trades

        # Compute aggregate stats
        total = len(trades)
        wins = sum(1 for t in trades if t.outcome == "win")
        losses = sum(1 for t in trades if t.outcome == "loss")
        total_pnl = sum(t.pnl for t in trades)
        win_rate = wins / total if total > 0 else 0

        reviews = []
        for advisor in self.advisors:
            try:
                review = self._advisor_post_mortem(
                    advisor, pm_input, wins, losses, total_pnl, win_rate
                )
                reviews.append(review)
            except Exception as e:
                reviews.append(
                    PostMortemResponse(
                        advisor_name=advisor.name,
                        overall_grade="N/A",
                        grade_explanation=f"Post-mortem failed: {e}",
                        what_was_good=[],
                        what_was_bad_luck=[],
                        what_was_bad_process=[],
                        lessons=[],
                        rules_to_add=[],
                        rules_to_change=[],
                        patterns_observed=[],
                        confidence=ConfidenceLevel.VERY_LOW,
                    )
                )

        # Aggregate
        output = self._aggregate_post_mortems(reviews, trades, pm_input.period)
        output.total_processing_time_ms = (time.time() - start_time) * 1000
        return output

    def _advisor_post_mortem(
        self,
        advisor: BaseAdvisor,
        pm_input: PostMortemInput,
        wins: int,
        losses: int,
        total_pnl: float,
        win_rate: float,
    ) -> PostMortemResponse:
        """Generate post-mortem from a single advisor."""
        start = time.time()
        trades = pm_input.closed_trades
        total = len(trades)

        good = []
        bad_luck = []
        bad_process = []
        lessons = []
        rules_add = []
        rules_change = []
        patterns = []

        # Analyze win rate
        if win_rate > 0.70:
            good.append(f"Strong win rate: {win_rate:.0%} ({wins}/{total})")
        elif win_rate < 0.50:
            bad_process.append(f"Win rate below 50%: {win_rate:.0%} — review entry criteria")
            rules_change.append("Tighten entry requirements (higher IV rank, better EV)")

        # Analyze P&L distribution
        if total_pnl > 0:
            good.append(f"Positive total P&L: ${total_pnl:.2f}")
        else:
            lessons.append(f"Net loss of ${abs(total_pnl):.2f} — reassess strategy sizing")

        # Analyze individual trades
        big_losses = [t for t in trades if t.pnl < -500]

        for t in big_losses:
            if t.exit_reason == "stop_loss":
                good.append(f"{t.ticker}: Stop loss discipline worked (${t.pnl:.0f})")
            elif t.exit_reason == "assigned":
                bad_luck.append(
                    f"{t.ticker}: Assignment loss (${t.pnl:.0f}) — market moved against"
                )
            elif t.max_drawdown_during < t.pnl * 2:
                bad_process.append(
                    f"{t.ticker}: Held too long (DD: {t.max_drawdown_during:.0f}%, final: ${t.pnl:.0f})"
                )

        # Pattern detection
        assigned_trades = [t for t in trades if t.exit_reason == "assigned"]
        if len(assigned_trades) > total * 0.3:
            patterns.append(
                f"High assignment rate ({len(assigned_trades)}/{total}) — strikes may be too aggressive"
            )
            rules_change.append("Consider wider OTM strikes (lower delta)")

        earnings_losses = [
            t for t in trades if t.outcome == "loss" and "earnings" in t.notes.lower()
        ]
        if earnings_losses:
            patterns.append(f"{len(earnings_losses)} losses related to earnings events")
            rules_add.append("Never sell premium within 10 days of earnings")

        # Check for sizing issues
        avg_hold = sum(t.hold_days for t in trades) / total if total > 0 else 0
        if avg_hold > 30:
            patterns.append(f"Average hold time {avg_hold:.0f} days — consider shorter DTE")

        # Grade
        if win_rate > 0.70 and total_pnl > 0:
            grade = "A" if win_rate > 0.80 else "B"
        elif win_rate > 0.55 and total_pnl > 0:
            grade = "B" if total_pnl > 1000 else "C"
        elif total_pnl > 0:
            grade = "C"
        elif win_rate > 0.50:
            grade = "C"
        else:
            grade = "D" if total_pnl > -5000 else "F"

        # Advisor flavor
        if "Munger" in advisor.name:
            lessons.append("Invert: which trades should you have NEVER taken?")
        elif "Taleb" in advisor.name:
            lessons.append("Check if losses were correlated — a fragility signal")
        elif "Simons" in advisor.name:
            lessons.append("Run these results through walk-forward validation before trusting")

        return PostMortemResponse(
            advisor_name=advisor.name,
            overall_grade=grade,
            grade_explanation=f"Win rate {win_rate:.0%}, total P&L ${total_pnl:.0f}, {total} trades",
            what_was_good=good[:4],
            what_was_bad_luck=bad_luck[:4],
            what_was_bad_process=bad_process[:4],
            lessons=lessons[:5],
            rules_to_add=rules_add[:3],
            rules_to_change=rules_change[:3],
            patterns_observed=patterns[:4],
            confidence=ConfidenceLevel.HIGH if total >= 20 else ConfidenceLevel.MEDIUM,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _aggregate_post_mortems(
        self,
        reviews: list[PostMortemResponse],
        trades: list[ClosedTradeRecord],
        period: str,
    ) -> PostMortemOutput:
        """Aggregate post-mortem reviews."""
        total = len(trades)
        wins = sum(1 for t in trades if t.outcome == "win")
        total_pnl = sum(t.pnl for t in trades)

        # Consensus grade (mode)
        grades = [r.overall_grade for r in reviews if r.overall_grade != "N/A"]
        consensus_grade = max(set(grades), key=grades.count) if grades else "N/A"

        # Collect and deduplicate lessons
        all_lessons = [lesson for r in reviews for lesson in r.lessons]
        all_improvements = [rule for r in reviews for rule in r.rules_to_add + r.rules_to_change]
        all_patterns = [p for r in reviews for p in r.patterns_observed]

        # Check for behavioral flags
        behavioral = []
        bad_process_items = [bp for r in reviews for bp in r.what_was_bad_process]
        if len(bad_process_items) > 3:
            behavioral.append("Multiple process errors detected — review trading discipline")
        if any("held too long" in bp.lower() for bp in bad_process_items):
            behavioral.append("Pattern: holding losing positions too long")
        if any("aggressive" in p.lower() for p in all_patterns):
            behavioral.append("Pattern: strike selection may be too aggressive")

        def _dedup(items):
            seen = set()
            result = []
            for item in items:
                key = item[:40].lower()
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result

        return PostMortemOutput(
            request_id="",
            period=period,
            advisor_reviews=reviews,
            consensus_grade=consensus_grade,
            total_trades=total,
            win_rate=wins / total if total > 0 else 0,
            total_pnl=total_pnl,
            avg_pnl_per_trade=total_pnl / total if total > 0 else 0,
            consensus_lessons=_dedup(all_lessons)[:5],
            process_improvements=_dedup(all_improvements)[:5],
            behavioral_flags=behavioral[:3],
        )


def format_committee_report(output: CommitteeOutput) -> str:
    """
    Format committee output as readable report.

    Args:
        output: CommitteeOutput from committee evaluation

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("INVESTMENT COMMITTEE REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Trade summary
    lines.append(f"TRADE: {output.trade_summary}")
    lines.append("")

    # Committee decision
    lines.append("-" * 40)
    lines.append("COMMITTEE DECISION")
    lines.append("-" * 40)
    lines.append(f"Judgment: {output.committee_judgment.value.upper()}")
    lines.append(f"Confidence: {output.committee_confidence.value}")
    lines.append(f"Reasoning: {output.committee_reasoning}")
    lines.append("")

    # Vote breakdown
    lines.append(
        f"Votes: {output.approval_count} approve, "
        f"{output.rejection_count} reject, {output.neutral_count} neutral"
    )
    lines.append("")

    # Individual advisor summaries
    lines.append("-" * 40)
    lines.append("ADVISOR OPINIONS")
    lines.append("-" * 40)
    for r in output.advisor_responses:
        lines.append(f"\n[{r.advisor_name}] - {r.judgment.value.upper()}")
        lines.append(f"  Philosophy: {r.advisor_philosophy}")
        lines.append(f"  Summary: {r.judgment_summary}")
        lines.append("  Key reasons:")
        for reason in r.key_reasons[:3]:
            lines.append(f"    - {reason}")
    lines.append("")

    # Agreement / Disagreement
    if output.areas_of_agreement:
        lines.append("-" * 40)
        lines.append("AREAS OF AGREEMENT")
        lines.append("-" * 40)
        for item in output.areas_of_agreement:
            lines.append(f"  - {item}")
        lines.append("")

    if output.areas_of_disagreement:
        lines.append("-" * 40)
        lines.append("AREAS OF DISAGREEMENT")
        lines.append("-" * 40)
        for item in output.areas_of_disagreement:
            lines.append(f"  - {item}")
        lines.append("")

    # Unresolved risks
    if output.unresolved_risks:
        lines.append("-" * 40)
        lines.append("UNRESOLVED RISKS")
        lines.append("-" * 40)
        for risk in output.unresolved_risks:
            lines.append(f"  - {risk}")
        lines.append("")

    # Required actions
    if output.required_before_trade:
        lines.append("-" * 40)
        lines.append("REQUIRED BEFORE TRADING")
        lines.append("-" * 40)
        for action in output.required_before_trade:
            lines.append(f"  [ ] {action}")
        lines.append("")

    # Recommended modifications
    if output.recommended_modifications:
        lines.append("-" * 40)
        lines.append("RECOMMENDED MODIFICATIONS")
        lines.append("-" * 40)
        for mod in output.recommended_modifications:
            lines.append(f"  - {mod}")
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Report generated: {output.analysis_timestamp.isoformat()}")
    lines.append(f"Processing time: {output.total_processing_time_ms:.0f}ms")
    lines.append("=" * 70)

    return "\n".join(lines)
