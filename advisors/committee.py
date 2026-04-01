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
    CommitteeOutput,
    ConfidenceLevel,
    Judgment,
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
                executor.submit(advisor.evaluate, input_data): advisor
                for advisor in self.advisors
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
        self,
        input_data: AdvisorInput,
        responses: list[AdvisorResponse]
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
            1 for r in responses
            if r.judgment in [Judgment.APPROVE, Judgment.STRONG_APPROVE]
        )
        reject_count = sum(
            1 for r in responses
            if r.judgment in [Judgment.REJECT, Judgment.STRONG_REJECT]
        )
        neutral_count = sum(
            1 for r in responses
            if r.judgment == Judgment.NEUTRAL
        )

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
            r.judgment in [Judgment.APPROVE, Judgment.STRONG_APPROVE]
            for r in responses
        )
        all_reject = all(
            r.judgment in [Judgment.REJECT, Judgment.STRONG_REJECT]
            for r in responses
        )

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
        common_risks = [
            term for term, count in risk_counts.items()
            if count >= majority_threshold
        ]

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
            term for term, count in question_counts.items()
            if count >= majority_threshold
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
        neutral_count: int
    ) -> tuple[Judgment, str]:
        """Determine final committee judgment."""
        total = len(responses)

        # Unanimous decisions
        if approve_count == total:
            return (
                Judgment.STRONG_APPROVE,
                "Unanimous approval from all advisors. Proceed with confidence."
            )

        if reject_count == total:
            return (
                Judgment.STRONG_REJECT,
                "Unanimous rejection from all advisors. Do not proceed."
            )

        # Majority decisions
        if approve_count > total / 2:
            return (
                Judgment.APPROVE,
                f"Majority approval ({approve_count}/{total}). Proceed with noted concerns."
            )

        if reject_count > total / 2:
            return (
                Judgment.REJECT,
                f"Majority rejection ({reject_count}/{total}). Significant concerns exist."
            )

        # Split decision
        return (
            Judgment.NEUTRAL,
            f"Split decision ({approve_count} approve, {reject_count} reject, "
            f"{neutral_count} neutral). Additional analysis recommended."
        )

    def _determine_confidence(
        self,
        responses: list[AdvisorResponse],
        approve_count: int,
        reject_count: int
    ) -> ConfidenceLevel:
        """Determine committee confidence based on consensus."""
        total = len(responses)

        # Strong consensus = high confidence
        if approve_count == total or reject_count == total:
            # Check individual confidence levels
            avg_confidence = sum(
                self._confidence_to_num(r.confidence) for r in responses
            ) / total
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
        self,
        responses: list[AdvisorResponse],
        judgment: Judgment
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
        self,
        responses: list[AdvisorResponse],
        judgment: Judgment
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
            "concentration", "earnings", "volatility", "iv", "regime",
            "assignment", "bias", "risk", "premium", "delta", "gamma",
            "theta", "vega", "sector", "position", "sizing", "kelly",
            "statistical", "probability", "edge", "model", "validation"
        }

        text_lower = text.lower()
        found = [kw for kw in keywords if kw in text_lower]

        return found[:3]


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
    lines.append(f"Votes: {output.approval_count} approve, "
                 f"{output.rejection_count} reject, {output.neutral_count} neutral")
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
