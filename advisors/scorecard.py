"""
Advisor Performance Scorecard

Tracks and analyzes advisor recommendation accuracy, calibration, and performance.
Essential for understanding advisor reliability and making data-driven improvements.

Key Metrics:
- Classification accuracy (approval/rejection correctness)
- Calibration (confidence vs actual accuracy)
- Financial performance (P&L of recommendations)
- Regime-specific analysis
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Trade outcome classification."""
    WIN = "win"           # Profitable trade
    LOSS = "loss"         # Unprofitable trade
    SCRATCH = "scratch"   # Break-even (within transaction costs)
    PENDING = "pending"   # Not yet closed


class JudgmentType(Enum):
    """Advisor judgment classification."""
    STRONG_APPROVE = "strong_approve"
    APPROVE = "approve"
    NEUTRAL = "neutral"
    REJECT = "reject"
    STRONG_REJECT = "strong_reject"


class ConfidenceLevel(Enum):
    """Advisor confidence levels."""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class AdvisorPrediction:
    """Record of an advisor's prediction."""
    prediction_id: str
    advisor_name: str
    judgment: JudgmentType
    confidence: ConfidenceLevel
    timestamp: datetime

    # Trade parameters
    ticker: str
    strategy: str  # e.g., "cash_secured_put", "covered_call"
    strike: float
    expiration_date: str
    premium: float

    # Context
    market_regime: str | None = None
    iv_rank: float | None = None
    key_reasons: list[str] = field(default_factory=list)
    hidden_risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.prediction_id,
            "advisor_name": self.advisor_name,
            "judgment": self.judgment.value,
            "confidence": self.confidence.value,
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "strategy": self.strategy,
            "strike": self.strike,
            "expiration_date": self.expiration_date,
            "premium": self.premium,
            "market_regime": self.market_regime,
            "iv_rank": self.iv_rank,
            "key_reasons": self.key_reasons,
            "hidden_risks": self.hidden_risks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdvisorPrediction":
        return cls(
            prediction_id=data["prediction_id"],
            advisor_name=data["advisor_name"],
            judgment=JudgmentType(data["judgment"]),
            confidence=ConfidenceLevel(data["confidence"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ticker=data["ticker"],
            strategy=data["strategy"],
            strike=data["strike"],
            expiration_date=data["expiration_date"],
            premium=data["premium"],
            market_regime=data.get("market_regime"),
            iv_rank=data.get("iv_rank"),
            key_reasons=data.get("key_reasons", []),
            hidden_risks=data.get("hidden_risks", []),
        )


@dataclass
class TradeOutcome:
    """Record of actual trade outcome."""
    outcome_id: str
    prediction_id: str  # Links to AdvisorPrediction
    outcome: OutcomeType
    pnl: float
    pnl_percent: float
    close_timestamp: datetime
    hold_days: int

    # Additional context
    exit_reason: str | None = None  # "expired", "early_close", "assigned", etc.
    max_drawdown: float | None = None

    def to_dict(self) -> dict:
        return {
            "outcome_id": self.outcome_id,
            "prediction_id": self.prediction_id,
            "outcome": self.outcome.value,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "close_timestamp": self.close_timestamp.isoformat(),
            "hold_days": self.hold_days,
            "exit_reason": self.exit_reason,
            "max_drawdown": self.max_drawdown,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TradeOutcome":
        return cls(
            outcome_id=data["outcome_id"],
            prediction_id=data["prediction_id"],
            outcome=OutcomeType(data["outcome"]),
            pnl=data["pnl"],
            pnl_percent=data["pnl_percent"],
            close_timestamp=datetime.fromisoformat(data["close_timestamp"]),
            hold_days=data["hold_days"],
            exit_reason=data.get("exit_reason"),
            max_drawdown=data.get("max_drawdown"),
        )


@dataclass
class CalibrationBucket:
    """Calibration data for a confidence bucket."""
    confidence_level: ConfidenceLevel
    total_predictions: int
    correct_predictions: int
    expected_accuracy: float  # Based on confidence level
    actual_accuracy: float

    @property
    def calibration_error(self) -> float:
        """Absolute calibration error for this bucket."""
        return abs(self.expected_accuracy - self.actual_accuracy)


@dataclass
class AdvisorMetrics:
    """Performance metrics for a single advisor."""
    advisor_name: str
    period_start: datetime
    period_end: datetime

    # Classification metrics
    total_predictions: int = 0
    total_outcomes: int = 0  # Predictions with known outcomes

    # Approval metrics
    approvals: int = 0
    approval_wins: int = 0
    approval_losses: int = 0

    # Rejection metrics
    rejections: int = 0
    rejection_would_have_won: int = 0  # Trades that would have been profitable
    rejection_would_have_lost: int = 0

    # Financial metrics
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float | None = None

    # Calibration
    calibration_buckets: list[CalibrationBucket] = field(default_factory=list)
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0

    # Regime breakdown
    regime_metrics: dict[str, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Overall accuracy: correct approvals + correct rejections / total."""
        if self.total_outcomes == 0:
            return 0.0
        correct = self.approval_wins + self.rejection_would_have_lost
        return correct / self.total_outcomes

    @property
    def precision(self) -> float:
        """Precision: wins / approvals."""
        if self.approvals == 0:
            return 0.0
        return self.approval_wins / self.approvals

    @property
    def recall(self) -> float:
        """Recall: wins / (wins + missed opportunities)."""
        actual_wins = self.approval_wins + self.rejection_would_have_won
        if actual_wins == 0:
            return 0.0
        return self.approval_wins / actual_wins

    @property
    def f1_score(self) -> float:
        """F1 score: harmonic mean of precision and recall."""
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def to_dict(self) -> dict:
        return {
            "advisor_name": self.advisor_name,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_predictions": self.total_predictions,
            "total_outcomes": self.total_outcomes,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "approvals": self.approvals,
            "approval_wins": self.approval_wins,
            "approval_losses": self.approval_losses,
            "rejections": self.rejections,
            "rejection_would_have_won": self.rejection_would_have_won,
            "rejection_would_have_lost": self.rejection_would_have_lost,
            "total_pnl": self.total_pnl,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "regime_metrics": self.regime_metrics,
        }


class AdvisorScorecard:
    """
    Tracks and analyzes advisor performance over time.

    Features:
    - Stores predictions and outcomes
    - Calculates accuracy, calibration, and financial metrics
    - Supports regime-specific analysis
    - Generates performance reports
    """

    # Expected accuracy by confidence level (calibration targets)
    CONFIDENCE_TARGETS = {
        ConfidenceLevel.VERY_LOW: 0.50,
        ConfidenceLevel.LOW: 0.60,
        ConfidenceLevel.MEDIUM: 0.70,
        ConfidenceLevel.HIGH: 0.80,
        ConfidenceLevel.VERY_HIGH: 0.90,
    }

    def __init__(self, storage_path: Path | str | None = None):
        """
        Initialize scorecard.

        Args:
            storage_path: Path to store prediction/outcome data. If None, in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.predictions: dict[str, AdvisorPrediction] = {}
        self.outcomes: dict[str, TradeOutcome] = {}

        if self.storage_path:
            self._load_data()

    def record_prediction(self, prediction: AdvisorPrediction) -> None:
        """Record a new advisor prediction."""
        self.predictions[prediction.prediction_id] = prediction
        logger.info(
            f"Recorded prediction {prediction.prediction_id} "
            f"from {prediction.advisor_name}: {prediction.judgment.value}"
        )
        if self.storage_path:
            self._save_data()

    def record_outcome(self, outcome: TradeOutcome) -> None:
        """Record the outcome of a trade."""
        if outcome.prediction_id not in self.predictions:
            logger.warning(
                f"Recording outcome for unknown prediction: {outcome.prediction_id}"
            )
        self.outcomes[outcome.outcome_id] = outcome
        logger.info(
            f"Recorded outcome {outcome.outcome_id}: "
            f"{outcome.outcome.value} P&L={outcome.pnl:.2f}"
        )
        if self.storage_path:
            self._save_data()

    def get_metrics(
        self,
        advisor_name: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        regime: str | None = None,
    ) -> AdvisorMetrics | dict[str, AdvisorMetrics]:
        """
        Calculate metrics for advisor(s).

        Args:
            advisor_name: Specific advisor or None for all
            start_date: Filter predictions after this date
            end_date: Filter predictions before this date
            regime: Filter by market regime

        Returns:
            AdvisorMetrics for single advisor or dict of metrics for all
        """
        # Filter predictions
        filtered_predictions = self._filter_predictions(
            advisor_name, start_date, end_date, regime
        )

        # Group by advisor
        by_advisor: dict[str, list[AdvisorPrediction]] = {}
        for pred in filtered_predictions:
            by_advisor.setdefault(pred.advisor_name, []).append(pred)

        # Calculate metrics for each advisor
        results = {}
        for name, preds in by_advisor.items():
            metrics = self._calculate_metrics(name, preds, start_date, end_date)
            results[name] = metrics

        if advisor_name and advisor_name in results:
            return results[advisor_name]
        return results

    def get_calibration_curve(
        self,
        advisor_name: str,
    ) -> list[CalibrationBucket]:
        """Get calibration data for an advisor."""
        predictions = [
            p for p in self.predictions.values()
            if p.advisor_name == advisor_name
        ]

        buckets = []
        for level in ConfidenceLevel:
            level_preds = [p for p in predictions if p.confidence == level]
            if not level_preds:
                continue

            # Count correct predictions
            correct = 0
            total = 0
            for pred in level_preds:
                outcome = self._get_outcome_for_prediction(pred.prediction_id)
                if outcome and outcome.outcome != OutcomeType.PENDING:
                    total += 1
                    # Correct if: approved and won, or rejected and would have lost
                    is_approval = pred.judgment in (
                        JudgmentType.APPROVE, JudgmentType.STRONG_APPROVE
                    )
                    is_win = outcome.outcome == OutcomeType.WIN
                    if (is_approval and is_win) or (not is_approval and not is_win):
                        correct += 1

            if total > 0:
                buckets.append(CalibrationBucket(
                    confidence_level=level,
                    total_predictions=total,
                    correct_predictions=correct,
                    expected_accuracy=self.CONFIDENCE_TARGETS[level],
                    actual_accuracy=correct / total,
                ))

        return buckets

    def generate_report(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive performance report."""
        all_metrics = self.get_metrics(start_date=start_date, end_date=end_date)

        report = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "summary": {
                "total_predictions": len(self.predictions),
                "total_outcomes": len(self.outcomes),
                "advisors_tracked": len(all_metrics),
            },
            "advisor_metrics": {
                name: metrics.to_dict()
                for name, metrics in all_metrics.items()
            } if isinstance(all_metrics, dict) else {},
            "calibration": {},
            "recommendations": [],
        }

        # Add calibration data
        for advisor_name in (all_metrics.keys() if isinstance(all_metrics, dict) else []):
            buckets = self.get_calibration_curve(advisor_name)
            report["calibration"][advisor_name] = [
                {
                    "confidence": b.confidence_level.name,
                    "expected": b.expected_accuracy,
                    "actual": b.actual_accuracy,
                    "error": b.calibration_error,
                    "n": b.total_predictions,
                }
                for b in buckets
            ]

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(all_metrics)

        return report

    def _filter_predictions(
        self,
        advisor_name: str | None,
        start_date: datetime | None,
        end_date: datetime | None,
        regime: str | None,
    ) -> list[AdvisorPrediction]:
        """Filter predictions by criteria."""
        result = []
        for pred in self.predictions.values():
            if advisor_name and pred.advisor_name != advisor_name:
                continue
            if start_date and pred.timestamp < start_date:
                continue
            if end_date and pred.timestamp > end_date:
                continue
            if regime and pred.market_regime != regime:
                continue
            result.append(pred)
        return result

    def _get_outcome_for_prediction(self, prediction_id: str) -> TradeOutcome | None:
        """Get outcome for a prediction."""
        for outcome in self.outcomes.values():
            if outcome.prediction_id == prediction_id:
                return outcome
        return None

    def _calculate_metrics(
        self,
        advisor_name: str,
        predictions: list[AdvisorPrediction],
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> AdvisorMetrics:
        """Calculate metrics for a set of predictions."""
        metrics = AdvisorMetrics(
            advisor_name=advisor_name,
            period_start=start_date or datetime.min,
            period_end=end_date or datetime.max,
            total_predictions=len(predictions),
        )

        pnls = []

        for pred in predictions:
            outcome = self._get_outcome_for_prediction(pred.prediction_id)
            is_approval = pred.judgment in (
                JudgmentType.APPROVE, JudgmentType.STRONG_APPROVE
            )

            if is_approval:
                metrics.approvals += 1
                if outcome and outcome.outcome != OutcomeType.PENDING:
                    metrics.total_outcomes += 1
                    if outcome.outcome == OutcomeType.WIN:
                        metrics.approval_wins += 1
                        pnls.append(outcome.pnl)
                    else:
                        metrics.approval_losses += 1
                        pnls.append(outcome.pnl)
            else:
                metrics.rejections += 1
                if outcome and outcome.outcome != OutcomeType.PENDING:
                    metrics.total_outcomes += 1
                    # For rejections, we track "would have" outcomes
                    if outcome.outcome == OutcomeType.WIN:
                        metrics.rejection_would_have_won += 1
                    else:
                        metrics.rejection_would_have_lost += 1

        # Financial metrics
        if pnls:
            metrics.total_pnl = sum(pnls)
            metrics.avg_pnl_per_trade = metrics.total_pnl / len(pnls)
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            if metrics.approvals > 0:
                metrics.win_rate = len(wins) / len(pnls)

            if losses:
                total_wins = sum(wins) if wins else 0
                total_losses = abs(sum(losses))
                if total_losses > 0:
                    metrics.profit_factor = total_wins / total_losses

            # Sharpe ratio (annualized, assuming daily returns)
            if len(pnls) > 1:
                returns = np.array(pnls)
                if returns.std() > 0:
                    metrics.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

        # Calibration
        buckets = self.get_calibration_curve(advisor_name)
        metrics.calibration_buckets = buckets
        if buckets:
            total_n = sum(b.total_predictions for b in buckets)
            if total_n > 0:
                metrics.expected_calibration_error = sum(
                    b.calibration_error * b.total_predictions / total_n
                    for b in buckets
                )

        return metrics

    def _generate_recommendations(
        self,
        metrics: dict[str, AdvisorMetrics]
    ) -> list[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        for name, m in metrics.items():
            # Check for overconfidence
            for bucket in m.calibration_buckets:
                if bucket.calibration_error > 0.15:
                    if bucket.actual_accuracy < bucket.expected_accuracy:
                        recommendations.append(
                            f"{name}: Overconfident at {bucket.confidence_level.name} level "
                            f"(expected {bucket.expected_accuracy:.0%}, "
                            f"actual {bucket.actual_accuracy:.0%}). "
                            f"Consider lowering confidence thresholds."
                        )
                    else:
                        recommendations.append(
                            f"{name}: Underconfident at {bucket.confidence_level.name} level. "
                            f"Consider raising confidence thresholds."
                        )

            # Check for low precision
            if m.precision < 0.5 and m.approvals > 10:
                recommendations.append(
                    f"{name}: Low precision ({m.precision:.0%}). "
                    f"Consider stricter approval criteria."
                )

            # Check for missed opportunities
            if m.rejection_would_have_won > m.approval_wins and m.rejections > 10:
                recommendations.append(
                    f"{name}: Missing profitable trades ({m.rejection_would_have_won} "
                    f"rejected trades were winners). Consider relaxing rejection criteria."
                )

        return recommendations

    def _load_data(self) -> None:
        """Load data from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        predictions_file = self.storage_path / "predictions.json"
        outcomes_file = self.storage_path / "outcomes.json"

        if predictions_file.exists():
            with open(predictions_file) as f:
                data = json.load(f)
                self.predictions = {
                    k: AdvisorPrediction.from_dict(v)
                    for k, v in data.items()
                }

        if outcomes_file.exists():
            with open(outcomes_file) as f:
                data = json.load(f)
                self.outcomes = {
                    k: TradeOutcome.from_dict(v)
                    for k, v in data.items()
                }

    def _save_data(self) -> None:
        """Save data to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        predictions_file = self.storage_path / "predictions.json"
        outcomes_file = self.storage_path / "outcomes.json"

        with open(predictions_file, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.predictions.items()},
                f,
                indent=2
            )

        with open(outcomes_file, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.outcomes.items()},
                f,
                indent=2
            )
