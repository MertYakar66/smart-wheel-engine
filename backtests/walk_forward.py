"""
Walk-Forward Validation Framework

Institutional-grade validation methodology:
- Anchored walk-forward (expanding window)
- Rolling walk-forward (fixed window)
- Combinatorial purged cross-validation
- Out-of-sample performance tracking
- Parameter stability analysis

Key principle: Never use future data to make past decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import date, datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class ValidationMethod(Enum):
    """Walk-forward validation methods."""
    ANCHORED = "anchored"  # Expanding training window
    ROLLING = "rolling"    # Fixed training window
    PURGED_KFOLD = "purged_kfold"  # Purged cross-validation


@dataclass
class ValidationFold:
    """Single validation fold."""
    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_days: int = 0  # Gap between train and test

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"Train [{self.train_start} -> {self.train_end}] ({self.train_days}d), "
            f"Test [{self.test_start} -> {self.test_end}] ({self.test_days}d)"
        )


@dataclass
class FoldResult:
    """Results from single validation fold."""
    fold: ValidationFold
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: Optional[pd.DataFrame] = None
    model_params: Optional[Dict] = None

    @property
    def overfit_ratio(self) -> float:
        """Ratio of test to train performance (Sharpe or returns)."""
        train_sharpe = self.train_metrics.get('sharpe_ratio', 0)
        test_sharpe = self.test_metrics.get('sharpe_ratio', 0)
        if train_sharpe == 0:
            return 0
        return test_sharpe / train_sharpe

    @property
    def is_overfit(self) -> bool:
        """Check if likely overfit (test << train)."""
        return self.overfit_ratio < 0.5


@dataclass
class WalkForwardResult:
    """Complete walk-forward validation result."""
    method: ValidationMethod
    folds: List[FoldResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    parameter_stability: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.folds:
            self._compute_aggregate_metrics()

    def _compute_aggregate_metrics(self):
        """Compute aggregate metrics across folds."""
        if not self.folds:
            return

        # Collect test metrics
        test_sharpes = [f.test_metrics.get('sharpe_ratio', 0) for f in self.folds]
        test_returns = [f.test_metrics.get('total_return', 0) for f in self.folds]
        test_drawdowns = [f.test_metrics.get('max_drawdown', 0) for f in self.folds]

        self.aggregate_metrics = {
            'mean_sharpe': np.mean(test_sharpes),
            'std_sharpe': np.std(test_sharpes),
            'min_sharpe': np.min(test_sharpes),
            'max_sharpe': np.max(test_sharpes),
            'mean_return': np.mean(test_returns),
            'std_return': np.std(test_returns),
            'mean_drawdown': np.mean(test_drawdowns),
            'max_drawdown': np.max(test_drawdowns),
            'pct_profitable_folds': np.mean([r > 0 for r in test_returns]),
            'mean_overfit_ratio': np.mean([f.overfit_ratio for f in self.folds]),
            'pct_overfit_folds': np.mean([f.is_overfit for f in self.folds])
        }

    @property
    def is_robust(self) -> bool:
        """Check if strategy shows robust out-of-sample performance."""
        if not self.aggregate_metrics:
            return False

        # Criteria for robustness
        checks = [
            self.aggregate_metrics.get('mean_sharpe', 0) > 0.5,
            self.aggregate_metrics.get('pct_profitable_folds', 0) >= 0.6,
            self.aggregate_metrics.get('pct_overfit_folds', 1) < 0.4,
            self.aggregate_metrics.get('std_sharpe', float('inf')) < 1.0
        ]
        return all(checks)

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            f"Walk-Forward Validation Results ({self.method.value})",
            "=" * 50,
            f"Number of folds: {len(self.folds)}",
            "",
            "Aggregate Test Performance:",
            f"  Mean Sharpe:    {self.aggregate_metrics.get('mean_sharpe', 0):.2f} "
            f"(+/- {self.aggregate_metrics.get('std_sharpe', 0):.2f})",
            f"  Mean Return:    {self.aggregate_metrics.get('mean_return', 0):.1%}",
            f"  Max Drawdown:   {self.aggregate_metrics.get('max_drawdown', 0):.1%}",
            f"  Profitable:     {self.aggregate_metrics.get('pct_profitable_folds', 0):.0%} of folds",
            "",
            "Robustness Indicators:",
            f"  Overfit Ratio:  {self.aggregate_metrics.get('mean_overfit_ratio', 0):.2f}",
            f"  Overfit Folds:  {self.aggregate_metrics.get('pct_overfit_folds', 0):.0%}",
            f"  Robust:         {'YES' if self.is_robust else 'NO'}",
        ]
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward validation engine.

    Implements proper out-of-sample testing with:
    - No lookahead bias
    - Embargo periods between train/test
    - Multiple validation methodologies
    """

    def __init__(
        self,
        method: ValidationMethod = ValidationMethod.ANCHORED,
        n_folds: int = 5,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,    # 1 quarter
        embargo_days: int = 5,         # Gap to prevent leakage
        min_train_days: int = 126      # Minimum 6 months training
    ):
        self.method = method
        self.n_folds = n_folds
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.embargo_days = embargo_days
        self.min_train_days = min_train_days

    def generate_folds(
        self,
        start_date: date,
        end_date: date
    ) -> List[ValidationFold]:
        """Generate validation folds based on method."""
        if self.method == ValidationMethod.ANCHORED:
            return self._generate_anchored_folds(start_date, end_date)
        elif self.method == ValidationMethod.ROLLING:
            return self._generate_rolling_folds(start_date, end_date)
        elif self.method == ValidationMethod.PURGED_KFOLD:
            return self._generate_purged_kfold(start_date, end_date)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _generate_anchored_folds(
        self,
        start_date: date,
        end_date: date
    ) -> List[ValidationFold]:
        """
        Anchored walk-forward: training window expands over time.

        Train: [start -> fold_end]
        Test: [fold_end + embargo -> fold_end + embargo + test_period]
        """
        folds = []
        total_days = (end_date - start_date).days

        # Reserve space for minimum training + embargo + test
        min_required = self.min_train_days + self.embargo_days + self.test_period_days
        if total_days < min_required:
            raise ValueError(f"Need at least {min_required} days, got {total_days}")

        # Calculate fold boundaries
        available_for_test = total_days - self.min_train_days - self.embargo_days
        test_per_fold = available_for_test // self.n_folds

        current_train_end = start_date + timedelta(days=self.min_train_days)

        for i in range(self.n_folds):
            test_start = current_train_end + timedelta(days=self.embargo_days)
            test_end = test_start + timedelta(days=test_per_fold)

            if test_end > end_date:
                test_end = end_date

            fold = ValidationFold(
                fold_id=i + 1,
                train_start=start_date,
                train_end=current_train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=self.embargo_days
            )
            folds.append(fold)

            # Expand training window for next fold
            current_train_end = test_end

        return folds

    def _generate_rolling_folds(
        self,
        start_date: date,
        end_date: date
    ) -> List[ValidationFold]:
        """
        Rolling walk-forward: fixed-size training window.

        Train: [fold_start -> fold_start + train_period]
        Test: [train_end + embargo -> train_end + embargo + test_period]
        """
        folds = []
        total_days = (end_date - start_date).days

        fold_size = self.train_period_days + self.embargo_days + self.test_period_days
        step_size = (total_days - fold_size) // (self.n_folds - 1) if self.n_folds > 1 else 0

        current_start = start_date

        for i in range(self.n_folds):
            train_end = current_start + timedelta(days=self.train_period_days)
            test_start = train_end + timedelta(days=self.embargo_days)
            test_end = test_start + timedelta(days=self.test_period_days)

            if test_end > end_date:
                break

            fold = ValidationFold(
                fold_id=i + 1,
                train_start=current_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=self.embargo_days
            )
            folds.append(fold)

            current_start = current_start + timedelta(days=step_size)

        return folds

    def _generate_purged_kfold(
        self,
        start_date: date,
        end_date: date
    ) -> List[ValidationFold]:
        """
        Purged K-fold: each fold tests on different time segment.

        Unlike rolling, ensures no overlap and complete coverage.
        """
        folds = []
        total_days = (end_date - start_date).days

        test_size = total_days // self.n_folds
        embargo = self.embargo_days

        for i in range(self.n_folds):
            # Test period
            test_start = start_date + timedelta(days=i * test_size + embargo)
            test_end = start_date + timedelta(days=(i + 1) * test_size - embargo)

            # Train on everything except test period (with embargo)
            # For simplicity, use data before test period
            train_start = start_date
            train_end = test_start - timedelta(days=embargo)

            if train_end <= train_start:
                continue

            fold = ValidationFold(
                fold_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_days=embargo
            )
            folds.append(fold)

        return folds

    def run_validation(
        self,
        data: pd.DataFrame,
        train_func: Callable[[pd.DataFrame], Any],
        test_func: Callable[[Any, pd.DataFrame], Dict[str, float]],
        date_column: str = 'date'
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: Full dataset with date column
            train_func: Function that trains model on data, returns model
            test_func: Function that tests model on data, returns metrics dict
            date_column: Name of date column in data

        Returns:
            WalkForwardResult with all fold results
        """
        # Ensure dates are proper format
        if date_column in data.columns:
            dates = pd.to_datetime(data[date_column])
            start_date = dates.min().date()
            end_date = dates.max().date()
        else:
            # Assume index is date
            start_date = data.index.min()
            end_date = data.index.max()
            if hasattr(start_date, 'date'):
                start_date = start_date.date()
                end_date = end_date.date()

        # Generate folds
        folds = self.generate_folds(start_date, end_date)

        # Run each fold
        fold_results = []
        for fold in folds:
            # Split data
            if date_column in data.columns:
                train_mask = (
                    (pd.to_datetime(data[date_column]).dt.date >= fold.train_start) &
                    (pd.to_datetime(data[date_column]).dt.date <= fold.train_end)
                )
                test_mask = (
                    (pd.to_datetime(data[date_column]).dt.date >= fold.test_start) &
                    (pd.to_datetime(data[date_column]).dt.date <= fold.test_end)
                )
            else:
                idx_dates = data.index
                if hasattr(idx_dates[0], 'date'):
                    idx_dates = pd.Series([d.date() for d in idx_dates])
                train_mask = (idx_dates >= fold.train_start) & (idx_dates <= fold.train_end)
                test_mask = (idx_dates >= fold.test_start) & (idx_dates <= fold.test_end)

            train_data = data[train_mask]
            test_data = data[test_mask]

            if len(train_data) == 0 or len(test_data) == 0:
                continue

            # Train
            model = train_func(train_data)

            # Evaluate on both train and test
            train_metrics = test_func(model, train_data)
            test_metrics = test_func(model, test_data)

            result = FoldResult(
                fold=fold,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                model_params=model if isinstance(model, dict) else None
            )
            fold_results.append(result)

        return WalkForwardResult(
            method=self.method,
            folds=fold_results
        )


class ParameterStabilityAnalyzer:
    """
    Analyze parameter stability across walk-forward folds.

    Unstable parameters suggest overfitting.
    """

    @staticmethod
    def analyze_stability(
        fold_results: List[FoldResult],
        param_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze how parameters vary across folds.

        Returns dict of param_name -> {mean, std, cv, range}
        """
        stability = {}

        for param in param_names:
            values = []
            for result in fold_results:
                if result.model_params and param in result.model_params:
                    values.append(result.model_params[param])

            if not values:
                continue

            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)

            stability[param] = {
                'mean': mean,
                'std': std,
                'cv': std / abs(mean) if mean != 0 else float('inf'),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values),
                'is_stable': std / abs(mean) < 0.3 if mean != 0 else False
            }

        return stability


class OutOfSampleTracker:
    """
    Track actual out-of-sample performance over time.

    Compare predictions made in real-time to actual outcomes.
    """

    def __init__(self):
        self.predictions: List[Dict] = []
        self.actuals: List[Dict] = []

    def record_prediction(
        self,
        prediction_date: date,
        symbol: str,
        predicted_outcome: float,
        model_version: str,
        metadata: Optional[Dict] = None
    ):
        """Record a prediction for later comparison."""
        self.predictions.append({
            'date': prediction_date,
            'symbol': symbol,
            'predicted': predicted_outcome,
            'model_version': model_version,
            'metadata': metadata or {}
        })

    def record_actual(
        self,
        prediction_date: date,
        symbol: str,
        actual_outcome: float
    ):
        """Record actual outcome for a prediction."""
        self.actuals.append({
            'date': prediction_date,
            'symbol': symbol,
            'actual': actual_outcome
        })

    def compute_accuracy(
        self,
        model_version: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute prediction accuracy metrics."""
        # Match predictions to actuals
        pred_df = pd.DataFrame(self.predictions)
        actual_df = pd.DataFrame(self.actuals)

        if len(pred_df) == 0 or len(actual_df) == 0:
            return {}

        if model_version:
            pred_df = pred_df[pred_df['model_version'] == model_version]

        merged = pred_df.merge(
            actual_df,
            on=['date', 'symbol'],
            how='inner'
        )

        if len(merged) == 0:
            return {}

        # Calculate metrics
        errors = merged['predicted'] - merged['actual']
        correct_direction = (merged['predicted'] * merged['actual']) > 0

        return {
            'n_predictions': len(merged),
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'direction_accuracy': np.mean(correct_direction),
            'mean_error': np.mean(errors),  # Bias
            'correlation': merged['predicted'].corr(merged['actual'])
        }


def run_anchored_walk_forward(
    data: pd.DataFrame,
    strategy_func: Callable,
    n_folds: int = 5,
    min_train_years: float = 1.0,
    test_months: int = 3
) -> WalkForwardResult:
    """
    Convenience function for anchored walk-forward validation.

    Args:
        data: Dataset with 'date' column
        strategy_func: Function(train_data) -> (model, metrics_func)
        n_folds: Number of test folds
        min_train_years: Minimum training period in years
        test_months: Test period in months

    Returns:
        WalkForwardResult
    """
    validator = WalkForwardValidator(
        method=ValidationMethod.ANCHORED,
        n_folds=n_folds,
        train_period_days=int(min_train_years * 252),
        test_period_days=test_months * 21,
        embargo_days=5
    )

    def train_wrapper(train_data):
        model, _ = strategy_func(train_data)
        return model

    def test_wrapper(model, test_data):
        _, metrics_func = strategy_func(test_data)
        return metrics_func(model, test_data)

    return validator.run_validation(
        data=data,
        train_func=train_wrapper,
        test_func=test_wrapper
    )
