"""
ML Model for Wheel Strategy Entry/Exit Timing.

STATUS: RESEARCH ONLY — NOT WIRED INTO PRODUCTION EV RANKING.

Audit (2026-04-14):
  * This module is NOT called by engine/ev_engine.py,
    engine/wheel_runner.py::rank_candidates_by_ev, or any live HTTP
    endpoint. It is imported only by src/backtest/wheel_backtest.py as
    an optional offline backtest feature.
  * Target labels at ``create_target`` / ``create_put_outcome_target``
    are correctly forward-looking (that is how supervised labels
    work), but the training code uses ``TimeSeriesSplit(n_splits=5)``
    WITHOUT a gap equal to ``forward_days``. This causes
    fold-boundary leakage: the validation fold's features overlap
    with the preceding fold's label window. Reported AUC / accuracy
    is consequently inflated.
  * The ``gap`` parameter (added below) defaults to the
    ``forward_days`` of the target so cross-validation now purges
    the label horizon. Without this, model metrics are not
    trustworthy.
  * Predictions from this module are NOT calibrated (no isotonic /
    Platt). Do not use raw predict_proba as EV inputs.
  * If this module is ever promoted to production, it MUST:
      1. Ship a calibration wrapper validated by
         ``ml/model_governance.py::DriftDetector.check_calibration``.
      2. Be gated by the same ``enforce_chain_quality_gate`` /
         ``enforce_history_gate`` flags as the EV ranker.
      3. Feed the ``EVEngine`` as an adjustment to the physical
         forward distribution, NOT as a direct ranking score.

Predicts optimal entry points for selling puts based on:
- Technical indicators
- Volatility regime
- Historical forward returns

Target: Forward N-day return > -X% (put expires worthless or minor loss)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import joblib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler


@dataclass
class WheelModelConfig:
    """Configuration for wheel ML model."""
    # Target definition
    forward_days: int = 30          # DTE for put
    put_delta: float = 0.30         # ~30 delta put
    max_loss_pct: float = 0.10      # Max acceptable drawdown (strike breached by 10%)
    min_premium_pct: float = 0.015  # Min premium as % of strike

    # Model params
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_leaf: int = 50

    # Training
    train_years: int = 5
    validation_years: int = 1
    min_train_samples: int = 5000


# Feature columns used for prediction
FEATURE_COLUMNS = [
    # Returns
    'return_1d', 'return_5d', 'return_21d', 'return_63d',
    # Moving averages
    'ma_ratio_20', 'ma_ratio_50', 'ma_ratio_200',
    # Momentum/oscillators
    'rsi_14', 'macd', 'macd_hist', 'bb_position',
    # Volatility
    'atr_pct_14', 'realized_vol_20', 'realized_vol_60', 'vol_ratio',
    'parkinson_vol_20', 'rv_rank_252',
    # Trend
    'trend_20d', 'trend_60d', 'above_sma_20', 'above_sma_50', 'above_sma_200',
    # Support/resistance
    'dist_to_support', 'dist_to_resistance', 'channel_position_20',
    # Drawdown
    'drawdown_52w',
    # Volume
    'volume_ratio_20'
]


def create_target(
    df: pd.DataFrame,
    forward_days: int = 30,
    max_drawdown_pct: float = 0.10,
    delta: float = 0.30
) -> pd.Series:
    """
    Create binary target for wheel entry.

    Target = 1 if:
    - Forward max drawdown over N days < threshold (put likely expires worthless)

    Args:
        df: DataFrame with close prices (per ticker)
        forward_days: Forward looking period
        max_drawdown_pct: Max acceptable drawdown threshold
        delta: Put delta (affects strike placement)

    Returns:
        Binary target series (1 = good entry, 0 = bad entry)
    """
    # Calculate forward max drawdown
    # For a 30-delta put, strike is roughly 5-8% below current price
    # We want the stock not to drop below strike by more than premium collected

    close = df['close'].values
    n = len(close)

    # Calculate forward min price over window
    forward_min = pd.Series(close).shift(-forward_days).rolling(forward_days, min_periods=1).min()

    # For proper forward-looking, we need to calculate min of NEXT N days
    # Using a workaround since pandas rolling is backward-looking
    forward_min_arr = np.full(n, np.nan)
    for i in range(n - forward_days):
        forward_min_arr[i] = close[i+1:i+1+forward_days].min()

    forward_min = pd.Series(forward_min_arr, index=df.index)

    # Max drawdown = (current - forward_min) / current
    max_dd = (df['close'] - forward_min) / df['close']

    # Target: 1 if drawdown is acceptable
    # For a 30-delta put ~5% OTM, we win if stock doesn't drop more than ~10-15%
    target = (max_dd < max_drawdown_pct).astype(int)

    return target


def create_put_outcome_target(
    df: pd.DataFrame,
    forward_days: int = 30,
    otm_pct: float = 0.05,
    premium_pct: float = 0.02
) -> pd.Series:
    """
    More realistic put outcome target.

    Simulates selling an OTM put and checks if it would have been profitable.

    Args:
        df: DataFrame with close prices
        forward_days: DTE
        otm_pct: How far OTM (e.g., 0.05 = 5% below current)
        premium_pct: Premium collected as % of strike

    Returns:
        Target: 1 if profitable, 0 if loss
    """
    close = df['close'].values
    n = len(close)

    # Strike = current price * (1 - otm_pct)
    strikes = close * (1 - otm_pct)
    premium = strikes * premium_pct

    # Forward price at expiration
    forward_price = np.full(n, np.nan)
    for i in range(n - forward_days):
        forward_price[i] = close[i + forward_days]

    forward_price = pd.Series(forward_price, index=df.index)

    # P&L calculation
    # If stock > strike: profit = full premium
    # If stock < strike: profit = premium - (strike - stock)
    pnl = np.where(
        forward_price >= strikes,
        premium,
        premium - (strikes - forward_price)
    )

    # Target: 1 if P&L > 0
    target = (pnl > 0).astype(int)

    return pd.Series(target, index=df.index)


def prepare_training_data(
    df: pd.DataFrame,
    config: WheelModelConfig,
    target_type: str = 'put_outcome'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.

    Args:
        df: Full processed DataFrame
        config: Model configuration
        target_type: 'drawdown' or 'put_outcome'

    Returns:
        X (features), y (target)
    """
    # Filter to available features
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    if len(available_features) < 10:
        raise ValueError(f"Only {len(available_features)} features available. Need at least 10.")

    # Create target per ticker
    targets = []
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()

        if target_type == 'drawdown':
            ticker_df['target'] = create_target(
                ticker_df,
                forward_days=config.forward_days,
                max_drawdown_pct=config.max_loss_pct
            )
        else:
            ticker_df['target'] = create_put_outcome_target(
                ticker_df,
                forward_days=config.forward_days,
                otm_pct=0.05,
                premium_pct=config.min_premium_pct
            )
        targets.append(ticker_df)

    df_with_target = pd.concat(targets, ignore_index=True)

    # Drop NaN
    df_clean = df_with_target.dropna(subset=available_features + ['target'])

    X = df_clean[available_features]
    y = df_clean['target']

    return X, y, df_clean


class WheelEntryModel:
    """
    ML model for wheel entry timing.
    """

    def __init__(self, config: Optional[WheelModelConfig] = None):
        self.config = config or WheelModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validate: bool = True
    ) -> Dict:
        """
        Train the wheel entry model.

        Args:
            X: Feature matrix
            y: Target labels
            validate: Whether to run time-series cross-validation

        Returns:
            Training metrics dict
        """
        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Initialize model
        self.model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=42,
            verbose=0
        )

        metrics = {}

        if validate:
            # Time-series cross-validation WITH gap equal to the label
            # horizon. Without this gap the validation fold's features
            # overlap the preceding fold's label window, producing
            # inflated AUC. This is the standard López de Prado
            # "purged CV" requirement for overlapping labels.
            forward_days = int(getattr(self.config, "forward_days", 30) or 30)
            tscv = TimeSeriesSplit(n_splits=5, gap=max(forward_days, 1))
            cv_scores = cross_val_score(
                self.model, X_scaled, y,
                cv=tscv, scoring='roc_auc'
            )
            metrics['cv_auc_mean'] = cv_scores.mean()
            metrics['cv_auc_std'] = cv_scores.std()
            metrics['cv_gap'] = forward_days

        # Fit on full data
        self.model.fit(X_scaled, y)

        # Training metrics
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)[:, 1]

        metrics['train_accuracy'] = accuracy_score(y, y_pred)
        metrics['train_precision'] = precision_score(y, y_pred)
        metrics['train_recall'] = recall_score(y, y_pred)
        metrics['train_f1'] = f1_score(y, y_pred)
        metrics['train_auc'] = roc_auc_score(y, y_proba)
        metrics['positive_rate'] = y.mean()

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of successful entry.

        Args:
            X: Feature matrix

        Returns:
            Probability of class 1 (good entry)
        """
        if self.model is None:
            raise ValueError("Model not trained")

        # Align features
        X_aligned = X[self.feature_names]
        X_scaled = self.scaler.transform(X_aligned)

        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Binary prediction.

        Args:
            X: Feature matrix
            threshold: Probability threshold

        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances."""
        if self.model is None:
            raise ValueError("Model not trained")

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }, path)

    @classmethod
    def load(cls, path: str) -> 'WheelEntryModel':
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(config=data['config'])
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        return instance


def train_wheel_model(
    data_path: str,
    output_path: str = 'models/wheel_entry_model.joblib',
    config: Optional[WheelModelConfig] = None
) -> Dict:
    """
    Full training pipeline.

    Args:
        data_path: Path to processed parquet file
        output_path: Where to save trained model
        config: Model configuration

    Returns:
        Training metrics
    """
    config = config or WheelModelConfig()

    print("Loading data...")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} rows, {df['ticker'].nunique()} tickers")

    print("Preparing training data...")
    X, y, df_clean = prepare_training_data(df, config)
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {len(X):,}")
    print(f"  Positive rate: {y.mean():.1%}")

    print("Training model...")
    model = WheelEntryModel(config)
    metrics = model.train(X, y, validate=True)

    print("\nTraining Results:")
    print(f"  CV AUC: {metrics['cv_auc_mean']:.3f} (+/- {metrics['cv_auc_std']:.3f})")
    print(f"  Train AUC: {metrics['train_auc']:.3f}")
    print(f"  Precision: {metrics['train_precision']:.3f}")
    print(f"  Recall: {metrics['train_recall']:.3f}")

    print("\nTop Features:")
    importance = model.get_feature_importance()
    print(importance.head(10).to_string(index=False))

    # Save model
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to {output_path}")

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to processed features parquet")
    parser.add_argument("-o", "--output", default="models/wheel_entry_model.joblib")
    args = parser.parse_args()

    train_wheel_model(args.data_path, args.output)
