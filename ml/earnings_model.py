"""
Earnings-Aware ML Module for Options Trading

Predicts post-earnings behavior for Wheel strategy optimization:
- IV crush magnitude prediction
- Earnings move vs implied move analysis
- Optimal entry/exit timing around earnings

Key insight: Earnings moves are predictable in STRUCTURE (not direction).
IV systematically drops post-earnings. The question is: by how much?
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import date, datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings


class EarningsAction(Enum):
    """Recommended action around earnings."""
    AVOID = "avoid"              # Don't trade - risk too high
    SELL_PRE_EARNINGS = "sell_pre"   # IV is overpriced, sell before
    SELL_POST_EARNINGS = "sell_post"  # Wait for crush, then sell
    NEUTRAL = "neutral"          # No strong signal


@dataclass
class EarningsFeatures:
    """Feature set for earnings prediction."""
    # Core identifiers
    symbol: str
    earnings_date: date

    # Implied vs Historical
    implied_move: float          # ATM straddle price / underlying (expected %)
    historical_avg_move: float   # Mean of last 8 quarters absolute moves
    historical_max_move: float   # Max move in last 8 quarters
    implied_vs_realized_ratio: float  # implied_move / historical_avg_move

    # IV metrics
    iv_rank_52w: float           # Current IV percentile (0-1)
    iv_30d: float                # 30-day ATM IV
    iv_7d: float                 # 7-day ATM IV (front month)
    iv_term_slope: float         # (iv_30d - iv_7d) / iv_7d

    # Historical earnings behavior
    avg_iv_crush_pct: float      # Average IV drop post-earnings (last 8 quarters)
    earnings_beat_rate: float    # % of quarters with positive surprise
    avg_surprise_magnitude: float # Average |EPS surprise| / estimate

    # Market context
    vix_level: float
    vix_percentile: float
    sector_iv_rank: float        # Sector average IV rank

    # Timing
    days_to_earnings: int
    is_pre_market: bool          # BMO vs AMC

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.implied_move,
            self.historical_avg_move,
            self.historical_max_move,
            self.implied_vs_realized_ratio,
            self.iv_rank_52w,
            self.iv_30d,
            self.iv_7d,
            self.iv_term_slope,
            self.avg_iv_crush_pct,
            self.earnings_beat_rate,
            self.avg_surprise_magnitude,
            self.vix_level,
            self.vix_percentile,
            self.sector_iv_rank,
            self.days_to_earnings,
            float(self.is_pre_market)
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretation."""
        return [
            'implied_move',
            'historical_avg_move',
            'historical_max_move',
            'implied_vs_realized_ratio',
            'iv_rank_52w',
            'iv_30d',
            'iv_7d',
            'iv_term_slope',
            'avg_iv_crush_pct',
            'earnings_beat_rate',
            'avg_surprise_magnitude',
            'vix_level',
            'vix_percentile',
            'sector_iv_rank',
            'days_to_earnings',
            'is_pre_market'
        ]


@dataclass
class EarningsPrediction:
    """Model prediction for earnings event."""
    symbol: str
    earnings_date: date

    # Predictions
    predicted_iv_crush_pct: float     # Expected IV drop (e.g., -0.30 = 30% drop)
    predicted_move_vs_implied: float  # Actual move / implied move ratio

    # Confidence
    confidence: float                 # Model confidence (0-1)

    # Recommended action
    action: EarningsAction
    action_reason: str

    # Risk metrics
    risk_score: float                 # 0-1, higher = more risky
    max_loss_scenario: float          # Worst case P&L estimate

    def __str__(self) -> str:
        return (
            f"{self.symbol} Earnings ({self.earnings_date}):\n"
            f"  Predicted IV Crush: {self.predicted_iv_crush_pct:+.1%}\n"
            f"  Move vs Implied: {self.predicted_move_vs_implied:.2f}x\n"
            f"  Action: {self.action.value} ({self.action_reason})\n"
            f"  Confidence: {self.confidence:.0%}, Risk: {self.risk_score:.0%}"
        )


class EarningsFeatureBuilder:
    """
    Build features for earnings prediction from raw data.

    Requires:
    - Historical earnings data (dates, EPS, surprises)
    - Historical price data (for realized moves)
    - Option data (for IV calculations)
    """

    def __init__(
        self,
        earnings_history: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame],
        lookback_quarters: int = 8
    ):
        """
        Args:
            earnings_history: DataFrame with columns:
                symbol, announcement_date, eps_estimate, eps_actual,
                eps_surprise_pct, iv_before, iv_after, price_move_pct
            price_data: Dict of symbol -> OHLCV DataFrame
            lookback_quarters: Number of past earnings to use
        """
        self.earnings_history = earnings_history
        self.price_data = price_data
        self.lookback_quarters = lookback_quarters

    def build_features(
        self,
        symbol: str,
        earnings_date: date,
        current_iv_30d: float,
        current_iv_7d: float,
        current_straddle_price: float,
        underlying_price: float,
        vix_level: float,
        vix_percentile: float,
        sector_iv_rank: float,
        is_pre_market: bool = True
    ) -> EarningsFeatures:
        """Build feature set for upcoming earnings."""

        # Get historical earnings for this symbol
        symbol_earnings = self.earnings_history[
            (self.earnings_history['symbol'] == symbol) &
            (self.earnings_history['announcement_date'] < pd.Timestamp(earnings_date))
        ].sort_values('announcement_date', ascending=False).head(self.lookback_quarters)

        # Calculate historical metrics
        if len(symbol_earnings) > 0:
            historical_moves = symbol_earnings['price_move_pct'].abs()
            historical_avg_move = historical_moves.mean()
            historical_max_move = historical_moves.max()

            iv_crushes = (
                (symbol_earnings['iv_after'] - symbol_earnings['iv_before']) /
                symbol_earnings['iv_before']
            )
            avg_iv_crush = iv_crushes.mean()

            earnings_beat_rate = (symbol_earnings['eps_surprise_pct'] > 0).mean()
            avg_surprise = symbol_earnings['eps_surprise_pct'].abs().mean()
        else:
            # Defaults for new stocks
            historical_avg_move = 0.05
            historical_max_move = 0.10
            avg_iv_crush = -0.25
            earnings_beat_rate = 0.50
            avg_surprise = 0.05

        # Implied move from straddle
        implied_move = current_straddle_price / underlying_price

        # IV metrics
        iv_term_slope = (current_iv_30d - current_iv_7d) / current_iv_7d if current_iv_7d > 0 else 0

        # IV rank (simplified - would use full history in production)
        iv_rank = min(current_iv_30d / 0.40, 1.0)  # Normalize to typical range

        # Days to earnings
        days_to = (earnings_date - date.today()).days

        return EarningsFeatures(
            symbol=symbol,
            earnings_date=earnings_date,
            implied_move=implied_move,
            historical_avg_move=historical_avg_move,
            historical_max_move=historical_max_move,
            implied_vs_realized_ratio=implied_move / historical_avg_move if historical_avg_move > 0 else 1.0,
            iv_rank_52w=iv_rank,
            iv_30d=current_iv_30d,
            iv_7d=current_iv_7d,
            iv_term_slope=iv_term_slope,
            avg_iv_crush_pct=avg_iv_crush,
            earnings_beat_rate=earnings_beat_rate,
            avg_surprise_magnitude=avg_surprise,
            vix_level=vix_level,
            vix_percentile=vix_percentile,
            sector_iv_rank=sector_iv_rank,
            days_to_earnings=days_to,
            is_pre_market=is_pre_market
        )


class EarningsPredictor:
    """
    ML model for earnings-related predictions.

    Supports multiple backends:
    - XGBoost (preferred, GPU-accelerated)
    - LightGBM (fast CPU alternative)
    - Simple heuristic (no training required)
    """

    def __init__(
        self,
        model_type: str = "heuristic",  # "xgboost", "lightgbm", or "heuristic"
        model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.feature_importance: Dict[str, float] = {}

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def train(
        self,
        X: np.ndarray,
        y_iv_crush: np.ndarray,
        y_move_ratio: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the earnings prediction model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_iv_crush: Target - IV crush percentage
            y_move_ratio: Target - actual move / implied move
            validation_split: Fraction for validation

        Returns:
            Dict of training metrics
        """
        if self.model_type == "heuristic":
            return {"note": "Heuristic model requires no training"}

        n_val = int(len(X) * validation_split)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_crush_train, y_crush_val = y_iv_crush[:-n_val], y_iv_crush[-n_val:]

        if self.model_type == "xgboost":
            return self._train_xgboost(X_train, y_crush_train, X_val, y_crush_val)
        elif self.model_type == "lightgbm":
            return self._train_lightgbm(X_train, y_crush_train, X_val, y_crush_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Train XGBoost model with optional GPU."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")

        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }

        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=EarningsFeatures.feature_names())
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=EarningsFeatures.feature_names())

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=False
        )

        # Feature importance
        importance = self.model.get_score(importance_type='gain')
        self.feature_importance = importance

        # Validation metrics
        val_pred = self.model.predict(dval)
        rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        mae = np.mean(np.abs(val_pred - y_val))

        return {
            'val_rmse': rmse,
            'val_mae': mae,
            'best_iteration': self.model.best_iteration
        }

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """Train LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        if self.use_gpu:
            params['device'] = 'gpu'

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=EarningsFeatures.feature_names())
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [lgb.early_stopping(20), lgb.log_evaluation(0)]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # Feature importance
        importance = dict(zip(
            EarningsFeatures.feature_names(),
            self.model.feature_importance(importance_type='gain')
        ))
        self.feature_importance = importance

        # Validation metrics
        val_pred = self.model.predict(X_val)
        rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        mae = np.mean(np.abs(val_pred - y_val))

        return {
            'val_rmse': rmse,
            'val_mae': mae,
            'best_iteration': self.model.best_iteration
        }

    def predict(self, features: EarningsFeatures) -> EarningsPrediction:
        """Generate prediction for earnings event."""

        if self.model_type == "heuristic":
            return self._predict_heuristic(features)

        X = features.to_array().reshape(1, -1)

        if self.model_type == "xgboost":
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X, feature_names=EarningsFeatures.feature_names())
            predicted_crush = float(self.model.predict(dmatrix)[0])
        else:
            predicted_crush = float(self.model.predict(X)[0])

        # Estimate move ratio from features
        predicted_move_ratio = self._estimate_move_ratio(features)

        # Determine action and confidence
        action, reason, confidence = self._determine_action(
            features, predicted_crush, predicted_move_ratio
        )

        # Risk assessment
        risk_score = self._calculate_risk_score(features, predicted_crush)
        max_loss = self._estimate_max_loss(features)

        return EarningsPrediction(
            symbol=features.symbol,
            earnings_date=features.earnings_date,
            predicted_iv_crush_pct=predicted_crush,
            predicted_move_vs_implied=predicted_move_ratio,
            confidence=confidence,
            action=action,
            action_reason=reason,
            risk_score=risk_score,
            max_loss_scenario=max_loss
        )

    def _predict_heuristic(self, features: EarningsFeatures) -> EarningsPrediction:
        """Simple heuristic prediction without ML."""

        # IV crush typically 20-40% post earnings
        base_crush = -0.30

        # Adjust based on IV level (higher IV = bigger crush)
        if features.iv_rank_52w > 0.70:
            predicted_crush = base_crush * 1.2
        elif features.iv_rank_52w < 0.30:
            predicted_crush = base_crush * 0.8
        else:
            predicted_crush = base_crush

        # Move ratio: implied usually overestimates
        # Ratio < 1 means actual move was less than implied
        if features.implied_vs_realized_ratio > 1.3:
            # Implied is much higher than historical - likely overpriced
            predicted_move_ratio = 0.7
        elif features.implied_vs_realized_ratio < 0.8:
            # Implied is lower than historical - might be underpriced
            predicted_move_ratio = 1.2
        else:
            predicted_move_ratio = 0.9

        # Determine action
        action, reason, confidence = self._determine_action(
            features, predicted_crush, predicted_move_ratio
        )

        risk_score = self._calculate_risk_score(features, predicted_crush)
        max_loss = self._estimate_max_loss(features)

        return EarningsPrediction(
            symbol=features.symbol,
            earnings_date=features.earnings_date,
            predicted_iv_crush_pct=predicted_crush,
            predicted_move_vs_implied=predicted_move_ratio,
            confidence=confidence * 0.7,  # Lower confidence for heuristic
            action=action,
            action_reason=reason,
            risk_score=risk_score,
            max_loss_scenario=max_loss
        )

    def _estimate_move_ratio(self, features: EarningsFeatures) -> float:
        """Estimate actual_move / implied_move ratio."""
        # Based on research: implied moves overestimate ~65% of the time
        base_ratio = 0.85

        # Adjustments
        if features.implied_vs_realized_ratio > 1.5:
            # Implied much higher than historical
            return 0.6
        elif features.implied_vs_realized_ratio < 0.7:
            # Implied lower than historical - be cautious
            return 1.1

        return base_ratio

    def _determine_action(
        self,
        features: EarningsFeatures,
        predicted_crush: float,
        predicted_move_ratio: float
    ) -> Tuple[EarningsAction, str, float]:
        """Determine recommended trading action."""

        confidence = 0.5  # Base confidence

        # Rule 1: If implied >> historical, IV is overpriced
        if features.implied_vs_realized_ratio > 1.4 and features.iv_rank_52w > 0.60:
            action = EarningsAction.SELL_PRE_EARNINGS
            reason = f"IV overpriced: implied {features.implied_move:.1%} vs historical {features.historical_avg_move:.1%}"
            confidence = 0.75

        # Rule 2: If implied << historical, avoid (underpriced risk)
        elif features.implied_vs_realized_ratio < 0.7:
            action = EarningsAction.AVOID
            reason = f"IV underpriced: implied {features.implied_move:.1%} vs historical {features.historical_avg_move:.1%}"
            confidence = 0.80

        # Rule 3: Historical max move > 15% = dangerous
        elif features.historical_max_move > 0.15:
            action = EarningsAction.AVOID
            reason = f"High historical volatility: max move {features.historical_max_move:.1%}"
            confidence = 0.70

        # Rule 4: VIX elevated = market uncertainty
        elif features.vix_percentile > 0.80:
            action = EarningsAction.AVOID
            reason = f"VIX elevated at {features.vix_percentile:.0%} percentile"
            confidence = 0.65

        # Rule 5: Good crush expected + reasonable IV = sell post
        elif predicted_crush < -0.25 and features.iv_rank_52w > 0.40:
            action = EarningsAction.SELL_POST_EARNINGS
            reason = f"Expected IV crush {predicted_crush:.1%}, wait for post-earnings entry"
            confidence = 0.60

        # Default: neutral
        else:
            action = EarningsAction.NEUTRAL
            reason = "No strong signal"
            confidence = 0.40

        return action, reason, confidence

    def _calculate_risk_score(
        self,
        features: EarningsFeatures,
        predicted_crush: float
    ) -> float:
        """Calculate risk score 0-1."""
        risk = 0.0

        # Factor 1: Historical volatility
        if features.historical_max_move > 0.15:
            risk += 0.30
        elif features.historical_max_move > 0.10:
            risk += 0.15

        # Factor 2: IV underpriced
        if features.implied_vs_realized_ratio < 0.8:
            risk += 0.25

        # Factor 3: VIX level
        if features.vix_percentile > 0.70:
            risk += 0.20

        # Factor 4: Low beat rate (unpredictable)
        if 0.35 < features.earnings_beat_rate < 0.65:
            risk += 0.15  # Close to 50/50 is actually higher risk

        # Factor 5: Days to earnings (closer = more gamma risk)
        if features.days_to_earnings <= 2:
            risk += 0.10

        return min(risk, 1.0)

    def _estimate_max_loss(self, features: EarningsFeatures) -> float:
        """Estimate worst-case loss as fraction of position."""
        # Max loss = stock goes to 0 minus premium collected
        # But more realistically, use historical max move
        gap_scenario = features.historical_max_move * 1.5  # 50% buffer
        return min(gap_scenario, 0.50)  # Cap at 50%

    def save_model(self, path: str) -> None:
        """Save trained model."""
        model_data = {
            'model_type': self.model_type,
            'model': self.model,
            'feature_importance': self.feature_importance
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, path: str) -> None:
        """Load trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model_type = model_data['model_type']
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']


def create_earnings_training_data(
    earnings_df: pd.DataFrame,
    options_df: pd.DataFrame,
    prices_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training data from historical earnings.

    Args:
        earnings_df: Historical earnings with columns:
            symbol, announcement_date, eps_estimate, eps_actual, eps_surprise_pct
        options_df: Historical options with columns:
            symbol, date, strike, expiration, iv_mid, underlying_price
        prices_df: OHLCV data

    Returns:
        X: Feature matrix
        y_iv_crush: IV crush targets
        y_move_ratio: Move ratio targets
    """
    features_list = []
    iv_crush_targets = []
    move_ratio_targets = []

    # This is a placeholder - actual implementation requires
    # Bloomberg data to be fully functional

    warnings.warn(
        "create_earnings_training_data requires historical options data. "
        "Implementation will be complete once Bloomberg data is integrated."
    )

    return np.array([]), np.array([]), np.array([])


def get_earnings_recommendation(
    symbol: str,
    earnings_date: date,
    current_iv: float,
    implied_move: float,
    historical_moves: List[float],
    vix_level: float = 20.0
) -> EarningsPrediction:
    """
    Quick utility function for earnings recommendation.

    Uses heuristic model - no training required.
    """
    # Build simplified features
    features = EarningsFeatures(
        symbol=symbol,
        earnings_date=earnings_date,
        implied_move=implied_move,
        historical_avg_move=np.mean(np.abs(historical_moves)) if historical_moves else 0.05,
        historical_max_move=np.max(np.abs(historical_moves)) if historical_moves else 0.10,
        implied_vs_realized_ratio=implied_move / (np.mean(np.abs(historical_moves)) if historical_moves else 0.05),
        iv_rank_52w=min(current_iv / 0.40, 1.0),
        iv_30d=current_iv,
        iv_7d=current_iv * 1.1,  # Front month typically higher near earnings
        iv_term_slope=-0.1,
        avg_iv_crush_pct=-0.30,
        earnings_beat_rate=0.55,
        avg_surprise_magnitude=0.05,
        vix_level=vix_level,
        vix_percentile=min(vix_level / 35.0, 1.0),
        sector_iv_rank=0.50,
        days_to_earnings=(earnings_date - date.today()).days,
        is_pre_market=True
    )

    predictor = EarningsPredictor(model_type="heuristic")
    return predictor.predict(features)
