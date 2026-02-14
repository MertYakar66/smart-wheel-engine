"""
Machine Learning Module for Smart Wheel Engine

ML-powered components for options trading:
- Earnings prediction (IV crush, move vs implied)
- Trade outcome prediction
- Regime classification
"""

from .earnings_model import (
    EarningsPredictor,
    EarningsFeatures,
    EarningsPrediction,
    EarningsAction,
    EarningsFeatureBuilder,
    get_earnings_recommendation,
    create_earnings_training_data
)

__all__ = [
    'EarningsPredictor',
    'EarningsFeatures',
    'EarningsPrediction',
    'EarningsAction',
    'EarningsFeatureBuilder',
    'get_earnings_recommendation',
    'create_earnings_training_data'
]
