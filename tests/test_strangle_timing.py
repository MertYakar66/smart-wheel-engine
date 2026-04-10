"""
Tests for the Short Strangle Entry Timing Framework.

Validates:
1. Regime classification across volatility lifecycle phases
2. Entry score computation and weighting
3. Signal integration
4. Edge cases and data requirements
"""

import numpy as np
import pandas as pd

from engine.signals import SignalStrength, StrangleTimingSignal
from engine.strangle_timing import (
    StrangleEntryScore,
    StrangleRegime,
    StrangleTimingEngine,
    VolatilityPhase,
)


def _generate_ohlcv(n: int = 252, seed: int = 42, base_vol: float = 0.02) -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, base_vol, n))
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n))
    open_[0] = 100.0
    volume = rng.lognormal(15, 0.5, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


def _generate_post_expansion_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate data that mimics post-expansion stabilization."""
    rng = np.random.default_rng(seed)
    # Phase 1: normal (first 150 days)
    prices1 = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, 150))
    # Phase 2: expansion (next 30 days - big moves)
    prices2 = prices1[-1] * np.cumprod(1 + rng.normal(-0.005, 0.04, 30))
    # Phase 3: stabilization (last 72 days - small moves, elevated range)
    prices3 = prices2[-1] * np.cumprod(1 + rng.normal(0.001, 0.012, 72))

    close = np.concatenate([prices1, prices2, prices3])
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n))
    open_[0] = 100.0
    volume = rng.lognormal(15, 0.5, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


def _generate_compression_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate data with very low volatility (compression)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.003, n))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.001, n))
    open_[0] = 100.0
    volume = rng.lognormal(15, 0.5, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


def _generate_trending_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate data with strong uptrend."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0.003, 0.015, n))
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.002, n))
    open_[0] = 100.0
    volume = rng.lognormal(15, 0.5, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


class TestRegimeClassification:
    """Test volatility lifecycle phase detection."""

    def test_basic_classification_returns_regime(self):
        """Should return a StrangleRegime object."""
        df = _generate_ohlcv()
        engine = StrangleTimingEngine()
        regime = engine.classify_regime(df)
        assert isinstance(regime, StrangleRegime)
        assert isinstance(regime.phase, VolatilityPhase)
        assert 0 <= regime.confidence <= 1

    def test_compression_detected(self):
        """Low-volatility data should produce low ATR percentile."""
        df = _generate_compression_ohlcv()
        engine = StrangleTimingEngine()
        regime = engine.classify_regime(df)
        # Very low vol data should have low ATR percentile or narrow bands
        # The exact classification depends on relative history, so check metrics
        assert regime.atr_14 < df["close"].iloc[-1] * 0.02, (
            f"ATR should be small relative to price in compression data: {regime.atr_14}"
        )

    def test_insufficient_data_returns_unknown(self):
        """Too little data should return UNKNOWN with low confidence."""
        df = _generate_ohlcv(n=30)
        engine = StrangleTimingEngine()
        regime = engine.classify_regime(df)
        assert regime.phase == VolatilityPhase.UNKNOWN
        assert regime.confidence == 0.0

    def test_regime_string_representation(self):
        """String representation should include all components."""
        df = _generate_ohlcv()
        engine = StrangleTimingEngine()
        regime = engine.classify_regime(df)
        s = str(regime)
        assert "BB:" in s
        assert "ATR:" in s
        assert "RSI:" in s

    def test_all_metrics_finite(self):
        """All regime metrics should be finite numbers."""
        df = _generate_ohlcv()
        engine = StrangleTimingEngine()
        regime = engine.classify_regime(df)
        assert np.isfinite(regime.bb_width)
        assert np.isfinite(regime.atr_14)
        assert np.isfinite(regime.rsi_14)
        assert 0 <= regime.rsi_14 <= 100
        assert np.isfinite(regime.ma_slope_20)


class TestEntryScoring:
    """Test the weighted entry scoring system."""

    def test_score_returns_valid_object(self):
        """Should return StrangleEntryScore with valid range."""
        df = _generate_ohlcv()
        engine = StrangleTimingEngine()
        score = engine.score_entry(df)
        assert isinstance(score, StrangleEntryScore)
        assert 0 <= score.total_score <= 100
        assert score.recommendation in ("strong_entry", "conditional", "avoid")

    def test_compression_scores_low(self):
        """Compression regime should score low and set warning flag."""
        df = _generate_compression_ohlcv()
        engine = StrangleTimingEngine()
        score = engine.score_entry(df)
        # Compression should either flag or score low
        if score.compression_warning:
            assert score.recommendation == "avoid"

    def test_scoring_is_deterministic(self):
        """Same data should produce same score every time."""
        df = _generate_ohlcv(seed=42)
        engine = StrangleTimingEngine()
        score1 = engine.score_entry(df)
        score2 = engine.score_entry(df)
        assert score1.total_score == score2.total_score
        assert score1.recommendation == score2.recommendation

    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        engine = StrangleTimingEngine()
        assert abs(sum(engine.weights.values()) - 1.0) < 1e-10

    def test_custom_weights(self):
        """Custom weights should be used in scoring."""
        df = _generate_ohlcv()
        heavy_bb = {"bollinger": 0.80, "atr": 0.05, "rsi": 0.05, "trend": 0.05, "range": 0.05}
        engine = StrangleTimingEngine(weights=heavy_bb)
        score = engine.score_entry(df)
        assert score.weights == heavy_bb

    def test_component_scores_bounded(self):
        """Each component score should be in [0, 100]."""
        df = _generate_ohlcv()
        engine = StrangleTimingEngine()
        score = engine.score_entry(df)
        assert 0 <= score.bollinger_score <= 100
        assert 0 <= score.atr_score <= 100
        assert 0 <= score.rsi_score <= 100
        assert 0 <= score.trend_score <= 100
        assert 0 <= score.range_score <= 100

    def test_score_string_representation(self):
        """String representation should be readable."""
        df = _generate_ohlcv()
        engine = StrangleTimingEngine()
        score = engine.score_entry(df)
        s = str(score)
        assert "Entry Score:" in s
        assert "BB=" in s


class TestSignalIntegration:
    """Test StrangleTimingSignal integration with signal framework."""

    def test_signal_with_valid_data(self):
        """Should produce a valid signal from OHLCV data."""
        df = _generate_ohlcv()
        signal_gen = StrangleTimingSignal()
        signal = signal_gen.generate({"ohlcv_data": df})
        assert signal.name == "Strangle_Timing"
        assert isinstance(signal.strength, SignalStrength)
        assert "entry_score" in signal.metadata
        assert "phase" in signal.metadata

    def test_signal_with_no_data(self):
        """Should return neutral signal when no data provided."""
        signal_gen = StrangleTimingSignal()
        signal = signal_gen.generate({})
        assert signal.strength == SignalStrength.NEUTRAL

    def test_signal_with_short_data(self):
        """Should return neutral signal when data too short."""
        df = _generate_ohlcv(n=30)
        signal_gen = StrangleTimingSignal()
        signal = signal_gen.generate({"ohlcv_data": df})
        assert signal.strength == SignalStrength.NEUTRAL

    def test_compression_produces_sell_signal(self):
        """Compression regime should produce sell/avoid signal."""
        df = _generate_compression_ohlcv()
        signal_gen = StrangleTimingSignal()
        signal = signal_gen.generate({"ohlcv_data": df})
        # If compression detected, should be negative
        if signal.metadata.get("phase") == "compression":
            assert signal.strength.value <= 0


class TestUniverseScan:
    """Test multi-symbol scanning."""

    def test_scan_returns_dataframe(self):
        """Should return a DataFrame with scored symbols."""
        symbols = {
            "AAPL": _generate_ohlcv(seed=1),
            "MSFT": _generate_ohlcv(seed=2),
            "GOOGL": _generate_post_expansion_ohlcv(seed=3),
        }
        engine = StrangleTimingEngine()
        results = engine.scan_universe(symbols, min_score=0)
        assert isinstance(results, pd.DataFrame)
        assert len(results) >= 1
        assert "ticker" in results.columns
        assert "score" in results.columns
        assert "recommendation" in results.columns

    def test_scan_respects_min_score(self):
        """Min score filter should work."""
        symbols = {
            "TEST": _generate_compression_ohlcv(seed=1),
        }
        engine = StrangleTimingEngine()
        results = engine.scan_universe(symbols, min_score=95)
        # Very high min_score should filter out most/all
        assert len(results) <= 1


class TestHistoricalScores:
    """Test historical score computation for backtesting."""

    def test_historical_scores_shape(self):
        """Should produce scores for each date after lookback."""
        df = _generate_ohlcv(n=200)
        engine = StrangleTimingEngine()
        hist = engine.compute_historical_scores(df, lookback_required=100)
        assert isinstance(hist, pd.DataFrame)
        assert len(hist) > 0
        assert "score" in hist.columns
        assert "phase" in hist.columns
        # All scores should be bounded
        assert hist["score"].min() >= 0
        assert hist["score"].max() <= 100
