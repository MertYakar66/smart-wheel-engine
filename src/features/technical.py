"""Technical analysis features."""

import numpy as np
import pandas as pd


class TechnicalFeatures:
    """Compute technical analysis features."""

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window).mean()

    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def ma_ratio(price: pd.Series, window: int) -> pd.Series:
        """Price relative to moving average."""
        return price / price.rolling(window).mean()

    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            close: Close prices
            window: RSI lookback period

        Returns:
            RSI values (0-100)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=window, adjust=False).mean()
        avg_loss = loss.ewm(span=window, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD indicator.

        Args:
            close: Close prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            close: Close prices
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    @staticmethod
    def bollinger_position(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """
        Position within Bollinger Bands (0 = lower, 1 = upper).

        Args:
            close: Close prices
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            Position within bands (0-1)
        """
        upper, middle, lower = TechnicalFeatures.bollinger_bands(close, window, num_std)
        return (close - lower) / (upper - lower)

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ATR lookback period

        Returns:
            ATR values
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window).mean()

    @staticmethod
    def atr_percent(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14,
    ) -> pd.Series:
        """ATR as percentage of close price."""
        return TechnicalFeatures.atr(high, low, close, window) / close * 100

    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
        """Current volume relative to average."""
        return volume / volume.rolling(window).mean()

    @staticmethod
    def returns(close: pd.Series, periods: int = 1) -> pd.Series:
        """Simple returns over N periods."""
        return close.pct_change(periods)

    @staticmethod
    def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
        """Log returns over N periods."""
        return np.log(close / close.shift(periods))

    @staticmethod
    def momentum(close: pd.Series, window: int = 12) -> pd.Series:
        """Price momentum (12-1 month return, skipping last month)."""
        return close.shift(21) / close.shift(252) - 1

    @staticmethod
    def support_resistance_distance(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        lookback: int = 20,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Distance to recent support and resistance levels.

        Args:
            close: Close prices
            high: High prices
            low: Low prices
            lookback: Lookback period

        Returns:
            Tuple of (distance_to_support, distance_to_resistance)
        """
        rolling_high = high.rolling(lookback).max()
        rolling_low = low.rolling(lookback).min()

        dist_to_resistance = (rolling_high - close) / close
        dist_to_support = (close - rolling_low) / close

        return dist_to_support, dist_to_resistance

    @staticmethod
    def price_channel_position(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        lookback: int = 20,
    ) -> pd.Series:
        """Position within price channel (0 = low, 1 = high)."""
        rolling_high = high.rolling(lookback).max()
        rolling_low = low.rolling(lookback).min()
        return (close - rolling_low) / (rolling_high - rolling_low)

    @staticmethod
    def gap(open_: pd.Series, close: pd.Series) -> pd.Series:
        """Overnight gap (open vs previous close)."""
        return (open_ - close.shift(1)) / close.shift(1)

    @staticmethod
    def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Intraday range as percentage of close."""
        return (high - low) / close

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features for a dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical features added
        """
        result = df.copy()

        # Returns
        for period in [1, 5, 21, 63]:
            result[f"return_{period}d"] = self.returns(result["close"], period)

        # Moving average ratios
        for window in [20, 50, 200]:
            result[f"ma_ratio_{window}"] = self.ma_ratio(result["close"], window)

        # RSI
        result["rsi_14"] = self.rsi(result["close"], 14)

        # MACD
        macd_line, signal_line, histogram = self.macd(result["close"])
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_hist"] = histogram

        # Bollinger
        result["bb_position"] = self.bollinger_position(result["close"])

        # ATR
        result["atr_14"] = self.atr(result["high"], result["low"], result["close"], 14)
        result["atr_pct_14"] = self.atr_percent(result["high"], result["low"], result["close"], 14)

        # Volume
        result["volume_ratio_20"] = self.volume_ratio(result["volume"], 20)

        # Momentum
        result["momentum_12m"] = self.momentum(result["close"])

        # Support/Resistance
        dist_support, dist_resistance = self.support_resistance_distance(
            result["close"], result["high"], result["low"]
        )
        result["dist_to_support"] = dist_support
        result["dist_to_resistance"] = dist_resistance

        # Price channel position
        result["channel_position_20"] = self.price_channel_position(
            result["close"], result["high"], result["low"], 20
        )

        # Gap
        result["gap"] = self.gap(result["open"], result["close"])

        # Intraday range
        result["intraday_range"] = self.intraday_range(
            result["high"], result["low"], result["close"]
        )

        return result
