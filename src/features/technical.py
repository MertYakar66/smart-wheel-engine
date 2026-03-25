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
        Relative Strength Index (Wilder's smoothing).

        Uses Wilder's smoothing (not standard EMA) as per the original
        RSI specification. Wilder's smoothing has alpha = 1/n, which is
        equivalent to EMA with span = 2n - 1.

        Args:
            close: Close prices
            window: RSI lookback period (typically 14)

        Returns:
            RSI values (0-100)
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Wilder's smoothing: alpha = 1/n, equivalent to EMA span = 2*n - 1
        wilder_span = 2 * window - 1
        avg_gain = gain.ewm(span=wilder_span, adjust=False).mean()
        avg_loss = loss.ewm(span=wilder_span, adjust=False).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI when no data

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
    def momentum(close: pd.Series, skip_recent: int = 21, lookback: int = 252) -> pd.Series:
        """
        Price momentum (12-1 month return, skipping recent month).

        Classic momentum factor: return from t-252 to t-21, excluding last 21 days.
        This avoids short-term reversal while capturing intermediate momentum.

        Args:
            close: Close price series
            skip_recent: Days to skip (avoid short-term reversal), default 21
            lookback: Total lookback period, default 252 (1 year)

        Returns:
            Momentum return (e.g., 0.15 = 15% return over period)
        """
        return close.shift(skip_recent) / close.shift(lookback) - 1

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

    @staticmethod
    def hurst_exponent(price: pd.Series, window: int = 100) -> pd.Series:
        """
        Hurst exponent (simplified R/S analysis).

        Measures mean-reversion vs trending tendency:
        - H < 0.5: Mean-reverting (good for wheel strategy)
        - H = 0.5: Random walk
        - H > 0.5: Trending (caution for wheel strategy)

        Uses rolling R/S (Rescaled Range) analysis.

        Args:
            price: Price series
            window: Lookback window for R/S calculation

        Returns:
            Rolling Hurst exponent estimate (0-1)
        """
        log_returns = np.log(price / price.shift(1))

        def calc_hurst(returns):
            if len(returns) < 20 or returns.isna().all():
                return np.nan

            returns = returns.dropna()
            n = len(returns)
            if n < 20:
                return np.nan

            # R/S for different sub-periods
            rs_values = []
            for divisor in [2, 4, 8, 16]:
                if n // divisor < 10:
                    continue
                sub_n = n // divisor
                rs_sub = []
                for i in range(divisor):
                    sub_returns = returns.iloc[i * sub_n:(i + 1) * sub_n]
                    if len(sub_returns) < 2:
                        continue
                    # Cumulative deviation from mean
                    mean_r = sub_returns.mean()
                    cumdev = (sub_returns - mean_r).cumsum()
                    R = cumdev.max() - cumdev.min()
                    S = sub_returns.std()
                    if S > 0:
                        rs_sub.append(R / S)
                if rs_sub:
                    rs_values.append((sub_n, np.mean(rs_sub)))

            if len(rs_values) < 2:
                return np.nan

            # Linear regression of log(R/S) vs log(n)
            log_n = np.log([x[0] for x in rs_values])
            log_rs = np.log([x[1] for x in rs_values])

            # Simple linear regression slope
            n_pts = len(log_n)
            slope = (n_pts * np.sum(log_n * log_rs) - np.sum(log_n) * np.sum(log_rs)) / \
                    (n_pts * np.sum(log_n ** 2) - np.sum(log_n) ** 2)

            return np.clip(slope, 0, 1)

        return log_returns.rolling(window).apply(calc_hurst, raw=False)

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

        # Hurst exponent (mean reversion indicator)
        # Only compute if sufficient history (slow, so check length first)
        if len(result) >= 100:
            result["hurst_100"] = self.hurst_exponent(result["close"], window=100)

        return result
