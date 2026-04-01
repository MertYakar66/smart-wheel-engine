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
        Relative Strength Index with canonical Wilder bootstrap.

        Implements RSI exactly as specified by Wilder (1978):
        1. First `window` periods are NaN (warm-up)
        2. First valid RSI uses simple average of first `window` gains/losses
        3. Subsequent values use Wilder's recursive smoothing:
           avg = (prev_avg * (n-1) + current) / n

        Args:
            close: Close prices
            window: RSI lookback period (typically 14)

        Returns:
            RSI values (0-100), with first `window` values as NaN
        """
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Initialize output
        rsi = pd.Series(np.nan, index=close.index)

        # Need at least window+1 data points for first valid RSI
        if len(close) <= window:
            return rsi

        # Canonical Wilder bootstrap: first avg uses simple mean
        first_avg_gain = gain.iloc[1 : window + 1].mean()
        first_avg_loss = loss.iloc[1 : window + 1].mean()

        # Build smoothed averages using Wilder's recursive formula
        avg_gains = np.zeros(len(close))
        avg_losses = np.zeros(len(close))
        avg_gains[: window + 1] = np.nan
        avg_losses[: window + 1] = np.nan
        avg_gains[window] = first_avg_gain
        avg_losses[window] = first_avg_loss

        # Wilder smoothing: avg = (prev_avg * (n-1) + current) / n
        for i in range(window + 1, len(close)):
            avg_gains[i] = (avg_gains[i - 1] * (window - 1) + gain.iloc[i]) / window
            avg_losses[i] = (avg_losses[i - 1] * (window - 1) + loss.iloc[i]) / window

        avg_gain = pd.Series(avg_gains, index=close.index)
        avg_loss = pd.Series(avg_losses, index=close.index)

        # Handle edge cases correctly:
        # - avg_loss == 0 (all gains): RSI should be 100
        # - avg_gain == 0 (all losses): RSI should be 0
        # - both == 0 (no movement): RSI is undefined, use 50

        # Normal case: both avg_gain and avg_loss > 0
        both_positive = (avg_gain > 0) & (avg_loss > 0)
        rs = avg_gain / avg_loss
        rsi[both_positive] = 100 - (100 / (1 + rs[both_positive]))

        # All gains, no losses: RSI = 100
        all_gains = (avg_gain > 0) & (avg_loss == 0)
        rsi[all_gains] = 100.0

        # All losses, no gains: RSI = 0
        all_losses = (avg_gain == 0) & (avg_loss > 0)
        rsi[all_losses] = 0.0

        # No movement (both zero): neutral RSI
        no_movement = (avg_gain == 0) & (avg_loss == 0)
        rsi[no_movement] = 50.0

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
        Average True Range with canonical Wilder smoothing.

        Implements ATR exactly as specified by Wilder (1978):
        1. First ATR uses simple average of first `window` true ranges
        2. Subsequent values use Wilder's recursive smoothing:
           ATR = (prev_ATR * (n-1) + current_TR) / n

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            window: ATR lookback period

        Returns:
            ATR values, with first `window-1` values as NaN
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Canonical Wilder smoothing (not SMA)
        atr = pd.Series(np.nan, index=close.index)

        if len(close) < window:
            return atr

        # First ATR: simple average of first `window` true ranges
        first_atr = true_range.iloc[:window].mean()
        atr.iloc[window - 1] = first_atr

        # Wilder smoothing: ATR = (prev_ATR * (n-1) + TR) / n
        for i in range(window, len(close)):
            atr.iloc[i] = (atr.iloc[i - 1] * (window - 1) + true_range.iloc[i]) / window

        return atr

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
                    sub_returns = returns.iloc[i * sub_n : (i + 1) * sub_n]
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
            slope = (n_pts * np.sum(log_n * log_rs) - np.sum(log_n) * np.sum(log_rs)) / (
                n_pts * np.sum(log_n**2) - np.sum(log_n) ** 2
            )

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
