"""Volatility feature engineering."""

import numpy as np
import pandas as pd


class VolatilityFeatures:
    """Compute realized and implied volatility features."""

    @staticmethod
    def realized_volatility_close(
        returns: pd.Series,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Close-to-close realized volatility.

        Args:
            returns: Series of log returns
            window: Rolling window size
            annualize: Whether to annualize (multiply by sqrt(252))

        Returns:
            Rolling realized volatility
        """
        rv = returns.rolling(window).std()
        if annualize:
            rv = rv * np.sqrt(252)
        return rv

    @staticmethod
    def realized_volatility_parkinson(
        high: pd.Series,
        low: pd.Series,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Parkinson volatility estimator using high-low range.

        More efficient than close-to-close as it uses intraday information.
        Assumes no overnight jumps.

        Args:
            high: High prices
            low: Low prices
            window: Rolling window size
            annualize: Whether to annualize

        Returns:
            Parkinson volatility estimate
        """
        log_hl = np.log(high / low)
        factor = 1.0 / (4.0 * np.log(2))
        rv = np.sqrt(factor * (log_hl ** 2).rolling(window).mean())
        if annualize:
            rv = rv * np.sqrt(252)
        return rv

    @staticmethod
    def realized_volatility_garman_klass(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Garman-Klass volatility estimator using OHLC data.

        Most efficient estimator using all OHLC information.
        Assumes no overnight jumps and drift.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            annualize: Whether to annualize

        Returns:
            Garman-Klass volatility estimate
        """
        log_hl = np.log(high / low) ** 2
        log_co = np.log(close / open_) ** 2

        term1 = 0.5 * log_hl
        term2 = (2 * np.log(2) - 1) * log_co

        rv = np.sqrt((term1 - term2).rolling(window).mean())
        if annualize:
            rv = rv * np.sqrt(252)
        return rv

    @staticmethod
    def realized_volatility_yang_zhang(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21,
        annualize: bool = True,
    ) -> pd.Series:
        """
        Yang-Zhang volatility estimator (2000).

        Handles overnight jumps and opening gaps.
        Minimum variance unbiased estimator combining:
        - Overnight variance (close-to-open)
        - Open-to-close variance
        - Rogers-Satchell intraday variance

        Reference: Yang & Zhang (2000) "Drift Independent Volatility Estimation"

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size
            annualize: Whether to annualize

        Returns:
            Yang-Zhang volatility estimate
        """
        # Yang-Zhang optimal k: minimizes variance of estimator
        # k = 0.34 / (1.34 + (n+1)/(n-1)) where n = window
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        # Overnight volatility (close-to-open)
        log_co = np.log(open_ / close.shift(1))
        overnight_var = log_co.rolling(window).var()

        # Open-to-close volatility
        log_oc = np.log(close / open_)
        open_close_var = log_oc.rolling(window).var()

        # Rogers-Satchell volatility (intraday)
        log_ho = np.log(high / open_)
        log_lo = np.log(low / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)

        rs_var = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()

        # Combine
        rv = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)
        if annualize:
            rv = rv * np.sqrt(252)
        return rv

    @staticmethod
    def iv_rank(iv: pd.Series, lookback: int = 252) -> pd.Series:
        """
        IV Rank: Current IV percentile vs lookback period.

        IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100

        This is NOT the same as IV Percentile, which counts what % of
        observations are below the current value.

        Args:
            iv: Implied volatility series
            lookback: Lookback period in days

        Returns:
            IV rank (0-100), where 0 = at historical low, 100 = at historical high
        """
        rolling_min = iv.rolling(lookback).min()
        rolling_max = iv.rolling(lookback).max()
        # Avoid division by zero when min == max
        range_iv = rolling_max - rolling_min
        range_iv = range_iv.replace(0, np.nan)
        return ((iv - rolling_min) / range_iv * 100).clip(0, 100)

    @staticmethod
    def iv_percentile(iv: pd.Series, lookback: int = 252) -> pd.Series:
        """
        IV Percentile: What percentage of observations over lookback
        period are BELOW the current IV value.

        Different from IV Rank:
        - IV Rank = position relative to min/max range
        - IV Percentile = % of observations below current value

        Args:
            iv: Implied volatility series
            lookback: Lookback period in days

        Returns:
            IV percentile (0-100)
        """
        def percentile_rank(x):
            if len(x) < 2:
                return np.nan
            current = x.iloc[-1]
            # Count values strictly less than current, divide by total count
            below = (x.iloc[:-1] < current).sum()
            return below / (len(x) - 1) * 100

        return iv.rolling(lookback).apply(percentile_rank, raw=False)

    @staticmethod
    def iv_rv_spread(iv: pd.Series, rv: pd.Series) -> pd.Series:
        """
        IV-RV spread (volatility risk premium).

        Positive spread indicates options are "expensive" relative to realized vol.

        Args:
            iv: Implied volatility
            rv: Realized volatility

        Returns:
            IV - RV spread
        """
        return iv - rv

    @staticmethod
    def iv_term_structure_slope(
        iv_30d: pd.Series,
        iv_60d: pd.Series,
        iv_90d: pd.Series,
    ) -> pd.Series:
        """
        IV term structure slope.

        Positive = contango (upward sloping)
        Negative = backwardation (inverted)

        Args:
            iv_30d: 30-day IV
            iv_60d: 60-day IV
            iv_90d: 90-day IV

        Returns:
            Average slope of term structure
        """
        # Simple average of slopes
        slope_30_60 = (iv_60d - iv_30d) / 30
        slope_60_90 = (iv_90d - iv_60d) / 30
        return (slope_30_60 + slope_60_90) / 2

    def compute_all(
        self,
        df: pd.DataFrame,
        iv_col: str = "atm_iv",
        include_iv: bool = True,
    ) -> pd.DataFrame:
        """
        Compute all volatility features for a dataframe.

        Args:
            df: DataFrame with OHLC and optionally IV data
            iv_col: Column name for implied volatility
            include_iv: Whether to include IV-based features

        Returns:
            DataFrame with volatility features added
        """
        result = df.copy()

        # Log returns
        result["log_return"] = np.log(result["close"] / result["close"].shift(1))

        # Realized volatility (multiple horizons)
        for window in [5, 10, 21, 63]:
            result[f"rv_{window}d"] = self.realized_volatility_close(
                result["log_return"], window=window
            )

        # Alternative estimators (21-day)
        result["rv_parkinson_21d"] = self.realized_volatility_parkinson(
            result["high"], result["low"], window=21
        )
        result["rv_garman_klass_21d"] = self.realized_volatility_garman_klass(
            result["open"], result["high"], result["low"], result["close"], window=21
        )
        result["rv_yang_zhang_21d"] = self.realized_volatility_yang_zhang(
            result["open"], result["high"], result["low"], result["close"], window=21
        )

        # IV-based features
        if include_iv and iv_col in result.columns:
            result["iv_rank"] = self.iv_rank(result[iv_col])
            result["iv_percentile"] = self.iv_percentile(result[iv_col])
            result["iv_rv_spread"] = self.iv_rv_spread(result[iv_col], result["rv_21d"])

        return result
