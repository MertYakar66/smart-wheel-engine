"""Data validation utilities."""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    stats: dict[str, Any]


class DataValidator:
    """Validates market data for quality and consistency."""

    def __init__(self, strict: bool = True):
        self.strict = strict

    def validate_ohlcv(self, df: pd.DataFrame) -> ValidationResult:
        """Validate OHLCV data."""
        errors = []
        warnings = []
        stats = {}

        required_cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)

        # OHLC relationship checks
        high_low_violations = (df["high"] < df["low"]).sum()
        if high_low_violations > 0:
            errors.append(f"High < Low violations: {high_low_violations} rows")

        high_open_violations = (df["high"] < df["open"]).sum()
        if high_open_violations > 0:
            errors.append(f"High < Open violations: {high_open_violations} rows")

        high_close_violations = (df["high"] < df["close"]).sum()
        if high_close_violations > 0:
            errors.append(f"High < Close violations: {high_close_violations} rows")

        low_open_violations = (df["low"] > df["open"]).sum()
        if low_open_violations > 0:
            errors.append(f"Low > Open violations: {low_open_violations} rows")

        low_close_violations = (df["low"] > df["close"]).sum()
        if low_close_violations > 0:
            errors.append(f"Low > Close violations: {low_close_violations} rows")

        # Negative values
        for col in ["open", "high", "low", "close"]:
            neg_count = (df[col] <= 0).sum()
            if neg_count > 0:
                errors.append(f"Non-positive {col} values: {neg_count} rows")

        neg_volume = (df["volume"] < 0).sum()
        if neg_volume > 0:
            errors.append(f"Negative volume: {neg_volume} rows")

        # Missing data
        for col in required_cols:
            missing_pct = df[col].isna().mean() * 100
            stats[f"{col}_missing_pct"] = missing_pct
            if missing_pct > 1:
                warnings.append(f"{col} has {missing_pct:.2f}% missing values")

        # Extreme returns (potential data errors)
        if "close" in df.columns:
            returns = df.groupby("ticker")["close"].pct_change()
            extreme_returns = (returns.abs() > 0.5).sum()
            if extreme_returns > 0:
                warnings.append(f"Extreme daily returns (>50%): {extreme_returns} occurrences")
            stats["max_return"] = returns.max()
            stats["min_return"] = returns.min()

        # Summary stats
        stats["row_count"] = len(df)
        stats["ticker_count"] = df["ticker"].nunique()
        stats["date_range"] = (df["date"].min(), df["date"].max())

        is_valid = len(errors) == 0 if self.strict else True
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_options_flow(self, df: pd.DataFrame) -> ValidationResult:
        """Validate options flow data."""
        errors = []
        warnings = []
        stats = {}

        required_cols = ["date", "ticker", "call_volume", "put_volume", "call_oi", "put_oi"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)

        # Non-negative checks
        for col in ["call_volume", "put_volume", "call_oi", "put_oi"]:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                errors.append(f"Negative {col}: {neg_count} rows")

        # IV bounds check
        if "atm_iv" in df.columns:
            invalid_iv = ((df["atm_iv"] < 0) | (df["atm_iv"] > 5)).sum()
            if invalid_iv > 0:
                warnings.append(f"IV outside [0, 5] range: {invalid_iv} rows")

        # IV rank/percentile bounds
        for col in ["iv_rank", "iv_percentile"]:
            if col in df.columns:
                invalid = ((df[col] < 0) | (df[col] > 100)).sum()
                if invalid > 0:
                    errors.append(f"{col} outside [0, 100]: {invalid} rows")

        stats["row_count"] = len(df)
        stats["ticker_count"] = df["ticker"].nunique()

        is_valid = len(errors) == 0 if self.strict else True
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_fundamentals(self, df: pd.DataFrame) -> ValidationResult:
        """Validate fundamental data."""
        errors = []
        warnings = []
        stats = {}

        required_cols = ["date", "ticker"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)

        # Non-negative checks
        non_neg_cols = ["market_cap", "debt_equity", "current_ratio"]
        for col in non_neg_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    warnings.append(f"Negative {col}: {neg_count} rows")

        # Margin bounds [-1, 1]
        margin_cols = ["gross_margin", "operating_margin", "net_margin"]
        for col in margin_cols:
            if col in df.columns:
                invalid = ((df[col] < -1) | (df[col] > 1)).sum()
                if invalid > 0:
                    warnings.append(f"{col} outside [-1, 1]: {invalid} rows")

        # Dividend yield bounds [0, 1]
        if "dividend_yield" in df.columns:
            invalid = ((df["dividend_yield"] < 0) | (df["dividend_yield"] > 1)).sum()
            if invalid > 0:
                warnings.append(f"dividend_yield outside [0, 1]: {invalid} rows")

        stats["row_count"] = len(df)
        stats["ticker_count"] = df["ticker"].nunique()

        is_valid = len(errors) == 0 if self.strict else True
        return ValidationResult(is_valid, errors, warnings, stats)

    def validate_realized_vol(self, df: pd.DataFrame) -> ValidationResult:
        """Validate realized volatility data."""
        errors = []
        warnings = []
        stats = {}

        required_cols = ["date", "ticker"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return ValidationResult(False, errors, warnings, stats)

        # RV should be non-negative and reasonable (< 500% annual)
        rv_cols = ["rv_5d", "rv_10d", "rv_21d", "rv_63d", "rv_parkinson", "rv_garman_klass"]
        for col in rv_cols:
            if col in df.columns:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    errors.append(f"Negative {col}: {neg_count} rows")
                extreme = (df[col] > 5).sum()
                if extreme > 0:
                    warnings.append(f"{col} > 500% annualized: {extreme} rows")

        stats["row_count"] = len(df)
        stats["ticker_count"] = df["ticker"].nunique()

        is_valid = len(errors) == 0 if self.strict else True
        return ValidationResult(is_valid, errors, warnings, stats)

    def check_continuity(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        ticker_col: str = "ticker",
        max_gap_days: int = 5,
    ) -> list[dict]:
        """Check for gaps in time series data."""
        gaps = []

        for ticker in df[ticker_col].unique():
            ticker_df = df[df[ticker_col] == ticker].sort_values(date_col)
            dates = pd.to_datetime(ticker_df[date_col])

            date_diffs = dates.diff().dt.days
            large_gaps = date_diffs[date_diffs > max_gap_days]

            for idx in large_gaps.index:
                gap_info = {
                    "ticker": ticker,
                    "gap_start": dates.loc[ticker_df.index[ticker_df.index.get_loc(idx) - 1]],
                    "gap_end": dates.loc[idx],
                    "gap_days": int(large_gaps.loc[idx]),
                }
                gaps.append(gap_info)

        return gaps

    def detect_outliers(
        self,
        df: pd.DataFrame,
        value_col: str,
        group_col: str = "ticker",
        method: str = "zscore",
        threshold: float = 4.0,
    ) -> pd.DataFrame:
        """Detect outliers in data."""
        if method == "zscore":
            df = df.copy()
            df["_zscore"] = df.groupby(group_col)[value_col].transform(
                lambda x: (x - x.mean()) / x.std()
            )
            outliers = df[df["_zscore"].abs() > threshold].copy()
            outliers.drop(columns=["_zscore"], inplace=True)
        elif method == "iqr":
            def iqr_outlier(x):
                q1 = x.quantile(0.25)
                q3 = x.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                return (x < lower) | (x > upper)

            mask = df.groupby(group_col)[value_col].transform(iqr_outlier)
            outliers = df[mask].copy()
        else:
            raise ValueError(f"Unknown method: {method}")

        return outliers
