"""
Data validation framework for option and OHLCV data.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Data is unusable
    WARNING = "warning"  # Data is suspicious but usable
    INFO = "info"        # Minor issue


@dataclass
class ValidationIssue:
    """Single validation issue."""
    field: str
    severity: ValidationSeverity
    message: str
    row_count: int = 0
    sample_values: List = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of data validation."""
    valid_df: pd.DataFrame
    invalid_df: pd.DataFrame
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.invalid_df) == 0 and not any(
            i.severity == ValidationSeverity.ERROR for i in self.issues
        )

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def log_summary(self):
        """Log validation summary."""
        logger.info(f"Validation: {len(self.valid_df)} valid, {len(self.invalid_df)} invalid")
        for issue in self.issues:
            if issue.severity == ValidationSeverity.ERROR:
                logger.error(f"{issue.field}: {issue.message} ({issue.row_count} rows)")
            elif issue.severity == ValidationSeverity.WARNING:
                logger.warning(f"{issue.field}: {issue.message} ({issue.row_count} rows)")
            else:
                logger.info(f"{issue.field}: {issue.message}")


def validate_and_normalize_iv(iv: float) -> Tuple[Optional[float], Optional[str]]:
    """
    Validate and normalize implied volatility to decimal form.

    Args:
        iv: Implied volatility value

    Returns:
        Tuple of (normalized_iv, warning_message)
        Returns (None, error_message) if invalid
    """
    if pd.isna(iv):
        return None, "IV is NaN"

    if iv <= 0:
        return None, f"IV is non-positive: {iv}"

    # Likely percentage format (e.g., 25 instead of 0.25)
    if iv > 10.0:
        normalized = iv / 100.0
        return normalized, f"IV appears to be percentage format, converted {iv} -> {normalized}"

    # Very high but plausible IV (>200%)
    if iv > 2.0:
        return iv, f"Unusually high IV: {iv:.1%}"

    # Very low IV (<5%)
    if iv < 0.05:
        return iv, f"Unusually low IV: {iv:.1%}"

    return iv, None


def validate_option_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate option data for quality issues.

    Args:
        df: DataFrame with option data

    Returns:
        ValidationResult with valid/invalid splits and issues list
    """
    if df.empty:
        return ValidationResult(df, pd.DataFrame(), [])

    df = df.copy()
    issues = []
    invalid_mask = pd.Series(False, index=df.index)

    # 1. Check for required columns
    required_cols = ['strike', 'bid', 'ask', 'implied_vol', 'underlying_price']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(ValidationIssue(
            field="columns",
            severity=ValidationSeverity.ERROR,
            message=f"Missing required columns: {missing_cols}"
        ))
        return ValidationResult(pd.DataFrame(), df, issues)

    # 2. Validate and normalize IV
    iv_issues = []
    normalized_ivs = []
    for idx, iv in df['implied_vol'].items():
        norm_iv, msg = validate_and_normalize_iv(iv)
        normalized_ivs.append(norm_iv)
        if norm_iv is None:
            iv_issues.append((idx, msg))
        elif msg:
            iv_issues.append((idx, msg))

    df['implied_vol_normalized'] = normalized_ivs
    invalid_iv = df['implied_vol_normalized'].isna()
    if invalid_iv.any():
        issues.append(ValidationIssue(
            field="implied_vol",
            severity=ValidationSeverity.WARNING,
            message="Invalid IV values removed",
            row_count=invalid_iv.sum(),
            sample_values=df.loc[invalid_iv, 'implied_vol'].head(5).tolist()
        ))
        invalid_mask |= invalid_iv

    # Replace original IV with normalized
    df['implied_vol'] = df['implied_vol_normalized']
    df = df.drop(columns=['implied_vol_normalized'])

    # 3. Bid-ask consistency
    bid_gt_ask = (df['bid'] > df['ask']) & df['bid'].notna() & df['ask'].notna()
    if bid_gt_ask.any():
        issues.append(ValidationIssue(
            field="bid_ask",
            severity=ValidationSeverity.ERROR,
            message="Bid > Ask (invalid market data)",
            row_count=bid_gt_ask.sum()
        ))
        invalid_mask |= bid_gt_ask

    # 4. Negative prices
    negative_bid = df['bid'] < 0
    negative_ask = df['ask'] < 0
    if negative_bid.any() or negative_ask.any():
        issues.append(ValidationIssue(
            field="prices",
            severity=ValidationSeverity.ERROR,
            message="Negative bid or ask prices",
            row_count=(negative_bid | negative_ask).sum()
        ))
        invalid_mask |= negative_bid | negative_ask

    # 5. Zero bid with positive ask (common but flag it)
    zero_bid = (df['bid'] == 0) & (df['ask'] > 0)
    if zero_bid.any():
        issues.append(ValidationIssue(
            field="bid",
            severity=ValidationSeverity.INFO,
            message="Zero bid (wide spread, possibly illiquid)",
            row_count=zero_bid.sum()
        ))

    # 6. Calculate and validate mid price
    df['mid_price_calc'] = (df['bid'].fillna(0) + df['ask'].fillna(0)) / 2
    if 'mid_price' in df.columns:
        mid_mismatch = abs(df['mid_price'] - df['mid_price_calc']) > 0.01
        if mid_mismatch.any():
            issues.append(ValidationIssue(
                field="mid_price",
                severity=ValidationSeverity.WARNING,
                message="Mid price doesn't match (bid+ask)/2",
                row_count=mid_mismatch.sum()
            ))

    # 7. Intrinsic value floor check
    if 'option_type' in df.columns and 'underlying_price' in df.columns:
        # Put intrinsic: max(0, strike - underlying)
        put_mask = df['option_type'].astype(str).str.upper().str.startswith('P')
        put_intrinsic = np.maximum(0, df['strike'] - df['underlying_price'])
        put_below_intrinsic = put_mask & (df['mid_price_calc'] < put_intrinsic * 0.95)

        # Call intrinsic: max(0, underlying - strike)
        call_mask = df['option_type'].astype(str).str.upper().str.startswith('C')
        call_intrinsic = np.maximum(0, df['underlying_price'] - df['strike'])
        call_below_intrinsic = call_mask & (df['mid_price_calc'] < call_intrinsic * 0.95)

        below_intrinsic = put_below_intrinsic | call_below_intrinsic
        if below_intrinsic.any():
            issues.append(ValidationIssue(
                field="intrinsic_value",
                severity=ValidationSeverity.WARNING,
                message="Option price below intrinsic value",
                row_count=below_intrinsic.sum()
            ))

    df = df.drop(columns=['mid_price_calc'], errors='ignore')

    # 8. Wide spread check (>50% of mid)
    spread = df['ask'] - df['bid']
    mid = (df['bid'] + df['ask']) / 2
    wide_spread = (spread / mid > 0.50) & (mid > 0)
    if wide_spread.any():
        issues.append(ValidationIssue(
            field="spread",
            severity=ValidationSeverity.INFO,
            message="Wide bid-ask spread (>50%)",
            row_count=wide_spread.sum()
        ))

    # Split valid/invalid
    valid_df = df[~invalid_mask].copy()
    invalid_df = df[invalid_mask].copy()

    return ValidationResult(valid_df, invalid_df, issues)


def validate_ohlcv_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate OHLCV data for quality issues.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        ValidationResult with valid/invalid splits
    """
    if df.empty:
        return ValidationResult(df, pd.DataFrame(), [])

    df = df.copy()
    issues = []
    invalid_mask = pd.Series(False, index=df.index)

    # 1. Check required columns
    required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        issues.append(ValidationIssue(
            field="columns",
            severity=ValidationSeverity.ERROR,
            message=f"Missing required columns: {missing_cols}"
        ))
        return ValidationResult(pd.DataFrame(), df, issues)

    # 2. Date validation
    invalid_date = pd.to_datetime(df['Date'], errors='coerce').isna()
    if invalid_date.any():
        issues.append(ValidationIssue(
            field="Date",
            severity=ValidationSeverity.ERROR,
            message="Invalid date values",
            row_count=invalid_date.sum()
        ))
        invalid_mask |= invalid_date

    # 3. Price sanity: High >= Low
    high_lt_low = df['High'] < df['Low']
    if high_lt_low.any():
        issues.append(ValidationIssue(
            field="High_Low",
            severity=ValidationSeverity.ERROR,
            message="High < Low (impossible)",
            row_count=high_lt_low.sum()
        ))
        invalid_mask |= high_lt_low

    # 4. Open/Close within High/Low
    open_outside = (df['Open'] < df['Low']) | (df['Open'] > df['High'])
    close_outside = (df['Close'] < df['Low']) | (df['Close'] > df['High'])
    if open_outside.any() or close_outside.any():
        issues.append(ValidationIssue(
            field="OHLC_consistency",
            severity=ValidationSeverity.WARNING,
            message="Open or Close outside High-Low range",
            row_count=(open_outside | close_outside).sum()
        ))

    # 5. Negative prices
    negative_prices = (df['Open'] < 0) | (df['High'] < 0) | (df['Low'] < 0) | (df['Close'] < 0)
    if negative_prices.any():
        issues.append(ValidationIssue(
            field="prices",
            severity=ValidationSeverity.ERROR,
            message="Negative prices",
            row_count=negative_prices.sum()
        ))
        invalid_mask |= negative_prices

    # 6. Extreme daily moves (>50% - likely data error)
    if len(df) > 1:
        df_sorted = df.sort_values('Date')
        pct_change = df_sorted['Close'].pct_change().abs()
        extreme_move = pct_change > 0.50
        if extreme_move.any():
            issues.append(ValidationIssue(
                field="price_change",
                severity=ValidationSeverity.WARNING,
                message="Extreme daily price change (>50%)",
                row_count=extreme_move.sum()
            ))

    # Split valid/invalid
    valid_df = df[~invalid_mask].copy()
    invalid_df = df[invalid_mask].copy()

    return ValidationResult(valid_df, invalid_df, issues)


def apply_liquidity_filter(
    df: pd.DataFrame,
    min_open_interest: int = 100,
    min_bid: float = 0.01,
    max_spread_pct: float = 0.50
) -> pd.DataFrame:
    """
    Filter options data by liquidity criteria.

    Args:
        df: Options DataFrame
        min_open_interest: Minimum open interest
        min_bid: Minimum bid price (>0 means real market)
        max_spread_pct: Maximum spread as percentage of mid

    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df

    df = df.copy()

    # Calculate spread percentage
    mid = (df['bid'] + df['ask']) / 2
    spread_pct = (df['ask'] - df['bid']) / mid
    spread_pct = spread_pct.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    mask = (
        (df.get('open_interest', 0) >= min_open_interest) &
        (df['bid'] >= min_bid) &
        (spread_pct <= max_spread_pct)
    )

    filtered = df[mask].copy()
    removed = len(df) - len(filtered)

    if removed > 0:
        logger.info(f"Liquidity filter removed {removed} options ({removed/len(df)*100:.1f}%)")

    return filtered
