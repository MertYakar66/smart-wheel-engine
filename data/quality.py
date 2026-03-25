"""
Data Quality Framework - Production-grade validation and monitoring.

Provides:
- Schema enforcement with Pydantic models
- Data contracts with SLA monitoring
- Anomaly detection (statistical + rules-based)
- Data freshness checks
- Quality dashboards and alerts
- Health checks with detailed diagnostics

Usage:
    from data.quality import DataQualityFramework

    dq = DataQualityFramework()

    # Validate a DataFrame
    result = dq.validate(df, schema="ohlcv", ticker="AAPL")

    # Check data freshness
    dq.check_freshness("ohlcv", max_age_hours=24)

    # Run all health checks
    health = dq.health_check()

    # Get quality report
    report = dq.quality_report()
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

# Import schemas
from src.data.schemas import (
    OHLCVSchema,
    OptionsFlowSchema,
    RealizedVolSchema,
    EarningsSchema,
    FundamentalsSchema,
    BorrowRateSchema,
)

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity levels for quality issues."""
    CRITICAL = "critical"    # Data unusable
    ERROR = "error"          # Significant issue
    WARNING = "warning"      # Minor issue
    INFO = "info"            # Informational


class CheckType(str, Enum):
    """Types of quality checks."""
    SCHEMA = "schema"
    COMPLETENESS = "completeness"
    FRESHNESS = "freshness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"
    ANOMALY = "anomaly"
    CUSTOM = "custom"


@dataclass
class QualityIssue:
    """A single quality issue."""
    check_type: CheckType
    severity: Severity
    column: Optional[str]
    message: str
    affected_rows: int = 0
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "check_type": self.check_type.value,
            "severity": self.severity.value,
            "column": self.column,
            "message": self.message,
            "affected_rows": self.affected_rows,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result of a validation run."""
    valid: bool
    schema_name: str
    ticker: Optional[str]
    row_count: int
    issues: List[QualityIssue] = field(default_factory=list)
    validated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validation_time_ms: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.CRITICAL)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "schema_name": self.schema_name,
            "ticker": self.ticker,
            "row_count": self.row_count,
            "issues": [i.to_dict() for i in self.issues],
            "validated_at": self.validated_at,
            "validation_time_ms": self.validation_time_ms,
            "summary": {
                "critical": self.critical_count,
                "error": self.error_count,
                "warning": self.warning_count,
            }
        }

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (
            f"[{status}] {self.schema_name}/{self.ticker}: "
            f"{self.row_count} rows, "
            f"{self.critical_count} critical, {self.error_count} errors, {self.warning_count} warnings"
        )


@dataclass
class DataContract:
    """Defines expectations for a data source."""
    name: str
    schema_name: str
    min_rows: int = 0
    max_staleness_hours: int = 24
    required_columns: List[str] = field(default_factory=list)
    null_thresholds: Dict[str, float] = field(default_factory=dict)  # column: max_null_pct
    value_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # column: (min, max)
    custom_checks: List[Callable[[pd.DataFrame], List[QualityIssue]]] = field(default_factory=list)


# Default data contracts
DEFAULT_CONTRACTS = {
    "ohlcv": DataContract(
        name="ohlcv",
        schema_name="ohlcv",
        min_rows=252,  # At least 1 year
        max_staleness_hours=24,
        required_columns=["date", "open", "high", "low", "close", "volume"],
        null_thresholds={
            "close": 0.01,  # Max 1% null
            "volume": 0.05,
        },
        value_ranges={
            "open": (0.01, 100000),
            "high": (0.01, 100000),
            "low": (0.01, 100000),
            "close": (0.01, 100000),
        },
    ),
    "options_flow": DataContract(
        name="options_flow",
        schema_name="options_flow",
        min_rows=100,
        required_columns=["date", "call_volume", "put_volume"],
        null_thresholds={
            "call_volume": 0.10,
            "put_volume": 0.10,
        },
    ),
    "volatility": DataContract(
        name="volatility",
        schema_name="volatility",
        min_rows=252,
        required_columns=["date", "rv_21d"],
        null_thresholds={
            "rv_21d": 0.10,
        },
        value_ranges={
            "rv_21d": (0.0, 5.0),  # 0% to 500% annualized vol
            "rv_63d": (0.0, 5.0),
        },
    ),
    "vol_edge": DataContract(
        name="vol_edge",
        schema_name="vol_edge",
        min_rows=252,
        required_columns=["date", "iv_rv_spread", "edge_score"],
        value_ranges={
            "iv_rv_spread": (-1.0, 1.0),  # -100% to +100%
            "edge_score": (0, 100),
            "iv_rank": (0, 100),
        },
    ),
}


class DataQualityFramework:
    """
    Production-grade data quality framework.

    Features:
    - Schema validation with Pydantic models
    - Data contracts with SLA monitoring
    - Statistical anomaly detection
    - Data freshness checks
    - Comprehensive health checks
    """

    # Map schema names to Pydantic models
    SCHEMA_MAP: Dict[str, Type[BaseModel]] = {
        "ohlcv": OHLCVSchema,
        "options_flow": OptionsFlowSchema,
        "volatility": RealizedVolSchema,
        "earnings": EarningsSchema,
        "fundamentals": FundamentalsSchema,
        "borrow_rates": BorrowRateSchema,
    }

    def __init__(
        self,
        contracts: Optional[Dict[str, DataContract]] = None,
        anomaly_zscore_threshold: float = 3.0,
        enable_strict_mode: bool = False,
    ):
        self.contracts = contracts or DEFAULT_CONTRACTS
        self.anomaly_zscore_threshold = anomaly_zscore_threshold
        self.strict_mode = enable_strict_mode

        # Validation history for tracking
        self._validation_history: List[ValidationResult] = []

        logger.info("DataQualityFramework initialized")

    def validate(
        self,
        df: pd.DataFrame,
        schema: str,
        ticker: Optional[str] = None,
        contract: Optional[DataContract] = None,
    ) -> ValidationResult:
        """
        Validate a DataFrame against schema and contract.

        Args:
            df: DataFrame to validate
            schema: Schema name (e.g., "ohlcv", "volatility")
            ticker: Ticker symbol (for logging)
            contract: Optional data contract (uses default if not provided)

        Returns:
            ValidationResult with all issues found
        """
        import time
        start_time = time.time()

        issues: List[QualityIssue] = []
        contract = contract or self.contracts.get(schema)

        # 1. Schema validation
        issues.extend(self._validate_schema(df, schema))

        # 2. Completeness checks
        issues.extend(self._check_completeness(df, contract))

        # 3. Consistency checks
        issues.extend(self._check_consistency(df, schema))

        # 4. Value range checks
        if contract:
            issues.extend(self._check_value_ranges(df, contract))

        # 5. Anomaly detection
        issues.extend(self._detect_anomalies(df))

        # 6. Custom contract checks
        if contract and contract.custom_checks:
            for check in contract.custom_checks:
                issues.extend(check(df))

        # Determine validity
        valid = not any(i.severity in (Severity.CRITICAL, Severity.ERROR) for i in issues)

        if self.strict_mode:
            valid = len(issues) == 0

        result = ValidationResult(
            valid=valid,
            schema_name=schema,
            ticker=ticker,
            row_count=len(df),
            issues=issues,
            validation_time_ms=int((time.time() - start_time) * 1000),
        )

        # Track history
        self._validation_history.append(result)

        if not valid:
            logger.warning(f"Validation failed: {result.summary()}")
        else:
            logger.debug(f"Validation passed: {result.summary()}")

        return result

    def _validate_schema(self, df: pd.DataFrame, schema: str) -> List[QualityIssue]:
        """Validate DataFrame against Pydantic schema."""
        issues = []
        schema_class = self.SCHEMA_MAP.get(schema)

        if schema_class is None:
            return issues  # No schema defined, skip

        # Sample validation (avoid validating every row for large DataFrames)
        sample_size = min(100, len(df))
        sample = df.sample(n=sample_size) if len(df) > sample_size else df

        validation_errors = 0
        error_columns = set()

        for idx, row in sample.iterrows():
            try:
                # Convert row to dict, handling date columns
                row_dict = {}
                for col, val in row.items():
                    if pd.isna(val):
                        row_dict[col.lower()] = None
                    elif isinstance(val, (pd.Timestamp, datetime)):
                        row_dict[col.lower()] = val.date() if hasattr(val, 'date') else val
                    else:
                        row_dict[col.lower()] = val

                schema_class.model_validate(row_dict)

            except ValidationError as e:
                validation_errors += 1
                for error in e.errors():
                    col = error.get("loc", [None])[0]
                    if col:
                        error_columns.add(col)

        if validation_errors > 0:
            error_pct = validation_errors / sample_size * 100
            severity = Severity.CRITICAL if error_pct > 50 else Severity.ERROR if error_pct > 10 else Severity.WARNING

            issues.append(QualityIssue(
                check_type=CheckType.SCHEMA,
                severity=severity,
                column=None,
                message=f"Schema validation failed for {error_pct:.1f}% of sampled rows",
                affected_rows=int(validation_errors * len(df) / sample_size),
                details={"error_columns": list(error_columns)},
            ))

        return issues

    def _check_completeness(
        self,
        df: pd.DataFrame,
        contract: Optional[DataContract],
    ) -> List[QualityIssue]:
        """Check data completeness."""
        issues = []

        # Check row count
        if contract and len(df) < contract.min_rows:
            issues.append(QualityIssue(
                check_type=CheckType.COMPLETENESS,
                severity=Severity.ERROR,
                column=None,
                message=f"Row count {len(df)} below minimum {contract.min_rows}",
                details={"actual": len(df), "required": contract.min_rows},
            ))

        # Check required columns
        if contract:
            missing_cols = set(contract.required_columns) - set(df.columns.str.lower())
            if missing_cols:
                issues.append(QualityIssue(
                    check_type=CheckType.COMPLETENESS,
                    severity=Severity.CRITICAL,
                    column=None,
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing": list(missing_cols)},
                ))

        # Check null thresholds
        if contract and contract.null_thresholds:
            for col, max_null_pct in contract.null_thresholds.items():
                col_lower = col.lower()
                matching_cols = [c for c in df.columns if c.lower() == col_lower]
                if matching_cols:
                    actual_col = matching_cols[0]
                    null_pct = df[actual_col].isna().mean()
                    if null_pct > max_null_pct:
                        issues.append(QualityIssue(
                            check_type=CheckType.COMPLETENESS,
                            severity=Severity.WARNING if null_pct < 0.5 else Severity.ERROR,
                            column=col,
                            message=f"Null rate {null_pct:.1%} exceeds threshold {max_null_pct:.1%}",
                            affected_rows=int(df[actual_col].isna().sum()),
                            details={"null_pct": null_pct, "threshold": max_null_pct},
                        ))

        return issues

    def _check_consistency(self, df: pd.DataFrame, schema: str) -> List[QualityIssue]:
        """Check data consistency rules."""
        issues = []

        # OHLCV consistency
        if schema == "ohlcv":
            issues.extend(self._check_ohlcv_consistency(df))

        # Options consistency
        if schema in ("options_flow", "options_chain"):
            issues.extend(self._check_options_consistency(df))

        # Time series consistency
        if "date" in df.columns:
            issues.extend(self._check_time_series_consistency(df))

        return issues

    def _check_ohlcv_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check OHLCV-specific consistency rules."""
        issues = []

        # Normalize column names
        df_check = df.copy()
        df_check.columns = df_check.columns.str.lower()

        # High >= Low
        if "high" in df_check.columns and "low" in df_check.columns:
            violations = df_check["high"] < df_check["low"]
            if violations.any():
                issues.append(QualityIssue(
                    check_type=CheckType.CONSISTENCY,
                    severity=Severity.CRITICAL,
                    column="high/low",
                    message="High price less than Low price",
                    affected_rows=int(violations.sum()),
                ))

        # High >= Open, Close
        for col in ["open", "close"]:
            if "high" in df_check.columns and col in df_check.columns:
                violations = df_check["high"] < df_check[col]
                if violations.any():
                    issues.append(QualityIssue(
                        check_type=CheckType.CONSISTENCY,
                        severity=Severity.ERROR,
                        column=f"high/{col}",
                        message=f"High price less than {col.title()} price",
                        affected_rows=int(violations.sum()),
                    ))

        # Low <= Open, Close
        for col in ["open", "close"]:
            if "low" in df_check.columns and col in df_check.columns:
                violations = df_check["low"] > df_check[col]
                if violations.any():
                    issues.append(QualityIssue(
                        check_type=CheckType.CONSISTENCY,
                        severity=Severity.ERROR,
                        column=f"low/{col}",
                        message=f"Low price greater than {col.title()} price",
                        affected_rows=int(violations.sum()),
                    ))

        # Volume >= 0
        if "volume" in df_check.columns:
            negative_vol = df_check["volume"] < 0
            if negative_vol.any():
                issues.append(QualityIssue(
                    check_type=CheckType.CONSISTENCY,
                    severity=Severity.CRITICAL,
                    column="volume",
                    message="Negative volume values",
                    affected_rows=int(negative_vol.sum()),
                ))

        return issues

    def _check_options_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check options-specific consistency rules."""
        issues = []
        df_check = df.copy()
        df_check.columns = df_check.columns.str.lower()

        # Non-negative volumes
        for col in ["call_volume", "put_volume", "call_oi", "put_oi"]:
            if col in df_check.columns:
                negative = df_check[col] < 0
                if negative.any():
                    issues.append(QualityIssue(
                        check_type=CheckType.CONSISTENCY,
                        severity=Severity.ERROR,
                        column=col,
                        message=f"Negative {col} values",
                        affected_rows=int(negative.sum()),
                    ))

        # IV in valid range (0 to 500%)
        iv_cols = [c for c in df_check.columns if "iv" in c.lower() or "impvol" in c.lower()]
        for col in iv_cols:
            if df_check[col].dtype in (np.float64, np.float32):
                invalid_iv = (df_check[col] < 0) | (df_check[col] > 5.0)
                if invalid_iv.any():
                    issues.append(QualityIssue(
                        check_type=CheckType.CONSISTENCY,
                        severity=Severity.WARNING,
                        column=col,
                        message=f"IV values outside valid range [0, 5.0]",
                        affected_rows=int(invalid_iv.sum()),
                    ))

        return issues

    def _check_time_series_consistency(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Check time series consistency."""
        issues = []

        df_check = df.copy()
        df_check.columns = df_check.columns.str.lower()

        if "date" not in df_check.columns:
            return issues

        dates = pd.to_datetime(df_check["date"])

        # Check for duplicates
        duplicates = dates.duplicated()
        if duplicates.any():
            issues.append(QualityIssue(
                check_type=CheckType.CONSISTENCY,
                severity=Severity.WARNING,
                column="date",
                message="Duplicate dates found",
                affected_rows=int(duplicates.sum()),
            ))

        # Check for gaps (more than 5 business days)
        if len(dates) > 1:
            sorted_dates = dates.sort_values()
            gaps = sorted_dates.diff().dt.days
            large_gaps = gaps > 7  # More than a week
            if large_gaps.any():
                max_gap = int(gaps.max())
                issues.append(QualityIssue(
                    check_type=CheckType.CONSISTENCY,
                    severity=Severity.INFO if max_gap < 30 else Severity.WARNING,
                    column="date",
                    message=f"Date gaps detected (max: {max_gap} days)",
                    details={"max_gap_days": max_gap},
                ))

        # Check chronological order
        if not dates.is_monotonic_increasing and not dates.is_monotonic_decreasing:
            issues.append(QualityIssue(
                check_type=CheckType.CONSISTENCY,
                severity=Severity.INFO,
                column="date",
                message="Dates not in chronological order",
            ))

        return issues

    def _check_value_ranges(
        self,
        df: pd.DataFrame,
        contract: DataContract,
    ) -> List[QualityIssue]:
        """Check value ranges from contract."""
        issues = []

        for col, (min_val, max_val) in contract.value_ranges.items():
            col_lower = col.lower()
            matching_cols = [c for c in df.columns if c.lower() == col_lower]

            if not matching_cols:
                continue

            actual_col = matching_cols[0]
            series = df[actual_col].dropna()

            if len(series) == 0:
                continue

            below_min = series < min_val
            above_max = series > max_val

            if below_min.any():
                issues.append(QualityIssue(
                    check_type=CheckType.ACCURACY,
                    severity=Severity.WARNING,
                    column=col,
                    message=f"Values below minimum {min_val}",
                    affected_rows=int(below_min.sum()),
                    details={"min_found": float(series.min())},
                ))

            if above_max.any():
                issues.append(QualityIssue(
                    check_type=CheckType.ACCURACY,
                    severity=Severity.WARNING,
                    column=col,
                    message=f"Values above maximum {max_val}",
                    affected_rows=int(above_max.sum()),
                    details={"max_found": float(series.max())},
                ))

        return issues

    def _detect_anomalies(self, df: pd.DataFrame) -> List[QualityIssue]:
        """Detect statistical anomalies using z-score."""
        issues = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 30:  # Need enough data for statistics
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            z_scores = np.abs((series - mean) / std)
            anomalies = z_scores > self.anomaly_zscore_threshold

            if anomalies.any():
                anomaly_count = int(anomalies.sum())
                if anomaly_count > len(series) * 0.01:  # More than 1% anomalies
                    issues.append(QualityIssue(
                        check_type=CheckType.ANOMALY,
                        severity=Severity.INFO,
                        column=col,
                        message=f"Statistical anomalies detected (z-score > {self.anomaly_zscore_threshold})",
                        affected_rows=anomaly_count,
                        details={
                            "anomaly_pct": anomaly_count / len(series),
                            "max_zscore": float(z_scores.max()),
                        },
                    ))

        return issues

    def check_freshness(
        self,
        df: pd.DataFrame,
        max_age_hours: int = 24,
    ) -> List[QualityIssue]:
        """Check data freshness."""
        issues = []

        # Check for date column
        date_col = None
        for col in ["date", "Date", "DATE", "timestamp", "Timestamp"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return issues

        max_date = pd.to_datetime(df[date_col]).max()
        age = datetime.now() - max_date.to_pydatetime()
        age_hours = age.total_seconds() / 3600

        if age_hours > max_age_hours:
            severity = Severity.WARNING if age_hours < max_age_hours * 3 else Severity.ERROR
            issues.append(QualityIssue(
                check_type=CheckType.FRESHNESS,
                severity=severity,
                column=date_col,
                message=f"Data is {age_hours:.1f} hours old (threshold: {max_age_hours}h)",
                details={
                    "max_date": str(max_date),
                    "age_hours": age_hours,
                    "threshold_hours": max_age_hours,
                },
            ))

        return issues

    def validate_contract(
        self,
        df: pd.DataFrame,
        contract_name: str,
        ticker: Optional[str] = None,
    ) -> ValidationResult:
        """Validate DataFrame against a named contract."""
        contract = self.contracts.get(contract_name)
        if contract is None:
            raise ValueError(f"Unknown contract: {contract_name}")

        return self.validate(df, contract.schema_name, ticker, contract)

    def add_contract(self, contract: DataContract) -> None:
        """Add or update a data contract."""
        self.contracts[contract.name] = contract
        logger.info(f"Added contract: {contract.name}")

    def get_validation_history(
        self,
        schema: Optional[str] = None,
        ticker: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[ValidationResult]:
        """Get validation history with optional filters."""
        history = self._validation_history

        if schema:
            history = [r for r in history if r.schema_name == schema]

        if ticker:
            history = [r for r in history if r.ticker == ticker]

        if since:
            history = [r for r in history if datetime.fromisoformat(r.validated_at) >= since]

        return history

    def quality_report(self) -> dict:
        """Generate a quality report from validation history."""
        if not self._validation_history:
            return {"message": "No validation history"}

        total = len(self._validation_history)
        valid_count = sum(1 for r in self._validation_history if r.valid)
        all_issues = [i for r in self._validation_history for i in r.issues]

        # Aggregate by severity
        severity_counts = {s.value: 0 for s in Severity}
        for issue in all_issues:
            severity_counts[issue.severity.value] += 1

        # Aggregate by check type
        check_counts = {c.value: 0 for c in CheckType}
        for issue in all_issues:
            check_counts[issue.check_type.value] += 1

        # Most common issues
        issue_messages = {}
        for issue in all_issues:
            key = (issue.check_type.value, issue.message)
            issue_messages[key] = issue_messages.get(key, 0) + 1

        top_issues = sorted(issue_messages.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "summary": {
                "total_validations": total,
                "valid": valid_count,
                "invalid": total - valid_count,
                "validity_rate": valid_count / total if total > 0 else 0,
            },
            "severity_breakdown": severity_counts,
            "check_type_breakdown": check_counts,
            "top_issues": [
                {"check_type": k[0], "message": k[1], "count": v}
                for k, v in top_issues
            ],
        }

    def health_check(self) -> dict:
        """Run comprehensive health checks."""
        issues = []
        checks_passed = 0
        checks_total = 0

        # Check validation history health
        checks_total += 1
        recent_validations = self.get_validation_history(
            since=datetime.now() - timedelta(hours=24)
        )
        if not recent_validations:
            issues.append("No validations in last 24 hours")
        else:
            checks_passed += 1

        # Check recent failure rate
        checks_total += 1
        if recent_validations:
            failure_rate = 1 - (sum(1 for r in recent_validations if r.valid) / len(recent_validations))
            if failure_rate > 0.5:
                issues.append(f"High failure rate: {failure_rate:.1%}")
            else:
                checks_passed += 1
        else:
            checks_passed += 1

        # Check for critical issues
        checks_total += 1
        critical_issues = [
            i for r in recent_validations for i in r.issues
            if i.severity == Severity.CRITICAL
        ]
        if critical_issues:
            issues.append(f"{len(critical_issues)} critical issues in last 24h")
        else:
            checks_passed += 1

        return {
            "healthy": checks_passed == checks_total,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "issues": issues,
            "validation_count_24h": len(recent_validations),
        }


# Convenience functions
def validate_ohlcv(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """Validate OHLCV data."""
    dq = DataQualityFramework()
    return dq.validate(df, "ohlcv", ticker)


def validate_options(df: pd.DataFrame, ticker: str) -> ValidationResult:
    """Validate options data."""
    dq = DataQualityFramework()
    return dq.validate(df, "options_flow", ticker)
