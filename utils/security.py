"""
Security Module for Smart Wheel Engine

Provides:
- Input validation and sanitization
- Secure configuration management
- Secrets handling
- Audit logging
- Rate limiting utilities
"""

import os
import re
import hashlib
import logging
import secrets
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import json


# =============================================================================
# Secure Logging
# =============================================================================

class AuditLogger:
    """
    Secure audit logging for trading operations.

    Logs all sensitive operations with timestamps and user context.
    Redacts sensitive information automatically.
    """

    SENSITIVE_PATTERNS = [
        r'api[_-]?key',
        r'secret',
        r'password',
        r'token',
        r'credential',
        r'auth',
    ]

    def __init__(self, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger('smart_wheel_audit')
        self.logger.setLevel(level)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console handler
            console = logging.StreamHandler()
            console.setFormatter(formatter)
            self.logger.addHandler(console)

            # File handler (if specified)
            if log_file:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def _redact_sensitive(self, message: str) -> str:
        """Redact sensitive information from log messages."""
        redacted = message
        for pattern in self.SENSITIVE_PATTERNS:
            # Redact values after sensitive keys
            redacted = re.sub(
                rf'({pattern})\s*[=:]\s*["\']?([^"\'\s,}}]+)',
                r'\1=***REDACTED***',
                redacted,
                flags=re.IGNORECASE
            )
        return redacted

    def log_trade(self, action: str, details: Dict[str, Any]) -> None:
        """Log trading operation."""
        safe_details = self._redact_sensitive(json.dumps(details))
        self.logger.info(f"TRADE | {action} | {safe_details}")

    def log_risk_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log risk management event."""
        safe_details = self._redact_sensitive(json.dumps(details))
        self.logger.warning(f"RISK | {event_type} | {safe_details}")

    def log_security_event(self, event_type: str, details: str) -> None:
        """Log security-related event."""
        safe_details = self._redact_sensitive(details)
        self.logger.critical(f"SECURITY | {event_type} | {safe_details}")

    def log_access(self, resource: str, action: str, success: bool) -> None:
        """Log access attempts."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"ACCESS | {resource} | {action} | {status}")


# =============================================================================
# Input Validation
# =============================================================================

class InputValidator:
    """
    Validates and sanitizes user inputs for trading operations.

    Prevents injection attacks and ensures data integrity.
    """

    # Valid ranges for financial inputs
    PRICE_MIN = 0.0001
    PRICE_MAX = 1_000_000
    QUANTITY_MIN = 1
    QUANTITY_MAX = 10_000
    VOLATILITY_MIN = 0.0
    VOLATILITY_MAX = 10.0  # 1000% vol
    RATE_MIN = -0.50
    RATE_MAX = 1.0
    DTE_MIN = 0
    DTE_MAX = 3650  # 10 years

    @staticmethod
    def validate_price(value: float, name: str = "price") -> float:
        """Validate and return a price value."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
        if value < InputValidator.PRICE_MIN:
            raise ValueError(f"{name} must be >= {InputValidator.PRICE_MIN}, got {value}")
        if value > InputValidator.PRICE_MAX:
            raise ValueError(f"{name} must be <= {InputValidator.PRICE_MAX}, got {value}")
        return float(value)

    @staticmethod
    def validate_quantity(value: int, name: str = "quantity") -> int:
        """Validate and return a quantity value."""
        if not isinstance(value, int):
            raise ValueError(f"{name} must be integer, got {type(value).__name__}")
        if value < InputValidator.QUANTITY_MIN:
            raise ValueError(f"{name} must be >= {InputValidator.QUANTITY_MIN}, got {value}")
        if value > InputValidator.QUANTITY_MAX:
            raise ValueError(f"{name} must be <= {InputValidator.QUANTITY_MAX}, got {value}")
        return value

    @staticmethod
    def validate_volatility(value: float, name: str = "volatility") -> float:
        """Validate and return a volatility value."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
        if value < InputValidator.VOLATILITY_MIN:
            raise ValueError(f"{name} must be >= {InputValidator.VOLATILITY_MIN}, got {value}")
        if value > InputValidator.VOLATILITY_MAX:
            raise ValueError(f"{name} must be <= {InputValidator.VOLATILITY_MAX}, got {value}")
        return float(value)

    @staticmethod
    def validate_rate(value: float, name: str = "rate") -> float:
        """Validate and return an interest rate value."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value).__name__}")
        if value < InputValidator.RATE_MIN:
            raise ValueError(f"{name} must be >= {InputValidator.RATE_MIN}, got {value}")
        if value > InputValidator.RATE_MAX:
            raise ValueError(f"{name} must be <= {InputValidator.RATE_MAX}, got {value}")
        return float(value)

    @staticmethod
    def validate_dte(value: int, name: str = "dte") -> int:
        """Validate and return days to expiration."""
        if not isinstance(value, int):
            raise ValueError(f"{name} must be integer, got {type(value).__name__}")
        if value < InputValidator.DTE_MIN:
            raise ValueError(f"{name} must be >= {InputValidator.DTE_MIN}, got {value}")
        if value > InputValidator.DTE_MAX:
            raise ValueError(f"{name} must be <= {InputValidator.DTE_MAX}, got {value}")
        return value

    @staticmethod
    def validate_option_type(value: str) -> str:
        """Validate option type."""
        if not isinstance(value, str):
            raise ValueError(f"option_type must be string, got {type(value).__name__}")
        value = value.lower().strip()
        if value not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put', got '{value}'")
        return value

    @staticmethod
    def validate_symbol(value: str) -> str:
        """Validate and sanitize stock symbol."""
        if not isinstance(value, str):
            raise ValueError(f"symbol must be string, got {type(value).__name__}")
        # Remove any non-alphanumeric characters
        sanitized = re.sub(r'[^A-Za-z0-9.]', '', value)
        if len(sanitized) == 0:
            raise ValueError("symbol cannot be empty after sanitization")
        if len(sanitized) > 10:
            raise ValueError(f"symbol too long: {len(sanitized)} chars")
        return sanitized.upper()

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize a general string input."""
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        # Truncate if too long
        return sanitized[:max_length]


# =============================================================================
# Secrets Management
# =============================================================================

@dataclass
class SecureConfig:
    """
    Secure configuration management.

    Loads configuration from environment variables and .env files.
    Never logs or exposes sensitive values.
    """

    _secrets: Dict[str, str] = field(default_factory=dict, repr=False)
    _loaded: bool = False

    def load_env_file(self, path: str = ".env") -> None:
        """Load environment variables from .env file."""
        env_path = Path(path)
        if not env_path.exists():
            return

        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    self._secrets[key] = value
                    # Also set in os.environ for compatibility
                    os.environ[key] = value

        self._loaded = True

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a configuration value."""
        # Check loaded secrets first, then environment
        return self._secrets.get(key, os.environ.get(key, default))

    def get_required(self, key: str) -> str:
        """Get a required configuration value."""
        value = self.get(key)
        if value is None:
            raise ValueError(f"Required configuration '{key}' not found")
        return value

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for a service."""
        key_names = [
            f"{service.upper()}_API_KEY",
            f"{service.upper()}_KEY",
            f"{service}_api_key",
        ]
        for name in key_names:
            value = self.get(name)
            if value:
                return value
        return None

    def is_loaded(self) -> bool:
        """Check if configuration has been loaded."""
        return self._loaded

    def list_keys(self) -> List[str]:
        """List available configuration keys (not values)."""
        return list(self._secrets.keys())


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    """
    Rate limiter for API calls and operations.

    Prevents excessive requests and protects against abuse.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: List[datetime] = []

    def is_allowed(self) -> bool:
        """Check if a request is allowed."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Remove old requests
        self._requests = [r for r in self._requests if r > cutoff]

        # Check limit
        if len(self._requests) >= self.max_requests:
            return False

        self._requests.append(now)
        return True

    def wait_time(self) -> float:
        """Get seconds to wait before next request is allowed."""
        if len(self._requests) < self.max_requests:
            return 0.0

        now = datetime.now()
        oldest = min(self._requests)
        wait = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
        return max(0.0, wait)

    def reset(self) -> None:
        """Reset the rate limiter."""
        self._requests = []


def rate_limited(limiter: RateLimiter):
    """Decorator to apply rate limiting to a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not limiter.is_allowed():
                wait = limiter.wait_time()
                raise RuntimeError(
                    f"Rate limit exceeded. Wait {wait:.1f} seconds."
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Security Utilities
# =============================================================================

def generate_session_token(length: int = 32) -> str:
    """Generate a secure random session token."""
    return secrets.token_urlsafe(length)


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """Hash sensitive data for storage or comparison."""
    if salt is None:
        salt = secrets.token_hex(16)
    combined = f"{salt}{data}".encode('utf-8')
    hashed = hashlib.sha256(combined).hexdigest()
    return f"{salt}:{hashed}"


def verify_hash(data: str, hashed: str) -> bool:
    """Verify data against a hash."""
    try:
        salt, expected_hash = hashed.split(':')
        combined = f"{salt}{data}".encode('utf-8')
        actual_hash = hashlib.sha256(combined).hexdigest()
        return secrets.compare_digest(actual_hash, expected_hash)
    except (ValueError, AttributeError):
        return False


def mask_sensitive(value: str, show_chars: int = 4) -> str:
    """Mask a sensitive string, showing only last few characters."""
    if len(value) <= show_chars:
        return '*' * len(value)
    return '*' * (len(value) - show_chars) + value[-show_chars:]


# =============================================================================
# Secure Decorators
# =============================================================================

def validate_inputs(**validators):
    """
    Decorator to validate function inputs.

    Usage:
        @validate_inputs(spot='price', strike='price', dte='dte')
        def price_option(spot, strike, dte):
            ...
    """
    validator_map = {
        'price': InputValidator.validate_price,
        'quantity': InputValidator.validate_quantity,
        'volatility': InputValidator.validate_volatility,
        'rate': InputValidator.validate_rate,
        'dte': InputValidator.validate_dte,
        'option_type': InputValidator.validate_option_type,
        'symbol': InputValidator.validate_symbol,
    }

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate keyword arguments
            for param_name, validator_name in validators.items():
                if param_name in kwargs:
                    validator = validator_map.get(validator_name)
                    if validator:
                        kwargs[param_name] = validator(kwargs[param_name], param_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def audit_logged(audit_logger: AuditLogger, action_type: str = "OPERATION"):
    """Decorator to log function calls for audit purposes."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                audit_logger.log_trade(
                    f"{action_type}:{func.__name__}",
                    {"status": "success", "duration_ms": (datetime.now() - start_time).total_seconds() * 1000}
                )
                return result
            except Exception as e:
                audit_logger.log_risk_event(
                    f"{action_type}:{func.__name__}:ERROR",
                    {"error": str(e), "duration_ms": (datetime.now() - start_time).total_seconds() * 1000}
                )
                raise
        return wrapper
    return decorator


# =============================================================================
# Module Initialization
# =============================================================================

# Global instances
config = SecureConfig()
audit_log = AuditLogger()

# Default rate limiters
api_limiter = RateLimiter(max_requests=100, window_seconds=60)  # 100 req/min
trade_limiter = RateLimiter(max_requests=10, window_seconds=1)  # 10 trades/sec


__all__ = [
    'AuditLogger',
    'InputValidator',
    'SecureConfig',
    'RateLimiter',
    'rate_limited',
    'generate_session_token',
    'hash_sensitive_data',
    'verify_hash',
    'mask_sensitive',
    'validate_inputs',
    'audit_logged',
    'config',
    'audit_log',
    'api_limiter',
    'trade_limiter',
]
