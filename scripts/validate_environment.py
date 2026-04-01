#!/usr/bin/env python3
"""
Environment Validation Script

Validates that the runtime environment meets all requirements for Smart Wheel Engine.
Used in CI/CD pipelines and local development to ensure environment parity.

Exit codes:
    0: All checks passed
    1: Critical checks failed (blocking)
    2: Warning checks failed (non-blocking)
"""

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CheckSeverity(Enum):
    """Severity levels for validation checks."""
    CRITICAL = "critical"  # Blocks execution
    WARNING = "warning"    # Logs warning but continues
    INFO = "info"          # Informational only


class CheckStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    status: CheckStatus
    severity: CheckSeverity
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: list[CheckResult] = field(default_factory=list)

    @property
    def has_critical_failures(self) -> bool:
        return any(
            r.status == CheckStatus.FAILED and r.severity == CheckSeverity.CRITICAL
            for r in self.results
        )

    @property
    def has_warnings(self) -> bool:
        return any(
            r.status == CheckStatus.FAILED and r.severity == CheckSeverity.WARNING
            for r in self.results
        )

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r.status == CheckStatus.PASSED),
            "failed": sum(1 for r in self.results if r.status == CheckStatus.FAILED),
            "skipped": sum(1 for r in self.results if r.status == CheckStatus.SKIPPED),
        }


class EnvironmentValidator:
    """Validates the runtime environment against requirements."""

    # Required Python version
    REQUIRED_PYTHON_MAJOR = 3
    REQUIRED_PYTHON_MINOR = 11

    # Core dependencies (always required)
    CORE_DEPENDENCIES = [
        ("numpy", "1.24.0"),
        ("pandas", "2.0.0"),
        ("scipy", "1.10.0"),
        ("scikit-learn", "1.3.0"),
        ("pydantic", "2.0.0"),
    ]

    # Development dependencies
    DEV_DEPENDENCIES = [
        ("pytest", "7.0.0"),
        ("pytest-cov", "4.0.0"),
        ("hypothesis", "6.0.0"),
        ("ruff", "0.1.0"),
    ]

    # Optional dependencies by group
    OPTIONAL_GROUPS = {
        "news": [
            ("playwright", "1.40.0"),
            ("aiohttp", "3.9.0"),
        ],
        "bloomberg": [
            ("blpapi", None),  # No minimum version
        ],
    }

    def __init__(self, check_dev: bool = False, check_optional: list[str] | None = None):
        self.check_dev = check_dev
        self.check_optional = check_optional or []
        self.report = ValidationReport()

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        self._check_python_version()
        self._check_core_dependencies()

        if self.check_dev:
            self._check_dev_dependencies()

        for group in self.check_optional:
            if group in self.OPTIONAL_GROUPS:
                self._check_optional_group(group)

        self._check_environment_variables()
        self._check_directory_structure()

        return self.report

    def _check_python_version(self) -> None:
        """Validate Python version meets requirements."""
        major, minor, micro = sys.version_info[:3]
        required = f"{self.REQUIRED_PYTHON_MAJOR}.{self.REQUIRED_PYTHON_MINOR}"
        actual = f"{major}.{minor}.{micro}"

        if major == self.REQUIRED_PYTHON_MAJOR and minor >= self.REQUIRED_PYTHON_MINOR:
            self.report.results.append(CheckResult(
                name="python_version",
                status=CheckStatus.PASSED,
                severity=CheckSeverity.CRITICAL,
                message=f"Python {actual} meets requirement >={required}",
                details={"required": required, "actual": actual}
            ))
        else:
            self.report.results.append(CheckResult(
                name="python_version",
                status=CheckStatus.FAILED,
                severity=CheckSeverity.CRITICAL,
                message=f"Python {actual} does not meet requirement >={required}",
                details={"required": required, "actual": actual}
            ))

    def _check_dependency(
        self,
        name: str,
        min_version: str | None,
        severity: CheckSeverity
    ) -> CheckResult:
        """Check if a dependency is installed with minimum version."""
        spec = importlib.util.find_spec(name)

        if spec is None:
            return CheckResult(
                name=f"dependency_{name}",
                status=CheckStatus.FAILED,
                severity=severity,
                message=f"Package '{name}' is not installed",
                details={"package": name, "required_version": min_version}
            )

        # Try to get version
        try:
            module = importlib.import_module(name)
            actual_version = getattr(module, "__version__", "unknown")
        except Exception:
            actual_version = "unknown"

        if min_version and actual_version != "unknown":
            if not self._version_satisfies(actual_version, min_version):
                return CheckResult(
                    name=f"dependency_{name}",
                    status=CheckStatus.FAILED,
                    severity=severity,
                    message=f"Package '{name}' version {actual_version} < {min_version}",
                    details={
                        "package": name,
                        "required_version": min_version,
                        "actual_version": actual_version
                    }
                )

        return CheckResult(
            name=f"dependency_{name}",
            status=CheckStatus.PASSED,
            severity=severity,
            message=f"Package '{name}' {actual_version} installed",
            details={
                "package": name,
                "required_version": min_version,
                "actual_version": actual_version
            }
        )

    def _version_satisfies(self, actual: str, required: str) -> bool:
        """Check if actual version satisfies required minimum version."""
        try:
            actual_parts = [int(x) for x in actual.split(".")[:3]]
            required_parts = [int(x) for x in required.split(".")[:3]]

            # Pad to same length
            while len(actual_parts) < 3:
                actual_parts.append(0)
            while len(required_parts) < 3:
                required_parts.append(0)

            return actual_parts >= required_parts
        except (ValueError, AttributeError):
            return True  # Can't parse, assume OK

    def _check_core_dependencies(self) -> None:
        """Check all core dependencies."""
        for name, min_version in self.CORE_DEPENDENCIES:
            result = self._check_dependency(name, min_version, CheckSeverity.CRITICAL)
            self.report.results.append(result)

    def _check_dev_dependencies(self) -> None:
        """Check development dependencies."""
        for name, min_version in self.DEV_DEPENDENCIES:
            result = self._check_dependency(name, min_version, CheckSeverity.WARNING)
            self.report.results.append(result)

    def _check_optional_group(self, group: str) -> None:
        """Check optional dependency group."""
        for name, min_version in self.OPTIONAL_GROUPS.get(group, []):
            result = self._check_dependency(name, min_version, CheckSeverity.WARNING)
            self.report.results.append(result)

    def _check_environment_variables(self) -> None:
        """Check required environment variables."""
        # These are optional but we'll note their status
        optional_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "IB_GATEWAY_HOST",
            "IB_GATEWAY_PORT",
        ]

        for var in optional_vars:
            value = os.environ.get(var)
            if value:
                self.report.results.append(CheckResult(
                    name=f"env_{var}",
                    status=CheckStatus.PASSED,
                    severity=CheckSeverity.INFO,
                    message=f"Environment variable {var} is set",
                    details={"variable": var, "set": True}
                ))
            else:
                self.report.results.append(CheckResult(
                    name=f"env_{var}",
                    status=CheckStatus.SKIPPED,
                    severity=CheckSeverity.INFO,
                    message=f"Environment variable {var} is not set (optional)",
                    details={"variable": var, "set": False}
                ))

    def _check_directory_structure(self) -> None:
        """Check expected directory structure exists."""
        project_root = Path(__file__).parent.parent
        required_dirs = [
            "engine",
            "advisors",
            "tests",
            "data",
        ]

        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if dir_path.is_dir():
                self.report.results.append(CheckResult(
                    name=f"directory_{dir_name}",
                    status=CheckStatus.PASSED,
                    severity=CheckSeverity.CRITICAL,
                    message=f"Directory '{dir_name}' exists",
                    details={"path": str(dir_path)}
                ))
            else:
                self.report.results.append(CheckResult(
                    name=f"directory_{dir_name}",
                    status=CheckStatus.FAILED,
                    severity=CheckSeverity.CRITICAL,
                    message=f"Directory '{dir_name}' not found",
                    details={"path": str(dir_path)}
                ))


def print_report(report: ValidationReport, verbose: bool = False) -> None:
    """Print validation report to stdout."""
    summary = report.summary()

    print("\n" + "=" * 60)
    print("ENVIRONMENT VALIDATION REPORT")
    print("=" * 60)

    # Print summary
    print(f"\nTotal checks: {summary['total']}")
    print(f"  Passed:  {summary['passed']}")
    print(f"  Failed:  {summary['failed']}")
    print(f"  Skipped: {summary['skipped']}")

    # Print failures
    failures = [r for r in report.results if r.status == CheckStatus.FAILED]
    if failures:
        print("\n" + "-" * 40)
        print("FAILURES:")
        for result in failures:
            severity_icon = "🔴" if result.severity == CheckSeverity.CRITICAL else "🟡"
            print(f"  {severity_icon} [{result.severity.value}] {result.name}: {result.message}")

    # Print all results in verbose mode
    if verbose:
        print("\n" + "-" * 40)
        print("ALL CHECKS:")
        for result in report.results:
            status_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}[result.status.value]
            print(f"  {status_icon} {result.name}: {result.message}")

    # Final status
    print("\n" + "=" * 60)
    if report.has_critical_failures:
        print("❌ VALIDATION FAILED (critical errors)")
    elif report.has_warnings:
        print("⚠️  VALIDATION PASSED WITH WARNINGS")
    else:
        print("✅ VALIDATION PASSED")
    print("=" * 60 + "\n")


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Smart Wheel Engine environment")
    parser.add_argument("--dev", action="store_true", help="Check development dependencies")
    parser.add_argument("--optional", nargs="*", default=[],
                       help="Optional dependency groups to check (news, bloomberg)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    validator = EnvironmentValidator(
        check_dev=args.dev,
        check_optional=args.optional
    )

    report = validator.validate_all()

    if args.json:
        import json
        output = {
            "summary": report.summary(),
            "has_critical_failures": report.has_critical_failures,
            "has_warnings": report.has_warnings,
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "severity": r.severity.value,
                    "message": r.message,
                    "details": r.details
                }
                for r in report.results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Return exit code
    if report.has_critical_failures:
        return 1
    elif report.has_warnings:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
