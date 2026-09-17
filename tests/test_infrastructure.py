"""
Tests for Infrastructure Components

Tests for:
- Environment validation
- Quant benchmarks
- Health checks
- SLO tracking
- Browser robustness
- Web vitals
"""


# =============================================================================
# ENVIRONMENT VALIDATION TESTS
# =============================================================================


class TestEnvironmentValidation:
    """Tests for environment validation script."""

    def test_validator_initialization(self):
        """EnvironmentValidator should initialize correctly."""
        from scripts.validate_environment import EnvironmentValidator

        validator = EnvironmentValidator()
        assert validator is not None
        assert validator.check_dev is False
        assert validator.check_optional == []

    def test_python_version_check(self):
        """Python version check should pass for 3.11+."""
        from scripts.validate_environment import CheckStatus, EnvironmentValidator

        validator = EnvironmentValidator()
        report = validator.validate_all()

        # Find python version result
        python_check = next((r for r in report.results if r.name == "python_version"), None)

        assert python_check is not None
        # Project requires Python 3.11+, so this should always pass
        assert python_check.status == CheckStatus.PASSED

    def test_validation_report_structure(self):
        """Validation report should have correct structure."""
        from scripts.validate_environment import EnvironmentValidator

        validator = EnvironmentValidator()
        report = validator.validate_all()

        assert hasattr(report, "results")
        assert hasattr(report, "has_critical_failures")
        assert hasattr(report, "has_warnings")
        assert hasattr(report, "summary")

        summary = report.summary()
        assert "total" in summary
        assert "passed" in summary
        assert "failed" in summary


# =============================================================================
# QUANT BENCHMARKS TESTS
# =============================================================================


class TestQuantBenchmarks:
    """Tests for quant benchmark registry."""

    def test_benchmark_registry_exists(self):
        """All benchmark categories should be defined."""
        from tests.quant_benchmarks import (
            ALL_BENCHMARKS,
            AMERICAN_BENCHMARKS,
            BLACKSCHOLES_BENCHMARKS,
            GREEKS_BENCHMARKS,
            MONTE_CARLO_BENCHMARKS,
        )

        assert len(BLACKSCHOLES_BENCHMARKS) > 0
        assert len(GREEKS_BENCHMARKS) > 0
        assert len(AMERICAN_BENCHMARKS) > 0
        assert len(MONTE_CARLO_BENCHMARKS) > 0
        assert len(ALL_BENCHMARKS) > 0

    def test_tolerance_check_absolute(self):
        """Absolute tolerance checks should work correctly."""
        from tests.quant_benchmarks import BenchmarkTolerance, ToleranceType, check_tolerance

        benchmark = BenchmarkTolerance(
            name="Test",
            tolerance_type=ToleranceType.ABSOLUTE,
            value=0.01,
            reference="Test",
            description="Test",
        )

        # Within tolerance
        passed, msg = check_tolerance(4.76, 4.7594, benchmark)
        assert passed is True

        # Outside tolerance
        passed, msg = check_tolerance(5.0, 4.7594, benchmark)
        assert passed is False

    def test_tolerance_check_bounds(self):
        """Bounds tolerance checks should work correctly."""
        from tests.quant_benchmarks import BenchmarkTolerance, ToleranceType, check_tolerance

        benchmark = BenchmarkTolerance(
            name="Test",
            tolerance_type=ToleranceType.BOUNDS,
            value=(0.0, 1.0),
            reference="Test",
            description="Test",
        )

        # Within bounds
        passed, msg = check_tolerance(0.5, None, benchmark)
        assert passed is True

        # Outside bounds
        passed, msg = check_tolerance(1.5, None, benchmark)
        assert passed is False

    def test_release_gate_benchmarks(self):
        """Release gate benchmarks should be properly flagged."""
        from tests.quant_benchmarks import ALL_BENCHMARKS, RELEASE_GATE_BENCHMARKS

        # All release gates should be in ALL_BENCHMARKS
        for name in RELEASE_GATE_BENCHMARKS:
            assert name in ALL_BENCHMARKS

        # Should have at least some release gates
        assert len(RELEASE_GATE_BENCHMARKS) > 0


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestHealthChecks:
    """Tests for health check infrastructure."""

    def test_health_checker_initialization(self):
        """HealthChecker should initialize correctly."""
        from utils.health import HealthChecker

        checker = HealthChecker(version="1.0.0")
        assert checker.version == "1.0.0"

    def test_register_check(self):
        """Should be able to register health checks."""
        from utils.health import CheckType, HealthChecker, HealthStatus

        checker = HealthChecker()

        def dummy_check():
            return (HealthStatus.HEALTHY, "OK", {})

        checker.register_check("test_check", dummy_check, CheckType.LIVENESS)

        status = checker.get_status()
        assert any(c.name == "test_check" for c in status.checks)

    def test_aggregate_status_healthy(self):
        """Aggregate status should be healthy when all checks pass."""
        from utils.health import HealthChecker, HealthStatus

        checker = HealthChecker()

        def healthy_check():
            return (HealthStatus.HEALTHY, "OK", {})

        checker.register_check("check1", healthy_check)
        checker.register_check("check2", healthy_check)

        status = checker.get_status()
        assert status.status == HealthStatus.HEALTHY
        assert status.is_healthy is True

    def test_aggregate_status_degraded(self):
        """Aggregate status should be degraded when dependency fails."""
        from utils.health import CheckType, HealthChecker, HealthStatus

        checker = HealthChecker()

        def healthy_check():
            return (HealthStatus.HEALTHY, "OK", {})

        def unhealthy_check():
            return (HealthStatus.UNHEALTHY, "Failed", {})

        checker.register_check("liveness", healthy_check, CheckType.LIVENESS)
        checker.register_check("dependency", unhealthy_check, CheckType.DEPENDENCY)

        status = checker.get_status()
        assert status.status == HealthStatus.DEGRADED

    def test_disk_space_check(self):
        """Disk space check should return valid result."""
        from utils.health import HealthStatus, check_disk_space

        status, message, details = check_disk_space("/")

        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "free_gb" in details


# =============================================================================
# SLO TRACKING TESTS
# =============================================================================


# =============================================================================
# BROWSER ROBUSTNESS TESTS
# =============================================================================


# =============================================================================
# ADDITIONAL QUANT BENCHMARK TESTS
# =============================================================================


class TestQuantBenchmarksExtended:
    """Extended tests for quant benchmark tolerances."""

    def test_tolerance_relative(self):
        """Relative tolerance checks should work correctly."""
        from tests.quant_benchmarks import BenchmarkTolerance, ToleranceType, check_tolerance

        benchmark = BenchmarkTolerance(
            name="Test",
            tolerance_type=ToleranceType.RELATIVE,
            value=0.05,  # 5% tolerance
            reference="Test",
            description="Test",
        )

        # Within 5% relative error
        passed, _ = check_tolerance(1.02, 1.0, benchmark)
        assert passed is True

        # Outside 5% relative error
        passed, _ = check_tolerance(1.10, 1.0, benchmark)
        assert passed is False

    def test_tolerance_non_negative(self):
        """Non-negative tolerance checks should work correctly."""
        from tests.quant_benchmarks import BenchmarkTolerance, ToleranceType, check_tolerance

        benchmark = BenchmarkTolerance(
            name="Test",
            tolerance_type=ToleranceType.NON_NEGATIVE,
            value=0.0,
            reference="Test",
            description="Test",
        )

        passed, _ = check_tolerance(0.5, None, benchmark)
        assert passed is True

        passed, _ = check_tolerance(-0.1, None, benchmark)
        assert passed is False

    def test_tolerance_probability(self):
        """Probability tolerance checks should work correctly."""
        from tests.quant_benchmarks import BenchmarkTolerance, ToleranceType, check_tolerance

        benchmark = BenchmarkTolerance(
            name="Test",
            tolerance_type=ToleranceType.PROBABILITY,
            value=(0.0, 1.0),
            reference="Test",
            description="Test",
        )

        passed, _ = check_tolerance(0.5, None, benchmark)
        assert passed is True

        passed, _ = check_tolerance(1.5, None, benchmark)
        assert passed is False

    def test_benchmark_helpers(self):
        """Test benchmark helper functions."""
        from tests.quant_benchmarks import (
            generate_benchmark_report,
            get_benchmark,
            list_benchmarks,
        )

        # Get specific benchmark
        benchmark = get_benchmark("put_call_parity")
        assert benchmark is not None
        assert benchmark.name == "Put-Call Parity"

        # List benchmarks by category
        bs_benchmarks = list_benchmarks("blackscholes")
        assert len(bs_benchmarks) > 0

        # Generate report
        report = generate_benchmark_report()
        assert "total_benchmarks" in report
        assert "release_gates" in report


# =============================================================================
# ADDITIONAL HEALTH CHECK TESTS
# =============================================================================


class TestHealthChecksExtended:
    """Extended health check tests."""

    def test_unregister_check(self):
        """Should be able to unregister health checks."""
        from utils.health import HealthChecker, HealthStatus

        checker = HealthChecker()

        def dummy_check():
            return (HealthStatus.HEALTHY, "OK", {})

        checker.register_check("temp_check", dummy_check)
        assert any(c.name == "temp_check" for c in checker.get_status().checks)

        checker.unregister_check("temp_check")
        assert not any(c.name == "temp_check" for c in checker.get_status().checks)

    def test_cache_behavior(self):
        """Cache should return cached results within TTL."""
        from utils.health import HealthChecker, HealthStatus

        call_count = [0]

        def counting_check():
            call_count[0] += 1
            return (HealthStatus.HEALTHY, f"Call {call_count[0]}", {})

        checker = HealthChecker(cache_ttl_seconds=60.0)
        checker.register_check("counting", counting_check)

        # First call
        checker.get_status()
        assert call_count[0] == 1

        # Second call should use cache
        checker.get_status()
        assert call_count[0] == 1  # Still 1, cached

        # Force refresh should bypass cache
        checker.get_status(force_refresh=True)
        assert call_count[0] == 2

    def test_check_exception(self):
        """Checks that raise exceptions should be caught."""
        from utils.health import HealthChecker, HealthStatus

        checker = HealthChecker()

        def failing_check():
            raise ValueError("Test error")

        checker.register_check("failing", failing_check)
        status = checker.get_status()

        failing_result = next(c for c in status.checks if c.name == "failing")
        assert failing_result.status == HealthStatus.UNHEALTHY

    def test_system_health_properties(self):
        """SystemHealth properties should work correctly."""
        from utils.health import HealthChecker, HealthStatus

        checker = HealthChecker(version="2.0.0")

        def healthy_check():
            return (HealthStatus.HEALTHY, "OK", {})

        checker.register_check("check", healthy_check)
        status = checker.get_status()

        assert status.is_healthy is True
        assert status.is_ready is True
        assert status.version == "2.0.0"
        assert status.uptime_seconds >= 0  # May be 0 on fast machines

        # Test to_dict
        status_dict = status.to_dict()
        assert "status" in status_dict
        assert "checks" in status_dict


# =============================================================================
# ADDITIONAL SLO TESTS
# =============================================================================


# =============================================================================
# ADDITIONAL BROWSER ROBUSTNESS TESTS
# =============================================================================


# =============================================================================
# ENVIRONMENT VALIDATION EXTENDED TESTS
# =============================================================================


class TestEnvironmentValidationExtended:
    """Extended environment validation tests."""

    def test_dev_dependencies_check(self):
        """Should check dev dependencies when requested."""
        from scripts.validate_environment import EnvironmentValidator

        validator = EnvironmentValidator(check_dev=True)
        report = validator.validate_all()

        # Should have more results with dev checks
        assert report.summary()["total"] > 5

    def test_version_satisfies(self):
        """Version comparison should work correctly."""
        from scripts.validate_environment import EnvironmentValidator

        validator = EnvironmentValidator()

        assert validator._version_satisfies("2.0.0", "1.0.0") is True
        assert validator._version_satisfies("1.0.0", "2.0.0") is False
        assert validator._version_satisfies("1.5.0", "1.5.0") is True
        assert validator._version_satisfies("invalid", "1.0.0") is True  # Can't parse, assume OK
