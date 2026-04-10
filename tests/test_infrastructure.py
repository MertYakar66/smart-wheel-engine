"""
Tests for Infrastructure Components

Tests for:
- Environment validation
- Quant benchmarks
- Health checks
- SLO tracking
- Advisor scorecard
- Browser robustness
- Web vitals
"""

from datetime import datetime, timedelta

import pytest

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


class TestSLOTracking:
    """Tests for SLO tracking."""

    def test_slo_tracker_initialization(self):
        """SLOTracker should initialize correctly."""
        from news_pipeline.slo import SLOTracker

        tracker = SLOTracker()
        assert tracker is not None

    def test_record_request(self):
        """Should be able to record requests."""
        from news_pipeline.slo import PipelineStage, SLOTracker

        tracker = SLOTracker()
        tracker.record_request(PipelineStage.DISCOVERY, 500.0, True)

        metrics = tracker.get_metrics(PipelineStage.DISCOVERY)
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1

    def test_latency_slo_check(self):
        """Latency SLO checks should work correctly."""
        from news_pipeline.slo import LatencySLO, PipelineStage, SLOStatus

        slo = LatencySLO(
            stage=PipelineStage.DISCOVERY,
            p50_target_ms=500,
            p95_target_ms=2000,
            p99_target_ms=5000,
            max_allowed_ms=10000,
        )

        # Good latencies
        status = slo.check([100, 200, 300, 400, 500])
        assert status == SLOStatus.MET

        # Poor latencies
        status = slo.check([5000, 6000, 7000, 8000, 15000])
        assert status == SLOStatus.VIOLATED

    def test_error_budget_status(self):
        """Error budget tracking should work correctly."""
        from news_pipeline.slo import PipelineStage, SLOTracker

        tracker = SLOTracker()

        # Record some successes and failures
        for _ in range(95):
            tracker.record_request(PipelineStage.DISCOVERY, 500.0, True)
        for _ in range(5):
            tracker.record_request(PipelineStage.DISCOVERY, 500.0, False)

        budget_status = tracker.get_error_budget_status("daily")
        assert "consumed_percent" in budget_status
        assert "status" in budget_status


# =============================================================================
# ADVISOR SCORECARD TESTS
# =============================================================================


class TestAdvisorScorecard:
    """Tests for advisor scorecard."""

    def test_scorecard_initialization(self):
        """AdvisorScorecard should initialize correctly."""
        from advisors.scorecard import AdvisorScorecard

        scorecard = AdvisorScorecard()
        assert scorecard is not None
        assert len(scorecard.predictions) == 0
        assert len(scorecard.outcomes) == 0

    def test_record_prediction(self):
        """Should be able to record predictions."""
        from advisors.scorecard import (
            AdvisorPrediction,
            AdvisorScorecard,
            ConfidenceLevel,
            JudgmentType,
        )

        scorecard = AdvisorScorecard()

        prediction = AdvisorPrediction(
            prediction_id="test_1",
            advisor_name="buffett",
            judgment=JudgmentType.APPROVE,
            confidence=ConfidenceLevel.HIGH,
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="cash_secured_put",
            strike=150.0,
            expiration_date="2024-12-20",
            premium=3.50,
        )

        scorecard.record_prediction(prediction)
        assert len(scorecard.predictions) == 1
        assert "test_1" in scorecard.predictions

    def test_record_outcome(self):
        """Should be able to record outcomes."""
        from advisors.scorecard import (
            AdvisorPrediction,
            AdvisorScorecard,
            ConfidenceLevel,
            JudgmentType,
            OutcomeType,
            TradeOutcome,
        )

        scorecard = AdvisorScorecard()

        prediction = AdvisorPrediction(
            prediction_id="test_1",
            advisor_name="buffett",
            judgment=JudgmentType.APPROVE,
            confidence=ConfidenceLevel.HIGH,
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="cash_secured_put",
            strike=150.0,
            expiration_date="2024-12-20",
            premium=3.50,
        )
        scorecard.record_prediction(prediction)

        outcome = TradeOutcome(
            outcome_id="outcome_1",
            prediction_id="test_1",
            outcome=OutcomeType.WIN,
            pnl=350.0,
            pnl_percent=2.5,
            close_timestamp=datetime.now(),
            hold_days=30,
        )
        scorecard.record_outcome(outcome)

        assert len(scorecard.outcomes) == 1

    def test_metrics_calculation(self):
        """Metrics should be calculated correctly."""
        from advisors.scorecard import AdvisorMetrics

        metrics = AdvisorMetrics(
            advisor_name="test",
            period_start=datetime.now() - timedelta(days=30),
            period_end=datetime.now(),
            total_predictions=100,
            total_outcomes=80,
            approvals=50,
            approval_wins=40,
            approval_losses=10,
            rejections=50,
            rejection_would_have_won=5,
            rejection_would_have_lost=25,
        )

        # Accuracy = (40 + 25) / 80 = 0.8125
        assert metrics.accuracy == pytest.approx(0.8125)

        # Precision = 40 / 50 = 0.8
        assert metrics.precision == pytest.approx(0.8)

        # Recall = 40 / (40 + 5) = 0.888...
        assert metrics.recall == pytest.approx(40 / 45)


# =============================================================================
# BROWSER ROBUSTNESS TESTS
# =============================================================================


class TestBrowserRobustness:
    """Tests for browser agent robustness tracking."""

    def test_robustness_tracker_initialization(self):
        """RobustnessTracker should initialize correctly."""
        from news_pipeline.browser_agents.robustness import RobustnessTracker

        tracker = RobustnessTracker()
        assert tracker is not None
        assert "claude" in tracker.agents
        assert "chatgpt" in tracker.agents
        assert "gemini" in tracker.agents

    def test_record_selector_attempt(self):
        """Should be able to record selector attempts."""
        from news_pipeline.browser_agents.robustness import RobustnessTracker

        tracker = RobustnessTracker()
        tracker.record_selector_attempt(
            agent_name="claude",
            purpose="prompt_input",
            success=True,
            locate_ms=50.0,
        )

        agent = tracker.get_agent_metrics("claude")
        assert agent is not None
        selector = agent.selectors.get("prompt_input")
        assert selector is not None
        assert selector.total_attempts == 1
        assert selector.successful_attempts == 1

    def test_selector_status_calculation(self):
        """Selector status should be calculated correctly."""
        from news_pipeline.browser_agents.robustness import (
            SelectorMetrics,
            SelectorStatus,
        )

        selector = SelectorMetrics(
            selector="div.test",
            element_type="div",
            purpose="test",
        )

        # Record mostly successful attempts
        for _ in range(95):
            selector.record_attempt(True, 50.0)
        for _ in range(5):
            selector.record_attempt(False, 100.0)

        assert selector.success_rate == 0.95
        assert selector.status == SelectorStatus.ACTIVE

    def test_dom_drift_detection(self):
        """DOM drift should be detected."""
        from news_pipeline.browser_agents.robustness import SelectorMetrics

        selector = SelectorMetrics(
            selector="div.test",
            element_type="div",
            purpose="test",
        )

        # First attempt
        selector.record_attempt(True, 50.0, dom_hash="hash1")
        assert selector.dom_changes_detected == 0

        # Same DOM
        selector.record_attempt(True, 50.0, dom_hash="hash1")
        assert selector.dom_changes_detected == 0

        # DOM changed
        selector.record_attempt(True, 50.0, dom_hash="hash2")
        assert selector.dom_changes_detected == 1


# =============================================================================
# WEB VITALS TESTS
# =============================================================================


class TestWebVitals:
    """Tests for web vitals tracking."""

    def test_vitals_tracker_initialization(self):
        """WebVitalsTracker should initialize correctly."""
        from dashboard.web_vitals import WebVitalsTracker

        tracker = WebVitalsTracker()
        assert tracker is not None

    def test_record_vital(self):
        """Should be able to record vitals."""
        from dashboard.web_vitals import WebVitalsTracker

        tracker = WebVitalsTracker()
        tracker.record("LCP", 2000.0, "dashboard")
        tracker.record("FID", 50.0, "dashboard")
        tracker.record("CLS", 0.05, "dashboard")

        metrics = tracker.get_page_metrics("dashboard")
        assert metrics.total_measurements == 3

    def test_vital_rating(self):
        """Vital ratings should be calculated correctly."""
        from datetime import datetime

        from dashboard.web_vitals import VitalMeasurement, VitalRating

        # Good LCP
        good_lcp = VitalMeasurement(
            metric="LCP",
            value=2000,
            timestamp=datetime.now(),
            page="test",
        )
        assert good_lcp.rating == VitalRating.GOOD

        # Poor LCP
        poor_lcp = VitalMeasurement(
            metric="LCP",
            value=5000,
            timestamp=datetime.now(),
            page="test",
        )
        assert poor_lcp.rating == VitalRating.POOR

    def test_page_metrics_aggregation(self):
        """Page metrics should aggregate correctly."""
        from dashboard.web_vitals import WebVitalsTracker

        tracker = WebVitalsTracker()

        # Record multiple LCP measurements
        for lcp in [1000, 1500, 2000, 2500, 3000]:
            tracker.record("LCP", lcp, "dashboard")

        metrics = tracker.get_page_metrics("dashboard")
        assert metrics.lcp_p75 is not None
        # p75 of [1000, 1500, 2000, 2500, 3000] should be ~2500
        assert 2000 <= metrics.lcp_p75 <= 3000

    def test_budget_check(self):
        """Budget checks should work correctly."""
        from dashboard.web_vitals import WebVitalsTracker

        tracker = WebVitalsTracker()

        # Record good vitals
        tracker.record("LCP", 1500, "dashboard")
        tracker.record("FID", 30, "dashboard")
        tracker.record("CLS", 0.03, "dashboard")

        result = tracker.check_all_budgets()
        assert "pages" in result
        assert "all_budgets_met" in result

    def test_generate_report(self):
        """Should generate performance reports."""
        from dashboard.web_vitals import WebVitalsTracker

        tracker = WebVitalsTracker()
        tracker.record("LCP", 1500, "dashboard")
        tracker.record("LCP", 5000, "news")  # Poor performance

        report = tracker.generate_report()
        assert "generated_at" in report
        assert "summary" in report
        assert "recommendations" in report


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


class TestSLOExtended:
    """Extended SLO tests."""

    def test_slo_compliance_check(self):
        """SLO compliance check should work correctly."""
        from news_pipeline.slo import PipelineStage, SLOTracker

        tracker = SLOTracker()

        # Record requests with various latencies
        for latency in [100, 200, 300, 400, 500]:
            tracker.record_request(PipelineStage.DISCOVERY, latency, True)

        report = tracker.check_slo_compliance(PipelineStage.DISCOVERY)
        assert "stage" in report
        assert "status" in report
        assert "latency" in report

    def test_generate_slo_report(self):
        """Should generate comprehensive SLO report."""
        from news_pipeline.slo import PipelineStage, SLOTracker

        tracker = SLOTracker()
        tracker.record_request(PipelineStage.DISCOVERY, 500.0, True)
        tracker.record_request(PipelineStage.VERIFICATION, 3000.0, True)

        report = tracker.generate_slo_report()
        assert "generated_at" in report
        assert "stages" in report
        assert "error_budgets" in report
        assert "overall_status" in report

    def test_slo_metrics_properties(self):
        """SLOMetrics properties should calculate correctly."""
        from news_pipeline.slo import PipelineStage, SLOMetrics

        metrics = SLOMetrics(
            period_start=datetime.now() - timedelta(hours=1),
            period_end=datetime.now(),
            stage=PipelineStage.DISCOVERY,
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            latencies_ms=[100, 200, 300, 400, 500],
        )

        assert metrics.error_rate == 0.05
        assert metrics.success_rate == 0.95
        assert metrics.p50_latency_ms is not None
        assert metrics.p95_latency_ms is not None


# =============================================================================
# ADDITIONAL ADVISOR SCORECARD TESTS
# =============================================================================


class TestAdvisorScorecardExtended:
    """Extended advisor scorecard tests."""

    def test_generate_report(self):
        """Should generate comprehensive report."""
        from advisors.scorecard import (
            AdvisorPrediction,
            AdvisorScorecard,
            ConfidenceLevel,
            JudgmentType,
            OutcomeType,
            TradeOutcome,
        )

        scorecard = AdvisorScorecard()

        # Add prediction and outcome
        prediction = AdvisorPrediction(
            prediction_id="test_1",
            advisor_name="buffett",
            judgment=JudgmentType.APPROVE,
            confidence=ConfidenceLevel.HIGH,
            timestamp=datetime.now(),
            ticker="AAPL",
            strategy="cash_secured_put",
            strike=150.0,
            expiration_date="2024-12-20",
            premium=3.50,
        )
        scorecard.record_prediction(prediction)

        outcome = TradeOutcome(
            outcome_id="outcome_1",
            prediction_id="test_1",
            outcome=OutcomeType.WIN,
            pnl=350.0,
            pnl_percent=2.5,
            close_timestamp=datetime.now(),
            hold_days=30,
        )
        scorecard.record_outcome(outcome)

        report = scorecard.generate_report()
        assert "generated_at" in report
        assert "summary" in report
        assert "advisor_metrics" in report

    def test_calibration_curve(self):
        """Should calculate calibration curve."""
        from advisors.scorecard import (
            AdvisorPrediction,
            AdvisorScorecard,
            ConfidenceLevel,
            JudgmentType,
            OutcomeType,
            TradeOutcome,
        )

        scorecard = AdvisorScorecard()

        # Add predictions with outcomes
        for i in range(10):
            pred = AdvisorPrediction(
                prediction_id=f"pred_{i}",
                advisor_name="test_advisor",
                judgment=JudgmentType.APPROVE,
                confidence=ConfidenceLevel.HIGH,
                timestamp=datetime.now(),
                ticker="AAPL",
                strategy="csp",
                strike=150.0,
                expiration_date="2024-12-20",
                premium=3.50,
            )
            scorecard.record_prediction(pred)

            outcome = TradeOutcome(
                outcome_id=f"out_{i}",
                prediction_id=f"pred_{i}",
                outcome=OutcomeType.WIN if i < 8 else OutcomeType.LOSS,
                pnl=350.0 if i < 8 else -150.0,
                pnl_percent=2.5 if i < 8 else -1.0,
                close_timestamp=datetime.now(),
                hold_days=30,
            )
            scorecard.record_outcome(outcome)

        buckets = scorecard.get_calibration_curve("test_advisor")
        assert len(buckets) > 0

    def test_metrics_edge_cases(self):
        """Test metrics edge cases."""
        from advisors.scorecard import AdvisorMetrics

        # Zero values
        metrics = AdvisorMetrics(
            advisor_name="test",
            period_start=datetime.now(),
            period_end=datetime.now(),
            total_predictions=0,
            total_outcomes=0,
        )

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0


# =============================================================================
# ADDITIONAL BROWSER ROBUSTNESS TESTS
# =============================================================================


class TestBrowserRobustnessExtended:
    """Extended browser robustness tests."""

    def test_record_session(self):
        """Should track session success/failure."""
        from news_pipeline.browser_agents.robustness import RobustnessTracker

        tracker = RobustnessTracker()

        tracker.record_session("claude", True)
        tracker.record_session("claude", True)
        tracker.record_session("claude", False)

        agent = tracker.get_agent_metrics("claude")
        assert agent.total_sessions == 3
        assert agent.successful_sessions == 2
        assert agent.failed_sessions == 1
        assert agent.session_success_rate == pytest.approx(2 / 3)

    def test_suggest_fallbacks(self):
        """Should suggest selector fallbacks."""
        from news_pipeline.browser_agents.robustness import RobustnessTracker

        tracker = RobustnessTracker()

        # Simulate failing selector
        for _ in range(10):
            tracker.record_selector_attempt("claude", "prompt_input", False, 100.0)

        fallbacks = tracker.suggest_fallbacks("claude", "prompt_input")
        assert len(fallbacks) > 0

    def test_generate_robustness_report(self):
        """Should generate robustness report."""
        from news_pipeline.browser_agents.robustness import RobustnessTracker

        tracker = RobustnessTracker()

        # Record some activity
        tracker.record_selector_attempt("claude", "prompt_input", True, 50.0)
        tracker.record_session("claude", True)

        report = tracker.generate_robustness_report()
        assert "generated_at" in report
        assert "summary" in report
        assert "agents" in report

    def test_selector_degraded_status(self):
        """Selector should show degraded status."""
        from news_pipeline.browser_agents.robustness import (
            SelectorMetrics,
            SelectorStatus,
        )

        selector = SelectorMetrics(
            selector="div.test",
            element_type="div",
            purpose="test",
        )

        # 85% success rate = degraded
        for _ in range(85):
            selector.record_attempt(True, 50.0)
        for _ in range(15):
            selector.record_attempt(False, 100.0)

        assert selector.status == SelectorStatus.DEGRADED

    def test_selector_failing_status(self):
        """Selector should show failing status."""
        from news_pipeline.browser_agents.robustness import (
            SelectorMetrics,
            SelectorStatus,
        )

        selector = SelectorMetrics(
            selector="div.test",
            element_type="div",
            purpose="test",
        )

        # 50% success rate = failing
        for _ in range(50):
            selector.record_attempt(True, 50.0)
        for _ in range(50):
            selector.record_attempt(False, 100.0)

        assert selector.status == SelectorStatus.FAILING


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
