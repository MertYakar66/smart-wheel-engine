"""
Integration tests for the data engineering pipeline.

Tests the full flow:
    Raw Data → Validation → Features → Store → Quality Checks
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.feature_store import FeatureStore
from data.observability import MetricsCollector, Tracer, metrics, trace
from data.quality import DataQualityFramework


try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")
class TestFeatureStore:  # pragma: no cover when pyarrow absent
    """Tests for the Feature Store."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a temporary feature store."""
        return FeatureStore(base_path=tmp_path / "features")

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        return pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.randint(1000000, 10000000, 100),
            }
        )

    def test_write_and_read_features(self, temp_store, sample_ohlcv):
        """Test basic write and read operations."""
        # Write
        metadata = temp_store.write_features(
            category="ohlcv",
            ticker="AAPL",
            df=sample_ohlcv,
            source_files=["test.csv"],
            source_category="raw",
            transformation="load",
        )

        assert metadata.category == "ohlcv"
        assert metadata.ticker == "AAPL"
        assert metadata.row_count == 100
        assert metadata.version == 1

        # Read
        df = temp_store.read_features("ohlcv", "AAPL")
        assert df is not None
        assert len(df) == 100
        assert "date" in df.columns
        assert "close" in df.columns

    def test_point_in_time_filtering(self, temp_store, sample_ohlcv):
        """Test point-in-time data filtering."""
        temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)

        # Filter to first 50 days
        df = temp_store.read_features(
            "ohlcv",
            "AAPL",
            as_of="2024-02-23",  # ~35 business days
        )

        assert df is not None
        assert len(df) < 100

    def test_version_increment(self, temp_store, sample_ohlcv):
        """Test that version increments on update."""
        # Write v1
        meta1 = temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)
        assert meta1.version == 1

        # Write v2 with force
        sample_ohlcv["close"] = sample_ohlcv["close"] * 1.01
        meta2 = temp_store.write_features("ohlcv", "AAPL", sample_ohlcv, force=True)
        assert meta2.version == 2

    def test_skip_unchanged_data(self, temp_store, sample_ohlcv):
        """Test that unchanged data is skipped."""
        meta1 = temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)

        # Write same data again
        meta2 = temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)

        # Should be same version (skipped)
        assert meta2.version == meta1.version

    def test_list_features(self, temp_store, sample_ohlcv):
        """Test listing features."""
        temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)
        temp_store.write_features("ohlcv", "MSFT", sample_ohlcv)
        temp_store.write_features("volatility", "AAPL", sample_ohlcv)

        features = temp_store.list_features()
        assert len(features) == 3

        ohlcv_features = temp_store.list_features("ohlcv")
        assert len(ohlcv_features) == 2

    def test_delete_features(self, temp_store, sample_ohlcv):
        """Test deleting features."""
        temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)

        # Delete
        result = temp_store.delete_features("ohlcv", "AAPL")
        assert result is True

        # Should not exist
        df = temp_store.read_features("ohlcv", "AAPL")
        assert df is None

    def test_health_check(self, temp_store, sample_ohlcv):
        """Test health check."""
        temp_store.write_features("ohlcv", "AAPL", sample_ohlcv)

        health = temp_store.health_check()
        assert health["healthy"] is True
        assert len(health["issues"]) == 0


class TestDataQualityFramework:
    """Tests for the Data Quality Framework."""

    @pytest.fixture
    def dq(self):
        return DataQualityFramework()

    @pytest.fixture
    def valid_ohlcv(self):
        """Create valid OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        n = len(dates)
        opens = np.random.uniform(100, 110, n)
        highs = opens + np.random.uniform(0, 10, n)
        lows = opens - np.random.uniform(0, 10, n)
        closes = np.random.uniform(lows, highs)

        return pd.DataFrame(
            {
                "date": dates,
                "ticker": ["AAPL"] * n,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.random.randint(1000000, 10000000, n),
            }
        )

    @pytest.fixture
    def invalid_ohlcv(self):
        """Create OHLCV data with violations."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": ["MSFT"] * 100,
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(90, 95, 100),  # High < Open (violation)
                "low": np.random.uniform(95, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.randint(-1000, 10000000, 100),  # Negative volume
            }
        )

    def test_valid_data_passes(self, dq, valid_ohlcv):
        """Test that valid data passes validation."""
        result = dq.validate(valid_ohlcv, "ohlcv", "AAPL")

        assert result.valid is True
        assert result.row_count == 300

    def test_invalid_data_fails(self, dq, invalid_ohlcv):
        """Test that invalid data fails validation."""
        result = dq.validate(invalid_ohlcv, "ohlcv", "AAPL")

        assert result.valid is False
        assert result.critical_count > 0 or result.error_count > 0

    def test_missing_columns(self, dq):
        """Test detection of missing required columns."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "close": np.random.uniform(100, 110, 100),
                # Missing open, high, low, volume
            }
        )

        result = dq.validate(df, "ohlcv", "AAPL")

        assert result.valid is False
        assert any(i.message.startswith("Missing required") for i in result.issues)

    def test_null_threshold_violations(self, dq):
        """Test null threshold violations."""
        dates = pd.date_range("2024-01-01", periods=100, freq="B")
        df = pd.DataFrame(
            {
                "date": dates,
                "open": [100.0] * 100,
                "high": [110.0] * 100,
                "low": [90.0] * 100,
                "close": [None] * 50 + [100.0] * 50,  # 50% null
                "volume": [1000000] * 100,
            }
        )

        result = dq.validate(df, "ohlcv", "AAPL")

        # Should have warning about null close
        null_issues = [i for i in result.issues if "null" in i.message.lower()]
        assert len(null_issues) > 0

    def test_quality_report(self, dq, valid_ohlcv, invalid_ohlcv):
        """Test quality report generation."""
        dq.validate(valid_ohlcv, "ohlcv", "AAPL")
        dq.validate(invalid_ohlcv, "ohlcv", "MSFT")

        report = dq.quality_report()

        assert report["summary"]["total_validations"] == 2
        assert report["summary"]["valid"] >= 1


class TestMetrics:
    """Tests for metrics collection."""

    @pytest.fixture
    def collector(self):
        return MetricsCollector()

    def test_counter(self, collector):
        """Test counter metrics."""
        collector.increment("test.counter")
        collector.increment("test.counter", 5)

        assert collector.get_counter("test.counter") == 6

    def test_counter_with_tags(self, collector):
        """Test counter with tags."""
        collector.increment("test.counter", tags={"ticker": "AAPL"})
        collector.increment("test.counter", tags={"ticker": "MSFT"})

        assert collector.get_counter("test.counter", {"ticker": "AAPL"}) == 1
        assert collector.get_counter("test.counter", {"ticker": "MSFT"}) == 1

    def test_gauge(self, collector):
        """Test gauge metrics."""
        collector.gauge("test.gauge", 100)
        assert collector.get_gauge("test.gauge") == 100

        collector.gauge("test.gauge", 200)
        assert collector.get_gauge("test.gauge") == 200

    def test_histogram(self, collector):
        """Test histogram metrics."""
        for i in range(100):
            collector.histogram("test.histogram", i)

        stats = collector.get_histogram_stats("test.histogram")
        assert stats is not None
        assert stats["count"] == 100
        assert stats["mean"] == pytest.approx(49.5, rel=0.1)

    def test_timing(self, collector):
        """Test timing metrics."""
        collector.timing("test.timing", 100)
        collector.timing("test.timing", 200)
        collector.timing("test.timing", 300)

        stats = collector.get_timing_stats("test.timing")
        assert stats is not None
        assert stats["count"] == 3
        assert stats["mean_ms"] == pytest.approx(200, rel=0.1)

    def test_timer_context_manager(self, collector):
        """Test timer context manager."""
        import time

        with collector.timer("test.timer"):
            time.sleep(0.01)

        stats = collector.get_timing_stats("test.timer")
        assert stats is not None
        assert stats["mean_ms"] >= 10


class TestTracer:
    """Tests for distributed tracing."""

    @pytest.fixture
    def tracer(self):
        return Tracer()

    def test_basic_span(self, tracer):
        """Test basic span creation."""
        with tracer.span("test_operation") as span:
            span.set_attribute("ticker", "AAPL")
            span.add_event("started")

        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.attributes["ticker"] == "AAPL"

    def test_nested_spans(self, tracer):
        """Test nested span creation."""
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass

        assert child.parent_id == parent.span_id
        assert child.trace_id == parent.trace_id

    def test_span_error(self, tracer):
        """Test span error handling."""
        try:
            with tracer.span("failing_operation") as span:
                raise ValueError("Test error")
        except ValueError:
            pass

        assert span.status == "ERROR"
        assert span.error == "Test error"


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")
class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directories."""
        data_dir = tmp_path / "data"
        features_dir = tmp_path / "features"
        data_dir.mkdir()
        features_dir.mkdir()
        return tmp_path

    def test_full_pipeline_flow(self, temp_dir):
        """Test the full pipeline flow with mock data."""
        from data.feature_store import FeatureStore
        from data.quality import DataQualityFramework

        # Create components
        store = FeatureStore(base_path=temp_dir / "features")
        dq = DataQualityFramework()

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        n = len(dates)
        ohlcv = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(100, 110, n),
                "high": np.random.uniform(110, 120, n),
                "low": np.random.uniform(90, 100, n),
                "close": np.random.uniform(100, 110, n),
                "volume": np.random.randint(1000000, 10000000, n),
            }
        )

        # Step 1: Validate
        validation = dq.validate(ohlcv, "ohlcv", "AAPL")
        assert validation.row_count == n

        # Step 2: Compute features (simplified - just add returns)
        ohlcv["returns"] = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
        ohlcv["rv_21d"] = ohlcv["returns"].rolling(21).std() * np.sqrt(252)

        # Step 3: Store
        metadata = store.write_features(
            category="volatility",
            ticker="AAPL",
            df=ohlcv,
            source_category="ohlcv",
            transformation="rv_computation",
        )

        assert metadata.version == 1

        # Step 4: Read back
        features = store.read_features("volatility", "AAPL")
        assert features is not None
        assert "rv_21d" in features.columns

        # Step 5: Health checks
        store_health = store.health_check()
        dq.health_check()

        assert store_health["healthy"] is True

    def test_observability_integration(self):
        """Test observability integration."""

        # Record some metrics
        metrics.increment("pipeline.tasks", tags={"stage": "load"})
        metrics.timing("pipeline.duration", 100)

        # Create trace
        with trace.span("pipeline_run") as span:
            span.set_attribute("tickers", 10)

        # Verify metrics collected
        all_metrics = metrics.get_all_metrics()
        assert "counters" in all_metrics

        # Verify trace recorded
        spans = trace.get_recent_spans(1)
        assert len(spans) >= 1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
