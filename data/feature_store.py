"""
Feature Store - Centralized persistence and versioning for computed features.

Production-grade feature storage with:
- Parquet persistence with partitioning
- Lineage tracking (source → feature mappings)
- Point-in-time correctness (no lookahead bias)
- Incremental updates with change detection
- Feature metadata and statistics
- TTL-based cache invalidation

Usage:
    store = FeatureStore()

    # Write features
    store.write_features("volatility", ticker="AAPL", df=vol_features)

    # Read features (point-in-time)
    features = store.read_features("volatility", ticker="AAPL", as_of="2024-01-15")

    # Get lineage
    lineage = store.get_lineage("volatility", ticker="AAPL")
"""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureCategory(str, Enum):
    """Categories of features in the store."""
    # Raw data (Layer 0)
    OHLCV = "ohlcv"
    OPTIONS_CHAIN = "options_chain"
    OPTIONS_FLOW = "options_flow"
    EARNINGS = "earnings"
    DIVIDENDS = "dividends"
    FUNDAMENTALS = "fundamentals"
    RATES = "rates"

    # Derived features (Layer 1)
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    OPTIONS_FEATURES = "options_features"

    # Edge features (Layer 2)
    DYNAMICS = "dynamics"
    VOL_EDGE = "vol_edge"
    ASSIGNMENT = "assignment"
    EVENTS = "events"
    REGIME = "regime"

    # Labels (Layer 3)
    LABELS = "labels"

    # Composite (combined features)
    COMPOSITE = "composite"


@dataclass
class FeatureMetadata:
    """Metadata for a feature set."""
    category: str
    ticker: str
    created_at: str
    updated_at: str
    row_count: int
    date_range: Tuple[str, str]
    columns: List[str]
    source_hash: str  # Hash of source data for change detection
    source_files: List[str]  # Lineage: which raw files produced this
    computation_time_ms: int
    version: int = 1
    schema_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureMetadata":
        d["date_range"] = tuple(d["date_range"])
        return cls(**d)


@dataclass
class FeatureStats:
    """Statistics for feature validation."""
    column: str
    dtype: str
    count: int
    null_count: int
    null_pct: float
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    unique_count: Optional[int] = None


@dataclass
class LineageRecord:
    """Tracks data lineage (source → feature)."""
    feature_category: str
    feature_ticker: str
    source_category: str
    source_ticker: str
    source_file: Optional[str]
    transformation: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureStore:
    """
    Production-grade feature store with Parquet persistence.

    Directory structure:
        data/features/
        ├── ohlcv/
        │   └── ticker=AAPL/
        │       ├── data.parquet
        │       └── metadata.json
        ├── volatility/
        │   └── ticker=AAPL/
        │       ├── data.parquet
        │       └── metadata.json
        ├── _lineage/
        │   └── lineage.parquet
        ├── _stats/
        │   └── stats.parquet
        └── _registry/
            └── registry.json
    """

    def __init__(
        self,
        base_path: Union[str, Path] = "data/features",
        cache_ttl_hours: int = 24,
        enable_compression: bool = True,
    ):
        self.base_path = Path(base_path)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.enable_compression = enable_compression

        # In-memory cache
        self._cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self._metadata_cache: Dict[str, FeatureMetadata] = {}

        # Lineage tracking
        self._lineage_records: List[LineageRecord] = []

        # Initialize directory structure
        self._init_directories()

        # Load registry
        self._registry = self._load_registry()

        logger.info(f"FeatureStore initialized at {self.base_path}")

    def _init_directories(self) -> None:
        """Create directory structure."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "_lineage").mkdir(exist_ok=True)
        (self.base_path / "_stats").mkdir(exist_ok=True)
        (self.base_path / "_registry").mkdir(exist_ok=True)
        (self.base_path / "_locks").mkdir(exist_ok=True)

    @contextmanager
    def _file_lock(self, lock_name: str):
        """
        Acquire an exclusive file lock for atomic operations.

        Uses fcntl.flock for cross-process locking on Unix systems.
        Lock is automatically released when context exits.
        """
        lock_path = self.base_path / "_locks" / f"{lock_name}.lock"
        lock_file = None
        try:
            lock_file = open(lock_path, 'w')
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

    def _atomic_write_parquet(
        self,
        df: pd.DataFrame,
        target_path: Path,
        compression: str | None = "snappy",
    ) -> None:
        """
        Write DataFrame to Parquet atomically using temp file + fsync + rename.

        Ensures:
        1. Partial writes never corrupt the target file
        2. fsync guarantees data is on disk before rename
        3. Atomic rename makes the new file visible in one operation
        """
        # Create temp file in same directory for atomic rename on same filesystem
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            suffix=".parquet.tmp",
            dir=target_dir,
        )
        temp_path = Path(temp_path)

        try:
            # Close the fd opened by mkstemp, we'll write via pandas/pyarrow
            os.close(fd)

            # Write to temp file
            if PYARROW_AVAILABLE:
                table = pa.Table.from_pandas(df)
                pq.write_table(table, str(temp_path), compression=compression)
            else:
                df.to_parquet(str(temp_path), compression=compression, index=False)

            # fsync to ensure data is on disk
            with open(temp_path, 'rb') as f:
                os.fsync(f.fileno())

            # Atomic rename (guaranteed atomic on POSIX for same filesystem)
            os.replace(str(temp_path), str(target_path))

            # fsync parent directory to ensure rename is persisted
            dir_fd = os.open(str(target_dir), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _atomic_write_json(
        self,
        data: dict,
        target_path: Path,
    ) -> None:
        """
        Write JSON file atomically using temp file + fsync + rename.
        """
        target_dir = target_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(
            suffix=".json.tmp",
            dir=target_dir,
        )
        temp_path = Path(temp_path)

        try:
            os.close(fd)

            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(str(temp_path), str(target_path))

            dir_fd = os.open(str(target_dir), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _load_registry(self) -> dict:
        """Load feature registry."""
        registry_path = self.base_path / "_registry" / "registry.json"
        if registry_path.exists():
            with open(registry_path) as f:
                return json.load(f)
        return {"features": {}, "last_updated": None}

    def _save_registry(self) -> None:
        """Save feature registry atomically with file lock."""
        self._registry["last_updated"] = datetime.now().isoformat()
        registry_path = self.base_path / "_registry" / "registry.json"
        with self._file_lock("registry"):
            self._atomic_write_json(self._registry, registry_path)

    def _get_feature_path(self, category: str, ticker: str) -> Path:
        """Get path for a feature set."""
        return self.base_path / category / f"ticker={ticker}"

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of DataFrame for change detection."""
        # Use a sample for large DataFrames
        sample = df.head(1000) if len(df) > 1000 else df
        content = pd.util.hash_pandas_object(sample).values.tobytes()
        return hashlib.sha256(content).hexdigest()[:16]

    def _compute_stats(self, df: pd.DataFrame) -> List[FeatureStats]:
        """Compute column statistics for validation."""
        stats = []
        for col in df.columns:
            series = df[col]
            stat = FeatureStats(
                column=col,
                dtype=str(series.dtype),
                count=len(series),
                null_count=int(series.isna().sum()),
                null_pct=float(series.isna().mean()),
            )

            if pd.api.types.is_numeric_dtype(series):
                stat.mean = float(series.mean()) if not series.isna().all() else None
                stat.std = float(series.std()) if not series.isna().all() else None
                stat.min = float(series.min()) if not series.isna().all() else None
                stat.max = float(series.max()) if not series.isna().all() else None
                stat.p25 = float(series.quantile(0.25)) if not series.isna().all() else None
                stat.p50 = float(series.quantile(0.50)) if not series.isna().all() else None
                stat.p75 = float(series.quantile(0.75)) if not series.isna().all() else None

            if pd.api.types.is_categorical_dtype(series) or series.dtype == object:
                stat.unique_count = int(series.nunique())

            stats.append(stat)

        return stats

    def write_features(
        self,
        category: Union[str, FeatureCategory],
        ticker: str,
        df: pd.DataFrame,
        source_files: Optional[List[str]] = None,
        source_category: Optional[str] = None,
        transformation: str = "unknown",
        force: bool = False,
    ) -> FeatureMetadata:
        """
        Write features to the store.

        Args:
            category: Feature category (e.g., "volatility", "dynamics")
            ticker: Ticker symbol
            df: DataFrame with features (must have date index or column)
            source_files: List of source files for lineage
            source_category: Category of source data (for lineage)
            transformation: Description of transformation applied
            force: Force overwrite even if data unchanged

        Returns:
            FeatureMetadata for the written features
        """
        import time
        start_time = time.time()

        if isinstance(category, FeatureCategory):
            category = category.value

        # Ensure date column/index
        df = df.copy()
        if "date" not in df.columns and df.index.name != "date":
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.columns = ["date"] + list(df.columns[1:])

        # Compute hash for change detection
        data_hash = self._compute_data_hash(df)

        # Check if update needed
        existing_meta = self._load_metadata(category, ticker)
        if existing_meta and existing_meta.source_hash == data_hash and not force:
            logger.info(f"Skipping {category}/{ticker} - data unchanged")
            return existing_meta

        # Prepare path
        feature_path = self._get_feature_path(category, ticker)
        feature_path.mkdir(parents=True, exist_ok=True)

        # Determine date range
        if "date" in df.columns:
            date_col = pd.to_datetime(df["date"])
            date_range = (str(date_col.min().date()), str(date_col.max().date()))
        else:
            date_range = ("unknown", "unknown")

        # Use file lock for concurrent write safety
        lock_name = f"{category}_{ticker}".replace("/", "_")
        with self._file_lock(lock_name):
            # Write Parquet atomically
            parquet_path = feature_path / "data.parquet"
            compression = "snappy" if self.enable_compression else None
            self._atomic_write_parquet(df, parquet_path, compression)

            # Create metadata
            computation_time = int((time.time() - start_time) * 1000)
            version = (existing_meta.version + 1) if existing_meta else 1

            metadata = FeatureMetadata(
                category=category,
                ticker=ticker,
                created_at=existing_meta.created_at if existing_meta else datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                row_count=len(df),
                date_range=date_range,
                columns=list(df.columns),
                source_hash=data_hash,
                source_files=source_files or [],
                computation_time_ms=computation_time,
                version=version,
            )

            # Save metadata atomically
            meta_path = feature_path / "metadata.json"
            self._atomic_write_json(metadata.to_dict(), meta_path)

            # Compute and save stats atomically
            stats = self._compute_stats(df)
            stats_path = feature_path / "stats.json"
            self._atomic_write_json([asdict(s) for s in stats], stats_path)

        # Record lineage
        if source_category:
            lineage = LineageRecord(
                feature_category=category,
                feature_ticker=ticker,
                source_category=source_category,
                source_ticker=ticker,
                source_file=source_files[0] if source_files else None,
                transformation=transformation,
                timestamp=datetime.now().isoformat(),
            )
            self._lineage_records.append(lineage)

        # Update registry
        feature_key = f"{category}/{ticker}"
        self._registry["features"][feature_key] = {
            "version": version,
            "updated_at": metadata.updated_at,
            "row_count": len(df),
        }
        self._save_registry()

        # Update cache
        cache_key = f"{category}:{ticker}"
        self._cache[cache_key] = (df, datetime.now())
        self._metadata_cache[cache_key] = metadata

        logger.info(f"Wrote {len(df)} rows to {category}/{ticker} (v{version}, {computation_time}ms)")
        return metadata

    def read_features(
        self,
        category: Union[str, FeatureCategory],
        ticker: str,
        as_of: Optional[Union[str, date, datetime]] = None,
        columns: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Read features from the store.

        Args:
            category: Feature category
            ticker: Ticker symbol
            as_of: Point-in-time filter (only return data up to this date)
            columns: Specific columns to read
            use_cache: Whether to use in-memory cache

        Returns:
            DataFrame with features, or None if not found
        """
        if isinstance(category, FeatureCategory):
            category = category.value

        cache_key = f"{category}:{ticker}"

        # Check cache
        if use_cache and cache_key in self._cache:
            df, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self.cache_ttl:
                df = df.copy()
                if columns:
                    df = df[[c for c in columns if c in df.columns]]
                if as_of:
                    df = self._filter_as_of(df, as_of)
                return df

        # Read from disk
        feature_path = self._get_feature_path(category, ticker)
        parquet_path = feature_path / "data.parquet"

        if not parquet_path.exists():
            return None

        if PYARROW_AVAILABLE and columns:
            table = pq.read_table(parquet_path, columns=columns)
            df = table.to_pandas()
        else:
            df = pd.read_parquet(parquet_path)
            if columns:
                df = df[[c for c in columns if c in df.columns]]

        # Update cache
        if use_cache:
            self._cache[cache_key] = (df.copy(), datetime.now())

        # Apply point-in-time filter
        if as_of:
            df = self._filter_as_of(df, as_of)

        return df

    def _filter_as_of(
        self,
        df: pd.DataFrame,
        as_of: Union[str, date, datetime],
    ) -> pd.DataFrame:
        """Filter DataFrame to point-in-time."""
        if isinstance(as_of, str):
            as_of = pd.to_datetime(as_of)
        elif isinstance(as_of, date):
            as_of = pd.Timestamp(as_of)

        if "date" in df.columns:
            df = df[pd.to_datetime(df["date"]) <= as_of]

        return df

    def _load_metadata(self, category: str, ticker: str) -> Optional[FeatureMetadata]:
        """Load metadata for a feature set."""
        cache_key = f"{category}:{ticker}"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        meta_path = self._get_feature_path(category, ticker) / "metadata.json"
        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            data = json.load(f)

        metadata = FeatureMetadata.from_dict(data)
        self._metadata_cache[cache_key] = metadata
        return metadata

    def get_metadata(
        self,
        category: Union[str, FeatureCategory],
        ticker: str,
    ) -> Optional[FeatureMetadata]:
        """Get metadata for a feature set."""
        if isinstance(category, FeatureCategory):
            category = category.value
        return self._load_metadata(category, ticker)

    def get_stats(
        self,
        category: Union[str, FeatureCategory],
        ticker: str,
    ) -> Optional[List[FeatureStats]]:
        """Get statistics for a feature set."""
        if isinstance(category, FeatureCategory):
            category = category.value

        stats_path = self._get_feature_path(category, ticker) / "stats.json"
        if not stats_path.exists():
            return None

        with open(stats_path) as f:
            data = json.load(f)

        return [FeatureStats(**s) for s in data]

    def list_features(
        self,
        category: Optional[Union[str, FeatureCategory]] = None,
    ) -> List[Tuple[str, str]]:
        """
        List all features in the store.

        Args:
            category: Filter by category (optional)

        Returns:
            List of (category, ticker) tuples
        """
        features = []

        categories = [category.value if isinstance(category, FeatureCategory) else category] if category else None

        for key in self._registry.get("features", {}):
            cat, ticker = key.split("/")
            if categories is None or cat in categories:
                features.append((cat, ticker))

        # Also check filesystem for any not in registry
        for cat_dir in self.base_path.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("_"):
                cat = cat_dir.name
                if categories and cat not in categories:
                    continue
                for ticker_dir in cat_dir.iterdir():
                    if ticker_dir.is_dir() and ticker_dir.name.startswith("ticker="):
                        ticker = ticker_dir.name.replace("ticker=", "")
                        if (cat, ticker) not in features:
                            features.append((cat, ticker))

        return sorted(features)

    def get_tickers(self, category: Union[str, FeatureCategory]) -> List[str]:
        """Get all tickers for a category."""
        if isinstance(category, FeatureCategory):
            category = category.value

        features = self.list_features(category)
        return sorted(set(ticker for cat, ticker in features if cat == category))

    def get_lineage(
        self,
        category: Union[str, FeatureCategory],
        ticker: str,
    ) -> List[LineageRecord]:
        """Get lineage records for a feature set."""
        if isinstance(category, FeatureCategory):
            category = category.value

        return [
            r for r in self._lineage_records
            if r.feature_category == category and r.feature_ticker == ticker
        ]

    def save_lineage(self) -> None:
        """Persist lineage records to disk atomically."""
        if not self._lineage_records:
            return

        lineage_path = self.base_path / "_lineage" / "lineage.parquet"
        df = pd.DataFrame([r.to_dict() for r in self._lineage_records])
        with self._file_lock("lineage"):
            self._atomic_write_parquet(df, lineage_path, compression=None)
        logger.info(f"Saved {len(self._lineage_records)} lineage records")

    def load_lineage(self) -> List[LineageRecord]:
        """Load lineage records from disk."""
        lineage_path = self.base_path / "_lineage" / "lineage.parquet"
        if not lineage_path.exists():
            return []

        df = pd.read_parquet(lineage_path)
        self._lineage_records = [LineageRecord(**row) for _, row in df.iterrows()]
        return self._lineage_records

    def delete_features(
        self,
        category: Union[str, FeatureCategory],
        ticker: str,
    ) -> bool:
        """Delete a feature set."""
        if isinstance(category, FeatureCategory):
            category = category.value

        feature_path = self._get_feature_path(category, ticker)
        if not feature_path.exists():
            return False

        shutil.rmtree(feature_path)

        # Update registry
        feature_key = f"{category}/{ticker}"
        if feature_key in self._registry["features"]:
            del self._registry["features"][feature_key]
            self._save_registry()

        # Clear cache
        cache_key = f"{category}:{ticker}"
        self._cache.pop(cache_key, None)
        self._metadata_cache.pop(cache_key, None)

        logger.info(f"Deleted {category}/{ticker}")
        return True

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        self._metadata_cache.clear()
        logger.info("Cache cleared")

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for path in self.base_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
                file_count += 1

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "feature_count": len(self._registry.get("features", {})),
            "cache_entries": len(self._cache),
        }

    def health_check(self) -> dict:
        """Run health checks on the feature store."""
        issues = []

        # Check registry consistency
        for key in self._registry.get("features", {}):
            cat, ticker = key.split("/")
            parquet_path = self._get_feature_path(cat, ticker) / "data.parquet"
            if not parquet_path.exists():
                issues.append(f"Missing parquet file for {key}")

        # Check for orphaned files
        for cat_dir in self.base_path.iterdir():
            if cat_dir.is_dir() and not cat_dir.name.startswith("_"):
                for ticker_dir in cat_dir.iterdir():
                    if ticker_dir.is_dir():
                        ticker = ticker_dir.name.replace("ticker=", "")
                        key = f"{cat_dir.name}/{ticker}"
                        if key not in self._registry.get("features", {}):
                            issues.append(f"Orphaned directory: {key}")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "storage": self.get_storage_stats(),
        }


# Singleton instance for convenience
_default_store: Optional[FeatureStore] = None


def get_feature_store(base_path: str = "data/features") -> FeatureStore:
    """Get the default feature store instance."""
    global _default_store
    if _default_store is None:
        _default_store = FeatureStore(base_path)
    return _default_store
