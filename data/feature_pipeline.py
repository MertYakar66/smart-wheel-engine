"""
Feature Pipeline - Unified computation engine for all feature modules.

This is the CRITICAL piece that connects:
    Layer 0 (Raw Data) → Layer 1 (Derived) → Layer 2 (Edge) → Layer 3 (Labels)

All the feature modules in src/features/ are wired together here.

Usage:
    from data.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    pipeline.compute_all("AAPL")  # Compute all features for ticker
    pipeline.compute_universe(["AAPL", "MSFT"])  # Batch compute

    # Get computed features
    features = pipeline.get_composite_features("AAPL")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Feature modules
from src.features.volatility import VolatilityFeatures
from src.features.technical import TechnicalFeatures
from src.features.options import OptionsFeatures
from src.features.dynamics import OptionsDynamics
from src.features.vol_edge import VolatilityEdge
from src.features.assignment import AssignmentFeatures
from src.features.events import EventVolatility
from src.features.regime import RegimeDetector, MarketRegime, VolRegime
from src.features.labels import LabelGenerator, OptionOutcome

# Feature store
from data.feature_store import FeatureStore, FeatureCategory, get_feature_store

logger = logging.getLogger(__name__)


class ComputeStatus(str, Enum):
    """Status of feature computation."""
    PENDING = "pending"
    COMPUTING = "computing"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ComputeResult:
    """Result of a single feature computation."""
    category: str
    ticker: str
    status: ComputeStatus
    row_count: int = 0
    computation_time_ms: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""
    ticker: str
    started_at: str
    completed_at: str
    total_time_ms: int
    results: List[ComputeResult] = field(default_factory=list)
    success: bool = True

    def summary(self) -> str:
        lines = [
            f"Pipeline Result for {self.ticker}",
            f"Time: {self.total_time_ms}ms",
            "=" * 50,
        ]
        for r in self.results:
            status_icon = "✓" if r.status == ComputeStatus.SUCCESS else "✗"
            lines.append(f"  {status_icon} {r.category}: {r.row_count} rows ({r.computation_time_ms}ms)")
            if r.error:
                lines.append(f"      Error: {r.error}")
        return "\n".join(lines)


class FeaturePipeline:
    """
    Unified feature computation pipeline.

    Computes all features in the correct dependency order:

    Layer 0: Raw data loading (from DataPipeline)
        ├── OHLCV
        ├── Options chains
        ├── Options flow
        ├── Earnings
        ├── Dividends
        ├── IV history
        └── Fundamentals

    Layer 1: Derived features
        ├── Technical indicators (from OHLCV)
        ├── Realized volatility (from OHLCV)
        └── Options flow features (from options)

    Layer 2: Edge features (THE ALPHA)
        ├── Dynamics (ΔOI, ΔIV) - from options flow
        ├── Vol edge (IV vs RV) - from volatility + IV history
        ├── Assignment risk - from options + OHLCV
        ├── Event volatility - from earnings + IV
        └── Regime detection - from OHLCV + volatility

    Layer 3: Labels (for ML training)
        └── Option outcomes
    """

    def __init__(
        self,
        data_pipeline: Optional["DataPipeline"] = None,
        feature_store: Optional[FeatureStore] = None,
        cache_results: bool = True,
        auto_load: bool = True,
    ):
        # Lazy import to avoid circular dependency
        if data_pipeline is None:
            from data.pipeline import DataPipeline
            data_pipeline = DataPipeline()
            if auto_load:
                data_pipeline.load_all()
        self.data = data_pipeline

        self.store = feature_store or get_feature_store()
        self.cache_results = cache_results

        # Initialize feature modules
        self.volatility = VolatilityFeatures()
        self.technical = TechnicalFeatures()
        self.options_features = OptionsFeatures()
        self.dynamics = OptionsDynamics()
        self.vol_edge = VolatilityEdge()
        self.assignment = AssignmentFeatures()
        self.events = EventVolatility()
        self.regime = RegimeDetector()
        self.labels = LabelGenerator()

        logger.info("FeaturePipeline initialized")

    def compute_all(
        self,
        ticker: str,
        force: bool = False,
        layers: Optional[List[int]] = None,
    ) -> PipelineResult:
        """
        Compute all features for a single ticker.

        Args:
            ticker: Ticker symbol
            force: Force recomputation even if cached
            layers: Specific layers to compute (default: all)

        Returns:
            PipelineResult with computation details
        """
        start_time = time.time()
        started_at = datetime.now().isoformat()
        results = []
        layers = layers or [1, 2, 3]

        logger.info(f"Starting feature computation for {ticker}")

        # Layer 1: Derived features
        if 1 in layers:
            results.extend(self._compute_layer1(ticker, force))

        # Layer 2: Edge features
        if 2 in layers:
            results.extend(self._compute_layer2(ticker, force))

        # Layer 3: Labels
        if 3 in layers:
            results.extend(self._compute_layer3(ticker, force))

        # Save lineage
        self.store.save_lineage()

        # Build result
        total_time = int((time.time() - start_time) * 1000)
        success = all(r.status in (ComputeStatus.SUCCESS, ComputeStatus.SKIPPED) for r in results)

        result = PipelineResult(
            ticker=ticker,
            started_at=started_at,
            completed_at=datetime.now().isoformat(),
            total_time_ms=total_time,
            results=results,
            success=success,
        )

        logger.info(f"Completed {ticker}: {len(results)} categories, {total_time}ms, success={success}")
        return result

    def _compute_layer1(self, ticker: str, force: bool) -> List[ComputeResult]:
        """Compute Layer 1: Derived features."""
        results = []

        # Technical indicators
        results.append(self._compute_technical(ticker, force))

        # Realized volatility
        results.append(self._compute_volatility(ticker, force))

        # Options flow features
        results.append(self._compute_options_flow(ticker, force))

        return results

    def _compute_layer2(self, ticker: str, force: bool) -> List[ComputeResult]:
        """Compute Layer 2: Edge features."""
        results = []

        # Dynamics (ΔOI, ΔIV)
        results.append(self._compute_dynamics(ticker, force))

        # Volatility edge (IV vs RV)
        results.append(self._compute_vol_edge(ticker, force))

        # Assignment risk
        results.append(self._compute_assignment(ticker, force))

        # Event volatility
        results.append(self._compute_events(ticker, force))

        # Regime detection
        results.append(self._compute_regime(ticker, force))

        return results

    def _compute_layer3(self, ticker: str, force: bool) -> List[ComputeResult]:
        """Compute Layer 3: Labels."""
        results = []

        # Option outcome labels
        results.append(self._compute_labels(ticker, force))

        return results

    def _compute_technical(self, ticker: str, force: bool) -> ComputeResult:
        """Compute technical indicators."""
        category = FeatureCategory.TECHNICAL.value
        start_time = time.time()

        try:
            # Get OHLCV data
            ohlcv = self.data.get_ohlcv(ticker)
            if ohlcv is None or ohlcv.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No OHLCV data available"],
                )

            # Ensure proper column names
            df = ohlcv.copy()
            df.columns = [c.lower() for c in df.columns]

            # Compute technical indicators
            df = self.technical.compute_all(df)

            # Store results
            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.OHLCV.value,
                transformation="technical_indicators",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing technical for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_volatility(self, ticker: str, force: bool) -> ComputeResult:
        """Compute realized volatility features."""
        category = FeatureCategory.VOLATILITY.value
        start_time = time.time()

        try:
            ohlcv = self.data.get_ohlcv(ticker)
            if ohlcv is None or ohlcv.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No OHLCV data available"],
                )

            df = ohlcv.copy()
            df.columns = [c.lower() for c in df.columns]

            # Compute log returns
            df["returns"] = np.log(df["close"] / df["close"].shift(1))

            # Compute realized volatility (multiple estimators)
            df["rv_5d"] = self.volatility.realized_volatility_close(df["returns"], window=5)
            df["rv_10d"] = self.volatility.realized_volatility_close(df["returns"], window=10)
            df["rv_21d"] = self.volatility.realized_volatility_close(df["returns"], window=21)
            df["rv_63d"] = self.volatility.realized_volatility_close(df["returns"], window=63)

            # Parkinson (uses high-low)
            df["rv_parkinson_21d"] = self.volatility.realized_volatility_parkinson(df["high"], df["low"], window=21)

            # Garman-Klass (uses OHLC)
            df["rv_garman_klass_21d"] = self.volatility.realized_volatility_garman_klass(
                df["open"], df["high"], df["low"], df["close"], window=21
            )

            # Yang-Zhang (most robust)
            df["rv_yang_zhang_21d"] = self.volatility.realized_volatility_yang_zhang(
                df["open"], df["high"], df["low"], df["close"], window=21
            )

            # Store
            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.OHLCV.value,
                transformation="realized_volatility",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing volatility for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_options_flow(self, ticker: str, force: bool) -> ComputeResult:
        """Compute options flow features."""
        category = FeatureCategory.OPTIONS_FEATURES.value
        start_time = time.time()

        try:
            # Try to get options flow data (aggregated daily)
            # This may come from bloomberg_loader's options flow or IV history
            iv_history = self.data.iv_history.get(ticker) if hasattr(self.data, 'iv_history') else None

            if iv_history is None or iv_history.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No options flow data available"],
                )

            df = iv_history.copy()

            # Compute options flow features (if we have volume/OI data)
            if any(col in df.columns for col in ["call_volume", "put_volume", "call_oi", "put_oi"]):
                df = self.options_features.compute_flow_features(df)
            # Otherwise just pass through the IV data

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category="iv_history",
                transformation="options_flow_features",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing options flow for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_dynamics(self, ticker: str, force: bool) -> ComputeResult:
        """Compute dynamics features (THE EDGE - ΔOI, ΔIV)."""
        category = FeatureCategory.DYNAMICS.value
        start_time = time.time()

        try:
            # Get IV history for dynamics computation
            iv_history = self.data.iv_history.get(ticker) if hasattr(self.data, 'iv_history') else None

            if iv_history is None or iv_history.empty:
                # Try to get from options flow or feature store
                iv_history = self.store.read_features(FeatureCategory.OPTIONS_FEATURES, ticker)

            if iv_history is None or iv_history.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No IV history for dynamics computation"],
                )

            df = iv_history.copy()

            # Map columns for dynamics computation
            # Dynamics module expects: atm_iv, call_oi, put_oi, call_volume, put_volume, close
            col_mapping = {
                "iv_atm_30d": "atm_iv",
                "30day_impvol_100.0%mny_df": "atm_iv",
                "iv_30d": "atm_iv",
            }

            for old_col, new_col in col_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]

            # Compute IV dynamics if we have IV data
            if "atm_iv" in df.columns:
                df["iv_change_1d"] = self.dynamics.iv_change(df["atm_iv"], 1)
                df["iv_change_5d"] = self.dynamics.iv_change(df["atm_iv"], 5)
                df["iv_velocity_5d"] = self.dynamics.iv_velocity(df["atm_iv"], 5)
                df["iv_acceleration"] = self.dynamics.iv_acceleration(df["atm_iv"])

            # Term structure dynamics if available
            if "iv_30d" in df.columns and "iv_60d" in df.columns:
                df["term_structure_change"] = self.dynamics.term_structure_change(
                    df["iv_30d"], df["iv_60d"]
                )
                df["term_structure_regime"] = self.dynamics.term_structure_regime(
                    df["iv_30d"], df["iv_60d"]
                )

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.OPTIONS_FEATURES.value,
                transformation="dynamics_features",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing dynamics for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_vol_edge(self, ticker: str, force: bool) -> ComputeResult:
        """Compute volatility edge features (THE CORE - IV vs RV)."""
        category = FeatureCategory.VOL_EDGE.value
        start_time = time.time()

        try:
            # Get volatility features (contains RV)
            vol_features = self.store.read_features(FeatureCategory.VOLATILITY, ticker)

            # Get IV history
            iv_history = self.data.iv_history.get(ticker) if hasattr(self.data, 'iv_history') else None

            if vol_features is None or vol_features.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No volatility features available"],
                )

            df = vol_features.copy()

            # Merge IV data if available
            if iv_history is not None and not iv_history.empty:
                iv_df = iv_history.copy()
                # Normalize column names
                col_mapping = {
                    "30day_impvol_100.0%mny_df": "atm_iv",
                    "iv_atm_30d": "atm_iv",
                    "iv_30d": "atm_iv",
                    "60day_impvol_100.0%mny_df": "iv_60d",
                    "90day_impvol_100.0%mny_df": "iv_90d",
                }
                for old_col, new_col in col_mapping.items():
                    if old_col in iv_df.columns:
                        iv_df[new_col] = iv_df[old_col]

                # Merge on date
                if "date" in df.columns and "date" in iv_df.columns:
                    df = df.merge(iv_df[["date", "atm_iv"]], on="date", how="left")
            else:
                # Use realized vol as proxy for IV (not ideal)
                if "rv_21d" in df.columns:
                    df["atm_iv"] = df["rv_21d"] * 1.1  # Slight premium

            if "atm_iv" not in df.columns or "rv_21d" not in df.columns:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["Missing IV or RV data for vol edge"],
                )

            # Compute vol edge features
            df["iv_rv_spread"] = self.vol_edge.iv_rv_spread(df["atm_iv"], df["rv_21d"])
            df["iv_rv_ratio"] = self.vol_edge.iv_rv_ratio(df["atm_iv"], df["rv_21d"])
            df["iv_rv_zscore"] = self.vol_edge.iv_rv_zscore(df["atm_iv"], df["rv_21d"])
            df["vrp_percentile"] = self.vol_edge.vrp_percentile(df["atm_iv"], df["rv_21d"])
            df["vol_regime"] = self.vol_edge.vol_regime(df["rv_21d"])
            df["vol_regime_transition"] = self.vol_edge.vol_regime_transition(df["rv_21d"])
            df["iv_mean_reversion"] = self.vol_edge.mean_reversion_signal(df["atm_iv"])

            # Compute IV rank from history
            if len(df) >= 252:
                df["iv_rank"] = df["atm_iv"].rolling(252).apply(
                    lambda x: (x < x.iloc[-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
                    raw=False
                )
            else:
                df["iv_rank"] = 50.0  # Default

            # Composite edge score
            df["edge_score"] = self.vol_edge.edge_score(
                df["atm_iv"],
                df["rv_21d"],
                df["iv_rank"],
                df["vrp_percentile"]
            )

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.VOLATILITY.value,
                transformation="vol_edge_features",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing vol edge for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_assignment(self, ticker: str, force: bool) -> ComputeResult:
        """Compute assignment risk features."""
        category = FeatureCategory.ASSIGNMENT.value
        start_time = time.time()

        try:
            # Assignment features need options chain data
            options = self.data.get_options(ticker)

            if options is None or options.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No options data for assignment features"],
                )

            df = options.copy()

            # Get current price
            spot_price = self.data.get_spot_price(ticker)
            if spot_price is None:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No spot price available"],
                )

            # Compute assignment features
            df = self.assignment.compute_all(df, spot_price=spot_price)

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.OPTIONS_CHAIN.value,
                transformation="assignment_features",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing assignment for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_events(self, ticker: str, force: bool) -> ComputeResult:
        """Compute event volatility features."""
        category = FeatureCategory.EVENTS.value
        start_time = time.time()

        try:
            # Get earnings data
            earnings = self.data.earnings.get(ticker) if hasattr(self.data, 'earnings') else None

            if earnings is None or earnings.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No earnings data for event features"],
                )

            df = earnings.copy()

            # Get IV history for event vol computation
            iv_history = self.data.iv_history.get(ticker) if hasattr(self.data, 'iv_history') else None

            # Get OHLCV for price data
            ohlcv = self.data.get_ohlcv(ticker)

            # EventVolatility.compute_earnings_features expects:
            #   df: DataFrame with price and IV data (indexed by date)
            #   earnings_dates: Series of earnings dates
            if ohlcv is None or ohlcv.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No OHLCV data for event features"],
                )

            # Build a price+IV DataFrame
            price_df = ohlcv.copy()
            price_df.columns = [c.lower() for c in price_df.columns]
            if "date" in price_df.columns:
                price_df = price_df.set_index(pd.to_datetime(price_df["date"]))

            # Merge IV if available
            if iv_history is not None and not iv_history.empty:
                iv_df = iv_history.copy()
                if "atm_iv" in iv_df.columns and "date" in iv_df.columns:
                    iv_df = iv_df.set_index(pd.to_datetime(iv_df["date"]))
                    price_df["atm_iv"] = iv_df["atm_iv"].reindex(price_df.index, method="ffill")

            # Extract earnings dates
            date_col = "earnings_date" if "earnings_date" in df.columns else "announcement_date"
            if date_col not in df.columns:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=[f"No {date_col} column in earnings data"],
                )

            earnings_dates = pd.to_datetime(df[date_col].dropna())

            # Compute event features
            result_df = self.events.compute_earnings_features(
                price_df,
                earnings_dates,
                iv_col="atm_iv" if "atm_iv" in price_df.columns else "close",
                close_col="close",
                open_col="open",
            )

            # Reset index for storage (handle duplicate 'date' column)
            if result_df.index.name == "date" or (hasattr(result_df.index, 'names') and 'date' in (result_df.index.names or [])):
                if "date" in result_df.columns:
                    result_df = result_df.drop(columns=["date"]).reset_index()
                else:
                    result_df = result_df.reset_index()

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=result_df,
                source_category=FeatureCategory.EARNINGS.value,
                transformation="event_volatility",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing events for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_regime(self, ticker: str, force: bool) -> ComputeResult:
        """Compute market regime features."""
        category = FeatureCategory.REGIME.value
        start_time = time.time()

        try:
            # Get volatility features
            vol_features = self.store.read_features(FeatureCategory.VOLATILITY, ticker)

            if vol_features is None or vol_features.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No volatility features for regime detection"],
                )

            df = vol_features.copy()

            # RegimeDetector.compute_all expects individual Series:
            #   price, rv, volume, vix (optional)
            if "close" not in df.columns or "rv_21d" not in df.columns or "volume" not in df.columns:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["Missing close/rv_21d/volume for regime detection"],
                )

            regime_df = self.regime.compute_all(
                price=df["close"].copy(),
                rv=df["rv_21d"].copy(),
                volume=df["volume"].copy(),
            )

            # Merge regime features back (use list() to avoid iteration issues)
            regime_cols = list(regime_df.columns)
            for col in regime_cols:
                df[col] = regime_df[col].values

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.VOLATILITY.value,
                transformation="regime_detection",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing regime for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def _compute_labels(self, ticker: str, force: bool) -> ComputeResult:
        """Compute outcome labels for ML training."""
        category = FeatureCategory.LABELS.value
        start_time = time.time()

        try:
            # Get vol edge features (includes RV for outcome computation)
            vol_edge = self.store.read_features(FeatureCategory.VOL_EDGE, ticker)

            if vol_edge is None or vol_edge.empty:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No vol edge features for label generation"],
                )

            df = vol_edge.copy()

            # LabelGenerator.generate_training_labels expects:
            #   df with price_col and iv_col
            price_col = "close" if "close" in df.columns else None
            iv_col = "atm_iv" if "atm_iv" in df.columns else None

            if price_col is None:
                return ComputeResult(
                    category=category,
                    ticker=ticker,
                    status=ComputeStatus.SKIPPED,
                    warnings=["No close price for label generation"],
                )

            kwargs = {"price_col": price_col}
            if iv_col:
                kwargs["iv_col"] = iv_col

            df = self.labels.generate_training_labels(df, **kwargs)

            self.store.write_features(
                category=category,
                ticker=ticker,
                df=df,
                source_category=FeatureCategory.VOL_EDGE.value,
                transformation="label_generation",
                force=force,
            )

            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.SUCCESS,
                row_count=len(df),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Error computing labels for {ticker}: {e}")
            return ComputeResult(
                category=category,
                ticker=ticker,
                status=ComputeStatus.FAILED,
                error=str(e),
                computation_time_ms=int((time.time() - start_time) * 1000),
            )

    def compute_universe(
        self,
        tickers: List[str],
        force: bool = False,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> Dict[str, PipelineResult]:
        """
        Compute features for multiple tickers.

        Args:
            tickers: List of ticker symbols
            force: Force recomputation
            parallel: Use parallel processing
            max_workers: Number of parallel workers

        Returns:
            Dict mapping ticker to PipelineResult
        """
        results = {}
        total_start = time.time()

        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.compute_all, ticker, force): ticker
                    for ticker in tickers
                }

                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        results[ticker] = future.result()
                    except Exception as e:
                        logger.error(f"Failed to compute {ticker}: {e}")
                        results[ticker] = PipelineResult(
                            ticker=ticker,
                            started_at=datetime.now().isoformat(),
                            completed_at=datetime.now().isoformat(),
                            total_time_ms=0,
                            success=False,
                        )
        else:
            for ticker in tickers:
                results[ticker] = self.compute_all(ticker, force)

        total_time = int((time.time() - total_start) * 1000)
        success_count = sum(1 for r in results.values() if r.success)

        logger.info(f"Computed universe: {success_count}/{len(tickers)} succeeded, {total_time}ms total")
        return results

    def get_composite_features(
        self,
        ticker: str,
        as_of: Optional[Union[str, date, datetime]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get all features combined into a single DataFrame.

        Args:
            ticker: Ticker symbol
            as_of: Point-in-time filter

        Returns:
            Combined feature DataFrame
        """
        # Start with vol edge (contains most features)
        composite = self.store.read_features(FeatureCategory.VOL_EDGE, ticker, as_of=as_of)

        if composite is None:
            # Fall back to volatility
            composite = self.store.read_features(FeatureCategory.VOLATILITY, ticker, as_of=as_of)

        if composite is None:
            # Fall back to technical
            composite = self.store.read_features(FeatureCategory.TECHNICAL, ticker, as_of=as_of)

        if composite is None:
            return None

        # Merge additional features
        for category in [FeatureCategory.DYNAMICS, FeatureCategory.REGIME, FeatureCategory.LABELS]:
            additional = self.store.read_features(category, ticker, as_of=as_of)
            if additional is not None and "date" in additional.columns:
                # Only add new columns
                new_cols = [c for c in additional.columns if c not in composite.columns and c != "date"]
                if new_cols:
                    composite = composite.merge(
                        additional[["date"] + new_cols],
                        on="date",
                        how="left"
                    )

        return composite

    def get_latest_features(self, ticker: str) -> Optional[pd.Series]:
        """Get the most recent feature values for a ticker."""
        composite = self.get_composite_features(ticker)
        if composite is None or composite.empty:
            return None

        return composite.iloc[-1]

    def status(self) -> dict:
        """Get pipeline status."""
        features = self.store.list_features()
        categories = {}

        for cat, ticker in features:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(ticker)

        return {
            "feature_store": self.store.get_storage_stats(),
            "categories": {k: len(v) for k, v in categories.items()},
            "total_features": len(features),
        }


# Convenience function
def compute_features(
    ticker: str,
    force: bool = False,
) -> PipelineResult:
    """Compute all features for a ticker."""
    pipeline = FeaturePipeline()
    return pipeline.compute_all(ticker, force)
