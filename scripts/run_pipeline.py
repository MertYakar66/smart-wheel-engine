#!/usr/bin/env python3
"""
Pipeline Runner CLI - Run the Smart Wheel data engineering pipeline.

Usage:
    python scripts/run_pipeline.py --help
    python scripts/run_pipeline.py load --tickers AAPL MSFT
    python scripts/run_pipeline.py features --tickers AAPL
    python scripts/run_pipeline.py full --tickers AAPL MSFT GOOGL
    python scripts/run_pipeline.py status
    python scripts/run_pipeline.py health

Examples:
    # Load data for all S&P 500 constituents
    python scripts/run_pipeline.py load --universe sp500

    # Compute features for specific tickers
    python scripts/run_pipeline.py features --tickers AAPL MSFT NVDA

    # Full pipeline with parallel execution
    python scripts/run_pipeline.py full --tickers AAPL MSFT --parallel

    # Check pipeline health
    python scripts/run_pipeline.py health
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.observability import metrics, setup_logging
from data.orchestrator import PipelineOrchestrator, StageType


def setup_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Smart Wheel Data Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Log to file in addition to console",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Load command
    load_parser = subparsers.add_parser("load", help="Load raw data")
    load_parser.add_argument("--tickers", nargs="+", help="Ticker symbols")
    load_parser.add_argument("--universe", choices=["sp500", "mag7"], help="Predefined universe")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data quality")
    validate_parser.add_argument("--tickers", nargs="+", help="Ticker symbols")

    # Features command
    features_parser = subparsers.add_parser("features", help="Compute features")
    features_parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols")
    features_parser.add_argument("--force", action="store_true", help="Force recomputation")
    features_parser.add_argument("--layers", nargs="+", type=int, choices=[1, 2, 3], help="Feature layers to compute")

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Run full pipeline")
    full_parser.add_argument("--tickers", nargs="+", help="Ticker symbols")
    full_parser.add_argument("--universe", choices=["sp500", "mag7"], help="Predefined universe")
    full_parser.add_argument("--parallel", action="store_true", help="Enable parallel execution")
    full_parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    full_parser.add_argument("--resume", help="Resume from checkpoint ID")

    # Status command
    subparsers.add_parser("status", help="Show pipeline status")

    # Health command
    subparsers.add_parser("health", help="Run health checks")

    # Metrics command
    subparsers.add_parser("metrics", help="Show collected metrics")

    return parser


def get_universe(name: str) -> list:
    """Get predefined ticker universe."""
    universes = {
        "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        "sp500": None,  # Will be loaded from constituents file
    }

    if name == "sp500":
        # Load from file
        constituents_path = Path("data_raw/sp500_constituents_current.csv")
        if constituents_path.exists():
            import pandas as pd
            df = pd.read_csv(constituents_path)
            return df["ticker"].tolist()
        else:
            print("Warning: S&P 500 constituents file not found. Using MAG7.")
            return universes["mag7"]

    return universes.get(name, [])


def cmd_load(args, orchestrator):
    """Run load command."""
    tickers = args.tickers or []
    if args.universe:
        tickers = get_universe(args.universe)

    if not tickers:
        print("Error: No tickers specified. Use --tickers or --universe")
        return 1

    print(f"Loading data for {len(tickers)} tickers...")
    result = orchestrator.run_stage(StageType.LOAD, tickers)
    print(result.summary())
    return 0 if result.success_count == len(tickers) else 1


def cmd_validate(args, orchestrator):
    """Run validate command."""
    tickers = args.tickers or []
    if not tickers:
        print("Error: No tickers specified. Use --tickers")
        return 1

    print(f"Validating data for {len(tickers)} tickers...")
    result = orchestrator.run_stage(StageType.VALIDATE, tickers)
    print(result.summary())
    return 0 if result.success_count == len(tickers) else 1


def cmd_features(args, orchestrator):
    """Run features command."""
    from data.feature_pipeline import FeaturePipeline

    tickers = args.tickers
    print(f"Computing features for {len(tickers)} tickers...")

    pipeline = FeaturePipeline()

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        result = pipeline.compute_all(
            ticker,
            force=args.force,
            layers=args.layers,
        )
        print(result.summary())

    return 0


def cmd_full(args, orchestrator):
    """Run full pipeline command."""
    tickers = args.tickers or []
    if args.universe:
        tickers = get_universe(args.universe)

    if not tickers:
        print("Error: No tickers specified. Use --tickers or --universe")
        return 1

    print(f"Running full pipeline for {len(tickers)} tickers...")
    print(f"Parallel: {args.parallel}, Workers: {args.workers}")

    if args.parallel:
        orchestrator.max_workers = args.workers

    result = orchestrator.run_full_pipeline(
        tickers,
        parallel=args.parallel,
        resume_from=args.resume,
    )

    print("\n" + "=" * 60)
    print(result.summary())

    return 0 if result.success_count == len(tickers) else 1


def cmd_status(args, orchestrator):
    """Show pipeline status."""
    status = orchestrator.status()

    print("Pipeline Status")
    print("=" * 40)
    print(f"Running: {status['running']}")
    print(f"Current Run: {status['current_run'] or 'None'}")
    print(f"Total Runs: {status['total_runs']}")
    print(f"Workers: {status['workers']}")
    print(f"Checkpoints: {status['checkpoints']}")

    print("\nMetrics:")
    for key, value in status['metrics'].items():
        print(f"  {key}: {value}")

    return 0


def cmd_health(args, orchestrator):
    """Run health checks."""
    print("Running health checks...")

    health = orchestrator.health_check()

    if health["healthy"]:
        print("\n[OK] All health checks passed")
    else:
        print("\n[WARN] Health issues detected:")
        for issue in health["issues"]:
            print(f"  - {issue}")

    print("\nStatus:")
    for key, value in health["status"].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    return 0 if health["healthy"] else 1


def cmd_metrics(args, orchestrator):
    """Show collected metrics."""
    all_metrics = metrics.get_all_metrics()

    print("Collected Metrics")
    print("=" * 40)

    for category, values in all_metrics.items():
        if values:
            print(f"\n{category.upper()}:")
            for name, data in values.items():
                if isinstance(data, dict):
                    print(f"  {name}:")
                    for k, v in data.items():
                        if isinstance(v, float):
                            print(f"    {k}: {v:.2f}")
                        else:
                            print(f"    {k}: {v}")
                else:
                    print(f"  {name}: {data}")

    return 0


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=level, log_file=args.log_file)

    # Create orchestrator
    orchestrator = PipelineOrchestrator()

    # Dispatch command
    commands = {
        "load": cmd_load,
        "validate": cmd_validate,
        "features": cmd_features,
        "full": cmd_full,
        "status": cmd_status,
        "health": cmd_health,
        "metrics": cmd_metrics,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        return cmd_func(args, orchestrator)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
