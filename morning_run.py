#!/usr/bin/env python3
"""
Morning News Pipeline Runner

Your morning financial news workflow in one command.

Usage:
    python morning_run.py                      # Run with defaults (overnight news)
    python morning_run.py --time-window 6h     # Last 6 hours only
    python morning_run.py --tickers AAPL,NVDA  # Focus on specific stocks
    python morning_run.py --categories fed,oil # Specific categories only
    python morning_run.py --dry-run            # Preview without publishing
    python morning_run.py --json               # Output as JSON

Example morning workflow:
    $ python morning_run.py --time-window overnight --min-confidence 7

Pipeline stages:
    1. Grok AI discovers breaking news via web search
    2. Gemini verifies each story against multiple sources
    3. ChatGPT formats verified facts into clear content
    4. Claude adds "why it matters" editorial polish
    5. Publisher sends to your website feed
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from news_pipeline import DiscoveryRequest, NewsPipelineOrchestrator, PipelineConfig
from news_pipeline.publisher import create_publisher_callback


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Morning News Pipeline - Multi-model AI financial news processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Discovery options
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers to monitor (e.g., AAPL,NVDA,TSLA)",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default="fed,earnings,sp500_events,oil,geopolitics",
        help="Comma-separated categories to include",
    )

    parser.add_argument(
        "--time-window",
        type=str,
        default="overnight",
        choices=["1h", "3h", "6h", "overnight", "today", "yesterday"],
        help="Time window for news discovery (default: overnight)",
    )

    parser.add_argument(
        "--max-stories",
        type=int,
        default=50,
        help="Maximum stories to discover (default: 50)",
    )

    # Verification options
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=6,
        choices=range(0, 11),
        metavar="[0-10]",
        help="Minimum verification confidence to publish (default: 6)",
    )

    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't publish - just show what would be published",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Write JSON output to file",
    )

    # Publishing options
    parser.add_argument(
        "--publish-mode",
        type=str,
        default="database",
        choices=["api", "database", "file"],
        help="Publishing mode (default: database)",
    )

    parser.add_argument(
        "--api-endpoint",
        type=str,
        help="API endpoint for mode=api",
    )

    # Operational options
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Check provider health and exit",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        help="Write logs to file",
    )

    return parser.parse_args()


def print_header(args: argparse.Namespace) -> None:
    """Print pipeline header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'=' * 60}")
    print(f"  MORNING NEWS PIPELINE - {now}")
    print(f"{'=' * 60}")
    print(f"  Time window:    {args.time_window}")
    print(f"  Categories:     {args.categories}")
    if args.tickers:
        print(f"  Tickers:        {args.tickers}")
    print(f"  Min confidence: {args.min_confidence}/10")
    if args.dry_run:
        print("  Mode:           DRY RUN (no publishing)")
    print(f"{'=' * 60}\n")


def print_story(story, index: int) -> None:
    """Pretty-print a finalized story."""
    print(f"\n{'─' * 60}")
    print(
        f"STORY #{index + 1} | Priority: {story.priority}/10 | "
        f"Confidence: {story.verification_confidence}/10"
    )
    if story.is_breaking:
        print("🔴 BREAKING NEWS")
    print(f"{'─' * 60}")

    print(f"\n📰 {story.title}")
    print(f"\n   Category: {story.category.upper()}")
    print(f"   Assets:   {', '.join(story.affected_assets)}")

    print("\n   WHAT HAPPENED:")
    print(f"   {story.what_happened}")

    print("\n   WHY IT MATTERS:")
    print(f"   {story.why_it_matters}")

    if story.market_implications:
        print("\n   MARKET IMPLICATIONS:")
        print(f"   {story.market_implications}")

    if story.trading_considerations:
        print("\n   TRADING CONSIDERATIONS:")
        print(f"   {story.trading_considerations}")

    print("\n   KEY POINTS:")
    for bp in story.bullet_points:
        print(f"   • {bp}")

    if story.tags:
        print(f"\n   Tags: {', '.join(story.tags)}")


def print_summary(result) -> None:
    """Print pipeline run summary."""
    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Run ID:       {result.run_id}")
    print(f"  Duration:     {result.duration_seconds:.1f}s")
    print(f"  Status:       {result.status.upper()}")
    print(f"{'─' * 60}")
    print(f"  Discovered:   {result.discovered_count} candidates")
    print(f"  Verified:     {result.verified_count} passed threshold")
    print(f"  Formatted:    {result.formatted_count} stories")
    print(f"  Finalized:    {result.finalized_count} stories")
    print(f"  Published:    {result.published_count} stories")
    print(f"{'─' * 60}")
    print(f"  Success rate: {result.success_rate:.1%}")
    print(f"{'=' * 60}\n")

    if result.errors:
        print("  ERRORS:")
        for error in result.errors:
            print(f"  - {error}")
        print()


async def run_health_check(config: PipelineConfig) -> bool:
    """Run health check on all providers."""
    print("\nChecking provider health...\n")

    orchestrator = NewsPipelineOrchestrator(config=config)

    try:
        await orchestrator.initialize()
        health = await orchestrator.health_check()

        all_healthy = True
        for provider, status in health.items():
            icon = "✓" if status else "✗"
            status_text = "OK" if status else "FAILED"
            print(f"  {icon} {provider.capitalize():12} {status_text}")
            if not status:
                all_healthy = False

        print()
        return all_healthy

    except Exception as e:
        print(f"  Health check error: {e}\n")
        return False


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_logging(
        verbose=args.verbose,
        log_file=args.log_file,
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    config = PipelineConfig.from_env()
    config.min_verification_confidence = args.min_confidence
    config.max_stories_per_run = args.max_stories

    # Health check mode
    if args.health_check:
        healthy = await run_health_check(config)
        return 0 if healthy else 1

    # Print header
    if not args.json:
        print_header(args)

    # Parse inputs
    tickers = tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())
    categories = tuple(c.strip().lower() for c in args.categories.split(",") if c.strip())

    # Map time window
    time_window_map = {
        "1h": "last_1h",
        "3h": "last_3h",
        "6h": "last_6h",
        "overnight": "overnight",
        "today": "today",
        "yesterday": "yesterday",
    }
    time_window = time_window_map.get(args.time_window, args.time_window)

    # Create discovery request
    request = DiscoveryRequest(
        tickers=tickers,
        categories=categories,
        time_window=time_window,
        max_results=args.max_stories,
    )

    # Setup publisher
    if args.dry_run:

        async def dry_run_publisher(item):
            logger.info(f"[DRY-RUN] Would publish: {item.title}")
            return True

        publisher = dry_run_publisher
    else:
        publisher = create_publisher_callback(
            mode=args.publish_mode,
            api_endpoint=args.api_endpoint,
        )

    # Create orchestrator
    orchestrator = NewsPipelineOrchestrator(
        config=config,
        publish_callback=publisher,
    )

    try:
        # Run pipeline
        result = await orchestrator.run(request)

        # Output results
        if args.json or args.output_file:
            output = result.to_dict()

            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(output, f, indent=2)
                logger.info(f"Results written to {args.output_file}")

            if args.json:
                print(json.dumps(output, indent=2))

        else:
            # Pretty print stories
            for i, story in enumerate(result.stories):
                print_story(story, i)

            print_summary(result)

        return 0 if result.status == "completed" else 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
