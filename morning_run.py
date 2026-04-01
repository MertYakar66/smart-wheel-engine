#!/usr/bin/env python3
"""
Morning News Pipeline Runner

Usage:
    python morning_run.py                    # Run with defaults (overnight news)
    python morning_run.py --time-window 6h   # Last 6 hours
    python morning_run.py --tickers AAPL,NVDA,TSLA
    python morning_run.py --categories earnings,fed
    python morning_run.py --dry-run          # Don't publish, just show results
    python morning_run.py --json             # Output as JSON

Example morning workflow:
    $ python morning_run.py --time-window overnight --min-confidence 7

This will:
1. Open Grok AI to discover overnight financial news
2. Send discoveries to Gemini for cross-source verification
3. Forward verified stories to ChatGPT for formatting
4. Polish with Claude to add "why it matters"
5. Publish to the website feed
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime

from news_pipeline import DiscoveryRequest, NewsPipelineOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the morning news pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers to monitor (default: major indices)",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default="sp500_events,fed,earnings,oil,geopolitics",
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
        "--max-results",
        type=int,
        default=50,
        help="Maximum stories to discover (default: 50)",
    )

    parser.add_argument(
        "--min-confidence",
        type=int,
        default=6,
        choices=range(0, 11),
        metavar="[0-10]",
        help="Minimum verification confidence to publish (default: 6)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't publish, just show what would be published",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Just check provider health and exit",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def mock_publisher(feed_item) -> bool:
    """Mock publisher for dry-run mode."""
    logger.info(f"[DRY-RUN] Would publish: {feed_item.title}")
    return True


async def web_publisher(feed_item) -> bool:
    """
    Publish feed item to the website.

    TODO: Implement actual website publishing logic.
    """
    logger.info(f"[PUBLISH] Publishing: {feed_item.title}")
    # Placeholder for actual publishing logic
    # Could POST to API, write to database, generate static file, etc.
    return True


def print_story(story, index: int) -> None:
    """Pretty-print a finalized story."""
    print(f"\n{'='*60}")
    print(f"STORY #{index + 1}")
    print(f"{'='*60}")
    print(f"TITLE: {story.title}")
    print(f"CONFIDENCE: {story.verification_confidence}/10")
    print(f"ASSETS: {', '.join(story.affected_assets)}")
    print(f"CATEGORY: {story.category}")
    print()
    print("WHAT HAPPENED:")
    print(f"  {story.what_happened}")
    print()
    print("WHY IT MATTERS:")
    print(f"  {story.why_it_matters}")
    print()
    print("KEY POINTS:")
    for bp in story.bullet_points:
        print(f"  • {bp}")


def print_summary(run) -> None:
    """Print pipeline run summary."""
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Run ID:      {run.run_id}")
    print(f"Duration:    {(run.completed_at - run.started_at).total_seconds():.1f}s")
    print(f"Discovered:  {run.discovered_count} candidates")
    print(f"Verified:    {run.verified_count} passed threshold")
    print(f"Published:   {run.published_count} stories")
    print()


async def run_health_check() -> bool:
    """Run health check on all providers."""
    print("Checking provider health...")
    orchestrator = NewsPipelineOrchestrator()

    try:
        await orchestrator.initialize()
        health = await orchestrator.health_check()

        all_healthy = True
        for provider, status in health.items():
            icon = "✓" if status else "✗"
            print(f"  {icon} {provider.capitalize()}: {'OK' if status else 'FAILED'}")
            if not status:
                all_healthy = False

        return all_healthy

    except Exception as e:
        print(f"Health check error: {e}")
        return False


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Health check mode
    if args.health_check:
        healthy = await run_health_check()
        return 0 if healthy else 1

    # Parse tickers and categories
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

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
        max_results=args.max_results,
    )

    # Set up publisher
    publisher = mock_publisher if args.dry_run else web_publisher

    # Create orchestrator
    orchestrator = NewsPipelineOrchestrator(
        min_confidence=args.min_confidence,
        publish_callback=publisher,
    )

    print(f"\n🌅 Morning News Pipeline - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Time window: {args.time_window}")
    print(f"Categories: {', '.join(categories)}")
    if tickers:
        print(f"Tickers: {', '.join(tickers)}")
    print(f"Min confidence: {args.min_confidence}/10")
    if args.dry_run:
        print("Mode: DRY RUN (no publishing)")
    print()

    try:
        # Run the pipeline
        run = await orchestrator.run_pipeline(request)

        # Output results
        if args.json:
            output = {
                "run_id": run.run_id,
                "started_at": run.started_at.isoformat(),
                "completed_at": run.completed_at.isoformat(),
                "discovered_count": run.discovered_count,
                "verified_count": run.verified_count,
                "published_count": run.published_count,
                "stories": [
                    {
                        "story_id": s.story_id,
                        "title": s.title,
                        "what_happened": s.what_happened,
                        "why_it_matters": s.why_it_matters,
                        "bullet_points": s.bullet_points,
                        "affected_assets": s.affected_assets,
                        "category": s.category,
                        "confidence": s.verification_confidence,
                    }
                    for s in run.stories
                ],
            }
            print(json.dumps(output, indent=2))
        else:
            # Pretty print stories
            for i, story in enumerate(run.stories):
                print_story(story, i)

            print_summary(run)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
