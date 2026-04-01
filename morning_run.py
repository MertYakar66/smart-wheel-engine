#!/usr/bin/env python3
"""
Morning News Pipeline Runner

Browser-based multi-model pipeline - ZERO API COST.
Uses your paid subscriptions (Claude Pro, ChatGPT Plus, Gemini Advanced).

Usage:
    # First time: authenticate browser sessions
    python morning_run.py --setup-auth

    # Daily run
    python morning_run.py

    # Custom options
    python morning_run.py --time-window 6h --min-confidence 7
    python morning_run.py --tickers AAPL,NVDA --categories fed,earnings
    python morning_run.py --dry-run  # Scrape only, no browser
    python morning_run.py --json     # Output as JSON

Pipeline Flow:
    1. SCRAPE: RSS feeds from Bloomberg, Reuters, Fed, SEC
    2. PREPROCESS: Local LLM filters and categorizes
    3. VERIFY: Claude (browser) verifies with web search
    4. FORMAT: ChatGPT (browser) structures content
    5. EDITORIAL: Claude (browser) adds "why it matters"
    6. PUBLISH: Save to database, ready for dashboard

Cost: $0 (uses existing subscriptions, not APIs)
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

from news_pipeline.browser_agents import (
    ChatGPTBrowserAgent,
    ClaudeBrowserAgent,
    GeminiBrowserAgent,
)
from news_pipeline.models import DiscoveryRequest
from news_pipeline.orchestrator import NewsPipelineOrchestrator, OrchestratorConfig
from news_pipeline.scrapers import NewsAggregator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Morning News Pipeline - Browser-based, zero API cost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Discovery options
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers to monitor",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="fed,earnings,sp500_events,oil,geopolitics",
        help="Comma-separated categories",
    )
    parser.add_argument(
        "--time-window",
        type=str,
        default="overnight",
        choices=["1h", "3h", "6h", "overnight", "today"],
        help="Time window for news (default: overnight)",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=20,
        help="Maximum stories to process (default: 20)",
    )

    # Verification
    parser.add_argument(
        "--min-confidence",
        type=int,
        default=6,
        choices=range(0, 11),
        metavar="[0-10]",
        help="Minimum verification confidence (default: 6)",
    )

    # Modes
    parser.add_argument(
        "--setup-auth",
        action="store_true",
        help="Setup browser authentication (interactive)",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Only scrape news, skip verification",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scrape and preprocess only, no browser sessions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Write output to file",
    )

    # Browser options
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser windows (for debugging)",
    )

    # Local LLM
    parser.add_argument(
        "--no-local-llm",
        action="store_true",
        help="Disable local LLM preprocessing",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model for preprocessing",
    )

    # General
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def print_header(args: argparse.Namespace) -> None:
    """Print pipeline header."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'=' * 60}")
    print(f"  MORNING NEWS PIPELINE - {now}")
    print("  Browser-Based | Zero API Cost")
    print(f"{'=' * 60}")
    print(f"  Time window:    {args.time_window}")
    print(f"  Categories:     {args.categories}")
    if args.tickers:
        print(f"  Tickers:        {args.tickers}")
    print(f"  Min confidence: {args.min_confidence}/10")
    if args.dry_run:
        print("  Mode:           DRY RUN (no browser)")
    if args.scrape_only:
        print("  Mode:           SCRAPE ONLY")
    print(f"{'=' * 60}\n")


def print_story(story, index: int) -> None:
    """Pretty-print a finalized story."""
    print(f"\n{'─' * 60}")
    print(
        f"STORY #{index + 1} | Priority: {story.priority}/10 | "
        f"Confidence: {story.verification_confidence}/10"
    )
    if story.is_breaking:
        print("BREAKING NEWS")
    print(f"{'─' * 60}")

    print(f"\n  {story.title}")
    print(f"\n  Category: {story.category.upper()}")
    print(f"  Assets:   {', '.join(story.affected_assets)}")

    print("\n  WHAT HAPPENED:")
    print(f"  {story.what_happened}")

    print("\n  WHY IT MATTERS:")
    print(f"  {story.why_it_matters}")

    if story.market_implications:
        print("\n  MARKET IMPLICATIONS:")
        print(f"  {story.market_implications}")

    print("\n  KEY POINTS:")
    for bp in story.bullet_points:
        print(f"  - {bp}")


def print_summary(result) -> None:
    """Print pipeline summary."""
    print(f"\n{'=' * 60}")
    print("  PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Run ID:       {result.run_id}")
    print(f"  Duration:     {result.duration_seconds:.1f}s")
    print(f"  Status:       {result.status.upper()}")
    print(f"{'─' * 60}")
    print(f"  Scraped:      {result.discovered_count} items")
    print(f"  Verified:     {result.verified_count} passed")
    print(f"  Formatted:    {result.formatted_count} stories")
    print(f"  Published:    {result.published_count} stories")
    print(f"{'=' * 60}\n")


async def setup_authentication() -> None:
    """Interactive setup for browser authentication."""
    print("\n" + "=" * 60)
    print("  BROWSER AUTHENTICATION SETUP")
    print("=" * 60)
    print("""
This will open browser windows for you to log into:
1. Claude (claude.ai)
2. ChatGPT (chat.openai.com)
3. Gemini (gemini.google.com)

Your login state will be saved for future runs.
Press Ctrl+C to skip any service.
""")

    agents = [
        ("Claude", ClaudeBrowserAgent),
        ("ChatGPT", ChatGPTBrowserAgent),
        ("Gemini", GeminiBrowserAgent),
    ]

    for name, AgentClass in agents:
        try:
            print(f"\n[{name}] Opening browser...")
            agent = AgentClass(headless=False)  # Visible for login
            await agent.initialize()

            if agent.is_authenticated:
                print(f"[{name}] Already authenticated!")
            else:
                print(f"[{name}] Please log in, then press Enter...")
                input()

                # Check again
                if await agent._check_authenticated():
                    print(f"[{name}] Authentication saved!")
                else:
                    print(f"[{name}] Authentication failed. Try again later.")

            await agent.close()

        except KeyboardInterrupt:
            print(f"\n[{name}] Skipped")
        except Exception as e:
            print(f"[{name}] Error: {e}")

    print("\nSetup complete! Run the pipeline with: python morning_run.py\n")


async def scrape_only_mode(args: argparse.Namespace) -> int:
    """Run scrape-only mode."""
    print("\nScraping news from RSS feeds...\n")

    aggregator = NewsAggregator(use_rss=True, use_browser=False)

    hours_map = {
        "overnight": 12,
        "1h": 1,
        "3h": 3,
        "6h": 6,
        "today": 24,
    }

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    items = await aggregator.fetch_news(
        categories=categories,
        tickers=tickers or None,
        max_items=args.max_stories,
        hours_back=hours_map.get(args.time_window, 12),
    )

    if args.json:
        output = [item.to_dict() for item in items]
        print(json.dumps(output, indent=2))
    else:
        print(f"Scraped {len(items)} news items:\n")
        for i, item in enumerate(items, 1):
            print(f"{i:2}. [{item.source_name}] {item.headline}")
            if item.tickers:
                print(f"    Tickers: {', '.join(item.tickers)}")
            print()

    await aggregator.close()
    return 0


async def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Setup auth mode
    if args.setup_auth:
        await setup_authentication()
        return 0

    # Scrape only mode
    if args.scrape_only or args.dry_run:
        return await scrape_only_mode(args)

    # Print header
    if not args.json:
        print_header(args)

    # Configure orchestrator
    config = OrchestratorConfig(
        min_confidence=args.min_confidence,
        max_stories_per_run=args.max_stories,
        headless=not args.visible,
        use_local_llm=not args.no_local_llm,
        ollama_model=args.ollama_model,
    )

    # Parse request
    tickers = tuple(t.strip().upper() for t in args.tickers.split(",") if t.strip())
    categories = tuple(c.strip().lower() for c in args.categories.split(",") if c.strip())

    time_window_map = {
        "1h": "last_1h",
        "3h": "last_3h",
        "6h": "last_6h",
        "overnight": "overnight",
        "today": "today",
    }

    request = DiscoveryRequest(
        tickers=tickers,
        categories=categories,
        time_window=time_window_map.get(args.time_window, "overnight"),
        max_results=args.max_stories,
    )

    # Run pipeline
    orchestrator = NewsPipelineOrchestrator(config=config)

    try:
        result = await orchestrator.run(request)

        # Output
        if args.json:
            print(result.to_json())
        else:
            for i, story in enumerate(result.stories):
                print_story(story, i)
            print_summary(result)

        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(result.to_json())
            logger.info(f"Output saved to {args.output_file}")

        return 0 if result.status == "completed" else 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted")
        return 130

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    finally:
        await orchestrator.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
