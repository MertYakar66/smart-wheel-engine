"""
Event-Aware Scheduler for Macro + SP500 Event Intelligence System

Implements:
- AM batch (05:30 ET): Overnight ingestion
- PM batch (17:30 ET): Post-close ingestion
- Event-aware mini-runs: CPI, NFP, FOMC, etc.

The scheduler reads from event_calendar to trigger runs,
not hardcoded cron logic.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any

try:
    import pytz
except ImportError:
    pytz = None

from financial_news.calendar import MacroCalendar
from financial_news.connectors import EIAConnector, FedConnector, SECEdgarConnector
from financial_news.schema import (
    ImportanceLevel,
    RunLog,
    RunStatus,
    ScheduledEvent,
)
from financial_news.storage import NewsDatabase

logger = logging.getLogger(__name__)


class JobType(Enum):
    """Types of scheduled jobs"""

    MORNING_BATCH = "morning_batch"  # 05:30 ET
    EVENING_BATCH = "evening_batch"  # 17:30 ET
    PRE_EVENT = "pre_event"  # Before major release
    POST_EVENT = "post_event"  # After major release
    MANUAL = "manual"  # Manual trigger


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""

    timezone: str = "America/New_York"

    # Core batch times (ET)
    morning_batch_time: time = field(default_factory=lambda: time(5, 30))
    evening_batch_time: time = field(default_factory=lambda: time(17, 30))

    # Event-aware triggers
    enable_event_triggers: bool = True
    pre_event_minutes: int = 10  # Run this many minutes before release
    post_event_minutes: int = 5  # Run this many minutes after release

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60

    # Source limits
    hours_lookback_morning: int = 14  # Look back to previous evening
    hours_lookback_evening: int = 12  # Look back to morning
    max_articles_per_source: int = 100


class Scheduler:
    """
    Event-aware scheduler for the news intelligence system.

    Core schedule (ET):
    - 05:30: Morning batch (overnight + premarket)
    - 17:30: Evening batch (post-close)

    Event-aware mini-runs:
    - CPI days: 08:20 (pre), 08:35 (post)
    - NFP days: 08:20 (pre), 08:35 (post)
    - FOMC days: 13:50 (pre), 14:05 (post)
    - EIA Petroleum: 10:25 (pre), 10:35 (post)
    """

    def __init__(
        self,
        db: NewsDatabase,
        config: SchedulerConfig | None = None,
    ):
        self.db = db
        self.config = config or SchedulerConfig()

        # Calendar for event lookup
        self.calendar = MacroCalendar()

        # Running state
        self._running = False
        self._current_job: str | None = None
        self._last_run: dict[str, datetime] = {}

        # Connectors (created on demand)
        self._connectors: dict[str, Any] = {}

    async def _get_connector(self, source_id: str):
        """Get or create a connector for a source."""
        if source_id not in self._connectors:
            if source_id == "fed":
                self._connectors[source_id] = FedConnector()
            elif source_id == "sec_edgar":
                self._connectors[source_id] = SECEdgarConnector()
            elif source_id == "eia":
                self._connectors[source_id] = EIAConnector()
            else:
                return None
        return self._connectors[source_id]

    async def run_batch(
        self,
        job_type: JobType,
        triggered_by: str = "scheduler",
        event: ScheduledEvent | None = None,
    ) -> RunLog:
        """
        Run a batch ingestion job.

        Args:
            job_type: Type of job (morning, evening, event)
            triggered_by: What triggered this run
            event: Associated event if event-triggered

        Returns:
            RunLog with job results
        """
        run_id = str(uuid.uuid4())[:8]
        job_name = job_type.value
        if event:
            job_name = f"{job_type.value}_{event.event_type.value}"

        # Create run log
        run_log = self.db.create_run_log(
            run_id=run_id,
            job_name=job_name,
            triggered_by=triggered_by,
            event_id=event.event_id if event else None,
        )

        logger.info(f"Starting {job_name} run [{run_id}]")

        try:
            # Determine lookback based on job type
            if job_type == JobType.MORNING_BATCH:
                hours = self.config.hours_lookback_morning
            elif job_type == JobType.EVENING_BATCH:
                hours = self.config.hours_lookback_evening
            else:
                hours = 2  # Short lookback for event runs

            since = datetime.utcnow() - timedelta(hours=hours)

            # Fetch from all active sources
            all_articles = []
            source_stats = {}

            # Determine which sources to fetch based on job type
            sources_to_fetch = self._get_sources_for_job(job_type, event)

            for source_id in sources_to_fetch:
                try:
                    connector = await self._get_connector(source_id)
                    if not connector:
                        continue

                    articles = await connector.fetch_latest(
                        since=since,
                        limit=self.config.max_articles_per_source,
                    )

                    # Save articles
                    saved = self.db.save_articles(articles)
                    all_articles.extend(articles)

                    source_stats[source_id] = {
                        "fetched": len(articles),
                        "saved": saved,
                        "status": "success",
                    }

                    # Update source status
                    self.db.update_source_fetch_status(source_id, success=True)

                    logger.info(f"  {source_id}: {len(articles)} articles")

                except Exception as e:
                    logger.error(f"Error fetching from {source_id}: {e}")
                    source_stats[source_id] = {
                        "status": "error",
                        "error": str(e),
                    }
                    self.db.update_source_fetch_status(source_id, success=False)

            # Update run log
            run_log.items_fetched = len(all_articles)
            run_log.items_processed = len(all_articles)
            run_log.source_stats = source_stats
            run_log.status = RunStatus.SUCCESS
            run_log.ended_at = datetime.utcnow()

            self.db.update_run_log(run_log)
            self._last_run[job_name] = datetime.utcnow()

            logger.info(
                f"Completed {job_name}: {len(all_articles)} articles from {len(sources_to_fetch)} sources"
            )

        except Exception as e:
            logger.error(f"Batch run failed: {e}")
            run_log.status = RunStatus.FAILED
            run_log.errors.append({"error": str(e)})
            run_log.ended_at = datetime.utcnow()
            self.db.update_run_log(run_log)

        return run_log

    def _get_sources_for_job(
        self,
        job_type: JobType,
        event: ScheduledEvent | None,
    ) -> list[str]:
        """Determine which sources to fetch for a job."""
        # For event runs, focus on the relevant source
        if event:
            sources = [event.source_id]
            # Add related sources
            if event.source_id in ["fed"]:
                sources.append("treasury")
            elif event.source_id in ["bls", "bea"]:
                sources.append("fed")  # Fed reaction
            return sources

        # For batch runs, fetch from all active sources
        active_sources = self.db.get_all_sources(active_only=True)
        return [s.source_id for s in active_sources if s.source_id in ["fed", "sec_edgar", "eia"]]

    async def run_morning_batch(self) -> RunLog:
        """Run morning batch (05:30 ET)."""
        return await self.run_batch(JobType.MORNING_BATCH, triggered_by="scheduler")

    async def run_evening_batch(self) -> RunLog:
        """Run evening batch (17:30 ET)."""
        return await self.run_batch(JobType.EVENING_BATCH, triggered_by="scheduler")

    async def run_event_batch(self, event: ScheduledEvent, pre_or_post: str) -> RunLog:
        """Run event-triggered batch."""
        job_type = JobType.PRE_EVENT if pre_or_post == "pre" else JobType.POST_EVENT
        return await self.run_batch(job_type, triggered_by="event", event=event)

    def get_upcoming_runs(self, hours: int = 24) -> list[dict[str, Any]]:
        """
        Get upcoming scheduled runs for the next N hours.

        Returns list of scheduled runs including:
        - Morning/evening batches
        - Event-triggered runs
        """
        runs = []
        now = datetime.utcnow()

        # Get timezone
        if pytz:
            tz = pytz.timezone(self.config.timezone)
            now_local = datetime.now(tz)
        else:
            now_local = now  # Fallback to UTC

        # Check for morning batch today
        morning_time = datetime.combine(now_local.date(), self.config.morning_batch_time)
        if morning_time > now_local:
            runs.append(
                {
                    "type": "morning_batch",
                    "scheduled_at": morning_time,
                    "description": "Morning batch - overnight + premarket",
                }
            )

        # Check for evening batch today
        evening_time = datetime.combine(now_local.date(), self.config.evening_batch_time)
        if evening_time > now_local:
            runs.append(
                {
                    "type": "evening_batch",
                    "scheduled_at": evening_time,
                    "description": "Evening batch - post-close",
                }
            )

        # Get event-triggered runs
        if self.config.enable_event_triggers:
            upcoming_events = self.db.get_upcoming_events(hours=hours)
            for event in upcoming_events:
                if event.importance in [ImportanceLevel.CRITICAL, ImportanceLevel.HIGH]:
                    # Pre-event run
                    pre_time = event.scheduled_at - timedelta(minutes=event.pre_run_offset_minutes)
                    if pre_time > now:
                        runs.append(
                            {
                                "type": "pre_event",
                                "scheduled_at": pre_time,
                                "event": event.title,
                                "description": f"Pre-{event.event_type.value} run",
                            }
                        )

                    # Post-event run
                    post_time = event.scheduled_at + timedelta(
                        minutes=event.post_run_offset_minutes
                    )
                    if post_time > now:
                        runs.append(
                            {
                                "type": "post_event",
                                "scheduled_at": post_time,
                                "event": event.title,
                                "description": f"Post-{event.event_type.value} run",
                            }
                        )

        # Sort by time
        runs.sort(key=lambda r: r["scheduled_at"])
        return runs

    async def start(self) -> None:
        """
        Start the scheduler loop.

        Checks every minute for:
        - Morning/evening batch times
        - Event pre/post triggers
        """
        self._running = True
        logger.info(f"Scheduler started (timezone: {self.config.timezone})")

        while self._running:
            try:
                now = datetime.utcnow()

                # Get current time in configured timezone
                if pytz:
                    tz = pytz.timezone(self.config.timezone)
                    now_local = datetime.now(tz)
                else:
                    now_local = now

                current_time = now_local.time()

                # Check for morning batch
                if (
                    current_time.hour == self.config.morning_batch_time.hour
                    and current_time.minute == self.config.morning_batch_time.minute
                ):
                    if self._should_run("morning_batch"):
                        await self.run_morning_batch()

                # Check for evening batch
                if (
                    current_time.hour == self.config.evening_batch_time.hour
                    and current_time.minute == self.config.evening_batch_time.minute
                ):
                    if self._should_run("evening_batch"):
                        await self.run_evening_batch()

                # Check for event triggers
                if self.config.enable_event_triggers:
                    await self._check_event_triggers(now)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")

            # Sleep until next minute
            await asyncio.sleep(60)

    def _should_run(self, job_name: str) -> bool:
        """Check if a job should run (prevent double-runs)."""
        last = self._last_run.get(job_name)
        if last is None:
            return True
        # Don't run if ran in last 30 minutes
        return (datetime.utcnow() - last).total_seconds() > 1800

    async def _check_event_triggers(self, now: datetime) -> None:
        """Check for event-triggered runs."""
        # Check pre-event runs
        pre_events = self.db.get_events_needing_prerun(minutes_ahead=15)
        for event in pre_events:
            job_name = f"pre_event_{event.event_id}"
            if self._should_run(job_name):
                logger.info(f"Triggering pre-event run for {event.title}")
                await self.run_event_batch(event, "pre")
                self._last_run[job_name] = now

        # Check post-event runs
        post_events = self.db.get_events_needing_postrun(minutes_since=30)
        for event in post_events:
            job_name = f"post_event_{event.event_id}"
            if self._should_run(job_name):
                logger.info(f"Triggering post-event run for {event.title}")
                await self.run_event_batch(event, "post")
                self._last_run[job_name] = now

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("Scheduler stopped")

    async def close(self) -> None:
        """Close all connectors."""
        for connector in self._connectors.values():
            await connector.close()
        self._connectors.clear()

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "timezone": self.config.timezone,
            "last_runs": {k: v.isoformat() for k, v in self._last_run.items()},
            "upcoming_runs": len(self.get_upcoming_runs(hours=24)),
            "active_connectors": list(self._connectors.keys()),
        }


# =============================================================================
# CLI Interface
# =============================================================================


async def main():
    """CLI entry point for scheduler."""
    import argparse

    parser = argparse.ArgumentParser(description="News Intelligence Scheduler")
    parser.add_argument("--run-morning", action="store_true", help="Run morning batch now")
    parser.add_argument("--run-evening", action="store_true", help="Run evening batch now")
    parser.add_argument("--start", action="store_true", help="Start scheduler loop")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--upcoming", action="store_true", help="Show upcoming runs")
    parser.add_argument("--db-path", default="data/news_intel.db", help="Database path")

    args = parser.parse_args()

    # Initialize
    db = NewsDatabase(args.db_path)

    # Populate calendar if empty
    calendar = MacroCalendar()
    calendar.populate_database(db)

    scheduler = Scheduler(db)

    try:
        if args.run_morning:
            run_log = await scheduler.run_morning_batch()
            print(f"Morning batch complete: {run_log.items_fetched} articles")

        elif args.run_evening:
            run_log = await scheduler.run_evening_batch()
            print(f"Evening batch complete: {run_log.items_fetched} articles")

        elif args.start:
            print("Starting scheduler (Ctrl+C to stop)...")
            await scheduler.start()

        elif args.status:
            status = scheduler.get_status()
            print("Scheduler Status:")
            print(f"  Running: {status['running']}")
            print(f"  Timezone: {status['timezone']}")
            print(f"  Last runs: {status['last_runs']}")

        elif args.upcoming:
            runs = scheduler.get_upcoming_runs(hours=24)
            print(f"Upcoming runs (next 24 hours): {len(runs)}")
            for run in runs[:10]:
                print(f"  {run['scheduled_at']}: {run['type']} - {run.get('description', '')}")

        else:
            parser.print_help()

    finally:
        await scheduler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
