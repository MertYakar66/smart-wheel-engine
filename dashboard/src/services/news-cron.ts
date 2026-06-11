import cron from "node-cron";
import { db } from "@/db";
import { storyEntities } from "@/db/schema";
import { eq } from "drizzle-orm";
import {
  runIngestionPipeline,
  shouldRunIngestion,
  type RunType,
} from "./scheduled-ingestion";
import { generateBriefing } from "./briefing-generator";
import { initializeDefaultCategories } from "./news-categories";
import { getValidationUniverse } from "./universe-cache";

// ─── Scheduled News Ingestion ─────────────────────────────────────────
// Started once per server boot from src/instrumentation.ts (Next.js
// instrumentation hook, nodejs runtime only). Schedule mirrors the
// contract documented in scheduled-ingestion.ts:
//   Morning 06:30 ET, Evening 18:30 ET, Hot every 15 min 9-16 ET weekdays.
// Every job re-checks shouldRunIngestion() so restarts can't double-run,
// and noOverlap stops a slow run from stacking on the next tick.
// Opt out with NEWS_CRON=0.

// globalThis guard: module-level state does not survive dev-mode module
// re-evaluation, but the timers do — without this, hot reload would
// accumulate duplicate cron jobs.
const g = globalThis as typeof globalThis & { __newsCronStarted?: boolean };

async function runScheduled(runType: RunType, withBriefing: boolean) {
  try {
    if (!(await shouldRunIngestion(runType))) return;
    const result = await runIngestionPipeline(runType);
    console.log(
      `[news-cron] ${runType} run ${result.status}: ` +
        `${result.headlinesIngested} headlines, ${result.storiesClustered} merges, ` +
        `${result.alertsRaised} alerts (${result.durationMs}ms)`
    );
    if (withBriefing && result.status === "completed") {
      const briefing = await generateBriefing(
        runType === "morning" ? "morning" : "evening"
      );
      console.log(
        `[news-cron] generated ${briefing.briefingType} briefing (${briefing.totalStories} stories)`
      );
    }
  } catch (err) {
    console.error(`[news-cron] ${runType} run failed:`, err);
  }
}

async function backfillTickerEntities(): Promise<void> {
  const universe = await getValidationUniverse();
  if (!universe) return;
  const rows = await db.query.storyEntities.findMany({
    where: eq(storyEntities.entityType, "ticker"),
  });
  let demoted = 0;
  for (const row of rows) {
    if (!universe.has(row.entityValue.toUpperCase())) {
      await db
        .update(storyEntities)
        .set({ entityType: "topic", entityValue: row.entityValue.toLowerCase() })
        .where(eq(storyEntities.id, row.id));
      demoted++;
    }
  }
  if (demoted > 0) {
    console.log(
      `[news-cron] demoted ${demoted} non-universe ticker entities to topics`
    );
  }
}

export function startNewsCron(): void {
  if (g.__newsCronStarted) return;
  if (process.env.NEWS_CRON === "0") {
    console.log("[news-cron] disabled via NEWS_CRON=0");
    return;
  }
  g.__newsCronStarted = true;

  // One-time seeding moved here from the /api/categories GET handler
  // (GETs must not mutate the DB). Fire-and-forget: a failure only delays
  // seeding until the first ingestion run, which also self-seeds.
  initializeDefaultCategories().catch((err) =>
    console.error("[news-cron] category seeding failed:", err)
  );

  // Backfill: rows written before universe validation existed can carry
  // regex junk ("FIFA", "NASA") as entityType=ticker. Demote anything
  // outside universe+book to topic. Idempotent; skipped entirely when the
  // engine is unreachable (cannot validate -> don't touch rows).
  backfillTickerEntities().catch((err) =>
    console.error("[news-cron] ticker-entity backfill failed:", err)
  );

  const tz = { timezone: "America/New_York", noOverlap: true };
  cron.schedule("30 6 * * *", () => runScheduled("morning", true), {
    ...tz,
    name: "news-morning",
  });
  cron.schedule("30 18 * * *", () => runScheduled("evening", true), {
    ...tz,
    name: "news-evening",
  });
  cron.schedule("*/15 9-16 * * 1-5", () => runScheduled("hot", false), {
    ...tz,
    name: "news-hot",
  });

  console.log(
    "[news-cron] scheduled: morning 06:30 ET, evening 18:30 ET, hot */15 9-16 ET weekdays"
  );
}
