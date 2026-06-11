import cron from "node-cron";
import { db } from "@/db";
import { storyEntities } from "@/db/schema";
import { eq } from "drizzle-orm";
import {
  runIngestionPipeline,
  shouldRunIngestion,
  getLastRun,
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
    // For the hot tick the cron fires exactly every 15 min, but startedAt is
    // recorded a DB round-trip after the tick — so the next tick's elapsed
    // measurement lands at ~14:59.9 s and shouldRunIngestion's strict
    // "> 15 min" check skips it, effectively halving the cadence to ~30 min.
    // Use a 14-min tolerance for hot runs; noOverlap already prevents overlap.
    if (runType === "hot") {
      const lastRun = await getLastRun("hot");
      if (
        lastRun &&
        Date.now() - new Date(lastRun.startedAt).getTime() <
          14 * 60 * 1000
      ) {
        return;
      }
    } else if (!(await shouldRunIngestion(runType))) {
      return;
    }
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
  const result = await getValidationUniverse();
  // null  → engine unreachable; cannot validate — leave all rows untouched.
  // !complete → positions leg failed (e.g. 503 on missing IBKR snapshot,
  //   the normal state on a fresh deploy); we can promote but must not demote
  //   because the set does not include off-universe held-book names (CLS etc.).
  if (!result) return;
  const { set: universe, complete } = result;

  // ── Re-promotion pass (runs regardless of completeness) ───────────────
  // topic rows that were written while the universe was unavailable (or that
  // were previously demoted by a partial-set run) get re-promoted to ticker
  // if the universe now confirms them.
  //
  // PROVENANCE GUARD (PR #403 Opus audit): only rows already stored in
  // canonical ALL-UPPERCASE qualify. Ticker candidates are always written
  // uppercase (entity-extraction.ts), while genuine LLM topics are free-form
  // — without this guard a topic like "all"/"key"/"now" would uppercase-
  // collide with a real S&P symbol (ALL/KEY/NOW) and be silently rewritten
  // into a phantom ticker, destroying the original topic string. The schema
  // has no provenance column, so casing IS the provenance signal; rows
  // lowercased by the pre-fix demotion path are deliberately left as topics
  // (safe direction). Residual risk: an LLM topic emitted as all-caps that
  // equals a symbol (e.g. "IT"/Gartner) — acceptable in the news layer; a
  // real entitySource column is the full fix (follow-up).
  const topicRows = await db.query.storyEntities.findMany({
    where: eq(storyEntities.entityType, "topic"),
  });
  let promoted = 0;
  for (const row of topicRows) {
    const value = row.entityValue;
    const isCanonicalTickerForm = value === value.toUpperCase();
    if (isCanonicalTickerForm && universe.has(value)) {
      await db
        .update(storyEntities)
        .set({ entityType: "ticker" })
        .where(eq(storyEntities.id, row.id));
      promoted++;
    }
  }

  // ── Demotion pass (only when the validation set is COMPLETE) ──────────
  // A complete set means BOTH the engine universe AND the held-book positions
  // were fetched successfully, so absence from the set proves the symbol is
  // neither tradeable nor held — safe to demote.
  let demoted = 0;
  if (complete) {
    const tickerRows = await db.query.storyEntities.findMany({
      where: eq(storyEntities.entityType, "ticker"),
    });
    for (const row of tickerRows) {
      if (!universe.has(row.entityValue.toUpperCase())) {
        // Preserve the original casing in entityValue so re-promotion on the
        // next complete-set backfill can round-trip without information loss.
        await db
          .update(storyEntities)
          .set({ entityType: "topic" })
          .where(eq(storyEntities.id, row.id));
        demoted++;
      }
    }
  }

  if (promoted > 0 || demoted > 0) {
    console.log(
      `[news-cron] ticker-entity backfill: ` +
        `promoted ${promoted}, demoted ${demoted} ` +
        `(complete-set=${complete})`
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

  // Backfill: convergence pass over all entityType rows.
  //   Re-promotion — topic rows whose value is in the universe get promoted
  //     to ticker (heals bad demotions and rows written during outages).
  //   Demotion — only when the validation set is COMPLETE (both universe +
  //     positions legs succeeded); absent symbols are truly not tradeable and
  //     not held.  Skipped on partial sets to protect off-universe held names.
  //   No-op when engine is entirely unreachable (cannot validate rows).
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
