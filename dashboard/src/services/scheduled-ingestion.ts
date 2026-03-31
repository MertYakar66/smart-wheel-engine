import { db } from "@/db";
import { ingestionRuns } from "@/db/schema";
import { eq, desc } from "drizzle-orm";
import { v4 as uuidv4 } from "uuid";
import { ingestAllFeeds } from "./rss-ingestion";
import { extractEntitiesForNewStories } from "./entity-extraction";
import { clusterStories } from "./story-clustering";
import { analyzeImpactForNewStories } from "./impact-analysis";
import { matchStoriesToCategories } from "./news-categories";

// ─── Scheduled Ingestion Service ──────────────────────────────────────
// Runs the full pipeline: fetch → extract → cluster → impact → categorize
// Supports morning/evening briefs and manual triggers.
//
// Schedule (recommended):
//   Morning: 06:30 ET (covers overnight news)
//   Evening: 18:30 ET (covers day's news)
//   Hot: every 15 min during market hours (optional)

export type RunType = "morning" | "evening" | "manual" | "hot";

export interface IngestionResult {
  runId: string;
  runType: RunType;
  headlinesIngested: number;
  entitiesProcessed: number;
  impactAnalyzed: number;
  storiesClustered: number;
  categoriesMatched: number;
  durationMs: number;
  status: "completed" | "failed";
  error?: string;
}

export async function runIngestionPipeline(
  runType: RunType = "manual"
): Promise<IngestionResult> {
  const runId = uuidv4();
  const startedAt = new Date().toISOString();
  const startMs = Date.now();

  // Record run start
  await db.insert(ingestionRuns).values({
    runId,
    runType,
    startedAt,
    status: "running",
  });

  try {
    // Step 1: Ingest RSS feeds
    const headlines = await ingestAllFeeds();

    // Step 2: Extract entities (tickers, people, orgs, topics)
    const entitiesProcessed = await extractEntitiesForNewStories();

    // Step 3: Analyze impact factors and generate narratives
    const impactAnalyzed = await analyzeImpactForNewStories();

    // Step 4: Cluster similar stories (dedup + contradiction detection)
    const storiesClustered = await clusterStories();

    // Step 5: Match stories to news categories
    const categoriesMatched = await matchStoriesToCategories();

    const durationMs = Date.now() - startMs;

    // Update run record
    await db
      .update(ingestionRuns)
      .set({
        completedAt: new Date().toISOString(),
        headlinesIngested: headlines.length,
        storiesClustered,
        impactAnalyzed,
        categoriesMatched,
        status: "completed",
      })
      .where(eq(ingestionRuns.runId, runId));

    return {
      runId,
      runType,
      headlinesIngested: headlines.length,
      entitiesProcessed,
      impactAnalyzed,
      storiesClustered,
      categoriesMatched,
      durationMs,
      status: "completed",
    };
  } catch (error) {
    const errMsg = error instanceof Error ? error.message : "Unknown error";

    await db
      .update(ingestionRuns)
      .set({
        completedAt: new Date().toISOString(),
        status: "failed",
        error: errMsg,
      })
      .where(eq(ingestionRuns.runId, runId));

    return {
      runId,
      runType,
      headlinesIngested: 0,
      entitiesProcessed: 0,
      impactAnalyzed: 0,
      storiesClustered: 0,
      categoriesMatched: 0,
      durationMs: Date.now() - startMs,
      status: "failed",
      error: errMsg,
    };
  }
}

// ─── Get Last Run Info ────────────────────────────────────────────────

export async function getLastRun(runType?: RunType) {
  if (runType) {
    return db.query.ingestionRuns.findFirst({
      where: eq(ingestionRuns.runType, runType),
      orderBy: [desc(ingestionRuns.startedAt)],
    });
  }
  return db.query.ingestionRuns.findFirst({
    orderBy: [desc(ingestionRuns.startedAt)],
  });
}

// ─── Determine If a Run is Needed ─────────────────────────────────────

export async function shouldRunIngestion(runType: RunType): Promise<boolean> {
  const lastRun = await getLastRun(runType);

  if (!lastRun) return true;

  const lastRunTime = new Date(lastRun.startedAt).getTime();
  const now = Date.now();

  switch (runType) {
    case "morning":
    case "evening":
      // Only run once per 10 hours for scheduled runs
      return now - lastRunTime > 10 * 60 * 60 * 1000;
    case "hot":
      // Run every 15 minutes during market hours
      return now - lastRunTime > 15 * 60 * 1000;
    case "manual":
      // Always allow manual runs, but rate-limit to 2 minutes
      return now - lastRunTime > 2 * 60 * 1000;
  }
}

// ─── Get Run History ──────────────────────────────────────────────────

export async function getRunHistory(limit: number = 20) {
  return db.query.ingestionRuns.findMany({
    orderBy: [desc(ingestionRuns.startedAt)],
    limit,
  });
}
