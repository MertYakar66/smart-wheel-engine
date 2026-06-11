import { v4 as uuidv4 } from "uuid";
import { db } from "@/db";
import { alerts, storyEntities, stories } from "@/db/schema";
import { eq, and, sql } from "drizzle-orm";

// ─── News-Alert Evaluator ─────────────────────────────────────────────
// Ingest-time writer for the alerts table (it previously had none, so
// the bell badge could never be nonzero). A story raises one alert per
// watchlist symbol it mentions via a VALIDATED ticker entity. Price-move
// alerts (alertThresholdPct) remain unimplemented — surfaces must not
// imply otherwise.

export async function raiseNewsAlerts(sinceIso: string): Promise<number> {
  const watched = await db.query.watchlists.findMany();
  if (watched.length === 0) return 0;
  const watchedSet = new Set(watched.map((w) => w.ticker.toUpperCase()));

  const newStories = await db.query.stories.findMany({
    where: sql`${stories.createdAt} >= ${sinceIso}`,
  });
  if (newStories.length === 0) return 0;

  let raised = 0;
  const now = new Date().toISOString();

  for (const story of newStories) {
    const entities = await db.query.storyEntities.findMany({
      where: eq(storyEntities.storyId, story.storyId),
    });
    const hits = [
      ...new Set(
        entities
          .filter((e) => e.entityType === "ticker")
          .map((e) => e.entityValue.toUpperCase())
          .filter((t) => watchedSet.has(t))
      ),
    ];

    for (const ticker of hits) {
      // One alert per (story, ticker) — re-runs must not re-alert.
      const existing = await db.query.alerts.findFirst({
        where: and(eq(alerts.storyId, story.storyId), eq(alerts.ticker, ticker)),
      });
      if (existing) continue;

      await db.insert(alerts).values({
        alertId: uuidv4(),
        storyId: story.storyId,
        ticker,
        triggerType: "news",
        triggeredAt: now,
        dismissed: 0,
      });
      raised++;
    }
  }

  return raised;
}
