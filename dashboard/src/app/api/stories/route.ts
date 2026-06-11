import { NextResponse } from "next/server";
import { db } from "@/db";
import { stories, storyEntities, storySources } from "@/db/schema";
import { desc, eq, sql } from "drizzle-orm";
import type { StoryCard, ImpactFactor, ExposureMechanism } from "@/types";
import { getUserExposures, rankStoriesByExposure } from "@/services/exposure-ranking";
import { getValidationUniverse } from "@/services/universe-cache";

type StoryRow = typeof stories.$inferSelect;

// Composite feed score: corroboration + source count, decayed by age.
// Every unmerged story carries impactScore=1, so sorting on impactScore
// alone collapsed to SQLite rowid (oldest-first) order.
function compositeScore(s: StoryRow, nowMs: number): number {
  const ageHours = Math.max(
    0,
    (nowMs - new Date(s.createdAt).getTime()) / 3_600_000
  );
  const substance = (s.sourceCount || 1) + 2 * (s.corroborationScore || 0);
  return substance / (1 + ageHours / 12);
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const limit = parseInt(url.searchParams.get("limit") || "50");
  const sector = url.searchParams.get("sector");
  const ticker = url.searchParams.get("ticker");
  const ranked = url.searchParams.get("ranked") !== "false"; // exposure-ranked by default
  const universeOnly = url.searchParams.get("universe") === "true";

  // S&P-universe filter: keep stories carrying >=1 validated in-universe
  // (or held-book) ticker entity. When the engine is unreachable we cannot
  // validate — report that via header rather than silently filtering on
  // nothing or pretending the filter applied.
  let universeFilterState: "applied" | "unavailable" | undefined;
  let universeStoryIds: Set<string> | null = null;
  if (universeOnly) {
    const universe = await getValidationUniverse();
    if (universe) {
      const tickerRows = await db.query.storyEntities.findMany({
        where: eq(storyEntities.entityType, "ticker"),
      });
      universeStoryIds = new Set(
        tickerRows
          .filter((r) => r.storyId && universe.has(r.entityValue.toUpperCase()))
          .map((r) => r.storyId as string)
      );
      universeFilterState = "applied";
    } else {
      universeFilterState = "unavailable";
    }
  }
  const responseHeaders = universeFilterState
    ? { "x-universe-filter": universeFilterState }
    : undefined;

  // Over-fetch window so post-fetch scoring/filtering still fills `limit`.
  const windowLimit = Math.max(limit * 4, 200);
  const nowMs = Date.now();

  let storyList: StoryRow[];

  if (ticker) {
    const entityRows = await db.query.storyEntities.findMany({
      where: eq(storyEntities.entityValue, ticker.toUpperCase()),
    });
    const storyIds = [...new Set(entityRows.map((e) => e.storyId))];
    if (storyIds.length === 0) {
      return NextResponse.json([], { headers: responseHeaders });
    }
    storyList = await db.query.stories.findMany({
      where: sql`${stories.storyId} IN (${sql.join(storyIds.map(id => sql`${id}`), sql`, `)})`,
      orderBy: [desc(stories.impactScore), desc(stories.createdAt)],
      limit,
    });
  } else if (sector) {
    const candidates = await db.query.stories.findMany({
      where: eq(stories.sector, sector),
      orderBy: [desc(stories.createdAt)],
      limit: windowLimit,
    });
    storyList = (
      universeStoryIds
        ? candidates.filter((s) => universeStoryIds.has(s.storyId))
        : candidates
    )
      .sort(
        (a, b) =>
          compositeScore(b, nowMs) - compositeScore(a, nowMs) ||
          b.createdAt.localeCompare(a.createdAt)
      )
      .slice(0, limit);
  } else {
    const candidates = await db.query.stories.findMany({
      orderBy: [desc(stories.createdAt)],
      limit: windowLimit,
    });
    storyList = (
      universeStoryIds
        ? candidates.filter((s) => universeStoryIds.has(s.storyId))
        : candidates
    )
      .sort(
        (a, b) =>
          compositeScore(b, nowMs) - compositeScore(a, nowMs) ||
          b.createdAt.localeCompare(a.createdAt)
      )
      .slice(0, limit);
  }

  // Enrich with entities, sources, and new intelligence fields
  const cards: StoryCard[] = await Promise.all(
    storyList.map(async (story) => {
      const entityRows = await db.query.storyEntities.findMany({
        where: eq(storyEntities.storyId, story.storyId),
      });
      // Cluster merges re-point the merged story's entity rows onto the
      // survivor, so shared tickers show up twice — dedupe for display.
      const entities = [
        ...new Map(
          entityRows.map((e) => [`${e.entityType}:${e.entityValue}`, e])
        ).values(),
      ];
      const sources = await db.query.storySources.findMany({
        where: eq(storySources.storyId, story.storyId),
      });

      // Parse JSON fields safely
      let impactTags: ImpactFactor[] = [];
      let exposureMechanisms: ExposureMechanism[] = [];
      try {
        if (story.impactTags) impactTags = JSON.parse(story.impactTags);
      } catch { /* ignore parse errors */ }
      try {
        if (story.exposureMechanisms) exposureMechanisms = JSON.parse(story.exposureMechanisms);
      } catch { /* ignore parse errors */ }

      return {
        storyId: story.storyId,
        canonicalTitle: story.canonicalTitle,
        summary: story.summary,
        impactScore: story.impactScore || 0,
        sector: story.sector,
        sourceCount: story.sourceCount || 1,
        createdAt: story.createdAt,
        entities: entities.map((e) => ({
          entityType: e.entityType as "ticker" | "person" | "org" | "topic" | "factor",
          entityValue: e.entityValue,
        })),
        sources: sources.map((s) => ({
          publisher: s.publisher || "Unknown",
          url: s.url,
          headline: s.headline,
        })),
        impactTags,
        exposureMechanisms,
        impactHorizon: story.impactHorizon as StoryCard["impactHorizon"],
        storyStatus: (story.storyStatus || "developing") as StoryCard["storyStatus"],
        contradictionFlag: story.contradictionFlag === 1,
        whyItMatters: story.whyItMatters,
        corroborationScore: story.corroborationScore || 0,
      };
    })
  );

  // Apply exposure-first ranking if enabled
  if (ranked && !ticker && !sector) {
    try {
      const exposures = await getUserExposures();
      if (exposures.length > 0) {
        return NextResponse.json(rankStoriesByExposure(cards, exposures), {
          headers: responseHeaders,
        });
      }
    } catch {
      // Fall through to unranked
    }
  }

  return NextResponse.json(cards, { headers: responseHeaders });
}
