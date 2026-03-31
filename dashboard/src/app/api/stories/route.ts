import { NextResponse } from "next/server";
import { db } from "@/db";
import { stories, storyEntities, storySources } from "@/db/schema";
import { desc, eq, sql } from "drizzle-orm";
import type { StoryCard, ImpactFactor, ExposureMechanism } from "@/types";
import { getUserExposures, rankStoriesByExposure } from "@/services/exposure-ranking";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const limit = parseInt(url.searchParams.get("limit") || "50");
  const sector = url.searchParams.get("sector");
  const ticker = url.searchParams.get("ticker");
  const ranked = url.searchParams.get("ranked") !== "false"; // exposure-ranked by default

  let storyList;

  if (ticker) {
    const entityRows = await db.query.storyEntities.findMany({
      where: eq(storyEntities.entityValue, ticker.toUpperCase()),
    });
    const storyIds = [...new Set(entityRows.map((e) => e.storyId))];
    if (storyIds.length === 0) {
      return NextResponse.json([]);
    }
    storyList = await db.query.stories.findMany({
      where: sql`${stories.storyId} IN (${sql.join(storyIds.map(id => sql`${id}`), sql`, `)})`,
      orderBy: [desc(stories.impactScore)],
      limit,
    });
  } else if (sector) {
    storyList = await db.query.stories.findMany({
      where: eq(stories.sector, sector),
      orderBy: [desc(stories.impactScore)],
      limit,
    });
  } else {
    storyList = await db.query.stories.findMany({
      orderBy: [desc(stories.impactScore)],
      limit,
    });
  }

  // Enrich with entities, sources, and new intelligence fields
  const cards: StoryCard[] = await Promise.all(
    storyList.map(async (story) => {
      const entities = await db.query.storyEntities.findMany({
        where: eq(storyEntities.storyId, story.storyId),
      });
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
        return NextResponse.json(rankStoriesByExposure(cards, exposures));
      }
    } catch {
      // Fall through to unranked
    }
  }

  return NextResponse.json(cards);
}
