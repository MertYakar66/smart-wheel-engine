import { NextResponse } from "next/server";
import { db } from "@/db";
import { stories, storyEntities, storySources, storyTimeline } from "@/db/schema";
import { eq, desc, sql } from "drizzle-orm";
import type { StoryDetail, StoryCard, ImpactFactor, ExposureMechanism, SourceRefDetailed, Sentiment } from "@/types";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  const story = await db.query.stories.findFirst({
    where: eq(stories.storyId, id),
  });

  if (!story) {
    return NextResponse.json({ error: "Story not found" }, { status: 404 });
  }

  // Get entities
  const entities = await db.query.storyEntities.findMany({
    where: eq(storyEntities.storyId, id),
  });

  // Get all sources with detailed fields
  const sources = await db.query.storySources.findMany({
    where: eq(storySources.storyId, id),
  });

  // Get timeline
  const timeline = await db.query.storyTimeline.findMany({
    where: eq(storyTimeline.storyId, id),
  });

  // Parse JSON fields
  let impactTags: ImpactFactor[] = [];
  let exposureMechanisms: ExposureMechanism[] = [];
  try { if (story.impactTags) impactTags = JSON.parse(story.impactTags); } catch {}
  try { if (story.exposureMechanisms) exposureMechanisms = JSON.parse(story.exposureMechanisms); } catch {}

  // Get related stories by shared entities
  const tickers = entities
    .filter((e) => e.entityType === "ticker")
    .map((e) => e.entityValue);

  let relatedStories: StoryCard[] = [];
  if (tickers.length > 0) {
    const relatedEntityRows = await db.query.storyEntities.findMany({
      where: sql`${storyEntities.entityValue} IN (${sql.join(tickers.map(t => sql`${t}`), sql`, `)}) AND ${storyEntities.storyId} != ${id}`,
    });
    const relatedIds = [...new Set(relatedEntityRows.map((e) => e.storyId))].slice(0, 5);

    if (relatedIds.length > 0) {
      const relatedList = await db.query.stories.findMany({
        where: sql`${stories.storyId} IN (${sql.join(relatedIds.map(rid => sql`${rid}`), sql`, `)})`,
        orderBy: [desc(stories.impactScore)],
        limit: 5,
      });

      relatedStories = await Promise.all(
        relatedList.map(async (s) => {
          const ents = await db.query.storyEntities.findMany({
            where: eq(storyEntities.storyId, s.storyId),
          });
          const srcs = await db.query.storySources.findMany({
            where: eq(storySources.storyId, s.storyId),
          });
          return {
            storyId: s.storyId,
            canonicalTitle: s.canonicalTitle,
            summary: s.summary,
            impactScore: s.impactScore || 0,
            sector: s.sector,
            sourceCount: s.sourceCount || 1,
            createdAt: s.createdAt,
            entities: ents.map((e) => ({
              entityType: e.entityType as "ticker" | "person" | "org" | "topic" | "factor",
              entityValue: e.entityValue,
            })),
            sources: srcs.map((sr) => ({
              publisher: sr.publisher || "Unknown",
              url: sr.url,
              headline: sr.headline,
            })),
          };
        })
      );
    }
  }

  const allSources: SourceRefDetailed[] = sources.map((s) => ({
    sourceId: s.sourceId,
    publisher: s.publisher || "Unknown",
    url: s.url,
    headline: s.headline,
    publishedAt: s.publishedAt,
    snippet: s.snippet,
    sentiment: (s.sentiment as Sentiment) || null,
    geography: s.geography || null,
    rightsRestricted: s.rightsRestricted === 1,
    retrievalProvider: s.retrievalProvider || "rss",
  }));

  const detail: StoryDetail = {
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
    sources: allSources.map((s) => ({
      publisher: s.publisher,
      url: s.url,
      headline: s.headline,
    })),
    impactTags,
    exposureMechanisms,
    impactHorizon: story.impactHorizon as StoryDetail["impactHorizon"],
    storyStatus: (story.storyStatus || "developing") as StoryDetail["storyStatus"],
    contradictionFlag: story.contradictionFlag === 1,
    whyItMatters: story.whyItMatters,
    corroborationScore: story.corroborationScore || 0,
    timeline: timeline.map((t) => ({
      id: t.id,
      eventType: t.eventType as "created" | "source_added" | "merged" | "status_change" | "contradiction_detected",
      description: t.description,
      metadata: t.metadata ? JSON.parse(t.metadata) : undefined,
      occurredAt: t.occurredAt,
    })),
    allSources,
    relatedStories,
  };

  return NextResponse.json(detail);
}
