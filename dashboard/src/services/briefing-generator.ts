import { db } from "@/db";
import { briefings, stories, storyEntities, storyCategories, newsCategories } from "@/db/schema";
import { desc, sql, eq } from "drizzle-orm";
import { v4 as uuidv4 } from "uuid";
import type { ImpactFactor } from "@/types";

// ─── Briefing Generator ──────────────────────────────────────────────
// Generates structured morning/evening briefings from recent stories.
// Each briefing has sections mapped to news categories with ranked stories.

export interface BriefingSection {
  categoryName: string;
  categoryColor: string;
  categoryIcon: string;
  stories: BriefingStory[];
}

export interface BriefingStory {
  storyId: string;
  title: string;
  whyItMatters: string | null;
  impactTags: ImpactFactor[];
  tickers: string[];
  sourceCount: number;
  sector: string | null;
}

export interface Briefing {
  briefingId: string;
  briefingType: "morning" | "evening" | "breaking";
  title: string;
  sections: BriefingSection[];
  generatedAt: string;
  periodStart: string;
  periodEnd: string;
  totalStories: number;
}

// ─── Generate Briefing ────────────────────────────────────────────────

export async function generateBriefing(
  type: "morning" | "evening" | "breaking" = "morning"
): Promise<Briefing> {
  const now = new Date();
  const briefingId = uuidv4();

  // Determine time window
  let periodStart: Date;
  const periodEnd = now;

  switch (type) {
    case "morning":
      // Cover from 6 PM yesterday to 7 AM today
      periodStart = new Date(now);
      periodStart.setHours(now.getHours() - 13); // ~13 hours back
      break;
    case "evening":
      // Cover from 7 AM today to 6 PM today
      periodStart = new Date(now);
      periodStart.setHours(now.getHours() - 11); // ~11 hours back
      break;
    case "breaking":
      // Last 2 hours
      periodStart = new Date(now.getTime() - 2 * 60 * 60 * 1000);
      break;
  }

  // Get all categories
  const categories = await db.query.newsCategories.findMany({
    where: eq(newsCategories.enabled, 1),
  });

  // Get stories in the time window
  const windowStories = await db.query.stories.findMany({
    where: sql`${stories.createdAt} > ${periodStart.toISOString()} AND ${stories.createdAt} <= ${periodEnd.toISOString()}`,
    orderBy: [desc(stories.impactScore)],
    limit: 100,
  });

  const sections: BriefingSection[] = [];
  const allStoryIds: string[] = [];

  for (const cat of categories.sort((a, b) => (a.sortOrder || 0) - (b.sortOrder || 0))) {
    // Get story IDs for this category
    const catStories = await db.query.storyCategories.findMany({
      where: sql`${storyCategories.categoryId} = ${cat.categoryId} AND ${storyCategories.matchedAt} > ${periodStart.toISOString()}`,
    });

    const catStoryIds = catStories.map((cs) => cs.storyId!).filter(Boolean);
    const matchedStories = windowStories.filter((s) =>
      catStoryIds.includes(s.storyId)
    );

    if (matchedStories.length === 0) continue;

    // Get entities for each story
    const sectionStories: BriefingStory[] = [];
    for (const story of matchedStories.slice(0, 5)) { // Top 5 per section
      const entities = await db.query.storyEntities.findMany({
        where: eq(storyEntities.storyId, story.storyId),
      });

      let impactTags: ImpactFactor[] = [];
      try {
        if (story.impactTags) impactTags = JSON.parse(story.impactTags);
      } catch { /* ignore */ }

      sectionStories.push({
        storyId: story.storyId,
        title: story.canonicalTitle,
        whyItMatters: story.whyItMatters,
        impactTags,
        tickers: entities
          .filter((e) => e.entityType === "ticker")
          .map((e) => e.entityValue),
        sourceCount: story.sourceCount || 1,
        sector: story.sector,
      });

      allStoryIds.push(story.storyId);
    }

    sections.push({
      categoryName: cat.name,
      categoryColor: cat.color || "#6B7280",
      categoryIcon: cat.icon || "Newspaper",
      stories: sectionStories,
    });
  }

  // Generate title
  const dateStr = now.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
  const title = type === "morning"
    ? `Morning Brief — ${dateStr}`
    : type === "evening"
    ? `Evening Brief — ${dateStr}`
    : `Breaking News Brief — ${now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`;

  const briefing: Briefing = {
    briefingId,
    briefingType: type,
    title,
    sections,
    generatedAt: now.toISOString(),
    periodStart: periodStart.toISOString(),
    periodEnd: periodEnd.toISOString(),
    totalStories: allStoryIds.length,
  };

  // Persist the briefing
  await db.insert(briefings).values({
    briefingId,
    briefingType: type,
    title,
    content: JSON.stringify(sections),
    storyIds: JSON.stringify(allStoryIds),
    generatedAt: now.toISOString(),
    periodStart: periodStart.toISOString(),
    periodEnd: periodEnd.toISOString(),
  });

  return briefing;
}

// ─── Get Latest Briefing ──────────────────────────────────────────────

export async function getLatestBriefing(
  type?: "morning" | "evening" | "breaking"
): Promise<Briefing | null> {
  const query = type
    ? sql`${briefings.briefingType} = ${type}`
    : sql`1=1`;

  const row = await db.query.briefings.findFirst({
    where: query,
    orderBy: [desc(briefings.generatedAt)],
  });

  if (!row) return null;

  return {
    briefingId: row.briefingId,
    briefingType: row.briefingType as "morning" | "evening" | "breaking",
    title: row.title,
    sections: JSON.parse(row.content),
    generatedAt: row.generatedAt,
    periodStart: row.periodStart,
    periodEnd: row.periodEnd,
    totalStories: JSON.parse(row.storyIds).length,
  };
}

// ─── Get Briefing History ─────────────────────────────────────────────

export async function getBriefingHistory(limit: number = 14) {
  return db.query.briefings.findMany({
    orderBy: [desc(briefings.generatedAt)],
    limit,
  });
}
