import { NextResponse } from "next/server";
import { db } from "@/db";
import { newsCategories, storyCategories, stories, storyEntities, storySources } from "@/db/schema";
import { eq, desc, sql } from "drizzle-orm";
import { v4 as uuidv4 } from "uuid";
import { initializeDefaultCategories, getStoriesByCategory } from "@/services/news-categories";
import type { StoryCard, ImpactFactor, ExposureMechanism } from "@/types";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const categoryId = url.searchParams.get("id");
  const withStories = url.searchParams.get("stories") === "true";

  if (categoryId) {
    // Get a specific category with its stories
    const cat = await db.query.newsCategories.findFirst({
      where: eq(newsCategories.categoryId, categoryId),
    });

    if (!cat) {
      return NextResponse.json({ error: "Category not found" }, { status: 404 });
    }

    if (withStories) {
      const storyIds = await getStoriesByCategory(categoryId, 30);
      const storyCards: StoryCard[] = [];

      for (const sid of storyIds) {
        const story = await db.query.stories.findFirst({
          where: eq(stories.storyId, sid),
        });
        if (!story) continue;

        const entities = await db.query.storyEntities.findMany({
          where: eq(storyEntities.storyId, sid),
        });
        const sources = await db.query.storySources.findMany({
          where: eq(storySources.storyId, sid),
        });

        let impactTags: ImpactFactor[] = [];
        let exposureMechanisms: ExposureMechanism[] = [];
        try { if (story.impactTags) impactTags = JSON.parse(story.impactTags); } catch {}
        try { if (story.exposureMechanisms) exposureMechanisms = JSON.parse(story.exposureMechanisms); } catch {}

        storyCards.push({
          storyId: story.storyId,
          canonicalTitle: story.canonicalTitle,
          summary: story.summary,
          impactScore: story.impactScore || 0,
          sector: story.sector,
          sourceCount: story.sourceCount || 1,
          createdAt: story.createdAt,
          entities: entities.map((e) => ({
            entityType: e.entityType as StoryCard["entities"][0]["entityType"],
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
        });
      }

      return NextResponse.json({ category: cat, stories: storyCards });
    }

    return NextResponse.json(cat);
  }

  // Get all categories
  let categories = await db.query.newsCategories.findMany({
    orderBy: [desc(newsCategories.sortOrder)],
  });

  // Initialize defaults if empty
  if (categories.length === 0) {
    await initializeDefaultCategories();
    categories = await db.query.newsCategories.findMany({
      orderBy: [desc(newsCategories.sortOrder)],
    });
  }

  // Count stories per category
  const result = await Promise.all(
    categories.map(async (cat) => {
      const count = await db.query.storyCategories.findMany({
        where: eq(storyCategories.categoryId, cat.categoryId),
      });
      return {
        ...cat,
        storyCount: count.length,
      };
    })
  );

  return NextResponse.json(result);
}

export async function POST(request: Request) {
  const body = await request.json();
  const { name, description, keywords, entities, tickers, sectors, countries, icon, color } = body;

  if (!name || !keywords || !Array.isArray(keywords)) {
    return NextResponse.json(
      { error: "name and keywords (array) are required" },
      { status: 400 }
    );
  }

  const now = new Date().toISOString();
  const categoryId = uuidv4();

  await db.insert(newsCategories).values({
    categoryId,
    name,
    description: description || null,
    queryKeywords: JSON.stringify(keywords),
    queryEntities: entities ? JSON.stringify(entities) : null,
    queryTickers: tickers ? JSON.stringify(tickers) : null,
    querySectors: sectors ? JSON.stringify(sectors) : null,
    queryCountries: countries ? JSON.stringify(countries) : null,
    icon: icon || null,
    color: color || null,
    sortOrder: 99,
    enabled: 1,
    createdAt: now,
    updatedAt: now,
  });

  return NextResponse.json({ categoryId, success: true });
}

export async function DELETE(request: Request) {
  const url = new URL(request.url);
  const categoryId = url.searchParams.get("id");

  if (!categoryId) {
    return NextResponse.json({ error: "id required" }, { status: 400 });
  }

  // Delete category mappings first
  await db.delete(storyCategories).where(eq(storyCategories.categoryId, categoryId));
  await db.delete(newsCategories).where(eq(newsCategories.categoryId, categoryId));

  return NextResponse.json({ success: true });
}
