import Parser from "rss-parser";
import { v4 as uuidv4 } from "uuid";
import { db } from "@/db";
import { storySources, stories } from "@/db/schema";
import { eq } from "drizzle-orm";
import { RSS_FEEDS } from "./rss-feeds";
import type { Headline } from "@/types";

const parser = new Parser({
  timeout: 10000,
  headers: {
    "User-Agent": "FinanceNewsPlatform/1.0",
  },
});

export async function ingestAllFeeds(): Promise<Headline[]> {
  const allHeadlines: Headline[] = [];

  const results = await Promise.allSettled(
    RSS_FEEDS.map((feed) => ingestFeed(feed.url, feed.publisher))
  );

  for (const result of results) {
    if (result.status === "fulfilled") {
      allHeadlines.push(...result.value);
    }
  }

  return allHeadlines;
}

async function ingestFeed(
  feedUrl: string,
  publisher: string
): Promise<Headline[]> {
  try {
    const feed = await parser.parseURL(feedUrl);
    const headlines: Headline[] = [];

    for (const item of feed.items) {
      if (!item.title || !item.link) continue;

      // Check for duplicate by URL
      const existing = await db.query.storySources.findFirst({
        where: eq(storySources.url, item.link),
      });
      if (existing) continue;

      const sourceId = uuidv4();
      const now = new Date().toISOString();
      const publishedAt =
        item.pubDate || item.isoDate || now;

      const headline: Headline = {
        sourceId,
        url: item.link,
        publisher,
        headline: item.title,
        publishedAt,
        snippet: item.contentSnippet?.slice(0, 500) || "",
      };

      // Create a story for each source initially; clustering merges them later
      const storyId = uuidv4();
      await db.insert(stories).values({
        storyId,
        canonicalTitle: item.title,
        summary: item.contentSnippet?.slice(0, 300) || null,
        impactScore: 1,
        sector: null,
        createdAt: now,
        updatedAt: now,
        sourceCount: 1,
      });

      await db.insert(storySources).values({
        sourceId,
        storyId,
        url: item.link,
        publisher,
        headline: item.title,
        publishedAt,
        snippet: item.contentSnippet?.slice(0, 500) || "",
      });

      // Entity extraction is owned by extractEntitiesForNewStories (pipeline
      // step 2), which validates tickers against the engine universe. A
      // second unvalidated writer here produced duplicate + junk ticker rows.

      headlines.push(headline);
    }

    return headlines;
  } catch {
    console.error(`Failed to ingest feed: ${feedUrl}`);
    return [];
  }
}
