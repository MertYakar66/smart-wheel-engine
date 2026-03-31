import type { RetrievalProvider, RetrievalQuery, RetrievalResult } from "@/types";
import Parser from "rss-parser";
import { RSS_FEEDS } from "../rss-feeds";

const parser = new Parser({
  timeout: 10000,
  headers: { "User-Agent": "FinanceNewsPlatform/1.0" },
});

export class RSSRetrievalProvider implements RetrievalProvider {
  name = "rss";

  async isAvailable(): Promise<boolean> {
    return true; // RSS is always available
  }

  async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
    if (query.domain && query.domain !== "news" && query.domain !== "all") {
      return []; // RSS only serves news
    }

    const results: RetrievalResult[] = [];
    const queryTerms = query.query.toLowerCase().split(/\s+/);
    const limit = query.limit || 20;

    const feedResults = await Promise.allSettled(
      RSS_FEEDS.map(async (feed) => {
        try {
          const parsed = await parser.parseURL(feed.url);
          return parsed.items.map((item) => ({
            url: item.link || "",
            title: item.title || "",
            snippet: item.contentSnippet?.slice(0, 500) || "",
            publisher: feed.publisher,
            publishedAt: item.pubDate || item.isoDate || new Date().toISOString(),
            rightsRestricted: false,
            item,
          }));
        } catch {
          return [];
        }
      })
    );

    for (const result of feedResults) {
      if (result.status !== "fulfilled") continue;
      for (const item of result.value) {
        if (!item.url || !item.title) continue;

        // Score relevance by query term matching
        const text = `${item.title} ${item.snippet}`.toLowerCase();
        let score = 0;
        for (const term of queryTerms) {
          if (text.includes(term)) score += 1;
        }
        // Normalize score
        const relevance = queryTerms.length > 0 ? score / queryTerms.length : 0.5;

        // Apply ticker filter
        if (query.filters?.tickers?.length) {
          const hasMatch = query.filters.tickers.some((t) =>
            text.includes(t.toLowerCase())
          );
          if (!hasMatch) continue;
        }

        results.push({
          url: item.url,
          title: item.title,
          snippet: item.snippet,
          publisher: item.publisher,
          publishedAt: item.publishedAt,
          relevanceScore: relevance,
          rightsRestricted: item.rightsRestricted,
        });
      }
    }

    results.sort((a, b) => b.relevanceScore - a.relevanceScore);
    return results.slice(0, limit);
  }
}
