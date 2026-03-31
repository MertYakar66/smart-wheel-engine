import type { RetrievalProvider, RetrievalQuery, RetrievalResult } from "@/types";
import { RSSRetrievalProvider } from "./rss-provider";
import { ValyuRetrievalProvider } from "./valyu-provider";

// ─── Unified Retrieval Layer ──────────────────────────────────────────
// Abstracts over multiple data providers (RSS, Valyu, etc.)
// Normalizes results into a canonical format for the story engine

class RetrievalOrchestrator {
  private providers: RetrievalProvider[] = [];

  constructor() {
    // Register providers — order determines priority
    this.providers.push(new RSSRetrievalProvider());
    this.providers.push(new ValyuRetrievalProvider());
  }

  async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
    const results: RetrievalResult[] = [];

    // Query all available providers in parallel
    const available = await Promise.all(
      this.providers.map(async (p) => ({
        provider: p,
        ready: await p.isAvailable(),
      }))
    );

    const activeProviders = available
      .filter((p) => p.ready)
      .map((p) => p.provider);

    if (activeProviders.length === 0) {
      console.warn("No retrieval providers available");
      return [];
    }

    const providerResults = await Promise.allSettled(
      activeProviders.map((p) => p.search(query))
    );

    for (const result of providerResults) {
      if (result.status === "fulfilled") {
        results.push(...result.value);
      }
    }

    // Deduplicate by URL
    const seen = new Set<string>();
    const deduped = results.filter((r) => {
      if (seen.has(r.url)) return false;
      seen.add(r.url);
      return true;
    });

    // Sort by relevance score descending
    deduped.sort((a, b) => b.relevanceScore - a.relevanceScore);

    return deduped.slice(0, query.limit || 50);
  }

  async getAvailableProviders(): Promise<string[]> {
    const checks = await Promise.all(
      this.providers.map(async (p) => ({
        name: p.name,
        available: await p.isAvailable(),
      }))
    );
    return checks.filter((c) => c.available).map((c) => c.name);
  }
}

// Singleton
export const retrievalOrchestrator = new RetrievalOrchestrator();
