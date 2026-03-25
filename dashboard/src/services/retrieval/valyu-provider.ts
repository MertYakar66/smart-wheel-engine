import type { RetrievalProvider, RetrievalQuery, RetrievalResult } from "@/types";

// ─── Valyu Unified Search API Provider ────────────────────────────────
// Valyu provides a single retrieval interface across web, news, financial
// data, filings, and research — reducing integration complexity.
// See: https://valyu.network/

const VALYU_API_KEY = process.env.VALYU_API_KEY || "";
const VALYU_BASE_URL = process.env.VALYU_API_URL || "https://api.valyu.network/v1";

interface ValyuSearchResponse {
  results: Array<{
    url: string;
    title: string;
    content: string;
    source: string;
    published_at?: string;
    relevance_score: number;
    metadata?: {
      rights_restricted?: boolean;
      geography?: string;
      data_type?: string;
    };
  }>;
  total_results: number;
}

export class ValyuRetrievalProvider implements RetrievalProvider {
  name = "valyu";

  async isAvailable(): Promise<boolean> {
    if (!VALYU_API_KEY) return false;
    try {
      const res = await fetch(`${VALYU_BASE_URL}/health`, {
        headers: { Authorization: `Bearer ${VALYU_API_KEY}` },
        signal: AbortSignal.timeout(3000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  async search(query: RetrievalQuery): Promise<RetrievalResult[]> {
    if (!VALYU_API_KEY) return [];

    try {
      const body: Record<string, unknown> = {
        query: query.query,
        max_results: query.limit || 20,
        search_type: "all",
      };

      // Map domain to Valyu data types
      if (query.domain && query.domain !== "all") {
        const domainMap: Record<string, string> = {
          news: "news",
          filings: "financial_filings",
          research: "academic",
          market: "financial_data",
        };
        body.data_type = domainMap[query.domain] || "all";
      }

      // Apply filters
      if (query.filters) {
        const filters: Record<string, unknown> = {};
        if (query.filters.tickers?.length) {
          filters.tickers = query.filters.tickers;
        }
        if (query.filters.dateFrom) {
          filters.date_from = query.filters.dateFrom;
        }
        if (query.filters.dateTo) {
          filters.date_to = query.filters.dateTo;
        }
        if (query.filters.countries?.length) {
          filters.countries = query.filters.countries;
        }
        if (Object.keys(filters).length > 0) {
          body.filters = filters;
        }
      }

      const res = await fetch(`${VALYU_BASE_URL}/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${VALYU_API_KEY}`,
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(15000),
      });

      if (!res.ok) {
        console.error(`Valyu search failed: ${res.status}`);
        return [];
      }

      const data: ValyuSearchResponse = await res.json();

      return data.results.map((r) => ({
        url: r.url,
        title: r.title,
        snippet: r.content?.slice(0, 500) || "",
        publisher: r.source || "Unknown",
        publishedAt: r.published_at || new Date().toISOString(),
        relevanceScore: r.relevance_score || 0,
        rightsRestricted: r.metadata?.rights_restricted || false,
        geography: r.metadata?.geography,
        metadata: r.metadata as Record<string, unknown>,
      }));
    } catch (err) {
      console.error("Valyu search error:", err);
      return [];
    }
  }
}
