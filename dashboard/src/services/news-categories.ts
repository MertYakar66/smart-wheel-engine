import { db } from "@/db";
import { newsCategories, storyCategories, stories, storyEntities } from "@/db/schema";
import { eq, desc, sql } from "drizzle-orm";
import { v4 as uuidv4 } from "uuid";

// ─── News Category System ─────────────────────────────────────────────
// Bloomberg-style "saved queries" that define what news to pull.
// Each category is a query definition that maps stories automatically.

export interface CategoryDefinition {
  name: string;
  description?: string;
  keywords: string[];
  entities?: string[];       // specific entity values (e.g., "FED", "ECB")
  tickers?: string[];
  sectors?: string[];
  countries?: string[];
  icon?: string;
  color?: string;
}

// ─── Default Categories (Bloomberg-inspired taxonomy) ─────────────────

export const DEFAULT_CATEGORIES: CategoryDefinition[] = [
  {
    name: "Central Banks & Monetary Policy",
    description: "Fed, ECB, BoJ, BoE rate decisions, dot plots, and policy statements",
    keywords: ["fed", "fomc", "rate decision", "interest rate", "central bank", "ecb", "boj", "boe", "monetary policy", "hawkish", "dovish", "dot plot", "quantitative", "tightening", "easing", "basis point"],
    entities: ["FED", "ECB", "BOJ", "BOE"],
    sectors: ["Macro"],
    icon: "Landmark",
    color: "#3B82F6",
  },
  {
    name: "Earnings & Corporate Results",
    description: "Quarterly earnings, revenue beats/misses, guidance updates",
    keywords: ["earnings", "revenue", "eps", "beat", "miss", "guidance", "quarterly", "annual results", "profit", "loss", "margin", "outlook", "forecast", "q1", "q2", "q3", "q4"],
    sectors: ["Technology", "Financials", "Healthcare", "Consumer", "Energy"],
    icon: "BarChart3",
    color: "#22C55E",
  },
  {
    name: "Macro & Economic Data",
    description: "CPI, jobs, GDP, PMI, retail sales, and economic indicators",
    keywords: ["cpi", "inflation", "jobs", "unemployment", "nonfarm", "gdp", "pmi", "retail sales", "consumer confidence", "housing", "industrial production", "trade balance", "economic data"],
    sectors: ["Macro"],
    icon: "TrendingUp",
    color: "#F59E0B",
  },
  {
    name: "Geopolitics & Trade",
    description: "Trade wars, sanctions, conflicts, and geopolitical tensions",
    keywords: ["tariff", "trade war", "sanction", "geopolitical", "conflict", "diplomatic", "military", "tension", "embargo", "trade deal", "treaty", "nato", "china", "russia", "middle east"],
    icon: "Globe",
    color: "#EF4444",
  },
  {
    name: "Energy & Commodities",
    description: "Oil, gas, OPEC, mining, agriculture, and commodity markets",
    keywords: ["oil", "crude", "opec", "brent", "wti", "natural gas", "gold", "silver", "copper", "mining", "commodity", "barrel", "energy", "renewable", "solar", "wind"],
    sectors: ["Energy"],
    icon: "Flame",
    color: "#F97316",
  },
  {
    name: "Technology & AI",
    description: "Tech earnings, AI developments, semiconductors, cloud, and crypto",
    keywords: ["ai", "artificial intelligence", "chip", "semiconductor", "cloud", "saas", "cybersecurity", "blockchain", "crypto", "bitcoin", "apple", "google", "microsoft", "nvidia", "meta", "amazon"],
    tickers: ["AAPL", "GOOGL", "MSFT", "NVDA", "META", "AMZN", "TSM", "AVGO"],
    sectors: ["Technology"],
    icon: "Cpu",
    color: "#8B5CF6",
  },
  {
    name: "Financials & Banking",
    description: "Banks, insurance, fintech, credit, and financial regulation",
    keywords: ["bank", "banking", "credit", "loan", "mortgage", "insurance", "fintech", "payment", "deposit", "capital", "basel", "stress test", "fdic"],
    tickers: ["JPM", "BAC", "GS", "MS", "WFC", "C"],
    sectors: ["Financials"],
    icon: "Building2",
    color: "#0EA5E9",
  },
  {
    name: "Regulation & Legal",
    description: "SEC enforcement, antitrust, lawsuits, and regulatory changes",
    keywords: ["sec", "regulation", "antitrust", "lawsuit", "fine", "penalty", "compliance", "ruling", "court", "doj", "ftc", "fda", "epa", "investigation", "subpoena"],
    icon: "Scale",
    color: "#DC2626",
  },
  {
    name: "IPOs & Deals",
    description: "IPOs, M&A, SPACs, buybacks, and corporate transactions",
    keywords: ["ipo", "merger", "acquisition", "deal", "buyout", "spac", "buyback", "spin-off", "divestiture", "takeover", "bid", "tender offer", "private equity"],
    icon: "Handshake",
    color: "#14B8A6",
  },
  {
    name: "Options & Volatility",
    description: "VIX, options flow, unusual activity, and volatility events",
    keywords: ["vix", "volatility", "options", "put", "call", "gamma", "delta", "theta", "implied volatility", "options flow", "unusual activity", "expiration", "opex", "squeeze"],
    icon: "Activity",
    color: "#A855F7",
  },
];

// ─── Initialize Default Categories ────────────────────────────────────

export async function initializeDefaultCategories(): Promise<number> {
  const existing = await db.query.newsCategories.findMany();
  if (existing.length > 0) return 0;

  const now = new Date().toISOString();
  let count = 0;

  for (let i = 0; i < DEFAULT_CATEGORIES.length; i++) {
    const cat = DEFAULT_CATEGORIES[i];
    await db.insert(newsCategories).values({
      categoryId: uuidv4(),
      name: cat.name,
      description: cat.description || null,
      queryKeywords: JSON.stringify(cat.keywords),
      queryEntities: cat.entities ? JSON.stringify(cat.entities) : null,
      queryTickers: cat.tickers ? JSON.stringify(cat.tickers) : null,
      querySectors: cat.sectors ? JSON.stringify(cat.sectors) : null,
      queryCountries: cat.countries ? JSON.stringify(cat.countries) : null,
      icon: cat.icon || null,
      color: cat.color || null,
      sortOrder: i,
      enabled: 1,
      createdAt: now,
      updatedAt: now,
    });
    count++;
  }

  return count;
}

// ─── Category Matching ────────────────────────────────────────────────
// Matches stories to categories using the hybrid rules + model approach.
// Layer A: deterministic keyword/entity matching
// Layer B: model classification (future)

export async function matchStoriesToCategories(): Promise<number> {
  const categories = await db.query.newsCategories.findMany({
    where: eq(newsCategories.enabled, 1),
  });

  if (categories.length === 0) {
    await initializeDefaultCategories();
    return matchStoriesToCategories(); // retry after init
  }

  // Get stories that haven't been categorized recently (last 1 hour)
  const cutoff = new Date(Date.now() - 60 * 60 * 1000).toISOString();
  const recentStories = await db.query.stories.findMany({
    where: sql`${stories.createdAt} > ${cutoff}`,
    orderBy: [desc(stories.createdAt)],
    limit: 100,
  });

  let matched = 0;

  for (const story of recentStories) {
    const storyText = `${story.canonicalTitle} ${story.summary || ""}`.toLowerCase();

    // Get story entities for ticker/sector matching
    const entities = await db.query.storyEntities.findMany({
      where: eq(storyEntities.storyId, story.storyId),
    });
    const storyTickers = entities
      .filter((e) => e.entityType === "ticker")
      .map((e) => e.entityValue.toUpperCase());

    for (const cat of categories) {
      const keywords: string[] = JSON.parse(cat.queryKeywords);
      const catTickers: string[] = cat.queryTickers ? JSON.parse(cat.queryTickers) : [];
      const catSectors: string[] = cat.querySectors ? JSON.parse(cat.querySectors) : [];
      const catEntities: string[] = cat.queryEntities ? JSON.parse(cat.queryEntities) : [];

      let score = 0;
      let maxScore = 0;

      // Keyword matching (weighted by specificity)
      maxScore += keywords.length;
      for (const kw of keywords) {
        if (storyText.includes(kw.toLowerCase())) {
          score += 1;
        }
      }

      // Ticker matching (high signal)
      if (catTickers.length > 0) {
        maxScore += catTickers.length * 2;
        for (const t of catTickers) {
          if (storyTickers.includes(t)) score += 2;
        }
      }

      // Sector matching
      if (catSectors.length > 0 && story.sector) {
        maxScore += 2;
        if (catSectors.includes(story.sector)) score += 2;
      }

      // Entity matching
      if (catEntities.length > 0) {
        maxScore += catEntities.length;
        for (const e of catEntities) {
          if (storyText.includes(e.toLowerCase())) score += 1;
        }
      }

      const matchScore = maxScore > 0 ? score / maxScore : 0;

      // Only categorize if above threshold (0.15 = at least some keyword/entity match)
      if (matchScore >= 0.15) {
        // Check if already categorized
        const existing = await db.query.storyCategories.findFirst({
          where: sql`${storyCategories.storyId} = ${story.storyId} AND ${storyCategories.categoryId} = ${cat.categoryId}`,
        });

        if (!existing) {
          await db.insert(storyCategories).values({
            storyId: story.storyId,
            categoryId: cat.categoryId,
            matchScore,
            matchedAt: new Date().toISOString(),
          });
          matched++;
        }
      }
    }
  }

  return matched;
}

// ─── Get Stories by Category ──────────────────────────────────────────

export async function getStoriesByCategory(
  categoryId: string,
  limit: number = 30
): Promise<string[]> {
  const rows = await db.query.storyCategories.findMany({
    where: eq(storyCategories.categoryId, categoryId),
    limit,
  });

  // Sort by match score descending
  return rows
    .sort((a, b) => (b.matchScore || 0) - (a.matchScore || 0))
    .map((r) => r.storyId!)
    .filter(Boolean);
}
