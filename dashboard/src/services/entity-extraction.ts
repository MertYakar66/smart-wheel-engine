import { db } from "@/db";
import { storyEntities, stories } from "@/db/schema";
import { eq, sql, isNull } from "drizzle-orm";

// Ollama-based entity extraction for richer NLP when available
// Falls back to regex-based extraction

const OLLAMA_BASE = process.env.OLLAMA_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5:7b";

interface ExtractedEntities {
  tickers: string[];
  people: string[];
  organizations: string[];
  topics: string[];
}

async function isOllamaAvailable(): Promise<boolean> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

async function extractWithOllama(
  headline: string
): Promise<ExtractedEntities> {
  const prompt = `Extract entities from this financial news headline. Return ONLY valid JSON with these keys: tickers (stock ticker symbols like AAPL, MSFT), people (person names), organizations (company/org names), topics (financial topics like "earnings", "merger", "interest rates").

Headline: "${headline}"

JSON:`;

  try {
    const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt,
        stream: false,
        options: { temperature: 0.1, num_predict: 200 },
      }),
      signal: AbortSignal.timeout(30000),
    });

    if (!res.ok) throw new Error("Ollama request failed");

    const data = await res.json();
    const text = data.response || "";

    // Parse JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) throw new Error("No JSON in response");

    const parsed = JSON.parse(jsonMatch[0]);
    return {
      tickers: Array.isArray(parsed.tickers) ? parsed.tickers : [],
      people: Array.isArray(parsed.people) ? parsed.people : [],
      organizations: Array.isArray(parsed.organizations) ? parsed.organizations : [],
      topics: Array.isArray(parsed.topics) ? parsed.topics : [],
    };
  } catch {
    // Fall back to regex
    return extractWithRegex(headline);
  }
}

function extractWithRegex(headline: string): ExtractedEntities {
  const commonWords = new Set([
    "A", "I", "AM", "PM", "THE", "AND", "FOR", "ARE", "BUT", "NOT",
    "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS",
    "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY",
    "WHO", "DID", "GET", "LET", "SAY", "SHE", "TOO", "USE", "CEO",
    "CFO", "IPO", "GDP", "CPI", "FED", "SEC", "ETF", "AI", "US",
    "UK", "EU", "NYSE", "FOMC", "OPEC", "FDA", "DOJ", "EPA", "TOP",
  ]);

  const tickerMatches = headline.match(/\b[A-Z]{2,5}\b/g) || [];
  const tickers = [...new Set(tickerMatches)].filter(
    (t) => !commonWords.has(t)
  );

  // Basic topic extraction
  const topicKeywords = [
    "earnings", "revenue", "profit", "loss", "merger", "acquisition",
    "IPO", "bankruptcy", "layoffs", "dividend", "buyback", "split",
    "inflation", "recession", "interest rate", "fed", "rally", "crash",
    "surge", "plunge", "beat", "miss", "guidance", "outlook",
  ];
  const lowerHeadline = headline.toLowerCase();
  const topics = topicKeywords.filter((k) =>
    lowerHeadline.includes(k.toLowerCase())
  );

  return {
    tickers: tickers.slice(0, 5),
    people: [],
    organizations: [],
    topics: topics.slice(0, 5),
  };
}

export async function extractEntitiesForNewStories(): Promise<number> {
  // Find stories without entity extraction
  const unprocessed = await db.query.stories.findMany({
    where: isNull(stories.sector),
    limit: 50,
  });

  const useOllama = await isOllamaAvailable();
  let processed = 0;

  for (const story of unprocessed) {
    const entities = useOllama
      ? await extractWithOllama(story.canonicalTitle)
      : extractWithRegex(story.canonicalTitle);

    // Save entities
    for (const ticker of entities.tickers) {
      await db.insert(storyEntities).values({
        storyId: story.storyId,
        entityType: "ticker",
        entityValue: ticker.toUpperCase(),
      }).onConflictDoNothing();
    }
    for (const person of entities.people) {
      await db.insert(storyEntities).values({
        storyId: story.storyId,
        entityType: "person",
        entityValue: person,
      }).onConflictDoNothing();
    }
    for (const org of entities.organizations) {
      await db.insert(storyEntities).values({
        storyId: story.storyId,
        entityType: "org",
        entityValue: org,
      }).onConflictDoNothing();
    }
    for (const topic of entities.topics) {
      await db.insert(storyEntities).values({
        storyId: story.storyId,
        entityType: "topic",
        entityValue: topic,
      }).onConflictDoNothing();
    }

    // Set sector based on topic heuristics
    const sector = inferSector(entities.topics, story.canonicalTitle);
    await db
      .update(stories)
      .set({ sector, updatedAt: new Date().toISOString() })
      .where(eq(stories.storyId, story.storyId));

    processed++;
  }

  return processed;
}

function inferSector(topics: string[], headline: string): string {
  const lower = headline.toLowerCase();
  if (lower.includes("tech") || lower.includes("software") || lower.includes("ai"))
    return "Technology";
  if (lower.includes("bank") || lower.includes("financ"))
    return "Financials";
  if (lower.includes("oil") || lower.includes("energy") || lower.includes("gas"))
    return "Energy";
  if (lower.includes("pharma") || lower.includes("health") || lower.includes("drug"))
    return "Healthcare";
  if (lower.includes("retail") || lower.includes("consumer"))
    return "Consumer";
  if (topics.includes("inflation") || topics.includes("interest rate") || topics.includes("fed"))
    return "Macro";
  return "General";
}
