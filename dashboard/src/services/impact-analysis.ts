import { db } from "@/db";
import { stories, storyEntities, storyTimeline } from "@/db/schema";
import { eq, isNull, sql } from "drizzle-orm";
import type { ImpactFactor, ExposureMechanism, ImpactHorizon, Sentiment } from "@/types";

// ─── Impact Analysis Service ──────────────────────────────────────────
// Analyzes stories to produce impact tags, exposure mechanisms,
// sentiment, and "why it matters" narratives.

const OLLAMA_BASE = process.env.OLLAMA_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "qwen2.5:7b";

// ─── Impact factor keyword mapping ───────────────────────────────────

const FACTOR_KEYWORDS: Record<ImpactFactor, string[]> = {
  rates: ["interest rate", "fed", "fomc", "yield", "treasury", "bond", "rate hike", "rate cut", "monetary", "basis point", "hawkish", "dovish"],
  oil: ["oil", "crude", "opec", "brent", "wti", "petroleum", "energy price", "barrel"],
  fx: ["dollar", "euro", "yen", "forex", "currency", "exchange rate", "usd", "eur", "gbp", "fx"],
  regulation: ["sec", "regulation", "compliance", "antitrust", "fine", "penalty", "lawsuit", "court", "ruling", "ban", "sanction", "policy"],
  earnings: ["earnings", "revenue", "profit", "loss", "eps", "beat", "miss", "guidance", "quarterly", "annual report", "q1", "q2", "q3", "q4"],
  demand: ["consumer", "spending", "retail", "sales", "demand", "orders", "consumer confidence"],
  supply_chain: ["supply chain", "shipping", "logistics", "shortage", "chip", "semiconductor", "inventory", "port", "tariff"],
  geopolitical: ["war", "conflict", "geopolitical", "tension", "sanction", "trade war", "diplomatic", "military", "crisis"],
  monetary_policy: ["central bank", "quantitative", "easing", "tightening", "inflation", "deflation", "cpi", "pce", "money supply"],
  labor: ["jobs", "employment", "unemployment", "layoff", "hiring", "wage", "strike", "labor", "workforce", "nonfarm"],
};

const HORIZON_KEYWORDS: Record<ImpactHorizon, string[]> = {
  intraday: ["today", "breaking", "just", "now", "intraday", "this morning", "premarket", "afterhours"],
  days: ["this week", "tomorrow", "near-term", "days", "overnight"],
  weeks: ["next week", "coming weeks", "outlook", "guidance", "forecast"],
  quarters: ["quarter", "annual", "year", "long-term", "structural", "secular"],
};

export function analyzeImpactFactors(title: string, summary: string | null): ImpactFactor[] {
  const text = `${title} ${summary || ""}`.toLowerCase();
  const factors: ImpactFactor[] = [];

  for (const [factor, keywords] of Object.entries(FACTOR_KEYWORDS)) {
    for (const kw of keywords) {
      if (text.includes(kw)) {
        factors.push(factor as ImpactFactor);
        break;
      }
    }
  }

  return factors;
}

export function analyzeImpactHorizon(title: string, summary: string | null): ImpactHorizon {
  const text = `${title} ${summary || ""}`.toLowerCase();

  for (const [horizon, keywords] of Object.entries(HORIZON_KEYWORDS)) {
    for (const kw of keywords) {
      if (text.includes(kw)) return horizon as ImpactHorizon;
    }
  }

  return "days"; // default
}

export function analyzeSentiment(text: string): Sentiment {
  const lower = text.toLowerCase();

  const positiveWords = ["surge", "rally", "gain", "rise", "jump", "beat", "strong", "record", "high", "boost", "upgrade", "bullish", "growth", "profit"];
  const negativeWords = ["crash", "plunge", "drop", "fall", "miss", "weak", "low", "loss", "decline", "downgrade", "bearish", "recession", "default", "crisis"];

  let positive = 0;
  let negative = 0;
  for (const w of positiveWords) if (lower.includes(w)) positive++;
  for (const w of negativeWords) if (lower.includes(w)) negative++;

  if (positive > 0 && negative > 0) return "mixed";
  if (positive > negative) return "positive";
  if (negative > positive) return "negative";
  return "neutral";
}

export function buildExposureMechanisms(
  factors: ImpactFactor[],
  title: string,
  summary: string | null
): ExposureMechanism[] {
  const text = `${title} ${summary || ""}`.toLowerCase();
  const sentiment = analyzeSentiment(text);

  return factors.map((factor) => {
    const direction = sentiment === "positive" ? "positive" as const
      : sentiment === "negative" ? "negative" as const
      : "uncertain" as const;

    return {
      factor,
      direction,
      confidence: factors.length === 1 ? 0.8 : 0.6,
    };
  });
}

// ─── Ollama-based "Why It Matters" Generation ─────────────────────────

async function generateWhyItMatters(
  title: string,
  summary: string | null,
  factors: ImpactFactor[],
  tickers: string[]
): Promise<string | null> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, {
      signal: AbortSignal.timeout(2000),
    });
    if (!res.ok) return null;
  } catch {
    return null;
  }

  const prompt = `You are a financial analyst. Given this news story, write a concise 2-3 sentence "Why It Matters" explanation for investors.

Title: "${title}"
${summary ? `Summary: "${summary}"` : ""}
Impact factors: ${factors.join(", ")}
${tickers.length > 0 ? `Related tickers: ${tickers.join(", ")}` : ""}

Focus on: who is exposed, what mechanism drives the impact, and over what time horizon. Be specific and evidence-based. Do not speculate.

Why It Matters:`;

  try {
    const res = await fetch(`${OLLAMA_BASE}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt,
        stream: false,
        options: { temperature: 0.3, num_predict: 200 },
      }),
      signal: AbortSignal.timeout(30000),
    });

    if (!res.ok) return null;
    const data = await res.json();
    return data.response?.trim() || null;
  } catch {
    return null;
  }
}

// ─── Batch Impact Analysis ────────────────────────────────────────────

export async function analyzeImpactForNewStories(): Promise<number> {
  const unprocessed = await db.query.stories.findMany({
    where: isNull(stories.impactTags),
    limit: 30,
  });

  let processed = 0;

  for (const story of unprocessed) {
    const factors = analyzeImpactFactors(story.canonicalTitle, story.summary);
    const horizon = analyzeImpactHorizon(story.canonicalTitle, story.summary);
    const mechanisms = buildExposureMechanisms(factors, story.canonicalTitle, story.summary);

    // Get tickers for this story
    const entities = await db.query.storyEntities.findMany({
      where: eq(storyEntities.storyId, story.storyId),
    });
    const tickers = entities
      .filter((e) => e.entityType === "ticker")
      .map((e) => e.entityValue);

    // Try AI-generated narrative
    const whyItMatters = await generateWhyItMatters(
      story.canonicalTitle,
      story.summary,
      factors,
      tickers
    );

    await db
      .update(stories)
      .set({
        impactTags: JSON.stringify(factors),
        exposureMechanisms: JSON.stringify(mechanisms),
        impactHorizon: horizon,
        whyItMatters,
        updatedAt: new Date().toISOString(),
      })
      .where(eq(stories.storyId, story.storyId));

    // Record in timeline
    await db.insert(storyTimeline).values({
      storyId: story.storyId,
      eventType: "created",
      description: `Impact analysis: ${factors.length > 0 ? factors.join(", ") : "general"} — horizon: ${horizon}`,
      metadata: JSON.stringify({ factors, horizon }),
      occurredAt: new Date().toISOString(),
    });

    processed++;
  }

  return processed;
}
