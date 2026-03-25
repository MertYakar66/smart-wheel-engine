import { db } from "@/db";
import { userExposures, storyEntities, watchlists } from "@/db/schema";
import type { StoryCard, UserExposure, ImpactFactor } from "@/types";

// ─── Exposure-First Ranking Engine ────────────────────────────────────
// Ranks stories by relevance to user's holdings, watchlist, sector
// exposure, and factor sensitivities — not by popularity or engagement.

export async function getUserExposures(): Promise<UserExposure[]> {
  // Combine explicit user exposures with watchlist tickers
  const explicit = await db.query.userExposures.findMany();
  const watched = await db.query.watchlists.findMany();

  const exposures: UserExposure[] = explicit.map((e) => ({
    id: e.id,
    exposureType: e.exposureType as UserExposure["exposureType"],
    exposureValue: e.exposureValue,
    weight: e.weight || 1,
  }));

  // Auto-add watchlist tickers as exposures
  for (const w of watched) {
    if (!exposures.some((e) => e.exposureType === "ticker" && e.exposureValue === w.ticker)) {
      exposures.push({
        id: 0,
        exposureType: "ticker",
        exposureValue: w.ticker,
        weight: 1.5, // watchlist tickers get higher weight
      });
    }
  }

  return exposures;
}

export function computeExposureRelevance(
  story: StoryCard,
  exposures: UserExposure[]
): number {
  if (exposures.length === 0) return story.impactScore;

  let relevance = 0;
  const storyTickers = story.entities
    .filter((e) => e.entityType === "ticker")
    .map((e) => e.entityValue.toUpperCase());
  const storyFactors = (story.impactTags || []) as ImpactFactor[];

  for (const exp of exposures) {
    switch (exp.exposureType) {
      case "ticker":
        if (storyTickers.includes(exp.exposureValue.toUpperCase())) {
          relevance += 3 * (exp.weight || 1); // Direct ticker match is highest signal
        }
        break;

      case "sector":
        if (story.sector?.toLowerCase() === exp.exposureValue.toLowerCase()) {
          relevance += 2 * (exp.weight || 1);
        }
        break;

      case "factor":
        if (storyFactors.includes(exp.exposureValue as ImpactFactor)) {
          relevance += 1.5 * (exp.weight || 1);
        }
        break;

      case "country": {
        // Check if country name appears in story text
        const storyText = `${story.canonicalTitle} ${story.summary || ""}`.toLowerCase();
        if (storyText.includes(exp.exposureValue.toLowerCase())) {
          relevance += 1 * (exp.weight || 1);
        }
        break;
      }

      case "theme": {
        const topicEntities = story.entities
          .filter((e) => e.entityType === "topic")
          .map((e) => e.entityValue.toLowerCase());
        if (topicEntities.includes(exp.exposureValue.toLowerCase())) {
          relevance += 1.5 * (exp.weight || 1);
        }
        break;
      }
    }
  }

  // Blend with base impact score
  const baseScore = story.impactScore || 0;
  const blendedScore = relevance > 0
    ? relevance * 2 + baseScore * 0.5
    : baseScore;

  // Boost for multi-source corroboration
  const corroborationBoost = Math.min(story.sourceCount / 3, 2);

  return blendedScore * (1 + corroborationBoost * 0.2);
}

export function rankStoriesByExposure(
  stories: StoryCard[],
  exposures: UserExposure[]
): StoryCard[] {
  return stories
    .map((story) => ({
      ...story,
      exposureRelevance: computeExposureRelevance(story, exposures),
    }))
    .sort((a, b) => (b.exposureRelevance || 0) - (a.exposureRelevance || 0));
}
