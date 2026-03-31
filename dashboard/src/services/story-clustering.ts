import { db } from "@/db";
import { stories, storySources, storyEntities, storyTimeline } from "@/db/schema";
import { eq, desc, sql } from "drizzle-orm";
import { analyzeSentiment } from "./impact-analysis";

// ─── Story Graph Clustering ──────────────────────────────────────────
// Clusters articles into "story objects" representing evolving events.
// Supports: deduplication, merging, timeline tracking, contradiction
// detection, and corroboration scoring.

function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\w\s]/g, "")
      .split(/\s+/)
      .filter((w) => w.length > 2)
  );
}

function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
  const intersection = new Set([...a].filter((x) => b.has(x)));
  const union = new Set([...a, ...b]);
  return union.size === 0 ? 0 : intersection.size / union.size;
}

const SIMILARITY_THRESHOLD = 0.35;
const TIME_WINDOW_MS = 6 * 60 * 60 * 1000; // 6 hours

// ─── Contradiction Detection ──────────────────────────────────────────

function detectContradiction(headlineA: string, headlineB: string): boolean {
  const contradictionPairs = [
    [/\brises?\b|\bsurge[sd]?\b|\bjump[sed]?\b|\brally\b|\bgain[sed]?\b/, /\bfall[sed]?\b|\bdrop[sed]?\b|\bplunge[sd]?\b|\bcrash\b|\bdecline[sd]?\b/],
    [/\bbeat[s]?\b|\bexceed[sed]?\b/, /\bmiss(es|ed)?\b|\bfell short\b/],
    [/\bupgrade[sd]?\b/, /\bdowngrade[sd]?\b/],
    [/\bapprove[sd]?\b/, /\breject[sed]?\b|\bdeny\b|\bdenied\b/],
    [/\bconfirm[sed]?\b/, /\bdeny\b|\bdenied\b/],
  ];

  const lowerA = headlineA.toLowerCase();
  const lowerB = headlineB.toLowerCase();

  for (const [patternPos, patternNeg] of contradictionPairs) {
    if (
      (patternPos.test(lowerA) && patternNeg.test(lowerB)) ||
      (patternNeg.test(lowerA) && patternPos.test(lowerB))
    ) {
      return true;
    }
  }

  return false;
}

// ─── Corroboration Scoring ────────────────────────────────────────────

function computeCorroborationScore(sourceHeadlines: string[]): number {
  if (sourceHeadlines.length <= 1) return 0;

  let agreements = 0;
  let contradictions = 0;
  let comparisons = 0;

  for (let i = 0; i < sourceHeadlines.length; i++) {
    for (let j = i + 1; j < sourceHeadlines.length; j++) {
      comparisons++;
      const sim = jaccardSimilarity(
        tokenize(sourceHeadlines[i]),
        tokenize(sourceHeadlines[j])
      );
      if (sim > 0.3) agreements++;
      if (detectContradiction(sourceHeadlines[i], sourceHeadlines[j])) {
        contradictions++;
      }
    }
  }

  if (comparisons === 0) return 0;
  return Math.max(0, (agreements - contradictions * 2) / comparisons);
}

// ─── Main Clustering Function ─────────────────────────────────────────

export async function clusterStories(): Promise<number> {
  const cutoff = new Date(
    Date.now() - 24 * 60 * 60 * 1000
  ).toISOString();

  const recentStories = await db.query.stories.findMany({
    where: sql`${stories.createdAt} > ${cutoff}`,
    orderBy: [desc(stories.createdAt)],
  });

  let mergeCount = 0;

  for (let i = 0; i < recentStories.length; i++) {
    const storyA = recentStories[i];
    if (!storyA.canonicalTitle) continue;

    const tokensA = tokenize(storyA.canonicalTitle);
    const timeA = new Date(storyA.createdAt).getTime();

    for (let j = i + 1; j < recentStories.length; j++) {
      const storyB = recentStories[j];
      if (!storyB.canonicalTitle) continue;

      const timeB = new Date(storyB.createdAt).getTime();
      if (Math.abs(timeA - timeB) > TIME_WINDOW_MS) continue;

      const tokensB = tokenize(storyB.canonicalTitle);
      const sim = jaccardSimilarity(tokensA, tokensB);

      if (sim >= SIMILARITY_THRESHOLD) {
        // Check for contradictions before merging
        const hasContradiction = detectContradiction(
          storyA.canonicalTitle,
          storyB.canonicalTitle
        );

        // Merge B into A
        await db
          .update(storySources)
          .set({ storyId: storyA.storyId })
          .where(eq(storySources.storyId, storyB.storyId));

        await db
          .update(storyEntities)
          .set({ storyId: storyA.storyId })
          .where(eq(storyEntities.storyId, storyB.storyId));

        // Move timeline entries
        await db
          .update(storyTimeline)
          .set({ storyId: storyA.storyId })
          .where(eq(storyTimeline.storyId, storyB.storyId));

        // Get all sources for corroboration scoring
        const sources = await db.query.storySources.findMany({
          where: eq(storySources.storyId, storyA.storyId),
        });

        const sourceHeadlines = sources.map((s) => s.headline);
        const corroborationScore = computeCorroborationScore(sourceHeadlines);

        // Update the story
        await db
          .update(stories)
          .set({
            sourceCount: sources.length,
            impactScore: sources.length * (1 / (1 + (Date.now() - timeA) / 3600000)),
            updatedAt: new Date().toISOString(),
            storyStatus: "evolving",
            contradictionFlag: hasContradiction ? 1 : (storyA.contradictionFlag || 0),
            corroborationScore,
          })
          .where(eq(stories.storyId, storyA.storyId));

        // Record timeline events
        await db.insert(storyTimeline).values({
          storyId: storyA.storyId,
          eventType: "merged",
          description: `Merged with: "${storyB.canonicalTitle}" (similarity: ${(sim * 100).toFixed(0)}%)`,
          metadata: JSON.stringify({
            mergedStoryId: storyB.storyId,
            similarity: sim,
            newSourceCount: sources.length,
          }),
          occurredAt: new Date().toISOString(),
        });

        if (hasContradiction) {
          await db.insert(storyTimeline).values({
            storyId: storyA.storyId,
            eventType: "contradiction_detected",
            description: `Contradicting headlines detected: "${storyA.canonicalTitle}" vs "${storyB.canonicalTitle}"`,
            metadata: JSON.stringify({
              headlineA: storyA.canonicalTitle,
              headlineB: storyB.canonicalTitle,
            }),
            occurredAt: new Date().toISOString(),
          });
        }

        // Delete merged story
        await db
          .delete(stories)
          .where(eq(stories.storyId, storyB.storyId));

        mergeCount++;
      }
    }
  }

  return mergeCount;
}
