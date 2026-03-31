import { NextResponse } from "next/server";
import { db } from "@/db";
import { stories, storyEntities } from "@/db/schema";
import { eq } from "drizzle-orm";
import { codeExecutionService } from "@/services/code-execution";
import { fetchQuoteFromFinnhub, getLatestSnapshot } from "@/services/market-data";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  const story = await db.query.stories.findFirst({
    where: eq(stories.storyId, id),
  });

  if (!story) {
    return NextResponse.json({ error: "Story not found" }, { status: 404 });
  }

  // Get tickers for this story
  const entities = await db.query.storyEntities.findMany({
    where: eq(storyEntities.storyId, id),
  });
  const tickers = entities
    .filter((e) => e.entityType === "ticker")
    .map((e) => e.entityValue);

  // Fetch current prices for all tickers
  const quotes = await Promise.all(
    tickers.slice(0, 5).map(async (ticker) => {
      const live = await fetchQuoteFromFinnhub(ticker);
      if (live) return live;
      return getLatestSnapshot(ticker);
    })
  );

  // Run event study template
  const eventStudy = await codeExecutionService.execute({
    code: "# template: event_study",
    language: "python",
  });

  // Run peer comparison template if multiple tickers
  let peerComparison = null;
  if (tickers.length > 1) {
    peerComparison = await codeExecutionService.execute({
      code: "# template: peer_comparison",
      language: "python",
    });
  }

  return NextResponse.json({
    storyId: id,
    tickers,
    quotes: quotes.filter(Boolean),
    eventStudy: eventStudy.success ? eventStudy : null,
    peerComparison: peerComparison?.success ? peerComparison : null,
    executor: await codeExecutionService.getAvailableExecutor(),
  });
}
