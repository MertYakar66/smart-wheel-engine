import { db } from "@/db";
import { stories } from "@/db/schema";
import { desc, sql } from "drizzle-orm";

// ─── Server-Sent Events (SSE) for Live Headlines ─────────────────────
// Push architecture: clients maintain a persistent connection.
// New stories are pushed as they appear. Polls DB every 10 seconds.

export async function GET() {
  const encoder = new TextEncoder();
  let lastCheckTime = new Date().toISOString();
  let isActive = true;

  const stream = new ReadableStream({
    async start(controller) {
      // Send initial heartbeat
      controller.enqueue(
        encoder.encode(`event: connected\ndata: ${JSON.stringify({ time: lastCheckTime })}\n\n`)
      );

      // Poll for new stories
      const interval = setInterval(async () => {
        if (!isActive) {
          clearInterval(interval);
          return;
        }

        try {
          const newStories = await db.query.stories.findMany({
            where: sql`${stories.createdAt} > ${lastCheckTime}`,
            orderBy: [desc(stories.createdAt)],
            limit: 10,
          });

          if (newStories.length > 0) {
            for (const story of newStories) {
              const event = {
                storyId: story.storyId,
                title: story.canonicalTitle,
                sector: story.sector,
                impactScore: story.impactScore,
                storyStatus: story.storyStatus,
                contradictionFlag: story.contradictionFlag === 1,
                createdAt: story.createdAt,
              };

              controller.enqueue(
                encoder.encode(`event: story\ndata: ${JSON.stringify(event)}\n\n`)
              );
            }

            lastCheckTime = newStories[0].createdAt;
          }

          // Send heartbeat every poll
          controller.enqueue(
            encoder.encode(`event: heartbeat\ndata: ${JSON.stringify({ time: new Date().toISOString(), pending: 0 })}\n\n`)
          );
        } catch (err) {
          console.error("SSE poll error:", err);
        }
      }, 10000); // 10 second poll interval

      // Cleanup when client disconnects
      const cleanup = () => {
        isActive = false;
        clearInterval(interval);
      };

      // Handle abort
      controller.enqueue(
        encoder.encode(`event: heartbeat\ndata: ${JSON.stringify({ time: new Date().toISOString() })}\n\n`)
      );
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
