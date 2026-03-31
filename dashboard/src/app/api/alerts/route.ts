import { NextResponse } from "next/server";
import { db } from "@/db";
import { alerts, stories } from "@/db/schema";
import { eq, desc, sql } from "drizzle-orm";

export async function GET() {
  const alertList = await db.query.alerts.findMany({
    where: eq(alerts.dismissed, 0),
    orderBy: [desc(alerts.triggeredAt)],
    limit: 50,
  });

  // Enrich with story titles
  const enriched = await Promise.all(
    alertList.map(async (alert) => {
      let storyTitle: string | undefined;
      if (alert.storyId) {
        const story = await db.query.stories.findFirst({
          where: eq(stories.storyId, alert.storyId),
        });
        storyTitle = story?.canonicalTitle;
      }
      return {
        alertId: alert.alertId,
        storyId: alert.storyId,
        ticker: alert.ticker,
        triggerType: alert.triggerType,
        triggeredAt: alert.triggeredAt,
        dismissed: alert.dismissed === 1,
        storyTitle,
      };
    })
  );

  return NextResponse.json(enriched);
}

export async function PATCH(request: Request) {
  const body = await request.json();
  const { alertId } = body;

  if (!alertId) {
    return NextResponse.json(
      { error: "alertId is required" },
      { status: 400 }
    );
  }

  await db
    .update(alerts)
    .set({ dismissed: 1 })
    .where(eq(alerts.alertId, alertId));

  return NextResponse.json({ success: true });
}
