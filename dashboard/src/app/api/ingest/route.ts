import { NextResponse } from "next/server";
import {
  runIngestionPipeline,
  shouldRunIngestion,
  getLastRun,
} from "@/services/scheduled-ingestion";

export async function POST() {
  // Honor the manual rate-limit (2 min) so a spammed "Ingest Now" button
  // can't hammer the outbound RSS feeds.
  if (!(await shouldRunIngestion("manual"))) {
    const lastRun = await getLastRun();
    return NextResponse.json(
      { success: false, skipped: true, reason: "Too soon since last run", lastRun },
      { status: 429 }
    );
  }

  const result = await runIngestionPipeline("manual");
  return NextResponse.json({
    success: result.status === "completed",
    ...result,
  });
}
