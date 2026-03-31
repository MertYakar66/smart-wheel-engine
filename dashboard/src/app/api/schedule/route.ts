import { NextResponse } from "next/server";
import {
  runIngestionPipeline,
  shouldRunIngestion,
  getLastRun,
  getRunHistory,
  type RunType,
} from "@/services/scheduled-ingestion";

// GET: Check schedule status and run history
export async function GET(request: Request) {
  const url = new URL(request.url);
  const action = url.searchParams.get("action");

  if (action === "history") {
    const history = await getRunHistory(20);
    return NextResponse.json(history);
  }

  if (action === "status") {
    const [morning, evening, lastAny] = await Promise.all([
      getLastRun("morning"),
      getLastRun("evening"),
      getLastRun(),
    ]);

    const [shouldMorning, shouldEvening, shouldHot] = await Promise.all([
      shouldRunIngestion("morning"),
      shouldRunIngestion("evening"),
      shouldRunIngestion("hot"),
    ]);

    return NextResponse.json({
      lastMorningRun: morning,
      lastEveningRun: evening,
      lastRun: lastAny,
      shouldRunMorning: shouldMorning,
      shouldRunEvening: shouldEvening,
      shouldRunHot: shouldHot,
    });
  }

  // Default: return last run
  const lastRun = await getLastRun();
  return NextResponse.json(lastRun || { message: "No runs yet" });
}

// POST: Trigger an ingestion run
export async function POST(request: Request) {
  const body = await request.json();
  const runType: RunType = body.runType || "manual";

  if (!["morning", "evening", "manual", "hot"].includes(runType)) {
    return NextResponse.json(
      { error: "runType must be morning, evening, manual, or hot" },
      { status: 400 }
    );
  }

  // Check if we should run
  if (runType !== "manual") {
    const shouldRun = await shouldRunIngestion(runType);
    if (!shouldRun) {
      const lastRun = await getLastRun(runType);
      return NextResponse.json({
        skipped: true,
        reason: "Too soon since last run",
        lastRun,
      });
    }
  }

  const result = await runIngestionPipeline(runType);
  return NextResponse.json(result);
}
