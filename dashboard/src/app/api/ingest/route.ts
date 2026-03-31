import { NextResponse } from "next/server";
import { runIngestionPipeline } from "@/services/scheduled-ingestion";

export async function POST() {
  const result = await runIngestionPipeline("manual");
  return NextResponse.json({
    success: result.status === "completed",
    ...result,
  });
}
