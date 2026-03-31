import { NextResponse } from "next/server";
import { generateBriefing, getLatestBriefing, getBriefingHistory } from "@/services/briefing-generator";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const type = url.searchParams.get("type") as "morning" | "evening" | "breaking" | null;
  const history = url.searchParams.get("history") === "true";

  if (history) {
    const briefings = await getBriefingHistory(14);
    return NextResponse.json(briefings);
  }

  const latest = await getLatestBriefing(type || undefined);
  if (!latest) {
    return NextResponse.json({ error: "No briefing available" }, { status: 404 });
  }

  return NextResponse.json(latest);
}

export async function POST(request: Request) {
  const body = await request.json();
  const type = body.type || "morning";

  if (!["morning", "evening", "breaking"].includes(type)) {
    return NextResponse.json(
      { error: "type must be morning, evening, or breaking" },
      { status: 400 }
    );
  }

  try {
    const briefing = await generateBriefing(type);
    return NextResponse.json(briefing);
  } catch (error) {
    console.error("Briefing generation failed:", error);
    return NextResponse.json(
      { error: "Failed to generate briefing" },
      { status: 500 }
    );
  }
}
