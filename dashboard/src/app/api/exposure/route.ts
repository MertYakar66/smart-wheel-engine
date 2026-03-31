import { NextResponse } from "next/server";
import { db } from "@/db";
import { userExposures } from "@/db/schema";
import { eq, and } from "drizzle-orm";
import { getUserExposures } from "@/services/exposure-ranking";

export async function GET() {
  const exposures = await getUserExposures();
  return NextResponse.json(exposures);
}

export async function POST(request: Request) {
  const body = await request.json();
  const { exposureType, exposureValue, weight } = body;

  if (!exposureType || !exposureValue) {
    return NextResponse.json(
      { error: "exposureType and exposureValue required" },
      { status: 400 }
    );
  }

  const validTypes = ["ticker", "sector", "factor", "country", "theme"];
  if (!validTypes.includes(exposureType)) {
    return NextResponse.json(
      { error: `exposureType must be one of: ${validTypes.join(", ")}` },
      { status: 400 }
    );
  }

  await db.insert(userExposures).values({
    exposureType,
    exposureValue,
    weight: weight || 1,
    addedAt: new Date().toISOString(),
  });

  return NextResponse.json({ success: true });
}

export async function DELETE(request: Request) {
  const url = new URL(request.url);
  const exposureType = url.searchParams.get("type");
  const exposureValue = url.searchParams.get("value");

  if (!exposureType || !exposureValue) {
    return NextResponse.json(
      { error: "type and value query params required" },
      { status: 400 }
    );
  }

  await db
    .delete(userExposures)
    .where(
      and(
        eq(userExposures.exposureType, exposureType),
        eq(userExposures.exposureValue, exposureValue)
      )
    );

  return NextResponse.json({ success: true });
}
