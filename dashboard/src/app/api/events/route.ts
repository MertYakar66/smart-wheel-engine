import { NextResponse } from "next/server";
import { db } from "@/db";
import { events } from "@/db/schema";
import { v4 as uuidv4 } from "uuid";

export async function GET() {
  const allEvents = await db.query.events.findMany();
  return NextResponse.json(allEvents);
}

export async function POST(request: Request) {
  const body = await request.json();
  const { eventType, ticker, eventDate, description } = body;

  if (!eventType || !eventDate) {
    return NextResponse.json(
      { error: "eventType and eventDate are required" },
      { status: 400 }
    );
  }

  const event = {
    eventId: uuidv4(),
    eventType,
    ticker: ticker || null,
    eventDate,
    description: description || null,
  };

  await db.insert(events).values(event);
  return NextResponse.json(event);
}
