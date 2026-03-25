import { NextResponse } from "next/server";
import { codeExecutionService } from "@/services/code-execution";

export async function POST(request: Request) {
  const body = await request.json();
  const { code, language, timeout } = body;

  if (!code) {
    return NextResponse.json({ error: "code is required" }, { status: 400 });
  }

  const result = await codeExecutionService.execute({
    code,
    language: language || "python",
    timeout: timeout || 30000,
  });

  return NextResponse.json(result);
}

export async function GET() {
  const executor = await codeExecutionService.getAvailableExecutor();
  return NextResponse.json({
    available: executor !== null,
    executor,
  });
}
