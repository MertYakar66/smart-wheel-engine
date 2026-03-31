import { streamText } from "ai";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { db } from "@/db";
import { chatSessions, messages } from "@/db/schema";
import { v4 as uuidv4 } from "uuid";
import { eq } from "drizzle-orm";

// Use OpenAI-compatible provider pointing at Ollama's OpenAI-compatible endpoint
const ollamaProvider = createOpenAICompatible({
  name: "ollama",
  baseURL: (process.env.OLLAMA_URL || "http://localhost:11434") + "/v1",
});

const model = ollamaProvider.chatModel(process.env.OLLAMA_MODEL || "qwen2.5:7b");

export async function POST(request: Request) {
  const body = await request.json();
  const { messages: userMessages, sessionId: existingSessionId } = body;

  const sessionId = existingSessionId || uuidv4();
  const now = new Date().toISOString();

  // Create or update session
  if (!existingSessionId) {
    await db.insert(chatSessions).values({
      sessionId,
      title:
        userMessages[0]?.content?.slice(0, 100) ||
        "New Research Session",
      createdAt: now,
      updatedAt: now,
    });
  } else {
    await db
      .update(chatSessions)
      .set({ updatedAt: now })
      .where(eq(chatSessions.sessionId, sessionId));
  }

  // Save user message
  const lastUserMsg = userMessages[userMessages.length - 1];
  if (lastUserMsg?.role === "user") {
    await db.insert(messages).values({
      messageId: uuidv4(),
      sessionId,
      role: "user",
      content: lastUserMsg.content,
      createdAt: now,
    });
  }

  const systemPrompt = `You are a financial research assistant. You help users analyze financial news, market data, SEC filings, and macroeconomic indicators. Be concise, factual, and data-driven. When referencing data, cite specific numbers and dates. If you don't have enough information to answer confidently, say so.`;

  try {
    const result = streamText({
      model,
      system: systemPrompt,
      messages: userMessages,
    });

    return result.toTextStreamResponse();
  } catch {
    // If Ollama is not available, return a helpful error
    return new Response(
      JSON.stringify({
        error:
          "Ollama is not available. Please ensure Ollama is running locally with a model installed (e.g., ollama pull qwen2.5:7b).",
      }),
      { status: 503, headers: { "Content-Type": "application/json" } }
    );
  }
}
