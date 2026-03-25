"use client";

import { useState, useRef, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Bot, User, AlertCircle } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

function ResearchContent() {
  const searchParams = useSearchParams();
  const storyTitle = searchParams.get("title");
  const ticker = searchParams.get("ticker");

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Pre-populate from URL params
  useEffect(() => {
    if (storyTitle && messages.length === 0) {
      setInput(`Analyze this financial news story: "${storyTitle}"`);
    } else if (ticker && messages.length === 0) {
      setInput(
        `Give me a brief analysis of ${ticker}. What's the current outlook?`
      );
    }
  }, [storyTitle, ticker, messages.length]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(
          errData.error ||
            "Failed to get response. Is Ollama running locally?"
        );
      }

      // Handle streaming response
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      let assistantContent = "";
      setMessages([
        ...newMessages,
        { role: "assistant", content: "" },
      ]);

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        // Parse SSE data lines
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("0:")) {
            // Vercel AI SDK text stream format
            try {
              const text = JSON.parse(line.slice(2));
              assistantContent += text;
              setMessages([
                ...newMessages,
                { role: "assistant", content: assistantContent },
              ]);
            } catch {
              // Skip unparseable lines
            }
          }
        }
      }

      if (!assistantContent) {
        // Non-streaming fallback
        assistantContent =
          "I received your message but couldn't generate a streaming response. Please ensure Ollama is running with a compatible model.";
        setMessages([
          ...newMessages,
          { role: "assistant", content: assistantContent },
        ]);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "An error occurred"
      );
      // Remove the loading message
      setMessages(newMessages);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col">
      <div className="mb-4">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
          Research Chat
        </h1>
        <p className="text-sm text-zinc-500 dark:text-zinc-400">
          AI-powered financial research assistant (via local Ollama)
        </p>
      </div>

      {/* Chat Messages */}
      <Card className="flex flex-1 flex-col overflow-hidden">
        <ScrollArea
          ref={scrollRef}
          className="flex-1 p-4"
        >
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center py-20">
              <div className="text-center">
                <Bot className="mx-auto h-12 w-12 text-zinc-300 dark:text-zinc-700" />
                <p className="mt-4 text-sm text-zinc-500">
                  Ask questions about financial news, market data, SEC filings,
                  or macroeconomic indicators.
                </p>
                <div className="mt-4 flex flex-wrap justify-center gap-2">
                  {[
                    "What are today's top market stories?",
                    "Analyze AAPL's recent performance",
                    "Explain the yield curve inversion",
                    "What does the latest CPI data mean?",
                  ].map((suggestion) => (
                    <Button
                      key={suggestion}
                      variant="outline"
                      size="sm"
                      className="text-xs"
                      onClick={() => setInput(suggestion)}
                    >
                      {suggestion}
                    </Button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex gap-3 ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {msg.role === "assistant" && (
                    <div className="mt-1 shrink-0">
                      <Bot className="h-5 w-5 text-blue-500" />
                    </div>
                  )}
                  <div
                    className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
                      msg.role === "user"
                        ? "bg-zinc-900 text-zinc-50 dark:bg-zinc-100 dark:text-zinc-900"
                        : "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-50"
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>
                  {msg.role === "user" && (
                    <div className="mt-1 shrink-0">
                      <User className="h-5 w-5 text-zinc-400" />
                    </div>
                  )}
                </div>
              ))}
              {isLoading &&
                messages[messages.length - 1]?.role !== "assistant" && (
                  <div className="flex gap-3">
                    <Bot className="mt-1 h-5 w-5 text-blue-500" />
                    <div className="rounded-lg bg-zinc-100 px-4 py-2 dark:bg-zinc-800">
                      <div className="flex gap-1">
                        <span className="animate-bounce text-zinc-400">.</span>
                        <span className="animate-bounce text-zinc-400" style={{ animationDelay: "0.1s" }}>.</span>
                        <span className="animate-bounce text-zinc-400" style={{ animationDelay: "0.2s" }}>.</span>
                      </div>
                    </div>
                  </div>
                )}
            </div>
          )}
        </ScrollArea>

        {/* Error */}
        {error && (
          <div className="mx-4 mb-2 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-600 dark:border-red-900 dark:bg-red-950 dark:text-red-400">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {/* Input */}
        <div className="border-t border-zinc-200 p-4 dark:border-zinc-800">
          <div className="flex gap-2">
            <Input
              placeholder="Ask about financial news, markets, or filings..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
              disabled={isLoading}
            />
            <Button onClick={sendMessage} disabled={isLoading || !input.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}

export default function ResearchPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <ResearchContent />
    </Suspense>
  );
}
