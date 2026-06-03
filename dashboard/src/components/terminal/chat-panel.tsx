"use client";

import { useState, useRef, useEffect } from "react";
import { TerminalPanel } from "./panel";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatPanelProps {
  initialQuery?: string;
}

export function ChatPanel({ initialQuery }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState(initialQuery || "");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (initialQuery && initialQuery !== input) {
      setInput(initialQuery);
    }
    // Intentionally only re-sync when a NEW initialQuery arrives; including
    // `input` would clobber the user's in-progress typing.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialQuery]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: input.trim() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: newMessages }),
      });

      if (!res.ok) {
        throw new Error("Chat request failed. Is Ollama running?");
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      let assistantContent = "";
      setMessages([...newMessages, { role: "assistant", content: "" }]);

      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // The server responds via toTextStreamResponse() — a raw UTF-8 text
        // stream of assistant tokens (no SSE/data-stream framing), so each
        // chunk is plain text to append directly.
        assistantContent += decoder.decode(value, { stream: true });
        setMessages([
          ...newMessages,
          { role: "assistant", content: assistantContent },
        ]);
      }

      if (!assistantContent) {
        setMessages([
          ...newMessages,
          {
            role: "assistant",
            content: "No response. Ensure Ollama is running.",
          },
        ]);
      }
    } catch {
      setMessages([
        ...newMessages,
        {
          role: "assistant",
          content: "⚠ Error: Could not reach Ollama. Check connection.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <TerminalPanel title="Research" tag="AI CHAT">
      <div className="flex h-full flex-col">
        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-2 mb-2">
          {messages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-terminal-dim text-[11px]">
              Ask financial research questions. Powered by local Ollama.
            </div>
          ) : (
            messages.map((msg, i) => (
              <div key={i}>
                <span
                  className={
                    msg.role === "user"
                      ? "text-terminal-amber"
                      : "text-terminal-green"
                  }
                >
                  {msg.role === "user" ? "YOU" : "AI"}:{" "}
                </span>
                <span className="text-terminal-text whitespace-pre-wrap">
                  {msg.content}
                </span>
                {isLoading &&
                  i === messages.length - 1 &&
                  msg.role === "assistant" &&
                  !msg.content && (
                    <span className="text-terminal-amber animate-pulse">
                      █
                    </span>
                  )}
              </div>
            ))
          )}
          {isLoading && messages[messages.length - 1]?.role === "user" && (
            <div>
              <span className="text-terminal-green">AI: </span>
              <span className="text-terminal-amber animate-pulse">█</span>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="flex items-center gap-1 border-t border-terminal-border pt-1.5">
          <span className="text-terminal-amber text-[10px]">ASK&gt;</span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) =>
              e.key === "Enter" && !e.shiftKey && sendMessage()
            }
            placeholder="Ask about markets, filings, macro..."
            className="flex-1 bg-transparent text-[11px] text-terminal-text placeholder:text-terminal-dim/40 outline-none caret-terminal-amber"
            disabled={isLoading}
          />
        </div>
      </div>
    </TerminalPanel>
  );
}
