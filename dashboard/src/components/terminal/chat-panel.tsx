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

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("0:")) {
            try {
              const text = JSON.parse(line.slice(2));
              assistantContent += text;
              setMessages([
                ...newMessages,
                { role: "assistant", content: assistantContent },
              ]);
            } catch {
              // skip
            }
          }
        }
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
