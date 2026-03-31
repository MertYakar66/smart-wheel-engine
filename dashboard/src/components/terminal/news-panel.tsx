"use client";

import { TerminalPanel, TerminalBadge } from "./panel";
import type { StoryCard } from "@/types";

interface NewsPanelProps {
  stories: StoryCard[];
  loading: boolean;
  onRefresh: () => void;
  refreshing: boolean;
  onSelectStory: (story: StoryCard) => void;
}

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

const FACTOR_COLORS: Record<string, string> = {
  rates: "text-blue-400",
  oil: "text-amber-400",
  fx: "text-purple-400",
  regulation: "text-red-400",
  earnings: "text-green-400",
  demand: "text-cyan-400",
  supply_chain: "text-orange-400",
  geopolitical: "text-rose-400",
  monetary_policy: "text-indigo-400",
  labor: "text-teal-400",
};

export function NewsPanel({
  stories,
  loading,
  onRefresh,
  refreshing,
  onSelectStory,
}: NewsPanelProps) {
  return (
    <TerminalPanel
      title="Story Intelligence"
      tag="GRAPH"
      headerRight={
        <button
          onClick={onRefresh}
          disabled={refreshing}
          className="text-[10px] text-terminal-amber hover:text-terminal-text disabled:text-terminal-dim"
        >
          {refreshing ? "INGESTING..." : "REFRESH"}
        </button>
      }
    >
      {loading ? (
        <div className="flex h-full items-center justify-center text-terminal-dim">
          Loading headlines...
        </div>
      ) : stories.length === 0 ? (
        <div className="flex h-full flex-col items-center justify-center gap-2 text-terminal-dim">
          <span>No stories loaded.</span>
          <button
            onClick={onRefresh}
            className="text-terminal-amber hover:underline"
          >
            Run REFRESH to ingest feeds
          </button>
        </div>
      ) : (
        <div className="space-y-1">
          {stories.map((story) => {
            const tickers = story.entities
              .filter((e) => e.entityType === "ticker")
              .map((e) => e.entityValue);
            const impactTags = story.impactTags || [];

            return (
              <button
                key={story.storyId}
                onClick={() => onSelectStory(story)}
                className="block w-full text-left hover:bg-terminal-border/30 px-1 py-0.5 transition-colors"
              >
                <div className="flex items-start gap-1">
                  <span className={`mt-[2px] ${
                    story.contradictionFlag ? "text-terminal-red" :
                    story.storyStatus === "evolving" ? "text-terminal-amber" :
                    "text-terminal-green"
                  }`}>
                    {story.contradictionFlag ? "▲" : story.storyStatus === "evolving" ? "◆" : "●"}
                  </span>
                  <div className="flex-1 min-w-0">
                    <div className="truncate text-terminal-text">
                      {story.canonicalTitle}
                    </div>
                    <div className="flex items-center gap-2 text-[10px] flex-wrap">
                      {tickers.slice(0, 3).map((t) => (
                        <span key={t} className="text-terminal-blue">
                          ${t}
                        </span>
                      ))}
                      {impactTags.slice(0, 2).map((tag) => (
                        <span key={tag} className={FACTOR_COLORS[tag] || "text-terminal-dim"}>
                          [{tag.replace("_", " ")}]
                        </span>
                      ))}
                      {story.sector && !impactTags.length && (
                        <span className="text-terminal-dim">
                          [{story.sector}]
                        </span>
                      )}
                      <span className="text-terminal-dim">
                        {timeAgo(story.createdAt)}
                      </span>
                      {story.sourceCount > 1 && (
                        <TerminalBadge variant="default">
                          {story.sourceCount} src
                        </TerminalBadge>
                      )}
                      {story.corroborationScore !== undefined && story.corroborationScore > 0.5 && (
                        <TerminalBadge variant="green">
                          ✓
                        </TerminalBadge>
                      )}
                      {story.exposureRelevance !== undefined && story.exposureRelevance > 0 && (
                        <TerminalBadge variant="blue">
                          ★{story.exposureRelevance.toFixed(0)}
                        </TerminalBadge>
                      )}
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </TerminalPanel>
  );
}
