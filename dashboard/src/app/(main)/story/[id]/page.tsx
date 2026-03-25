"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Clock,
  ExternalLink,
  AlertTriangle,
  CheckCircle,
  TrendingUp,
  BarChart3,
  Users,
  FileText,
  ArrowLeft,
  GitBranch,
  Layers,
  Zap,
} from "lucide-react";
import type { StoryDetail, StoryTimelineEvent, SourceRefDetailed } from "@/types";

const IMPACT_TAG_COLORS: Record<string, string> = {
  rates: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300",
  oil: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  fx: "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300",
  regulation: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  earnings: "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
  demand: "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-300",
  supply_chain: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  geopolitical: "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300",
  monetary_policy: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/30 dark:text-indigo-300",
  labor: "bg-teal-100 text-teal-800 dark:bg-teal-900/30 dark:text-teal-300",
};

const STATUS_ICONS: Record<string, typeof CheckCircle> = {
  developing: Clock,
  evolving: GitBranch,
  resolved: CheckCircle,
};

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function TimelineItem({ event }: { event: StoryTimelineEvent }) {
  const iconMap: Record<string, string> = {
    created: "bg-blue-500",
    source_added: "bg-green-500",
    merged: "bg-purple-500",
    status_change: "bg-amber-500",
    contradiction_detected: "bg-red-500",
  };

  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <div className={`h-3 w-3 rounded-full ${iconMap[event.eventType] || "bg-zinc-400"}`} />
        <div className="w-px flex-1 bg-zinc-200 dark:bg-zinc-700" />
      </div>
      <div className="pb-4">
        <p className="text-sm text-zinc-700 dark:text-zinc-300">{event.description}</p>
        <p className="text-xs text-zinc-400">{timeAgo(event.occurredAt)}</p>
      </div>
    </div>
  );
}

function SourceCard({ source }: { source: SourceRefDetailed }) {
  return (
    <div className="rounded-lg border border-zinc-200 p-3 dark:border-zinc-700">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1">
          <a
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-zinc-900 hover:text-blue-600 dark:text-zinc-100 dark:hover:text-blue-400"
          >
            {source.headline}
            <ExternalLink className="ml-1 inline h-3 w-3" />
          </a>
          <div className="mt-1 flex items-center gap-2 text-xs text-zinc-500">
            <span>{source.publisher}</span>
            {source.publishedAt && (
              <>
                <span>·</span>
                <span>{timeAgo(source.publishedAt)}</span>
              </>
            )}
            {source.sentiment && (
              <Badge
                variant="outline"
                className={
                  source.sentiment === "positive" ? "border-green-300 text-green-600" :
                  source.sentiment === "negative" ? "border-red-300 text-red-600" :
                  source.sentiment === "mixed" ? "border-amber-300 text-amber-600" :
                  "border-zinc-300 text-zinc-600"
                }
              >
                {source.sentiment}
              </Badge>
            )}
            {source.rightsRestricted && (
              <Badge variant="outline" className="border-orange-300 text-orange-600">
                restricted
              </Badge>
            )}
          </div>
          {source.snippet && (
            <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400 line-clamp-2">
              {source.snippet}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

export default function StoryDetailPage() {
  const params = useParams();
  const storyId = params.id as string;
  const [story, setStory] = useState<StoryDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStory() {
      try {
        const res = await fetch(`/api/stories/${storyId}`);
        if (res.ok) {
          const data = await res.json();
          setStory(data);
        }
      } catch (err) {
        console.error("Failed to fetch story:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchStory();
  }, [storyId]);

  if (loading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-40" />
          <Skeleton className="h-40" />
        </div>
      </div>
    );
  }

  if (!story) {
    return (
      <div className="text-center py-12">
        <p className="text-zinc-500">Story not found.</p>
        <Link href="/feed">
          <Button variant="ghost" className="mt-4">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back to Feed
          </Button>
        </Link>
      </div>
    );
  }

  const StatusIcon = STATUS_ICONS[story.storyStatus || "developing"] || Clock;

  return (
    <div className="space-y-6">
      {/* Back button */}
      <Link href="/feed">
        <Button variant="ghost" size="sm">
          <ArrowLeft className="mr-2 h-4 w-4" /> Feed
        </Button>
      </Link>

      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-2">
          <Badge variant="outline" className="capitalize">
            <StatusIcon className="mr-1 h-3 w-3" />
            {story.storyStatus || "developing"}
          </Badge>
          {story.contradictionFlag && (
            <Badge variant="destructive">
              <AlertTriangle className="mr-1 h-3 w-3" />
              Contradicting sources
            </Badge>
          )}
          {story.sourceCount > 1 && (
            <Badge variant="outline">
              <Layers className="mr-1 h-3 w-3" />
              {story.sourceCount} sources
            </Badge>
          )}
          {story.corroborationScore !== undefined && story.corroborationScore > 0 && (
            <Badge variant="outline" className="border-green-300 text-green-600">
              <CheckCircle className="mr-1 h-3 w-3" />
              {(story.corroborationScore * 100).toFixed(0)}% corroborated
            </Badge>
          )}
        </div>

        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50">
          {story.canonicalTitle}
        </h1>

        <p className="mt-1 text-sm text-zinc-500">
          {timeAgo(story.createdAt)}
          {story.sector && <> · {story.sector}</>}
          {story.impactHorizon && <> · Horizon: {story.impactHorizon}</>}
        </p>
      </div>

      {/* Impact Tags */}
      {story.impactTags && story.impactTags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {story.impactTags.map((tag) => (
            <span
              key={tag}
              className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${IMPACT_TAG_COLORS[tag] || "bg-zinc-100 text-zinc-800"}`}
            >
              <Zap className="mr-1 h-3 w-3" />
              {tag.replace("_", " ")}
            </span>
          ))}
        </div>
      )}

      {/* Why It Matters */}
      {story.whyItMatters && (
        <Card className="border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-blue-800 dark:text-blue-300">
              Why It Matters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-blue-900 dark:text-blue-200">
              {story.whyItMatters}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Summary & Entity Tags */}
      {story.summary && (
        <p className="text-sm text-zinc-600 dark:text-zinc-400">{story.summary}</p>
      )}

      <div className="flex flex-wrap gap-1.5">
        {story.entities
          .filter((e) => e.entityType === "ticker")
          .map((e) => (
            <Link key={e.entityValue} href={`/ticker/${e.entityValue}`}>
              <Badge variant="ticker" className="cursor-pointer">
                ${e.entityValue}
              </Badge>
            </Link>
          ))}
        {story.entities
          .filter((e) => e.entityType === "topic")
          .map((e) => (
            <Badge key={e.entityValue} variant="secondary">
              {e.entityValue}
            </Badge>
          ))}
      </div>

      {/* Story-to-Workstation Bridge Buttons */}
      <div className="flex flex-wrap gap-2">
        <Link href={`/research?story=${story.storyId}&title=${encodeURIComponent(story.canonicalTitle)}`}>
          <Button variant="outline" size="sm">
            <TrendingUp className="mr-2 h-4 w-4" />
            Explain Market Impact
          </Button>
        </Link>
        {story.entities.filter((e) => e.entityType === "ticker").length > 0 && (
          <Link href={`/ticker/${story.entities.find((e) => e.entityType === "ticker")!.entityValue}`}>
            <Button variant="outline" size="sm">
              <BarChart3 className="mr-2 h-4 w-4" />
              Show Price Reaction
            </Button>
          </Link>
        )}
        {story.entities.filter((e) => e.entityType === "ticker").length > 1 && (
          <Link
            href={`/research?story=${story.storyId}&title=${encodeURIComponent("Compare peers: " + story.entities.filter((e) => e.entityType === "ticker").map((e) => e.entityValue).join(", "))}`}
          >
            <Button variant="outline" size="sm">
              <Users className="mr-2 h-4 w-4" />
              Compare Peers
            </Button>
          </Link>
        )}
        <a
          href={story.sources[0]?.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Button variant="outline" size="sm">
            <FileText className="mr-2 h-4 w-4" />
            Primary Sources
          </Button>
        </a>
      </div>

      {/* Two-column layout: Sources & Timeline */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Sources (2/3 width) */}
        <div className="lg:col-span-2 space-y-3">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50">
            Sources ({story.allSources.length})
          </h2>
          {story.allSources.map((source) => (
            <SourceCard key={source.sourceId} source={source} />
          ))}
        </div>

        {/* Timeline (1/3 width) */}
        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
            Timeline
          </h2>
          {story.timeline.length > 0 ? (
            <div>
              {story.timeline
                .sort((a, b) => new Date(b.occurredAt).getTime() - new Date(a.occurredAt).getTime())
                .map((event) => (
                  <TimelineItem key={event.id} event={event} />
                ))}
            </div>
          ) : (
            <p className="text-sm text-zinc-400">No timeline events yet.</p>
          )}
        </div>
      </div>

      {/* Exposure Mechanisms */}
      {story.exposureMechanisms && story.exposureMechanisms.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Exposure Mechanisms</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {story.exposureMechanisms.map((m, i) => (
                <div key={i} className="flex items-center justify-between text-sm">
                  <span className="capitalize">{m.factor.replace("_", " ")}</span>
                  <div className="flex items-center gap-2">
                    <span className={
                      m.direction === "positive" ? "text-green-600" :
                      m.direction === "negative" ? "text-red-600" :
                      "text-zinc-500"
                    }>
                      {m.direction === "positive" ? "+" : m.direction === "negative" ? "-" : "?"}
                    </span>
                    <div className="w-20 bg-zinc-200 rounded-full h-1.5 dark:bg-zinc-700">
                      <div
                        className="bg-blue-600 h-1.5 rounded-full"
                        style={{ width: `${m.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-zinc-400">{(m.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Related Stories */}
      {story.relatedStories.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
            Related Stories
          </h2>
          <div className="space-y-2">
            {story.relatedStories.map((related) => (
              <Link key={related.storyId} href={`/story/${related.storyId}`}>
                <Card className="transition-shadow hover:shadow-md cursor-pointer">
                  <CardContent className="py-3">
                    <p className="text-sm font-medium">{related.canonicalTitle}</p>
                    <div className="flex gap-1.5 mt-1">
                      {related.sector && (
                        <Badge variant="secondary" className="text-xs">{related.sector}</Badge>
                      )}
                      <span className="text-xs text-zinc-400">{timeAgo(related.createdAt)}</span>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
