"use client";

import { useState } from "react";
import {
  Code2,
  Copy,
  ExternalLink,
  GitCommit,
  Hash,
  Network,
  Tag,
  Check,
} from "lucide-react";
import { CodeViewer } from "./CodeViewer";

export interface SearchResultItem {
  unit_id: string;
  score: number;
  rel_path: string;
  language: string;
  symbol_name: string | null;
  unit_type: string;
  start_line: number;
  end_line: number;
  extraction_method: string;
  preview: string;
  source_code: string;
  model_ranks?: Record<string, number>;
  rrf_score?: number;
  graph_boost?: number;
  graph_score?: number;
}

interface SearchResultCardProps {
  result: SearchResultItem;
  rank: number;
  searchQuery?: string;
  onInspectUnit?: (unitId: string) => void;
}

export function SearchResultCard({
  result,
  rank,
  searchQuery = "",
  onInspectUnit,
}: SearchResultCardProps) {
  const [copiedPath, setCopiedPath] = useState(false);

  const handleCopyPath = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(result.rel_path);
      setCopiedPath(true);
      setTimeout(() => setCopiedPath(false), 2000);
    } catch {
      // Fallback
    }
  };

  // Determine score color badge
  const scoreBadgeColor =
    result.score >= 0.75
      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
      : result.score >= 0.50
      ? "bg-indigo-500/10 text-indigo-400 border-indigo-500/30"
      : result.score >= 0.30
      ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
      : "bg-slate-500/10 text-slate-400 border-slate-500/30";

  // Determine unit type badge color
  const typeBadgeColor =
    result.unit_type === "class"
      ? "bg-purple-500/10 text-purple-300 border-purple-500/20"
      : result.unit_type === "function"
      ? "bg-cyan-500/10 text-cyan-300 border-cyan-500/20"
      : result.unit_type === "method"
      ? "bg-blue-500/10 text-blue-300 border-blue-500/20"
      : "bg-slate-500/10 text-slate-300 border-slate-500/20";

  return (
    <article className="glass-panel glass-panel-hover rounded-2xl p-5 shadow-lg transition-all duration-200">
      {/* Top Header Row */}
      <div className="flex flex-wrap items-start justify-between gap-3 pb-3 border-b border-white/5">
        {/* Left: Rank, Symbol & Type */}
        <div className="flex items-center gap-2.5 flex-wrap">
          {/* Rank Badge */}
          <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-slate-800 text-xs font-bold text-slate-300 border border-white/5 shadow-inner">
            #{rank}
          </span>

          {/* Symbol Name */}
          <h3 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
            {result.symbol_name ? (
              <span className="text-indigo-200 hover:text-indigo-100 transition">
                {result.symbol_name}
              </span>
            ) : (
              <span className="italic text-slate-400">Anonymous Unit</span>
            )}
          </h3>

          {/* Unit Type Badge */}
          <span
            className={`rounded-full px-2.5 py-0.5 text-[11px] font-medium border capitalize ${typeBadgeColor}`}
          >
            {result.unit_type}
          </span>

          {/* Language Badge */}
          <span className="rounded-full bg-slate-800/80 px-2.5 py-0.5 text-[11px] font-medium text-slate-300 border border-white/5 uppercase tracking-wider">
            {result.language}
          </span>
        </div>

        {/* Right: Similarity Score & Graph Action */}
        <div className="flex items-center gap-2">
          {onInspectUnit && (
            <button
              onClick={() => onInspectUnit(result.unit_id)}
              className="flex items-center gap-1 rounded-lg border border-white/10 bg-slate-800/60 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-700 hover:text-white transition"
              title="Inspect Graph Neighborhood"
            >
              <Network className="h-3.5 w-3.5 text-amber-400" />
              <span className="hidden sm:inline">Graph</span>
            </button>
          )}

          {/* Score Badge */}
          <div
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-mono font-semibold ${scoreBadgeColor}`}
            title="Similarity Relevance Score"
          >
            <span className="text-[10px] uppercase tracking-wider text-slate-400 font-sans">
              Score
            </span>
            <span>{result.score.toFixed(4)}</span>
          </div>
        </div>
      </div>

      {/* Path and Line Range Row */}
      <div className="flex items-center justify-between gap-2 py-2.5 text-xs text-slate-400">
        {/* Clickable File Path */}
        <div className="flex items-center gap-1.5 font-mono text-slate-300 truncate">
          <Code2 className="h-3.5 w-3.5 shrink-0 text-slate-400" />
          <span className="truncate">{result.rel_path}</span>
          <button
            onClick={handleCopyPath}
            className="text-slate-500 hover:text-slate-300 transition p-0.5"
            title="Copy path"
          >
            {copiedPath ? (
              <Check className="h-3 w-3 text-emerald-400" />
            ) : (
              <Copy className="h-3 w-3" />
            )}
          </button>
        </div>

        {/* Line range */}
        <div className="shrink-0 font-mono text-[11px] text-slate-400 bg-slate-900/80 px-2 py-0.5 rounded border border-white/5">
          L{result.start_line}–L{result.end_line}
        </div>
      </div>

      {/* Hybrid RRF Provenance Details */}
      {result.model_ranks && Object.keys(result.model_ranks).length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap pb-2.5 text-[11px]">
          <span className="text-slate-500 font-medium">RRF Consensus:</span>
          {Object.entries(result.model_ranks).map(([modelKey, rankNum]) => (
            <span
              key={modelKey}
              className="rounded bg-indigo-950/60 border border-indigo-500/30 px-2 py-0.5 text-indigo-300 font-mono"
            >
              {modelKey}: #{rankNum}
            </span>
          ))}
          {result.graph_boost !== undefined && result.graph_boost !== 0 && (
            <span
              className={`rounded px-2 py-0.5 font-mono ${
                result.graph_boost > 0
                  ? "bg-amber-950/60 border border-amber-500/30 text-amber-300"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              graph: {result.graph_boost > 0 ? `+${result.graph_boost.toFixed(4)}` : result.graph_boost.toFixed(4)}
            </span>
          )}
        </div>
      )}

      {/* Syntax-Highlighted Source Preview */}
      <div className="mt-2">
        <CodeViewer
          code={result.source_code || result.preview}
          language={result.language}
          startLine={result.start_line}
          maxPreviewLines={8}
          highlightSymbol={result.symbol_name}
        />
      </div>
    </article>
  );
}
