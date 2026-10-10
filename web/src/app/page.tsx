"use client";

import { useEffect, useState } from "react";
import {
  AlertCircle,
  Cpu,
  Layers,
  Network,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Zap,
} from "lucide-react";
import { SearchResultCard, SearchResultItem } from "@/components/SearchResultCard";
import { CompareView } from "@/components/CompareView";
import { GraphNeighborhoodModal } from "@/components/GraphNeighborhoodModal";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const EXAMPLE_QUERIES = [
  "merge multiple PDF files",
  "cleanup temporary files",
  "validate uploaded PDF",
  "show custom snackbar error message",
  "PDF preview screen state management",
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [model, setModel] = useState<string>("tfidf");
  const [topK, setTopK] = useState<number>(10);
  const [rerank, setRerank] = useState<boolean>(false);
  const [compareMode, setCompareMode] = useState<boolean>(false);

  // Search Results State
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [singleResults, setSingleResults] = useState<SearchResultItem[]>([]);
  const [compareResults, setCompareResults] = useState<Record<string, any>>({});
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [totalCompareLatencyMs, setTotalCompareLatencyMs] = useState<number>(0);
  const [hasSearched, setHasSearched] = useState<boolean>(false);

  // Inspected unit for graph neighborhood modal
  const [inspectedUnitId, setInspectedUnitId] = useState<string | null>(null);

  // Trigger search execution
  const executeSearch = async (overrideQuery?: string) => {
    const activeQuery = (overrideQuery ?? query).trim();
    if (!activeQuery) return;

    setLoading(true);
    setError(null);
    setHasSearched(true);

    try {
      if (compareMode) {
        // Multi-engine comparison mode
        const res = await fetch(`${API_BASE}/api/compare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: activeQuery,
            top_k: topK,
            with_graph: rerank,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Search failed with status ${res.status}`);
        }

        const data = await res.json();
        setCompareResults(data.systems || {});
        setTotalCompareLatencyMs(data.total_latency_ms || 0);
      } else {
        // Single model search
        const res = await fetch(`${API_BASE}/api/search`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: activeQuery,
            model,
            top_k: topK,
            rerank,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || `Search failed with status ${res.status}`);
        }

        const data = await res.json();
        setSingleResults(data.results || []);
        setLatencyMs(data.latency_ms);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred while connecting to the backend API.");
    } finally {
      setLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    executeSearch();
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
    executeSearch(example);
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 space-y-8">
      {/* Top Hero Section */}
      <div className="text-center max-w-2xl mx-auto space-y-3">
        <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1 text-xs font-medium text-indigo-300">
          <Sparkles className="h-3.5 w-3.5" />
          <span>Code Intelligence & Retrieval System</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
          Repository Code Search
        </h1>
        <p className="text-sm text-slate-400">
          Query the DocuCharm codebase using lexical TF-IDF, general-purpose sentence
          embeddings, and code-specific Transformer representations with optional graph reranking.
        </p>
      </div>

      {/* Main Search Box Card */}
      <div className="glass-panel rounded-2xl p-4 sm:p-6 shadow-2xl space-y-4">
        <form onSubmit={handleFormSubmit} className="space-y-4">
          {/* Query Input Bar */}
          <div className="relative flex items-center">
            <Search className="absolute left-4 h-5 w-5 text-slate-400 pointer-events-none" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by intent, concept, or function (e.g. 'merge multiple PDF files')..."
              className="w-full rounded-xl border border-white/10 bg-slate-900/90 py-3.5 pl-12 pr-28 text-sm text-white placeholder-slate-500 shadow-inner focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 font-sans transition"
            />
            <div className="absolute right-2 flex items-center gap-2">
              <button
                type="submit"
                disabled={loading || !query.trim()}
                className="flex items-center gap-1.5 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-md hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {loading ? "Searching..." : "Search"}
              </button>
            </div>
          </div>

          {/* Controls Bar: Model, Top-K, Graph Toggle, Compare Mode */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-white/5 text-xs">
            {/* Left Controls: Model Select & Top-K */}
            <div className="flex flex-wrap items-center gap-3">
              {/* Compare Mode Toggle */}
              <div className="flex items-center rounded-lg border border-white/10 bg-slate-900/80 p-0.5">
                <button
                  type="button"
                  onClick={() => setCompareMode(false)}
                  className={`rounded-md px-3 py-1.5 font-medium transition ${
                    !compareMode
                      ? "bg-indigo-600 text-white shadow"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  Single Model
                </button>
                <button
                  type="button"
                  onClick={() => setCompareMode(true)}
                  className={`flex items-center gap-1 rounded-md px-3 py-1.5 font-medium transition ${
                    compareMode
                      ? "bg-indigo-600 text-white shadow"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <Layers className="h-3 w-3" />
                  <span>3-Way Compare</span>
                </button>
              </div>

              {/* Single Model Selector (disabled in compare mode) */}
              {!compareMode && (
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400 font-medium">Model:</span>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="rounded-lg border border-white/10 bg-slate-900 px-3 py-1.5 text-xs text-white focus:border-indigo-500 focus:outline-none"
                  >
                    <option value="tfidf">TF-IDF (Lexical Baseline)</option>
                    <option value="semantic">General Semantic (all-MiniLM-L6-v2)</option>
                    <option value="code">Code Transformer (st-codesearch)</option>
                  </select>
                </div>
              )}

              {/* Top-K Selector */}
              <div className="flex items-center gap-1.5">
                <span className="text-slate-400 font-medium">Top-K:</span>
                <select
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="rounded-lg border border-white/10 bg-slate-900 px-2.5 py-1.5 text-xs text-white focus:border-indigo-500 focus:outline-none"
                >
                  <option value={3}>3</option>
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={15}>15</option>
                  <option value={20}>20</option>
                </select>
              </div>
            </div>

            {/* Right Controls: Graph Reranking Switch */}
            <div className="flex items-center gap-2">
              <label
                htmlFor="graph-rerank-toggle"
                className="flex items-center gap-2 cursor-pointer rounded-lg border border-white/10 bg-slate-900/60 px-3 py-1.5 hover:bg-slate-900 transition"
              >
                <Network className={`h-3.5 w-3.5 ${rerank ? "text-amber-400" : "text-slate-500"}`} />
                <span className="text-slate-300 font-medium">Graph Reranker</span>
                <input
                  id="graph-rerank-toggle"
                  type="checkbox"
                  checked={rerank}
                  onChange={(e) => setRerank(e.target.checked)}
                  className="rounded border-slate-700 text-indigo-600 focus:ring-indigo-500 h-4 w-4 bg-slate-800"
                />
              </label>
            </div>
          </div>
        </form>

        {/* Quick Example Query Pills */}
        <div className="pt-2 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500">Suggested:</span>
          {EXAMPLE_QUERIES.map((example, idx) => (
            <button
              key={idx}
              onClick={() => handleExampleClick(example)}
              className="rounded-full border border-white/5 bg-slate-900/60 px-2.5 py-1 text-[11px] text-slate-400 hover:border-indigo-500/30 hover:bg-indigo-500/10 hover:text-indigo-300 transition"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 p-4 text-xs text-rose-300 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="h-4 w-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => executeSearch()}
            className="flex items-center gap-1 rounded bg-rose-500/20 px-2.5 py-1 text-rose-200 hover:bg-rose-500/30 transition"
          >
            <RotateCcw className="h-3 w-3" />
            <span>Retry</span>
          </button>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && (
        <div className="space-y-4">
          <div className="h-6 w-48 bg-slate-800/60 rounded animate-pulse" />
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="glass-panel rounded-2xl p-5 space-y-3 animate-pulse border border-white/5"
            >
              <div className="flex justify-between">
                <div className="h-4 w-1/3 bg-slate-800 rounded" />
                <div className="h-4 w-16 bg-slate-800 rounded" />
              </div>
              <div className="h-3 w-1/2 bg-slate-800/40 rounded" />
              <div className="h-28 bg-[#060911] rounded-xl" />
            </div>
          ))}
        </div>
      )}

      {/* Search Results Display */}
      {!loading && hasSearched && (
        <div>
          {compareMode ? (
            /* 3-Way Comparison View */
            <CompareView
              systems={compareResults}
              query={query}
              totalLatencyMs={totalCompareLatencyMs}
              onInspectUnit={(unitId) => setInspectedUnitId(unitId)}
            />
          ) : (
            /* Single Model Results List */
            <div className="space-y-4">
              {/* Results Header */}
              <div className="flex items-center justify-between text-xs text-slate-400 px-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white">
                    {singleResults.length} Results
                  </span>
                  <span>•</span>
                  <span>Model: <strong className="text-indigo-300 uppercase">{model}</strong></span>
                  {rerank && (
                    <span className="rounded bg-amber-500/10 text-amber-300 px-1.5 py-0.5 border border-amber-500/20 text-[10px]">
                      Graph Reranked
                    </span>
                  )}
                </div>

                {latencyMs !== null && (
                  <div className="font-mono text-emerald-400">
                    Latency: {latencyMs.toFixed(2)} ms
                  </div>
                )}
              </div>

              {/* Cards List */}
              {singleResults.length === 0 ? (
                <div className="glass-panel rounded-2xl p-12 text-center text-slate-400 space-y-2">
                  <Search className="h-8 w-8 mx-auto text-slate-600 mb-2" />
                  <p className="font-semibold text-white">No code units matched your query</p>
                  <p className="text-xs">
                    Try adjusting keywords, selecting another retrieval engine, or reducing Top-K filters.
                  </p>
                </div>
              ) : (
                singleResults.map((item, idx) => (
                  <SearchResultCard
                    key={item.unit_id}
                    result={item}
                    rank={idx + 1}
                    searchQuery={query}
                    onInspectUnit={(unitId) => setInspectedUnitId(unitId)}
                  />
                ))
              )}
            </div>
          )}
        </div>
      )}

      {/* Initial Empty State before searching */}
      {!loading && !hasSearched && (
        <div className="glass-panel rounded-2xl p-10 text-center space-y-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mx-auto">
            <Search className="h-6 w-6" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Ready to Explore Code</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto mt-1">
              Enter a natural-language description of functionality or choose one of the suggested
              queries above to compare retrieval behavior across lexical, semantic, and code models.
            </p>
          </div>
        </div>
      )}

      {/* Graph Neighborhood Modal */}
      <GraphNeighborhoodModal
        unitId={inspectedUnitId}
        onClose={() => setInspectedUnitId(null)}
      />
    </div>
  );
}
