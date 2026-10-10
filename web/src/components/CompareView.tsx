"use client";

import { useState } from "react";
import {
  ArrowRight,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  Zap,
} from "lucide-react";
import { SearchResultCard, SearchResultItem } from "./SearchResultCard";

interface CompareSystemResult {
  name: string;
  model_slug: string;
  latency_ms: number;
  results: SearchResultItem[];
}

interface CompareViewProps {
  systems: Record<string, CompareSystemResult>;
  query: string;
  totalLatencyMs: number;
  onInspectUnit?: (unitId: string) => void;
}

export function CompareView({
  systems,
  query,
  totalLatencyMs,
  onInspectUnit,
}: CompareViewProps) {
  const [selectedMobileTab, setSelectedMobileTab] = useState<string>("tfidf");

  const systemKeys = Object.keys(systems);
  if (systemKeys.length === 0) {
    return (
      <div className="rounded-2xl border border-white/5 bg-slate-900/40 p-8 text-center text-slate-400">
        No comparative results returned.
      </div>
    );
  }

  // Model-specific styles & icons
  const getSystemMeta = (slug: string) => {
    switch (slug) {
      case "tfidf":
        return {
          title: "TF-IDF Lexical",
          subtitle: "Sublinear token n-grams",
          badge: "Lexical Baseline",
          badgeColor: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
          icon: Zap,
        };
      case "semantic":
        return {
          title: "General Semantic",
          subtitle: "all-MiniLM-L6-v2 (384-dim)",
          badge: "Sentence-Transformers",
          badgeColor: "bg-indigo-500/10 text-indigo-300 border-indigo-500/20",
          icon: Sparkles,
        };
      case "code":
        return {
          title: "Code Transformer",
          subtitle: "st-codesearch (768-dim)",
          badge: "CodeSearchNet",
          badgeColor: "bg-violet-500/10 text-violet-300 border-violet-500/20",
          icon: Cpu,
        };
      default:
        return {
          title: slug,
          subtitle: "Graph Reranked",
          badge: "Hybrid Graph",
          badgeColor: "bg-amber-500/10 text-amber-300 border-amber-500/20",
          icon: Layers,
        };
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview Header Card */}
      <div className="glass-panel rounded-2xl p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-indigo-400 font-semibold font-mono">
                Multi-Regime IR Comparison
              </span>
              <span className="text-slate-600">•</span>
              <span className="text-xs text-slate-400">Query: &quot;{query}&quot;</span>
            </div>
            <h2 className="text-lg font-bold text-white mt-1">
              3-Way Simultaneous Retrieval Comparison
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-slate-900/80 px-3 py-1.5 border border-white/5 text-xs text-slate-300">
              <span className="text-slate-400">Total Round-trip:</span>{" "}
              <span className="font-mono font-semibold text-emerald-400">
                {totalLatencyMs.toFixed(1)}ms
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Tab Selector (shown only on small screens) */}
      <div className="flex lg:hidden overflow-x-auto gap-2 pb-2">
        {systemKeys.map((key) => {
          const sys = systems[key];
          const meta = getSystemMeta(key);
          const isSelected = selectedMobileTab === key;
          return (
            <button
              key={key}
              onClick={() => setSelectedMobileTab(key)}
              className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold whitespace-nowrap transition border ${
                isSelected
                  ? "bg-indigo-600 text-white border-indigo-500"
                  : "bg-slate-900/60 text-slate-400 border-white/5 hover:bg-slate-800"
              }`}
            >
              <span>{meta.title}</span>
              <span className="font-mono text-[10px] opacity-80">
                ({sys.latency_ms.toFixed(1)}ms)
              </span>
            </button>
          );
        })}
      </div>

      {/* Side-by-Side 3-Way Columns Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {systemKeys.map((key) => {
          const sys = systems[key];
          const meta = getSystemMeta(key);
          const Icon = meta.icon;

          return (
            <div
              key={key}
              className={`flex flex-col rounded-2xl border border-white/10 bg-[#0a0f1d]/90 shadow-xl overflow-hidden ${
                selectedMobileTab !== key ? "hidden lg:flex" : "flex"
              }`}
            >
              {/* Column Header */}
              <div className="border-b border-white/5 bg-slate-900/90 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                      <Icon className="h-4 w-4" />
                    </div>
                    <div>
                      <h3 className="font-bold text-sm text-white">{meta.title}</h3>
                      <p className="text-[11px] text-slate-400">{meta.subtitle}</p>
                    </div>
                  </div>

                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-semibold border ${meta.badgeColor}`}
                  >
                    {sys.latency_ms.toFixed(1)} ms
                  </span>
                </div>
              </div>

              {/* Column Results List */}
              <div className="p-3 sm:p-4 space-y-4 flex-1 overflow-y-auto max-h-[850px]">
                {sys.results.length === 0 ? (
                  <div className="p-8 text-center text-xs text-slate-500 italic">
                    No results returned for this engine.
                  </div>
                ) : (
                  sys.results.map((item, idx) => (
                    <SearchResultCard
                      key={`${key}-${item.unit_id}`}
                      result={item}
                      rank={idx + 1}
                      searchQuery={query}
                      onInspectUnit={onInspectUnit}
                    />
                  ))
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
