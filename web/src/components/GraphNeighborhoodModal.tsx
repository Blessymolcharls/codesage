"use client";

import { useEffect, useState } from "react";
import { ArrowLeftRight, ArrowRight, Code2, Network, X } from "lucide-react";

interface GraphNeighborhoodModalProps {
  unitId: string | null;
  onClose: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function GraphNeighborhoodModal({ unitId, onClose }: GraphNeighborhoodModalProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!unitId) return;

    let isMounted = true;
    async function fetchUnit() {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`${API_BASE}/api/units/${encodeURIComponent(unitId!)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load unit`);
        const json = await res.json();
        if (isMounted) setData(json);
      } catch (err: any) {
        if (isMounted) setError(err.message || "Failed to load graph relationships");
      } finally {
        if (isMounted) setLoading(false);
      }
    }

    fetchUnit();
    return () => {
      isMounted = false;
    };
  }, [unitId]);

  if (!unitId) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fadeIn">
      <div className="relative w-full max-w-2xl rounded-2xl border border-white/10 bg-[#0c1222] p-6 shadow-2xl">
        {/* Modal Header */}
        <div className="flex items-start justify-between border-b border-white/5 pb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
              <Network className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Repository Graph Context</h3>
              <p className="text-xs text-slate-400 font-mono truncate max-w-md">{unitId}</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="mt-4 max-h-[70vh] overflow-y-auto space-y-4 pr-1 text-xs">
          {loading && (
            <div className="p-8 text-center text-slate-400">Loading graph relationships...</div>
          )}

          {error && (
            <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4 text-rose-300">
              {error}
            </div>
          )}

          {data && (
            <>
              {/* Unit Info Box */}
              <div className="rounded-xl border border-white/5 bg-slate-900/60 p-4">
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-semibold block">Symbol</span>
                    <span className="font-semibold text-white">{data.unit.symbol_name || "Anonymous"}</span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-semibold block">Type</span>
                    <span className="capitalize text-slate-300">{data.unit.unit_type}</span>
                  </div>
                  <div>
                    <span className="text-[10px] uppercase text-slate-500 font-semibold block">Language</span>
                    <span className="uppercase text-slate-300">{data.unit.language}</span>
                  </div>
                  <div className="col-span-2 sm:col-span-3">
                    <span className="text-[10px] uppercase text-slate-500 font-semibold block">Path & Range</span>
                    <span className="font-mono text-slate-300">
                      {data.unit.repository_relative_path} (L{data.unit.start_line}–L{data.unit.end_line})
                    </span>
                  </div>
                </div>
              </div>

              {/* Outgoing Edges */}
              <div>
                <h4 className="font-semibold text-slate-300 mb-2 flex items-center gap-1.5 text-xs">
                  <ArrowRight className="h-3.5 w-3.5 text-indigo-400" />
                  <span>Outgoing Relationships ({data.graph_context.outgoing.length})</span>
                </h4>
                {data.graph_context.outgoing.length === 0 ? (
                  <p className="text-slate-500 italic">No outgoing syntactic relationships recorded.</p>
                ) : (
                  <div className="space-y-1.5">
                    {data.graph_context.outgoing.map((e: any, idx: number) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between rounded-lg border border-white/5 bg-slate-900/40 px-3 py-2 text-slate-300 font-mono text-[11px]"
                      >
                        <span className="font-semibold text-indigo-300">{e.relationship}</span>
                        <span className="text-slate-400 truncate max-w-xs">{e.target}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Incoming Edges */}
              <div>
                <h4 className="font-semibold text-slate-300 mb-2 flex items-center gap-1.5 text-xs">
                  <ArrowLeftRight className="h-3.5 w-3.5 text-emerald-400" />
                  <span>Incoming Relationships ({data.graph_context.incoming.length})</span>
                </h4>
                {data.graph_context.incoming.length === 0 ? (
                  <p className="text-slate-500 italic">No incoming syntactic relationships recorded.</p>
                ) : (
                  <div className="space-y-1.5">
                    {data.graph_context.incoming.map((e: any, idx: number) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between rounded-lg border border-white/5 bg-slate-900/40 px-3 py-2 text-slate-300 font-mono text-[11px]"
                      >
                        <span className="font-semibold text-emerald-300">{e.relationship}</span>
                        <span className="text-slate-400 truncate max-w-xs">{e.source}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
