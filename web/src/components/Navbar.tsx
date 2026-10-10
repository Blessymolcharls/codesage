"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Activity, BarChart3, Database, GitBranch, Layers, Search, Sparkles } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function Navbar() {
  const pathname = usePathname();
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [unitCount, setUnitCount] = useState<number | null>(null);

  useEffect(() => {
    let isMounted = true;
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
        if (res.ok) {
          const data = await res.json();
          if (isMounted) {
            setBackendStatus("online");
            setUnitCount(data.indexed_units || 89);
          }
        } else {
          if (isMounted) setBackendStatus("offline");
        }
      } catch {
        if (isMounted) setBackendStatus("offline");
      }
    }

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#090d16]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6 lg:px-8">
        {/* Brand */}
        <div className="flex items-center gap-6">
          <Link href="/" className="group flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-indigo-600 to-violet-700 shadow-glow transition group-hover:scale-105">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold tracking-tight text-white group-hover:text-indigo-300 transition">
                  CodeSage
                </span>
                <span className="rounded-full bg-indigo-500/10 px-2 py-0.5 text-[10px] font-medium text-indigo-400 border border-indigo-500/20">
                  IR Intelligence
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                DocuCharm Multi-Regime Code Retrieval
              </p>
            </div>
          </Link>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 rounded-xl bg-slate-900/80 p-1 border border-white/5">
            <Link
              href="/"
              className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-medium transition ${
                pathname === "/"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Search className="h-3.5 w-3.5" />
              <span>Repository Search</span>
            </Link>
            <Link
              href="/evaluation"
              className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-medium transition ${
                pathname === "/evaluation"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" />
              <span>Evaluation & Study</span>
            </Link>
          </nav>
        </div>

        {/* Right Info Badges */}
        <div className="flex items-center gap-3">
          {/* Workspace info */}
          <div className="hidden md:flex items-center gap-2 rounded-lg border border-white/5 bg-slate-900/60 px-2.5 py-1 text-xs text-slate-300">
            <Database className="h-3.5 w-3.5 text-cyan-400" />
            <span>DocuCharm</span>
            {unitCount !== null && (
              <span className="rounded bg-slate-800 px-1.5 py-0.2 text-[10px] text-slate-400 font-mono">
                {unitCount} units
              </span>
            )}
          </div>

          {/* Backend Status indicator */}
          <div className="flex items-center gap-2 rounded-lg border border-white/5 bg-slate-900/60 px-2.5 py-1 text-xs">
            <span
              className={`h-2 w-2 rounded-full ${
                backendStatus === "online"
                  ? "bg-emerald-400 animate-pulse shadow-[0_0_8px_rgba(52,211,153,0.8)]"
                  : backendStatus === "checking"
                  ? "bg-amber-400 animate-ping"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-slate-300 hidden sm:inline">
              {backendStatus === "online"
                ? "Backend Ready"
                : backendStatus === "checking"
                ? "Connecting..."
                : "Backend Offline"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
