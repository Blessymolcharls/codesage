"use client";

import React, { useEffect, useState } from "react";
import {
  Database,
  Plus,
  RefreshCw,
  Trash2,
  CheckCircle2,
  AlertTriangle,
  Folder,
  Layers,
  ChevronDown,
  X,
  Play,
  Clock,
  Sparkles,
} from "lucide-react";

export interface WorkspaceMetadata {
  id: string;
  name: string;
  display_path: string;
  repo_path: string;
  created_at: string;
  indexed_at: string | null;
  status: "unindexed" | "indexing" | "ready" | "error";
  unit_count: number;
  file_count: number;
  languages: string[];
  last_error: string | null;
  is_canonical?: boolean;
}

interface WorkspaceSelectorProps {
  apiBase: string;
  currentWorkspaceId: string;
  onSelectWorkspace: (workspace: WorkspaceMetadata) => void;
  className?: string;
}

export function WorkspaceSelector({
  apiBase,
  currentWorkspaceId,
  onSelectWorkspace,
  className = "",
}: WorkspaceSelectorProps) {
  const [workspaces, setWorkspaces] = useState<WorkspaceMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"list" | "register">("list");

  // Registration Form State
  const [newId, setNewId] = useState("");
  const [newPath, setNewPath] = useState("");
  const [newName, setNewName] = useState("");
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionSuccess, setActionSuccess] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Fetch workspaces
  const fetchWorkspaces = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${apiBase}/api/workspaces`, { cache: "no-store" });
      if (res.ok) {
        const data: WorkspaceMetadata[] = await res.json();
        setWorkspaces(data);
        const current = data.find((w) => w.id === currentWorkspaceId);
        if (current) {
          onSelectWorkspace(current);
        } else if (data.length > 0) {
          onSelectWorkspace(data[0]);
        }
      }
    } catch (e) {
      // Backend may be offline
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkspaces();
  }, [apiBase]);

  const currentWorkspace = workspaces.find((w) => w.id === currentWorkspaceId) || {
    id: currentWorkspaceId,
    name: currentWorkspaceId === "default" ? "DocuCharm (Canonical)" : currentWorkspaceId,
    display_path: "../DocuCharm",
    repo_path: "../DocuCharm",
    created_at: "",
    indexed_at: null,
    status: "ready",
    unit_count: 89,
    file_count: 16,
    languages: ["dart", "javascript"],
    last_error: null,
    is_canonical: currentWorkspaceId === "default",
  };

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setActionError(null);
    setActionSuccess(null);
    setIsProcessing(true);

    try {
      const res = await fetch(`${apiBase}/api/workspaces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: newId.trim(),
          repo_path: newPath.trim(),
          name: newName.trim() || undefined,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to register repository workspace");
      }

      const created: WorkspaceMetadata = await res.json();
      setActionSuccess(`Registered workspace '${created.name}' successfully.`);
      setNewId("");
      setNewPath("");
      setNewName("");
      await fetchWorkspaces();
      onSelectWorkspace(created);
      setActiveTab("list");
    } catch (err: any) {
      setActionError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTriggerIndex = async (workspaceId: string, incremental: boolean = false) => {
    setActionError(null);
    setActionSuccess(null);
    setIsProcessing(true);

    try {
      const res = await fetch(`${apiBase}/api/workspaces/${workspaceId}/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ incremental }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Indexing failed");
      }

      const updated: WorkspaceMetadata = await res.json();
      setActionSuccess(
        `Workspace '${updated.name}' indexed successfully (${updated.unit_count} code units).`
      );
      await fetchWorkspaces();
      if (updated.id === currentWorkspaceId) {
        onSelectWorkspace(updated);
      }
    } catch (err: any) {
      setActionError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDeleteWorkspace = async (workspaceId: string) => {
    if (!confirm(`Remove index for workspace '${workspaceId}'? Source files will NOT be touched.`)) {
      return;
    }
    setActionError(null);
    setActionSuccess(null);
    setIsProcessing(true);

    try {
      const res = await fetch(`${apiBase}/api/workspaces/${workspaceId}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to delete workspace index");
      }

      setActionSuccess(`Workspace index '${workspaceId}' removed.`);
      await fetchWorkspaces();
      if (currentWorkspaceId === workspaceId) {
        onSelectWorkspace(workspaces.find((w) => w.id === "default") || workspaces[0]);
      }
    } catch (err: any) {
      setActionError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <>
      {/* Workspace Quick-Selector Button */}
      <div className={`relative inline-flex items-center ${className}`}>
        <button
          type="button"
          onClick={() => setIsModalOpen(true)}
          className="group flex items-center gap-2.5 rounded-xl border border-white/10 bg-slate-900/80 px-3.5 py-1.5 text-xs text-slate-200 shadow-sm transition hover:border-indigo-500/40 hover:bg-slate-800/80 focus:outline-none focus:ring-2 focus:ring-indigo-500/20"
          title="Click to manage or switch repository workspaces"
        >
          <div className="flex h-5 w-5 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400 group-hover:scale-105 transition">
            <Database className="h-3 w-3" />
          </div>
          <div className="flex flex-col text-left">
            <span className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
              Repository Workspace
            </span>
            <span className="font-semibold text-white group-hover:text-indigo-300 transition max-w-[140px] truncate">
              {currentWorkspace.name}
            </span>
          </div>
          <span className="rounded bg-slate-800/90 px-1.5 py-0.5 text-[10px] font-mono text-cyan-400 border border-cyan-500/20">
            {currentWorkspace.unit_count} units
          </span>
          <ChevronDown className="h-3.5 w-3.5 text-slate-400 group-hover:text-slate-200 transition ml-1" />
        </button>
      </div>

      {/* Modal / Management Dialog */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4 animate-in fade-in duration-150">
          <div
            className="w-full max-w-2xl rounded-2xl border border-white/15 bg-[#0c111d] shadow-2xl overflow-hidden flex flex-col max-h-[90vh]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-white/10 px-6 py-4 bg-slate-900/50">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-700 shadow-glow">
                  <Layers className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-white">Repository Workspaces</h2>
                  <p className="text-xs text-slate-400">
                    Index, switch, and search isolated local codebases independently
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Tabs */}
            <div className="flex border-b border-white/10 px-6 bg-slate-900/20">
              <button
                type="button"
                onClick={() => setActiveTab("list")}
                className={`flex items-center gap-2 border-b-2 px-4 py-3 text-xs font-semibold transition ${
                  activeTab === "list"
                    ? "border-indigo-500 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                <Database className="h-3.5 w-3.5" />
                <span>Available Workspaces ({workspaces.length})</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("register")}
                className={`flex items-center gap-2 border-b-2 px-4 py-3 text-xs font-semibold transition ${
                  activeTab === "register"
                    ? "border-indigo-500 text-indigo-400"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                <Plus className="h-3.5 w-3.5" />
                <span>Register Local Repository</span>
              </button>
            </div>

            {/* Alerts */}
            {actionError && (
              <div className="mx-6 mt-4 flex items-start gap-2.5 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-xs text-rose-300">
                <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5 text-rose-400" />
                <div>{actionError}</div>
              </div>
            )}
            {actionSuccess && (
              <div className="mx-6 mt-4 flex items-start gap-2.5 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 text-xs text-emerald-300">
                <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5 text-emerald-400" />
                <div>{actionSuccess}</div>
              </div>
            )}

            {/* Modal Body */}
            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              {activeTab === "list" ? (
                <div className="space-y-3">
                  {workspaces.map((ws) => {
                    const isSelected = ws.id === currentWorkspaceId;
                    return (
                      <div
                        key={ws.id}
                        className={`rounded-xl border p-4 transition ${
                          isSelected
                            ? "border-indigo-500/60 bg-indigo-500/10 shadow-sm"
                            : "border-white/5 bg-slate-900/60 hover:border-white/15 hover:bg-slate-900/90"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-4">
                          <div className="space-y-1.5 flex-1">
                            <div className="flex items-center gap-2.5">
                              <h3 className="font-semibold text-white text-sm">{ws.name}</h3>
                              <span className="rounded bg-slate-800 px-2 py-0.5 text-[10px] font-mono text-slate-300 border border-white/10">
                                {ws.id}
                              </span>
                              {ws.is_canonical && (
                                <span className="rounded-full bg-cyan-500/15 px-2 py-0.5 text-[10px] font-medium text-cyan-400 border border-cyan-500/30">
                                  Benchmark Baseline
                                </span>
                              )}
                            </div>
                            <div className="flex items-center gap-3 text-xs text-slate-400">
                              <span className="flex items-center gap-1 font-mono text-[11px]">
                                <Folder className="h-3 w-3 text-slate-500" />
                                {ws.display_path}
                              </span>
                              <span>•</span>
                              <span>{ws.unit_count} units</span>
                              <span>•</span>
                              <span>{ws.file_count} files</span>
                            </div>
                            {ws.languages.length > 0 && (
                              <div className="flex items-center gap-1.5 pt-1">
                                {ws.languages.map((lang) => (
                                  <span
                                    key={lang}
                                    className="rounded bg-slate-800/80 px-1.5 py-0.5 text-[10px] text-slate-400 border border-white/5 uppercase"
                                  >
                                    {lang}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>

                          {/* Action Buttons */}
                          <div className="flex items-center gap-2">
                            {isSelected ? (
                              <span className="flex items-center gap-1 rounded-lg bg-emerald-500/20 px-2.5 py-1 text-xs font-semibold text-emerald-400 border border-emerald-500/30">
                                <CheckCircle2 className="h-3.5 w-3.5" />
                                Active
                              </span>
                            ) : (
                              <button
                                type="button"
                                onClick={() => {
                                  onSelectWorkspace(ws);
                                  setIsModalOpen(false);
                                }}
                                className="rounded-lg bg-indigo-600 px-3 py-1 text-xs font-medium text-white hover:bg-indigo-500 transition shadow-sm"
                              >
                                Select
                              </button>
                            )}

                            {!ws.is_canonical && (
                              <>
                                <button
                                  type="button"
                                  disabled={isProcessing}
                                  onClick={() => handleTriggerIndex(ws.id, false)}
                                  className="rounded-lg border border-white/10 bg-slate-800 p-1.5 text-slate-300 hover:bg-slate-700 hover:text-white transition disabled:opacity-50"
                                  title="Re-run full index"
                                >
                                  <RefreshCw className="h-3.5 w-3.5" />
                                </button>
                                <button
                                  type="button"
                                  disabled={isProcessing}
                                  onClick={() => handleDeleteWorkspace(ws.id)}
                                  className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-1.5 text-rose-300 hover:bg-rose-500/20 transition disabled:opacity-50"
                                  title="Remove workspace index (preserves sources)"
                                >
                                  <Trash2 className="h-3.5 w-3.5" />
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                /* Registration Form */
                <form onSubmit={handleRegister} className="space-y-4">
                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-300">
                      Workspace Identifier <span className="text-rose-400">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={newId}
                      onChange={(e) => setNewId(e.target.value)}
                      placeholder="e.g. backend-service, project-alpha"
                      className="w-full rounded-xl border border-white/10 bg-slate-900/90 px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono"
                    />
                    <p className="text-[11px] text-slate-500">
                      Alphanumeric slug (letters, numbers, hyphens, underscores). No paths or slashes.
                    </p>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-300">
                      Local Repository Root Directory <span className="text-rose-400">*</span>
                    </label>
                    <input
                      type="text"
                      required
                      value={newPath}
                      onChange={(e) => setNewPath(e.target.value)}
                      placeholder="e.g. ../DocuCharm or C:/Projects/my-app"
                      className="w-full rounded-xl border border-white/10 bg-slate-900/90 px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 font-mono"
                    />
                    <p className="text-[11px] text-slate-500">
                      Local filesystem directory path to target codebase. Sources are scanned read-only.
                    </p>
                  </div>

                  <div className="space-y-1">
                    <label className="text-xs font-semibold text-slate-300">
                      Display Title <span className="text-slate-500">(Optional)</span>
                    </label>
                    <input
                      type="text"
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      placeholder="e.g. Core Payment Engine"
                      className="w-full rounded-xl border border-white/10 bg-slate-900/90 px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    />
                  </div>

                  <div className="pt-2 flex justify-end gap-3">
                    <button
                      type="button"
                      onClick={() => setActiveTab("list")}
                      className="rounded-xl border border-white/10 bg-slate-800/80 px-4 py-2 text-xs font-medium text-slate-300 hover:bg-slate-700 transition"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isProcessing || !newId.trim() || !newPath.trim()}
                      className="rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 px-5 py-2 text-xs font-semibold text-white shadow-glow hover:opacity-95 transition disabled:opacity-50"
                    >
                      {isProcessing ? "Registering..." : "Register Workspace"}
                    </button>
                  </div>
                </form>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
