"use client";

import { useState } from "react";
import { Check, ChevronDown, ChevronUp, Copy, FileCode } from "lucide-react";

interface CodeViewerProps {
  code: string;
  language: string;
  startLine?: number;
  maxPreviewLines?: number;
  highlightSymbol?: string | null;
}

// Tokenize and format line with syntax coloring for JavaScript and Dart
function renderHighlightedLine(lineText: string, language: string, highlightSymbol?: string | null) {
  if (!lineText) return <span>&nbsp;</span>;

  // Comment line
  if (lineText.trim().startsWith("//") || lineText.trim().startsWith("/*") || lineText.trim().startsWith("*")) {
    return <span className="text-slate-500 italic">{lineText}</span>;
  }

  const jsDartKeywords = new Set([
    "const", "let", "var", "function", "async", "await", "return", "class", "extends",
    "import", "export", "from", "require", "if", "else", "for", "while", "switch",
    "case", "break", "try", "catch", "throw", "new", "this", "super", "null", "true",
    "false", "Widget", "BuildContext", "override", "final", "void", "bool", "int",
    "double", "String", "List", "Map", "Set", "Future", "static", "abstract"
  ]);

  // Regex splitting on word boundaries, strings, and operators
  const tokens = lineText.split(/(\/\/[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b[a-zA-Z_$][a-zA-Z0-9_$]*\b|[{}()[\].,;:+\-*/%=<>!&|^~?])/);

  return (
    <>
      {tokens.map((token, idx) => {
        if (!token) return null;

        // String literals
        if (token.startsWith('"') || token.startsWith("'") || token.startsWith("`")) {
          return (
            <span key={idx} className="text-emerald-400">
              {token}
            </span>
          );
        }

        // Inline comments
        if (token.startsWith("//")) {
          return (
            <span key={idx} className="text-slate-500 italic">
              {token}
            </span>
          );
        }

        // Keywords
        if (jsDartKeywords.has(token)) {
          return (
            <span key={idx} className="text-indigo-400 font-medium">
              {token}
            </span>
          );
        }

        // Highlighted symbol query match
        if (highlightSymbol && token.toLowerCase() === highlightSymbol.toLowerCase()) {
          return (
            <span key={idx} className="bg-amber-400/20 text-amber-300 font-bold px-1 rounded border border-amber-400/30">
              {token}
            </span>
          );
        }

        // Numbers
        if (/^\d+(\.\d+)?$/.test(token)) {
          return (
            <span key={idx} className="text-amber-400">
              {token}
            </span>
          );
        }

        // Method / function invocation
        if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(token) && /^[A-Z]/.test(token)) {
          return (
            <span key={idx} className="text-cyan-300">
              {token}
            </span>
          );
        }

        return <span key={idx} className="text-slate-200">{token}</span>;
      })}
    </>
  );
}

export function CodeViewer({
  code,
  language,
  startLine = 1,
  maxPreviewLines = 8,
  highlightSymbol = null,
}: CodeViewerProps) {
  const [copied, setCopied] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const lines = code.trimEnd().split("\n");
  const isTruncated = lines.length > maxPreviewLines;
  const displayedLines = expanded || !isTruncated ? lines : lines.slice(0, maxPreviewLines);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
    }
  };

  return (
    <div className="group relative rounded-xl border border-white/5 bg-[#060911] shadow-inner font-mono text-xs overflow-hidden">
      {/* Top action header */}
      <div className="flex items-center justify-between border-b border-white/5 bg-slate-900/60 px-3 py-1.5 text-[11px] text-slate-400">
        <div className="flex items-center gap-2">
          <FileCode className="h-3.5 w-3.5 text-slate-400" />
          <span className="uppercase tracking-wider font-semibold text-slate-300">
            {language}
          </span>
          <span className="text-slate-600">•</span>
          <span>{lines.length} lines</span>
        </div>

        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded px-2 py-0.5 text-[11px] text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="h-3 w-3 text-emerald-400" />
              <span className="text-emerald-400">Copied</span>
            </>
          ) : (
            <>
              <Copy className="h-3 w-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code body with line numbers */}
      <div className="overflow-x-auto p-3">
        <pre className="grid">
          {displayedLines.map((line, idx) => {
            const lineNum = startLine + idx;
            return (
              <div key={idx} className="flex hover:bg-white/[0.02] rounded px-1 -mx-1">
                <span className="w-9 shrink-0 select-none text-right pr-3 text-slate-600 font-mono text-[11px]">
                  {lineNum}
                </span>
                <span className="flex-1 whitespace-pre font-mono">
                  {renderHighlightedLine(line, language, highlightSymbol)}
                </span>
              </div>
            );
          })}
        </pre>
      </div>

      {/* Expand / collapse footer */}
      {isTruncated && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex w-full items-center justify-center gap-1.5 border-t border-white/5 bg-slate-900/40 py-1.5 text-[11px] font-medium text-indigo-400 hover:bg-slate-900/80 hover:text-indigo-300 transition"
        >
          {expanded ? (
            <>
              <ChevronUp className="h-3.5 w-3.5" />
              <span>Collapse ({lines.length - maxPreviewLines} lines hidden)</span>
            </>
          ) : (
            <>
              <ChevronDown className="h-3.5 w-3.5" />
              <span>Show all {lines.length} lines (+{lines.length - maxPreviewLines} more)</span>
            </>
          )}
        </button>
      )}
    </div>
  );
}
