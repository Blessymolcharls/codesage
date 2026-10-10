import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CodeSage — Codebase Intelligence & Retrieval Benchmark",
  description:
    "Interactive multi-regime code search system evaluating TF-IDF, general sentence embeddings, code Transformers, and graph reranking on the DocuCharm codebase.",
  keywords: [
    "code search",
    "information retrieval",
    "TF-IDF",
    "embeddings",
    "transformers",
    "graph reranking",
    "codebase intelligence",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`dark ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-[#090d16] text-slate-100 font-sans antialiased flex flex-col selection:bg-indigo-500/30 selection:text-indigo-200">
        <Navbar />
        <main className="flex-1 pb-16">{children}</main>
        <footer className="border-t border-white/5 bg-[#060911] py-6 text-center text-xs text-slate-500">
          <div className="mx-auto max-w-7xl px-4 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-400">CodeSage</span>
              <span>•</span>
              <span>DocuCharm Repository Index (89 Code Units)</span>
            </div>
            <div className="font-mono text-[11px] text-slate-500">
              TF-IDF • all-MiniLM-L6-v2 • st-codesearch • Graph Context
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
