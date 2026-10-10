import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#090d16",
        foreground: "#f1f5f9",
        card: {
          DEFAULT: "#0f172a",
          foreground: "#f8fafc",
          muted: "#1e293b",
        },
        brand: {
          50: "#eef2ff",
          100: "#e0e7ff",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
        },
        accent: {
          emerald: "#10b981",
          violet: "#8b5cf6",
          amber: "#f59e0b",
          cyan: "#06b6d4",
          rose: "#f43f5e",
        },
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "JetBrains Mono", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px -4px rgba(99, 102, 241, 0.15)",
        "glow-emerald": "0 0 24px -4px rgba(16, 185, 129, 0.2)",
        "glow-violet": "0 0 24px -4px rgba(139, 92, 246, 0.2)",
      },
    },
  },
  plugins: [],
};

export default config;
