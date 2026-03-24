import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        mono: ["var(--font-jetbrains-mono)", "monospace"],
      },
      colors: {
        bg: {
          primary: "#0f1117",
          secondary: "#1a1d27",
          input: "#242736",
        },
        border: {
          subtle: "#2e3347",
          default: "#3d4462",
        },
        text: {
          primary: "#e8eaf6",
          secondary: "#9aa0c4",
          muted: "#5a6285",
        },
        accent: {
          DEFAULT: "#6366f1",
          hover: "#818cf8",
          muted: "rgba(99,102,241,0.15)",
        },
        grounded: { DEFAULT: "#22c55e", bg: "#14532d" },
        directional: { DEFAULT: "#f59e0b", bg: "#78350f" },
        assumed: { DEFAULT: "#f87171", bg: "#450a0a" },
        severity: { low: "#22c55e", mid: "#f59e0b", high: "#ef4444" },
        compliance: "#6366f1",
        relocation: "#f97316",
        investment: "#06b6d4",
        enforcement: "#a855f7",
      },
    },
  },
  plugins: [],
};
export default config;
