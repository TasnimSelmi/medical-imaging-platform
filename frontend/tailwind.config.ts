import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: "#ffffff",
        bg: "#fafafa",
        border: "#e5e5e5",
        text: "#1a1a1a",
        "text-secondary": "#6b6b6b",
        accent: "#1e40af",
        "mask-pred": "rgba(220, 38, 38, 0.5)",
        "mask-gt": "rgba(34, 197, 94, 0.5)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
