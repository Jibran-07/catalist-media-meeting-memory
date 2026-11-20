import type { Config } from "tailwindcss";
export default {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: { extend: { colors: { ink: "#233041", border: "#dae1e8" } } },
  plugins: []
} satisfies Config;
