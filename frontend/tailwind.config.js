export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        verdict: {
          true: "#00C896",
          false: "#F6584C",
          unsure: "#FFB347"
        },
        card: "#0F172A",
        background: "#050E1F"
      },
      fontFamily: {
        sans: ["Space Grotesk", "Inter", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "SFMono-Regular", "monospace"]
      },
      boxShadow: {
        glow: "0 10px 35px rgba(15, 23, 42, 0.35)"
      }
    }
  },
  plugins: []
};
