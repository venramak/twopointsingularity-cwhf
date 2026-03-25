# CWHF Coherence Lab – Validation Prototype

This is an interactive web application that demonstrates the **Coherence‑Weighted Human Feedback (CWHF)** framework. It allows users to:

- Explore the theoretical framework (contamination layers, bias theorem, rate‑distortion, scalability).
- View a mock simulation of 10 personas × 10 cognitive states (heatmap).
- Participate in a **Live H1** session to compute their own coherence weight `c` and observer effect `Δc`.
- Export data (CSV, JSON, JSON‑LD) and simulate secure sharing of aggregated metrics with Merkle integrity.

**Privacy‑first design:** All raw data stays in your browser’s IndexedDB. Only aggregated weights are optionally shared (mock).

## Deploy on Vercel / Netlify

1. Push this repository to GitHub.
2. Import the repository on [Vercel](https://vercel.com) or [Netlify](https://netlify.com).
3. Both will automatically detect the static HTML file and serve it.

## Run Locally

Simply open `index.html` in any modern browser.

## License

CC BY 4.0 – Venkat Rao, Two Point Singularity, 2026.
