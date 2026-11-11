# Research & Review Log (`docs/research_log.md`)

Chronological log of architectural reviews, experimental findings, negative results, and mental-model updates across Phases 1–3.

---

## 2026-11-10 — Pre-Initialization Design Audit & 2026 Data Source Verification

- **Summary:** Reviewed `QUANT_PROJECT_DESIGN.md` (v0.1) prior to writing any code; identified five architectural improvements and audited all market data vendors for 2026 availability and terms.
- **Key Architectural Changes Adopted (v0.2 & [ADR-0001](adr/0001-initial-architecture-decisions.md)):**
  1. **US Stock Market Scope (`XNYS`/`XNAS`):** Scoped Phase 1–2 strictly to the US stock market using a two-tier universe (14 US Market/Sector ETFs + 50–100 liquid US single stocks) to eliminate multi-currency and multi-timezone noise.
  2. **Raw OHLCV + Corporate Actions Separation:** Replaced reliance on backward-adjusted `Adj Close` with immutable raw prices + explicit split/dividend adjustment tables so content hashes remain immutable.
  3. **Two-File Experiment Ledger:** Paired `experiments/ledger.jsonl` with `experiments/trial_returns.parquet` so CSCV/PBO and effective independent trials ($\hat{N}_{\text{eff}}$) in the Deflated Sharpe Ratio have access to the full $T \times K$ return matrix.
  4. **Data Vendor Modernization:** Adopted Tiingo EOD + Alpaca (`feed="sip"`, `end <= now - 16min`) + FRED/ALFRED (`realtime_start`) + Ken French + Databento/WRDS (`@ualberta.ca`); dropped Alpha Vantage (25 req/day limit; adjusted close paywalled) and Binance `data.binance.vision` (geo-blocked in North America).
  5. **Hardware Guardrail:** Documented the 8 GB unified memory limit on the primary Apple M2 laptop and standardized on Parquet + DuckDB/Polars from Day 1 with a typed `float64` NumPy array boundary.
