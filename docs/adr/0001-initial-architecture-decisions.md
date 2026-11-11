# ADR 0001: Initial Architecture, Data, and Scope Decisions

- **Status:** Accepted
- **Date:** 2026-11-10
- **Context:** Pre-initialization design review of `QUANT_PROJECT_DESIGN.md` (v0.1 → v0.2), verification of 2026 market data API terms, and resolution of the seven open project decisions (§9).

---

## Decisions

### 1. Phase 2 Flagship: Option A — Cost-Aware, Regime-Aware US Portfolio Construction & Risk System
- **Decision:** Build Option A as the single production-grade flagship for Phase 2.
- **Rationale:** Return prediction on daily public equity data has low signal-to-noise, whereas covariance shrinkage/denoising (Ledoit-Wolf, Marchenko–Pastur RMT), hierarchical clustering (HRP), Black-Litterman, regime-conditioned risk scaling, and convex optimization with explicit transaction costs (`cvxpy`) provide robust, verifiable out-of-sample improvements that can be forward-tested in paper trading.

### 2. Asset Universe: North American (US) Stock Market (`XNYS` / `XNAS`)
- **Decision:** Restrict Phase 1–2 strictly to the US stock market using a two-tier universe:
  - **Tier 1 (Survivorship-Free Core):** 14 liquid US Market & Sector ETFs (`SPY`, `QQQ`, `IWM`, `XLK`, `XLF`, `XLV`, `XLE`, `XLI`, `XLY`, `XLP`, `XLU`, `XLB`, `XLRE`, `XLC`).
  - **Tier 2 (Cross-Sectional Single-Stock Panel):** 50–100 liquid US equities for high-dimensional covariance denoising, PCA/Fama-French factor modeling, corporate actions, and survivorship-bias testing.
- **Rationale:** Focusing exclusively on the US stock market eliminates multi-currency, multi-timezone, and 24/7 cross-asset calendar alignment noise. Every equity/ETF bar adheres to the `XNYS` trading calendar (`09:30–16:00 America/New_York`). University affiliation (`@ualberta.ca`) also enables optional access to WRDS/CRSP and Databento academic programs.

### 3. Data Frequency & Verified 2026 Vendor Stack
- **Decision:** Use **daily (`1D`) bars** for all Phase 1–2 core pipelines, scoping intraday/L2 tick data strictly to microstructure modules (Notebook 11 and Phase 3 Track B/G).
- **Vendor Stack:**
  - **Primary EOD Equities/ETFs:** **Tiingo REST API** (raw OHLCV + split factors + cash dividends; 30+ yrs history) and **Alpaca Market Data** (`feed="sip"` with `end <= now - 16min` for consolidated US volume).
  - **Secondary / Exploratory:** `yfinance` (with `curl_cffi` impersonation and rate pacing) and **Stooq** (direct CSV/bulk downloads with HTTP 200 quota-text detection).
  - **Macro & Factors:** **FRED / ALFRED** (using `realtime_start` vintage dates for point-in-time discipline) and **Kenneth French Data Library** (converted from percentage points to decimal returns).
  - **Microstructure / Options:** **LOBSTER** sample files and **Databento** (\$125 free credit + university academic program).
  - **Dropped Vendors:** Alpha Vantage (free tier limited to 25 req/day; adjusted close paywalled) and Binance `data.binance.vision` (geo-blocked for US/North American IPs and out of US-equity scope).

### 4. Paper-Trading Broker: Alpaca Paper Trading API
- **Decision:** Use **Alpaca (`alpaca-py`)** for Phase 2 automated paper trading and order reconciliation.
- **Rationale:** Provides a stateless REST/WebSocket API suitable for unattended Docker/cron execution without the local Java gateway (`TWS`/`IB Gateway`) and manual weekly 2FA required by Interactive Brokers.

### 5. Language Strategy: 100% Python in Phase 1–2; Rust (`PyO3`/`maturin`) in Phase 3 Track G
- **Decision:** Keep Phase 1 and Phase 2 exclusively in Python 3.12 (managed via `uv`), deferring Rust to Phase 3 Track G (high-performance LOB matching engine and event simulator).
- **Rationale:** Numerical bottlenecks in Phase 1–2 are already handled by compiled C/C++/Fortran libraries (NumPy, Polars, DuckDB, Clarabel/OSQP, PyTorch).

### 6. Repository Visibility & Data Hygiene: Public from Day 1
- **Decision:** Maintain a public GitHub repository (`github.com/Arthur7Li/quant`) from Day 1 with `.gitignore` and pre-commit hooks blocking secrets and vendor data files.
- **Rationale:** Enforces production-grade engineering discipline and provides an externally verifiable portfolio artifact. All unit tests and GitHub Actions CI runs execute on deterministic synthetic data (`quantlab.data.synthetic`) to comply with market data vendor non-redistribution terms.

### 7. Experiment Tracking: Git-Tracked `ledger.jsonl` + `trial_returns.parquet`
- **Decision:** Track all experiments in Git via `experiments/ledger.jsonl` (metadata, hashes, summary statistics) paired with `experiments/trial_returns.parquet` (out-of-sample per-period return vectors).
- **Rationale:** Avoids external SaaS dependencies, syncs seamlessly via Git across the macOS M2 (8 GB RAM) and Windows RTX 3080 Ti machines, and preserves the full $T \times K$ trial return matrix required to compute the Probability of Backtest Overfitting (PBO) and effective independent trial count ($\hat{N}_{\text{eff}}$) for the Deflated Sharpe Ratio (DSR).

---

## Consequences
- All time-series pipelines must store raw (unadjusted) prices alongside explicit split/dividend records, plus dual timestamps (`period_ts` and `available_ts`).
- Because the primary development machine is an Apple M2 with **8 GB Unified Memory**, data storage uses Parquet + DuckDB/Polars from Day 1, and core numerical routines operate on typed `float64` NumPy arrays.
