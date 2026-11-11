# Experiment Ledger (`experiments/`)

As specified in [QUANT_PROJECT_DESIGN.md](../QUANT_PROJECT_DESIGN.md) §6.3 and [ADR-0001](../docs/adr/0001-initial-architecture-decisions.md), every strategy or model trial evaluated in `quant-lab`—including failed trials—is logged to two Git-tracked files in this directory:

1. **`experiments/ledger.jsonl`:** Append-only JSON Lines file where each line records a single trial's metadata and summary statistics:
   - `trial_id` (UUID / deterministic hash)
   - `created_at_utc` (ISO-8601 UTC timestamp)
   - `hypothesis_id` (reference to task spec or hypothesis registry)
   - `git_sha` (commit hash that produced the trial)
   - `config_hash` (SHA-256 of typed configuration)
   - `data_hash` (SHA-256 content hash of input PIT dataset)
   - `start_date`, `end_date`, `n_observations`, `n_assets`
   - `sharpe_gross`, `sharpe_net`, `annualized_return_net`, `annualized_vol`, `max_drawdown`, `mean_turnover`
   - `return_skewness`, `return_kurtosis` (required for Bailey & López de Prado non-normal Sharpe standard error)

2. **`experiments/trial_returns.parquet`:** Columnar store with schema `(trial_id: string, period_ts: timestamp[ns, tz=UTC], net_return: double)`.
   - Preserves the full $T \times K$ matrix of out-of-sample trial return streams required to compute:
     - **Effective number of independent trials ($\hat{N}_{\text{eff}}$)** from cross-trial return correlations for the **Deflated Sharpe Ratio (DSR)**, and
     - **Probability of Backtest Overfitting (PBO)** via Combinatorial Symmetric Cross-Validation (CSCV).
