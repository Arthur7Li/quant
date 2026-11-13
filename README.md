# quant-lab

> A three-phase quantitative finance engineering & research repository:
> 1. **Phase 1 — Foundations:** Learn quantitative finance through the lens of Computer Science, build an event-driven backtester and leakage suite from scratch, and prove why naive backtests fail.
> 2. **Phase 2 — Production Flagship:** Build a cost-aware, regime-aware **North American (US) Stock Market Portfolio Construction & Risk System**, validated under Combinatorial Purged Cross-Validation (CPCV), Deflated Sharpe Ratio (DSR), and Probability of Backtest Overfitting (PBO), forward-tested on Alpaca Paper Trading.
> 3. **Phase 3 — Frontier:** An open-ended research platform with pre-registered hypotheses, automated statistical gatekeepers, and high-performance systems (Rust `PyO3` LOB matching engine, deep hedging, market microstructure).

---

## Getting Started

### 1. Prerequisites
- **[uv](https://docs.astral.sh/uv/)** package and Python version manager (`brew install uv` on macOS).
- **Python 3.12+** (automatically provisioned via `uv sync`).
- Tested locally on **macOS Apple Silicon (M2, 8 GB Unified Memory)** and **Linux / Windows WSL2 (CUDA)**.

### 2. Installation
Clone the repository and synchronize the deterministic environment lockfile (`uv.lock`):

```bash
git clone https://github.com/Arthur7Li/quant.git
cd quant

# Core numerical stack + dev/test/lint toolchain
uv sync --group dev
```

Optional feature extras (install as needed):

```bash
# Live/historical market providers (exchange_calendars, httpx, yfinance, alpaca-py)
uv sync --group dev --extra market

# Optimization & econometrics (cvxpy, scikit-learn, statsmodels, arch, lightgbm)
uv sync --group dev --extra research

# Interactive JupyterLab environment for Phase 1 notebooks
uv sync --group dev --extra notebooks
```

### 3. Environment Variables (Optional for Offline Work)
All unit tests, leakage validation suites, and Notebook 01 run **100% offline** using deterministic synthetic data (`quantlab.data.SyntheticDataProvider`) and hand-computed fixtures. When pulling live/historical market bars locally into gitignored `data/raw/`, export API credentials via `.env` (never committed):

```bash
TIINGO_API_KEY=...
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
FRED_API_KEY=...
```

### 4. Running Quality Gates & Tests
Before committing any change, run the four mandatory checks enforced in CI:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=quantlab --cov-report=term-missing
```

### 5. Running Phase 1 Notebooks
Launch JupyterLab or execute notebooks using the project `.venv`:

```bash
uv run --extra notebooks jupyter lab notebooks/phase1/01_markets_prices_returns.ipynb
```

Notebook 01 (`01_markets_prices_returns.ipynb`) is also automatically verified top-to-bottom on every test run via [`tests/test_notebooks.py`](tests/test_notebooks.py).

---

## Repository Map & Documentation

- **[QUANT_PROJECT_DESIGN.md](QUANT_PROJECT_DESIGN.md):** Master architectural specification and roadmap (v0.2).
- **[ADR-0001: Initial Architecture, Data, and Scope Decisions](docs/adr/0001-initial-architecture-decisions.md):** Locked decisions on the Phase 2 flagship, US stock market two-tier universe, 2026 data vendor stack, Alpaca paper trading, Rust deferral to Phase 3, and two-file experiment ledger.
- **[AGENTS.md](AGENTS.md):** Standing rules, Definition of Done, 8 GB M2 memory guardrails, and forbidden anti-patterns enforced on AI agents.
- **[docs/conventions.md](docs/conventions.md):** Mathematical conventions (raw OHLCV + corporate actions, dual UTC timestamps `period_ts`/`available_ts`, `signal[t] -> fill[t+1]` execution lag, `XNYS` 252-day annualization).
- **[docs/glossary.md](docs/glossary.md):** CS → Quantitative Finance concept mapping.
- **[docs/risk_register.md](docs/risk_register.md):** Active statistical, data, numerical, and hardware risks.
- **[docs/tasks/](docs/tasks/):** Concrete task specifications driving all Phase 1 implementation work.

---

## Implemented Core Modules (Phase 1 Kickoff)

| Module | Purpose |
|---|---|
| [`quantlab.compute`](src/quantlab/compute.py) | `cuda -> mps -> cpu` device selection, deterministic seeding, and 8 GB M2 memory budget checks |
| [`quantlab.data`](src/quantlab/data/) | Dual-timestamp schemas (`BARS_SCHEMA`, `ACTIONS_SCHEMA`), corporate action adjustments, bar validation, Tiingo/Stooq/`yfinance`/synthetic providers, and DuckDB/Parquet PIT store |
| [`quantlab.core`](src/quantlab/core/) | Simple/log return conversions, multi-period compounding, portfolio aggregation, CAGR/vol/Sharpe/Sortino/Calmar/MDD, drifted-weight turnover, and Mertens/Lo (2002) Sharpe standard errors |
| [`quantlab.validation`](src/quantlab/validation/) | Causal shift probes (`shift_signals`), `check_future_poison`, `evaluate_signal_lag_and_shift_probe`, `check_in_fold_preprocessor`, `check_shuffled_labels`, and `random_signal_sharpe_baseline` |

---

## Disclaimer

Educational and research project only. Nothing in this repository constitutes financial or investment advice. No live trading with real capital is part of this system.
