# quant-lab

> A three-phase quantitative finance engineering & research repository:
> 1. **Phase 1 — Foundations:** Learn quantitative finance through the lens of Computer Science, build an event-driven backtester and leakage suite from scratch, and prove why naive backtests fail.
> 2. **Phase 2 — Production Flagship:** Build a cost-aware, regime-aware **North American (US) Stock Market Portfolio Construction & Risk System**, validated under Combinatorial Purged Cross-Validation (CPCV), Deflated Sharpe Ratio (DSR), and Probability of Backtest Overfitting (PBO), forward-tested on Alpaca Paper Trading.
> 3. **Phase 3 — Frontier:** An open-ended research platform with pre-registered hypotheses, automated statistical gatekeepers, and high-performance systems (Rust `PyO3` LOB matching engine, deep hedging, market microstructure).

---

## Documentation & Architecture

- **[QUANT_PROJECT_DESIGN.md](QUANT_PROJECT_DESIGN.md):** Master architectural specification and roadmap (v0.2).
- **[ADR-0001: Initial Architecture, Data, and Scope Decisions](docs/adr/0001-initial-architecture-decisions.md):** Locked decisions on the Phase 2 flagship, US stock market two-tier universe, 2026 data vendor stack, Alpaca paper trading, and two-file experiment ledger.
- **[AGENTS.md](AGENTS.md):** Standing rules, Definition of Done, and anti-patterns enforced on all AI agents.
- **[docs/conventions.md](docs/conventions.md):** Mathematical conventions (raw OHLCV + corporate actions, dual UTC timestamps `period_ts`/`available_ts`, `signal[t] -> fill[t+1]` execution lag, `XNYS` 252-day annualization).
- **[docs/glossary.md](docs/glossary.md):** CS → Quantitative Finance mapping.
- **[docs/risk_register.md](docs/risk_register.md):** Active statistical, data, numerical, and hardware risks.

---

## Quickstart (macOS M2 / Linux WSL2)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12:

```bash
# 1. Install dependencies (creates .venv with Python 3.12)
uv sync --group dev

# 2. Run linter & formatter check
uv run ruff check .
uv run ruff format --check .

# 3. Run strict static type checking
uv run mypy src tests

# 4. Run test suite with coverage
uv run pytest --cov=quantlab
```

---

## Disclaimer

Educational and research project only. Nothing in this repository constitutes financial or investment advice. No live trading with real capital is part of this system.
