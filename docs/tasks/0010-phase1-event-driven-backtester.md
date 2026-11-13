# Task: 0010 — Event-Driven Backtester & Honest Simulation (`backtest/engine.py` & Notebook 10)

## Goal
Build an event-driven backtesting engine (`src/quantlab/backtest/engine.py`) from scratch enforcing the strict `signal[t] -> fill[t+1]` execution contract, explicit transaction costs, and dual accounting modes—**Mode A (Share & Cash with raw prices + splits/dividends)** and **Mode B (Total-Return Weight with drifted turnover)**—and verify both against `make_toy_5bar_fixture` and Notebook 10 (`10_backtesting_honestly.ipynb`).

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.1, §1.2 (Notebook 10), §1.4 (`backtest/engine.py`), §1.5, §1.6, §6.2
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §2–§4

## Inputs
- `quantlab.data.make_toy_5bar_fixture`, `compute_total_returns`
- `quantlab.core.metrics`: `compute_turnover`, `summarize_performance`
- `quantlab.validation.leakage`: `run_leakage_suite`

## Outputs (Files & APIs)
- `src/quantlab/backtest/engine.py` & `src/quantlab/backtest/__init__.py`:
  - `CostModel` protocol / `USBasisPointCostModel` dataclass (`half_spread_bps`, `commission_bps`, `impact_coeff`)
  - `Strategy` protocol (`on_bar(t, history_bars, portfolio_state) -> npt.NDArray[np.float64]`) receiving *only* `available_ts <= close_ts[t]` slices
  - `BacktestResult` frozen dataclass (`period_ts`, `equity_curve`, `net_returns`, `gross_returns`, `weights_history`, `turnover`, `total_costs_paid`, `summary`)
  - `run_backtest_share_cash(bars, actions, strategy, *, initial_cash=100_000.0, cost_model=..., fill_at="next_open") -> BacktestResult` (Mode A)
  - `run_backtest_total_return(total_returns_2d, target_weights_2d, *, period_ts=None, cost_model=..., initial_equity=1.0) -> BacktestResult` (Mode B)
- `tests/test_backtest_engine.py`
- `notebooks/phase1/10_backtesting_honestly.ipynb` (registered in `tests/test_notebooks.py`)

## Acceptance Tests (Concrete, Runnable)
- [ ] **Known-answer tests (`QUANT_PROJECT_DESIGN.md` §1.5):**
  - Mode A (`run_backtest_share_cash`) and Mode B (`run_backtest_total_return`) reproduce the hand-computed 5-bar `make_toy_5bar_fixture` (2-for-1 split on Bar 2, `$1.00` dividend on Bar 3, explicit transaction costs) to machine precision ($10^{-12}$), satisfying $\text{Equity}_t = \text{Cash}_t + \sum_i \text{Shares}_{i,t} \cdot P_{i,t}^{\text{raw}}$.
  - Zero-cost buy-and-hold total-return weights match vectorized `portfolio_returns` identically.
- [ ] **Leakage & bias tests:**
  - Engine passes `check_future_poison` cleanly (altering future bars after `t` never alters `on_bar(t)` inputs or resulting trade fills prior to `t+2`).
  - Notebook 10 deliberately demonstrates two failure modes: (a) injecting look-ahead vs. causal lag, and (b) single-stock survivorship bias comparing surviving-winner selection vs. an unbiased cohort including delisted/failing paths.
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- L2 limit order book matching (Task 0011) and cross-verification memo against external Python backtesting library (`§1.6` Phase 1 → 2 gate, tracked in Task 0012).

## Risks / Assumptions to Verify
- Split/dividend ex-date execution timing: if an order fills at `open[t+1]` on `ex_date = t+1`, shares held overnight from `t` receive the split/dividend while newly added shares execute post-ex. Document and test exact ex-date share adjustment order cleanly.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations)
- **Coverage target:** ≥ 95% on `src/quantlab/backtest/engine.py`
- **Type checking:** `mypy --strict`
