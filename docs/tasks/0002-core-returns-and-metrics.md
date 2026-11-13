# Task: 0002 — Core Returns & Performance Metrics (`quantlab.core`)

## Goal
Implement `quantlab.core.returns` and `quantlab.core.metrics` (`QUANT_PROJECT_DESIGN.md` §1.4 and §8 Item 6) providing typed `float64` NumPy operations for return conversions, compounding, portfolio aggregation, risk-adjusted performance ratios, non-normal and autocorrelation-adjusted Sharpe standard errors, drawdowns, and drifted-weight turnover.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.4, §6.1, §8 (Item 6)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §2, §3, §4, §5

## Inputs
- `quantlab.compute.check_array_memory` for 8 GB M2 allocation guardrails
- Standardized 1D `(T,)` and 2D `(T, N)` `np.float64` return/price/weight panels in decimal fractions and `252` `XNYS` trading days per year.

## Outputs (Files & APIs)
- `src/quantlab/core/returns.py`: `TRADING_DAYS_PER_YEAR`, `simple_to_log_returns`, `log_to_simple_returns`, `prices_to_returns`, `compound_returns`, `wealth_index`, `aggregate_multi_period_returns`, `portfolio_returns`, `excess_returns`
- `src/quantlab/core/metrics.py`: `PerformanceSummary`, `annualized_return`, `annualized_volatility`, `sharpe_ratio`, `sharpe_standard_error`, `sharpe_confidence_interval`, `sortino_ratio`, `drawdown_series`, `max_drawdown`, `calmar_ratio`, `compute_turnover`, `summarize_performance`
- `src/quantlab/core/__init__.py`: Public package exports
- `tests/test_core.py`: Known-answer, edge-case, and Hypothesis property tests

## Acceptance Tests (Concrete, Runnable)
- [x] **Known-answer tests:** Exact verification of simple/log conversions, compounding, wealth index, multi-period aggregation, cross-sectional portfolio returns, drawdown (capturing a bar-0 loss via $W_0 = 1.0$), drifted-weight turnover, and Mertens (2002) / Lo (2002) Sharpe standard errors and confidence intervals.
- [x] **Property tests (`hypothesis`):**
  - `log_to_simple_returns(simple_to_log_returns(R)) == R` for all $R > -1$.
  - Time-sum of log returns equals $\ln(1 + R_{\text{compound}})$.
  - Long-only portfolio return lies in $[\min_i R_{i,t}, \max_i R_{i,t}]$.
  - Positive scaling invariance of Sharpe and Sortino ratios (when risk-free/target rate is 0).
  - Drawdown bounds $-1.0 < \text{DD}_t \le 0.0$ and $0.0 \le \text{MDD} < 1.0$ for $R_t > -1.0$.
- [x] Passes `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- `XNYS` exchange calendar alignment wrapper (`core/calendars.py` when `market` extra is installed) and Value-at-Risk / CVaR (`risk/`).

## Risks / Assumptions to Verify
- `scipy.stats` typing under `mypy --strict` and `scipy==1.18.1`.
- Zero-volatility and zero-drawdown edge cases return `NaN` rather than raising `ZeroDivisionError`.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations — Kickoff Task)
- **Coverage target:** ≥ 90% on `src/quantlab/core/`
- **Type checking:** `mypy --strict`
