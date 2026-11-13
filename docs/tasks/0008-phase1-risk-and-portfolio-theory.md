# Task: 0008 — Risk Measures & Portfolio Theory (`risk/measures.py`, `portfolio/{baselines,markowitz}.py` & Notebooks 06–07)

## Goal
Implement downside tail risk estimators (`value_at_risk`, `conditional_var` / Expected Shortfall in historical, parametric, and Monte Carlo modes) alongside foundational portfolio allocators (`1/N` equal weight, inverse-volatility, global minimum variance, and Markowitz mean-variance frontier) with Phase 1 Notebooks 06 and 07.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.2 (Notebooks 06–07), §1.4 (`portfolio/markowitz.py`, `portfolio/baselines.py`), §6.2
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §2, §4, §5 (`(T, N)` returns, `(N, N)` PSD covariance within $10^{-10}$)

## Inputs
- `quantlab.compute.check_array_memory`
- `quantlab.core`: `portfolio_returns`, `sharpe_ratio`, `max_drawdown`
- `scipy.optimize.minimize` (SLSQP) for constrained quadratic portfolio optimization

## Outputs (Files & APIs)
- `src/quantlab/risk/measures.py` & `src/quantlab/risk/__init__.py`:
  - `value_at_risk(returns_1d, alpha=0.95, method="historical", *, df=None) -> float` (returns positive loss magnitude at tail quantile $1 - \alpha$)
  - `conditional_var(returns_1d, alpha=0.95, method="historical", *, df=None) -> float` (Expected Shortfall $\mathbb{E}[-R \mid -R \ge \text{VaR}_\alpha]$)
  - `parametric_var_cvar(mean, vol, alpha=0.95) -> tuple[float, float]`
- `src/quantlab/portfolio/baselines.py`:
  - `equal_weight(n_assets) -> npt.NDArray[np.float64]`
  - `inverse_volatility_weights(cov_matrix, *, min_vol=1e-8) -> npt.NDArray[np.float64]`
- `src/quantlab/portfolio/markowitz.py` & `src/quantlab/portfolio/__init__.py`:
  - `sample_covariance(returns_2d, *, periods_per_year=252, enforce_psd=True) -> npt.NDArray[np.float64]`
  - `minimum_variance_weights(cov_matrix, *, long_only=True, max_weight=1.0) -> npt.NDArray[np.float64]`
  - `mean_variance_weights(expected_returns, cov_matrix, *, risk_aversion=1.0, long_only=True, max_weight=1.0) -> npt.NDArray[np.float64]`
  - `efficient_frontier(expected_returns, cov_matrix, *, n_points=25, long_only=True) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]`
- `tests/test_risk_and_portfolio.py`
- `notebooks/phase1/06_risk_measures.ipynb`
- `notebooks/phase1/07_portfolio_theory.ipynb` (registered in `tests/test_notebooks.py`)

## Acceptance Tests (Concrete, Runnable)
- [ ] **Known-answer tests:**
  - Two-asset closed-form minimum-variance weights $w_1 = (\sigma_2^2 - \rho\sigma_1\sigma_2) / (\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2)$ match `minimum_variance_weights` to $10^{-8}$.
  - Analytical Gaussian VaR/CVaR formulas match numerical empirical estimates on large synthetic draws ($N = 100\_000$).
- [ ] **Property tests (`hypothesis`):**
  - $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$ for all non-degenerate distributions (`0.5 < alpha < 1.0`).
  - VaR fails subadditivity on a discrete bimodal credit-default fixture while CVaR preserves subadditivity ($\text{CVaR}(A+B) \le \text{CVaR}(A) + \text{CVaR}(B)$).
  - Portfolio weights sum to `1.0` within $10^{-10}$ and obey `[0.0, max_weight]` bounds when `long_only=True`.
  - Covariance matrices are symmetric PSD within $10^{-10}$.
- [ ] **In-sample vs. out-of-sample fragility test (Notebook 07):** Demonstrates sample covariance condition number explosion as $N / T$ increases and OOS dominance of `equal_weight` / `inverse_volatility_weights` over unregularized sample Markowitz.
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Ledoit-Wolf shrinkage, Marchenko-Pastur RMT denoising, HRP, Black-Litterman, and `cvxpy` transaction-cost MPC (Phase 2 Layers 1 and 3).

## Risks / Assumptions to Verify
- Ill-conditioned or nearly singular sample covariance matrices when $T < N$ causing SLSQP solver non-convergence; mitigate with PSD eigenvalue clipping / condition-number diagnostics.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations)
- **Coverage target:** ≥ 90% on `src/quantlab/risk/` and `src/quantlab/portfolio/`
- **Type checking:** `mypy --strict`
