# Task: 0006 — Stylized Facts & Monte Carlo Simulation (`pricing/monte_carlo.py` & Notebooks 02–03)

## Goal
Implement `src/quantlab/pricing/monte_carlo.py` (`QUANT_PROJECT_DESIGN.md` §1.4) along with Phase 1 Notebooks 02 (`notebooks/phase1/02_stylized_facts.ipynb`) and 03 (`notebooks/phase1/03_randomness_and_monte_carlo.ipynb`), connecting empirical US equity return properties (fat tails, volatility clustering, leverage effect) to Brownian motion, Geometric Brownian Motion (GBM), variance-reduction techniques, and $O(1/\sqrt{N})$ Monte Carlo convergence.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.2 (Notebooks 02–03), §1.4 (`pricing/monte_carlo.py`), §1.5
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §4–§5

## Inputs
- `quantlab.compute`: `check_array_memory`, `get_device`, `set_global_seed`
- `quantlab.data.SyntheticDataProvider` and `quantlab.core.returns` (`TRADING_DAYS_PER_YEAR`)
- `scipy.stats` (skewness, excess kurtosis, Jarque-Bera test)

## Outputs (Files & APIs)
- `src/quantlab/pricing/monte_carlo.py` & `src/quantlab/pricing/__init__.py`:
  - `MCPriceEstimate` frozen dataclass (`price`, `std_error`, `ci_95`, `n_paths`)
  - `simulate_gbm_paths(s0, r, sigma, maturity, n_steps, n_paths, *, antithetic=False, seed=42) -> npt.NDArray[np.float64]` (exact log-normal Euler/Milstein discretization $\Delta \ln S_t = (r - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z$)
  - `price_european_mc(s0, k, r, sigma, maturity, n_paths, *, option_type="call", antithetic=True, control_variate=False, seed=42) -> MCPriceEstimate`
  - `empirical_stylized_facts(returns_1d, max_lag=20) -> dict[str, Any]` (excess kurtosis, skewness, Jarque-Bera p-value, ACF of $r_t$, ACF of $|r_t|$ and $r_t^2$, return-volatility leverage correlation $\text{Corr}(r_t, r_{t+1}^2)$)
- `tests/test_pricing_mc.py`
- `notebooks/phase1/02_stylized_facts.ipynb`
- `notebooks/phase1/03_randomness_and_monte_carlo.ipynb` (registered in `tests/test_notebooks.py`)

## Acceptance Tests (Concrete, Runnable)
- [ ] **Known-answer / convergence tests:**
  - `price_european_mc` matches analytical Black-Scholes within $3\sigma$ standard error (`|mc.price - bs_price| <= 3.0 * mc.std_error`) for both calls and puts (`QUANT_PROJECT_DESIGN.md` §1.5, §6.2).
  - Empirical slope of $\ln(\text{std\_error})$ vs. $\ln(N_{\text{paths}})$ across `N in {1_000, 4_000, 16_000, 64_000}` is $\approx -0.50 \pm 0.08$.
- [ ] **Property / variance-reduction tests:**
  - Antithetic variates + control variates strictly reduce `std_error` compared to naive Monte Carlo at fixed `n_paths`.
  - Under simulated GBM/GARCH synthetic series, ACF of $|r_t|$ and $r_t^2$ exhibits persistent positive autocorrelation while linear ACF of $r_t$ stays near zero.
  - `check_array_memory` blocks oversized path tensors within the 8 GB M2 memory guardrail.
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Closed-form Black-Scholes and CRR binomial trees (covered in Task 0007).
- Stochastic volatility (Heston/rough Bergomi) models (deferred to Phase 3).

## Risks / Assumptions to Verify
- Memory exhaustion on `(n_paths, n_steps)` dense matrices on 8 GB M2: mitigate via chunked path batching (`batch_size`) for `n_paths >= 100_000` plus terminal-only fast path when intermediate steps are unneeded for European payoffs.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations)
- **Coverage target:** ≥ 90% on `src/quantlab/pricing/monte_carlo.py`
- **Type checking:** `mypy --strict`
