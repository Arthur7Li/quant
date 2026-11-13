# Task: 0007 — Option Pricing & Greeks (`pricing/black_scholes.py`, `binomial.py`, `greeks.py` & Notebooks 04–05)

## Goal
Implement analytical Black-Scholes pricing (`black_scholes.py`), closed-form & finite-difference option sensitivities (`greeks.py`), and Cox-Ross-Rubinstein (CRR) binomial tree dynamic programming (`binomial.py`) for European and American US equity options, along with Phase 1 Notebooks 04 and 05.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.2 (Notebooks 04–05), §1.4 (`pricing/`), §1.5, §6.2
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §2 (USD & decimal conventions), §5 (`np.float64`)

## Inputs
- `scipy.stats.norm` (standard normal CDF/PDF) and `numpy` (`float64`)
- `quantlab.compute.check_array_memory` for $O(N_{\text{steps}})$ memory-efficient binomial backward induction

## Outputs (Files & APIs)
- `src/quantlab/pricing/black_scholes.py`:
  - `bs_price(s0, k, r, sigma, maturity, *, option_type="call", q=0.0) -> float`
  - `implied_volatility(market_price, s0, k, r, maturity, *, option_type="call", q=0.0) -> float` (Brent's root-finding within no-arbitrage bounds)
- `src/quantlab/pricing/greeks.py`:
  - `OptionGreeks` frozen dataclass (`delta`, `gamma`, `vega`, `theta`, `rho`)
  - `bs_greeks(s0, k, r, sigma, maturity, *, option_type="call", q=0.0) -> OptionGreeks`
  - `finite_difference_greeks(pricer_fn, s0, k, r, sigma, maturity, *, option_type="call", q=0.0) -> OptionGreeks`
- `src/quantlab/pricing/binomial.py`:
  - `crr_binomial_price(s0, k, r, sigma, maturity, n_steps, *, option_type="call", american=False, q=0.0) -> float`
  - `crr_early_exercise_boundary(...) -> npt.NDArray[np.float64]`
- `src/quantlab/pricing/__init__.py` & `tests/test_pricing_options.py`
- `notebooks/phase1/04_option_pricing_1_black_scholes.ipynb`
- `notebooks/phase1/05_option_pricing_2_binomial_trees.ipynb` (registered in `tests/test_notebooks.py`)

## Acceptance Tests (Concrete, Runnable)
- [ ] **Known-answer tests:**
  - Standard textbook Hull BS call/put fixture (`S=42, K=40, r=0.10, sigma=0.20, T=0.5`) matches analytical benchmark to $10^{-10}$.
  - Closed-form `bs_greeks` matches central `finite_difference_greeks` within $10^{-5}$.
- [ ] **Property & convergence tests (`hypothesis`):**
  - Put-call parity invariant $C - P = S_0 e^{-q T} - K e^{-r T}$ holds to $10^{-12}$ across random parameter combinations (`QUANT_PROJECT_DESIGN.md` §1.5, §6.2).
  - No-arbitrage bounds: $\max(S_0 e^{-qT} - K e^{-rT}, 0) \le C \le S_0 e^{-qT}$; American price $\ge$ European price ($\forall$ inputs); American put strictly exceeds European put when $r > 0$ and $K \ge S_0$.
  - High-$N$ European CRR binomial tree converges to `bs_price` within $O(1/N)$ ($< 5 \times 10^{-3}$ at $N = 500$).
  - Implied volatility solver recovers exact input `sigma` to $10^{-8}$.
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Implied volatility surface interpolation (SVI/SABR) and Delta hedging simulator (deferred to Phase 2 / Phase 3 Track C).

## Risks / Assumptions to Verify
- Expiration $T \to 0$ and zero volatility $\sigma \to 0$ boundary behavior cleanly returns intrinsic payoff without `ZeroDivisionError` or `NaN`.
- No-arbitrage failure in CRR parameterization if $\Delta t$ is too large ($u = e^{\sigma \sqrt{\Delta t}}, d = 1/u$, enforce $d < e^{(r-q)\Delta t} < u$).

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations)
- **Coverage target:** ≥ 95% on `src/quantlab/pricing/`
- **Type checking:** `mypy --strict`
