# Task: 0009 — Factor Models & Trading Signals (`models/{factors,signals}.py` & Notebooks 08–09)

## Goal
Implement statistical & fundamental factor models (`capm_fama_french_regression`, in-fold SVD/PCA eigenportfolios) in `src/quantlab/models/factors.py` and classical causal trading signals (time-series momentum, cross-sectional momentum, volatility-scaled mean reversion) in `src/quantlab/models/signals.py`, accompanied by Phase 1 Notebooks 08 and 09.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.2 (Notebooks 08–09), §1.4, §6.2
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §3 (`signal[t] -> fill[t+1]`), §5 (`(T, N)` arrays)

## Inputs
- `quantlab.compute.check_array_memory`
- `quantlab.data.SyntheticDataProvider` (and optional Ken French daily factor parser dividing percentage values by `100.0`)
- `quantlab.validation.leakage`: `check_future_poison`, `check_in_fold_preprocessor`, `evaluate_signal_lag_and_shift_probe`, `run_leakage_suite`

## Outputs (Files & APIs)
- `src/quantlab/models/factors.py`:
  - `FactorRegressionResult` frozen dataclass (`alpha_annualized`, `betas`, `r_squared`, `residual_vol_annualized`, `t_stats`)
  - `fit_factor_regression(excess_asset_returns, factor_returns, *, periods_per_year=252) -> FactorRegressionResult`
  - `PCAEigenportfolioModel`: strictly causal `fit(train_returns_2d)` / `transform(returns_2d)` SVD eigenportfolio extractor isolating test-fold rows (`fit` never sees test data)
- `src/quantlab/models/signals.py` & `src/quantlab/models/__init__.py`:
  - `time_series_momentum_signal(returns_panel, *, lookback=20, vol_window=20) -> npt.NDArray[np.float64]`
  - `cross_sectional_momentum_signal(returns_panel, *, lookback=20, top_quantile=0.25, dollar_neutral=True) -> npt.NDArray[np.float64]`
  - `mean_reversion_zscore_signal(prices_or_returns, *, window=20, entry_z=1.5) -> npt.NDArray[np.float64]`
- `tests/test_models_factors_and_signals.py`
- `notebooks/phase1/08_factor_models.ipynb`
- `notebooks/phase1/09_strategies_and_signals.ipynb` (registered in `tests/test_notebooks.py`)

## Acceptance Tests (Concrete, Runnable)
- [ ] **Known-answer tests:**
  - `fit_factor_regression` recovers planted factor $\alpha$ and $\beta$ coefficients on noise-free linear combinations to $10^{-10}$.
  - Orthonormality of PCA eigenvectors ($V^T V = I$) and variance-explained sum matching total trace of sample covariance.
- [ ] **Leakage & sanity tests (mandatory):**
  - `PCAEigenportfolioModel` passes `check_in_fold_preprocessor` (training projections invariant to test-fold scaling/NaNs).
  - Every function in `signals.py` passes `check_future_poison` (`max_abs_diff == 0.0`) and `evaluate_signal_lag_and_shift_probe` (`lag=+1` execution contract).
  - Planted Ken French parser fixture divides raw percentages by `100.0` (`1.25 -> 0.0125`).
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Ridge/LightGBM/PatchTST forecasting models, regime-switching HMMs, and Kalman pairs filters (Phase 2 Layer 4).

## Risks / Assumptions to Verify
- Lookback windows producing warmup `NaN`s or leading zeros must never forward-propagate into strategy weights (`np.where(np.isnan(...), 0.0, ...)`).

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations)
- **Coverage target:** ≥ 90% on `src/quantlab/models/`
- **Type checking:** `mypy --strict`
