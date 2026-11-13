# Task: 0003 — Leakage & Sanity Validation Suite (`quantlab.validation.leakage`)

## Goal
Implement `src/quantlab/validation/leakage.py` (`QUANT_PROJECT_DESIGN.md` §1.4, §6.2, and §8 Item 7) providing automated detectors for look-ahead bias (`future_poison`), signal-to-fill alignment errors (`+1` vs. `-1` `shift_probe`), in-fold stateful preprocessing leakage (`in_fold_preprocessor`), target label leakage via permutation tests (`shuffled_labels`), and deterministic random-signal Sharpe baselines (`random_signal_sharpe_baseline`).

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.4, §1.5, §6.2, §8 (Item 7)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §3 (`signal[t] -> fill[t+1]`, `shift(+1)` vs. `shift(-1)`) and §5

## Inputs
- `quantlab.compute.check_array_memory` for 8 GB M2 allocation guardrails
- `quantlab.core.returns.portfolio_returns` for 2D `(T, N)` strategy return aggregation
- `quantlab.core.metrics.sharpe_ratio` for annualized Sharpe evaluation
- Standardized 1D `(T,)` and 2D `(T, N)` `np.float64` feature, price, return, and weight panels

## Outputs (Files & APIs)
- `src/quantlab/validation/leakage.py`:
  - `LeakageViolationError`, `LeakageCheckResult`, `LeakageSuiteReport`
  - `SignalFunc`, `PreprocessorFunc`, `FitPredictFunc`
  - `shift_signals`, `check_future_poison`, `evaluate_signal_lag_and_shift_probe`, `check_in_fold_preprocessor`, `check_shuffled_labels`, `random_signal_sharpe_baseline`, `run_leakage_suite`
- `src/quantlab/validation/__init__.py`: Public exports for `quantlab.validation`
- `tests/test_validation_leakage.py`: Known-answer, Hypothesis property, causal-pass, and intentionally leaky-fail tests

## Acceptance Tests (Concrete, Runnable)
- [x] Causal trailing moving-average / momentum signal passes `check_future_poison` and `evaluate_signal_lag_and_shift_probe`.
- [x] Deliberately leaky future-peeking signal (`w[t] = sign(r[t+1])`) and full-sample normalized signal fail `check_future_poison` and `evaluate_signal_lag_and_shift_probe` with explicit diagnostic details.
- [x] Causal in-fold z-score scaler passes `check_in_fold_preprocessor`; a leaky preprocessor fitting on the full sample before splitting or refitting on the test fold fails.
- [x] Causal predictor passes `check_shuffled_labels` (near-zero mean OOS Sharpe under shuffled training labels and invariant to test-label perturbation); a predictor peeking at `y_test` fails.
- [x] `random_signal_sharpe_baseline` produces a deterministic null Sharpe distribution on 1D and 2D synthetic returns within the 8 GB M2 memory budget.
- [x] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Cross-sectionally synchronized Purged K-Fold, CPCV, Deflated Sharpe Ratio (DSR), and Probability of Backtest Overfitting (PBO) (deferred to Phase 2 Layer 1–2 validation modules).

## Risks / Assumptions to Verify
- Strict `np.float64` handling, NaN-mask comparison during future-poison checks, and zero-variance prediction / zero-volatility Sharpe edge cases returning `0.0` or `NaN` safely without raising `ZeroDivisionError`.
- Protocol and Callable type annotations compatible with `mypy --strict`.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations — Kickoff Task)
- **Coverage target:** ≥ 95% on `src/quantlab/validation/leakage.py`
- **Type checking:** `mypy --strict`
