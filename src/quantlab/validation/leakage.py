"""Mandatory leakage, alignment, preprocessing, and null-baseline validation suite.

Implements `QUANT_PROJECT_DESIGN.md` §1.4, §6.2, and §8 Item 7, enforcing
`docs/conventions.md` §3 (`signal[t] -> fill[t+1]` execution lag):
- `shift_signals`: Causal positive delay (`shift(+k)`) and look-ahead probe (`shift(-k)`).
- `check_future_poison`: Verifies $w_t = f(\\mathcal{F}_t)$ is invariant to future poisoning.
- `evaluate_signal_lag_and_shift_probe`: Evaluates `+1`, `+2`, `0`, and `-1` alignments.
- `check_in_fold_preprocessor`: Detects full-sample or test-fold stateful fit leakage.
- `check_shuffled_labels`: Detects direct test-label leaks and non-null shuffled skill.
- `random_signal_sharpe_baseline`: Deterministic random-signal Sharpe null distribution.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from quantlab.compute import check_array_memory
from quantlab.core.metrics import sharpe_ratio
from quantlab.core.returns import portfolio_returns

SignalFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
PreprocessorFunc = Callable[
    [npt.NDArray[np.float64], int],
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
]
FitPredictFunc = Callable[
    [
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ],
    npt.NDArray[np.float64],
]


class LeakageViolationError(RuntimeError):
    """Raised when one or more checks in a `LeakageSuiteReport` fail."""


@dataclass(frozen=True)
class LeakageCheckResult:
    """Outcome of a single leakage or sanity check."""

    name: str
    passed: bool
    max_abs_diff: float
    details: str
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LeakageSuiteReport:
    """Aggregated report across all executed leakage and sanity checks."""

    passed: bool
    checks: tuple[LeakageCheckResult, ...]

    def raise_if_failed(self) -> None:
        """Raise `LeakageViolationError` listing all failed checks if `not self.passed`."""
        if self.passed:
            return
        failures = [f"- {c.name}: {c.details}" for c in self.checks if not c.passed]
        msg = "Leakage validation suite failed:\n" + "\n".join(failures)
        raise LeakageViolationError(msg)


def _as_panel(
    data: npt.ArrayLike,
    *,
    name: str,
    min_rows: int = 1,
) -> npt.NDArray[np.float64]:
    """Validate and convert an input to a 1D `(T,)` or 2D `(T, N)` float64 array."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim not in (1, 2):
        msg = f"{name} must be 1D (T,) or 2D (T, N), got {arr.ndim}D with shape {arr.shape}."
        raise ValueError(msg)
    if arr.shape[0] < min_rows:
        msg = f"{name} requires at least {min_rows} time steps, got {arr.shape[0]}."
        raise ValueError(msg)
    if arr.ndim == 2 and arr.shape[1] < 1:
        msg = f"{name} 2D array must have at least 1 column, got shape {arr.shape}."
        raise ValueError(msg)
    check_array_memory(arr.shape, np.float64)
    return arr


def _arrays_match(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    """Compare two same-shape float64 arrays including exact NaN-mask equality."""
    if a.shape != b.shape:
        return False, float("inf")
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False, float("inf")
    finite_mask = ~nan_a
    if not np.any(finite_mask):
        return True, 0.0
    max_diff = float(np.max(np.abs(a[finite_mask] - b[finite_mask])))
    matched = bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    return matched, max_diff


def _strategy_returns(
    returns: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute 1D period strategy returns for 1D or 2D `(T, N)` inputs."""
    clean_w = np.where(np.isnan(weights), 0.0, weights)
    if returns.ndim == 1:
        return np.asarray(clean_w * returns, dtype=np.float64)
    return portfolio_returns(returns, clean_w)


def _safe_sharpe(strategy_returns: npt.NDArray[np.float64]) -> float:
    """Compute annualized Sharpe ratio, returning `0.0` for constant/insufficient series."""
    valid_count = int(np.sum(~np.isnan(strategy_returns)))
    if valid_count < 2:
        return 0.0
    sr = float(sharpe_ratio(strategy_returns))
    return 0.0 if math.isnan(sr) else sr


def shift_signals(
    signals: npt.ArrayLike,
    lag: int = 1,
    *,
    fill_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Shift signals along the time axis (`docs/conventions.md` §3.2).

    Parameters
    ----------
    signals : npt.ArrayLike
        1D `(T,)` or 2D `(T, N)` signal/weight panel.
    lag : int, default 1
        Positive lag delays signals forward in time (`out[t] = signals[t - lag]`,
        causal execution). Negative lag peeks into the future (`out[t] = signals[t + |lag|]`,
        look-ahead probe). Zero returns a copy.
    fill_value : float, default 0.0
        Boundary fill value for vacated leading (or trailing) time steps.
    """
    arr = _as_panel(signals, name="signals", min_rows=1)
    t_len = arr.shape[0]
    if abs(lag) >= t_len:
        msg = f"abs(lag)={abs(lag)} must be strictly less than T={t_len}."
        raise ValueError(msg)

    out = np.full(arr.shape, fill_value, dtype=np.float64)
    if lag == 0:
        out[...] = arr
    elif lag > 0:
        out[lag:] = arr[: t_len - lag]
    else:
        step = -lag
        out[: t_len - step] = arr[step:]
    return out


def check_future_poison(
    signal_fn: SignalFunc,
    panel: npt.ArrayLike,
    *,
    cutoffs: Sequence[int] | None = None,
    poison_values: Sequence[float] = (np.nan, 1e9, -1e9),
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> LeakageCheckResult:
    """Verify causal invariance: altering `panel[t_c + 1:]` must not alter `signal[:t_c + 1]`."""
    arr = _as_panel(panel, name="panel", min_rows=4)
    t_len = arr.shape[0]

    baseline = np.asarray(signal_fn(arr.copy()), dtype=np.float64)
    if baseline.shape != arr.shape:
        msg = f"signal_fn must return shape {arr.shape}, got {baseline.shape}."
        raise ValueError(msg)

    resolved_cutoffs = (
        [t_len // 4, t_len // 2, (3 * t_len) // 4] if cutoffs is None else list(cutoffs)
    )
    if not resolved_cutoffs:
        msg = "cutoffs must contain at least one index."
        raise ValueError(msg)

    worst_diff = 0.0
    first_failure: str | None = None

    for cutoff in resolved_cutoffs:
        if cutoff < 0 or cutoff >= t_len - 1:
            msg = f"Cutoff {cutoff} must satisfy 0 <= cutoff < T - 1 ({t_len - 1})."
            raise ValueError(msg)

        base_prefix = baseline[: cutoff + 1]
        for p_val in poison_values:
            poisoned = arr.copy()
            poisoned[cutoff + 1 :] = p_val
            out_poisoned = np.asarray(signal_fn(poisoned), dtype=np.float64)
            if out_poisoned.shape != arr.shape:
                return LeakageCheckResult(
                    name="future_poison",
                    passed=False,
                    max_abs_diff=float("inf"),
                    details=(
                        f"signal_fn returned incompatible shape {out_poisoned.shape} "
                        f"under future poison {p_val} at cutoff {cutoff}."
                    ),
                    metadata={"n_cutoffs": float(len(resolved_cutoffs))},
                )

            poison_prefix = out_poisoned[: cutoff + 1]
            matched, diff = _arrays_match(base_prefix, poison_prefix, atol=atol, rtol=rtol)
            if math.isinf(diff):
                worst_diff = float("inf")
            elif not math.isinf(worst_diff):
                worst_diff = max(worst_diff, diff)

            if not matched and first_failure is None:
                first_failure = (
                    f"Future poison ({p_val!r}) after t={cutoff} altered signals at t <= {cutoff} "
                    f"(max_abs_diff={diff:.3e})."
                )

    passed = first_failure is None
    details = (
        "All causal prefixes remained invariant under future poisoning."
        if passed
        else str(first_failure)
    )
    return LeakageCheckResult(
        name="future_poison",
        passed=passed,
        max_abs_diff=worst_diff,
        details=details,
        metadata={"n_cutoffs": float(len(resolved_cutoffs))},
    )


def evaluate_signal_lag_and_shift_probe(
    signal_fn: SignalFunc,
    inputs: npt.ArrayLike,
    period_returns: npt.ArrayLike,
    *,
    max_contemporaneous_sharpe_jump: float = 1.5,
    atol: float = 1e-12,
) -> LeakageCheckResult:
    """Audit `signal_fn` causality and measure Sharpe sensitivity across `+1`/`+2`/`0`/`-1` lags."""
    arr_inputs = _as_panel(inputs, name="inputs", min_rows=10)
    arr_returns = _as_panel(period_returns, name="period_returns", min_rows=10)
    if arr_inputs.shape != arr_returns.shape:
        msg = (
            f"inputs shape {arr_inputs.shape} does not match "
            f"period_returns shape {arr_returns.shape}."
        )
        raise ValueError(msg)

    poison_check = check_future_poison(signal_fn, arr_inputs, atol=atol)
    raw_w = np.asarray(signal_fn(arr_inputs.copy()), dtype=np.float64)
    clean_w = np.where(np.isnan(raw_w), 0.0, raw_w)

    sr_lag1 = _safe_sharpe(_strategy_returns(arr_returns, shift_signals(clean_w, lag=1)))
    sr_lag2 = _safe_sharpe(_strategy_returns(arr_returns, shift_signals(clean_w, lag=2)))
    sr_lag0 = _safe_sharpe(_strategy_returns(arr_returns, shift_signals(clean_w, lag=0)))
    sr_peek1 = _safe_sharpe(_strategy_returns(arr_returns, shift_signals(clean_w, lag=-1)))

    contemp_jump = abs(sr_lag0 - sr_lag1)
    jump_note = (
        f" (contemporaneous lag=0 Sharpe jump {contemp_jump:.2f} exceeds "
        f"threshold {max_contemporaneous_sharpe_jump:.2f}, confirming lag=0 would leak)"
        if contemp_jump > max_contemporaneous_sharpe_jump
        else ""
    )

    if poison_check.passed:
        details = (
            f"Signal passed causality check: lag+1 SR={sr_lag1:.2f}, "
            f"lag+2 SR={sr_lag2:.2f}, lag0 SR={sr_lag0:.2f}, peek-1 SR={sr_peek1:.2f}"
            f"{jump_note}."
        )
    else:
        details = f"Signal failed internal causality check before lag shift: {poison_check.details}"

    return LeakageCheckResult(
        name="shift_probe",
        passed=poison_check.passed,
        max_abs_diff=poison_check.max_abs_diff,
        details=details,
        metadata={
            "sharpe_causal_lag1": sr_lag1,
            "sharpe_delayed_lag2": sr_lag2,
            "sharpe_contemporaneous_lag0": sr_lag0,
            "sharpe_peek_minus1": sr_peek1,
            "contemporaneous_sharpe_jump": contemp_jump,
        },
    )


def check_in_fold_preprocessor(
    preprocessor_fn: PreprocessorFunc,
    panel: npt.ArrayLike,
    *,
    split_idx: int | None = None,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> LeakageCheckResult:
    """Verify a stateful preprocessor fits only on `panel[:split_idx]` without test leakage."""
    arr = _as_panel(panel, name="panel", min_rows=4)
    t_len = arr.shape[0]
    resolved_split = t_len // 2 if split_idx is None else split_idx
    if resolved_split < 2 or resolved_split > t_len - 2:
        msg = f"split_idx must satisfy 2 <= split_idx <= T - 2 ({t_len - 2}), got {resolved_split}."
        raise ValueError(msg)

    train_clean, test_clean = preprocessor_fn(arr.copy(), resolved_split)
    if train_clean.shape[0] != resolved_split or test_clean.shape[0] != t_len - resolved_split:
        msg = (
            f"preprocessor_fn must split T={t_len} into ({resolved_split}, ...) and "
            f"({t_len - resolved_split}, ...), got {train_clean.shape} and {test_clean.shape}."
        )
        raise ValueError(msg)

    worst_diff = 0.0
    failures: list[str] = []

    # Test 1: Training outputs must be invariant to extreme scaling/shift or NaN in the test fold.
    poison_specs: list[tuple[str, npt.NDArray[np.float64]]] = []
    scaled = arr.copy()
    scaled[resolved_split:] = scaled[resolved_split:] * 1000.0 + 500.0
    poison_specs.append(("extreme scaling/shift", scaled))

    with_nan = arr.copy()
    with_nan[resolved_split:] = np.nan
    poison_specs.append(("NaN test fold", with_nan))

    for label, perturbed in poison_specs:
        train_pert, _ = preprocessor_fn(perturbed, resolved_split)
        matched, diff = _arrays_match(train_clean, train_pert, atol=atol, rtol=rtol)
        worst_diff = diff if math.isinf(diff) or math.isinf(worst_diff) else max(worst_diff, diff)
        if not matched:
            failures.append(
                f"Test 1 ({label} in test fold) altered transformed training data "
                f"(max_abs_diff={diff:.3e}; indicates full-sample fit before CV split)."
            )
            break

    # Test 2: Early test rows must be invariant to extreme perturbations in later test rows.
    mid_test = resolved_split + (t_len - resolved_split) // 2
    tail_poisoned = arr.copy()
    tail_poisoned[mid_test:] = tail_poisoned[mid_test:] + 1e6
    _, test_pert = preprocessor_fn(tail_poisoned, resolved_split)
    head_len = mid_test - resolved_split
    matched_test, diff_test = _arrays_match(
        test_clean[:head_len],
        test_pert[:head_len],
        atol=atol,
        rtol=rtol,
    )
    worst_diff = (
        diff_test if math.isinf(diff_test) or math.isinf(worst_diff) else max(worst_diff, diff_test)
    )
    if not matched_test:
        failures.append(
            f"Test 2 (late-test poisoning after t={mid_test}) altered early test outputs "
            f"(max_abs_diff={diff_test:.3e}; indicates preprocessor refit on the test fold)."
        )

    passed = not failures
    details = (
        "Preprocessor fit is strictly isolated to the training fold."
        if passed
        else " ".join(failures)
    )
    return LeakageCheckResult(
        name="in_fold_preprocessor",
        passed=passed,
        max_abs_diff=worst_diff,
        details=details,
        metadata={"split_idx": float(resolved_split)},
    )


def check_shuffled_labels(
    fit_predict_fn: FitPredictFunc,
    features: npt.ArrayLike,
    target_returns: npt.ArrayLike,
    *,
    split_idx: int | None = None,
    n_permutations: int = 40,
    max_abs_mean_sharpe: float = 1.0,
    atol: float = 1e-12,
    seed: int = 42,
) -> LeakageCheckResult:
    """Detect direct test-label leakage and excessive OOS Sharpe under permuted training labels."""
    if n_permutations < 1:
        msg = f"n_permutations must be >= 1, got {n_permutations}."
        raise ValueError(msg)

    x_arr = _as_panel(features, name="features", min_rows=6)
    y_arr = _as_panel(target_returns, name="target_returns", min_rows=6)
    if x_arr.shape[0] != y_arr.shape[0]:
        msg = (
            f"features length T={x_arr.shape[0]} does not match "
            f"target_returns length T={y_arr.shape[0]}."
        )
        raise ValueError(msg)

    t_len = x_arr.shape[0]
    resolved_split = t_len // 2 if split_idx is None else split_idx
    if resolved_split < 2 or resolved_split > t_len - 2:
        msg = f"split_idx must satisfy 2 <= split_idx <= T - 2 ({t_len - 2}), got {resolved_split}."
        raise ValueError(msg)

    x_train, x_test = x_arr[:resolved_split], x_arr[resolved_split:]
    y_train, y_test = y_arr[:resolved_split], y_arr[resolved_split:]

    pred_base = np.asarray(
        fit_predict_fn(x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy()),
        dtype=np.float64,
    )
    if pred_base.shape != y_test.shape:
        msg = f"fit_predict_fn must return shape {y_test.shape}, got {pred_base.shape}."
        raise ValueError(msg)

    # Part A: Test-label poisoning must not change out-of-sample predictions.
    y_test_poison = -100.0 * y_test + 999.0
    pred_poison = np.asarray(
        fit_predict_fn(x_train.copy(), y_train.copy(), x_test.copy(), y_test_poison),
        dtype=np.float64,
    )
    if pred_poison.shape != y_test.shape:
        msg = f"fit_predict_fn returned incompatible shape {pred_poison.shape} on poisoned y_test."
        raise ValueError(msg)

    labels_invariant, label_diff = _arrays_match(pred_base, pred_poison, atol=atol, rtol=atol)
    if not labels_invariant:
        return LeakageCheckResult(
            name="shuffled_labels",
            passed=False,
            max_abs_diff=label_diff,
            details=(
                "Model predictions changed when y_test was perturbed "
                f"(direct test-label leakage, max_abs_diff={label_diff:.3e})."
            ),
            metadata={"n_permutations": float(n_permutations)},
        )

    # Part B: Permuting training labels must destroy predictive OOS Sharpe.
    rng = np.random.default_rng(seed)
    perm_sharpes = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_idx = rng.permutation(y_train.shape[0])
        y_train_perm = y_train[perm_idx]
        pred_perm = np.asarray(
            fit_predict_fn(x_train.copy(), y_train_perm, x_test.copy(), y_test.copy()),
            dtype=np.float64,
        )
        strat_ret = _strategy_returns(y_test, pred_perm)
        perm_sharpes[i] = _safe_sharpe(strat_ret)

    mean_sr = float(np.mean(perm_sharpes))
    std_sr = float(np.std(perm_sharpes, ddof=1)) if n_permutations > 1 else 0.0
    null_ok = abs(mean_sr) <= max_abs_mean_sharpe

    details = (
        f"Permuted training labels yielded null-consistent mean OOS Sharpe {mean_sr:.3f} "
        f"(std={std_sr:.3f}, threshold={max_abs_mean_sharpe:.3f})."
        if null_ok
        else (
            f"Permuted training labels retained non-null mean OOS Sharpe {mean_sr:.3f} "
            f"(exceeds max_abs_mean_sharpe={max_abs_mean_sharpe:.3f}; possible target leakage)."
        )
    )
    return LeakageCheckResult(
        name="shuffled_labels",
        passed=null_ok,
        max_abs_diff=label_diff,
        details=details,
        metadata={
            "mean_shuffled_sharpe": mean_sr,
            "std_shuffled_sharpe": std_sr,
            "n_permutations": float(n_permutations),
        },
    )


def random_signal_sharpe_baseline(
    period_returns: npt.ArrayLike,
    *,
    n_trials: int = 200,
    long_only: bool = True,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Generate a deterministic null distribution of annualized Sharpes from random weights."""
    if n_trials < 1:
        msg = f"n_trials must be >= 1, got {n_trials}."
        raise ValueError(msg)

    arr = _as_panel(period_returns, name="period_returns", min_rows=2)
    t_len = arr.shape[0]
    n_assets = 1 if arr.ndim == 1 else arr.shape[1]
    check_array_memory((n_trials, t_len, n_assets), np.float64)

    rng = np.random.default_rng(seed)
    sharpes = np.empty(n_trials, dtype=np.float64)
    sign_choices = np.array([-1.0, 1.0], dtype=np.float64)

    for idx in range(n_trials):
        weights: npt.NDArray[np.float64]
        if arr.ndim == 1:
            if long_only:
                weights = np.asarray(rng.uniform(0.0, 1.0, size=t_len), dtype=np.float64)
            else:
                weights = np.asarray(rng.choice(sign_choices, size=t_len), dtype=np.float64)
        else:
            if long_only:
                raw = rng.exponential(1.0, size=(t_len, n_assets))
                weights = np.asarray(raw / np.sum(raw, axis=1, keepdims=True), dtype=np.float64)
            elif n_assets == 1:
                weights = np.asarray(rng.choice(sign_choices, size=(t_len, 1)), dtype=np.float64)
            else:
                raw = rng.normal(0.0, 1.0, size=(t_len, n_assets))
                centered = raw - np.mean(raw, axis=1, keepdims=True)
                l1_norm = np.sum(np.abs(centered), axis=1, keepdims=True)
                fallback = np.zeros_like(centered)
                fallback[:, 0] = 0.5
                fallback[:, 1] = -0.5
                weights = np.asarray(
                    np.where(
                        l1_norm > 0.0,
                        centered / np.where(l1_norm > 0.0, l1_norm, 1.0),
                        fallback,
                    ),
                    dtype=np.float64,
                )

        sharpes[idx] = _safe_sharpe(_strategy_returns(arr, weights))

    return sharpes, float(np.mean(sharpes)), float(np.percentile(sharpes, 95.0))


def run_leakage_suite(
    signal_fn: SignalFunc,
    panel: npt.ArrayLike,
    period_returns: npt.ArrayLike,
    *,
    preprocessor_fn: PreprocessorFunc | None = None,
    fit_predict_fn: FitPredictFunc | None = None,
    split_idx: int | None = None,
    seed: int = 42,
) -> LeakageSuiteReport:
    """Run all applicable leakage and sanity checks and aggregate into a report."""
    checks: list[LeakageCheckResult] = [
        check_future_poison(signal_fn, panel),
        evaluate_signal_lag_and_shift_probe(signal_fn, panel, period_returns),
    ]
    if preprocessor_fn is not None:
        checks.append(check_in_fold_preprocessor(preprocessor_fn, panel, split_idx=split_idx))
    if fit_predict_fn is not None:
        checks.append(
            check_shuffled_labels(
                fit_predict_fn,
                panel,
                period_returns,
                split_idx=split_idx,
                seed=seed,
            )
        )
    return LeakageSuiteReport(passed=all(c.passed for c in checks), checks=tuple(checks))
