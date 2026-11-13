"""Unit, known-answer, property, and leaky-fixture tests for `quantlab.validation.leakage`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.validation import (
    LeakageCheckResult,
    LeakageSuiteReport,
    LeakageViolationError,
    check_future_poison,
    check_in_fold_preprocessor,
    check_shuffled_labels,
    evaluate_signal_lag_and_shift_probe,
    random_signal_sharpe_baseline,
    run_leakage_suite,
    shift_signals,
)


def _causal_trailing_momentum_1d(
    panel: npt.NDArray[np.float64], window: int = 5
) -> npt.NDArray[np.float64]:
    """Causal trailing rolling mean sign signal: depends strictly on `panel[:t + 1]`."""
    out = np.zeros_like(panel, dtype=np.float64)
    for t in range(window - 1, panel.shape[0]):
        trailing = panel[t - window + 1 : t + 1]
        if np.any(np.isnan(trailing)):
            out[t] = np.nan
        else:
            out[t] = 1.0 if float(np.mean(trailing)) >= 0.0 else -1.0
    return out


def _causal_trailing_momentum_2d(
    panel: npt.NDArray[np.float64], window: int = 5
) -> npt.NDArray[np.float64]:
    """Causal cross-sectional normalized trailing momentum weights on `(T, N)`."""
    t_len, n_assets = panel.shape
    out = np.zeros((t_len, n_assets), dtype=np.float64)
    for t in range(window - 1, t_len):
        trailing = panel[t - window + 1 : t + 1]
        if np.any(np.isnan(trailing)):
            out[t, :] = np.nan
        else:
            raw = np.where(np.mean(trailing, axis=0) >= 0.0, 1.0, 0.0)
            total = float(np.sum(raw))
            out[t, :] = raw / total if total > 0.0 else 1.0 / n_assets
    return out


# ---------------------------------------------------------------------------
# 1. shift_signals: Known-Answer, Edge Cases & Hypothesis Property Tests
# ---------------------------------------------------------------------------


def test_shift_signals_known_answer_1d_and_2d() -> None:
    sig_1d = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    np.testing.assert_array_equal(shift_signals(sig_1d, lag=0), sig_1d)
    np.testing.assert_array_equal(shift_signals(sig_1d, lag=1), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        shift_signals(sig_1d, lag=2, fill_value=-1.0), [-1.0, -1.0, 1.0, 2.0]
    )
    np.testing.assert_array_equal(shift_signals(sig_1d, lag=-1), [2.0, 3.0, 4.0, 0.0])

    sig_2d = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float64)
    expected_2d_lag1 = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]], dtype=np.float64)
    np.testing.assert_array_equal(shift_signals(sig_2d, lag=1), expected_2d_lag1)


@given(
    t_len=st.integers(min_value=5, max_value=40),
    n_cols=st.integers(min_value=1, max_value=5),
    lag=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=40)
def test_shift_signals_causal_and_peek_prefix_conservation(
    t_len: int,
    n_cols: int,
    lag: int,
) -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(t_len, n_cols))

    delayed = shift_signals(x, lag=lag)
    np.testing.assert_array_equal(delayed[:lag], 0.0)
    np.testing.assert_array_equal(delayed[lag:], x[: t_len - lag])

    peeked = shift_signals(x, lag=-lag)
    np.testing.assert_array_equal(peeked[: t_len - lag], x[lag:])
    np.testing.assert_array_equal(peeked[t_len - lag :], 0.0)


def test_shift_signals_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        shift_signals(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="at least 1 column"):
        shift_signals(np.zeros((4, 0)))
    with pytest.raises(ValueError, match="strictly less than T"):
        shift_signals(np.array([1.0, 2.0, 3.0]), lag=3)
    with pytest.raises(ValueError, match="strictly less than T"):
        shift_signals(np.array([1.0, 2.0, 3.0]), lag=-3)


# ---------------------------------------------------------------------------
# 2. check_future_poison: Causal Pass vs. Deliberately Leaky Failures
# ---------------------------------------------------------------------------


def test_future_poison_passes_on_causal_1d_and_2d_signals() -> None:
    rng = np.random.default_rng(42)
    rets_1d = rng.normal(0.0, 0.01, size=60)
    res_1d = check_future_poison(_causal_trailing_momentum_1d, rets_1d)
    assert isinstance(res_1d, LeakageCheckResult)
    assert res_1d.passed
    assert res_1d.max_abs_diff == 0.0

    rets_2d = rng.normal(0.0, 0.01, size=(60, 4))
    res_2d = check_future_poison(_causal_trailing_momentum_2d, rets_2d)
    assert res_2d.passed
    assert res_2d.max_abs_diff == 0.0


def test_future_poison_fails_on_leaky_fixtures() -> None:
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0, 0.01, size=50)

    # Leaky 1: Direct future peek w[t] = sign(r[t + 1])
    def future_peek(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.zeros_like(x)
        out[:-1] = np.sign(x[1:])
        return out

    res_peek = check_future_poison(future_peek, rets)
    assert not res_peek.passed
    assert res_peek.name == "future_poison"
    assert "altered signals" in res_peek.details

    # Leaky 2: Full-sample z-score normalization
    def full_sample_zscore(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray((x - np.mean(x)) / np.std(x), dtype=np.float64)

    res_z = check_future_poison(full_sample_zscore, rets)
    assert not res_z.passed
    assert np.isinf(res_z.max_abs_diff) or res_z.max_abs_diff > 1e-3

    # Leaky 3: Future NaN propagation back in time
    def reverse_cummean(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.cumsum(x[::-1])[::-1], dtype=np.float64)

    res_nan = check_future_poison(reverse_cummean, rets)
    assert not res_nan.passed
    assert np.isinf(res_nan.max_abs_diff)


def test_future_poison_edge_and_error_cases() -> None:
    rets = np.zeros(20, dtype=np.float64)
    with pytest.raises(ValueError, match="at least 4 time steps"):
        check_future_poison(_causal_trailing_momentum_1d, np.zeros(3))
    with pytest.raises(ValueError, match="at least one index"):
        check_future_poison(_causal_trailing_momentum_1d, rets, cutoffs=[])
    with pytest.raises(ValueError, match="0 <= cutoff < T - 1"):
        check_future_poison(_causal_trailing_momentum_1d, rets, cutoffs=[19])
    with pytest.raises(ValueError, match="must return shape"):
        check_future_poison(lambda x: x[:-1], rets)

    def shape_breaking_on_nan(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if np.any(np.isnan(x)):
            return x[:-1]
        return x.copy()

    res_bad_shape = check_future_poison(shape_breaking_on_nan, rets)
    assert not res_bad_shape.passed
    assert "incompatible shape" in res_bad_shape.details


# ---------------------------------------------------------------------------
# 3. evaluate_signal_lag_and_shift_probe: +1 / +2 / 0 / -1 Alignment Audits
# ---------------------------------------------------------------------------


def test_shift_probe_demonstrates_contemporaneous_jump_and_catches_internal_peek() -> None:
    rng = np.random.default_rng(101)
    rets = rng.normal(0.0, 0.015, size=252)

    # Causal contemporaneous sign signal w[t] = sign(r[t]): legal at close[t],
    # must earn r[t+1] (lag=+1), NOT r[t] (lag=0)!
    def sign_of_today(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sign(x)

    res = evaluate_signal_lag_and_shift_probe(sign_of_today, rets, rets)
    assert res.passed
    assert res.name == "shift_probe"
    # lag=0 on w[t] = sign(r[t]) earns |r[t]| every bar -> absurdly inflated Sharpe
    assert res.metadata["sharpe_contemporaneous_lag0"] > 5.0
    assert abs(res.metadata["sharpe_causal_lag1"]) < 2.0
    assert res.metadata["contemporaneous_sharpe_jump"] > 4.0
    assert "lag=0 would leak" in res.details

    # 2D portfolio evaluation also works cleanly
    rets_2d = rng.normal(0.0002, 0.01, size=(120, 3))
    res_2d = evaluate_signal_lag_and_shift_probe(_causal_trailing_momentum_2d, rets_2d, rets_2d)
    assert res_2d.passed
    assert "sharpe_delayed_lag2" in res_2d.metadata
    assert "sharpe_peek_minus1" in res_2d.metadata

    # Signal with internal future peek must fail shift_probe
    def internal_peek(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.zeros_like(x)
        out[:-1] = np.sign(x[1:])
        return out

    res_fail = evaluate_signal_lag_and_shift_probe(internal_peek, rets, rets)
    assert not res_fail.passed
    assert "failed internal causality" in res_fail.details

    with pytest.raises(ValueError, match="does not match period_returns shape"):
        evaluate_signal_lag_and_shift_probe(sign_of_today, rets, rets[:-1])


# ---------------------------------------------------------------------------
# 4. check_in_fold_preprocessor: In-Fold Isolation vs. Full-Sample / Test Refit
# ---------------------------------------------------------------------------


def test_in_fold_preprocessor_causal_pass_and_leaky_failures() -> None:
    rng = np.random.default_rng(17)
    panel = rng.normal(0.05, 0.2, size=(80, 3))

    def causal_in_fold_zscore(
        arr: npt.NDArray[np.float64],
        split_idx: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        train = arr[:split_idx]
        test = arr[split_idx:]
        mu = np.mean(train, axis=0, keepdims=True)
        sd = np.where(
            np.std(train, axis=0, keepdims=True) > 0.0, np.std(train, axis=0, keepdims=True), 1.0
        )
        return np.asarray((train - mu) / sd, dtype=np.float64), np.asarray(
            (test - mu) / sd, dtype=np.float64
        )

    res_ok = check_in_fold_preprocessor(causal_in_fold_zscore, panel, split_idx=40)
    assert res_ok.passed
    assert res_ok.max_abs_diff == 0.0

    # Leaky 1: Full-sample fit before splitting
    def full_sample_scaler(
        arr: npt.NDArray[np.float64],
        split_idx: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        mu = np.mean(arr, axis=0, keepdims=True)
        sd = np.std(arr, axis=0, keepdims=True)
        scaled = (arr - mu) / sd
        return np.asarray(scaled[:split_idx], dtype=np.float64), np.asarray(
            scaled[split_idx:], dtype=np.float64
        )

    res_full = check_in_fold_preprocessor(full_sample_scaler, panel, split_idx=40)
    assert not res_full.passed
    assert "Test 1" in res_full.details

    # Leaky 2: Separate fit on test fold
    def test_refit_scaler(
        arr: npt.NDArray[np.float64],
        split_idx: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        train = arr[:split_idx]
        test = arr[split_idx:]
        tr_out = (train - np.mean(train, axis=0, keepdims=True)) / np.std(
            train, axis=0, keepdims=True
        )
        te_out = (test - np.mean(test, axis=0, keepdims=True)) / np.std(test, axis=0, keepdims=True)
        return np.asarray(tr_out, dtype=np.float64), np.asarray(te_out, dtype=np.float64)

    res_refit = check_in_fold_preprocessor(test_refit_scaler, panel, split_idx=40)
    assert not res_refit.passed
    assert "Test 2" in res_refit.details

    with pytest.raises(ValueError, match="split_idx must satisfy"):
        check_in_fold_preprocessor(causal_in_fold_zscore, panel, split_idx=1)
    with pytest.raises(ValueError, match="must split"):
        check_in_fold_preprocessor(lambda a, s: (a[: s - 1], a[s:]), panel, split_idx=40)


# ---------------------------------------------------------------------------
# 5. check_shuffled_labels: Causal Ridge Pass vs. Direct/Oracle Label Leakage
# ---------------------------------------------------------------------------


def test_shuffled_labels_causal_and_leaky_models() -> None:
    rng = np.random.default_rng(2026)
    x_feat = rng.normal(size=(160, 4))
    y_ret = rng.normal(0.0, 0.01, size=160)

    # Causal Ridge predictor fit strictly on (X_train, y_train)
    def causal_ridge(
        x_tr: npt.NDArray[np.float64],
        y_tr: npt.NDArray[np.float64],
        x_te: npt.NDArray[np.float64],
        _y_te: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        reg = 10.0 * np.eye(x_tr.shape[1], dtype=np.float64)
        beta = np.linalg.solve(x_tr.T @ x_tr + reg, x_tr.T @ y_tr)
        return np.asarray(np.sign(x_te @ beta), dtype=np.float64)

    res_ok = check_shuffled_labels(causal_ridge, x_feat, y_ret, n_permutations=30, seed=99)
    assert res_ok.passed
    assert abs(res_ok.metadata["mean_shuffled_sharpe"]) <= 1.0

    # 2D target returns also supported (multi-output ridge fit on X_train, Y_train)
    y_2d = rng.normal(0.0, 0.01, size=(160, 2))

    def causal_2d_ridge(
        x_tr: npt.NDArray[np.float64],
        y_tr: npt.NDArray[np.float64],
        x_te: npt.NDArray[np.float64],
        _y_te: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        reg = 10.0 * np.eye(x_tr.shape[1], dtype=np.float64)
        beta = np.linalg.solve(x_tr.T @ x_tr + reg, x_tr.T @ y_tr)
        return np.asarray(np.sign(x_te @ beta), dtype=np.float64)

    res_2d_ok = check_shuffled_labels(causal_2d_ridge, x_feat, y_2d, n_permutations=25)
    assert res_2d_ok.passed

    # Leaky 1: Direct peek into y_test -> caught by Part A immediately
    def direct_label_leak(
        x_tr: npt.NDArray[np.float64],
        y_tr: npt.NDArray[np.float64],
        x_te: npt.NDArray[np.float64],
        y_te: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        base = causal_ridge(x_tr, y_tr, x_te, y_te)
        return np.asarray(base + 0.1 * y_te, dtype=np.float64)

    res_direct = check_shuffled_labels(direct_label_leak, x_feat, y_ret)
    assert not res_direct.passed
    assert "direct test-label leakage" in res_direct.details

    # Leaky 2: Oracle feature embedded in X[:, 0] == target_returns (caught by Part B)
    x_with_oracle = np.column_stack([y_ret, x_feat[:, 1:]])

    def cheat_feature_model(
        _x_tr: npt.NDArray[np.float64],
        _y_tr: npt.NDArray[np.float64],
        x_te: npt.NDArray[np.float64],
        _y_te: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.asarray(np.sign(x_te[:, 0]), dtype=np.float64)

    res_oracle = check_shuffled_labels(cheat_feature_model, x_with_oracle, y_ret, n_permutations=15)
    assert not res_oracle.passed
    assert "retained non-null mean OOS Sharpe" in res_oracle.details

    with pytest.raises(ValueError, match="n_permutations must be >= 1"):
        check_shuffled_labels(causal_ridge, x_feat, y_ret, n_permutations=0)
    with pytest.raises(ValueError, match="does not match target_returns length"):
        check_shuffled_labels(causal_ridge, x_feat[:-1], y_ret)
    with pytest.raises(ValueError, match="split_idx must satisfy"):
        check_shuffled_labels(causal_ridge, x_feat, y_ret, split_idx=1)
    with pytest.raises(ValueError, match="must return shape"):
        check_shuffled_labels(lambda _a, _b, x_te, _d: x_te[:, 0][:-1], x_feat, y_ret)


# ---------------------------------------------------------------------------
# 6. random_signal_sharpe_baseline & run_leakage_suite
# ---------------------------------------------------------------------------


def test_random_signal_sharpe_baseline_and_suite_report() -> None:
    rng = np.random.default_rng(55)
    rets_1d = rng.normal(0.0, 0.01, size=252)
    dist1, mean1, p95_1 = random_signal_sharpe_baseline(
        rets_1d, n_trials=60, long_only=False, seed=7
    )
    dist2, mean2, p95_2 = random_signal_sharpe_baseline(
        rets_1d, n_trials=60, long_only=False, seed=7
    )
    assert dist1.shape == (60,)
    np.testing.assert_array_equal(dist1, dist2)
    assert mean1 == pytest.approx(mean2)
    assert p95_1 == pytest.approx(p95_2)
    assert abs(mean1) < 1.0

    # 1D long-only & 2D long-only / long-short
    dist_lo, _, _ = random_signal_sharpe_baseline(rets_1d, n_trials=20, long_only=True, seed=1)
    assert dist_lo.shape == (20,)

    rets_2d = rng.normal(0.0, 0.01, size=(100, 4))
    dist_2d_lo, _, _ = random_signal_sharpe_baseline(rets_2d, n_trials=20, long_only=True, seed=2)
    dist_2d_ls, _, _ = random_signal_sharpe_baseline(rets_2d, n_trials=20, long_only=False, seed=3)
    dist_2d_single, _, _ = random_signal_sharpe_baseline(
        rets_2d[:, :1], n_trials=10, long_only=False, seed=4
    )
    assert dist_2d_lo.shape == (20,)
    assert dist_2d_ls.shape == (20,)
    assert dist_2d_single.shape == (10,)

    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        random_signal_sharpe_baseline(rets_1d, n_trials=0)

    # Full run_leakage_suite: causal passes silently
    panel_2d = rets_2d.copy()

    def causal_pre(
        arr: npt.NDArray[np.float64],
        split_idx: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        tr = arr[:split_idx]
        mu = np.mean(tr, axis=0, keepdims=True)
        return np.asarray(tr - mu, dtype=np.float64), np.asarray(
            arr[split_idx:] - mu, dtype=np.float64
        )

    def causal_model(
        _x_tr: npt.NDArray[np.float64],
        y_tr: npt.NDArray[np.float64],
        x_te: npt.NDArray[np.float64],
        _y_te: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        signs = np.where(np.mean(y_tr, axis=0) >= 0.0, 1.0 / y_tr.shape[1], 0.0)
        return np.broadcast_to(signs, (x_te.shape[0], y_tr.shape[1])).copy()

    report_ok = run_leakage_suite(
        _causal_trailing_momentum_2d,
        panel_2d,
        rets_2d,
        preprocessor_fn=causal_pre,
        fit_predict_fn=causal_model,
    )
    assert isinstance(report_ok, LeakageSuiteReport)
    assert report_ok.passed
    assert len(report_ok.checks) == 4
    report_ok.raise_if_failed()

    # Failing suite raises LeakageViolationError
    report_bad = run_leakage_suite(
        lambda x: np.vstack([np.sign(x[1:]), np.zeros((1, x.shape[1]))]),
        panel_2d,
        rets_2d,
    )
    assert not report_bad.passed
    with pytest.raises(LeakageViolationError, match="Leakage validation suite failed"):
        report_bad.raise_if_failed()
