"""Leakage test suite, synchronized Purged K-Fold, CPCV, Deflated Sharpe Ratio, and PBO."""

from quantlab.validation.leakage import (
    FitPredictFunc,
    LeakageCheckResult,
    LeakageSuiteReport,
    LeakageViolationError,
    PreprocessorFunc,
    SignalFunc,
    check_future_poison,
    check_in_fold_preprocessor,
    check_shuffled_labels,
    evaluate_signal_lag_and_shift_probe,
    random_signal_sharpe_baseline,
    run_leakage_suite,
    shift_signals,
)

__all__ = [
    "FitPredictFunc",
    "LeakageCheckResult",
    "LeakageSuiteReport",
    "LeakageViolationError",
    "PreprocessorFunc",
    "SignalFunc",
    "check_future_poison",
    "check_in_fold_preprocessor",
    "check_shuffled_labels",
    "evaluate_signal_lag_and_shift_probe",
    "random_signal_sharpe_baseline",
    "run_leakage_suite",
    "shift_signals",
]
