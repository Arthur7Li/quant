"""Known-answer, edge-case, and Hypothesis property tests for `quantlab.core`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.stats
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.core import (
    TRADING_DAYS_PER_YEAR,
    PerformanceSummary,
    aggregate_multi_period_returns,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compound_returns,
    compute_turnover,
    drawdown_series,
    excess_returns,
    log_to_simple_returns,
    max_drawdown,
    portfolio_returns,
    prices_to_returns,
    sharpe_confidence_interval,
    sharpe_ratio,
    sharpe_standard_error,
    simple_to_log_returns,
    sortino_ratio,
    summarize_performance,
    wealth_index,
)


def test_known_answer_returns_conversions_and_compounding() -> None:
    """Verify simple/log conversions, price-to-return, compounding, and wealth index."""
    simple = np.array([0.10, -0.10, 0.25])
    log_r = simple_to_log_returns(simple)
    expected_log = np.log(np.array([1.10, 0.90, 1.25]))
    np.testing.assert_allclose(log_r, expected_log, rtol=1e-12)
    np.testing.assert_allclose(log_to_simple_returns(log_r), simple, rtol=1e-12)

    # Prices to returns (1D and prepend_nan)
    prices = np.array([100.0, 110.0, 99.0])
    np.testing.assert_allclose(prices_to_returns(prices), np.array([0.10, -0.10]), rtol=1e-12)
    prep = prices_to_returns(prices, log=True, prepend_nan=True)
    assert np.isnan(prep[0])
    np.testing.assert_allclose(prep[1:], np.log(np.array([1.10, 0.90])), rtol=1e-12)

    # 2D prices with prepend_nan
    prices_2d = np.array([[100.0, 50.0], [110.0, 55.0]])
    prep_2d = prices_to_returns(prices_2d, prepend_nan=True)
    assert prep_2d.shape == (2, 2)
    assert np.all(np.isnan(prep_2d[0, :]))
    np.testing.assert_allclose(prep_2d[1, :], np.array([0.10, 0.10]), rtol=1e-12)

    # Compounding: 1.1 * 0.9 * 1.25 - 1 = 0.2375
    tot = compound_returns(simple)
    assert isinstance(tot, float)
    assert tot == pytest.approx(0.2375, rel=1e-12)

    # Wealth index with initial_wealth and include_initial
    w = wealth_index(simple, initial_wealth=100.0, include_initial=True)
    np.testing.assert_allclose(w, np.array([100.0, 110.0, 99.0, 123.75]), rtol=1e-12)

    # Multi-period block compounding (5 bars -> 2 blocks of 2, trailing bar dropped)
    five = np.array([0.10, 0.10, -0.20, 0.25, 0.50])
    agg = aggregate_multi_period_returns(five, window=2)
    np.testing.assert_allclose(agg, np.array([0.21, 0.0]), rtol=1e-12, atol=1e-12)

    # Portfolio returns (static 1D weights and time-varying 2D weights)
    rets_2d = np.array([[0.10, -0.05], [0.02, 0.04]])
    p_static = portfolio_returns(rets_2d, np.array([0.6, 0.4]))
    np.testing.assert_allclose(p_static, np.array([0.04, 0.028]), rtol=1e-12)

    w_2d = np.array([[1.0, 0.0], [0.5, 0.5]])
    p_dyn = portfolio_returns(rets_2d, w_2d)
    np.testing.assert_allclose(p_dyn, np.array([0.10, 0.03]), rtol=1e-12)

    # Excess returns with annualized scalar rf and 1D time-varying rf
    ex = excess_returns(np.array([[0.01, 0.02], [0.03, 0.04]]), 0.0252, periods_per_year=252)
    np.testing.assert_allclose(ex, np.array([[0.0099, 0.0199], [0.0299, 0.0399]]), rtol=1e-12)
    ex_1d_rf = excess_returns(
        np.array([[0.01, 0.02], [0.03, 0.04]]),
        np.array([0.001, 0.002]),
        rf_is_annualized=False,
    )
    np.testing.assert_allclose(ex_1d_rf, np.array([[0.009, 0.019], [0.028, 0.038]]), rtol=1e-12)


def test_known_answer_metrics_drawdown_turnover_and_sharpe_se() -> None:
    """Verify annualized metrics, bar-0 drawdown anchor, turnover, and Sharpe SE."""
    # Drawdown anchored at W_0 = 1.0: immediate -10% loss on bar 0 must be -0.10!
    rets = np.array([-0.10, 0.20, -0.25, 0.10])
    # Wealth path from W0=1: [1.0, 0.90, 1.08, 0.81, 0.891]
    # Running peak:        [1.0, 1.00, 1.08, 1.08, 1.080]
    expected_dd = np.array([-0.10, 0.0, -0.25, 0.891 / 1.08 - 1.0])
    dd = drawdown_series(rets)
    np.testing.assert_allclose(dd, expected_dd, rtol=1e-12, atol=1e-12)
    assert max_drawdown(rets) == pytest.approx(0.25, rel=1e-12)

    # Annualized return & vol on 2-bar simple example (periods_per_year=2)
    r_two = np.array([0.10, 0.10])
    assert annualized_return(r_two, periods_per_year=2, geometric=True) == pytest.approx(0.21)
    assert annualized_return(r_two, periods_per_year=2, geometric=False) == pytest.approx(0.20)
    assert math.isnan(float(sharpe_ratio(r_two)))
    assert math.isnan(float(calmar_ratio(r_two)))

    # 2D metrics panel
    r_mat = np.array([[0.01, -0.02], [0.03, 0.01], [-0.01, 0.02], [0.02, -0.01]])
    vols = annualized_volatility(r_mat)
    assert isinstance(vols, np.ndarray) and vols.shape == (2,)
    srs = sharpe_ratio(r_mat)
    assert isinstance(srs, np.ndarray) and srs.shape == (2,)
    sorts = sortino_ratio(r_mat)
    assert isinstance(sorts, np.ndarray) and sorts.shape == (2,)
    mdds = max_drawdown(r_mat)
    assert isinstance(mdds, np.ndarray) and mdds.shape == (2,)
    calms = calmar_ratio(r_mat)
    assert isinstance(calms, np.ndarray) and calms.shape == (2,)

    # Hand-computed Mertens / Bailey-López de Prado Sharpe SE
    x = np.array([0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.04, 0.01])
    sr_sub = float(np.mean(x)) / float(np.std(x, ddof=1))
    skew_x = float(scipy.stats.skew(x, bias=False))
    kurt_x = float(scipy.stats.kurtosis(x, fisher=False, bias=False))
    expected_var_sub = (1.0 - skew_x * sr_sub + ((kurt_x - 1.0) / 4.0) * (sr_sub**2)) / (len(x) - 1)
    expected_se = math.sqrt(TRADING_DAYS_PER_YEAR * expected_var_sub)
    actual_se = sharpe_standard_error(x, max_autocorr_lags=0)
    assert actual_se == pytest.approx(expected_se, rel=1e-12)

    # Lo (2002) positive autocorrelation inflates SE relative to i.i.d. SE
    pos_ac_series = np.array(
        [0.01, 0.012, 0.014, 0.011, 0.009, -0.005, -0.008, -0.006, 0.004, 0.007]
    )
    se_iid = sharpe_standard_error(pos_ac_series, max_autocorr_lags=0)
    se_lo = sharpe_standard_error(pos_ac_series, max_autocorr_lags=2)
    assert se_lo > se_iid

    ci_lo, ci_hi = sharpe_confidence_interval(x, confidence_level=0.95)
    sr_val = sharpe_ratio(x)
    assert isinstance(sr_val, float)
    assert ci_lo < sr_val < ci_hi
    assert (ci_hi - ci_lo) == pytest.approx(2.0 * 1.959963984540054 * actual_se, rel=1e-6)

    # Hand-computed drifted-weight turnover:
    # t=0: target [0.5, 0.5] from zero cash -> 0.5 * (0.5 + 0.5) = 0.5
    # During t=0, asset returns are [+1.0, 0.0] -> portfolio return = +0.5 -> W_end = 1.5
    # Drifted weights entering t=1: [0.5 * 2.0 / 1.5, 0.5 * 1.0 / 1.5] = [2/3, 1/3]
    # Rebalancing back to [0.5, 0.5] requires |0.5 - 2/3| + |0.5 - 1/3| = 1/3 -> 0.5 * 1/3 = 1/6
    w_tgt = np.array([[0.5, 0.5], [0.5, 0.5]])
    r_ast = np.array([[1.0, 0.0], [0.0, 0.0]])
    tau_drifted, mean_drifted = compute_turnover(w_tgt, r_ast)
    np.testing.assert_allclose(tau_drifted, np.array([0.5, 1.0 / 6.0]), rtol=1e-12)
    assert mean_drifted == pytest.approx((0.5 + 1.0 / 6.0) / 2.0, rel=1e-12)

    # Undrifted turnover when asset_simple_returns is None
    tau_undrifted, _ = compute_turnover(w_tgt)
    np.testing.assert_allclose(tau_undrifted, np.array([0.5, 0.0]), rtol=1e-12)

    # Summarize performance
    summary = summarize_performance(x)
    assert isinstance(summary, PerformanceSummary)
    assert summary.n_observations == len(x)
    assert summary.sharpe == pytest.approx(sr_val, rel=1e-12)
    assert summary.sharpe_se == pytest.approx(actual_se, rel=1e-12)


@settings(max_examples=50, deadline=None)
@given(
    rets=st.lists(
        st.floats(min_value=-0.50, max_value=0.50, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=40,
    )
)
def test_property_log_simple_roundtrip_time_additivity_and_drawdown_bounds(
    rets: list[float],
) -> None:
    """Property tests:
    1. log_to_simple_returns(simple_to_log_returns(R)) == R.
    2. Sum of log returns equals ln(1 + compound_returns(R)).
    3. Drawdowns lie in (-1.0, 0.0] and MDD lies in [0.0, 1.0).
    """
    arr = np.array(rets, dtype=np.float64)
    log_r = simple_to_log_returns(arr)
    roundtrip = log_to_simple_returns(log_r)
    np.testing.assert_allclose(roundtrip, arr, rtol=1e-11, atol=1e-12)

    comp = compound_returns(arr)
    assert isinstance(comp, float)
    assert float(np.sum(log_r)) == pytest.approx(math.log1p(comp), rel=1e-10, abs=1e-12)

    dd = drawdown_series(arr)
    assert np.all(dd <= 1e-15)
    assert np.all(dd > -1.0)
    mdd = max_drawdown(arr)
    assert isinstance(mdd, float)
    assert 0.0 <= mdd < 1.0


@settings(max_examples=40, deadline=None)
@given(
    n_rows=st.integers(min_value=5, max_value=30),
    scale=st.floats(min_value=0.05, max_value=10.0, allow_nan=False, allow_infinity=False),
    data=st.data(),
)
def test_property_portfolio_convex_hull_and_sharpe_scale_invariance(
    n_rows: int,
    scale: float,
    data: st.DataObject,
) -> None:
    """Property tests:
    1. Long-only portfolio return with weights summing to 1 lies in [min_i R_i, max_i R_i].
    2. Scaling returns by c > 0 leaves Sharpe and Sortino ratios (with rf=0, target=0) invariant.
    """
    n_assets = 4
    mat_vals = data.draw(
        st.lists(
            st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
            min_size=n_rows * n_assets,
            max_size=n_rows * n_assets,
        )
    )
    rets_2d = np.array(mat_vals, dtype=np.float64).reshape(n_rows, n_assets)

    raw_w = data.draw(
        st.lists(
            st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=n_assets,
            max_size=n_assets,
        )
    )
    w = np.array(raw_w, dtype=np.float64)
    w /= np.sum(w)

    p_ret = portfolio_returns(rets_2d, w)
    row_mins = np.min(rets_2d, axis=1)
    row_maxs = np.max(rets_2d, axis=1)
    assert np.all(p_ret >= row_mins - 1e-12)
    assert np.all(p_ret <= row_maxs + 1e-12)

    # Scale invariance of Sharpe & Sortino when std > 0 and downside > 0
    if float(np.std(p_ret, ddof=1)) > 1e-8 and float(np.min(p_ret)) < -1e-8:
        sr1 = sharpe_ratio(p_ret, risk_free_rate=0.0)
        sr2 = sharpe_ratio(p_ret * scale, risk_free_rate=0.0)
        assert isinstance(sr1, float) and isinstance(sr2, float)
        assert sr2 == pytest.approx(sr1, rel=1e-10, abs=1e-10)

        so1 = sortino_ratio(p_ret, target_return=0.0)
        so2 = sortino_ratio(p_ret * scale, target_return=0.0)
        assert isinstance(so1, float) and isinstance(so2, float)
        assert so2 == pytest.approx(so1, rel=1e-10, abs=1e-10)


def test_core_validation_and_error_edge_cases() -> None:
    """Verify strict error handling on invalid dimensions, NaNs, and impossible returns."""
    with pytest.raises(ValueError, match="1D or 2D"):
        simple_to_log_returns(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="non-empty"):
        simple_to_log_returns([])

    with pytest.raises(ValueError, match=r"strictly > -1\.0"):
        simple_to_log_returns([-1.0])

    with pytest.raises(ValueError, match="at least 2 observations"):
        prices_to_returns([100.0])

    with pytest.raises(ValueError, match="strictly positive"):
        prices_to_returns([100.0, 0.0])

    with pytest.raises(ValueError, match=r"cannot be < -1\.0"):
        compound_returns([-1.5])

    with pytest.raises(ValueError, match="initial_wealth must be positive"):
        wealth_index([0.01], initial_wealth=0.0)

    with pytest.raises(ValueError, match="window must be >= 1"):
        aggregate_multi_period_returns([0.01, 0.02], window=0)

    with pytest.raises(ValueError, match="must be >= window"):
        aggregate_multi_period_returns([0.01, 0.02], window=3)

    with pytest.raises(ValueError, match="must be 2D"):
        portfolio_returns([0.01, 0.02], [1.0])

    with pytest.raises(ValueError, match="does not match n_assets"):
        portfolio_returns(np.zeros((3, 2)), [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="does not match T="):
        excess_returns([0.01, 0.02], [0.0])

    with pytest.raises(ValueError, match=r"strictly > -1\.0"):
        annualized_return([-1.0, 0.1], geometric=True)

    with pytest.raises(ValueError, match="At least 3 non-NaN"):
        sharpe_standard_error([0.01, 0.02])

    with pytest.raises(ValueError, match="confidence_level must be in"):
        sharpe_confidence_interval([0.01, 0.02, 0.03], confidence_level=1.0)

    with pytest.raises(ValueError, match="dropped to <= 0"):
        compute_turnover(np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([[-1.0, 0.0], [0.0, 0.0]]))


def test_core_2d_and_remaining_branches() -> None:
    """Cover 2D return helpers, zero-variance metrics, initial_weights, and error branches."""
    rets_2d = np.array([[0.10, 0.20], [0.10, -0.10], [0.00, 0.05], [0.05, 0.05]])

    # 2D compound_returns, wealth_index(include_initial=True), aggregate_multi_period_returns
    comp_2d = compound_returns(rets_2d)
    assert isinstance(comp_2d, np.ndarray) and comp_2d.shape == (2,)
    np.testing.assert_allclose(
        comp_2d, np.array([1.1 * 1.1 * 1.05 - 1.0, 1.2 * 0.9 * 1.05 * 1.05 - 1.0]), rtol=1e-12
    )

    w_2d = wealth_index(rets_2d, initial_wealth=10.0, include_initial=True)
    assert w_2d.shape == (5, 2)
    np.testing.assert_allclose(w_2d[0], np.array([10.0, 10.0]))

    agg_2d = aggregate_multi_period_returns(rets_2d, window=2)
    assert agg_2d.shape == (2, 2)
    np.testing.assert_allclose(agg_2d[0], np.array([0.21, 0.08]), rtol=1e-12)

    # 2D excess_returns + error branches
    rf_2d = np.full_like(rets_2d, 0.001)
    np.testing.assert_allclose(
        excess_returns(rets_2d, rf_2d, rf_is_annualized=False), rets_2d - 0.001, rtol=1e-12
    )
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        excess_returns(rets_2d, 0.0, periods_per_year=0)
    with pytest.raises(ValueError, match="2D risk_free_rate shape"):
        excess_returns(rets_2d, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="scalar, 1D, or 2D"):
        excess_returns(rets_2d, np.zeros((1, 1, 1)))

    # returns.py remaining validation branches
    with pytest.raises(ValueError, match=r"cannot be < -1\.0"):
        wealth_index([-1.5, 0.1])
    with pytest.raises(ValueError, match=r"cannot be < -1\.0"):
        aggregate_multi_period_returns([-1.5, 0.1], window=2)
    with pytest.raises(ValueError, match="weights must not contain NaN"):
        portfolio_returns(rets_2d, [0.5, float("nan")])
    with pytest.raises(ValueError, match="2D weights shape"):
        portfolio_returns(rets_2d, np.ones((2, 2)))

    # metrics.py error branches
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        annualized_return([0.01, 0.02], periods_per_year=0)
    with pytest.raises(ValueError, match="at least 1 non-NaN"):
        annualized_return([float("nan"), float("nan")])
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        annualized_volatility([0.01, 0.02], periods_per_year=0)
    with pytest.raises(ValueError, match="ddof must be non-negative"):
        annualized_volatility([0.01, 0.02], ddof=-1)
    with pytest.raises(ValueError, match="strictly more than ddof=1"):
        annualized_volatility([0.01, float("nan")], ddof=1)
    with pytest.raises(ValueError, match="At least 2 non-NaN"):
        sharpe_ratio([0.01, float("nan")])
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        sharpe_standard_error([0.01, 0.02, 0.03], periods_per_year=0)
    with pytest.raises(ValueError, match="max_autocorr_lags must be non-negative"):
        sharpe_standard_error([0.01, 0.02, 0.03], max_autocorr_lags=-1)
    with pytest.raises(ValueError, match="requires a 1D return series"):
        sharpe_standard_error(rets_2d)
    with pytest.raises(ValueError, match="requires a 1D return series"):
        sharpe_confidence_interval(rets_2d)
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        sortino_ratio([0.01, 0.02], periods_per_year=0)
    with pytest.raises(ValueError, match="at least 1 non-NaN"):
        sortino_ratio([float("nan")])
    with pytest.raises(ValueError, match="requires a 1D return series"):
        summarize_performance(rets_2d)
    with pytest.raises(ValueError, match="At least 3 non-NaN"):
        summarize_performance([0.01, 0.02])

    # Zero-variance & zero-downside-deviation metrics return NaN cleanly
    flat = np.zeros(5)
    assert math.isnan(sharpe_standard_error(flat))
    ci_lo, ci_hi = sharpe_confidence_interval(flat)
    assert math.isnan(ci_lo) and math.isnan(ci_hi)
    all_pos = np.array([0.01, 0.02, 0.03, 0.01])
    assert math.isnan(float(sortino_ratio(all_pos, target_return=0.0)))
    flat_summary = summarize_performance(flat)
    assert math.isnan(flat_summary.sharpe)
    assert math.isnan(flat_summary.skewness)
    assert math.isnan(flat_summary.kurtosis)

    # compute_turnover with explicit initial_weights and error branches
    w_tgt = np.array([[0.6, 0.4], [0.5, 0.5]])
    tau_init, mean_init = compute_turnover(w_tgt, initial_weights=np.array([0.5, 0.5]))
    np.testing.assert_allclose(tau_init, np.array([0.1, 0.1]), rtol=1e-12)
    assert mean_init == pytest.approx(0.1, rel=1e-12)

    with pytest.raises(ValueError, match="must be 2D"):
        compute_turnover([0.5, 0.5])
    with pytest.raises(ValueError, match="target_weights must not contain NaN"):
        compute_turnover(np.array([[0.5, float("nan")]]))
    with pytest.raises(ValueError, match="initial_weights shape"):
        compute_turnover(w_tgt, initial_weights=np.array([1.0]))
    with pytest.raises(ValueError, match="initial_weights must not contain NaN"):
        compute_turnover(w_tgt, initial_weights=np.array([0.5, float("nan")]))
    with pytest.raises(ValueError, match="asset_simple_returns shape"):
        compute_turnover(w_tgt, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_turnover(w_tgt, np.array([[0.0, float("nan")], [0.0, 0.0]]))
