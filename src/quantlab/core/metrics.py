"""Performance, risk-adjusted return, Sharpe standard error, drawdown, and turnover metrics.

Implements `QUANT_PROJECT_DESIGN.md` §1.4 and `docs/conventions.md` §4:
- Annualized return (CAGR and arithmetic mean) and volatility (`ddof=1`, $A = 252$).
- Sharpe ratio and Mertens (2002) / Bailey & López de Prado (2014) non-normal Sharpe
  standard error with optional Lo (2002) Bartlett-kernel autocorrelation adjustment.
- Sortino ratio, drawdown series anchored at $W_0 = 1.0$, maximum drawdown, Calmar ratio,
  and one-way drifted-weight turnover.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.stats

from quantlab.core.returns import (
    TRADING_DAYS_PER_YEAR,
    _as_float64_array,
    compound_returns,
    excess_returns,
    wealth_index,
)


@dataclass(frozen=True)
class PerformanceSummary:
    """Comprehensive performance summary for a single 1D return stream."""

    n_observations: int
    total_return: float
    cagr: float
    annualized_mean: float
    annualized_vol: float
    sharpe: float
    sharpe_se: float
    sharpe_ci_95: tuple[float, float]
    sortino: float
    max_drawdown: float
    calmar: float
    skewness: float
    kurtosis: float


def annualized_return(
    simple_returns: npt.ArrayLike,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    geometric: bool = True,
) -> npt.NDArray[np.float64] | float:
    """Compute annualized CAGR (`geometric=True`) or arithmetic mean (`geometric=False`)."""
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year}."
        raise ValueError(msg)

    arr = _as_float64_array(simple_returns, name="simple_returns")
    valid_counts = np.sum(~np.isnan(arr), axis=0)
    if np.any(valid_counts == 0):
        msg = "Each return series must contain at least 1 non-NaN observation."
        raise ValueError(msg)

    if geometric:
        if np.any(np.isfinite(arr) & (arr <= -1.0)):
            msg = "Geometric annualized return requires all returns strictly > -1.0."
            raise ValueError(msg)
        clean = np.where(np.isnan(arr), 0.0, arr)
        growth = np.prod(1.0 + clean, axis=0)
        ann = np.power(growth, float(periods_per_year) / valid_counts) - 1.0
    else:
        ann = float(periods_per_year) * np.nanmean(arr, axis=0)

    if arr.ndim == 1:
        return float(ann)
    return np.asarray(ann, dtype=np.float64)


def annualized_volatility(
    simple_returns: npt.ArrayLike,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    ddof: int = 1,
) -> npt.NDArray[np.float64] | float:
    """Compute annualized sample volatility $\\sqrt{A} \\cdot s_R$."""
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year}."
        raise ValueError(msg)
    if ddof < 0:
        msg = f"ddof must be non-negative, got {ddof}."
        raise ValueError(msg)

    arr = _as_float64_array(simple_returns, name="simple_returns")
    valid_counts = np.sum(~np.isnan(arr), axis=0)
    if np.any(valid_counts <= ddof):
        msg = f"Each series requires strictly more than ddof={ddof} non-NaN observations."
        raise ValueError(msg)

    vol = math.sqrt(float(periods_per_year)) * np.nanstd(arr, ddof=ddof, axis=0)
    if arr.ndim == 1:
        return float(vol)
    return np.asarray(vol, dtype=np.float64)


def sharpe_ratio(
    simple_returns: npt.ArrayLike,
    *,
    risk_free_rate: float | npt.ArrayLike = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf_is_annualized: bool = True,
) -> npt.NDArray[np.float64] | float:
    """Compute annualized Sharpe ratio $\\sqrt{A} \\cdot \\bar{R}^e / s_{R^e}$ (`ddof=1`).

    Returns `NaN` for any series with zero excess return standard deviation.
    """
    ex = excess_returns(
        simple_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
    )
    valid_counts = np.sum(~np.isnan(ex), axis=0)
    if np.any(valid_counts < 2):
        msg = "At least 2 non-NaN observations per series are required for Sharpe ratio."
        raise ValueError(msg)

    mean_ex = np.nanmean(ex, axis=0)
    std_ex = np.nanstd(ex, ddof=1, axis=0)
    scale = math.sqrt(float(periods_per_year))

    if ex.ndim == 1:
        std_val = float(std_ex)
        return float(scale * float(mean_ex) / std_val) if std_val > 0.0 else float("nan")

    res = np.full(ex.shape[1], np.nan, dtype=np.float64)
    pos = std_ex > 0.0
    res[pos] = scale * mean_ex[pos] / std_ex[pos]
    return res


def sharpe_standard_error(
    simple_returns: npt.ArrayLike,
    *,
    risk_free_rate: float | npt.ArrayLike = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf_is_annualized: bool = True,
    max_autocorr_lags: int = 0,
) -> float:
    """Compute the annualized standard error of the Sharpe ratio for a 1D return stream.

    Uses the Mertens (2002) / Bailey & López de Prado (2014) non-normal variance formula:
    $$\\widehat{\\text{Var}}(\\widehat{\\text{SR}}_{\\text{sub}}) =
    \\frac{1 - \\hat{\\gamma}_3 \\widehat{\\text{SR}}_{\\text{sub}} +
    \\frac{\\hat{\\gamma}_4 - 1}{4} \\widehat{\\text{SR}}_{\\text{sub}}^2}{T - 1}$$
    scaled by `periods_per_year` and, when `max_autocorr_lags > 0`, multiplied by
    Lo's (2002) Bartlett-kernel autocorrelation factor
    $\\eta(q) = 1 + 2 \\sum_{k=1}^{q} (1 - \\frac{k}{q+1}) \\hat{\\rho}_k$.
    """
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year}."
        raise ValueError(msg)
    if max_autocorr_lags < 0:
        msg = f"max_autocorr_lags must be non-negative, got {max_autocorr_lags}."
        raise ValueError(msg)

    ex = excess_returns(
        simple_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
    )
    if ex.ndim != 1:
        msg = f"sharpe_standard_error requires a 1D return series, got shape {ex.shape}."
        raise ValueError(msg)

    x = ex[~np.isnan(ex)]
    n_obs = int(x.shape[0])
    if n_obs < 3:
        msg = f"At least 3 non-NaN observations are required, got {n_obs}."
        raise ValueError(msg)

    std_x = float(np.std(x, ddof=1))
    if std_x == 0.0:
        return float("nan")

    sr_sub = float(np.mean(x)) / std_x
    skew_val = float(scipy.stats.skew(x, bias=False))
    kurt_val = float(scipy.stats.kurtosis(x, fisher=False, bias=False))

    var_sub = (1.0 - skew_val * sr_sub + ((kurt_val - 1.0) / 4.0) * (sr_sub**2)) / (n_obs - 1)
    var_sub = max(var_sub, 0.0)

    eta = 1.0
    if max_autocorr_lags > 0:
        q = min(max_autocorr_lags, n_obs - 2)
        x_c = x - float(np.mean(x))
        denom = float(np.dot(x_c, x_c))
        if denom > 0.0 and q >= 1:
            for k in range(1, q + 1):
                rho_k = float(np.dot(x_c[:-k], x_c[k:])) / denom
                eta += (2.0 * (q + 1 - k) / (q + 1)) * rho_k

    return float(math.sqrt(float(periods_per_year) * var_sub * max(eta, 0.0)))


def sharpe_confidence_interval(
    simple_returns: npt.ArrayLike,
    *,
    confidence_level: float = 0.95,
    risk_free_rate: float | npt.ArrayLike = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf_is_annualized: bool = True,
    max_autocorr_lags: int = 0,
) -> tuple[float, float]:
    """Compute a two-sided asymptotic normal confidence interval for the annualized Sharpe."""
    if not 0.0 < confidence_level < 1.0:
        msg = f"confidence_level must be in (0, 1), got {confidence_level}."
        raise ValueError(msg)

    sr = sharpe_ratio(
        simple_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
    )
    if not isinstance(sr, float):
        msg = "sharpe_confidence_interval requires a 1D return series."
        raise ValueError(msg)

    se = sharpe_standard_error(
        simple_returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
        max_autocorr_lags=max_autocorr_lags,
    )
    if math.isnan(sr) or math.isnan(se):
        return (float("nan"), float("nan"))

    z = float(scipy.stats.norm.ppf(0.5 + 0.5 * confidence_level))
    return (sr - z * se, sr + z * se)


def sortino_ratio(
    simple_returns: npt.ArrayLike,
    *,
    target_return: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target_is_annualized: bool = True,
) -> npt.NDArray[np.float64] | float:
    """Compute annualized Sortino ratio using downside deviation ($d^-_t = \\min(0, d_t)$)."""
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year}."
        raise ValueError(msg)

    arr = _as_float64_array(simple_returns, name="simple_returns")
    valid_counts = np.sum(~np.isnan(arr), axis=0)
    if np.any(valid_counts < 1):
        msg = "Each return series must contain at least 1 non-NaN observation."
        raise ValueError(msg)

    tau = target_return / float(periods_per_year) if target_is_annualized else target_return
    diff = arr - tau
    downside = np.minimum(0.0, diff)
    down_dev = np.sqrt(float(periods_per_year) * np.nanmean(downside**2, axis=0))
    ann_excess = float(periods_per_year) * np.nanmean(diff, axis=0)

    if arr.ndim == 1:
        dd_val = float(down_dev)
        return float(float(ann_excess) / dd_val) if dd_val > 0.0 else float("nan")

    res = np.full(arr.shape[1], np.nan, dtype=np.float64)
    pos = down_dev > 0.0
    res[pos] = ann_excess[pos] / down_dev[pos]
    return res


def drawdown_series(simple_returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compute underwater drawdown series anchored at $W_0 = 1.0$ so bar-0 losses are captured."""
    w = wealth_index(simple_returns, initial_wealth=1.0, include_initial=True)
    running_peak = np.maximum.accumulate(w, axis=0)
    dd = np.asarray(w / running_peak - 1.0, dtype=np.float64)
    return dd[1:]


def max_drawdown(simple_returns: npt.ArrayLike) -> npt.NDArray[np.float64] | float:
    """Compute maximum drawdown as a non-negative fraction in $[0, 1]$."""
    dd = drawdown_series(simple_returns)
    if dd.ndim == 1:
        return float(np.max(-dd))
    return np.asarray(np.max(-dd, axis=0), dtype=np.float64)


def calmar_ratio(
    simple_returns: npt.ArrayLike,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> npt.NDArray[np.float64] | float:
    """Compute Calmar ratio (`CAGR / max_drawdown`), returning `NaN` when `max_drawdown == 0`."""
    cagr = annualized_return(simple_returns, periods_per_year=periods_per_year, geometric=True)
    mdd = max_drawdown(simple_returns)

    if isinstance(cagr, float) and isinstance(mdd, float):
        return float(cagr / mdd) if mdd > 0.0 else float("nan")

    cagr_arr = np.asarray(cagr, dtype=np.float64)
    mdd_arr = np.asarray(mdd, dtype=np.float64)
    res = np.full(cagr_arr.shape, np.nan, dtype=np.float64)
    pos = mdd_arr > 0.0
    res[pos] = cagr_arr[pos] / mdd_arr[pos]
    return res


def compute_turnover(
    target_weights: npt.ArrayLike,
    asset_simple_returns: npt.ArrayLike | None = None,
    *,
    initial_weights: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], float]:
    """Compute one-way portfolio turnover $\\tau_t = \\frac{1}{2} \\sum_i |w_{i,t} - w_{i,t-1}^+|$.

    Parameters
    ----------
    target_weights : npt.ArrayLike
        2D `(T, N)` array of target portfolio weights at each rebalance `t`.
    asset_simple_returns : npt.ArrayLike | None, optional
        2D `(T, N)` array of asset returns realized under `target_weights[t - 1]` prior
        to rebalancing to `target_weights[t]`. When provided, pre-trade drifted weights
        $w_{i,t-1}^+$ are computed before taking the difference with `target_weights[t]`.
    initial_weights : npt.ArrayLike | None, optional
        1D `(N,)` array of weights held before `t = 0`. Defaults to zeros (all cash).

    Returns
    -------
    tuple[npt.NDArray[np.float64], float]
        `(turnover_series, mean_turnover)`.
    """
    w = _as_float64_array(target_weights, name="target_weights")
    if w.ndim != 2:
        msg = f"target_weights must be 2D (T, N), got shape {w.shape}."
        raise ValueError(msg)
    if np.any(np.isnan(w)):
        msg = "target_weights must not contain NaN."
        raise ValueError(msg)

    t_len, n_assets = w.shape
    tau = np.empty(t_len, dtype=np.float64)

    if initial_weights is None:
        w_init = np.zeros(n_assets, dtype=np.float64)
    else:
        w_init = _as_float64_array(initial_weights, name="initial_weights")
        if w_init.shape != (n_assets,):
            msg = f"initial_weights shape {w_init.shape} does not match n_assets={n_assets}."
            raise ValueError(msg)
        if np.any(np.isnan(w_init)):
            msg = "initial_weights must not contain NaN."
            raise ValueError(msg)

    tau[0] = 0.5 * float(np.sum(np.abs(w[0] - w_init)))

    if t_len > 1:
        if asset_simple_returns is None:
            tau[1:] = 0.5 * np.sum(np.abs(w[1:] - w[:-1]), axis=1)
        else:
            r = _as_float64_array(asset_simple_returns, name="asset_simple_returns")
            if r.shape != (t_len, n_assets):
                msg = f"asset_simple_returns shape {r.shape} does not match {w.shape}."
                raise ValueError(msg)
            if np.any(np.isnan(r[:-1])):
                msg = "asset_simple_returns[:-1] must not contain NaN for drifted weights."
                raise ValueError(msg)

            w_prev = w[:-1]
            r_prev = r[:-1]
            port_ret = np.sum(w_prev * r_prev, axis=1, keepdims=True)
            denom = 1.0 + port_ret
            if np.any(denom <= 0.0):
                msg = "Portfolio value dropped to <= 0 during pre-trade drift calculation."
                raise ValueError(msg)
            w_drifted = w_prev * (1.0 + r_prev) / denom
            tau[1:] = 0.5 * np.sum(np.abs(w[1:] - w_drifted), axis=1)

    return tau, float(np.mean(tau))


def summarize_performance(
    simple_returns: npt.ArrayLike,
    *,
    risk_free_rate: float | npt.ArrayLike = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf_is_annualized: bool = True,
    max_autocorr_lags: int = 0,
) -> PerformanceSummary:
    """Compute a complete `PerformanceSummary` for a 1D return stream."""
    arr = _as_float64_array(simple_returns, name="simple_returns")
    if arr.ndim != 1:
        msg = f"summarize_performance requires a 1D return series, got shape {arr.shape}."
        raise ValueError(msg)

    clean = arr[~np.isnan(arr)]
    n_obs = int(clean.shape[0])
    if n_obs < 3:
        msg = f"At least 3 non-NaN observations are required for summary, got {n_obs}."
        raise ValueError(msg)

    tot_ret = compound_returns(arr)
    cagr = annualized_return(arr, periods_per_year=periods_per_year, geometric=True)
    ann_mean = annualized_return(arr, periods_per_year=periods_per_year, geometric=False)
    ann_vol = annualized_volatility(arr, periods_per_year=periods_per_year)
    sr = sharpe_ratio(
        arr,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
    )
    sr_se = sharpe_standard_error(
        arr,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
        max_autocorr_lags=max_autocorr_lags,
    )
    sr_ci = sharpe_confidence_interval(
        arr,
        confidence_level=0.95,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        rf_is_annualized=rf_is_annualized,
        max_autocorr_lags=max_autocorr_lags,
    )
    sortino = sortino_ratio(arr, periods_per_year=periods_per_year)
    mdd = max_drawdown(arr)
    calmar = calmar_ratio(arr, periods_per_year=periods_per_year)

    std_clean = float(np.std(clean, ddof=1))
    if std_clean > 0.0:
        skewness = float(scipy.stats.skew(clean, bias=False))
        kurtosis = float(scipy.stats.kurtosis(clean, fisher=False, bias=False))
    else:
        skewness = float("nan")
        kurtosis = float("nan")

    assert isinstance(tot_ret, float)
    assert isinstance(cagr, float)
    assert isinstance(ann_mean, float)
    assert isinstance(ann_vol, float)
    assert isinstance(sr, float)
    assert isinstance(sortino, float)
    assert isinstance(mdd, float)
    assert isinstance(calmar, float)

    return PerformanceSummary(
        n_observations=n_obs,
        total_return=tot_ret,
        cagr=cagr,
        annualized_mean=ann_mean,
        annualized_vol=ann_vol,
        sharpe=sr,
        sharpe_se=sr_se,
        sharpe_ci_95=sr_ci,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        skewness=skewness,
        kurtosis=kurtosis,
    )
