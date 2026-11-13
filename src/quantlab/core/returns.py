"""Return conversions, compounding, wealth indices, and portfolio return aggregation.

Follows the mathematical conventions in `docs/conventions.md` §2, §4, and §5:
- All returns are decimal fractions (`0.01` means 1%).
- Arrays have shape `(T,)` or `(T, N)` with `dtype=np.float64`.
- Log returns aggregate additively across time (`axis=0`).
- Simple returns aggregate linearly across assets (`axis=1`) via portfolio weights.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

from quantlab.compute import check_array_memory

TRADING_DAYS_PER_YEAR: Final[int] = 252


def _as_float64_array(arr: npt.ArrayLike, *, name: str = "array") -> npt.NDArray[np.float64]:
    """Convert `arr` to a non-empty 1D or 2D `np.float64` array within M2 memory limits."""
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim not in (1, 2):
        msg = f"{name} must be 1D or 2D, got {out.ndim}D with shape {out.shape}."
        raise ValueError(msg)
    if out.size == 0:
        msg = f"{name} must be non-empty, got shape {out.shape}."
        raise ValueError(msg)
    check_array_memory(out.shape, dtype=np.float64)
    return out


def simple_to_log_returns(simple_returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert simple returns $R_t$ to log returns $r_t = \\ln(1 + R_t)$."""
    arr = _as_float64_array(simple_returns, name="simple_returns")
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask & (arr <= -1.0)):
        msg = "Simple returns must be strictly > -1.0 for log conversion."
        raise ValueError(msg)
    return np.asarray(np.log1p(arr), dtype=np.float64)


def log_to_simple_returns(log_returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert log returns $r_t$ to simple returns $R_t = \\exp(r_t) - 1$."""
    arr = _as_float64_array(log_returns, name="log_returns")
    return np.asarray(np.expm1(arr), dtype=np.float64)


def prices_to_returns(
    prices: npt.ArrayLike,
    *,
    log: bool = False,
    prepend_nan: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute period-over-period simple or log returns from a 1D `(T,)` or 2D `(T, N)` price array.

    Parameters
    ----------
    prices : npt.ArrayLike
        Strictly positive asset prices along chronological `axis=0`.
    log : bool, default False
        If True, return continuously compounded log returns; otherwise simple returns.
    prepend_nan : bool, default False
        If True, prepend a leading `NaN` along `axis=0` so output shape matches `prices`.
    """
    arr = _as_float64_array(prices, name="prices")
    if arr.shape[0] < 2:
        msg = f"prices must have at least 2 observations along axis=0, got shape {arr.shape}."
        raise ValueError(msg)

    finite_mask = np.isfinite(arr)
    if np.any(finite_mask & (arr <= 0.0)):
        msg = "All finite prices must be strictly positive (> 0.0)."
        raise ValueError(msg)

    simple = np.asarray(arr[1:] / arr[:-1] - 1.0, dtype=np.float64)
    rets = simple_to_log_returns(simple) if log else simple

    if not prepend_nan:
        return rets

    if rets.ndim == 1:
        return np.concatenate([np.array([np.nan], dtype=np.float64), rets])
    nan_row = np.full((1, rets.shape[1]), np.nan, dtype=np.float64)
    return np.vstack([nan_row, rets])


def compound_returns(simple_returns: npt.ArrayLike) -> npt.NDArray[np.float64] | float:
    """Compound simple returns across `axis=0`: $\\prod_t (1 + R_t) - 1$.

    `NaN` entries are treated as zero return (`0.0`) during compounding.
    """
    arr = _as_float64_array(simple_returns, name="simple_returns")
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask & (arr < -1.0)):
        msg = "Simple returns cannot be < -1.0."
        raise ValueError(msg)

    clean = np.where(np.isnan(arr), 0.0, arr)
    prod = np.prod(1.0 + clean, axis=0) - 1.0
    if arr.ndim == 1:
        return float(prod)
    return np.asarray(prod, dtype=np.float64)


def wealth_index(
    simple_returns: npt.ArrayLike,
    *,
    initial_wealth: float = 1.0,
    include_initial: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute cumulative wealth path starting from `initial_wealth`.

    `NaN` entries are treated as zero return (`0.0`). When `include_initial=True`,
    prepends `initial_wealth` at index `0` along `axis=0`.
    """
    if initial_wealth <= 0.0:
        msg = f"initial_wealth must be positive, got {initial_wealth}."
        raise ValueError(msg)

    arr = _as_float64_array(simple_returns, name="simple_returns")
    finite_mask = np.isfinite(arr)
    if np.any(finite_mask & (arr < -1.0)):
        msg = "Simple returns cannot be < -1.0."
        raise ValueError(msg)

    clean = np.where(np.isnan(arr), 0.0, arr)
    growth = np.asarray(np.cumprod(1.0 + clean, axis=0) * initial_wealth, dtype=np.float64)

    if not include_initial:
        return growth

    if arr.ndim == 1:
        return np.concatenate([np.array([initial_wealth], dtype=np.float64), growth])
    init_row = np.full((1, arr.shape[1]), initial_wealth, dtype=np.float64)
    return np.vstack([init_row, growth])


def aggregate_multi_period_returns(
    simple_returns: npt.ArrayLike,
    window: int,
) -> npt.NDArray[np.float64]:
    """Compound simple returns into non-overlapping blocks of length `window` along `axis=0`.

    Trailing observations beyond `T // window * window` are truncated.
    """
    if window < 1:
        msg = f"window must be >= 1, got {window}."
        raise ValueError(msg)

    arr = _as_float64_array(simple_returns, name="simple_returns")
    if arr.shape[0] < window:
        msg = f"Input length ({arr.shape[0]}) must be >= window ({window})."
        raise ValueError(msg)

    finite_mask = np.isfinite(arr)
    if np.any(finite_mask & (arr < -1.0)):
        msg = "Simple returns cannot be < -1.0."
        raise ValueError(msg)

    n_blocks = arr.shape[0] // window
    trimmed = arr[: n_blocks * window]
    clean = np.where(np.isnan(trimmed), 0.0, trimmed)

    if arr.ndim == 1:
        blocks_2d = clean.reshape(n_blocks, window)
        return np.asarray(np.prod(1.0 + blocks_2d, axis=1) - 1.0, dtype=np.float64)
    blocks_3d = clean.reshape(n_blocks, window, arr.shape[1])
    return np.asarray(np.prod(1.0 + blocks_3d, axis=1) - 1.0, dtype=np.float64)


def portfolio_returns(
    asset_simple_returns: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Compute portfolio simple returns $R_{p,t} = \\sum_i w_{i,t} R_{i,t}$.

    Parameters
    ----------
    asset_simple_returns : npt.ArrayLike
        2D array of shape `(T, N)` of asset simple returns.
    weights : npt.ArrayLike
        Either 1D `(N,)` static pre-period weights or 2D `(T, N)` where row `t`
        holds the causal pre-period weights active during period `t`.
    """
    ret_2d = _as_float64_array(asset_simple_returns, name="asset_simple_returns")
    if ret_2d.ndim != 2:
        msg = f"asset_simple_returns must be 2D (T, N), got shape {ret_2d.shape}."
        raise ValueError(msg)

    w = _as_float64_array(weights, name="weights")
    if np.any(np.isnan(w)):
        msg = "weights must not contain NaN."
        raise ValueError(msg)

    t_len, n_assets = ret_2d.shape
    if w.ndim == 1:
        if w.shape != (n_assets,):
            msg = f"1D weights shape {w.shape} does not match n_assets={n_assets}."
            raise ValueError(msg)
        w_2d = w[np.newaxis, :]
    else:
        if w.shape != (t_len, n_assets):
            msg = f"2D weights shape {w.shape} does not match returns shape {ret_2d.shape}."
            raise ValueError(msg)
        w_2d = w

    return np.asarray(np.sum(ret_2d * w_2d, axis=1), dtype=np.float64)


def excess_returns(
    simple_returns: npt.ArrayLike,
    risk_free_rate: float | npt.ArrayLike = 0.0,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf_is_annualized: bool = True,
) -> npt.NDArray[np.float64]:
    """Subtract a scalar or time-varying risk-free rate from simple returns."""
    if periods_per_year <= 0:
        msg = f"periods_per_year must be positive, got {periods_per_year}."
        raise ValueError(msg)

    arr = _as_float64_array(simple_returns, name="simple_returns")
    rf_arr = np.asarray(risk_free_rate, dtype=np.float64)
    rf_period = rf_arr / float(periods_per_year) if rf_is_annualized else rf_arr

    if rf_period.ndim == 0:
        return np.asarray(arr - float(rf_period), dtype=np.float64)

    if rf_period.ndim == 1:
        if rf_period.shape != (arr.shape[0],):
            msg = f"1D risk_free_rate length {rf_period.shape[0]} does not match T={arr.shape[0]}."
            raise ValueError(msg)
        if arr.ndim == 2:
            return np.asarray(arr - rf_period[:, np.newaxis], dtype=np.float64)
        return np.asarray(arr - rf_period, dtype=np.float64)

    if rf_period.ndim == 2:
        if rf_period.shape != arr.shape:
            msg = f"2D risk_free_rate shape {rf_period.shape} does not match {arr.shape}."
            raise ValueError(msg)
        return np.asarray(arr - rf_period, dtype=np.float64)

    msg = f"risk_free_rate must be scalar, 1D, or 2D, got {rf_period.ndim}D."
    raise ValueError(msg)
