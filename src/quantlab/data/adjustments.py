"""Corporate action adjustments: splits, cash dividends, total returns, and forward TRI.

Implements the mathematical definitions in `docs/conventions.md` §2:
- Raw OHLCV prices remain immutable.
- On ex-date `t` with split factor `S_t` (new shares per old share) and cash dividend
  `D_t` (USD per pre-split share):
      R_t^{tot} = (P_t^{raw} * S_t + D_t) / P_{t-1}^{raw} - 1
- Forward Total Return Index (`tri`) compounds forward from `initial_value` (100.0) so
  future dividends or splits never retroactively mutate historical index levels.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl

from quantlab.data.schemas import ACTIONS_SCHEMA, BARS_SCHEMA, ensure_schema


def _prepare_actions(
    actions: pl.DataFrame | None,
    as_of_ts: datetime | None = None,
) -> pl.DataFrame:
    """Validate, optionally filter by `as_of_ts`, and aggregate corporate actions by date."""
    if actions is None or actions.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String(),
                "ex_date": pl.Datetime("us", "UTC"),
                "split_factor": pl.Float64(),
                "div_cash": pl.Float64(),
            }
        )

    ensure_schema(actions, ACTIONS_SCHEMA, name="actions")
    filtered = actions
    if as_of_ts is not None:
        filtered = filtered.filter(pl.col("available_ts") <= as_of_ts)

    if filtered.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String(),
                "ex_date": pl.Datetime("us", "UTC"),
                "split_factor": pl.Float64(),
                "div_cash": pl.Float64(),
            }
        )

    return (
        filtered.group_by(["ticker", "ex_date"])
        .agg(
            pl.col("split_factor").product().alias("split_factor"),
            pl.col("div_cash").sum().alias("div_cash"),
        )
        .sort(["ticker", "ex_date"])
    )


def compute_total_returns(
    bars: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    as_of_ts: datetime | None = None,
) -> pl.DataFrame:
    """Compute raw price returns and corporate-action-adjusted total returns per ticker.

    Parameters
    ----------
    bars : pl.DataFrame
        Daily OHLCV bars conforming to `BARS_SCHEMA`.
    actions : pl.DataFrame | None, optional
        Corporate actions conforming to `ACTIONS_SCHEMA`.
    as_of_ts : datetime | None, optional
        If provided, filters both `bars` and `actions` to records with
        `available_ts <= as_of_ts` before computing returns.

    Returns
    -------
    pl.DataFrame
        Sorted DataFrame with additional columns:
        `split_factor`, `div_cash`, `price_return`, and `total_return` (`float64`,
        with `NaN` for the first observation of each ticker).
    """
    ensure_schema(bars, BARS_SCHEMA, name="bars")

    working_bars = bars if as_of_ts is None else bars.filter(pl.col("available_ts") <= as_of_ts)
    if working_bars.is_empty():
        return working_bars.with_columns(
            pl.lit(1.0, dtype=pl.Float64).alias("split_factor"),
            pl.lit(0.0, dtype=pl.Float64).alias("div_cash"),
            pl.lit(np.nan, dtype=pl.Float64).alias("price_return"),
            pl.lit(np.nan, dtype=pl.Float64).alias("total_return"),
        )

    agg_actions = _prepare_actions(actions, as_of_ts=as_of_ts)

    joined = (
        working_bars.sort(["ticker", "period_ts"])
        .join(
            agg_actions,
            left_on=["ticker", "period_ts"],
            right_on=["ticker", "ex_date"],
            how="left",
        )
        .with_columns(
            pl.col("split_factor").fill_null(1.0),
            pl.col("div_cash").fill_null(0.0),
        )
    )

    prev_close = pl.col("close").shift(1).over("ticker")
    price_ret = (pl.col("close") / prev_close - 1.0).fill_null(np.nan)
    total_ret = (
        (pl.col("close") * pl.col("split_factor") + pl.col("div_cash")) / prev_close - 1.0
    ).fill_null(np.nan)

    return joined.with_columns(
        price_ret.alias("price_return"),
        total_ret.alias("total_return"),
    )


def compute_total_return_index(
    bars: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    initial_value: float = 100.0,
    as_of_ts: datetime | None = None,
) -> pl.DataFrame:
    """Compute a forward-compounded Total Return Index (`tri`) per ticker.

    Unlike vendor backward-adjusted `Adj Close`, a forward-compounded Total Return
    Index starts at `initial_value` on each ticker's first bar and compounds
    forward via `tri[t] = tri[t-1] * (1 + total_return[t])`. Future splits and
    dividends never alter historical `tri` values.

    Parameters
    ----------
    bars : pl.DataFrame
        Daily OHLCV bars conforming to `BARS_SCHEMA`.
    actions : pl.DataFrame | None, optional
        Corporate actions conforming to `ACTIONS_SCHEMA`.
    initial_value : float, default 100.0
        Starting index level on each ticker's first observation (must be positive).
    as_of_ts : datetime | None, optional
        Point-in-time cutoff timestamp.

    Returns
    -------
    pl.DataFrame
        DataFrame from `compute_total_returns` with an added `tri` (`Float64`) column.
    """
    if initial_value <= 0.0:
        msg = f"initial_value must be positive, got {initial_value}."
        raise ValueError(msg)

    with_returns = compute_total_returns(bars, actions=actions, as_of_ts=as_of_ts)
    if with_returns.is_empty():
        return with_returns.with_columns(pl.lit(np.nan, dtype=pl.Float64).alias("tri"))

    growth_factor = (
        pl.when(pl.col("total_return").is_nan() | pl.col("total_return").is_null())
        .then(1.0)
        .otherwise(1.0 + pl.col("total_return"))
    )
    return with_returns.with_columns(
        (growth_factor.cum_prod().over("ticker") * initial_value).alias("tri")
    )


def apply_split_adjustments(
    bars: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    as_of_ts: datetime | None = None,
) -> pl.DataFrame:
    """Compute point-in-time split-adjusted OHLCV columns using splits known at `as_of_ts`.

    For each bar at `t`, let `cum_future_split[t]` be the product of all `split_factor`
    events for that ticker occurring strictly after `t` (with `available_ts <= as_of_ts`).
    Then:
    - `price_split_adj[t] = price_raw[t] / cum_future_split[t]`
    - `volume_split_adj[t] = volume_raw[t] * cum_future_split[t]`

    Parameters
    ----------
    bars : pl.DataFrame
        Daily OHLCV bars conforming to `BARS_SCHEMA`.
    actions : pl.DataFrame | None, optional
        Corporate actions conforming to `ACTIONS_SCHEMA`.
    as_of_ts : datetime | None, optional
        Point-in-time cutoff (`available_ts <= as_of_ts`).

    Returns
    -------
    pl.DataFrame
        Sorted DataFrame with added columns:
        `cum_future_split`, `open_split_adj`, `high_split_adj`, `low_split_adj`,
        `close_split_adj`, and `volume_split_adj`.
    """
    with_returns = compute_total_returns(bars, actions=actions, as_of_ts=as_of_ts)
    if with_returns.is_empty():
        return with_returns.with_columns(
            pl.lit(1.0, dtype=pl.Float64).alias("cum_future_split"),
            pl.lit(np.nan, dtype=pl.Float64).alias("open_split_adj"),
            pl.lit(np.nan, dtype=pl.Float64).alias("high_split_adj"),
            pl.lit(np.nan, dtype=pl.Float64).alias("low_split_adj"),
            pl.lit(np.nan, dtype=pl.Float64).alias("close_split_adj"),
            pl.lit(np.nan, dtype=pl.Float64).alias("volume_split_adj"),
        )

    # Product of split_factor strictly after period_ts within each ticker:
    # Reverse cum_prod of split_factor shifted by 1 in reverse order.
    cum_future_split = (
        pl.col("split_factor")
        .reverse()
        .cum_prod()
        .shift(1, fill_value=1.0)
        .reverse()
        .over("ticker")
    )

    return with_returns.with_columns(cum_future_split.alias("cum_future_split")).with_columns(
        (pl.col("open") / pl.col("cum_future_split")).alias("open_split_adj"),
        (pl.col("high") / pl.col("cum_future_split")).alias("high_split_adj"),
        (pl.col("low") / pl.col("cum_future_split")).alias("low_split_adj"),
        (pl.col("close") / pl.col("cum_future_split")).alias("close_split_adj"),
        (pl.col("volume") * pl.col("cum_future_split")).alias("volume_split_adj"),
    )
