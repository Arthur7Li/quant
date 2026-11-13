"""Deterministic synthetic market data generator and known-answer fixtures.

Provides:
1. `make_toy_5bar_fixture()`: Hand-calculated 5-bar US equity fixture with a 2-for-1 split
   on day 2 and a $1.00 cash dividend on day 3 (used across unit tests for adjustments,
   returns, and the event-driven backtester).
2. `SyntheticDataProvider`: Seeded multi-asset Geometric Brownian Motion (GBM) + shared
   market factor generator implementing `DataProvider` for offline CI and leakage tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import polars as pl

from quantlab.compute import check_array_memory
from quantlab.data.schemas import (
    ACTIONS_SCHEMA,
    BARS_SCHEMA,
    date_to_period_ts,
    empty_actions_frame,
    empty_bars_frame,
    normalize_tickers,
    us_equity_close_available_ts,
)


def business_days_between(start_date: date, end_date: date) -> list[date]:
    """Return chronological Monday-Friday dates in `[start_date, end_date]` inclusive."""
    if end_date < start_date:
        msg = f"end_date ({end_date}) must be >= start_date ({start_date})."
        raise ValueError(msg)
    days: list[date] = []
    cur = start_date
    one_day = timedelta(days=1)
    while cur <= end_date:
        if cur.weekday() < 5:
            days.append(cur)
        cur += one_day
    return days


def make_toy_5bar_fixture(ticker: str = "TOY") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create a hand-verified 5-bar US equity fixture with a split and a cash dividend.

    Timeline & Hand-Computed Values
    -------------------------------
    - Bar 0 (`2025-01-06`): Raw Close = $100.00. Initial TRI = 100.0.
    - Bar 1 (`2025-01-07`): Raw Close = $110.00 (+10% gain). TRI = 110.0.
    - Bar 2 (`2025-01-08`): **2-for-1 Split (`split_factor = 2.0`)**.
      Post-split Raw Close = $55.00. Total return = `(55 * 2 + 0) / 110 - 1 = 0.0`.
      TRI = 110.0.
    - Bar 3 (`2025-01-09`): **$1.00 Cash Dividend (`div_cash = 1.0`)**.
      Ex-dividend Raw Close = $54.00. Total return = `(54 * 1 + 1) / 55 - 1 = 0.0`.
      TRI = 110.0.
    - Bar 4 (`2025-01-10`): Raw Close = $59.40 (+10% gain).
      Total return = `(59.40 * 1 + 0) / 54 - 1 = +0.10`. TRI = 121.0.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        `(bars, actions)` conforming to `BARS_SCHEMA` and `ACTIONS_SCHEMA`.
    """
    clean_tkr = ticker.strip().upper()
    dates = [
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 8),
        date(2025, 1, 9),
        date(2025, 1, 10),
    ]
    closes = [100.0, 110.0, 55.0, 54.0, 59.4]
    opens = [99.5, 100.5, 55.0, 54.0, 54.2]
    highs = [101.0, 111.0, 56.0, 54.5, 60.0]
    lows = [99.0, 100.0, 54.5, 53.5, 54.0]
    volumes = [1_000_000.0, 1_200_000.0, 2_400_000.0, 2_100_000.0, 2_500_000.0]

    bars = pl.DataFrame(
        {
            "ticker": [clean_tkr] * 5,
            "period_ts": [date_to_period_ts(d) for d in dates],
            "available_ts": [us_equity_close_available_ts(d) for d in dates],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "source": ["synthetic_toy"] * 5,
        },
        schema=BARS_SCHEMA,
    )

    actions = pl.DataFrame(
        {
            "ticker": [clean_tkr, clean_tkr],
            "ex_date": [date_to_period_ts(dates[2]), date_to_period_ts(dates[3])],
            "available_ts": [
                us_equity_close_available_ts(dates[2]),
                us_equity_close_available_ts(dates[3]),
            ],
            "split_factor": [2.0, 1.0],
            "div_cash": [0.0, 1.0],
            "source": ["synthetic_toy", "synthetic_toy"],
        },
        schema=ACTIONS_SCHEMA,
    )
    return bars, actions


class SyntheticDataProvider:
    """Deterministic multi-asset US equity bar and corporate action generator for CI."""

    name: str = "synthetic"

    def __init__(
        self,
        *,
        seed: int = 42,
        annual_drift: float = 0.08,
        annual_vol: float = 0.18,
        market_beta_Correlation: float = 0.5,
        include_corporate_actions: bool = True,
    ) -> None:
        if annual_vol <= 0.0:
            msg = f"annual_vol must be positive, got {annual_vol}."
            raise ValueError(msg)
        if not 0.0 <= market_beta_Correlation <= 1.0:
            msg = f"market_beta_Correlation must be in [0, 1], got {market_beta_Correlation}."
            raise ValueError(msg)

        self.seed = seed
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol
        self.market_corr = market_beta_Correlation
        self.include_corporate_actions = include_corporate_actions

    def fetch_bars(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Generate deterministic daily OHLCV bars for `tickers` in `[start_date, end_date]`."""
        bars, _ = self._generate(tickers, start_date, end_date)
        return bars

    def fetch_corporate_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Generate deterministic corporate actions for `tickers` in `[start_date, end_date]`."""
        _, actions = self._generate(tickers, start_date, end_date)
        return actions

    def _generate(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        clean_tickers = normalize_tickers(tickers)
        days = business_days_between(start_date, end_date)
        n_days = len(days)
        n_assets = len(clean_tickers)

        if n_days == 0 or n_assets == 0:
            return empty_bars_frame(), empty_actions_frame()

        check_array_memory((n_days, n_assets), dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        dt = 1.0 / 252.0
        daily_drift = (self.annual_drift - 0.5 * self.annual_vol**2) * dt
        daily_vol = self.annual_vol * np.sqrt(dt)

        # Shared market mode + idiosyncratic shock
        market_z = rng.standard_normal((n_days, 1))
        idio_z = rng.standard_normal((n_days, n_assets))
        rho = self.market_corr
        combined_z = np.sqrt(rho) * market_z + np.sqrt(1.0 - rho) * idio_z
        log_rets = daily_drift + daily_vol * combined_z
        log_rets[0, :] = 0.0

        period_ts_list = [date_to_period_ts(d) for d in days]
        avail_ts_list = [us_equity_close_available_ts(d) for d in days]

        bar_frames: list[pl.DataFrame] = []
        action_rows: list[dict[str, object]] = []

        for col_idx, tkr in enumerate(clean_tickers):
            # Continuous latent total-value path starting at $100
            latent_close = 100.0 * np.exp(np.cumsum(log_rets[:, col_idx]))
            raw_close = latent_close.copy()

            if self.include_corporate_actions and n_days >= 20:
                split_idx = n_days // 2
                # Apply a 2-for-1 split from split_idx onward in raw price space
                raw_close[split_idx:] /= 2.0
                action_rows.append(
                    {
                        "ticker": tkr,
                        "ex_date": period_ts_list[split_idx],
                        "available_ts": avail_ts_list[split_idx],
                        "split_factor": 2.0,
                        "div_cash": 0.0,
                        "source": self.name,
                    }
                )

            # Construct realistic OHLC around raw_close
            intraday_spread = np.abs(rng.normal(0.004, 0.002, size=n_days))
            open_Offset = rng.uniform(-0.5, 0.5, size=n_days) * intraday_spread
            raw_open = raw_close * (1.0 + open_Offset)
            raw_high = np.maximum(raw_open, raw_close) * (1.0 + 0.5 * intraday_spread)
            raw_low = np.minimum(raw_open, raw_close) * (1.0 - 0.5 * intraday_spread)
            volume = np.round(rng.lognormal(mean=14.0, sigma=0.3, size=n_days))

            bar_frames.append(
                pl.DataFrame(
                    {
                        "ticker": [tkr] * n_days,
                        "period_ts": period_ts_list,
                        "available_ts": avail_ts_list,
                        "open": raw_open.tolist(),
                        "high": raw_high.tolist(),
                        "low": raw_low.tolist(),
                        "close": raw_close.tolist(),
                        "volume": volume.tolist(),
                        "source": [self.name] * n_days,
                    },
                    schema=BARS_SCHEMA,
                )
            )

        bars_df = pl.concat(bar_frames).sort(["ticker", "period_ts"])
        actions_df = (
            pl.DataFrame(action_rows, schema=ACTIONS_SCHEMA).sort(["ticker", "ex_date"])
            if action_rows
            else empty_actions_frame()
        )
        return bars_df, actions_df
