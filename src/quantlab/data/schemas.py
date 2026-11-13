"""Canonical Polars schemas, US equity universe constants, and UTC/XNYS timestamp helpers.

Follows the dual-timestamp (`period_ts`, `available_ts`) and raw-OHLCV + corporate-action
conventions defined in `docs/conventions.md` §1-2.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timezone
from typing import Final
from zoneinfo import ZoneInfo

import polars as pl

UTC_TZ: Final[timezone] = UTC
NY_TZ: Final[ZoneInfo] = ZoneInfo("America/New_York")
US_EQUITY_CLOSE_TIME_ET: Final[time] = time(16, 0, 0)

# Tier 1 Survivorship-Free Core: 14 liquid US Market & Sector ETFs (ADR-0001)
TIER1_US_ETFS: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLV",
    "XLE",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
)

BARS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "ticker": pl.String(),
        "period_ts": pl.Datetime("us", "UTC"),
        "available_ts": pl.Datetime("us", "UTC"),
        "open": pl.Float64(),
        "high": pl.Float64(),
        "low": pl.Float64(),
        "close": pl.Float64(),
        "volume": pl.Float64(),
        "source": pl.String(),
    }
)

ACTIONS_SCHEMA: Final[pl.Schema] = pl.Schema(
    {
        "ticker": pl.String(),
        "ex_date": pl.Datetime("us", "UTC"),
        "available_ts": pl.Datetime("us", "UTC"),
        "split_factor": pl.Float64(),
        "div_cash": pl.Float64(),
        "source": pl.String(),
    }
)


def date_to_period_ts(d: date) -> datetime:
    """Convert a calendar date to a midnight UTC `period_ts` timestamp."""
    return datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=UTC_TZ)


def us_equity_close_available_ts(
    d: date,
    close_time_et: time = US_EQUITY_CLOSE_TIME_ET,
) -> datetime:
    """Convert a US trading date to its UTC `available_ts` at market close (`16:00 ET`).

    Correctly accounts for US Daylight Saving Time (`EDT` -> `20:00:00Z`,
    `EST` -> `21:00:00Z`).
    """
    local_dt = datetime(
        d.year,
        d.month,
        d.day,
        close_time_et.hour,
        close_time_et.minute,
        close_time_et.second,
        tzinfo=NY_TZ,
    )
    return local_dt.astimezone(UTC_TZ)


def empty_bars_frame() -> pl.DataFrame:
    """Return an empty Polars DataFrame adhering to `BARS_SCHEMA`."""
    return pl.DataFrame(schema=BARS_SCHEMA)


def empty_actions_frame() -> pl.DataFrame:
    """Return an empty Polars DataFrame adhering to `ACTIONS_SCHEMA`."""
    return pl.DataFrame(schema=ACTIONS_SCHEMA)


def ensure_schema(df: pl.DataFrame, expected_schema: pl.Schema, name: str = "DataFrame") -> None:
    """Verify that `df` contains all columns and dtypes required by `expected_schema`.

    Raises
    ------
    ValueError
        If required columns are missing or have incompatible Polars dtypes.
    """
    missing = [col for col in expected_schema if col not in df.columns]
    if missing:
        msg = f"{name} is missing required columns: {missing}. Found: {df.columns}"
        raise ValueError(msg)

    mismatches: list[str] = []
    for col, expected_dtype in expected_schema.items():
        actual_dtype = df.schema[col]
        if actual_dtype != expected_dtype:
            mismatches.append(f"{col}: expected {expected_dtype}, got {actual_dtype}")

    if mismatches:
        msg = f"{name} has schema dtype mismatches: {', '.join(mismatches)}"
        raise ValueError(msg)


def normalize_tickers(tickers: Sequence[str]) -> list[str]:
    """Return deduplicated, stripped, uppercase tickers preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in tickers:
        clean = raw.strip().upper()
        if not clean:
            msg = "Ticker symbol cannot be empty or whitespace."
            raise ValueError(msg)
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
