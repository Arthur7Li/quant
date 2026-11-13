"""Unified `DataProvider` protocol and concrete US equity market data providers.

Implements `QUANT_PROJECT_DESIGN.md` §4.2:
- `DataProvider` runtime-checkable protocol.
- `TiingoProvider`: Primary EOD US equity/ETF provider returning raw OHLCV + explicit
  `splitFactor` and `divCash` corporate actions.
- `StooqProvider`: Direct CSV downloader with explicit detection of Stooq's HTTP 200
  `"Exceeded the daily hits limit"` quota message.
- `YFinanceProvider`: Optional wrapper around `yfinance` (with `curl_cffi` session
  support and rate pacing).
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import os
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, Protocol, runtime_checkable

import polars as pl

from quantlab.data.schemas import (
    ACTIONS_SCHEMA,
    BARS_SCHEMA,
    date_to_period_ts,
    empty_actions_frame,
    empty_bars_frame,
    normalize_tickers,
    us_equity_close_available_ts,
)


class DataProviderError(RuntimeError):
    """Base exception raised when a market data provider fails."""


class DataProviderQuotaError(DataProviderError):
    """Raised when a market data provider throttles or exhausts its API/IP quota."""


@runtime_checkable
class DataProvider(Protocol):
    """Unified interface implemented by all market data sources in `quantlab`."""

    name: str

    def fetch_bars(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Return daily OHLCV bars conforming to `BARS_SCHEMA`."""
        ...

    def fetch_corporate_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Return corporate actions (splits & cash dividends) conforming to `ACTIONS_SCHEMA`."""
        ...


HttpGetter = Callable[[str, dict[str, str]], str]


def _default_http_get(url: str, headers: dict[str, str]) -> str:
    """Execute an HTTPS GET request and decode UTF-8 text."""
    if not url.startswith("https://"):
        msg = f"Only https:// URLs are permitted, got {url!r}."
        raise ValueError(msg)
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        raw_bytes: bytes = resp.read()
    return raw_bytes.decode("utf-8")


class TiingoProvider:
    """Tiingo End-of-Day US Equity & ETF data provider (`api.tiingo.com`).

    Parses both raw OHLCV bars (`BARS_SCHEMA`) and corporate actions (`ACTIONS_SCHEMA`)
    from Tiingo's daily prices endpoint.
    """

    name: str = "tiingo"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        http_get: HttpGetter = _default_http_get,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("TIINGO_API_KEY", "")
        if not resolved_key.strip():
            msg = (
                "Tiingo API key is required. Pass `api_key` or set the "
                "`TIINGO_API_KEY` environment variable."
            )
            raise ValueError(msg)
        self._api_key = resolved_key.strip()
        self._http_get = http_get

    def fetch_bars(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch raw daily OHLCV bars from Tiingo."""
        bars, _ = self.fetch_bars_and_actions(tickers, start_date, end_date)
        return bars

    def fetch_corporate_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch split and cash dividend corporate actions from Tiingo."""
        _, actions = self.fetch_bars_and_actions(tickers, start_date, end_date)
        return actions

    def fetch_bars_and_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch both raw bars and corporate actions in a single request per ticker."""
        if end_date < start_date:
            msg = f"end_date ({end_date}) must be >= start_date ({start_date})."
            raise ValueError(msg)

        clean_tickers = normalize_tickers(tickers)
        if not clean_tickers:
            return empty_bars_frame(), empty_actions_frame()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self._api_key}",
        }

        bar_rows: list[dict[str, object]] = []
        action_rows: list[dict[str, object]] = []

        for tkr in clean_tickers:
            url = (
                f"https://api.tiingo.com/tiingo/daily/{tkr}/prices"
                f"?startDate={start_date.isoformat()}&endDate={end_date.isoformat()}&format=json"
            )
            payload_text = self._http_get(url, headers)
            records = json.loads(payload_text)
            if not isinstance(records, list):
                msg = f"Unexpected Tiingo response for {tkr}: {payload_text[:200]}"
                raise DataProviderError(msg)

            for item in records:
                raw_date_str = str(item["date"])[:10]
                bar_date = date.fromisoformat(raw_date_str)
                p_ts = date_to_period_ts(bar_date)
                a_ts = us_equity_close_available_ts(bar_date)

                bar_rows.append(
                    {
                        "ticker": tkr,
                        "period_ts": p_ts,
                        "available_ts": a_ts,
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": float(item["volume"]),
                        "source": self.name,
                    }
                )

                split_val = float(item.get("splitFactor", 1.0))
                div_val = float(item.get("divCash", 0.0))
                if split_val != 1.0 or div_val > 0.0:
                    action_rows.append(
                        {
                            "ticker": tkr,
                            "ex_date": p_ts,
                            "available_ts": a_ts,
                            "split_factor": split_val,
                            "div_cash": div_val,
                            "source": self.name,
                        }
                    )

        bars_df = (
            pl.DataFrame(bar_rows, schema=BARS_SCHEMA).sort(["ticker", "period_ts"])
            if bar_rows
            else empty_bars_frame()
        )
        actions_df = (
            pl.DataFrame(action_rows, schema=ACTIONS_SCHEMA).sort(["ticker", "ex_date"])
            if action_rows
            else empty_actions_frame()
        )
        return bars_df, actions_df


class StooqProvider:
    """Direct Stooq CSV provider (`stooq.com/q/d/l/`) with HTTP 200 quota detection.

    Note: Stooq's public CSV endpoint returns pre-adjusted (split + dividend) prices
    and returns HTTP 200 with body text `"Exceeded the daily hits limit"` when throttled.
    """

    name: str = "stooq"

    def __init__(
        self,
        *,
        market_suffix: str = ".us",
        http_get: HttpGetter = _default_http_get,
    ) -> None:
        self._suffix = market_suffix
        self._http_get = http_get

    def fetch_bars(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch daily bars from Stooq CSV endpoint."""
        if end_date < start_date:
            msg = f"end_date ({end_date}) must be >= start_date ({start_date})."
            raise ValueError(msg)

        clean_tickers = normalize_tickers(tickers)
        if not clean_tickers:
            return empty_bars_frame()

        d1 = start_date.strftime("%Y%m%d")
        d2 = end_date.strftime("%Y%m%d")
        headers = {"User-Agent": "quantlab-research/0.1"}

        bar_rows: list[dict[str, object]] = []
        for tkr in clean_tickers:
            symbol = f"{tkr.lower()}{self._suffix}"
            url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
            csv_text = self._http_get(url, headers).strip()

            if "Exceeded the daily hits limit" in csv_text:
                msg = f"Stooq daily quota exceeded while fetching {tkr}."
                raise DataProviderQuotaError(msg)

            if not csv_text or csv_text.startswith("No data") or "<html" in csv_text.lower():
                msg = f"Stooq returned invalid or empty CSV payload for {tkr}: {csv_text[:120]!r}"
                raise DataProviderError(msg)

            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                bar_date = date.fromisoformat(str(row["Date"]).strip())
                if bar_date < start_date or bar_date > end_date:
                    continue
                bar_rows.append(
                    {
                        "ticker": tkr,
                        "period_ts": date_to_period_ts(bar_date),
                        "available_ts": us_equity_close_available_ts(bar_date),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row.get("Volume") or 0.0),
                        "source": self.name,
                    }
                )

        if not bar_rows:
            return empty_bars_frame()
        return pl.DataFrame(bar_rows, schema=BARS_SCHEMA).sort(["ticker", "period_ts"])

    def fetch_corporate_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Return an empty actions DataFrame (Stooq CSVs embed adjustments directly)."""
        return empty_actions_frame()


class YFinanceProvider:
    """Yahoo Finance wrapper (`yfinance`) with optional `curl_cffi` session and pacing."""

    name: str = "yfinance"

    def __init__(self, *, min_delay_seconds: float = 1.5) -> None:
        if min_delay_seconds < 0.0:
            msg = f"min_delay_seconds must be >= 0, got {min_delay_seconds}."
            raise ValueError(msg)
        self._min_delay = min_delay_seconds

    def fetch_bars(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch raw daily OHLCV bars via `yfinance`."""
        bars, _ = self.fetch_bars_and_actions(tickers, start_date, end_date)
        return bars

    def fetch_corporate_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch splits and cash dividends via `yfinance`."""
        _, actions = self.fetch_bars_and_actions(tickers, start_date, end_date)
        return actions

    def fetch_bars_and_actions(
        self,
        tickers: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Fetch raw bars and corporate actions from `yfinance` with rate pacing."""
        try:
            yf_mod: Any = importlib.import_module("yfinance")
        except ImportError as exc:
            msg = "yfinance is not installed. Run `uv sync --extra market` to install it."
            raise DataProviderError(msg) from exc

        session: Any = None
        try:
            curl_requests: Any = importlib.import_module("curl_cffi.requests")
            session = curl_requests.Session(impersonate="chrome")
        except ImportError:
            session = None

        clean_tickers = normalize_tickers(tickers)
        bar_rows: list[dict[str, object]] = []
        action_rows: list[dict[str, object]] = []

        for idx, tkr in enumerate(clean_tickers):
            if idx > 0 and self._min_delay > 0.0:
                time.sleep(self._min_delay)

            ticker_obj = (
                yf_mod.Ticker(tkr, session=session) if session is not None else yf_mod.Ticker(tkr)
            )
            hist_df = ticker_obj.history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=False,
                actions=True,
            )
            if hist_df is None or hist_df.empty:
                continue

            for idx_ts, row in hist_df.iterrows():
                bar_date: date = idx_ts.date()
                p_ts = date_to_period_ts(bar_date)
                a_ts = us_equity_close_available_ts(bar_date)

                bar_rows.append(
                    {
                        "ticker": tkr,
                        "period_ts": p_ts,
                        "available_ts": a_ts,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"]),
                        "source": self.name,
                    }
                )

                raw_split = float(row.get("Stock Splits", 0.0) or 0.0)
                split_factor = raw_split if raw_split > 0.0 else 1.0
                div_cash = float(row.get("Dividends", 0.0) or 0.0)
                if split_factor != 1.0 or div_cash > 0.0:
                    action_rows.append(
                        {
                            "ticker": tkr,
                            "ex_date": p_ts,
                            "available_ts": a_ts,
                            "split_factor": split_factor,
                            "div_cash": div_cash,
                            "source": self.name,
                        }
                    )

        bars_df = (
            pl.DataFrame(bar_rows, schema=BARS_SCHEMA).sort(["ticker", "period_ts"])
            if bar_rows
            else empty_bars_frame()
        )
        actions_df = (
            pl.DataFrame(action_rows, schema=ACTIONS_SCHEMA).sort(["ticker", "ex_date"])
            if action_rows
            else empty_actions_frame()
        )
        return bars_df, actions_df
