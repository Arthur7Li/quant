"""Known-answer, property, validation, PIT look-ahead, and provider tests for `quantlab.data`."""

from __future__ import annotations

import builtins
import importlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantlab.data import (
    ACTIONS_SCHEMA,
    BARS_SCHEMA,
    TIER1_US_ETFS,
    UTC_TZ,
    DataProvider,
    DataProviderError,
    DataProviderQuotaError,
    PointInTimeStore,
    StooqProvider,
    SyntheticDataProvider,
    TiingoProvider,
    YFinanceProvider,
    apply_split_adjustments,
    business_days_between,
    compute_dataframe_hash,
    compute_total_return_index,
    compute_total_returns,
    date_to_period_ts,
    empty_actions_frame,
    empty_bars_frame,
    ensure_schema,
    make_toy_5bar_fixture,
    normalize_tickers,
    us_equity_close_available_ts,
    validate_bars,
)


def test_us_equity_close_available_ts_handles_est_and_edt() -> None:
    """Verify 16:00 America/New_York converts to 21:00 UTC in EST and 20:00 UTC in EDT."""
    winter_ts = us_equity_close_available_ts(date(2025, 1, 6))
    summer_ts = us_equity_close_available_ts(date(2025, 7, 7))

    assert winter_ts == datetime(2025, 1, 6, 21, 0, 0, tzinfo=UTC_TZ)
    assert summer_ts == datetime(2025, 7, 7, 20, 0, 0, tzinfo=UTC_TZ)


def test_known_answer_5bar_corporate_actions_and_tri() -> None:
    """Verify hand-calculated 5-bar split + dividend fixture to machine precision."""
    bars, actions = make_toy_5bar_fixture("toy")

    # 1. Total returns and raw price returns
    res = compute_total_returns(bars, actions)
    price_rets = res["price_return"].to_numpy()
    total_rets = res["total_return"].to_numpy()

    assert np.isnan(price_rets[0])
    assert np.isnan(total_rets[0])

    # Raw price returns drop -50% on split day (bar 2) and -1.818% on dividend day (bar 3)
    expected_price_rets = np.array([0.10, -0.50, 54.0 / 55.0 - 1.0, 0.10])
    np.testing.assert_allclose(price_rets[1:], expected_price_rets, rtol=1e-12, atol=1e-12)

    # Total returns are +10%, 0%, 0%, +10%
    expected_total_rets = np.array([0.10, 0.0, 0.0, 0.10])
    np.testing.assert_allclose(total_rets[1:], expected_total_rets, rtol=1e-12, atol=1e-12)

    # 2. Forward Total Return Index (TRI)
    tri_df = compute_total_return_index(bars, actions, initial_value=100.0)
    expected_tri = np.array([100.0, 110.0, 110.0, 110.0, 121.0])
    np.testing.assert_allclose(tri_df["tri"].to_numpy(), expected_tri, rtol=1e-12, atol=1e-12)

    # 3. Point-in-time split adjustments
    split_df = apply_split_adjustments(bars, actions)
    expected_split_adj_close = np.array([50.0, 55.0, 55.0, 54.0, 59.4])
    expected_split_adj_vol = np.array(
        [2_000_000.0, 2_400_000.0, 2_400_000.0, 2_100_000.0, 2_500_000.0]
    )
    np.testing.assert_allclose(
        split_df["close_split_adj"].to_numpy(), expected_split_adj_close, rtol=1e-12
    )
    np.testing.assert_allclose(
        split_df["volume_split_adj"].to_numpy(), expected_split_adj_vol, rtol=1e-12
    )

    # 4. Point-in-time cutoff BEFORE the split on bar 2 (as of bar 1 close):
    # At bar 1, the 2-for-1 split on bar 2 has not happened yet, so close_split_adj == raw close!
    as_of_bar1 = bars["available_ts"][1]
    split_pit_bar1 = apply_split_adjustments(bars, actions, as_of_ts=as_of_bar1)
    assert split_pit_bar1.height == 2
    np.testing.assert_allclose(
        split_pit_bar1["close_split_adj"].to_numpy(),
        np.array([100.0, 110.0]),
        rtol=1e-12,
    )


@settings(max_examples=40, deadline=None)
@given(
    log_returns=st.lists(
        st.floats(min_value=-0.15, max_value=0.15, allow_nan=False, allow_infinity=False),
        min_size=6,
        max_size=25,
    ),
    split_factor=st.floats(min_value=0.25, max_value=5.0, allow_nan=False, allow_infinity=False),
    div_yield=st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False),
)
def test_property_split_invariance_and_forward_tri_pit_immutability(
    log_returns: list[float],
    split_factor: float,
    div_yield: float,
) -> None:
    """Property test:
    1. Applying any split factor `S_t` at bar `k` leaves `total_return` and `tri` invariant.
    2. Forward `tri` up to `t_0` is strictly unchanged when future bars, splits, or
       dividends are appended at `t > t_0`.
    """
    n_bars = len(log_returns)
    split_idx = n_bars // 2
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(n_bars)]
    period_ts = [date_to_period_ts(d) for d in dates]
    avail_ts = [us_equity_close_available_ts(d) for d in dates]

    latent_prices = 100.0 * np.exp(np.cumsum([0.0, *log_returns[1:]]))

    # Baseline unsplit bars (no corporate actions)
    unsplit_bars = pl.DataFrame(
        {
            "ticker": ["PROP"] * n_bars,
            "period_ts": period_ts,
            "available_ts": avail_ts,
            "open": latent_prices.tolist(),
            "high": (latent_prices * 1.01).tolist(),
            "low": (latent_prices * 0.99).tolist(),
            "close": latent_prices.tolist(),
            "volume": [10_000.0] * n_bars,
            "source": ["prop"] * n_bars,
        },
        schema=BARS_SCHEMA,
    )

    # Split-modified raw prices from split_idx onward
    split_prices = latent_prices.copy()
    split_prices[split_idx:] /= split_factor

    split_bars = unsplit_bars.with_columns(
        pl.Series("open", split_prices.tolist(), dtype=pl.Float64),
        pl.Series("high", (split_prices * 1.01).tolist(), dtype=pl.Float64),
        pl.Series("low", (split_prices * 0.99).tolist(), dtype=pl.Float64),
        pl.Series("close", split_prices.tolist(), dtype=pl.Float64),
    )
    split_actions = pl.DataFrame(
        {
            "ticker": ["PROP"],
            "ex_date": [period_ts[split_idx]],
            "available_ts": [avail_ts[split_idx]],
            "split_factor": [split_factor],
            "div_cash": [0.0],
            "source": ["prop"],
        },
        schema=ACTIONS_SCHEMA,
    )

    tri_unsplit = compute_total_return_index(unsplit_bars)
    tri_split = compute_total_return_index(split_bars, split_actions)

    np.testing.assert_allclose(
        tri_split["total_return"].to_numpy()[1:],
        tri_unsplit["total_return"].to_numpy()[1:],
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tri_split["tri"].to_numpy(),
        tri_unsplit["tri"].to_numpy(),
        rtol=1e-10,
        atol=1e-10,
    )

    # Property 2: Adding a cash dividend and split AFTER cutoff_idx does not change
    # TRI <= cutoff_idx
    cutoff_idx = max(1, split_idx - 1)
    cutoff_ts = avail_ts[cutoff_idx]

    future_div_actions = pl.concat(
        [
            split_actions,
            pl.DataFrame(
                {
                    "ticker": ["PROP"],
                    "ex_date": [period_ts[-1]],
                    "available_ts": [avail_ts[-1]],
                    "split_factor": [1.0],
                    "div_cash": [float(split_prices[-1] * div_yield)],
                    "source": ["prop"],
                },
                schema=ACTIONS_SCHEMA,
            ),
        ]
    )

    tri_full = compute_total_return_index(split_bars, future_div_actions)
    tri_pit = compute_total_return_index(split_bars, future_div_actions, as_of_ts=cutoff_ts)

    np.testing.assert_allclose(
        tri_pit["tri"].to_numpy(),
        tri_full["tri"].to_numpy()[: cutoff_idx + 1],
        rtol=1e-12,
        atol=1e-12,
    )


def test_validation_suite_passes_clean_and_catches_all_corruptions() -> None:
    """Verify `validate_bars` on clean synthetic data and intentional corruption modes."""
    provider = SyntheticDataProvider(seed=7)
    assert isinstance(provider, DataProvider)

    bars = provider.fetch_bars(["SPY", "XLK"], date(2025, 1, 1), date(2025, 3, 1))
    actions = provider.fetch_corporate_actions(["SPY", "XLK"], date(2025, 1, 1), date(2025, 3, 1))

    clean_report = validate_bars(bars, actions)
    assert clean_report.is_valid
    assert clean_report.errors == ()
    clean_report.raise_if_invalid()

    # 1. Missing corporate actions causes the 2-for-1 synthetic split (-50% raw drop) to be flagged!
    no_action_report = validate_bars(bars, actions=None, max_unexplained_abs_return=0.40)
    assert not no_action_report.is_valid
    assert any("unexplained extreme return jump" in err for err in no_action_report.errors)
    with pytest.raises(ValueError, match="Data validation failed"):
        no_action_report.raise_if_invalid()

    # 2. Negative price and OHLC violation
    bad_price_bars = bars.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 0)
        .then(-10.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    rep_neg = validate_bars(bad_price_bars, actions)
    assert not rep_neg.is_valid
    assert any("non-positive price" in err for err in rep_neg.errors)
    assert any("violating OHLC bounds" in err for err in rep_neg.errors)

    # 3. Look-ahead timestamp violation (available_ts < period_ts)
    bad_pit_bars = bars.with_columns(
        (pl.col("period_ts") - pl.duration(hours=1)).alias("available_ts")
    )
    rep_pit = validate_bars(bad_pit_bars, actions)
    assert not rep_pit.is_valid
    assert any("available_ts < period_ts" in err for err in rep_pit.errors)

    # 4. Duplicate row violation
    dup_bars = pl.concat([bars, bars.head(1)])
    rep_dup = validate_bars(dup_bars, actions)
    assert not rep_dup.is_valid
    assert any("duplicate" in err for err in rep_dup.errors)

    # 5. Stale price streak warning
    toy_bars, _ = make_toy_5bar_fixture("STALE")
    flat_bars = toy_bars.with_columns(
        pl.lit(100.0).alias("open"),
        pl.lit(101.0).alias("high"),
        pl.lit(99.0).alias("low"),
        pl.lit(100.0).alias("close"),
    )
    rep_stale = validate_bars(flat_bars, max_stale_streak=3)
    assert rep_stale.is_valid
    assert any("consecutive unchanged close" in w for w in rep_stale.warnings)


def test_point_in_time_store_ingest_query_and_return_panel(tmp_path: Path) -> None:
    """Verify DuckDB/Parquet PIT store look-ahead filtering, hashing, and `(T, N)` panel."""
    bars_a, actions_a = make_toy_5bar_fixture("AAA")
    bars_b, actions_b = make_toy_5bar_fixture("BBB")

    combined_bars = pl.concat([bars_a, bars_b])
    combined_actions = pl.concat([actions_a, actions_b])

    store = PointInTimeStore(tmp_path / "pit")
    with pytest.raises(FileNotFoundError, match="No PIT manifest"):
        store.get_dataset_hash()
    assert store.query_bars(us_equity_close_available_ts(date(2025, 1, 8))).is_empty()
    assert store.query_actions(us_equity_close_available_ts(date(2025, 1, 8))).is_empty()
    empty_mat, empty_ts, empty_tkrs = store.to_return_panel(
        us_equity_close_available_ts(date(2025, 1, 8))
    )
    assert empty_mat.shape == (0, 0)
    assert empty_ts == []
    assert empty_tkrs == []

    h1 = store.ingest(combined_bars, combined_actions)
    assert len(h1) == 64
    assert store.get_dataset_hash() == h1

    # Re-ingesting identical data produces the exact same content hash (idempotency)
    h2 = store.ingest(combined_bars, combined_actions)
    assert h1 == h2

    # Query as of Day 2 (`2025-01-08 21:00:00Z`): Days 3 and 4 (`2025-01-09`, `2025-01-10`)
    # must be invisible!
    cutoff_day2 = us_equity_close_available_ts(date(2025, 1, 8))
    pit_bars = store.query_bars(cutoff_day2)
    assert pit_bars.height == 6  # 3 bars * 2 tickers
    assert pit_bars["period_ts"].max() == date_to_period_ts(date(2025, 1, 8))

    # Filter by single ticker and start/end window
    sub_bars = store.query_bars(
        cutoff_day2,
        tickers=["AAA"],
        start_ts=date_to_period_ts(date(2025, 1, 7)),
        end_ts=date_to_period_ts(date(2025, 1, 8)),
    )
    assert sub_bars.height == 2
    assert sub_bars["ticker"].unique().to_list() == ["AAA"]

    # Query actions as of Day 2: only the split on Day 2 is visible; Day 3 dividend is invisible
    pit_actions = store.query_actions(cutoff_day2, tickers=["AAA"])
    assert pit_actions.height == 1
    assert all(s == 2.0 for s in pit_actions["split_factor"].to_list())

    # Full `(T, N)` return panel as of Day 4 (`2025-01-10`)
    cutoff_day4 = us_equity_close_available_ts(date(2025, 1, 10))
    mat, timestamps, tickers = store.to_return_panel(cutoff_day4)
    assert mat.shape == (4, 2)
    assert mat.dtype == np.float64
    assert tickers == ["AAA", "BBB"]
    assert len(timestamps) == 4
    expected_col = np.array([0.10, 0.0, 0.0, 0.10])
    np.testing.assert_allclose(mat[:, 0], expected_col, rtol=1e-12)
    np.testing.assert_allclose(mat[:, 1], expected_col, rtol=1e-12)

    # Naive datetime raises ValueError
    with pytest.raises(ValueError, match="timezone-aware"):
        store.query_bars(datetime(2025, 1, 8, 21, 0, 0))


def test_tiingo_and_stooq_providers_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Tiingo JSON parsing and Stooq CSV parsing + HTTP 200 quota detection."""
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Tiingo API key is required"):
        TiingoProvider()

    sample_tiingo_payload = json.dumps(
        [
            {
                "date": "2025-01-06T00:00:00.000Z",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 500000,
                "splitFactor": 1.0,
                "divCash": 0.0,
            },
            {
                "date": "2025-01-07T00:00:00.000Z",
                "open": 50.5,
                "high": 52.0,
                "low": 50.0,
                "close": 51.0,
                "volume": 1000000,
                "splitFactor": 2.0,
                "divCash": 0.50,
            },
        ]
    )

    tiingo = TiingoProvider(api_key="test-key", http_get=lambda _u, _h: sample_tiingo_payload)
    assert isinstance(tiingo, DataProvider)
    t_bars = tiingo.fetch_bars(["spy"], date(2025, 1, 6), date(2025, 1, 7))
    t_actions = tiingo.fetch_corporate_actions(["spy"], date(2025, 1, 6), date(2025, 1, 7))

    assert t_bars.height == 2
    assert t_actions.height == 1
    assert t_actions["split_factor"][0] == 2.0
    assert t_actions["div_cash"][0] == 0.50

    # Stooq valid CSV
    stooq_csv = (
        "Date,Open,High,Low,Close,Volume\n"
        "2025-01-06,100.0,102.0,99.0,101.0,500000\n"
        "2025-01-07,101.0,103.0,100.5,102.5,600000\n"
    )
    stooq = StooqProvider(http_get=lambda _u, _h: stooq_csv)
    assert isinstance(stooq, DataProvider)
    s_bars = stooq.fetch_bars(["SPY"], date(2025, 1, 6), date(2025, 1, 7))
    assert s_bars.height == 2
    assert stooq.fetch_corporate_actions(["SPY"], date(2025, 1, 6), date(2025, 1, 7)).is_empty()

    # Stooq HTTP 200 quota message raises DataProviderQuotaError
    stooq_throttled = StooqProvider(
        http_get=lambda _u, _h: "Exceeded the daily hits limit. Please try again tomorrow."
    )
    with pytest.raises(DataProviderQuotaError, match="quota exceeded"):
        stooq_throttled.fetch_bars(["SPY"], date(2025, 1, 6), date(2025, 1, 7))

    # Stooq "No data" raises DataProviderError
    stooq_nodata = StooqProvider(http_get=lambda _u, _h: "No data")
    with pytest.raises(DataProviderError, match="invalid or empty CSV"):
        stooq_nodata.fetch_bars(["SPY"], date(2025, 1, 6), date(2025, 1, 7))


def test_yfinance_provider_mocked_and_missing_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify YFinanceProvider with a fake `yfinance` module and when `yfinance` is absent."""
    rows = [
        (
            datetime(2025, 1, 6, tzinfo=UTC_TZ),
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Volume": 1000.0,
                "Dividends": 0.0,
                "Stock Splits": 0.0,
            },
        ),
        (
            datetime(2025, 1, 7, tzinfo=UTC_TZ),
            {
                "Open": 50.0,
                "High": 51.0,
                "Low": 49.5,
                "Close": 50.25,
                "Volume": 2000.0,
                "Dividends": 0.25,
                "Stock Splits": 2.0,
            },
        ),
    ]
    fake_hist = SimpleNamespace(empty=False, iterrows=lambda: iter(rows))

    class _FakeTicker:
        def __init__(self, _tkr: str, session: Any = None) -> None:
            self.session = session

        def history(self, **_kwargs: Any) -> Any:
            return fake_hist

    def _fake_import(name: str) -> Any:
        if name == "yfinance":
            return SimpleNamespace(Ticker=_FakeTicker)
        raise builtins.ImportError(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    yf_prov = YFinanceProvider(min_delay_seconds=0.0)
    assert isinstance(yf_prov, DataProvider)
    bars = yf_prov.fetch_bars(["AAPL"], date(2025, 1, 6), date(2025, 1, 8))
    actions = yf_prov.fetch_corporate_actions(["AAPL"], date(2025, 1, 6), date(2025, 1, 8))
    assert bars.height == 2
    assert actions.height == 1
    assert actions["split_factor"][0] == 2.0
    assert actions["div_cash"][0] == 0.25

    # Missing yfinance raises DataProviderError
    def _raise_import(_name: str) -> Any:
        raise builtins.ImportError("no yfinance")

    monkeypatch.setattr(importlib, "import_module", _raise_import)
    with pytest.raises(DataProviderError, match="yfinance is not installed"):
        yf_prov.fetch_bars(["AAPL"], date(2025, 1, 6), date(2025, 1, 8))


def test_schema_and_helper_edge_cases() -> None:
    """Verify schema validation errors, empty frames, and ticker normalization."""
    assert len(TIER1_US_ETFS) == 14
    assert normalize_tickers([" spy ", "QQQ", "spy"]) == ["SPY", "QQQ"]

    with pytest.raises(ValueError, match="cannot be empty"):
        normalize_tickers([""])

    with pytest.raises(ValueError, match="missing required columns"):
        ensure_schema(pl.DataFrame({"ticker": ["SPY"]}), BARS_SCHEMA)

    bad_dtype_df = empty_bars_frame().with_columns(pl.col("close").cast(pl.Int64))
    with pytest.raises(ValueError, match="schema dtype mismatches"):
        ensure_schema(bad_dtype_df, BARS_SCHEMA)

    assert compute_total_returns(empty_bars_frame()).is_empty()
    assert compute_total_return_index(empty_bars_frame()).is_empty()
    assert apply_split_adjustments(empty_bars_frame()).is_empty()
    assert validate_bars(empty_bars_frame()).is_valid
    assert len(compute_dataframe_hash(empty_bars_frame(), ["ticker", "period_ts"])) == 64
    assert business_days_between(date(2025, 1, 4), date(2025, 1, 5)) == []
    assert empty_actions_frame().is_empty()
