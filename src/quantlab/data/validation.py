"""Automated data quality and point-in-time sanity checks (`quantlab.data.validation`).

Implements `QUANT_PROJECT_DESIGN.md` §4 Rule 6:
Every ingestion run validates schema types, nulls/NaNs, positive prices, OHLC ordering,
timestamp monotonicity, point-in-time causal ordering (`available_ts >= period_ts`),
unexplained extreme jumps (>50% without a split/dividend), and stale price streaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from quantlab.data.adjustments import compute_total_returns
from quantlab.data.schemas import ACTIONS_SCHEMA, BARS_SCHEMA, ensure_schema


@dataclass(frozen=True)
class ValidationReport:
    """Result of running the data validation suite on a bar dataset."""

    is_valid: bool
    row_count: int
    ticker_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def raise_if_invalid(self) -> None:
        """Raise `ValueError` if `is_valid` is False."""
        if not self.is_valid:
            joined = "; ".join(self.errors)
            msg = f"Data validation failed with {len(self.errors)} error(s): {joined}"
            raise ValueError(msg)


def validate_bars(
    bars: pl.DataFrame,
    actions: pl.DataFrame | None = None,
    *,
    max_unexplained_abs_return: float = 0.50,
    max_stale_streak: int = 5,
) -> ValidationReport:
    """Run all integrity and point-in-time checks on `bars` (and optional `actions`).

    Parameters
    ----------
    bars : pl.DataFrame
        Daily OHLCV bars conforming to `BARS_SCHEMA`.
    actions : pl.DataFrame | None, optional
        Corporate actions conforming to `ACTIONS_SCHEMA`.
    max_unexplained_abs_return : float, default 0.50
        Threshold for flagging single-day raw price jumps (`|R_t| > 50%`) that do
        not have a corresponding corporate action on that date.
    max_stale_streak : int, default 5
        Maximum number of consecutive identical close prices (with positive volume)
        before emitting a stale-price warning.

    Returns
    -------
    ValidationReport
        Structured validation report containing `is_valid`, `errors`, and `warnings`.
    """
    ensure_schema(bars, BARS_SCHEMA, name="bars")
    if actions is not None and not actions.is_empty():
        ensure_schema(actions, ACTIONS_SCHEMA, name="actions")

    errors: list[str] = []
    warnings: list[str] = []

    row_count = bars.height
    ticker_count = int(bars["ticker"].n_unique()) if row_count > 0 else 0

    if row_count == 0:
        warnings.append("Dataset is empty (0 rows).")
        return ValidationReport(
            is_valid=True,
            row_count=0,
            ticker_count=0,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    # 1. Check nulls and NaNs across required columns
    for col in BARS_SCHEMA:
        null_count = int(bars[col].null_count())
        if null_count > 0:
            errors.append(f"Column {col!r} contains {null_count} null value(s).")
        if bars.schema[col] == pl.Float64:
            nan_count = int(bars[col].is_nan().sum())
            if nan_count > 0:
                errors.append(f"Column {col!r} contains {nan_count} NaN value(s).")

    # 2. Check positive prices and non-negative volume
    for price_col in ("open", "high", "low", "close"):
        non_pos = int((bars[price_col] <= 0.0).sum())
        if non_pos > 0:
            errors.append(f"Column {price_col!r} has {non_pos} non-positive price(s) (<= 0).")

    neg_vol = int((bars["volume"] < 0.0).sum())
    if neg_vol > 0:
        errors.append(f"Column 'volume' has {neg_vol} negative value(s).")

    # 3. Check OHLC bar ordering invariants: low <= open <= high and low <= close <= high
    bad_ohlc = int(
        (
            (bars["low"] > bars["high"])
            | (bars["open"] < bars["low"])
            | (bars["open"] > bars["high"])
            | (bars["close"] < bars["low"])
            | (bars["close"] > bars["high"])
        ).sum()
    )
    if bad_ohlc > 0:
        errors.append(
            f"Found {bad_ohlc} bar(s) violating OHLC bounds (low <= open, close <= high)."
        )

    # 4. Check Point-in-Time causal ordering: available_ts >= period_ts
    pit_violations = int((bars["available_ts"] < bars["period_ts"]).sum())
    if pit_violations > 0:
        errors.append(
            f"Found {pit_violations} bar(s) with available_ts < period_ts (look-ahead timestamp)."
        )

    # 5. Check per-ticker duplicate or non-monotonic period_ts / available_ts
    dup_count = int(bars.select(["ticker", "period_ts"]).is_duplicated().sum())
    if dup_count > 0:
        errors.append(f"Found {dup_count} duplicate (ticker, period_ts) row(s).")

    for ticker_tuple, group in bars.group_by("ticker", maintain_order=True):
        tkr = str(ticker_tuple[0])
        if not group["period_ts"].is_sorted():
            errors.append(f"Ticker {tkr!r} has non-monotonic period_ts ordering.")
        if not group["available_ts"].is_sorted():
            errors.append(f"Ticker {tkr!r} has non-monotonic available_ts ordering.")

    # 6. Check corporate actions validity if provided
    if actions is not None and not actions.is_empty():
        bad_splits = int((actions["split_factor"] <= 0.0).sum())
        if bad_splits > 0:
            errors.append(f"actions has {bad_splits} non-positive split_factor value(s).")
        bad_divs = int((actions["div_cash"] < 0.0).sum())
        if bad_divs > 0:
            errors.append(f"actions has {bad_divs} negative div_cash value(s).")

    # Stop before return-based checks if basic structural errors exist
    if errors:
        return ValidationReport(
            is_valid=False,
            row_count=row_count,
            ticker_count=ticker_count,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    # 7. Check unexplained extreme jumps (> max_unexplained_abs_return without split/dividend)
    enriched = compute_total_returns(bars, actions=actions)
    unexplained_jumps = enriched.filter(
        pl.col("price_return").is_not_nan()
        & (pl.col("price_return").abs() > max_unexplained_abs_return)
        & (pl.col("split_factor") == 1.0)
        & (pl.col("div_cash") == 0.0)
    )
    if unexplained_jumps.height > 0:
        sample = unexplained_jumps.select(["ticker", "period_ts", "price_return"]).head(3)
        records = sample.to_dicts()
        errors.append(
            f"Found {unexplained_jumps.height} unexplained extreme return jump(s) "
            f"(|price_return| > {max_unexplained_abs_return:.0%}) with no corporate action: "
            f"{records}"
        )

    # 8. Check stale repeated close streaks (consecutive identical close prices with volume > 0)
    for ticker_tuple, group in enriched.group_by("ticker", maintain_order=True):
        tkr = str(ticker_tuple[0])
        closes = group["close"].to_list()
        vols = group["volume"].to_list()
        streak = 0
        max_seen = 0
        for idx in range(1, len(closes)):
            if closes[idx] == closes[idx - 1] and vols[idx] > 0.0:
                streak += 1
                max_seen = max(max_seen, streak)
            else:
                streak = 0
        if max_seen >= max_stale_streak:
            warnings.append(
                f"Ticker {tkr!r} has {max_seen} consecutive unchanged close transitions."
            )

    return ValidationReport(
        is_valid=len(errors) == 0,
        row_count=row_count,
        ticker_count=ticker_count,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
