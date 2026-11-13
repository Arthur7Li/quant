"""Point-in-Time (PIT) Parquet store with DuckDB/Polars out-of-core queries and content hashing.

Implements `QUANT_PROJECT_DESIGN.md` §3, §4 Rules 2–3, and `docs/conventions.md` §1, §5:
- Raw OHLCV bars and corporate actions are persisted as Parquet files alongside a
  deterministic SHA-256 content manifest (`manifest.json`).
- All queries require an explicit `as_of_ts` (UTC timestamp) and push down the
  filter `available_ts <= as_of_ts` so future bars and future corporate actions
  are strictly invisible.
- Bridges directly to the `(T, N)` typed `float64` NumPy array boundary via
  `to_return_panel()`.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt
import polars as pl

from quantlab.compute import check_array_memory
from quantlab.data.adjustments import compute_total_returns
from quantlab.data.schemas import (
    ACTIONS_SCHEMA,
    BARS_SCHEMA,
    empty_actions_frame,
    empty_bars_frame,
    ensure_schema,
    normalize_tickers,
)
from quantlab.data.validation import validate_bars


def compute_dataframe_hash(df: pl.DataFrame, sort_keys: Sequence[str]) -> str:
    """Compute a deterministic SHA-256 hex digest of a Polars DataFrame.

    Sorts `df` by `sort_keys` and serializes to canonical CSV bytes in memory
    so the digest is independent of row arrival order or Parquet metadata headers.
    """
    sorted_df = df.sort(list(sort_keys)) if df.height > 0 else df
    buf = io.BytesIO()
    sorted_df.write_csv(buf, float_precision=12)
    return hashlib.sha256(buf.getvalue()).hexdigest()


class PointInTimeStore:
    """Local Parquet-backed Point-in-Time store with SHA-256 content versioning."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir)
        self.bars_path = self.root_dir / "bars.parquet"
        self.actions_path = self.root_dir / "actions.parquet"
        self.manifest_path = self.root_dir / "manifest.json"

    def ingest(
        self,
        bars: pl.DataFrame,
        actions: pl.DataFrame | None = None,
        *,
        validate: bool = True,
    ) -> str:
        """Validate, deduplicate, and persist `bars` and `actions` to Parquet.

        Parameters
        ----------
        bars : pl.DataFrame
            Daily OHLCV bars conforming to `BARS_SCHEMA`.
        actions : pl.DataFrame | None, optional
            Corporate actions conforming to `ACTIONS_SCHEMA`.
        validate : bool, default True
            Run `validate_bars` before persisting and raise if invalid.

        Returns
        -------
        str
            Combined SHA-256 content hash of the stored dataset.
        """
        ensure_schema(bars, BARS_SCHEMA, name="bars")
        clean_actions = actions if actions is not None else empty_actions_frame()
        ensure_schema(clean_actions, ACTIONS_SCHEMA, name="actions")

        if self.bars_path.exists():
            existing_bars = pl.read_parquet(self.bars_path)
            bars = (
                pl.concat([existing_bars, bars])
                .unique(subset=["ticker", "period_ts"], keep="last")
                .sort(["ticker", "period_ts"])
            )
        else:
            bars = (
                bars.unique(subset=["ticker", "period_ts"], keep="last").sort(
                    ["ticker", "period_ts"]
                )
            )

        if self.actions_path.exists():
            existing_actions = pl.read_parquet(self.actions_path)
            clean_actions = (
                pl.concat([existing_actions, clean_actions])
                .unique(subset=["ticker", "ex_date", "source"], keep="last")
                .sort(["ticker", "ex_date"])
            )
        else:
            clean_actions = (
                clean_actions.unique(subset=["ticker", "ex_date", "source"], keep="last").sort(
                    ["ticker", "ex_date"]
                )
            )

        if validate:
            report = validate_bars(bars, actions=clean_actions)
            report.raise_if_invalid()

        self.root_dir.mkdir(parents=True, exist_ok=True)
        bars.write_parquet(self.bars_path)
        clean_actions.write_parquet(self.actions_path)

        bars_hash = compute_dataframe_hash(bars, sort_keys=["ticker", "period_ts"])
        actions_hash = compute_dataframe_hash(clean_actions, sort_keys=["ticker", "ex_date"])
        combined_hash = hashlib.sha256(f"{bars_hash}:{actions_hash}".encode()).hexdigest()

        manifest: dict[str, Any] = {
            "dataset_hash": combined_hash,
            "bars_hash": bars_hash,
            "actions_hash": actions_hash,
            "bar_rows": bars.height,
            "action_rows": clean_actions.height,
            "tickers": sorted(bars["ticker"].unique().to_list()) if bars.height > 0 else [],
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return combined_hash

    def get_dataset_hash(self) -> str:
        """Return the current SHA-256 dataset hash from `manifest.json`."""
        if not self.manifest_path.exists():
            msg = f"No PIT manifest found at {self.manifest_path}."
            raise FileNotFoundError(msg)
        data: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return str(data["dataset_hash"])

    def query_bars(
        self,
        as_of_ts: datetime,
        *,
        tickers: Sequence[str] | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ) -> pl.DataFrame:
        """Query bars strictly knowable at `as_of_ts` (`available_ts <= as_of_ts`).

        Uses DuckDB out-of-core Parquet execution so only matching rows enter memory.
        """
        if as_of_ts.tzinfo is None:
            msg = "as_of_ts must be timezone-aware (UTC)."
            raise ValueError(msg)
        if not self.bars_path.exists():
            return empty_bars_frame()

        clauses = ["available_ts <= $as_of_ts"]
        params: dict[str, Any] = {"as_of_ts": as_of_ts}

        if tickers is not None:
            clean_tickers = normalize_tickers(tickers)
            if not clean_tickers:
                return empty_bars_frame()
            clauses.append("ticker IN (SELECT UNNEST($tickers))")
            params["tickers"] = clean_tickers

        if start_ts is not None:
            clauses.append("period_ts >= $start_ts")
            params["start_ts"] = start_ts

        if end_ts is not None:
            clauses.append("period_ts <= $end_ts")
            params["end_ts"] = end_ts

        where_sql = " AND ".join(clauses)
        parquet_file = str(self.bars_path)
        query = (
            f"SELECT ticker, period_ts, available_ts, open, high, low, close, volume, source "  # noqa: S608
            f"FROM read_parquet({parquet_file!r}) "
            f"WHERE {where_sql} "
            f"ORDER BY ticker, period_ts"
        )

        with duckdb.connect(":memory:") as con:
            arrow_tbl = con.sql(query, params=params).to_arrow_table()
        df_any: Any = pl.from_arrow(arrow_tbl)
        assert isinstance(df_any, pl.DataFrame)
        if df_any.is_empty():
            return empty_bars_frame()
        return df_any.cast(BARS_SCHEMA)

    def query_actions(
        self,
        as_of_ts: datetime,
        *,
        tickers: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        """Query corporate actions strictly knowable at `as_of_ts` (`available_ts <= as_of_ts`)."""
        if as_of_ts.tzinfo is None:
            msg = "as_of_ts must be timezone-aware (UTC)."
            raise ValueError(msg)
        if not self.actions_path.exists():
            return empty_actions_frame()

        lazy_q = pl.scan_parquet(self.actions_path).filter(pl.col("available_ts") <= as_of_ts)
        if tickers is not None:
            clean_tickers = normalize_tickers(tickers)
            if not clean_tickers:
                return empty_actions_frame()
            lazy_q = lazy_q.filter(pl.col("ticker").is_in(clean_tickers))

        result = lazy_q.sort(["ticker", "ex_date"]).collect()
        if result.is_empty():
            return empty_actions_frame()
        return result.cast(ACTIONS_SCHEMA)

    def to_return_panel(
        self,
        as_of_ts: datetime,
        *,
        tickers: Sequence[str] | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        use_total_return: bool = True,
    ) -> tuple[npt.NDArray[np.float64], list[datetime], list[str]]:
        """Build a `(T, N)` `float64` return panel strictly as of `as_of_ts`.

        Parameters
        ----------
        as_of_ts : datetime
            Point-in-time cutoff (`available_ts <= as_of_ts`).
        tickers : Sequence[str] | None, optional
            Subset of tickers to include.
        start_ts, end_ts : datetime | None, optional
            Optional `period_ts` window bounds.
        use_total_return : bool, default True
            If True, uses split- and dividend-adjusted `total_return`; otherwise
            uses raw `price_return`.

        Returns
        -------
        tuple[npt.NDArray[np.float64], list[datetime], list[str]]
            `(returns_2d, period_timestamps, sorted_tickers)` where `returns_2d`
            has shape `(T, N)` and `dtype=np.float64` (dropping the initial `NaN`
            row if all assets start on the same date, or preserving `np.nan` where
            missing).
        """
        bars = self.query_bars(as_of_ts, tickers=tickers, start_ts=start_ts, end_ts=end_ts)
        if bars.is_empty():
            return np.empty((0, 0), dtype=np.float64), [], []

        actions = self.query_actions(as_of_ts, tickers=tickers)
        enriched = compute_total_returns(bars, actions=actions, as_of_ts=as_of_ts)
        ret_col = "total_return" if use_total_return else "price_return"

        sorted_tickers = sorted(enriched["ticker"].unique().to_list())
        n_dates = int(enriched["period_ts"].n_unique())
        check_array_memory((n_dates, len(sorted_tickers)), dtype=np.float64)

        pivoted = (
            enriched.select(["period_ts", "ticker", ret_col])
            .pivot(on="ticker", index="period_ts", values=ret_col)
            .sort("period_ts")
        )
        # Drop the very first row if all tickers are NaN on the initial price observation
        ordered_cols = ["period_ts", *sorted_tickers]
        pivoted = pivoted.select(ordered_cols)

        period_list: list[datetime] = pivoted["period_ts"].to_list()
        mat = pivoted.select(sorted_tickers).to_numpy().astype(np.float64, copy=False)

        if mat.shape[0] > 0 and bool(np.all(np.isnan(mat[0, :]))):
            mat = mat[1:, :]
            period_list = period_list[1:]

        return mat, period_list, sorted_tickers
