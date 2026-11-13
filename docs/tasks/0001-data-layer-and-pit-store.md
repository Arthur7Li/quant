# Task: 0001 — Data Layer, Corporate Actions, Validation Suite & Point-in-Time Parquet Store

## Goal
Implement the `quantlab.data` package (§4 & §8 Item 5): canonical dual-timestamp (`period_ts`, `available_ts`) schemas, the `DataProvider` protocol with deterministic synthetic (`SyntheticDataProvider`) and US equity vendor implementations (`TiingoProvider`, `StooqProvider`, `YFinanceProvider`), corporate action adjustments (raw OHLCV + splits/dividends $\to$ total returns & forward Total Return Index), automated ingestion validation (`validate_bars`), and the DuckDB/Polars-backed `PointInTimeStore` with SHA-256 content hashing.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §4, §6.1, §8 (Item 5)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §1, §2, §5

## Inputs
- `quantlab.compute.check_array_memory` and `quantlab.compute.set_global_seed`
- Conventions: `XNYS` (`16:00:00 America/New_York` close converted to UTC for `available_ts`), raw USD prices + `(split_factor, div_cash)` corporate actions table, `(T, N)` `float64` return panels.

## Outputs (Files & APIs)
- `src/quantlab/data/schemas.py`: `BARS_SCHEMA`, `ACTIONS_SCHEMA`, `TIER1_US_ETFS`, `us_equity_close_available_ts`, `validate_schema`
- `src/quantlab/data/providers.py`: `DataProvider` protocol, `DataProviderError`, `DataProviderQuotaError`, `TiingoProvider`, `StooqProvider`, `YFinanceProvider`
- `src/quantlab/data/synthetic.py`: `SyntheticDataProvider`, `make_toy_5bar_fixture`
- `src/quantlab/data/adjustments.py`: `compute_total_returns`, `compute_total_return_index`, `apply_split_adjustments`
- `src/quantlab/data/validation.py`: `ValidationReport`, `validate_bars`
- `src/quantlab/data/pit_store.py`: `PointInTimeStore`, `compute_dataframe_hash`
- `src/quantlab/data/__init__.py`: Public exports
- `tests/test_data.py`: Known-answer, property (`hypothesis`), PIT look-ahead isolation, vendor parser (mocked HTTP), and validation tests

## Acceptance Tests (Concrete, Runnable)
- [x] **Known-answer corporate action test:** On the 5-bar hand-computed fixture (`P_raw = [100, 110, 55, 54, 59.4]`, with a 2-for-1 split at bar 2 and a \$1.00 cash dividend at bar 3), `compute_total_returns` produces `[NaN, +0.10, 0.00, 0.00, +0.10]` to machine precision ($10^{-12}$), and forward `compute_total_return_index` produces `[100.0, 110.0, 110.0, 110.0, 121.0]`.
- [x] **Point-in-time immutability property test (`hypothesis`):** Appending future bars or future dividends/splits at $t > t_0$ never alters `compute_total_return_index` or `PointInTimeStore.query_bars(as_of_ts=t_0)` for any $t \le t_0$.
- [x] **Validation suite tests:** `validate_bars` passes on clean synthetic data and accurately catches negative prices, `high < low` violations, non-monotonic timestamps, `available_ts < period_ts`, unexplained $>50\%$ jumps without corporate actions, and stale price streaks.
- [x] **Vendor parser & quota tests:** `TiingoProvider` parses JSON payloads into raw bars + corporate actions; `StooqProvider` detects HTTP 200 `"Exceeded the daily hits limit"` responses and raises `DataProviderQuotaError`.
- [x] Passes `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src tests`, and `uv run pytest`.

## Out of Scope
- Live API key registration or committing downloaded vendor data (forbidden by Standing Rule 8).
- Log return / Sharpe / drawdown metrics (`core/returns.py` and `core/metrics.py`, covered in Task 0002 / §8 Item 6).

## Risks / Assumptions to Verify
- Verify US Eastern DST transitions (`America/New_York` $\to$ UTC `20:00Z` in EDT vs. `21:00Z` in EST) are handled deterministically via Python's standard `zoneinfo`.
- Verify DuckDB / Polars Parquet reads and writes preserve `datetime[ns, UTC]` timezones across platforms.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations — Kickoff Task)
- **Coverage target:** ≥ 90% on `src/quantlab/data/`
- **Type checking:** `mypy --strict`
