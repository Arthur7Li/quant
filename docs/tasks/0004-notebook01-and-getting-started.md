# Task: 0004 — Notebook 01 (`01_markets_prices_returns.ipynb`) & `README.md` "Getting Started"

## Goal
Create the first Phase 1 curriculum notebook ([01_markets_prices_returns.ipynb](../../notebooks/phase1/01_markets_prices_returns.ipynb)), an automated top-to-bottom notebook execution test ([tests/test_notebooks.py](../../tests/test_notebooks.py)), and a complete onboarding "Getting Started" guide in [README.md](../../README.md) (`QUANT_PROJECT_DESIGN.md` §1.2 Notebook 01 & §8 Item 8).

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.1, §1.2 (Notebook 01), §8 (Item 8)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §1–§5

## Inputs
- `quantlab.data`: `TIER1_US_ETFS`, `date_to_period_ts`, `us_equity_close_available_ts`, `make_toy_5bar_fixture`, `SyntheticDataProvider`, `compute_total_returns`, `compute_total_return_index`, `validate_bars`, `PointInTimeStore`, `compute_dataframe_hash`
- `quantlab.core`: `simple_to_log_returns`, `log_to_simple_returns`, `compound_returns`, `portfolio_returns`, `summarize_performance`
- `quantlab.validation`: `evaluate_signal_lag_and_shift_probe`, `shift_signals`

## Outputs (Files & APIs)
- `docs/tasks/0004-notebook01-and-getting-started.md`
- `notebooks/phase1/01_markets_prices_returns.ipynb` (executed top-to-bottom with verified cell outputs)
- `tests/test_notebooks.py` (validates nbformat v4 schema, populated code-cell outputs, and sequential cell execution)
- `README.md` (updated with "Getting Started", environment setup, quality gates, and repository map)

## Acceptance Tests (Concrete, Runnable)
- [x] `notebooks/phase1/01_markets_prices_returns.ipynb` follows `concept -> CS analogy -> math -> code -> exercise -> pitfall` and has all code cells pre-executed cleanly.
- [x] Code demonstrates EST/EDT dual timestamps (`period_ts`, `available_ts`), raw vs. corporate-action-adjusted returns on `make_toy_5bar_fixture`, PIT Parquet queries on synthetic Tier-1 US ETFs, Jensen's inequality error when averaging log returns across assets, and `lag=0` vs. `lag=+1` Sharpe inflation via `evaluate_signal_lag_and_shift_probe`.
- [x] `tests/test_notebooks.py` executes every code cell sequentially inside CI within the 8 GB M2 memory budget with zero errors.
- [x] `README.md` documents prerequisites, `uv sync`, optional `.env` variables, quality/test commands, notebook execution, and Phase 1 links.
- [x] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Live/network market vendor fetches in CI tests (all CI notebook tests use `SyntheticDataProvider` and `make_toy_5bar_fixture`).
- Phase 1 Notebooks 02–12 (specified in Task 0005 and implemented in subsequent tasks).

## Risks / Assumptions to Verify
- `jupyterlab` / `nbconvert` is an optional extra (`[project.optional-dependencies].notebooks`), so generating and testing `.ipynb` files in the base `dev` environment should use Python's stdlib `json` + `exec` + `contextlib.redirect_stdout` so CI (`--group dev`) stays dependency-light and fast.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations — Kickoff Task)
- **Coverage target:** ≥ 90% on `src/` overall; top-to-bottom notebook cell execution tested in CI
- **Type checking:** `mypy --strict`
