# Task: 0012 — Experiment Ledger & Capstone Report ("Why Most Backtests Lie", Notebook 12)

## Goal
Implement the Git-tracked two-file experiment ledger utility (`src/quantlab/experiments/ledger.py` managing `experiments/ledger.jsonl` and `experiments/trial_returns.parquet`) and deliver Phase 1 Capstone Notebook 12 (`12_capstone_why_most_backtests_lie.ipynb`) demonstrating deliberate backtest overfitting on synthetic/US ETF data and exposing it via OOS decay, cost erosion, survivorship bias, and random-signal Sharpe baselines, satisfying the §1.5 & §1.6 Phase 1 → 2 gate.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.1–§1.2 (Notebook 12), §1.5, §1.6 (Phase 1 → 2 Gate), §6.3 (Two-File Experiment Ledger)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §4–§5

## Inputs
- `quantlab.data`: `SyntheticDataProvider`, `PointInTimeStore`, `TIER1_US_ETFS`
- `quantlab.core`: `summarize_performance`, `sharpe_ratio`, `max_drawdown`
- `quantlab.backtest.engine`: `run_backtest_total_return`
- `quantlab.validation.leakage`: `random_signal_sharpe_baseline`, `check_future_poison`, `evaluate_signal_lag_and_shift_probe`

## Outputs (Files & APIs)
- `src/quantlab/experiments/ledger.py` & `src/quantlab/experiments/__init__.py`:
  - `TrialRecord` Pydantic / frozen dataclass matching §6.3 (`trial_id`, `timestamp_utc`, `hypothesis_id`, `git_sha`, `config_hash`, `dataset_hash`, `start_date`, `end_date`, `is_sharpe`, `oos_gross_sharpe`, `oos_net_sharpe`, `annual_turnover`, `max_drawdown`, `skewness`, `kurtosis`, `status`)
  - `append_experiment_trial(record, oos_returns, period_ts, *, ledger_path="experiments/ledger.jsonl", returns_path="experiments/trial_returns.parquet") -> None`
  - `load_trial_return_matrix(...) -> tuple[npt.NDArray[np.float64], list[datetime], list[str]]` returning `(T, K)` aligned OOS trial returns cleanly for Phase 2 DSR / PBO
- `tests/test_experiments_ledger.py`
- `notebooks/phase1/12_capstone_why_most_backtests_lie.ipynb` (registered in `tests/test_notebooks.py`)
- Phase 1 → 2 gate artifacts (§1.6): backtester cross-check note and "what I understand / what I'm unsure about" note in `docs/research_log.md`

## Acceptance Tests (Concrete, Runnable)
- [ ] **Experiment ledger tests:**
  - Appending multiple synthetic trials writes atomic valid JSON Lines to a temporary `ledger.jsonl` and aligned `(trial_id, period_ts, net_return)` rows to `trial_returns.parquet`; `load_trial_return_matrix` reconstructs the exact `(T, K)` `np.float64` matrix.
  - Failed/negative trials are logged identically to successful trials without dropping negative results (`AGENTS.md` Rule 7).
- [ ] **Capstone report criteria (`QUANT_PROJECT_DESIGN.md` §1.5):**
  - Grid search over N strategy variants on US Tier-1 synthetic/ETF panels logs all trials to a temporary test ledger inside CI tests, demonstrating: (1) sharp in-sample vs. out-of-sample Sharpe decay of the "winning" parameter tuple, (2) transaction cost erosion across $0\times \to 5\times$ cost multipliers, (3) quantified single-stock survivorship bias inflation, and (4) the winner lying inside the `random_signal_sharpe_baseline` null distribution.
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Computing Phase 2 Deflated Sharpe Ratio ($\hat{N}_{\text{eff}}$) and CSCV Probability of Backtest Overfitting (PBO) (Phase 2 Layer 1–2 validation tasks).

## Risks / Assumptions to Verify
- CI tests must write trial artifacts to `tmp_path` rather than mutating tracked `experiments/ledger.jsonl` during unit tests; production/capstone runs commit intentional ledger updates explicitly.

## Phase & Rigor Level
- **Phase:** Phase 1 (Capstone & Phase 1 → 2 Gate)
- **Coverage target:** ≥ 90% on `src/quantlab/experiments/`; ≥ 70% overall across `src/` (§1.5 criterion)
- **Type checking:** `mypy --strict`
