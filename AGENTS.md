# AGENTS.md — AI-Agent Scaffolding Contract (`quant-lab`)

This file defines the mandatory operating contract for all AI agents working in this repository. It is derived from [QUANT_PROJECT_DESIGN.md](QUANT_PROJECT_DESIGN.md) §5 and [ADR-0001](docs/adr/0001-initial-architecture-decisions.md).

---

## 1. Standing Rules for Every Agent Task

1. **Work from a task spec.** No task starts without a file in `docs/tasks/` (using `docs/tasks/TEMPLATE.md`) stating goal, inputs, outputs, acceptance tests, and out-of-scope items.
2. **Small, reviewable PRs / checkpoints.** One concern per task/PR. No drive-by refactors.
3. **Tests first for numerics.** Any numerical routine requires:
   - A known-answer test (closed-form or hand-computed fixture),
   - A property test (`hypothesis` or mathematical invariant such as put-call parity, PSD covariance, weights summing to 1), and
   - A convergence or accuracy test where applicable.
4. **No look-ahead.** Any code touching time series must obey the dual-timestamp (`period_ts`, `available_ts`) and `signal[t] -> fill[t+1]` conventions in `docs/conventions.md` and pass the leakage test suite (`src/quantlab/validation/leakage.py`).
5. **No silent assumptions.** Assumptions, units, conventions (log vs. simple returns, annualization factor = 252 `XNYS` trading days, decimal returns, UTC timestamps) must be documented in docstrings and follow `docs/conventions.md`.
6. **Never fabricate results.** Reported numbers must come from committed code that reproduces them. Agents must never write unverified result numbers into docs by hand.
7. **Experiment ledger is mandatory.** Every strategy or model variant evaluated must be logged to `experiments/ledger.jsonl` (metadata & metrics) and `experiments/trial_returns.parquet` (out-of-sample return series), including failed trials.
8. **No secrets or vendor data in the repo.** API keys must be read from environment variables / `.env` (gitignored). Market data lives in `data/` (gitignored). CI tests must use deterministic synthetic data (`quantlab.data.synthetic`) or tiny hand-computed fixtures in `tests/fixtures/`.
9. **Hardware & 8 GB M2 memory discipline.** Primary development runs on an Apple M2 with **8 GB Unified Memory**. All device selection goes through `quantlab.compute.get_device()`. Every experiment must support a `--smoke` flag that runs on CPU within the M2 memory budget (`quantlab.compute.check_array_memory()`).
10. **Human gate.** Merging to `main`, adding dependencies in `pyproject.toml`, modifying `src/quantlab/validation/`, and any code touching live/paper broker connectivity require explicit human approval.
11. **Uncertainty reporting.** At the end of every task, agents must explicitly state **what was verified** (with command outputs) versus **what was assumed**.

---

## 2. Definition of Done (All Phases)

- [ ] Code passes `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src tests`, and `uv run pytest`
- [ ] New logic has unit tests; numerical code has known-answer + property tests
- [ ] Time-dependent logic passes leakage tests (`future_poison`, `shift_probe`, `in_fold_preprocessing`)
- [ ] Runs comfortably within the 8 GB M2 memory budget
- [ ] Docstrings and `docs/conventions.md` respected
- [ ] Experiment ledger (`experiments/ledger.jsonl` + `experiments/trial_returns.parquet`) updated if any strategy/model was evaluated
- [ ] Reproducible with a single documented command
- [ ] All acceptance criteria in the corresponding `docs/tasks/<task>.md` spec checked off

---

## 3. Phase-Scaled Rigor

| Requirement | Phase 1 (Foundations) | Phase 2 (Production Flagship) | Phase 3 (Research Frontier) |
|---|---|---|---|
| **Test coverage (`src/`)** | ≥ 70% | ≥ 85% | ≥ 85% core; experimental flagged |
| **Type checking** | `mypy` (strict on completed modules) | `mypy --strict` across `src/` | `mypy --strict` on core |
| **Validation** | Chronological train/test split + demonstrated pitfalls | Cross-sectionally synchronized Purged K-Fold, CPCV, Deflated Sharpe ($\hat{N}_{\text{eff}}$), PBO | Automated gatekeeper blocks any claim failing validation |
| **Review** | Human reviews all tasks | Human reviews design + numerics; agent cross-review | Agent adversarial review + human spot-audit |
| **Reproducibility** | Notebook reruns + `uv.lock` | One-command pipeline + Docker | Versioned experiments, deterministic seeds, full provenance |

---

## 4. Explicitly Forbidden Anti-Patterns

- Tuning hyperparameters on the test set or full sample, then reporting in-sample or leaked results.
- Pre-computing stateful transforms (PCA, Ledoit-Wolf shrinkage, RMT denoising, volatility scaling, fractional differentiation) on the full dataset before CV splitting.
- Splitting multi-asset panels independently per ticker rather than synchronizing train/test/purge/embargo timestamps across the entire US equity cross-section.
- Reporting Sharpe ratios without transaction costs or without logging the trial return vector to `experiments/trial_returns.parquet`.
- "Fixing" a failing leakage, accounting, or statistical validation test by weakening the test assertion.
- Adding a heavyweight dependency to avoid writing 20 lines of clear NumPy/SciPy code.
- Describing an unexecuted code path as verified.
