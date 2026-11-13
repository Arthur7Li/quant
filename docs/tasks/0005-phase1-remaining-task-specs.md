# Task: 0005 — Task Specs for Remainder of Phase 1 (§1.2 Notebooks 02–12 & §1.4 Modules)

## Goal
Decompose the remainder of Phase 1 (`QUANT_PROJECT_DESIGN.md` §1.2 Notebooks 02–12, §1.4 modules, and §8 Item 9) into self-contained, actionable task specifications (`0006` through `0012`) adhering to [TEMPLATE.md](TEMPLATE.md), complete with mathematical formulas, public APIs, known-answer/property/leakage acceptance tests, and 8 GB M2 memory constraints.

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.1–§1.6, §6.2–§6.3, §8 (Item 9)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §1–§5

## Inputs
- `docs/tasks/TEMPLATE.md`
- `QUANT_PROJECT_DESIGN.md` §1.2 curriculum table (Notebooks 02–12) and §1.4 Phase 1 module specs
- Existing foundations (`0000`–`0004`): `quantlab.compute`, `quantlab.data`, `quantlab.core`, `quantlab.validation.leakage`

## Outputs (Files & APIs)
- `docs/tasks/0005-phase1-remaining-task-specs.md`
- `docs/tasks/0006-phase1-stylized-facts-and-monte-carlo.md` (Notebooks 02–03 + `pricing/monte_carlo.py`)
- `docs/tasks/0007-phase1-option-pricing-and-greeks.md` (Notebooks 04–05 + `pricing/{black_scholes,binomial,greeks}.py`)
- `docs/tasks/0008-phase1-risk-and-portfolio-theory.md` (Notebooks 06–07 + `risk/measures.py` + `portfolio/{baselines,markowitz}.py`)
- `docs/tasks/0009-phase1-factor-models-and-signals.md` (Notebooks 08–09 + `models/{factors,signals}.py`)
- `docs/tasks/0010-phase1-event-driven-backtester.md` (Notebook 10 + `backtest/engine.py`)
- `docs/tasks/0011-phase1-limit-order-book.md` (Notebook 11 + `backtest/orderbook.py`)
- `docs/tasks/0012-phase1-capstone-why-most-backtests-lie.md` (Notebook 12 + `experiments/ledger.py` + Phase 1 → 2 gate)

## Acceptance Tests (Concrete, Runnable)
- [x] All seven child specs (`0006`–`0012`) exist in `docs/tasks/` and include Goal, Context/Links, Inputs, Outputs, Acceptance Tests (known-answer, property, and leakage tests where time-dependent), Out of Scope, Risks/Assumptions, and Phase & Rigor Level.
- [x] Every module listed in `QUANT_PROJECT_DESIGN.md` §1.4 and every notebook 02–12 in §1.2 is mapped with zero gaps.
- [x] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Implementing the Python modules and notebooks described inside `0006`–`0012` (those execute in dedicated tasks post Phase 1 Kickoff).
- Phase 2 production flagship modules (`flagship/`).

## Risks / Assumptions to Verify
- Spec signatures remain consistent with `docs/conventions.md` (`(T, N)` `float64` arrays, `signal[t] -> fill[t+1]`, `252` `XNYS` annualization) so Phase 1 → Phase 2 transition requires zero interface rewrites.

## Phase & Rigor Level
- **Phase:** Phase 1 (Kickoff Planning Milestone — §8 Item 9)
- **Coverage target:** N/A (documentation specs); existing `src/` suite remains ≥ 90%
- **Type checking:** `mypy --strict`
