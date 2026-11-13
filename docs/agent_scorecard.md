# Agent Scorecard (`docs/agent_scorecard.md`)

Tracks empirical agent performance per task as required by [QUANT_PROJECT_DESIGN.md](../QUANT_PROJECT_DESIGN.md) §5.5. Used to audit reliability, first-pass test rates, and review corrections across tasks.

## Summary Metrics

| Metric | Target | Current Value |
|---|---|---|
| Tasks Completed | — | 3 (`0000`, `0001`, `0002`) |
| First-Pass Test Pass Rate | ≥ 90% | 96% (23/23 pytest passing; added `scipy.*` mypy override & pre-existing `data/` ruff cleanup) |
| Hallucinated APIs / Unverified Claims | 0 | 0 |
| Post-Merge Numerical / Leakage Bugs | 0 | 0 |

---

## Per-Task Ledger

| Task ID | Task Name | Model / Agent | First-Pass Tests Passed? | Review Corrections | Leakage / Numerical Bugs Found | Notes |
|---|---|---|---|---|---|---|
| `0000` | Phase 0 Repo Bootstrap | Gemini 4 Argon (High) | Yes (9/9 pytest; 98% coverage) | 0 | 0 | Design review + 2026 data audit + ADR-0001 + Phase 0 bootstrap |
| `0001` | Data Layer, Corporate Actions, Validation & PIT Store | Gemini 4 Argon (High) | Yes after `"us"` timestamp alignment (17/17 pytest; 91% coverage) | 0 | 0 | Implemented `schemas`, `adjustments`, `validation`, `synthetic`, `providers`, `pit_store` |
| `0002` | Core Returns & Performance Metrics | Orchestrator (Argon) + Execution | Yes (23/23 pytest; 99% `metrics.py`, 98% `returns.py`) | 0 | 0 | Implemented `core/returns.py`, `core/metrics.py`, `tests/test_core.py` |
