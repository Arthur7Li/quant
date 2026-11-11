# Agent Scorecard (`docs/agent_scorecard.md`)

Tracks empirical agent performance per task as required by [QUANT_PROJECT_DESIGN.md](../QUANT_PROJECT_DESIGN.md) §5.5. Used to audit reliability, first-pass test rates, and review corrections across tasks.

## Summary Metrics

| Metric | Target | Current Value |
|---|---|---|
| Tasks Completed | — | 1 (`0000`) |
| First-Pass Test Pass Rate | ≥ 90% | 100% (9/9 pytest passed on first run; 1 mypy re-export adjustment in test file) |
| Hallucinated APIs / Unverified Claims | 0 | 0 |
| Post-Merge Numerical / Leakage Bugs | 0 | 0 |

---

## Per-Task Ledger

| Task ID | Task Name | Model / Agent | First-Pass Tests Passed? | Review Corrections | Leakage / Numerical Bugs Found | Notes |
|---|---|---|---|---|---|---|
| `0000` | Phase 0 Repo Bootstrap | Gemini 4 Argon (High) | Yes (9/9 pytest; 98% coverage) | 0 | 0 | Design review + 2026 data audit + ADR-0001 + Phase 0 bootstrap |
