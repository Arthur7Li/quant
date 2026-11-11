# Task: <short name>

## Goal
State clearly what this task accomplishes in 1–3 sentences.

## Context / Links
- Design doc section: `QUANT_PROJECT_DESIGN.md` §...
- Related ADRs: `docs/adr/...`

## Inputs
- Existing modules, data schemas, or mathematical formulas relied upon.

## Outputs (Files & APIs)
- List of files to create or modify and public functions/classes exposed.

## Acceptance Tests (Concrete, Runnable)
- [ ] Known-answer tests:
- [ ] Property / edge-case tests:
- [ ] Leakage / sanity tests (if time-dependent):
- [ ] Passes `uv run ruff check .`, `uv run mypy src tests`, `uv run pytest`

## Out of Scope
- Items explicitly excluded from this task to prevent scope creep.

## Risks / Assumptions to Verify
- Numerical, statistical, or systems risks and how they are mitigated.

## Phase & Rigor Level
- **Phase:** Phase 0 / 1 / 2 / 3
- **Coverage target:** ≥ 70% (Phase 1) / ≥ 85% (Phase 2–3)
- **Type checking:** mypy strict
