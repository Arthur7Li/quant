# Task: 0000 — Phase 0 Repository Bootstrap

## Goal
Initialize the `quant-lab` repository skeleton, Python 3.12 `uv` environment, static analysis (`ruff`, `mypy`), test runner (`pytest`), pre-commit guardrails, GitHub Actions CI, governance documentation, and the device/memory utility module (`quantlab.compute`).

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §3, §5, §6.1, §8 (Items 1–4)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)

## Inputs
- Finalized design specification (`QUANT_PROJECT_DESIGN.md` v0.2)
- Hardware targets: macOS Apple M2 (8 GB Unified Memory) and Windows WSL2 RTX 3080 Ti (12 GB VRAM)

## Outputs (Files & APIs)
- `pyproject.toml`, `uv.lock`, `.gitignore`, `.pre-commit-config.yaml`
- `.github/workflows/ci.yml`
- `AGENTS.md`, `README.md`, `docs/DESIGN.md`, `docs/conventions.md`, `docs/glossary.md`, `docs/risk_register.md`, `docs/agent_scorecard.md`, `docs/research_log.md`
- `experiments/ledger.jsonl` (initialized empty ledger)
- `src/quantlab/__init__.py`, `src/quantlab/py.typed`, `src/quantlab/compute.py`, and subpackage `__init__.py` stubs
- `tests/test_compute.py`

## Acceptance Tests (Concrete, Runnable)
- [x] `quantlab.compute.get_device()` selects `cuda` → `mps` → `cpu` deterministically and falls back cleanly when PyTorch is absent or overridden.
- [x] `quantlab.compute.check_array_memory()` enforces memory caps for the 8 GB M2 guardrail.
- [x] `quantlab.compute.set_global_seed()` seeds `random` and `numpy` reproducibly.
- [x] Passes `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`.

## Out of Scope
- Phase 1 market data providers (`data/providers/`), return/metric calculations (`core/`), and pricing/backtesting logic (covered in Tasks 0001+).

## Risks / Assumptions to Verify
- Verify that `uv` and Python 3.12 run cleanly on macOS M2 and that `pytest`, `mypy`, and `ruff` pass with zero warnings.

## Phase & Rigor Level
- **Phase:** Phase 0 (Bootstrap)
- **Coverage target:** ≥ 90% on `src/quantlab/compute.py`
- **Type checking:** `mypy --strict`
