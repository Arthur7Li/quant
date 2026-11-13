"""Automated execution tests for Phase 1 curriculum Jupyter notebooks."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "phase1"
    / ("01_markets_prices_returns.ipynb")
)


def test_notebook_01_executes_cleanly() -> None:
    """Validate nbformat structure, stored outputs, and sequential cell execution."""
    assert NOTEBOOK_PATH.exists(), f"Missing notebook at {NOTEBOOK_PATH}"
    nb: dict[str, Any] = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert nb["nbformat"] == 4
    cells: list[dict[str, Any]] = nb["cells"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    assert len(markdown_cells) >= 4
    assert len(code_cells) >= 4

    namespace: dict[str, Any] = {}
    for idx, cell in enumerate(code_cells, start=1):
        assert cell.get("execution_count") is not None, f"Code cell {idx} was not pre-executed."
        assert cell.get("outputs"), f"Code cell {idx} has empty stored outputs."

        source = "".join(cell["source"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(source, f"{NOTEBOOK_PATH.name}#cell{idx}", "exec"), namespace)
        assert buf.getvalue().strip(), f"Code cell {idx} produced no stdout on re-execution."
