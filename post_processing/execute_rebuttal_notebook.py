"""Execute every code cell in the rebuttal notebook in one fresh Python process.

This lightweight verifier is useful in environments that have the analysis
dependencies but not the Jupyter/nbformat command-line packages installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def display(value: Any) -> None:
    """Provide the notebook display name during source-only verification."""
    print(value)


def main() -> None:
    notebook_path = Path(__file__).with_name("rebuttal_analysis.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {
        "__name__": "__rebuttal_notebook__",
        "display": display,
    }
    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    for index, cell in enumerate(code_cells, start=1):
        print(f"Executing code cell {index}/{len(code_cells)}")
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)
    print(f"Executed all {len(code_cells)} code cells successfully.")


if __name__ == "__main__":
    main()
