"""Local synthetic regression checks for notebook cell dependencies/order."""

from __future__ import annotations

import ast
import json
from pathlib import Path


def test_preparation_cell_imports_sys_and_allocates_a_fresh_stage():
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "geant_h96_colab_matrix3_recovery.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    preparation = next(
        cell for cell in cells
        if cell["cell_type"] == "code"
        and any("fresh staging session" in line for line in cell["source"])
    )
    source = "".join(preparation["source"])
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "sys" in imported
    assert "create_fresh_preparation_session" in source
    assert "reset_prior_staged_project_imports" in source
    assert "REPO.mkdir(exist_ok=False)" in source


def test_upload_cell_does_not_pin_reusable_preparation_directory():
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "geant_h96_colab_matrix3_recovery.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    upload = next(
        cell for cell in cells
        if cell["cell_type"] == "code"
        and any("files.upload()" in line for line in cell["source"])
    )
    source = "".join(upload["source"])
    assert "SESSION.mkdir(" not in source
    assert "SESSION_ID=uuid.uuid4().hex" not in source
