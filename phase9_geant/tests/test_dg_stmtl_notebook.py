from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[2] / "notebooks" / "geant_dg_stmtl_h96_colab.ipynb"


def test_notebook_code_cells_parse_and_run_gate_defaults_closed():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 5
    for cell_source in code_cells:
        compile("".join(cell_source), str(NOTEBOOK), "exec")
    setup = "".join(code_cells[0])
    staging = "".join(code_cells[1])
    preflight = "".join(code_cells[2])
    training = "".join(code_cells[3])
    assert "TRAINING_ENABLED = False" in setup
    assert "SOURCE_MANIFEST_SHA256 = 'f5bdf0ef45089942195bd32d1277507023a545aefa129a5b95ee56454f2e67fc'" in setup
    assert "DATASET_METADATA_CANONICAL_SHA256 = '71e949092897d273aee39d2b0893ad48b31861b477808c98379f49190c4612d4'" in setup
    assert "files.upload()" in staging
    assert "safe_extract" in staging
    assert staging.index("safe_extract(role_bytes['source_zip']") < staging.index("safe_extract(role_bytes['dataset_zip']")
    assert "test_result = subprocess.run" in preflight
    assert preflight.index("test_result = subprocess.run") < preflight.index("experiment._verified_inputs(DATA_ROOT, PROJECT_ROOT)")
    assert "RESOURCE_POLICY_ID" in preflight and "PROCESS_PEAK_LIMIT_BYTES" in preflight
    assert "workflow.require_authorization(seed, response)" in training
    assert "backup_completed_run" in training
    assert training.index("workflow.require_authorization(seed, response)") < training.index("experiment.train_one")
    assert "drive.mount('/content/drive')" in preflight
    assert preflight.index("drive.mount('/content/drive')") < preflight.index("DRIVE_BACKUP_ROOT.exists()")
    assert "DRIVE_BACKUP_ROOT.parent.is_dir()" in preflight
    flattened = "".join("".join(cell_source) for cell_source in code_cells)
    assert "test" in flattened
    assert "best.pt" in staging or "best.pt" in training


def test_notebook_has_fixed_seed_list_and_no_auto_training_call():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "SEEDS = (42, 123, 2026)" in code
    assert "if TRAINING_ENABLED is not True:" in code
    assert "--run-training" not in code


def test_notebook_requires_exact_four_uploaded_hashes_and_new_session():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    setup, staging = code[0], code[1]
    for pin in ("SOURCE_ZIP_SHA256", "SOURCE_MANIFEST_SHA256", "DATASET_ZIP_SHA256", "DATASET_MANIFEST_SHA256"):
        assert pin in setup and pin in staging
    assert "uuid.uuid4().hex" in setup
    assert "SESSION_ROOT.mkdir(exist_ok=False)" in setup
    assert "hashlib.sha256(content).hexdigest()" in staging
    assert "if digest not in EXPECTED_UPLOADS" in staging
    assert "if role in role_bytes" in staging
    assert "DATA_NAMES" in staging and "window_index_validation_h96.json" in staging


def test_notebook_training_and_backup_destinations_refuse_reuse():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    preflight, training = code[2], code[3]
    assert "DRIVE_BACKUP_ROOT.exists()" in preflight
    assert "if (PROJECT_ROOT / 'data/processed' / experiment.OUTPUT_ROOT_NAME).exists()" in preflight
    assert training.index("workflow.require_authorization(seed, response)") < training.index("DRIVE_BACKUP_ROOT.mkdir(parents=False, exist_ok=False)")
    assert "backup_completed_run" in training
    assert "validate_completed_run" in training
    assert "Refusing existing" in preflight
