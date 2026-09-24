"""Run the Model D controlled comparison experiment.

Compares Model B (current), Model C (proposed), and Model D (model_d) under
the identical Phase 4.5 protocol.  Artifacts go to data/experiments/model_d/.
Do NOT overwrite previous experiment directories.

Usage:
    python -m scripts.run_model_d_experiment
"""

import json
import sys
from pathlib import Path

# Allow running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.gnn.run_experiment import MODEL_D_MODELS, run_study

CONFIG_PATH = Path("configs/simulation.yaml")
OUTPUT_DIR = Path("data/experiments/model_d")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)["gnn_experiment"]

    print(f"Model D experiment: models={list(MODEL_D_MODELS)}, "
          f"seeds={config['seeds']}, scenarios=10")
    print(f"Output: {OUTPUT_DIR}/results.json")

    report = run_study(config, str(OUTPUT_DIR), models=list(MODEL_D_MODELS))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Written: {out}")
    return str(out)


if __name__ == "__main__":
    print(main())
