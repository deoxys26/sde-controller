import json
from pathlib import Path

import numpy as np
import torch

from phase9_geant.run_neural import GeantUtilDataset, build_model


ROOT = Path("data/processed/geant")


def test_prompt6_windows_are_chronological_and_block_local():
    data = GeantUtilDataset(ROOT)
    expected = {
        "train": (7344, 7, 8019),
        "val": (1715, 8025, 9739),
        "test": (1651, 9745, 11395),
    }
    for split, (count, first_target, last_target) in expected.items():
        starts = data.indices[split]
        block = data.split[split]
        assert len(starts) == count
        assert np.all(np.diff(starts) > 0)
        assert starts[0] + 5 == first_target
        assert starts[-1] + 5 == last_target
        assert np.all(starts >= block["start_index"])
        assert np.all(starts + 5 < block["end_index"])


def test_prompt6_scaler_uses_training_period_only():
    data = GeantUtilDataset(ROOT)
    train = data.split["train"]
    steps = np.arange(train["start_index"], train["end_index"])[
        data.valid[train["start_index"]:train["end_index"]]
    ]
    assert np.isclose(data.mean, data.util[steps].mean(dtype=np.float64))
    assert np.isclose(data.scale, data.util[steps].std(dtype=np.float64))


def test_future_target_change_does_not_change_historical_input():
    data = GeantUtilDataset(ROOT)
    start = int(data.indices["test"][0])
    target_index = start + 5
    before = [snapshot.edge_attr.clone() for snapshot in data.sequence([start])]
    original = data.util[target_index].copy()
    data.util[target_index] = original + 123.0
    after = data.sequence([start])
    assert all(torch.equal(old, new.edge_attr) for old, new in zip(before, after))
    assert not np.array_equal(data.target([start]), original[None, :])


def test_prompt6_results_cover_all_models_seeds_and_validation_selection():
    payload = json.loads((ROOT / "neural_prompt6" / "results.json").read_text())
    assert set(payload["models"]) == {"B-U", "C-U", "D-U"}
    for model in payload["models"].values():
        rows = model["seed_results"]
        assert [row["seed"] for row in rows] == [42, 123, 2026]
        assert all(row["best_validation_loss"] == min(
            item["validation_loss"] for item in row["history"]
        ) for row in rows)
        assert all(row["metrics"]["test"]["overall"]["mae"] >= 0 for row in rows)
    assert payload["baseline_comparison"]["blocks"]["test"]["methods"]["seasonal_naive_96"]["coverage"] == 1.0


def test_prompt6_model_output_shapes_and_finiteness():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    snapshots = []
    for step in range(5):
        from src.gnn.dataset import GraphSnapshot
        snapshots.append(GraphSnapshot(
            x=torch.ones((3, 1)), edge_index=edge_index,
            edge_attr=torch.randn((4, 1)), timestep=step,
        ))
    for name in ("B-U", "D-U"):
        output = build_model(name)(snapshots)
        assert output.shape == (4, 1)
        assert torch.isfinite(output).all()
    c_snapshots = [
        type(snapshots[0])(
            x=snapshots[0].x, edge_index=edge_index,
            edge_attr=torch.randn((2, 4, 1)), timestep=step,
        ) for step in range(5)
    ]
    output = build_model("C-U").forward_batched(c_snapshots)
    assert output.shape == (2, 4, 1)
    assert torch.isfinite(output).all()
