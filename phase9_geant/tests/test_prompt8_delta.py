import json
from pathlib import Path

import numpy as np
import torch

from phase9_geant.run_delta import (
    fit_residual_scaler,
    last_observed,
    predict_batches,
    reconstruct_utilization,
    residual_targets,
)
from phase9_geant.run_neural import CONFIG, GeantUtilDataset, build_model, set_seed


ROOT = Path("data/processed/geant")
OUTPUT = ROOT / "neural_prompt8_delta"


def test_delta_target_is_next_value_minus_final_history_value():
    data = GeantUtilDataset(ROOT)
    starts = data.indices["train"][:4]
    actual = residual_targets(data, starts)
    expected = data.target(starts) - last_observed(data, starts)
    np.testing.assert_array_equal(actual, expected)


def test_reconstruction_adds_delta_to_last_observation():
    last = np.asarray([[0.2, 0.5], [0.8, 0.1]])
    delta = np.asarray([[0.03, -0.1], [0.4, -0.3]])
    reconstructed = reconstruct_utilization(last, delta)
    np.testing.assert_array_equal(reconstructed, last + delta)
    assert reconstructed[1, 0] > 1 and reconstructed[1, 1] < 0


def test_residual_scaler_fits_training_window_targets_only():
    data = GeantUtilDataset(ROOT)
    train_starts = data.indices["train"]
    mean, scale = fit_residual_scaler(data, train_starts)
    expected = residual_targets(data, train_starts)
    assert np.isclose(mean, expected.mean(dtype=np.float64))
    assert np.isclose(scale, expected.std(dtype=np.float64))


def test_validation_loss_selects_saved_checkpoint():
    payload = json.loads((OUTPUT / "results.json").read_text(encoding="utf-8"))
    assert [row["seed"] for row in payload["seed_results"]] == [42, 123, 2026]
    data = GeantUtilDataset(ROOT)
    residual_mean, residual_scale = fit_residual_scaler(data, data.indices["train"])
    torch.set_num_threads(1)
    loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
    for row in payload["seed_results"]:
        losses = [item["validation_loss_standardized_residual"] for item in row["history"]]
        selected = min(row["history"], key=lambda item: item["validation_loss_standardized_residual"])
        assert row["best_validation_loss_standardized_residual"] == min(losses)
        assert row["best_epoch"] == selected["epoch"]
        assert Path(row["checkpoint"]).is_file()
        model = build_model("D-U")
        model.load_state_dict(torch.load(row["checkpoint"], map_location="cpu", weights_only=True))
        model.eval()
        total, count = 0.0, 0
        starts_all = data.indices["val"]
        with torch.no_grad():
            for begin in range(0, len(starts_all), CONFIG["batch_size"]):
                starts = starts_all[begin:begin + CONFIG["batch_size"]]
                prediction = model(data.sequence(starts)).reshape(len(starts), len(data.edges), 1)
                target = torch.as_tensor(
                    ((residual_targets(data, starts) - residual_mean) / residual_scale)[:, :, None],
                    dtype=torch.float32,
                )
                total += float(loss_fn(prediction, target)) * target.numel()
                count += target.numel()
        assert np.isclose(
            total / count,
            row["best_validation_loss_standardized_residual"],
            rtol=1e-6,
            atol=1e-8,
        )


def test_test_target_change_does_not_change_residual_forecast_or_scaler():
    data = GeantUtilDataset(ROOT)
    residual_mean, residual_scale = fit_residual_scaler(data, data.indices["train"])
    test_start = int(data.indices["test"][0])
    target_index = test_start + 5
    original = data.util[target_index].copy()
    set_seed(91)
    model = build_model("D-U")
    before = predict_batches(model, data, [test_start], residual_mean, residual_scale)
    try:
        data.util[target_index] = original + 10.0
        after = predict_batches(model, data, [test_start], residual_mean, residual_scale)
        assert fit_residual_scaler(data, data.indices["train"]) == (residual_mean, residual_scale)
        np.testing.assert_array_equal(before, after)
    finally:
        data.util[target_index] = original


def test_prompt8_saved_predictions_are_aligned_and_finite():
    predictions = np.load(OUTPUT / "predictions.npz")
    baselines = np.load(ROOT / "baseline_predictions.npz")
    np.testing.assert_array_equal(predictions["test_target_indices"], baselines["distance_test_target_ts"])
    np.testing.assert_array_equal(predictions["test_targets"], baselines["distance_test_y"])
    for seed in (42, 123, 2026):
        values = predictions[f"test_prediction_seed{seed}"]
        assert values.shape == predictions["test_targets"].shape
        assert np.isfinite(values).all()
