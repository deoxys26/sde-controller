"""Metric and persistence-baseline helpers for link QoS forecasting."""

from __future__ import annotations

import numpy as np
import torch


def regression_metrics(prediction, target, target_names):
    prediction, target = np.asarray(prediction), np.asarray(target)
    result = {}
    for index, name in enumerate(target_names):
        error = prediction[..., index] - target[..., index]
        result[name] = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
    error = prediction - target
    result["aggregate"] = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
    return result


@torch.no_grad()
def evaluate_model(model, dataset, indices):
    model.eval()
    predicted, observed, persistence = [], [], []
    for index in indices:
        sequence, target, _ = dataset[index]
        predicted.append(model(sequence).cpu().numpy())
        observed.append(target.numpy())
        persistence.append(sequence[-1].edge_attr[:, dataset.target_indices].numpy())
    if not predicted:
        return {}, {}
    predicted = dataset.inverse_targets(np.concatenate(predicted))
    observed = dataset.inverse_targets(np.concatenate(observed))
    persistence = dataset.inverse_targets(np.concatenate(persistence))
    return regression_metrics(predicted, observed, dataset.target_columns), regression_metrics(persistence, observed, dataset.target_columns)
