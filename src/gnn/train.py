"""Reproducible CPU-friendly training for the Phase 4 baseline."""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch

from src.gnn.evaluate import evaluate_model


def _save_checkpoint(state, checkpoint):
    """Retry transient OneDrive/antivirus file locks without changing training."""
    os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
    for attempt in range(5):
        try:
            torch.save(state, checkpoint)
            return
        except RuntimeError:
            if attempt == 4:
                raise
            time.sleep(.25 * (attempt + 1))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_model(model, dataset, config):
    set_seed(int(config.get("seed", 42)))
    # Callers may have instantiated the model before entering this function.
    # Reset its parameterized children after seeding so repeat runs are identical.
    for module in model.modules():
        if module is not model and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", .005)))
    criterion = torch.nn.SmoothL1Loss()
    train_indices, validation_indices = dataset.indices("train"), dataset.indices("validation")
    history, best_loss = [], float("inf")
    best_epoch, stale_epochs = 0, 0
    checkpoint = config.get("checkpoint_path", "data/generated/models/gnn_qos_best.pt")
    batch_size = max(1, int(config.get("batch_size", 4)))
    for epoch in range(int(config.get("epochs", 12))):
        model.train(); losses = []
        for start in range(0, len(train_indices), batch_size):
            batch = train_indices[start:start + batch_size]
            optimizer.zero_grad()
            batch_losses = [criterion(model(dataset[index][0]), dataset[index][1]) for index in batch]
            loss = torch.stack(batch_losses).mean()
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        validation_loss = 0.0
        if validation_indices:
            model.eval()
            with torch.no_grad():
                validation_loss = float(np.mean([criterion(model(dataset[i][0]), dataset[i][1]).item() for i in validation_indices]))
        selection_loss = validation_loss if validation_indices else float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
        if selection_loss < best_loss:
            best_loss = selection_loss
            _save_checkpoint(model.state_dict(), checkpoint)
            best_epoch, stale_epochs = epoch + 1, 0
        else:
            stale_epochs += 1
        patience = config.get("early_stopping_patience")
        if patience is not None and stale_epochs >= int(patience):
            break
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    gnn_metrics, persistence_metrics = evaluate_model(model, dataset, dataset.indices("test"))
    if history:
        history[-1]["best_validation_epoch"] = best_epoch
    return history, gnn_metrics, persistence_metrics
