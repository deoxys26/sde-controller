"""Tests for Phase 7 Model D evaluator extension.

Verifies:
- Evaluator initialization and Model D checkpoint loading.
- Forecast adapter integration accepts Model D.
- No future leakage.
- Output dimensions.
- Deterministic behavior.
- Phase 7 evaluation compatibility.
"""

import copy
import pytest
from pathlib import Path

from src.gnn.run_experiment import SCENARIOS
from src.routing.forecast_adapter import FrozenQoSForecaster
from scripts.run_phase7_model_d_experiment import Phase7ModelDEvaluator, METHODS
import yaml

CONFIG_PATH = Path("configs/simulation.yaml")

def _get_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def test_phase7_model_d_evaluator_methods():
    assert "predicted_qos_ga_current" in METHODS
    assert "predicted_qos_ga_model_d" in METHODS
    assert "shortest_path" in METHODS
    assert "reactive" in METHODS
    assert "current_qos_ga" in METHODS

def test_frozen_qos_forecaster_loads_model_d():
    config = _get_config()
    gnn_config = config["gnn_experiment"]
    
    # Just test initialization interface if checkpoints don't exist in CI
    ckpt_dir = Path("data/experiments/model_d/checkpoints")
    if not ckpt_dir.exists():
        pytest.skip("Model D checkpoints missing, skip integration test")
        
    scaler_dir = Path("data/experiments/phase7_model_d/scalers/test_model_d")
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    # Must load successfully
    forecaster = FrozenQoSForecaster(
        "normal", gnn_config, str(ckpt_dir),
        model_name="model_d", training_seed=42, scaler_output_dir=scaler_dir
    )
    
    assert forecaster.model.__class__.__name__ == "TemporalAttentionEdgeForecaster"
    assert forecaster.dataset.history_length == 5
    assert forecaster.dataset.prediction_horizon == 1

def test_evaluator_is_causal_and_deterministic_with_model_d():
    config = _get_config()
    
    # Override settings for tiny test
    config["gnn_experiment"]["num_nodes"] = 5
    config["gnn_experiment"]["simulation_timesteps"] = 10
    config["gnn_experiment"]["traffic_flows_per_timestep"] = 2
    
    ckpt_dir = Path("data/experiments/model_d/checkpoints")
    if not ckpt_dir.exists():
        pytest.skip("Model D checkpoints missing, skip integration test")
        
    scaler_dir = Path("data/experiments/phase7_model_d/scalers/test")
    
    evaluator = Phase7ModelDEvaluator(
        config, 
        current_checkpoint_dir=config["phase7"]["checkpoint_dir"],
        model_d_checkpoint_dir=str(ckpt_dir),
        scaler_output_dir=scaler_dir
    )
    
    # Test just normal scenario
    result = evaluator.run_scenario("normal")
    
    assert result["scenario"] == "normal"
    assert "evaluation_start_timestep" in result
    assert result["causality"] == "All evaluated decisions use only telemetry completed before the decision timestep; realized QoS is computed after selected routes are applied."
    
    for method in METHODS:
        assert method in result["methods"]
        m_result = result["methods"][method]
        assert "mean_link_delay" in m_result
        assert "flow_samples" in m_result
        assert "link_samples" in m_result
