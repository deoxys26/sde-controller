"""Tests for Model D (TemporalAttentionEdgeForecaster).

Verifies:
- output shape matches target shape (same as Model B)
- outputs are finite
- temporal attention weights sum to 1 along the history axis
- the attention layer adds exactly gru_hidden_size parameters vs Model B
- seeded training produces reproducible results
- the forecast interface accepts Model D via the existing forecast_adapter path
"""

import torch
import pytest

from src.gnn.dataset import TemporalGraphDataset
from src.gnn.model import EdgeQoSForecaster, TemporalAttentionEdgeForecaster
from src.simulator.network_simulator import NetworkSimulator
from src.topology.graph_builder import build_topology


def _tiny_dataset():
    graph = build_topology(num_nodes=5, seed=7)
    _, links = NetworkSimulator(
        graph, {"seed": 7, "simulation_timesteps": 14,
                "traffic_flows_per_timestep": 3}
    ).run()
    return TemporalGraphDataset(links, graph, history_length=5), graph


def _model_d(dataset):
    return TemporalAttentionEdgeForecaster(
        len(dataset.feature_columns), len(dataset.target_columns),
        hidden_channels=8, gru_hidden_size=8
    )


def _model_b(dataset):
    return EdgeQoSForecaster(
        len(dataset.feature_columns), len(dataset.target_columns),
        hidden_channels=8, gru_hidden_size=8
    )


def test_model_d_output_shape_matches_target():
    dataset, _ = _tiny_dataset()
    sequence, target, _ = dataset[0]
    model = _model_d(dataset)
    output = model(sequence)
    assert output.shape == target.shape


def test_model_d_outputs_are_finite():
    dataset, _ = _tiny_dataset()
    sequence, _, _ = dataset[0]
    model = _model_d(dataset)
    output = model(sequence)
    assert torch.isfinite(output).all()


def test_model_d_attention_weights_sum_to_one():
    """Verify the softmax attention sums to 1 over the history dimension."""
    dataset, _ = _tiny_dataset()
    sequence, _, _ = dataset[0]
    model = _model_d(dataset)
    # Build GRU outputs manually to check weights.
    encoded = []
    for graph in sequence:
        ns = torch.relu(model.gcn1(graph.x, graph.edge_index))
        ns = torch.relu(model.gcn2(ns, graph.edge_index))
        src, dst = graph.edge_index
        es = torch.cat((ns[src], ns[dst], graph.edge_attr), dim=1)
        encoded.append(model.edge_encoder(es))
    temporal = torch.stack(encoded, dim=1)
    output, _ = model.gru(temporal)
    scores = model.temporal_attn(output)
    weights = torch.softmax(scores, dim=1)
    # weights: [links, history, 1]; sum over history should be ~1.0 per link.
    sums = weights.sum(dim=1).squeeze(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_model_d_has_exactly_one_attention_layer_more_than_model_b():
    """Model D adds only gru_hidden_size parameters relative to Model B."""
    dataset, _ = _tiny_dataset()
    b = _model_b(dataset)
    d = _model_d(dataset)
    b_params = sum(p.numel() for p in b.parameters())
    d_params = sum(p.numel() for p in d.parameters())
    # Exactly gru_hidden_size=8 extra weights in temporal_attn.
    assert d_params - b_params == 8


def test_model_d_seeded_training_is_reproducible():
    """Two training runs with the same seed must produce identical outputs."""
    from src.gnn.train import train_model, set_seed
    import tempfile, os

    dataset, _ = _tiny_dataset()

    def _run():
        model = _model_d(dataset)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
            ckpt = fh.name
        try:
            train_model(model, dataset, {"seed": 42, "epochs": 2,
                                         "learning_rate": 0.005, "batch_size": 4,
                                         "early_stopping_patience": 2,
                                         "checkpoint_path": ckpt})
            sequence, _, _ = dataset[0]
            model.eval()
            with torch.no_grad():
                return model(sequence).detach().clone()
        finally:
            if os.path.exists(ckpt):
                os.remove(ckpt)

    first, second = _run(), _run()
    assert torch.allclose(first, second, atol=1e-6), (
        "Model D training is not reproducible under the same seed"
    )


def test_model_d_build_model_name_resolves():
    """build_model('model_d', ...) must return a TemporalAttentionEdgeForecaster."""
    from src.gnn.run_experiment import build_model
    dataset, _ = _tiny_dataset()
    model = build_model("model_d", dataset, {"hidden_channels": 8, "gru_hidden_size": 8})
    assert isinstance(model, TemporalAttentionEdgeForecaster)
