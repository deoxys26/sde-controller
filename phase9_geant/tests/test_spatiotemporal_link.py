from __future__ import annotations

import inspect

import pytest
import torch

from src.gnn.spatiotemporal_link import (
    SEASONAL_CHANNELS,
    SHUFFLE_SEED,
    SpatioTemporalLinkForecaster,
    build_binary_link_adjacency,
    build_link_adjacency,
)


EDGES = (
    ("A", "B"),
    ("B", "A"),
    ("B", "C"),
    ("C", "B"),
    ("D", "E"),
    ("E", "D"),
)

MODEL_EDGES = (
    ("A", "B"), ("B", "A"),
    ("B", "C"), ("C", "B"),
    ("C", "D"), ("D", "C"),
    ("D", "E"), ("E", "D"),
)


def _model(*, adjacency_mode: str = "topology", hidden_size: int = 8) -> SpatioTemporalLinkForecaster:
    return SpatioTemporalLinkForecaster(
        MODEL_EDGES,
        history_length=8,
        horizons=(4, 96),
        hidden_size=hidden_size,
        alpha=0.8,
        ewma_steps=5,
        include_seasonal_features=False,
        adjacency_mode=adjacency_mode,  # type: ignore[arg-type]
    )


def _seasonal_model(*, adjacency_mode: str = "topology") -> SpatioTemporalLinkForecaster:
    return SpatioTemporalLinkForecaster(
        MODEL_EDGES,
        history_length=672,
        horizons=(4, 96),
        hidden_size=8,
        alpha=0.8,
        ewma_steps=5,
        include_seasonal_features=True,
        adjacency_mode=adjacency_mode,  # type: ignore[arg-type]
    )


def test_shapes_finite_predictions_and_zero_initialized_ewma_residual() -> None:
    torch.manual_seed(7)
    model = _model()
    history = torch.rand(3, 8, len(MODEL_EDGES))
    prediction = model(history)

    assert prediction.shape == (3, 2, len(MODEL_EDGES))
    assert torch.isfinite(prediction).all()
    assert torch.equal(prediction[:, 0], prediction[:, 1])
    weights = model.ewma_weights
    expected = torch.einsum("bte,t->be", history[:, -5:, :], weights)
    torch.testing.assert_close(prediction[:, 0], expected)
    expected_parameters = 3 * 8 + 3 * 8 * 8 + 6 * 8 + (2 * 8) + (8 * 8 + 8) + 8 * 8 + (2 * 8 + 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters


def test_link_adjacency_preserves_directed_edge_order_and_components() -> None:
    adjacency = build_link_adjacency(EDGES)
    assert adjacency.shape == (6, 6)
    torch.testing.assert_close(adjacency[:4, :4], torch.full((4, 4), 0.25))
    torch.testing.assert_close(adjacency[4:, 4:], torch.full((2, 2), 0.5))
    assert torch.count_nonzero(adjacency[:4, 4:]) == 0
    assert torch.count_nonzero(adjacency[4:, :4]) == 0
    model = _model()
    assert model.edge_order == MODEL_EDGES
    torch.testing.assert_close(model.topology_adjacency, build_link_adjacency(MODEL_EDGES))
    torch.testing.assert_close(model.identity_adjacency, torch.eye(len(MODEL_EDGES)))


@pytest.mark.parametrize("mode", ["topology", "identity", "shuffled"])
def test_batch_samples_are_independent(mode: str) -> None:
    torch.manual_seed(11)
    model = _model(adjacency_mode=mode).eval()
    first = torch.rand(1, 8, len(MODEL_EDGES))
    second = torch.rand(1, 8, len(MODEL_EDGES)) * 10
    with torch.no_grad():
        together = model(torch.cat((first, second), dim=0))
        alone = model(first)
    torch.testing.assert_close(together[:1], alone)


def test_forward_accepts_history_only_and_rejects_future_target_argument() -> None:
    model = _model()
    assert tuple(inspect.signature(model.forward).parameters) == (
        "history", "seasonal_features", "adjacency_mode"
    )
    history = torch.rand(2, 8, len(MODEL_EDGES))
    future_target = torch.rand(2, 2, len(MODEL_EDGES))
    with pytest.raises(TypeError):
        model(history, target=future_target)  # type: ignore[call-arg]


def test_gradients_reach_temporal_and_graph_message_parameters() -> None:
    torch.manual_seed(19)
    model = _model()
    # The default zero head starts at EWMA and intentionally blocks upstream
    # gradients on step zero; set a nonzero synthetic head to test propagation.
    with torch.no_grad():
        model.residual_head.weight.fill_(0.1)
    history = torch.rand(2, 8, len(MODEL_EDGES), requires_grad=True)
    loss = model(history).square().mean()
    loss.backward()

    assert history.grad is not None and torch.isfinite(history.grad).all()
    for module in (model.temporal_encoder, model.message_projection, model.residual_head):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients if gradient is not None)


def _configure_neighbor_sensitivity(model: SpatioTemporalLinkForecaster) -> None:
    hidden = model.temporal_encoder.hidden_size
    with torch.no_grad():
        for parameter in model.temporal_encoder.parameters():
            parameter.zero_()
        # GRU gate order is reset, update, new; this makes each link state a
        # deterministic positive function of that link's history.
        model.temporal_encoder.weight_ih_l0[2 * hidden, 0] = 1.0
        model.self_projection.weight.zero_()
        model.self_projection.bias.zero_()
        model.message_projection.weight.copy_(torch.eye(hidden))
        model.residual_head.weight.fill_(0.2)
        model.residual_head.bias.zero_()


def test_neighbor_history_changes_prediction_only_when_graph_enabled() -> None:
    torch.manual_seed(23)
    graph_model = _model(adjacency_mode="topology").eval()
    _configure_neighbor_sensitivity(graph_model)
    before = torch.zeros(1, 8, len(MODEL_EDGES))
    after = before.clone()
    # Link 0 (A->B) is adjacent to target link 2 (B->C) in the declared edge order.
    after[:, :, 0] = 1.0
    with torch.no_grad():
        graph_before = graph_model(before)
        graph_after = graph_model(after)
    assert not torch.isclose(graph_before[0, :, 2], graph_after[0, :, 2]).all()

    no_graph = _model(adjacency_mode="identity").eval()
    _configure_neighbor_sensitivity(no_graph)
    with torch.no_grad():
        independent_before = no_graph(before)
        independent_after = no_graph(after)
    torch.testing.assert_close(independent_before[0, :, 2], independent_after[0, :, 2])


def test_nonfinite_or_misaligned_history_fails_closed() -> None:
    model = _model()
    with pytest.raises(ValueError, match="Expected"):
        model(torch.zeros(1, 8, len(MODEL_EDGES) - 1))
    nonfinite = torch.zeros(1, 8, len(MODEL_EDGES))
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model(nonfinite)


def test_seasonal_feature_shape_order_and_causal_sources() -> None:
    model = _seasonal_model().eval()
    # Link-specific synthetic values make the source index arithmetic visible.
    history = torch.arange(672, dtype=torch.float32).view(1, 672, 1).expand(
        1, 672, len(MODEL_EDGES)
    ).clone()
    features = model.seasonal_residual_features(history)
    baseline = model._ewma(history)
    assert features.shape == (1, 2, len(MODEL_EDGES), len(SEASONAL_CHANNELS))
    # t=671: daily sources are t-92 and t; weekly sources are t-668 and t-576.
    torch.testing.assert_close(features[0, 0, 0, 0], history[0, 579, 0] - baseline[0, 0])
    torch.testing.assert_close(features[0, 0, 0, 1], history[0, 3, 0] - baseline[0, 0])
    torch.testing.assert_close(features[0, 1, 0, 0], history[0, 671, 0] - baseline[0, 0])
    torch.testing.assert_close(features[0, 1, 0, 1], history[0, 95, 0] - baseline[0, 0])
    torch.testing.assert_close(model(history, seasonal_features=features), model(history))


def test_future_derived_seasonal_features_are_rejected() -> None:
    model = _seasonal_model().eval()
    history = torch.rand(2, 672, len(MODEL_EDGES))
    supplied = model.seasonal_residual_features(history).clone()
    # A changed seasonal value cannot be accepted as an unverified covariate.
    supplied[:, 1, :, 0] += 1.0
    with pytest.raises(ValueError, match="derived from the supplied history"):
        model(history, seasonal_features=supplied)
    with pytest.raises(TypeError):
        model(history, target=torch.rand(2, 2, len(MODEL_EDGES)))  # type: ignore[call-arg]


def test_history_derived_seasonal_features_reach_the_forecast_head() -> None:
    model = _seasonal_model().eval()
    with torch.no_grad():
        model.temporal_encoder.weight_ih_l0.zero_()
        model.temporal_encoder.weight_hh_l0.zero_()
        model.temporal_encoder.bias_ih_l0.zero_()
        model.temporal_encoder.bias_hh_l0.zero_()
        model.seasonal_projection.weight.fill_(1.0)
        model.self_projection.weight.copy_(torch.eye(8))
        model.self_projection.bias.zero_()
        model.message_projection.weight.zero_()
        model.residual_head.weight.fill_(0.1)
        model.residual_head.bias.zero_()
    baseline_history = torch.zeros(1, 672, len(MODEL_EDGES))
    changed_history = baseline_history.clone()
    # For t=671,h=4, daily source is index 579; it is still in the input.
    changed_history[:, 579, 0] = 1.0
    with torch.no_grad():
        baseline_prediction = model(baseline_history)
        changed_prediction = model(changed_history)
    assert not torch.allclose(baseline_prediction, changed_prediction)


def test_horizon_specific_head_channels_align_with_requested_horizons() -> None:
    model = _seasonal_model().eval()
    with torch.no_grad():
        model.temporal_encoder.weight_ih_l0.zero_()
        model.temporal_encoder.weight_hh_l0.zero_()
        model.temporal_encoder.bias_ih_l0.zero_()
        model.temporal_encoder.bias_hh_l0.zero_()
        model.seasonal_projection.weight.zero_()
        model.self_projection.weight.zero_()
        model.self_projection.bias.zero_()
        model.message_projection.weight.zero_()
        model.residual_head.weight.zero_()
        model.residual_head.bias.copy_(torch.tensor([0.25, -0.5]))
    history = torch.full((1, 672, len(MODEL_EDGES)), 0.4)
    prediction = model(history)
    baseline = model._ewma(history)
    torch.testing.assert_close(prediction[:, 0], baseline + 0.25)
    torch.testing.assert_close(prediction[:, 1], baseline - 0.5)


def test_adjacency_modes_share_parameters_and_use_one_operator() -> None:
    counts = {
        mode: sum(parameter.numel() for parameter in _model(adjacency_mode=mode).parameters())
        for mode in ("topology", "identity", "shuffled")
    }
    assert len(set(counts.values())) == 1
    assert counts["topology"] == 434
    default_counts = {
        mode: sum(
            parameter.numel()
            for parameter in SpatioTemporalLinkForecaster(MODEL_EDGES, adjacency_mode=mode).parameters()
        )
        for mode in ("topology", "identity", "shuffled")
    }
    assert default_counts == {mode: 1506 for mode in default_counts}


def test_shuffled_adjacency_is_seeded_symmetric_and_degree_preserving() -> None:
    raw = build_binary_link_adjacency(MODEL_EDGES)
    shuffled = build_link_adjacency(MODEL_EDGES, mode="shuffled", shuffle_seed=SHUFFLE_SEED)
    repeated = build_link_adjacency(MODEL_EDGES, mode="shuffled", shuffle_seed=SHUFFLE_SEED)
    torch.testing.assert_close(shuffled, repeated)
    assert not torch.equal(shuffled, build_link_adjacency(MODEL_EDGES))
    torch.testing.assert_close(shuffled.sum(dim=1), torch.ones(len(MODEL_EDGES)))
    # Recover the binary adjacency from row normalization, then inspect its
    # undirected symmetry and degree sequence.
    original_degree = raw.sum(dim=1)
    recovered_degree = 1.0 / shuffled.diagonal()
    torch.testing.assert_close(recovered_degree, original_degree)
    recovered_binary = shuffled * recovered_degree.unsqueeze(1)
    torch.testing.assert_close(recovered_binary, recovered_binary.T)


@pytest.mark.parametrize("mode", ["topology", "identity", "shuffled"])
@pytest.mark.parametrize("seasonal", [False, True])
def test_zero_residual_head_equals_fixed_ewma_for_every_mode(mode: str, seasonal: bool) -> None:
    model = (
        _seasonal_model(adjacency_mode=mode)
        if seasonal
        else _model(adjacency_mode=mode)
    ).eval()
    history = torch.rand(2, model.history_length, len(MODEL_EDGES))
    with torch.no_grad():
        prediction = model(history)
        baseline = model._ewma(history).unsqueeze(1).expand_as(prediction)
    torch.testing.assert_close(prediction, baseline)
