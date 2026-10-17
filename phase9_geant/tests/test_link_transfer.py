import numpy as np

from phase9_geant.run_link_transfer import (
    build_ordered_path_incidence,
    build_physical_weights,
    build_valid_starts,
    fit_ridge_predict,
    make_features,
    shuffle_incidence_weights,
)


def test_valid_windows_exclude_invalid_steps_and_fold_boundaries():
    valid = np.ones(20, dtype=bool)
    valid[8] = False
    starts = build_valid_starts(valid, 3, 16, history=5)
    assert starts.tolist() == [9, 10]
    assert all(valid[s : s + 6].all() and 3 <= s and s + 5 < 16 for s in starts)


def test_feature_construction_uses_only_history_and_recent_known_demand():
    util = np.arange(12 * 3, dtype=float).reshape(12, 3) / 100.0
    demand = np.arange(12, dtype=float) * 10.0
    weights = np.eye(3, dtype=float)
    starts = np.asarray([2])
    before = make_features(util, demand, starts, weights)
    changed_future = util.copy()
    changed_future[starts[0] + 5, :] += 500.0
    after = make_features(changed_future, demand, starts, weights)
    np.testing.assert_array_equal(before["own"], after["own"])
    np.testing.assert_array_equal(before["spatial"], after["spatial"])
    assert not np.array_equal(before["target_delta"][0], after["target_delta"][0])


def test_physical_weights_are_directed_predecessor_to_successor_and_row_normalized():
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("b", "a")]
    weights = build_physical_weights(edges)
    assert weights[1, 0] == 1.0  # a->b informs b->c
    assert weights[0, 1] == 0.0  # b->c does not inform a->b
    assert weights[3, 0] == 1.0  # a->b may inform b->a as a directed U-turn
    row_sums = weights.sum(axis=1)
    np.testing.assert_array_equal(row_sums, np.ones(4))


def test_od_incidence_weights_respect_route_order_and_direction():
    nodes = ["a", "b", "c"]
    edges = [("a", "b"), ("b", "c"), ("c", "b"), ("b", "a")]
    paths = {
        "a→b": ["a", "b"],
        "a→c": ["a", "b", "c"],
        "b→a": ["b", "a"],
        "b→c": ["b", "c"],
        "c→a": ["c", "b", "a"],
        "c→b": ["c", "b"],
    }
    weights = build_ordered_path_incidence(paths, nodes, edges, np.ones(6))
    assert weights[1, 0] > 0  # a->b precedes b->c on a->c
    assert weights[0, 1] == 0  # reverse route order is not included
    assert weights[3, 2] > 0  # c->b precedes b->a on c->a
    assert weights[2, 3] == 0
    np.testing.assert_allclose(weights.sum(axis=1), [0.0, 1.0, 0.0, 1.0])


def test_shuffled_incidence_is_seeded_and_preserves_each_target_weight_multiset():
    weights = np.asarray(
        [[0.0, 0.2, 0.3, 0.5], [0.4, 0.0, 0.6, 0.0], [0.1, 0.2, 0.0, 0.7], [0.0, 0.3, 0.7, 0.0]]
    )
    first = shuffle_incidence_weights(weights, seed=99)
    second = shuffle_incidence_weights(weights, seed=99)
    np.testing.assert_array_equal(first, second)
    for target in range(len(weights)):
        assert first[target, target] == 0
        np.testing.assert_array_equal(np.sort(first[target]), np.sort(weights[target]))


def test_ridge_scaling_and_predictions_are_reproducible_and_eval_independent():
    rng = np.random.default_rng(15)
    train_x = rng.normal(size=(40, 3, 4))
    train_y = rng.normal(size=(40, 3))
    eval_x = rng.normal(size=(8, 3, 4))
    pred1, params1 = fit_ridge_predict(train_x, train_y, eval_x, alpha=1.0)
    pred2, params2 = fit_ridge_predict(train_x, train_y, eval_x + 50.0, alpha=1.0)
    np.testing.assert_array_equal(pred1, fit_ridge_predict(train_x, train_y, eval_x, alpha=1.0)[0])
    for one, two in zip(params1, params2):
        np.testing.assert_array_equal(one["x_mean"], two["x_mean"])
        np.testing.assert_array_equal(one["x_scale"], two["x_scale"])
        np.testing.assert_array_equal(one["coef"], two["coef"])
    assert not np.array_equal(pred1, pred2)
