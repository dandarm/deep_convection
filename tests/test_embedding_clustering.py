import numpy as np
import pytest

pytest.importorskip('sklearn', reason='Optional requirements-analysis.txt dependency')

from emma_gpm.embedding_clustering import (
    pool_encoder_tokens, hierarchy, explained_variance, cross_day_neighbor_gain,
    day_mean_interval,
)


def test_temporal_pooling_retains_signed_change_and_correct_token_order():
    import torch
    # Eight temporal tubelets, four spatial tokens, known linear time signal.
    series = torch.arange(8) * 10 + 2.5
    tokens = series.reshape(1, 8, 1, 1).expand(1, 8, 4, 2).reshape(1, 32, 2)
    pooled = pool_encoder_tokens(tokens)
    assert torch.allclose(pooled['mean'], torch.full((1, 2), 37.5))
    assert torch.allclose(pooled['mean_slope'][:, 2:], torch.full((1, 2), 75.))


def test_external_agreement_detects_good_and_bad_partitions_and_constants():
    values = np.array([[0, 5], [0, 5], [10, 5], [10, 5]])
    good = explained_variance(values, np.array([0, 0, 1, 1]))
    bad = explained_variance(values, np.array([0, 1, 0, 1]))
    assert good[0] == pytest.approx(1)
    assert bad[0] < 0
    assert np.isnan(good[1])


def test_neighbors_exclude_same_day_and_score_relative_to_eligible_pairs():
    values = np.array([[0], [100], [0], [100]], dtype=float)
    distances = np.abs(values - values.T)
    days = np.array(['a', 'a', 'b', 'b'])
    gains, neighbors = cross_day_neighbor_gain(distances, values, days, k=1)
    assert np.array_equal(neighbors.ravel(), [2, 3, 0, 1])
    assert np.all(gains == 1)
    assert np.all(days[neighbors[:, 0]] != days)


def test_hierarchy_uses_valid_metric_and_rejects_collapsed_embeddings():
    values = np.array([[1., 0], [1., .1], [0, 1], [.1, 1]])
    for method in ['average', 'ward']:
        tree, distances = hierarchy(values, method)
        assert tree.shape == (3, 4)
        assert np.allclose(distances, distances.T)
        assert np.all(np.diag(distances) == 0)
    with pytest.raises(ValueError):
        hierarchy(np.zeros((4, 2)), 'average')


def test_day_bootstrap_weights_days_equally_and_preserves_paired_difference():
    days = np.array(['a'] * 100 + ['b'] * 2)
    values = np.array([0.] * 100 + [1.] * 2)[:, None]
    mean, low, high = day_mean_interval(values, days)
    assert mean[0] == pytest.approx(.5)
    assert low[0] <= mean[0] <= high[0]
    difference = (values + .2) - values
    mean, low, high = day_mean_interval(difference, days)
    assert np.allclose([mean[0], low[0], high[0]], .2)
