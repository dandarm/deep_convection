import numpy as np
import pytest
pytest.importorskip('sklearn')

from emma_gpm.density_clustering import density_partition, density_scores


def test_density_discovers_groups_without_prescribed_count():
    rng = np.random.default_rng(31)
    values = np.concatenate([rng.normal(loc=center, scale=.015, size=(50, 3))
                             for center in ([1, 0, 0], [0, 1, 0])])
    labels, strength, minimum = density_partition(values)
    assert len(np.unique(labels[labels >= 0])) == 2
    assert minimum == 10
    assert np.all((strength >= 0) & (strength <= 1))


def test_unassigned_points_are_excluded_and_coverage_is_explicit():
    values = np.array([[1, 0], [1, .1], [0, 1], [.1, 1], [-1, -1]], dtype=float)
    descriptors = np.array([[0], [0], [10], [10], [1000]], dtype=float)
    labels = np.array([0, 0, 1, 1, -1])
    metrics, physical, _ = density_scores(values, descriptors, labels)
    assert metrics['clusters'] == 2
    assert metrics['coverage'] == .8
    assert physical[0] == pytest.approx(1)


def test_all_noise_is_not_reported_as_perfect_clustering():
    rng = np.random.default_rng(2)
    metrics, physical, _ = density_scores(rng.normal(size=(20, 4)), rng.normal(size=(20, 2)), np.full(20, -1))
    assert metrics['clusters'] == 0 and metrics['coverage'] == 0
    assert np.isnan(metrics['silhouette']) and np.isnan(physical).all()
