import numpy as np
import pytest
from emma_gpm.physical_cluster_profiles import physical_profiles, physical_dispersion, physical_separation


def test_means_and_intercluster_distance_are_not_explained_variance():
    values = np.repeat([[250., 2.], [270., 2.]], 5, axis=0)
    labels = np.repeat([0, 1], 5)
    groups, counts, means, distance = physical_profiles(values, labels, [260., 2.], [10., 0.], ['thermal', 'spectral'])
    assert np.array_equal(groups, [0, 1]) and np.array_equal(counts, [5, 5])
    assert means[:, 0].tolist() == [250, 270]
    assert distance['thermal'][0, 1] == pytest.approx(2)
    assert distance['thermal'][0, 0] == 0
    assert np.isnan(distance['spectral']).all()


def test_equal_means_can_hide_different_internal_variability():
    values = np.array([-2., -1., 0., 1., 2., 0., 0., 0., 0., 0.])[:, None]
    _, _, means, distance = physical_profiles(values, np.repeat([0, 1], 5), [0.], [1.], ['dynamic'])
    assert np.all(means == 0)
    assert distance['dynamic'][0, 1] == 0  # Distinct spreads, same centroid.


def test_noise_and_small_groups_cannot_create_artificial_profile_separation():
    values = np.array([0.] * 5 + [10.] * 2 + [1000.])[:, None]
    groups, counts, means, distance = physical_profiles(values, np.array([0]*5+[1]*2+[-1]), [5.], [5.], ['thermal'])
    assert np.array_equal(groups, [0, 1])
    assert np.array_equal(counts, [5, 2])
    assert means[:, 0].tolist() == [0, 10]
    assert np.isnan(distance['thermal'][0, 1])


def test_diagnostic_unit_changes_preserve_standardized_profile_distances():
    values = np.repeat([[2.], [8.]], 5, axis=0)
    labels = np.repeat([0, 1], 5)
    original = physical_profiles(values, labels, [5.], [2.], ['thermal'])[3]['thermal']
    transformed = physical_profiles(values*3+100, labels, [115.], [6.], ['thermal'])[3]['thermal']
    assert np.allclose(original, transformed)


def test_pair_matrix_matches_explicit_empirical_clip_pairs():
    values = np.array([-2., -1., 0., 1., 2., 8., 9., 10., 11., 12.])[:, None]
    labels = np.repeat([0, 1], 5)
    matrices, summary = physical_dispersion(values, labels, [5.], [2.], ['thermal'])
    expected = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            difference = (values[labels == i, 0, None] - values[labels == j, 0]) / 2
            expected[i, j] = np.sqrt(np.mean(difference**2))
    np.testing.assert_allclose(matrices['thermal'], expected)
    assert summary['thermal']['between'] == pytest.approx(5)
    assert summary['thermal']['within'] == pytest.approx(1)
    assert summary['thermal']['ratio'] == pytest.approx(5)


def test_summary_excludes_diagonal_small_groups_and_weights_internal_scatter():
    values = np.r_[np.arange(5), np.repeat([10., 14.], 5), [1000., 1000.]][:, None]
    labels = np.r_[np.zeros(5), np.ones(10), [2, 2]]
    _, summary = physical_dispersion(values, labels, [0.], [1.], ['spatial'])
    assert summary['spatial']['between'] == pytest.approx(10)
    assert summary['spatial']['within'] == pytest.approx(np.sqrt((5*4+10*8)/15))


def test_global_summary_weights_families_equally_not_descriptor_counts():
    values = np.repeat([[0., 0., 0.], [2., 2., 10.]], 5, axis=0)
    _, summary = physical_dispersion(values, np.repeat([0, 1], 5), [0.]*3, [1.]*3,
                                     ['thermal', 'thermal', 'dynamic'])
    assert summary['global']['between'] == pytest.approx(6)
    assert np.isnan(summary['global']['ratio'])  # Zero within-cluster spread.


def test_no_groups_and_constant_family_do_not_produce_perfect_summary():
    for labels in [np.full(10, -1), np.repeat([0, 1], 5)]:
        matrices, summary = physical_dispersion(np.ones((10, 1)), labels, [1.], [0.], ['thermal'])
        assert np.isnan(matrices['thermal']).all()
        assert np.isnan(summary['global']['between'])


def test_relative_separation_matches_known_spread_and_is_symmetric():
    values = np.r_[np.arange(-2, 3), np.arange(8, 13)][:, None]
    result = physical_separation(values, np.repeat([0, 1], 5), [5.], [2.], ['thermal'])
    matrix, spread, score = result[3]['thermal'], result[4]['thermal'], result[5]['thermal']
    np.testing.assert_allclose(matrix, [[0, 5], [5, 0]])
    np.testing.assert_allclose(spread, np.sqrt(2)/2)
    assert score == pytest.approx(5)
    shifted = physical_separation(values*3+7, np.repeat([0, 1], 5), [22.], [6.], ['thermal'])
    np.testing.assert_allclose(shifted[3]['thermal'], matrix)


def test_zero_spread_is_undefined_and_equal_means_are_not_separated():
    values = np.repeat([[0.], [10.]], 5, axis=0)
    result = physical_separation(values, np.repeat([0, 1], 5), [5.], [1.], ['thermal'])
    assert np.isnan(result[3]['thermal'][0, 1])
    assert np.isnan(result[5]['global'])
    assert np.all(np.diag(result[3]['thermal']) == 0)
    values = np.r_[np.arange(-2, 3), np.arange(-2, 3)*10][:, None]
    result = physical_separation(values, np.repeat([0, 1], 5), [0.], [1.], ['thermal'])
    assert result[3]['thermal'][0, 1] == 0


def test_separation_masks_small_groups_and_all_noise():
    for labels in [np.full(7, -1), np.r_[np.zeros(5), [1, 1]]]:
        result = physical_separation(np.arange(7)[:, None], labels, [0.], [1.], ['thermal'])
        assert np.isnan(result[5]['global'])
        if len(result[0]):
            assert np.isnan(result[3]['thermal'][0, 1])
