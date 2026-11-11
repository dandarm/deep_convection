import numpy as np
import pandas as pd
import pytest

from emma_gpm.ssl_incremental_sampling import (IncrementalCropSampler,
    augment_events, audit_added_pairs, balanced_extension)


def test_new_clip_checks_future_fixed_events_and_combined_reuse():
    # A quarter of the crop area in common gives IoU 1/7: spatially admissible.
    sampler = IncrementalCropSampler([[10, 0, 0]], size=20)
    assert sampler.try_add([10, 0, 15]) == 'shared_observations'
    assert sampler.try_add([0, 0, 0]) == 'spatial_iou'  # Future fixed event.
    assert sampler.try_add([2, 0, 15]) == 'accepted'  # Eight frames, 12.5% observations.
    assert sampler.try_add([10, 30, 30]) == 'accepted'  # All frames, distinct area.
    assert sampler.try_add([26, 0, 0]) == 'accepted'  # Consecutive clips, no shared frame.
    audit = audit_added_pairs(sampler.events, 1, size=20)
    assert audit['maximum_active_iou'] <= .2
    assert audit['maximum_new_pair_shared_fraction'] <= .2


def test_fixed_pairs_are_retained_and_new_pairs_respect_the_extra_limit():
    fixed = np.array([[0, 0, 0], [0, 0, 15]])
    sampler = IncrementalCropSampler(fixed, size=20)
    assert sampler.try_add([0, 40, 40]) == 'accepted'
    np.testing.assert_array_equal(sampler.events[:2], fixed)
    audit = audit_added_pairs(sampler.events, 2, size=20)
    assert audit['maximum_fixed_pair_shared_fraction'] == .25
    assert audit['maximum_new_pair_shared_fraction'] == 0
    with pytest.raises(ValueError, match='unique'):
        IncrementalCropSampler([[0, 0, 0], [0, 0, 0]])


def test_rejection_stopping_counts_individual_proposals_and_is_reproducible():
    added, blocks, reason = augment_events([[0, 0, 0]], {0: [0]}, 1, seed=3,
        size=2, block_size=10, patience=3, max_proposals=100)
    assert len(added) == 0 and reason == 'low_acceptance'
    assert len(blocks) == 3 and sum(b['proposals'] for b in blocks) == 30
    choices = {i: np.arange(21*21) for i in range(30)}
    result = augment_events([], choices, 21, seed=5, size=10, block_size=50, max_proposals=500)
    repeat = augment_events([], choices, 21, seed=5, size=10, block_size=50, max_proposals=500)
    np.testing.assert_array_equal(result[0], repeat[0])
    assert result[1:] == repeat[1:]
    audit = audit_added_pairs(result[0], 0, size=10)
    assert audit['maximum_active_iou'] <= .2+1e-12
    assert audit['maximum_new_pair_shared_fraction'] <= .2+1e-12


def test_calendar_extension_keeps_every_parent_and_balances_full_totals():
    rows = []
    for month, total in ((1, 24), (2, 40), (3, 31)):
        for i in range(total):
            day = f'2022-{month:02d}-{1+i%2:02d}'
            rows.append(dict(clip_id=f'{month}_{i}', day=day, time_bin=i%4,
                start_time=f'{day}T{i%24:02d}:00:00Z', origin_y=i, origin_x=0))
    candidates = pd.DataFrame(rows)
    fixed = candidates.groupby(candidates.day.str[:7]).head(6)
    selected = balanced_extension(fixed, candidates, seed=4)
    assert len(selected) == 72 and len(selected)%4 == 0
    assert set(fixed.clip_id) <= set(selected.clip_id)
    assert selected.clip_id.is_unique
    assert set(selected.groupby(selected.day.str[:7]).size()) == {24}
    pd.testing.assert_frame_equal(selected, balanced_extension(fixed, candidates, seed=4))


def test_longer_search_keeps_original_blocks_candidates_and_pairwise_limits():
    choices = {i: np.arange(31*31) for i in range(40)}
    options = dict(seed=11, size=10, block_size=64, minimum_acceptance=1., max_proposals=1024)
    short = augment_events([[0, 0, 0]], choices, 31, patience=3, **options)
    long = augment_events([[0, 0, 0]], choices, 31, patience=10, **options)
    assert len(short[1]) == 3 and len(long[1]) == 10
    assert long[1][:3] == short[1]
    np.testing.assert_array_equal(long[0][:len(short[0])], short[0])
    audit = audit_added_pairs(np.vstack([[[0,0,0]], long[0]]), 1, size=10)
    assert audit['maximum_active_iou'] <= .2+1e-12
    assert audit['maximum_new_pair_shared_fraction'] <= .2+1e-12
