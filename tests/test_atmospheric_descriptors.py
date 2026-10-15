import numpy as np
import pytest

from emma_gpm.atmospheric_descriptors import describe_clip


def test_constant_kelvin_cube_has_zero_dynamics_and_defined_empty_regions():
    values, schema = describe_clip(np.full((16, 7, 224, 224), 280, dtype=np.float32))
    assert all(values[name] == 0 for name, spec in schema.items() if spec['group'] == 'dynamic')
    assert values['IR_108_below260_components'] == 0
    assert values['IR_108_below260_largest_pct'] == 0
    assert values['IR_108_q10_k'] == 280
    assert values['WV_062_minus_IR_108_mean_k'] == 0


def test_tendency_sign_units_and_spectral_difference_are_preserved():
    clip = np.full((16, 7, 224, 224), 280, dtype=np.float32)
    clip -= np.arange(16, dtype=np.float32)[:, None, None, None] * 2
    clip[:, 0] += 12
    values, _ = describe_clip(clip)
    assert values['IR_108_tendency_k_15min'] == pytest.approx(-6)
    assert values['IR_108_q10_tendency_k_15min'] == pytest.approx(-6)
    assert values['IR_108_temporal_abs_k_5min'] == pytest.approx(2)
    assert values['IR_108_temporal_pattern_std_k_5min'] == 0
    assert values['WV_062_minus_IR_108_mean_k'] == 12
    assert values['WV_062_minus_IR_108_tendency_k_15min'] == 0


def test_eight_connected_regions_and_pixel_fraction_have_no_tracking_assumptions():
    clip = np.full((16, 7, 224, 224), 280, dtype=np.float32)
    clip[:, 4, :2, :2] = 250
    clip[:, 4, 100:102, 100:102] = 250
    values, _ = describe_clip(clip)
    assert values['IR_108_below260_components'] == 2
    assert values['IR_108_below260_pct'] == pytest.approx(800 / (224 * 224))
    assert values['IR_108_below260_largest_pct'] == pytest.approx(400 / (224 * 224))


def test_descriptor_boundary_rejects_incomplete_or_nonfinite_clips():
    with pytest.raises(ValueError):
        describe_clip(np.zeros((15, 7, 224, 224)))
    clip = np.full((16, 7, 224, 224), 280, dtype=np.float32)
    clip[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        describe_clip(clip)
