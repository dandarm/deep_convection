import numpy as np
import pandas as pd
import pytest
from emma_gpm.rdt_teacher import label_rdt_clip, RDTConfig, projected_pixel_areas
from emma_gpm.seviri import RSS_CRS


def run(clip, area=10):
    return label_rdt_clip(clip, pd.date_range('2020-06-01', periods=len(clip), freq='5min'),
                          np.full(clip.shape[2:], area))


def evolution():
    clip = np.full((3,7,20,20), 280.)
    for t, (size, minimum) in enumerate([(5,200), (7,198), (7,199)]):
        clip[t,4,5:5+size,5:5+size] = 235
        clip[t,4,7,7] = minimum
        clip[t,0,7,7] = minimum+2
    return clip


def test_developing_mature_and_ot_require_tracked_history():
    result = run(evolution())
    rows = result['objects']
    assert rows.class_id.tolist() == [1,2,3]
    assert rows.track_id.nunique() == 1
    assert rows.cooling_k_15min.iloc[1] == -6
    assert rows.growth_km2_15min.iloc[1] == 720
    assert result['classes'][0,7,7] == 1
    assert result['classes'][1,7,7] == 4
    assert result['classes'][2,7,7] == 4
    assert result['uncertain'][0,7,7]
    assert not result['uncertain'][1,7,7]


def test_area_filter_uses_ground_km2_and_eight_connectivity():
    clip = np.full((1,7,5,5), 280.)
    clip[0,4,1,1] = clip[0,4,2,2] = 230
    assert run(clip, area=49)['objects'].empty
    result = run(clip, area=60)
    assert len(result['objects']) == 1
    assert result['objects'].area_km2.iloc[0] == 120


def test_split_merge_record_lineage_and_reset_derivatives():
    clip = np.full((3,7,20,20), 280.)
    clip[:,4,5:10,5:16] = 230
    clip[1,4,5:10,10] = 280
    result = run(clip)
    assert result['objects'].event.tolist() == ['new','split','split','merge']
    assert not result['objects'].temporal_valid.any()
    assert result['objects'].track_id.nunique() == 4
    assert len(result['objects'].parents.iloc[-1]) == 2
    assert result['objects'].cooling_k_15min.isna().all()


def test_border_objects_cannot_create_false_temporal_growth():
    clip = evolution()
    clip[:,:,0:7,5:12] = clip[:,:,5:12,5:12]
    result = run(clip)
    border = result['objects'][result['objects'].border]
    assert not border.temporal_valid.any()
    assert (border.class_id == 1).all()


def test_only_wv062_ir108_affect_labels_and_run_is_reproducible():
    clip = evolution(); first = run(clip)
    altered = clip.copy(); altered[:,[1,2,3,5,6]] = 140
    second = run(altered)
    np.testing.assert_array_equal(first['classes'], second['classes'])
    pd.testing.assert_frame_equal(first['objects'], second['objects'])
    np.testing.assert_array_equal(first['tracks'], run(clip)['tracks'])


def test_invalid_kelvin_time_and_area_are_rejected():
    clip = evolution(); times = pd.date_range('2020-06-01',periods=3,freq='5min')
    for bad_times in [times[::-1], times.delete(1), pd.date_range('2020',periods=3,freq='10min')]:
        with pytest.raises(ValueError): label_rdt_clip(clip,bad_times,np.ones((20,20)))
    with pytest.raises(ValueError):label_rdt_clip(clip/100,times,np.ones((20,20)))
    clip[0,0,0,0]=np.nan
    with pytest.raises(ValueError):run(clip)
    with pytest.raises(ValueError): RDTConfig(min_aor=0)


def test_projected_geodesic_areas_respect_axis_orientation():
    x=np.arange(3)*3000.; y=3e6+np.arange(3)*3000
    areas=projected_pixel_areas(x,y,RSS_CRS.to_wkt())
    assert np.isfinite(areas).all() and (areas>9).all()
    reverse=projected_pixel_areas(x[::-1],y[::-1],RSS_CRS.to_wkt())
    np.testing.assert_allclose(reverse,areas[::-1,::-1])
    with pytest.raises(ValueError):projected_pixel_areas([0,1,3],y,RSS_CRS.to_wkt())
