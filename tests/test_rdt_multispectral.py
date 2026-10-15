from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.rdt import RDTDomainTeacher, RDTTeacherConfig, build_rdt_teacher_labels
from emma_gpm.rdt.features import spectral_features
from emma_gpm.rdt.detection import detect_cells
from emma_gpm.rdt.utls_proxy import estimate_proxy, thermal_depth
from emma_gpm.rdt.tracking import associate, primary_lineage


def scene(size=65, radius=14, minimum=205, offset=0):
    y, x = np.indices((size, size))
    r = np.hypot(y-size//2, x-size//2)
    ir = np.full((size, size), 290., dtype=np.float32)
    ir[r <= radius] = 225.
    ring = (r > radius) & (r < radius+8)
    ir[ring] = 225+65*(r[ring]-radius)/8
    ir[r <= 3] = minimum
    channels = np.stack([ir+2, ir-1, ir+.5, ir-3, ir, ir-.5, ir-4])+offset
    return channels, np.full(ir.shape, 9.)


def test_features_raw_kelvin_canonical_signs():
    frame, _ = scene()
    features = spectral_features(frame)
    assert np.allclose(features[:, 20, 20], [2, -1, .5, -3, .5, -4])


def test_isolated_closed_support_and_temperature_translation():
    frame, area = scene()
    cfg = RDTTeacherConfig(smoothing_sigma_pixels=0)
    a, cells = detect_cells(frame[4], spectral_features(frame), area, np.ones(area.shape, bool), cfg)
    b, shifted = detect_cells(frame[4]+17, spectral_features(frame+17), area, np.ones(area.shape, bool), cfg)
    assert len(cells) == len(shifted) == 1
    assert cells[1]["edge_method"] == "closed_gradient_spectral_support"
    assert 225 < cells[1]["t_edge_k"] < 290
    assert np.array_equal(a, b)
    assert shifted[1]["t_edge_k"]-cells[1]["t_edge_k"] == pytest.approx(17)
    one = build_rdt_teacher_labels(frame[None], times=["2020-07-11"], pixel_area_km2=area, config=cfg)
    two = build_rdt_teacher_labels((frame+17)[None], times=["2020-07-11"], pixel_area_km2=area, config=cfg)
    assert np.allclose(one["thermal_depth"], two["thermal_depth"], equal_nan=True)


def test_two_minima_end_below_known_saddle():
    yy, xx = np.indices((51, 81))
    # Two paraboloid towers meet at x=40 at exactly 250 K.
    a = 210+(.1*(xx-20)**2+.1*(yy-25)**2)
    b = 210+(.1*(xx-60)**2+.1*(yy-25)**2)
    ir = np.minimum(a, b)
    frame = np.stack([ir+2, ir-1, ir+.5, ir-3, ir, ir-.5, ir-4]).astype(np.float32)
    cfg = RDTTeacherConfig(smoothing_sigma_pixels=0)
    labels, cells = detect_cells(frame[4], spectral_features(frame), np.full(ir.shape, 9.), np.ones(ir.shape, bool), cfg)
    assert len(cells) == 2
    assert all(c["edge_method"] == "pre_merge_saddle" and 249 <= c["t_edge_k"] < 250.1 for c in cells.values())
    assert labels[25, 20] and labels[25, 60] and labels[25, 20] != labels[25, 60]


def test_plateau_resists_small_cold_core_and_invalid_denominator():
    frame, area = scene()
    mask = frame[4] < 250
    cfg = RDTTeacherConfig()
    p = estimate_proxy(frame[4], spectral_features(frame), mask, area, 250, cfg)
    cold = frame.copy(); cold[:, 31:34, 31:34] -= 20
    q = estimate_proxy(cold[4], spectral_features(cold), mask, area, 250, cfg)
    assert p.instant_k == q.instant_k == 225
    assert p.confidence > cfg.min_proxy_confidence
    bad = thermal_depth(frame[4], mask, 225, 225, cfg)
    assert np.isnan(bad).all()


def test_actual_15min_growth_cooling_maturity_and_ot_routing():
    cfg = RDTTeacherConfig(smoothing_sigma_pixels=0)
    teacher = RDTDomainTeacher(np.full((65, 65), 9.), cfg)
    results = []
    minima = [205, 203, 201, 199, 199, 204, 206]
    for i, minimum in enumerate(minima):
        frame, _ = scene(radius=14+i, minimum=minimum)
        result = teacher.step(frame, pd.Timestamp("2020-07-11")+pd.Timedelta(minutes=5*i))
        results.append(result)
    assert all(np.isnan(r.objects.cooling_k_15min).all() for r in results[:3])
    assert results[3].objects.iloc[0].cooling_k_15min == -6
    assert results[3].objects.iloc[0].class_id == 2
    assert results[-1].objects.iloc[0].class_id == 3
    assert all(not r.fields["ot_mask"].any() for r in results[:3])
    assert results[3].fields["provisional_ot"].any()
    assert results[4].fields["ot_mask"].any()
    assert np.all(results[4].fields["semantic_label"][results[4].fields["ot_mask"]] == 4)


def test_causal_chunk_resume_and_gap_keep_unique_ids():
    frame, area = scene()
    times = pd.date_range("2020-07-11T23:45", periods=7, freq="5min")
    continuous = RDTDomainTeacher(area)
    chunked = RDTDomainTeacher(area)
    for i, stamp in enumerate(times):
        a = continuous.step(frame, stamp)
        if i == 3:
            chunked = deepcopy(chunked)
        b = chunked.step(frame, stamp)
        for field in a.fields:
            assert np.array_equal(a.fields[field], b.fields[field], equal_nan=True)
    old = int(b.objects.track_id.iloc[0])
    gap = chunked.step(frame, times[-1]+pd.Timedelta(minutes=10))
    assert gap.reset_reason == "observation_gap"
    assert gap.objects.track_id.iloc[0] > old
    assert not gap.objects.temporal_valid.any()


def test_invalid_input_and_explicit_domain_missing_pixels():
    frame, area = scene()
    teacher = RDTDomainTeacher(area)
    with pytest.raises(ValueError, match="seven-channel"):
        teacher.step(frame, "2020-07-11", channels=THERMAL_CHANNELS[::-1])
    bad = frame.copy(); bad[:, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid observations"):
        teacher.step(bad, "2020-07-11")
    result = teacher.step(bad, "2020-07-11", allow_invalid=True)
    assert not result.fields["valid_observation"][0, 0]
    assert result.fields["uncertainty"][0, 0]
    assert np.isnan(result.fields["thermal_depth"][0, 0])
    with pytest.raises(ValueError, match="increase"):
        teacher.step(frame, "2020-07-11")
    with pytest.raises(ValueError, match="chronological"):
        build_rdt_teacher_labels(np.stack([frame]*2), times=["2020-07-11", "2020-07-11T00:10"], pixel_area_km2=area)


def test_aor_primary_split_merge_and_complex():
    def obj(y0, y1, x0, x1, area):
        return dict(bbox=(slice(y0, y1), slice(x0, x1)), mask=np.ones((y1-y0, x1-x0), bool), area_km2=area)
    previous = {1: obj(0, 4, 0, 4, 16)}
    current = {1: obj(0, 4, 0, 3, 12), 2: obj(0, 4, 3, 4, 4)}
    labels = np.zeros((4, 4), np.int32); labels[:, :3] = 1; labels[:, 3:] = 2
    ps, cs, edges = associate(previous, current, labels, np.ones((4, 4)), RDTTeacherConfig())
    primary, events = primary_lineage(previous, current, ps, cs)
    assert primary == {1: 1} and events == {1: "split_main", 2: "split_new"}
    assert edges[1, 1]["aor"] == 1
    primary, events = primary_lineage(current, previous, {1: [1, 2]}, {1: [1], 2: [1]})
    assert primary == {1: 1} and events == {1: "merge_main"}
    primary, events = primary_lineage(current, current, {1: [1, 2], 2: [1]}, {1: [1, 2], 2: [1]})
    assert primary == {} and set(events.values()) == {"complex"}


def test_domain_masks_corrupt_kelvin_pixels_without_repair_but_clips_reject():
    frame, area = scene()
    frame[1, 0, 0] = -.434326171875
    frame[1, 32, 32] = -.434326171875
    original = frame.copy()
    with pytest.raises(ValueError, match="Raw Kelvin outside"):
        RDTDomainTeacher(area).step(frame,"2020-07-11")
    result = RDTDomainTeacher(area).step(frame,"2020-07-11",allow_invalid=True)
    assert result.fields['invalid_radiometry'].sum()==2
    for y,x in [(0,0),(32,32)]:
        assert not result.fields['valid_observation'][y,x]
        assert result.fields['semantic_label'][y,x]==0
        assert result.fields['uncertainty'][y,x]
        assert np.isnan(result.fields['thermal_depth'][y,x])
    assert np.array_equal(frame,original)
    with pytest.raises(ValueError,match="Raw Kelvin outside"):
        RDTDomainTeacher(area).step(np.zeros_like(frame),'2020-07-11',allow_invalid=True)


@pytest.mark.parametrize("shape",[(1,24),(24,1)])
def test_thin_proxy_support_abstains_without_fabricating_temperature(shape):
    cfg=RDTTeacherConfig()
    ir=np.full(shape,225.,np.float32)
    features=np.zeros((6,*shape),np.float32)
    result=estimate_proxy(ir,features,np.ones(shape,bool),np.full(shape,12.),250,cfg,history=[224.])
    assert result.reason=="insufficient_spatial_support"
    assert np.isnan(result.instant_k) and np.isnan(result.filtered_k)
    assert result.confidence==0 and not result.plateau.any()
    assert np.isnan(thermal_depth(ir,np.ones(shape,bool),250,result.filtered_k,cfg)).all()
