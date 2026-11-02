import numpy as np
import torch

from emma_gpm.forecasting import ForecastWindow
from emma_gpm.ssl_model import history_config
from emma_gpm.ssl_diagnostics import pool_embeddings,radiometric_descriptors,DESCRIPTOR_NAMES,load_run_model,purely_advected_history
from emma_gpm.ssl_dashboard import spatial_coverage, crop_origin_counts, crop_origin_locations


def test_pooled_temporal_slope_retains_order_and_channel_identity():
    config=history_config(ForecastWindow(),image_size=32,patch_size=8,hidden_size=6)
    time=torch.linspace(-1,1,5)
    intercept=torch.arange(6).float()
    slopes=torch.arange(6).float()+1
    tokens=(intercept[None,None,None,:]+time[None,:,None,None]*slopes[None,None,None,:]).expand(2,5,16,6).reshape(2,80,6)
    pooled=pool_embeddings(tokens,config)
    torch.testing.assert_close(pooled[0,:6],intercept)
    torch.testing.assert_close(pooled[0,6:],slopes)
    reversed_tokens=tokens.reshape(2,5,16,6).flip(1).reshape(2,80,6)
    torch.testing.assert_close(pool_embeddings(reversed_tokens,config)[:,6:],-pooled[:,6:])


def test_descriptors_use_real_fifteen_and_thirty_minute_changes():
    history=torch.arange(10).float()[None,:,None,None,None].expand(1,10,7,8,8)+260
    targets=torch.tensor([272.,275.])[None,:,None,None,None].expand(1,2,7,8,8)
    values=radiometric_descriptors(history,targets)[0]
    assert len(values)==len(DESCRIPTOR_NAMES)
    assert float(values[7])==1
    assert float(values[8])==3
    assert float(values[9])==0
    torch.testing.assert_close(values[10:12],torch.tensor([3.,6.]))


def test_diagnostic_model_reads_json_equivalent_atomic_identity(tmp_path):
    from scripts.pretrain_seviri_ssl import main
    main(['--synthetic-smoke','--device','cpu','--cpu-threads','1','--output-dir',str(tmp_path),
          '--head-width','12','--epochs','1'])
    model,config=load_run_model(tmp_path)
    assert model.config.num_frames==10
    assert not model.training


def test_spatial_coverage_counts_exact_crop_footprints():
    import pandas as pd
    frame=pd.DataFrame({'origin_y':[0,112],'origin_x':[0,112]})
    coverage=spatial_coverage(frame,336,336)
    assert coverage.sum()==2*224**2
    assert coverage[150,150]==2
    assert coverage[0,335]==0


def test_crop_origin_is_tensor_top_left_after_coordinate_reversal():
    import pandas as pd
    frame=pd.DataFrame({'origin_y':[0,0,112],'origin_x':[0,0,112]})
    points=crop_origin_counts(frame,336,448,x_increasing=False,y_increasing=True)
    repeated=points[points.clips==2].iloc[0]
    assert repeated.north_up_y==112 and repeated.west_left_x==224
    assert repeated.clip_fraction_percent==200/3
    assert points.clips.sum()==3 and len(points)==2
    for x_increasing in (False,True):
        for y_increasing in (False,True):
            point=crop_origin_counts(frame.iloc[[2]],336,448,
                x_increasing=x_increasing,y_increasing=y_increasing).iloc[0]
            assert point.north_up_y==(0 if y_increasing else 112)
            assert point.west_left_x==112


def test_geographic_origins_align_different_native_grids(monkeypatch,tmp_path):
    import pandas as pd
    import xarray as xr
    x=np.arange(448)*.1-20
    first=xr.Dataset(coords={'x':x[::-1],'y':np.arange(336)*.1+30},attrs={'projection_wkt':'EPSG:4326'})
    second=xr.Dataset(coords={'x':x,'y':(np.arange(337)*.1+30)[::-1]},attrs={'projection_wkt':'EPSG:4326'})
    monkeypatch.setattr(xr,'open_zarr',lambda path,**kwargs: first if path.name=='a.zarr' else second)
    frame=pd.DataFrame({'store':['a.zarr','b.zarr'],'origin_y':[0,113],'origin_x':[0,224]})
    points=crop_origin_locations(frame,tmp_path)
    assert len(points)==1 and points.clips.iloc[0]==2
    np.testing.assert_allclose(points[['longitude','latitude']].iloc[0],[2.4,52.3])
    assert points.clip_fraction_percent.iloc[0]==100


def test_advected_history_preserves_translation_and_reports_border_support():
    torch.manual_seed(17)
    first=260+torch.randn(1,7,32,32)
    history=torch.stack([torch.roll(first,shift,-1) for shift in range(10)],dim=1)
    advected,support=purely_advected_history(history,return_support=True)
    torch.testing.assert_close(advected[...,9:],history[...,9:],atol=1e-4,rtol=0)
    torch.testing.assert_close(support,torch.tensor([1-4.5/32]))
