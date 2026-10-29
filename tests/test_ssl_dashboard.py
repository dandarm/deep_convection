import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from emma_gpm.seviri import THERMAL_CHANNELS
from emma_gpm.ssl_dashboard import forecast_metrics_frame,comparison_figure


def write_saved_reports(root):
    plan={'stages':[{'name':'future','kind':'pretraining'},
                    {'name':'probe_mvm','kind':'frozen_common_head'}]}
    (root/'comparison_plan.json').write_text(json.dumps(plan))
    def score(samples,factor=1):
        return {'samples':samples,'per_channel_horizon':[
            dict(channel=channel,horizon_minutes=horizon,mse_k2=mse*factor,
                 rmse_k=np.sqrt(mse*factor))
            for horizon,mse in [(15,4),(30,16)] for channel in THERMAL_CHANNELS]}
    for name in ('future','probe_mvm'):
        (root/name).mkdir()
        report={method:score(512) for method in ('persistence','advection','neural')}
        report['by_historical_variability']={'high':{
            method:score(141,2) for method in ('persistence','advection','neural')}}
        (root/name/'validation_latest.json').write_text(json.dumps(report))


def test_frozen_probe_comparison_preserves_common_reference_population(tmp_path):
    write_saved_reports(tmp_path)
    frame=forecast_metrics_frame(tmp_path,'frozen_common_head','high')
    assert set(frame.run)=={'persistence','advection','probe_mvm'}
    assert frame.groupby('run').size().to_dict()=={'advection':14,'persistence':14,'probe_mvm':14}
    assert set(frame.samples)=={141}
    assert set(frame.mse_k2)=={8,32}


def test_global_rmse_is_derived_from_pooled_squared_error(tmp_path):
    write_saved_reports(tmp_path)
    figure,table=comparison_figure(tmp_path,'pretraining')
    try:
        assert np.isclose(table.loc['Future','Global RMSE (K)'],np.sqrt(10))
        assert table.loc['Future','Number of clips']==512
        assert len(figure.axes)==2
    finally:
        plt.close(figure)


@pytest.mark.parametrize('family', ['pretraining','frozen_common_head'])
@pytest.mark.parametrize('variability', ['all','high'])
def test_forecast_horizons_share_vertical_scale(tmp_path,family,variability):
    write_saved_reports(tmp_path)
    figure,_=comparison_figure(tmp_path,family,variability=variability)
    try:
        first,second=figure.axes
        assert first.get_ylim()==second.get_ylim()
        assert first.get_ylim()[0]==0
        assert first.get_shared_y_axes().joined(first,second)
        tallest=max(bar.get_height() for axis in figure.axes for bar in axis.patches)
        assert first.get_ylim()[1]>tallest
        assert np.array_equal(first.get_yticks(),second.get_yticks())
    finally:
        plt.close(figure)
