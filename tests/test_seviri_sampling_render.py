from types import SimpleNamespace

import cartopy.crs as ccrs
import numpy as np
from pyproj import CRS, Transformer
import pytest

from scripts.render_seviri_sampling_gif import geostationary_crs, native_rectangle, pixel_edges


def test_display_projection_matches_native_satellite_geolocation():
    source = CRS.from_proj4('+proj=geos +lon_0=9.5 +h=35785831 +a=6378169 '
                           '+b=6356583.8 +sweep=y +units=m +no_defs')
    display = geostationary_crs(source)
    lon = np.array([-20., 0., 20., 40.])
    lat = np.array([30., 50., 60., 70.])
    expected = np.column_stack(Transformer.from_crs('EPSG:4326',source,always_xy=True).transform(lon,lat))
    actual = display.transform_points(ccrs.PlateCarree(),lon,lat)[:,:2]
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-5)
    assert display.proj4_params['lon_0'] == 9.5
    assert display.proj4_params['h'] == 35785831


@pytest.mark.parametrize('reverse_x,reverse_y',[(False,False),(True,False),(False,True),(True,True)])
def test_tile_footprint_follows_native_pixel_edges_in_all_axis_directions(reverse_x,reverse_y):
    x = np.arange(10,dtype=float)*3000
    y = 1_000_000 + np.arange(8,dtype=float)*3000
    if reverse_x:x=x[::-1]
    if reverse_y:y=y[::-1]
    rectangle = native_rectangle(SimpleNamespace(origin_x=2,origin_y=1),pixel_edges(x),pixel_edges(y),size=3)
    np.testing.assert_allclose(rectangle.min(axis=0),[x[2:5].min()-1500,y[1:4].min()-1500])
    np.testing.assert_allclose(rectangle.max(axis=0),[x[2:5].max()+1500,y[1:4].max()+1500])
    assert np.ptp(rectangle[:,0]) == np.ptp(rectangle[:,1]) == 9000


def test_projection_and_grid_errors_are_explicit():
    with pytest.raises(ValueError,match='geostationary'):
        geostationary_crs('EPSG:3857')
    with pytest.raises(ValueError,match='Regular'):
        pixel_edges([0,1,3])
    with pytest.raises(ValueError,match='outside'):
        native_rectangle(SimpleNamespace(origin_x=1,origin_y=0),pixel_edges([0,1,2]),pixel_edges([0,1,2]),size=3)


def test_video_includes_previous_day_clips_and_excludes_unseen_tail_starts():
    import pandas as pd
    from scripts.render_seviri_sampling_video import visible_clip_rows
    from emma_gpm.ssl_spaced_sampling import active_clip_rows
    frame = pd.DataFrame(dict(start_time=['2022-12-31T23:30Z', '2023-01-01T00:00Z',
        '2023-01-01T23:45Z', '2023-01-01T23:50Z'], clip_id=['previous','first','last','after']))
    timestamps = pd.date_range('2023-01-01', periods=96, freq='15min', tz='UTC')
    visible = visible_clip_rows(frame, timestamps)
    assert visible.clip_id.tolist() == ['previous', 'first', 'last']
    displayed = set().union(*(set(active_clip_rows(frame, t).clip_id) for t in timestamps))
    assert set(visible.clip_id) == displayed
