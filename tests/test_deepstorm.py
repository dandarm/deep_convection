from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from emma_gpm.deepstorm import (
    GMI_TB_COLUMNS,
    aggregate_csatgpm_arrays,
    decode_cloud_type,
    read_csatgpm_file,
)


def test_cloud_scenario_bits_one_to_four_hold_cloud_type():
    encoded = np.array([[1 | (8 << 1), 0], [1 | (1 << 1) | (3 << 5), -999]])
    result = decode_cloud_type(encoded)
    assert result[0, 0] == 8.0
    assert np.isnan(result[0, 1])
    assert result[1, 0] == 1.0
    assert np.isnan(result[1, 1])


def test_repeated_cloudsat_rays_are_aggregated_to_unique_gmi_footprints():
    tb = np.vstack([np.arange(13) + 200, np.arange(13) + 200, np.arange(13) + 230])
    result = aggregate_csatgpm_arrays(
        source_file="2B.CSATGPM.COIN.X.20140505-S011851-E012159.001032.V04.nc4",
        scan_index=np.array([10, 10, 11]),
        pixel_index=np.array([20, 20, 21]),
        latitude=np.array([40.0, 40.1, 50.0]),
        longitude=np.array([10.0, 10.1, 20.0]),
        brightness_temperature=tb,
        ice_water_path_kg_m2=np.array([0.2, 0.8, np.nan]),
        cloud_scenario=np.array([[0, 1 | (8 << 1)], [1 | (1 << 1), 0], [1 | (2 << 1), 1 | (3 << 1)]]),
        observation_time=pd.date_range("2014-05-05", periods=3, freq="s", tz="UTC"),
    )
    assert len(result) == 2
    first = result.iloc[0]
    assert first.n_cloudsat_rays == 2
    assert first.target_dc_any == 1
    assert first.target_dc_fraction == 0.5
    assert first.target_iwp_max_kg_m2 == 0.8
    assert first.target_iwp_mean_kg_m2 == 0.5
    assert first.target_iwp_gt_0p5 == 1
    assert first.tb_repeat_max_abs_diff_k == 0
    assert first[GMI_TB_COLUMNS[0]] == 200
    second = result.iloc[1]
    assert pd.isna(second.target_iwp_gt_0p5)


def test_reader_handles_v04_style_groups_and_iwp_units(tmp_path: Path):
    path = tmp_path / "2B.CSATGPM.COIN.X.20140505-S011851-E012159.001032.V04.nc4"
    with h5py.File(path, "w") as handle:
        cs = handle.create_group("CS")
        cs.create_dataset("Latitude", data=[40.0, 40.1])
        cs.create_dataset("Longitude", data=[10.0, 10.1])
        cs.create_dataset("scan_index_S1", data=[2, 2])
        cs.create_dataset("pix_index_S1", data=[3, 3])
        cs.create_dataset("Profile_time", data=[0.0, 1.0])
        cs.create_dataset("Tb", data=np.tile(np.arange(13) + 200.0, (2, 1)))
        ice = handle.create_group("2C-ICE")
        iwp = ice.create_dataset("ice_water_path", data=[200.0, 800.0])
        iwp.attrs["units"] = "g m-2"
        cld = handle.create_group("2B-CLDCLASS")
        cld.create_dataset("cloud_scenario", data=[[0, 1 | (8 << 1)], [1 | (1 << 1), 0]])

    result = read_csatgpm_file(path)
    assert len(result) == 1
    assert result.loc[0, "target_iwp_max_kg_m2"] == 0.8
    assert result.loc[0, "target_dc_any"] == 1
    assert result.loc[0, "observation_time"] == pd.Timestamp("2014-05-05 01:18:51.5Z")


def test_reader_handles_v05_combined_scan_indices_and_unix_time(tmp_path: Path):
    path = tmp_path / "2B.CSATGPM.COIN.COIN2022.20190114-S193327-E193913.027724.V05.nc4"
    with h5py.File(path, "w") as handle:
        geog = handle.create_group("2B-GEOPROF")
        geog.create_dataset("Latitude", data=[-50.1, -50.2])
        geog.create_dataset("Longitude", data=[-77.1, -77.2])
        gmi = handle.create_group("1C-R.GPM.GMI")
        gmi.create_dataset("scan_indices", data=[[12, 34], [12, 34]])
        scan_time = gmi.create_dataset("scan_time", data=[1547495104.0, 1547495104.0])
        scan_time.attrs["units"] = "seconds"
        s1 = gmi.create_group("S1")
        s1.create_dataset("Latitude", data=[40.0, 40.0])
        s1.create_dataset("Longitude", data=[10.0, 10.0])
        s1.create_dataset("Tc", data=np.tile(np.arange(13) + 200.0, (2, 1)))
        ice = handle.create_group("2C-ICE")
        iwp = ice.create_dataset("ice_water_path", data=[-7777.0, 700.0])
        iwp.attrs["units"] = "g/m^2"
        cld = handle.create_group("2B-CLDCLASS")
        cld.create_dataset("cloud_scenario", data=[[0, 1 | (8 << 1)], [1, 0]])

    result = read_csatgpm_file(
        path,
        domain={"lat_min": 25.0, "lat_max": 55.0, "lon_min": -10.0, "lon_max": 45.0},
    )
    assert len(result) == 1
    assert result.loc[0, "gmi_scan"] == 12
    assert result.loc[0, "gmi_pixel"] == 34
    assert result.loc[0, "latitude"] == 40.0
    assert result.loc[0, "target_iwp_valid_rays"] == 1
    assert result.loc[0, "target_iwp_max_kg_m2"] == 0.7
    assert result.loc[0, "observation_time"] == pd.Timestamp(1547495104, unit="s", tz="UTC")
