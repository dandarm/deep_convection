from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from plot_orbital_pixel_feature_maps import nearest_index, rasterize_components


def test_nearest_index_uses_nearest_grid_centre_and_breaks_ties_left():
    grid = np.array([0.0, 1.0, 2.0])
    result = nearest_index(np.array([0.49, 0.50, 1.51, 3.0]), grid)
    assert result.tolist() == [0, 0, 2, 2]


def test_component_resampling_keeps_union_and_majority_component_on_collision():
    frame = pd.DataFrame(
        {
            "lat": [0.01, 0.02, 0.03, 1.01],
            "lon": [0.01, 0.02, 0.03, 1.01],
            "component_uid": ["GMI:1:A", "GMI:1:A", "GMI:1:B", "GMI:1:B"],
        }
    )
    union, components, identifiers = rasterize_components(
        frame, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    assert union.sum() == 2
    assert identifiers == ["GMI:1:A", "GMI:1:B"]
    assert components[0, 0] == 0.0
    assert components[1, 1] == 1.0
