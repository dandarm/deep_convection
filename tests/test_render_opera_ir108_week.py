import pandas as pd

from scripts.render_opera_ir108_week import minimum_gap_sample


def test_minimum_gap_sample_keeps_chronological_frames() -> None:
    times = pd.date_range("2020-05-01", periods=9, freq="15min")
    selected = minimum_gap_sample(times, 60)
    assert selected.tolist() == [times[0], times[4], times[8]]
