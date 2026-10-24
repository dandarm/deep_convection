from datetime import datetime

from scripts.download_msg_seviri import matches_cadence


def test_rss_sensing_start_uses_nominal_five_minute_slot():
    assert matches_cadence(datetime(2022, 1, 1, 0, 4, 14), 15)
    assert not matches_cadence(datetime(2022, 1, 1, 0, 9, 14), 15)
    assert not matches_cadence(datetime(2022, 1, 1, 0, 14, 14), 15)
    assert matches_cadence(datetime(2022, 1, 1, 0, 19, 14), 15)
    assert matches_cadence(datetime(2022, 1, 1, 1, 4, 14), 15)


def test_zero_cadence_keeps_every_product():
    assert matches_cadence(datetime(2022, 1, 1, 0, 9, 14), 0)
