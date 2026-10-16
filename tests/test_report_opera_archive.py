import pandas as pd

from scripts.report_opera_archive import monthly_summary


def test_monthly_summary_uses_leap_year_slot_counts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2020-01-01T00:00", "2020-02-29T23:45"]
            ),
            "bytes": [10, 20],
        }
    )
    summary = monthly_summary(frame, 2020).set_index("month")
    assert summary.loc[1, "expected_slots"] == 31 * 96
    assert summary.loc[2, "expected_slots"] == 29 * 96
    assert summary.loc[2, "files"] == 1
