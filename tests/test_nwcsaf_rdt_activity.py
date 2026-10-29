import io
from datetime import timedelta

import numpy as np
import pandas as pd
from PIL import Image

import scripts.rank_nwcsaf_rdt_activity as ranker
import scripts.scan_nwcsaf_rdt_activity as scanner
import scripts.summarize_nwcsaf_rdt_candidates as summarizer


def test_activity_analysis_counts_target_components_and_excludes_split(monkeypatch):
    monkeypatch.setitem(
        scanner.EXPECTED_LAYOUTS,
        "v2021",
        {"size": (32, 24), "map_bbox": (2, 2, 30, 20)},
    )
    image = np.full((24, 32, 3), 128, dtype=np.uint8)
    image[5, 5:12] = scanner.PHASE_RGB["triggering"]
    image[8, 15:23] = scanner.PHASE_RGB["mature"]
    image[12, 6:15] = scanner.PHASE_RGB["triggering_from_split"]
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="GIF")
    result = scanner.analyse_gif_bytes(stream.getvalue(), "v2021", color_tolerance=0)
    assert result["triggering_components"] == 1
    assert result["mature_components"] == 1
    assert result["triggering_from_split_components"] == 1
    assert result["target_components"] == 2
    assert result["phase_richness"] == 2
    assert len([row for row in __import__("json").loads(result["objects_json"])
                if row["phase"] != "triggering_from_split"]) == 2


def test_window_ranking_prefers_sustained_multiphase_activity():
    times = pd.date_range("2025-06-01", periods=8, freq="1h")
    rows = []
    for index, timestamp in enumerate(times):
        active = 2 <= index <= 5
        rows.append(
            {
                "period": "p",
                "archive": "v2021",
                "time": timestamp,
                "status": "ok",
                "target_line_pixels": 400 if active else 0,
                "target_components": 8 if active else 0,
                "phase_richness": 4 if active else 0,
                "triggering_components": 2 if active else 0,
                "growing_components": 2 if active else 0,
                "mature_components": 3 if active else 0,
                "decay_components": 1 if active else 0,
                "objects_json": "[]",
            }
        )
    windows = ranker.build_windows(pd.DataFrame(rows), window_hours=3)
    best = windows.iloc[0]
    assert best.start == pd.Timestamp("2025-06-01T02:00:00")
    assert best.integrated_components == 24
    assert best.phase_presence == 4
    selected = ranker.balanced_candidates(windows, count=1, separation_hours=3)
    assert selected.iloc[0].start == best.start


def test_candidate_summary_measures_persistence():
    rows = []
    for period, components in (("candidate_001", [0, 2, 3, 0]), ("candidate_002", [1, 1, 1, 1])):
        for index, count in enumerate(components):
            rows.append(
                {
                    "period": period,
                    "archive": "v2025_msg",
                    "time": pd.Timestamp("2026-07-01") + timedelta(minutes=15 * index),
                    "status": "ok",
                    "target_components": count,
                    "target_line_pixels": count * 10,
                    "triggering_components": count,
                    "growing_components": 0,
                    "mature_components": 0,
                    "decay_components": 0,
                }
            )
    summary = summarizer.summarize(pd.DataFrame(rows))
    persistent = summary.set_index("period").loc["candidate_002"]
    assert persistent.active_fraction_available == 1.0
    assert persistent.longest_active_run_slots == 4
    assert persistent.integrated_components == 4
