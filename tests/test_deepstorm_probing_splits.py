from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location(
    "prepare_deepstorm_probing_splits", ROOT / "scripts" / "prepare_deepstorm_probing_splits.py"
)
assert SPEC and SPEC.loader
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_group_split_assignment_keeps_groups_whole_and_populates_every_split():
    rows = []
    for group, positives in {
        "a": 8,
        "b": 6,
        "c": 5,
        "d": 3,
        "e": 2,
        "f": 1,
        "g": 1,
        "h": 0,
    }.items():
        rows.extend(
            {
                "split_group": group,
                "target_iwp_gt_0p5": int(index < positives),
                "target_dc_any": int(index < max(0, positives // 3)),
            }
            for index in range(20)
        )
    frame = pd.DataFrame(rows)

    assignment = MODULE.assign_group_splits(
        frame,
        {"train": 0.6, "validation": 0.2, "test": 0.2},
        "target_iwp_gt_0p5",
    )

    assert set(assignment) == set(frame["split_group"])
    assert set(assignment.values()) == {"train", "validation", "test"}
    assert assignment["a"] in {"train", "validation", "test"}
