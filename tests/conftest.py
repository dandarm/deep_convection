"""Keep simulated acquisition/writer tests independent of host disk capacity."""
from collections import namedtuple
from pathlib import Path
import shutil

import pytest


@pytest.fixture(autouse=True)
def simulated_pipeline_disk_capacity(request, monkeypatch):
    # These modules simulate archive writes using tiny temporary payloads. Their
    # unrelated control-flow assertions require a deterministic capacity value.
    # Production capacity checks and other tests retain the real disk report.
    if request.path.name not in {
        'test_acquire_seviri_year.py', 'test_rdt_catalog_control.py', 'test_rdt_execution.py'
    }:
        return
    directory = request.getfixturevalue('tmp_path').resolve()
    original = shutil.disk_usage
    usage = namedtuple('usage', 'total used free')

    def capacity(path):
        if Path(path).resolve().is_relative_to(directory):
            return usage(200*2**30, 100*2**30, 100*2**30)
        return original(path)

    monkeypatch.setattr(shutil, 'disk_usage', capacity)
