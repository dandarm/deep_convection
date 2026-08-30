from pathlib import Path

from emma_gpm.config import resolve_project_path


def test_data_tree_can_be_relocated(monkeypatch):
    monkeypatch.setenv("EMMA_GPM_DATA_ROOT", "/datasets/emma-gpm")
    assert resolve_project_path("/repo", "data/raw/emma/file.nc") == Path(
        "/datasets/emma-gpm/raw/emma/file.nc"
    )


def test_repository_outputs_stay_in_project(monkeypatch):
    monkeypatch.setenv("EMMA_GPM_DATA_ROOT", "/datasets/emma-gpm")
    assert resolve_project_path("/repo", "catalogs/output.parquet") == Path(
        "/repo/catalogs/output.parquet"
    )
