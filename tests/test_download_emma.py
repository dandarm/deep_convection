import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from emma_gpm import download
from scripts.download_emma import FIRST_EMMA_YEAR, LAST_EMMA_YEAR, parse_years


class MetadataResponse:
    def __init__(self, item):
        self.item = item

    def raise_for_status(self):
        pass

    def json(self):
        return {"files": [self.item]}


def test_parse_years_defaults_deduplicates_and_expands_all():
    assert parse_years(None, 2020) == [2020]
    assert parse_years(["2021", "2020", "2021"], 2020) == [2020, 2021]
    assert parse_years(["all"], 2020) == list(
        range(FIRST_EMMA_YEAR, LAST_EMMA_YEAR + 1)
    )
    with pytest.raises(SystemExit, match="invalid"):
        parse_years([str(FIRST_EMMA_YEAR - 1)], 2020)


def test_fetch_emma_year_reuses_verified_archive_and_existing_extraction(
    tmp_path: Path, monkeypatch
):
    raw = tmp_path / "raw"
    extracted = tmp_path / "extracted"
    raw.mkdir()
    archive = raw / "emma_2020_test.tar.gz"
    payload = b"verified EMMA payload"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("2020/data.bin")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    checksum = hashlib.md5(archive.read_bytes()).hexdigest()
    item = {
        "key": archive.name,
        "checksum": f"md5:{checksum}",
        "links": {"self": "https://example.invalid/archive"},
    }
    monkeypatch.setattr(download.requests, "get", lambda *args, **kwargs: MetadataResponse(item))
    monkeypatch.setattr(
        download,
        "stream_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert download.fetch_emma_year(1, 2020, raw, extracted) == archive
    target = extracted / "2020" / "data.bin"
    assert target.read_bytes() == payload
    target.write_bytes(b"already extracted")
    download.fetch_emma_year(1, 2020, raw, extracted)
    assert target.read_bytes() == b"already extracted"
