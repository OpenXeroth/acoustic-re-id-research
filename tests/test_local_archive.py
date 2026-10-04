"""Independent runs keep evidence without contacting the authors' infrastructure."""

import json

import pytest
import typer

from xinyenyana import cli
from xinyenyana.archive import archive_result_locally, sha256_file


def test_local_archive_is_content_addressed_idempotent_and_never_overwrites(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"accuracy": 0.5}\n')
    archive = tmp_path / "archive"
    receipt = archive_result_locally(result, archive)
    assert receipt == archive_result_locally(result, archive)
    target = archive / f"{sha256_file(result)}.json"
    assert target.read_bytes() == result.read_bytes()
    target.write_text("corrupt")
    with pytest.raises(ValueError, match="not overwritten"):
        archive_result_locally(result, archive)
    assert target.read_text() == "corrupt"


def test_independent_mode_keeps_provenance_without_a_fabricated_lease(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_PROVENANCE", {})
    monkeypatch.delenv("XENWARDEN_LEASE", raising=False)
    monkeypatch.delenv("XINYENYANA_RESULT_ARCHIVE", raising=False)
    cli._snapshot_provenance(local_archive=tmp_path / "archive")
    cli._require_lease_and_archive()
    output = tmp_path / "result.json"
    cli._emit({"synthetic": True}, output, require_archive=True)
    result = json.loads(output.read_text())
    assert result["source_sha256"]
    assert result["execution_mode"].startswith("independent compute")
    assert (
        tmp_path / "archive" / f"{sha256_file(output)}.json"
    ).read_bytes() == output.read_bytes()
    # A subsequent command must return to the unchanged managed-compute policy.
    cli._snapshot_provenance()
    with pytest.raises(typer.BadParameter, match="XenWarden"):
        cli._require_lease_and_archive()
