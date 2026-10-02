"""Tests for making a result survive the machine that produced it."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from xinyenyana.archive import (
    archive_object_name,
    archive_result,
    sha256_file,
    source_digest,
)

PREFIX = "gs://example-bucket/results"


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def _recorder(returncode: int = 0, stderr: str = "") -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []

    def run(command: Any) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return _completed(returncode, stderr)

    return calls, run


# --- the source digest ------------------------------------------------------


def test_the_source_digest_is_stable_across_calls(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("x = 1\n")
    (tmp_path / "two.py").write_text("y = 2\n")
    assert source_digest(tmp_path) == source_digest(tmp_path)


def test_changing_one_line_of_source_changes_the_digest(tmp_path: Path) -> None:
    """This is what a commit hash stops doing once the branch is squash-merged."""

    (tmp_path / "one.py").write_text("x = 1\n")
    before = source_digest(tmp_path)
    (tmp_path / "one.py").write_text("x = 2\n")
    assert source_digest(tmp_path) != before


def test_renaming_a_file_changes_the_digest(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("x = 1\n")
    before = source_digest(tmp_path)
    (tmp_path / "one.py").rename(tmp_path / "renamed.py")
    assert source_digest(tmp_path) != before


def test_a_tree_with_no_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no source files"):
        source_digest(tmp_path)


# --- where a result goes ----------------------------------------------------


def test_a_result_is_named_by_the_digest_of_its_own_bytes(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"accuracy": 0.44}\n')
    assert archive_object_name(path, prefix=PREFIX) == f"{PREFIX}/{sha256_file(path)}.json"


def test_two_results_with_the_same_content_share_one_object(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text('{"accuracy": 0.44}\n')
    second.write_text('{"accuracy": 0.44}\n')
    assert archive_object_name(first, prefix=PREFIX) == archive_object_name(second, prefix=PREFIX)


def test_a_changed_result_is_a_different_object(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"accuracy": 0.44}\n')
    before = archive_object_name(path, prefix=PREFIX)
    path.write_text('{"accuracy": 0.45}\n')
    assert archive_object_name(path, prefix=PREFIX) != before


# --- the upload -------------------------------------------------------------


def test_the_upload_refuses_to_replace_anything(tmp_path: Path) -> None:
    """A generation-zero precondition prevents replacing an existing object."""

    path = tmp_path / "result.json"
    path.write_text('{"accuracy": 0.44}\n')
    calls, run = _recorder()
    archive_result(path, prefix=PREFIX, run=run)
    assert calls[0][:3] == ["gcloud", "storage", "cp"]
    assert "--if-generation-match=0" in calls[0]
    assert calls[0][-1] == archive_object_name(path, prefix=PREFIX)


def test_an_upload_is_only_reported_archived_once_the_object_is_there(tmp_path: Path) -> None:
    """A successful send is not a later read. The object is checked for."""

    path = tmp_path / "result.json"
    path.write_text('{"accuracy": 0.44}\n')
    seen: list[list[str]] = []

    def run(command: Any) -> subprocess.CompletedProcess[str]:
        seen.append(list(command))
        return _completed(0 if command[2] == "cp" else 1)

    outcome = archive_result(path, prefix=PREFIX, run=run)
    assert [command[2] for command in seen] == ["cp", "objects"]
    assert outcome["archived"] is False


def test_a_failed_upload_reports_what_the_tool_said(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"accuracy": 0.44}\n')
    _, run = _recorder(returncode=1, stderr="AccessDeniedException: 403")
    outcome = archive_result(path, prefix=PREFIX, run=run)
    assert outcome["archived"] is False
    assert "403" in outcome["error"]


def test_archiving_a_file_that_is_not_there_is_refused(tmp_path: Path) -> None:
    _, run = _recorder()
    with pytest.raises(FileNotFoundError):
        archive_result(tmp_path / "absent.json", prefix=PREFIX, run=run)


@pytest.mark.parametrize(
    "different_size,different_checksum", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("copy_returncode", [0, 1])
def test_archive_verification_checks_the_bytes_not_just_the_name(
    tmp_path: Path, different_size: bool, different_checksum: bool, copy_returncode: int
) -> None:
    path = tmp_path / "result.json"
    content = b'{"synthetic":true}\n'
    path.write_bytes(content)
    metadata = {
        "generation": "12345",
        "size": len(content) + int(different_size),
        "md5_hash": base64.b64encode(hashlib.md5(content, usedforsecurity=False).digest()).decode(),
    }
    if different_checksum:
        metadata["md5_hash"] = "different"

    def run(command: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0 if command[2] == "objects" else copy_returncode,
            stdout=json.dumps(metadata) if command[2] == "objects" else "",
            stderr="",
        )

    outcome = archive_result(path, prefix=PREFIX, run=run)
    assert outcome["archived"] is (not different_size and not different_checksum)
    if outcome["archived"]:
        assert outcome["generation"] == "12345"


def test_malformed_success_response_is_not_verified(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}")
    _, run = _recorder()
    assert archive_result(path, prefix=PREFIX, run=run)["archived"] is False


@pytest.mark.parametrize("upload_status", [200, 412])
def test_token_archive_uses_exact_objects_without_listing(
    tmp_path: Path, monkeypatch: Any, upload_status: int
) -> None:
    path = tmp_path / "result.json"
    content = b'{"synthetic":true}\n'
    path.write_bytes(content)
    token = tmp_path / "token"
    token.write_text("test-token")
    monkeypatch.setenv("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", str(token))
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.host == "storage.googleapis.com"
        assert request.headers["Authorization"] == "Bearer test-token"
        if request.method == "POST":
            assert request.url.params["ifGenerationMatch"] == "0"
            assert request.url.params["name"] == f"results/{sha256_file(path)}.json"
            assert request.read() == content
            return httpx.Response(upload_status, json={})
        assert request.method == "GET"
        assert request.url.path.endswith(f"/o/results/{sha256_file(path)}.json")
        return httpx.Response(
            200,
            json={
                "size": str(len(content)),
                "generation": "12345",
                "md5Hash": base64.b64encode(
                    hashlib.md5(content, usedforsecurity=False).digest()
                ).decode(),
            },
        )

    outcome = archive_result(path, prefix=PREFIX, transport=httpx.MockTransport(handle))
    assert outcome["archived"] is True
    assert len(calls) == 2


def test_expired_token_does_not_report_an_archived_result(tmp_path: Path, monkeypatch: Any) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}")
    token = tmp_path / "token"
    token.write_text("expired-test-token")
    monkeypatch.setenv("CLOUDSDK_AUTH_ACCESS_TOKEN_FILE", str(token))
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={}))
    outcome = archive_result(path, prefix=PREFIX, transport=transport)
    assert outcome["archived"] is False
    assert "401" in outcome["error"]
    assert "expired-test-token" not in outcome["error"]
