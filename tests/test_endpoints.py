"""Tests for the local-archive endpoint loader.

These use synthetic files, so they run in CI where no archive exists. The
archives themselves are checked on xen1 by ``check-local-benchmarks``.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from xinyenyana.archive import canonical_sha256, sha256_file
from xinyenyana.endpoints import (
    BENCHMARK_ROOT_VARIABLE,
    DEFAULT_BENCHMARK_ROOT,
    RECORDED_FILES,
    ROOKID_ARCHIVE,
    benchmark_root,
    check_recorded_files,
    check_rookid_members,
    extract_rookid_annotations,
    read_csv_column,
)

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_benchmark_root_defaults_and_honours_the_variable(monkeypatch) -> None:
    monkeypatch.delenv(BENCHMARK_ROOT_VARIABLE, raising=False)
    assert benchmark_root() == DEFAULT_BENCHMARK_ROOT
    monkeypatch.setenv(BENCHMARK_ROOT_VARIABLE, "/somewhere/else")
    assert benchmark_root() == Path("/somewhere/else")


def test_every_recorded_file_names_a_result_document_and_a_full_digest() -> None:
    assert RECORDED_FILES, "the table must not be empty"
    for entry in RECORDED_FILES:
        assert len(entry.sha256) == 64, entry.relative_path
        assert int(entry.sha256, 16) >= 0, entry.relative_path
        assert entry.recorded_in == "docs/measured.md", entry.relative_path


def test_sha256_file_reads_a_file_larger_than_one_block(tmp_path: Path) -> None:
    import hashlib

    payload = bytes(range(256)) * 20_000
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    assert sha256_file(path, block=4096) == hashlib.sha256(payload).hexdigest()


def test_check_recorded_files_reports_a_missing_file_as_not_ok(tmp_path: Path) -> None:
    results = check_recorded_files(root=tmp_path)
    assert results, "the table must produce at least one check"
    assert all(not check.present for check in results)
    assert all(not check.ok for check in results)
    assert all(check.actual_sha256 is None for check in results)


def test_check_recorded_files_reports_a_wrong_digest_as_not_ok(tmp_path: Path) -> None:
    entry = RECORDED_FILES[0]
    target = tmp_path / entry.relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    results = {check.relative_path: check for check in check_recorded_files(root=tmp_path)}
    check = results[entry.relative_path]
    assert check.present
    assert check.actual_sha256 == EMPTY_SHA256
    assert not check.ok


def _rookid_protocol(tsv: bytes, wav: bytes, stem: str) -> dict:
    import zlib

    return {
        "source": {
            "recordings": [
                {
                    "stem": stem,
                    "split": "enrollment",
                    "tsv_bytes": len(tsv),
                    "tsv_crc32": f"{zlib.crc32(tsv) & 0xFFFFFFFF:08x}",
                    "wav_bytes": len(wav),
                    "wav_crc32": f"{zlib.crc32(wav) & 0xFFFFFFFF:08x}",
                }
            ]
        }
    }


def _write_rookid_archive(root: Path, stem: str, tsv: bytes, wav: bytes) -> None:
    archive = root / ROOKID_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"RookID/{stem}.tsv", tsv)
        bundle.writestr(f"RookID/{stem}.wav", wav)


def test_check_rookid_members_passes_on_matching_size_and_crc(tmp_path: Path) -> None:
    tsv, wav, stem = b"Source\tEvent\n", b"RIFFfake", "20200214_090050"
    _write_rookid_archive(tmp_path, stem, tsv, wav)
    checks = check_rookid_members(protocol=_rookid_protocol(tsv, wav, stem), root=tmp_path)
    assert len(checks) == 2
    assert all(check.ok for check in checks)


def test_check_rookid_members_fails_when_a_member_differs(tmp_path: Path) -> None:
    tsv, wav, stem = b"Source\tEvent\n", b"RIFFfake", "20200214_090050"
    protocol = _rookid_protocol(tsv, wav, stem)
    _write_rookid_archive(tmp_path, stem, tsv, b"RIFFdiff")
    checks = {
        check.member: check for check in check_rookid_members(protocol=protocol, root=tmp_path)
    }
    assert checks[f"RookID/{stem}.tsv"].ok
    wav_check = checks[f"RookID/{stem}.wav"]
    assert not wav_check.ok
    assert wav_check.actual_crc32 != wav_check.expected_crc32


def test_check_rookid_members_reports_an_absent_member(tmp_path: Path) -> None:
    tsv, wav, stem = b"Source\tEvent\n", b"RIFFfake", "20200214_090050"
    protocol = _rookid_protocol(tsv, wav, stem)
    archive = tmp_path / ROOKID_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"RookID/{stem}.tsv", tsv)
    checks = {
        check.member: check for check in check_rookid_members(protocol=protocol, root=tmp_path)
    }
    assert not checks[f"RookID/{stem}.wav"].present
    assert not checks[f"RookID/{stem}.wav"].ok


def test_extract_rookid_annotations_writes_the_registered_tsvs(tmp_path: Path) -> None:
    tsv, wav, stem = b"Source\tEvent\tComment\tStart\tEnd\n", b"RIFFfake", "20200214_090050"
    _write_rookid_archive(tmp_path, stem, tsv, wav)
    destination = tmp_path / "work"
    written = extract_rookid_annotations(
        protocol=_rookid_protocol(tsv, wav, stem), destination=destination, root=tmp_path
    )
    assert written == [destination / "sources" / f"{stem}.tsv"]
    assert written[0].read_bytes() == tsv


def test_canonical_sha256_matches_the_e01d_helper() -> None:
    from xinyenyana.e01d import _canonical_sha256

    value = [{"b": 2, "a": [1, "x"]}, {"c": None}]
    assert canonical_sha256(value) == _canonical_sha256(value)
    assert canonical_sha256(value) != canonical_sha256(json.loads(json.dumps(value))[:1])


def test_read_csv_column_returns_the_column_in_file_order(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("ring,year\nB1,2021\nB2,2022\n", encoding="utf-8")
    assert read_csv_column(path, "ring") == ["B1", "B2"]
    with pytest.raises(ValueError):
        read_csv_column(path, "missing")
