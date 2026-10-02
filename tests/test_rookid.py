"""Tests for reading the RookID annotation files.

Every RookID clip's identity, time and split comes from these tab-separated
files, so a silent misreading of one would put the wrong bird on a clip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xinyenyana.rookid import _safe_path, read_rookid_annotations

PROTOCOL = {
    "source": {
        "recordings": [
            {"stem": "20200214_090050", "split": "enrollment"},
            {"stem": "20200302_091000", "split": "query"},
        ]
    }
}

HEADER = "Source\tEvent\tComment\tStart\tEnd\n"


def _write(root: Path, stem: str, rows: str) -> None:
    path = root / "sources" / f"{stem}.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + rows)


def test_rows_keep_their_identity_time_split_and_source_line(tmp_path: Path) -> None:
    _write(tmp_path, "20200214_090050", "Balbo\tapb\tstill\t1.5\t1.9\nBashir\tapb\t\t2.0\t2.4\n")
    _write(tmp_path, "20200302_091000", "Brain\tapb\t\t3.0\t3.5\n")
    annotations = read_rookid_annotations(protocol=PROTOCOL, source_root=tmp_path)

    assert sorted(annotations) == ["20200214_090050", "20200302_091000"]
    first, second = annotations["20200214_090050"]
    assert (first.identity, first.event, first.comment) == ("Balbo", "apb", "still")
    assert (first.start, first.end) == (1.5, 1.9)
    assert first.split == "enrollment"
    # Row 1 is the header, so the first data row is line 2 of the file.
    assert (first.row_number, second.row_number) == (2, 3)
    assert annotations["20200302_091000"][0].split == "query"


def test_a_file_with_a_different_schema_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, "20200214_090050", "")
    (tmp_path / "sources" / "20200214_090050.tsv").write_text(
        "Bird\tEvent\tComment\tStart\tEnd\nBalbo\tapb\t\t1.0\t2.0\n"
    )
    with pytest.raises(ValueError, match="unexpected RookID TSV schema"):
        read_rookid_annotations(
            protocol={"source": {"recordings": [{"stem": "20200214_090050", "split": "e"}]}},
            source_root=tmp_path,
        )


@pytest.mark.parametrize("interval", ["-0.5\t1.0", "2.0\t2.0", "3.0\t1.0"])
def test_an_impossible_interval_is_refused(tmp_path: Path, interval: str) -> None:
    _write(tmp_path, "20200214_090050", f"Balbo\tapb\t\t{interval}\n")
    with pytest.raises(ValueError, match="invalid RookID interval"):
        read_rookid_annotations(
            protocol={"source": {"recordings": [{"stem": "20200214_090050", "split": "e"}]}},
            source_root=tmp_path,
        )


def test_a_path_that_escapes_the_root_is_refused(tmp_path: Path) -> None:
    assert _safe_path(tmp_path, "sources/a.tsv") == tmp_path / "sources" / "a.tsv"
    with pytest.raises(ValueError):
        _safe_path(tmp_path, "../escaped.tsv")
    with pytest.raises(ValueError):
        _safe_path(tmp_path, "/etc/passwd")
