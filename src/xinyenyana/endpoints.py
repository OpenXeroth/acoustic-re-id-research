"""Local benchmark archives, and the endpoints Phase A builds from them.

Phase A reads its data from xen1's own disk. GCS holds the verified original of
every archive so a disk failure costs a re-download; this module reads the local
copy and checks it against a written digest, so a local file that differs is an
error rather than a wrong number.

Nothing here downloads. The archives are put in place once, by
``scripts/fetch_benchmarks.py`` or by ``gsutil cp`` from the bucket.
"""

from __future__ import annotations

import csv
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.archive import sha256_file

DEFAULT_BENCHMARK_ROOT = Path("/mnt/data/xinyenyana/benchmarks")
BENCHMARK_ROOT_VARIABLE = "XINYENYANA_BENCHMARK_ROOT"

_READ_BLOCK = 8 * 1024 * 1024


def benchmark_root() -> Path:
    """Return the directory holding the local archives."""

    value = os.environ.get(BENCHMARK_ROOT_VARIABLE)
    return Path(value) if value else DEFAULT_BENCHMARK_ROOT


@dataclass(frozen=True)
class RecordedFile:
    """A local file whose digest is written down.

    ``recorded_in`` names the document that carries it, so a mismatch is traced
    to a written value rather than argued about.
    """

    relative_path: str
    sha256: str
    recorded_in: str


# Only files whose digest is written down in a merged document appear here. A
# file with no recorded digest is not listed rather than listed with a digest
# computed today, which would check nothing.
RECORDED_FILES: tuple[RecordedFile, ...] = (
    RecordedFile(
        "zebra-finch-figshare-11905533/AdultVocalizations.zip",
        "498804bf357e49fa5b3f0d66b35d05430fea22907540a871f2e7423240b7b576",
        "docs/measured.md",
    ),
    RecordedFile(
        "wytham-great-tit-osf-n8ac9/main.csv",
        "497cc05221f93cef597f95390ea7a8122cd42d7da5dba6cc9ed0dab3d475f4d9",
        "docs/measured.md",
    ),
    RecordedFile(
        "wytham-great-tit-osf-n8ac9/morphometrics.csv",
        "67953ccd0a54dcd7a786b64bd0ab9ec347e40a4ef34a4c356ea2d12d7588ec4d",
        "docs/measured.md",
    ),
    RecordedFile(
        "wytham-great-tit-osf-n8ac9/great-tit-hits.csv",
        "580fee8440d7e213b7c8507903818d7748efaf4052ed94740a25ec8c1e5ccb80",
        "docs/measured.md",
    ),
)

ROOKID_ARCHIVE = "rookid-zenodo-6091940/RookID.zip"


@dataclass(frozen=True)
class FileCheck:
    """The outcome of checking one local file against its recorded digest."""

    relative_path: str
    present: bool
    expected_sha256: str
    actual_sha256: str | None
    recorded_in: str

    @property
    def ok(self) -> bool:
        return self.present and self.actual_sha256 == self.expected_sha256


def check_recorded_files(*, root: Path | None = None) -> list[FileCheck]:
    """Check every file with a recorded digest against the local copy."""

    base = root if root is not None else benchmark_root()
    results: list[FileCheck] = []
    for entry in RECORDED_FILES:
        path = base / entry.relative_path
        present = path.is_file()
        results.append(
            FileCheck(
                relative_path=entry.relative_path,
                present=present,
                expected_sha256=entry.sha256,
                actual_sha256=sha256_file(path) if present else None,
                recorded_in=entry.recorded_in,
            )
        )
    return results


@dataclass(frozen=True)
class MemberCheck:
    """The outcome of checking one archive member against a registered value."""

    archive: str
    member: str
    present: bool
    expected_bytes: int
    actual_bytes: int | None
    expected_crc32: str
    actual_crc32: str | None

    @property
    def ok(self) -> bool:
        return (
            self.present
            and self.actual_bytes == self.expected_bytes
            and self.actual_crc32 == self.expected_crc32
        )


def check_rookid_members(
    *, protocol: dict[str, Any], root: Path | None = None
) -> list[MemberCheck]:
    """Check the RookID members the E01d protocol registered.

    The protocol records each source recording's uncompressed size and CRC-32
    for both its TSV and its WAV. The zip's central directory carries the same
    two values, so this reads no audio and completes in under a second on a
    22 GB archive.
    """

    base = root if root is not None else benchmark_root()
    archive = base / ROOKID_ARCHIVE
    checks: list[MemberCheck] = []
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        for recording in protocol["source"]["recordings"]:
            stem = str(recording["stem"])
            for kind in ("tsv", "wav"):
                member = f"RookID/{stem}.{kind}"
                present = member in names
                info = bundle.getinfo(member) if present else None
                checks.append(
                    MemberCheck(
                        archive=ROOKID_ARCHIVE,
                        member=member,
                        present=present,
                        expected_bytes=int(recording[f"{kind}_bytes"]),
                        actual_bytes=info.file_size if info else None,
                        expected_crc32=str(recording[f"{kind}_crc32"]),
                        actual_crc32=f"{info.CRC:08x}" if info else None,
                    )
                )
    return checks


def extract_rookid_annotations(
    *, protocol: dict[str, Any], destination: Path, root: Path | None = None
) -> list[Path]:
    """Write the registered RookID TSVs to ``destination/sources``.

    ``read_rookid_annotations`` expects them on disk under ``sources/``. They are
    together 155 kB, so they are extracted rather than read through a zip handle
    each time.
    """

    base = root if root is not None else benchmark_root()
    archive = base / ROOKID_ARCHIVE
    sources = destination / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for recording in protocol["source"]["recordings"]:
            stem = str(recording["stem"])
            target = sources / f"{stem}.tsv"
            target.write_bytes(bundle.read(f"RookID/{stem}.tsv"))
            written.append(target)
    return written


def read_csv_column(path: Path, column: str) -> list[str]:
    """Return one column of a CSV, in file order."""

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"{path} has no column {column!r}")
        return [row[column] for row in reader]
