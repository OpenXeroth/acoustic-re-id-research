"""The RookID archive: its record, its annotation files, and two helpers.

Four things about the dataset rather than about any experiment: where it is
published, how its tab-separated annotation files are read, and the path and
digest checks the clip cutter needs.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(frozen=True)
class RookIDEvent:
    """One source-authored annotation with stable row provenance."""

    recording: str
    split: str
    row_number: int
    identity: str
    event: str
    comment: str
    start: float
    end: float


ROOKID_RECORD = "6091940"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe RookID path: {relative}")
    destination = root.joinpath(*candidate.parts)
    if root.resolve() not in destination.resolve().parents:
        raise ValueError(f"RookID path escapes output root: {relative}")
    return destination


def read_rookid_annotations(
    *, protocol: dict[str, Any], source_root: Path
) -> dict[str, list[RookIDEvent]]:
    """Read registered TSVs and retain their source row numbers."""

    annotations: dict[str, list[RookIDEvent]] = {}
    for recording in protocol["source"]["recordings"]:
        stem = str(recording["stem"])
        path = source_root / "sources" / f"{stem}.tsv"
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if reader.fieldnames != ["Source", "Event", "Comment", "Start", "End"]:
                raise ValueError(f"unexpected RookID TSV schema: {path}")
            values: list[RookIDEvent] = []
            for row_number, row in enumerate(reader, start=2):
                start = float(row["Start"])
                end = float(row["End"])
                if start < 0 or end <= start:
                    raise ValueError(f"invalid RookID interval: {path}:{row_number}")
                values.append(
                    RookIDEvent(
                        recording=stem,
                        split=str(recording["split"]),
                        row_number=row_number,
                        identity=row["Source"].strip(),
                        event=row["Event"].strip(),
                        comment=row["Comment"].strip(),
                        start=start,
                        end=end,
                    )
                )
        annotations[stem] = values
    return annotations
