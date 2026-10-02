"""The Wytham great tit endpoint at full width, across years.

The sixteen-bird great tit endpoint was built by a chain that has since been
removed from this repository. This builder works from the published tables
alone, so the rule that decides which bird a song belongs to can be read here.

**Identity.** A song belongs to the nest attempt whose recording it was cut
from (``great-tit-hits.csv``, column ``ID``). The attempt's father is its BTO
ring in ``main.csv``. The ring must also appear in ``morphometrics.csv`` as a
great tit (species ``g``) sexed male, the same two-source requirement the
sixteen-bird endpoint used. A bird is known by its nestbox, not seen calling,
which is why the place control is run beside it.

**Split.** A bird enters when it has songs in at least two breeding years.
It is enrolled on songs from its first recorded year and scored on songs from
its next recorded year. Up to ten songs per bird per year are kept, taken in
the order of a salted hash of their file names, so no score chooses them.

**Clip.** Each published song WAV is copied as it is, under an opaque token.
The fraction of samples at full scale is recorded because the open-set slices
read it. The manifest names real rings and stays on xen1, as the data
governance rules require; only this code is in the repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from xinyenyana.a2 import Endpoint, load_great_tit

ENDPOINT = "great-tit-full"
MANIFEST = "great-tit-full-manifest.json"
SONGS_PER_BIRD_YEAR = 10
SELECTION_SALT = "xyy-great-tit-full-20260924"
ARCHIVES = tuple(f"song-files.zip.part{i}" for i in range(1, 5))


def male_great_tits(morphometrics: Path) -> set[str]:
    with morphometrics.open(newline="") as handle:
        return {
            row["bto_ring"]
            for row in csv.DictReader(handle)
            if row.get("species") == "g" and row.get("sex") == "m" and row.get("bto_ring")
        }


def fathers_by_attempt(main: Path, males: set[str]) -> dict[str, dict[str, Any]]:
    """Nest attempt -> father ring, year, nestbox and nestbox position."""

    attempts: dict[str, dict[str, Any]] = {}
    with main.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ring = row.get("father", "")
            if not ring or ring not in males:
                continue
            attempts[row["pnum"]] = {
                "identity": ring,
                "year": int(row["year"]),
                "nestbox": row["nestbox"],
                "nest_x": float(row["x"]) if row.get("x") else float("nan"),
                "nest_y": float(row["y"]) if row.get("y") else float("nan"),
            }
    return attempts


def _rank(name: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{name}".encode()).hexdigest()


def select_songs(hits: Path, attempts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every kept song, with its bird, year, split and source recording."""

    by_bird_year: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    with hits.open(newline="") as handle:
        for row in csv.DictReader(handle):
            attempt = attempts.get(row["ID"])
            if attempt is None:
                continue
            stem = row[""]
            by_bird_year[(attempt["identity"], attempt["year"])].append(
                {
                    **attempt,
                    "attempt": row["ID"],
                    "member": f"WAV/{stem}.wav",
                    "source_recording": stem.rsplit("_", 1)[0],
                }
            )
    years: dict[str, list[int]] = defaultdict(list)
    for identity, year in by_bird_year:
        years[identity].append(year)
    kept: list[dict[str, Any]] = []
    for identity in sorted(years):
        ordered = sorted(years[identity])
        if len(ordered) < 2:
            continue
        for split, year in (("enrollment", ordered[0]), ("query", ordered[1])):
            songs = sorted(by_bird_year[(identity, year)], key=lambda s: _rank(s["member"]))
            for song in songs[:SONGS_PER_BIRD_YEAR]:
                kept.append({**song, "split": split, "cohort": f"{ordered[0]}-{ordered[1]}"})
    return kept


def _full_scale_fraction(data: bytes) -> float:
    import io
    import wave

    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as stream:
        width = stream.getsampwidth()
        frames = stream.readframes(stream.getnframes())
    if width != 2 or not frames:
        return 0.0
    samples = np.frombuffer(frames, dtype="<i2").astype(np.int32)
    return float(np.mean(np.abs(samples) >= 32767))


def build_sample(*, source: Path, destination: Path) -> dict[str, Any]:
    """Write the clips and the manifest; returns the counts for the build record."""

    males = male_great_tits(source / "morphometrics.csv")
    attempts = fathers_by_attempt(source / "main.csv", males)
    kept = select_songs(source / "great-tit-hits.csv", attempts)
    wanted = {song["member"]: song for song in kept}
    clips = destination / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for archive in ARCHIVES:
        with zipfile.ZipFile(source / archive) as bundle:
            for member in bundle.namelist():
                song = wanted.get(member)
                if song is None:
                    continue
                data = bundle.read(member)
                token = hashlib.sha256(member.encode()).hexdigest()[:24]
                path = clips / f"{token}.wav"
                path.write_bytes(data)
                records.append(
                    {
                        "event_token": token,
                        "clip_path": str(path),
                        "clip_sha256": hashlib.sha256(data).hexdigest(),
                        "identity": song["identity"],
                        "split": song["split"],
                        "year": song["year"],
                        "cohort": song["cohort"],
                        "nestbox": song["nestbox"],
                        "attempt": song["attempt"],
                        "source_recording": song["source_recording"],
                        "member": member,
                        "archive": archive,
                        "context": {"nest_x": song["nest_x"], "nest_y": song["nest_y"]},
                        "quality": {
                            "metrics": {"clipped_sample_fraction": _full_scale_fraction(data)}
                        },
                        "annotation": {},
                    }
                )
    missing = sorted(set(wanted) - {r["member"] for r in records})
    if missing:
        raise ValueError(f"{len(missing)} selected songs are in no archive, e.g. {missing[:3]}")
    records.sort(key=lambda r: (r["identity"], r["split"], r["event_token"]))
    manifest = {"experiment": "PA-V6 great tit at full width", "records": records}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    birds = sorted({r["identity"] for r in records})
    return {
        "male_great_tits_in_morphometrics": len(males),
        "attempts_with_a_qualifying_father": len(attempts),
        "birds_with_songs_in_two_or_more_years": len(birds),
        "clips": len(records),
        "clips_by_split": {
            split: sum(1 for r in records if r["split"] == split)
            for split in ("enrollment", "query")
        },
        "birds_scored_at_the_same_nestbox": sum(
            1
            for bird in birds
            if {r["nestbox"] for r in records if r["identity"] == bird and r["split"] == "query"}
            & {
                r["nestbox"]
                for r in records
                if r["identity"] == bird and r["split"] == "enrollment"
            }
        ),
    }


def load_great_tit_full(sample_root: Path) -> Endpoint:
    loaded = load_great_tit(manifest_path=sample_root / MANIFEST, sample_root=sample_root)
    return replace(loaded, name=ENDPOINT)
