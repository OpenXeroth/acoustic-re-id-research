"""The 2019 paper's own remedy, tested on a representation it predates.

Stowell and colleagues did not only publish the diagnostic that finds the
recording confound. They published a fix for it, and their recommendation list
names it: improve robustness by structured data augmentation using the
background recordings. This project has run their diagnostic for weeks and has
never run their fix, which leaves the obvious question about their paper
unanswered: does the prescription still work on the embeddings the field uses
now?

**Their method, in their words.** "Each training item had been mixed with an
example of background sound from each other individual", so "the dataset size
increases by a factor of K". The intent is to break the correlation between an
individual and its own ambient sound by making every individual's ambient sound
appear under every individual's calls.

**What differs here, and why.** Their mixing is `sox -m`, a plain sum of two
waveforms at their own levels. This project's existing background challenge
mixes at a declared level ratio instead, because it is asking a different
question about sensitivity. Their fix is reproduced with plain addition, so
that what is tested is their method rather than this project's variant of it.
The sum is scaled once to fit int16 without clipping, which sox also has to do,
and that scaling is applied to every augmented clip alike.

**The query side is never touched.** Only the training material changes, which
is what an augmentation is. The diagnostic that scores it is unchanged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint

#: No score chooses which background a training clip is mixed with. The donor
#: clip within a donor identity is fixed by this hash of the pair.
DONOR_SALT = "xyy-stowell-augmentation-20260915"

#: Peak the summed waveform is scaled to before int16 rounding, matching the
#: headroom the existing challenge mixer uses.
PEAK_TARGET = 0.95

AUGMENTED = "stratified-background-augmentation"
UNTOUCHED = "untouched"


def _donor_clip(foreground: ClipRecord, candidates: Sequence[ClipRecord]) -> ClipRecord:
    """Which of a donor identity's background clips this training clip gets."""

    ordered = sorted(candidates, key=lambda record: record.filename)
    digest = hashlib.sha256(f"{DONOR_SALT}:{foreground.filename}:{ordered[0].identity}".encode())
    return ordered[int.from_bytes(digest.digest()[:8], "big") % len(ordered)]


def plan_augmentation(records: Sequence[ClipRecord]) -> list[tuple[ClipRecord, ClipRecord]]:
    """Every (training clip, donor background) pair their method calls for.

    One pair per training clip per other identity, taken from the enrolment
    split alone so that no query clip reaches the training material.
    """

    enrolled = [record for record in records if record.split == "enrollment"]
    foreground = [
        record
        for record in enrolled
        if str(record.context.get("condition", "foreground")) == "foreground"
    ]
    backgrounds = [
        record for record in enrolled if str(record.context.get("condition")) == "background"
    ]
    if not backgrounds:
        raise ValueError(
            "this endpoint publishes no background recordings, so the 2019 "
            "augmentation has nothing to mix in"
        )
    by_identity: dict[str, list[ClipRecord]] = {}
    for record in backgrounds:
        by_identity.setdefault(record.identity, []).append(record)
    pairs: list[tuple[ClipRecord, ClipRecord]] = []
    for record in foreground:
        for identity in sorted(by_identity):
            if identity == record.identity:
                continue
            pairs.append((record, _donor_clip(record, by_identity[identity])))
    return pairs


def write_mixed_clip(foreground: ClipRecord, donor: ClipRecord, destination: Path) -> Path:
    """One training clip summed with one other individual's ambient recording."""

    import wave
    from array import array

    import numpy as np

    def read(path: Path) -> tuple[Any, int]:
        with wave.open(str(path), "rb") as stream:
            if stream.getnchannels() != 1:
                raise ValueError(f"{path} is not mono")
            rate = stream.getframerate()
            raw = array("h")
            raw.frombytes(stream.readframes(stream.getnframes()))
        return np.asarray(raw, dtype=np.float64) / 32768.0, rate

    signal, rate = read(foreground.path)
    ambient, donor_rate = read(donor.path)
    if donor_rate != rate:
        raise ValueError(f"{foreground.filename} is {rate} Hz and {donor.filename} {donor_rate} Hz")
    if not signal.size or not ambient.size:
        raise ValueError("mixing needs two nonempty waveforms")
    mixed = signal + np.resize(ambient, signal.size)
    peak = float(np.max(np.abs(mixed)))
    if peak:
        mixed = mixed * (PEAK_TARGET / peak)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(destination), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.round(mixed * 32768.0).astype("<i2").tobytes())
    return destination


def augmented_endpoint(endpoint: Endpoint, *, scratch: Path) -> tuple[Endpoint, dict[str, Any]]:
    """The endpoint with its training material enlarged as the 2019 paper did.

    Query records are carried through untouched. Enrolment records are carried
    through and joined by one mixed copy per other identity, so the training
    material grows by a factor of the identity count and every individual's
    ambient sound appears under every individual's calls.
    """

    pairs = plan_augmentation(endpoint.records)
    scratch.mkdir(parents=True, exist_ok=True)
    added: list[ClipRecord] = []
    for index, (foreground, donor) in enumerate(pairs):
        name = f"aug-{index:06d}-{foreground.filename}-with-{donor.identity}"
        added.append(
            replace(
                foreground,
                filename=name,
                path=write_mixed_clip(foreground, donor, scratch / f"{name}.wav"),
                context={**foreground.context, "augmented_with": donor.identity},
            )
        )
    identities = sorted({record.identity for record in endpoint.records})
    return (
        replace(
            endpoint,
            name=f"{endpoint.name}-stowell-augmented",
            records=(*endpoint.records, *added),
        ),
        {
            "method": AUGMENTED,
            "source": "Stowell et al. 2019, stratified data augmentation",
            "mixing": "plain sum at the clips' own levels, scaled once to a 0.95 peak",
            "donor_salt": DONOR_SALT,
            "identities": len(identities),
            "training_clips_before": sum(
                1
                for record in endpoint.records
                if record.split == "enrollment"
                and str(record.context.get("condition", "foreground")) == "foreground"
            ),
            "mixed_clips_added": len(added),
            "query_clips_touched": 0,
        },
    )
