"""What size of identity signal the pipeline would have found on a null endpoint.

On chiffchaff across a year BirdNET does not clear its permutation test. That
null means something only if the same pipeline, on the same recordings, would
have found an identity signal of a known size. This builds such a signal:
calls from chiffchaff within a year, where identity is strong, are mixed into
the across-year endpoint's own background recordings, one donor bird to one
target territory, enrolment calls into enrolment-year backgrounds and scoring
calls into scoring-year backgrounds, at the three mixing levels of the
background challenge. The planted endpoint is then scored exactly as every
endpoint is.

If the planted calls are named well above the null, the across-year null is not
a failure of the pipeline to see identity through across-year recording
conditions: identity of within-year strength would have been seen.

Registered in ``docs/measurement-protocol.md`` (PA-V6):

* donors are the ten within-year birds with the most enrolment calls, ties
  broken by name; targets are the ten across-year birds in name order; donor k
  is planted into target k's territory;
* every donor call gets one background clip of its target in the matching
  split, chosen by a salted hash of the call's file name;
* mixing uses ``background_challenge.mix_background`` at call-over-background
  ratios of -10, 0 and +10 dB.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint

PLANTING_SALT = "xyy-sensitivity-floor-20260924"
DONORS = 10


def _foreground(records: Sequence[ClipRecord]) -> list[ClipRecord]:
    return [r for r in records if str(r.context.get("condition", "foreground")) == "foreground"]


def _backgrounds(records: Sequence[ClipRecord]) -> list[ClipRecord]:
    return [r for r in records if str(r.context.get("condition")) == "background"]


def pair_donors(donor: Endpoint, target: Endpoint) -> list[tuple[str, str]]:
    """(donor identity, target identity) pairs, fixed before any score is read."""

    calls = _foreground(donor.records)
    counts: dict[str, int] = {}
    for record in calls:
        if record.split == "enrollment":
            counts[record.identity] = counts.get(record.identity, 0) + 1
    ranked = sorted(counts, key=lambda name: (-counts[name], name))
    targets = sorted({r.identity for r in _backgrounds(target.records)})
    count = min(DONORS, len(ranked), len(targets))
    if count < 2:
        raise ValueError("planting needs at least two donors and two targets")
    return list(zip(ranked[:count], targets[:count], strict=True))


def _choose(call: ClipRecord, options: Sequence[ClipRecord]) -> ClipRecord:
    ordered = sorted(options, key=lambda r: r.filename)
    digest = hashlib.sha256(f"{PLANTING_SALT}:{call.filename}".encode()).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def plan_planting(donor: Endpoint, target: Endpoint) -> list[tuple[ClipRecord, ClipRecord]]:
    """Every (donor call, target background) pair, in donor-record order."""

    pairs = dict(pair_donors(donor, target))
    backgrounds = _backgrounds(target.records)
    planned: list[tuple[ClipRecord, ClipRecord]] = []
    for call in _foreground(donor.records):
        if call.identity not in pairs:
            continue
        options = [
            r for r in backgrounds if r.identity == pairs[call.identity] and r.split == call.split
        ]
        if not options:
            raise ValueError(f"{pairs[call.identity]} has no {call.split} backgrounds")
        planned.append((call, _choose(call, options)))
    return planned


def _read(path: Path) -> tuple[Any, int]:
    import wave
    from array import array

    import numpy as np

    with wave.open(str(path), "rb") as stream:
        if stream.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        rate = stream.getframerate()
        raw = array("h")
        raw.frombytes(stream.readframes(stream.getnframes()))
    return np.asarray(raw, dtype=np.float64) / 32768.0, rate


def _write(path: Path, signal: Any, rate: int) -> Path:
    import wave

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.round(np.asarray(signal) * 32767.0).astype("<i2").tobytes())
    return path


def planted_endpoint(
    donor: Endpoint, target: Endpoint, *, level_db: float, scratch: Path
) -> Endpoint:
    """The donor calls, each mixed into its target territory's background."""

    from xinyenyana.background_challenge import mix_background

    records: list[ClipRecord] = []
    for index, (call, background) in enumerate(plan_planting(donor, target)):
        signal, rate = _read(call.path)
        ambient, ambient_rate = _read(background.path)
        if rate != ambient_rate:
            raise ValueError(
                f"{call.filename} is {rate} Hz and {background.filename} {ambient_rate}"
            )
        mixed = mix_background(signal, ambient, foreground_over_background_db=level_db)
        name = f"plant-{level_db:+.0f}-{index:06d}-{call.filename}"
        records.append(
            replace(
                call,
                filename=name,
                path=_write(scratch / f"{name}.wav", mixed, rate),
                context={
                    **call.context,
                    "planted_into": background.filename,
                    "target_territory": background.identity,
                },
            )
        )
    return replace(target, name=f"{target.name}-planted-{level_db:+.0f}dB", records=tuple(records))
