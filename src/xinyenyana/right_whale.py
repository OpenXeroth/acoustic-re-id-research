"""The North Atlantic right whale endpoint: calls and the ambient sound beside them.

Tolkova and colleagues published 234 upcalls from 11 tagged whales and, for
every call, a recording of the ambient sound from the same tag with no call in
it. That pairing is what this project's validity gate reads, and outside the
2019 Stowell release it is the only one this project has.

Two properties of the release decide what can be measured on it and are
registered in ``docs/measurement-protocol.md``.

Each whale carried one tag deployment, so identity and recording are the same
variable. Every split that can be made here puts the same recording on both
sides, which is prohibited everywhere else in this project. The two diagonals
are therefore within-recording tests and carry little; the two crossings are
what the endpoint is registered for, because they compare a gallery built from
one condition against probes from the other.

The upcall band is 50 to 500 Hz. The authors moved it upward before embedding
it and ``shift_frequency`` is their arithmetic. Nothing here asserts where
BirdNET's sensitivity begins: the shifted and unshifted arms measure that.

The release carries no licence file. Its audio and any embedding of it stay on
xen1.
"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.archive import canonical_sha256, sha256_file

#: The release's own directory of paired recordings.
RECORDINGS = "Recordings"

#: The manifest a build writes and a load reads.
RIGHT_WHALE_MANIFEST = "right-whale-manifest.json"

ENDPOINT = "right-whale-aiid"

#: The published sample rate. Every one of the 468 files carries it.
SOURCE_RATE = 2_000

#: The rate ``shiftFrequency`` resamples to before modulating.
SHIFTED_RATE = 24_000

#: The band the authors name, and the high-pass offset above the shift.
BAND_LOW_HZ = 50.0
BAND_HIGH_HZ = 500.0

#: Tenth order, applied forwards and backwards, which is what the authors ran.
FILTER_ORDER = 10

#: The ten shifts the release itself carries, plus the unshifted arm at 0.
#: 10,000 is the largest the authors' own aliasing assertion allows at 24 kHz:
#: ``24000 > (500 + shift) * 2`` fails from 11,000.
SHIFTS: tuple[int, ...] = (0, 1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000)

#: The split salt. Changing it redraws the split, so it is part of the rule.
SPLIT_SALT = "xyy-right-whale-20260915"

#: A stem whose digest starts below this byte is scored, the rest enrol. 77 of
#: 256 is about three in ten, matching the share the registration states.
QUERY_THRESHOLD = 77

#: One gain is applied to a whole arm so that relative level between clips
#: survives the write. A per-clip peak scaling would make the level control
#: representation constant and destroy it.
PEAK_TARGET = 0.95

CONDITIONS = {"call": "foreground", "noise": "background"}


def query_side(stem: str) -> bool:
    """Whether a stem is scored rather than enrolled. No score moves this."""

    digest = hashlib.sha256(f"{SPLIT_SALT}:{stem}".encode()).digest()
    return digest[0] < QUERY_THRESHOLD


def read_pcm16(path: Path) -> tuple[Any, int]:
    """Raw samples in [-1, 1) and the file's rate, with the level left alone.

    ``representations.read_pcm_wav`` divides every clip by its own standard
    deviation, which is right for reading a clip into a representation and
    wrong for building one: it would erase the level differences between clips
    before the transform, and the level control exists to read them.
    """

    import numpy as np

    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if audio.getsampwidth() != 2:
            raise ValueError(f"{path} is not PCM16")
        rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    signal = np.frombuffer(frames, dtype="<i2").astype(np.float64) / 32768.0
    if not signal.size:
        raise ValueError(f"empty WAV: {path}")
    return signal, rate


def shift_frequency(signal: Any, rate: int, shift_hz: int) -> tuple[Any, int]:
    """The authors' ``shiftFrequency``, with this project's resampler.

    Resample to 24 kHz, multiply by a complex exponential at ``shift_hz``, take
    the real part, and high-pass at ``shift_hz + 50`` Hz with a tenth-order
    Butterworth applied forwards and backwards. A shift of zero returns the
    signal at its own rate, which is the unshifted arm.

    The authors resampled with librosa; this uses the Fourier method, which is
    what every other resampling step in this project uses and which the audio
    path's own registration measured against the published embeddings.
    """

    import numpy as np
    from scipy.signal import butter, filtfilt, resample

    if shift_hz == 0:
        return np.asarray(signal, dtype=np.float64), rate
    if SHIFTED_RATE <= (BAND_HIGH_HZ + shift_hz) * 2:
        raise ValueError(
            f"a shift of {shift_hz} Hz aliases at {SHIFTED_RATE} Hz; "
            f"the authors' own assertion allows up to {SHIFTS[-1]}"
        )
    wide = resample(np.asarray(signal, dtype=np.float64), round(len(signal) / rate * SHIFTED_RATE))
    steps = np.arange(wide.size)
    modulated = np.real(wide * np.exp(-1j * 2 * np.pi * shift_hz * steps / SHIFTED_RATE))
    coefficients = butter(
        FILTER_ORDER, 2 * (shift_hz + BAND_LOW_HZ) / SHIFTED_RATE, btype="high", analog=False
    )
    return filtfilt(coefficients[0], coefficients[1], modulated), SHIFTED_RATE


def _write_pcm16(path: Path, signal: Any, rate: int) -> None:
    import numpy as np

    samples = np.clip(np.round(np.asarray(signal) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())


def _stems(release_root: Path) -> list[str]:
    folder = release_root / RECORDINGS
    if not folder.is_dir():
        raise ValueError(f"the release holds no {RECORDINGS} folder at {folder}")
    stems = sorted({path.name.rsplit("_", 1)[0] for path in folder.glob("*.wav")})
    for stem in stems:
        for suffix in CONDITIONS:
            member = folder / f"{stem}_{suffix}.wav"
            if not member.is_file():
                raise ValueError(f"{stem} has no {suffix} file; every call carries an ambient twin")
    if not stems:
        raise ValueError(f"{folder} holds no WAV files")
    return stems


def build_right_whale_sample(
    *, release_root: Path, output_root: Path, shift_hz: int
) -> dict[str, Any]:
    """Write one arm of the endpoint and the manifest that records how.

    An arm is the whole release at one shift. Both conditions of a stem get the
    same transform and the same gain, so the crossings compare like with like.
    """

    import numpy as np

    if shift_hz not in SHIFTS:
        raise ValueError(f"unregistered shift {shift_hz}: {list(SHIFTS)}")
    folder = release_root / RECORDINGS
    stems = _stems(release_root)

    transformed: dict[tuple[str, str], tuple[Any, int]] = {}
    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for stem in stems:
        for suffix in CONDITIONS:
            member = folder / f"{stem}_{suffix}.wav"
            signal, rate = read_pcm16(member)
            if rate != SOURCE_RATE:
                raise ValueError(f"{member} reads {rate} Hz, not the published {SOURCE_RATE}")
            transformed[(stem, suffix)] = shift_frequency(signal, rate, shift_hz)
            sources[(stem, suffix)] = {
                "source_sha256": sha256_file(member),
                "source_samples": int(signal.size),
            }

    peak = max(float(np.max(np.abs(values))) for values, _ in transformed.values())
    if not peak > 0:
        raise ValueError("every clip in this arm is silent")
    gain = PEAK_TARGET / peak

    arm = output_root / f"shift-{shift_hz}"
    arm.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for stem in stems:
        identity = stem.split("_", 1)[0]
        split = "query" if query_side(stem) else "enrollment"
        for suffix, condition in CONDITIONS.items():
            values, rate = transformed[(stem, suffix)]
            relative = f"shift-{shift_hz}/{stem}_{suffix}.wav"
            _write_pcm16(output_root / relative, values * gain, rate)
            entries.append(
                {
                    "filename": f"{stem}_{suffix}",
                    "stem": stem,
                    "identity": identity,
                    "condition": condition,
                    "split": split,
                    "local_path": relative,
                    "sample_rate": int(rate),
                    "samples": int(values.size),
                    "sha256": sha256_file(output_root / relative),
                    **sources[(stem, suffix)],
                }
            )

    dropped = sorted(
        {entry["identity"] for entry in entries}
        - {entry["identity"] for entry in entries if entry["split"] == "query"}
    )
    manifest = {
        "endpoint": ENDPOINT,
        "source": "github.com/avokloti/narw-acoustic-identification",
        "commit": "3cad65da424a7234ea182c00e18283a92d65b4b5",
        "licence": "none declared; local measurement only, nothing redistributed",
        "shift_hz": shift_hz,
        "shifted_rate_hz": SHIFTED_RATE if shift_hz else SOURCE_RATE,
        "filter_order": FILTER_ORDER,
        "resampler": "fourier",
        "split_salt": SPLIT_SALT,
        "query_threshold": QUERY_THRESHOLD,
        "peak_target": PEAK_TARGET,
        "common_gain": gain,
        "identities_without_a_query_stem": dropped,
        "entries": entries,
    }
    (arm / RIGHT_WHALE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {key: value for key, value in manifest.items() if key != "entries"}


def load_right_whale(*, manifest_path: Path, sample_root: Path) -> Endpoint:
    """The endpoint one arm wrote, in two conditions, with its drops applied."""

    if not manifest_path.is_file():
        raise ValueError(
            f"no right whale sample at {manifest_path}; "
            "build it with `xinyenyana build-right-whale-sample`"
        )
    manifest = json.loads(manifest_path.read_text())
    if str(manifest["endpoint"]) != ENDPOINT:
        raise ValueError(f"{manifest_path} is not a {ENDPOINT} manifest")
    dropped = set(manifest["identities_without_a_query_stem"])
    records = tuple(
        ClipRecord(
            filename=str(entry["filename"]),
            path=sample_root / str(entry["local_path"]),
            identity=str(entry["identity"]),
            split=str(entry["split"]),
            context={
                "condition": str(entry["condition"]),
                "species": "north-atlantic-right-whale",
                "stem": str(entry["stem"]),
                "shift_hz": int(manifest["shift_hz"]),
                "tag_deployment": str(entry["identity"]),
            },
        )
        for entry in manifest["entries"]
        if str(entry["identity"]) not in dropped
    )
    if not records:
        raise ValueError(f"{ENDPOINT} carries no clips")
    return Endpoint(
        name=ENDPOINT,
        records=records,
        categorical_targets=(),
        manifest_sha256=canonical_sha256(
            sorted(str(entry["source_sha256"]) for entry in manifest["entries"])
        ),
        source_document="docs/benchmark-data.md",
    )


def coverage(endpoint: Endpoint) -> dict[str, Any]:
    """Per-whale counts by condition and split, so a thin arm is visible."""

    per_identity: dict[str, dict[str, int]] = {}
    for record in endpoint.records:
        entry = per_identity.setdefault(
            record.identity,
            {"foreground": 0, "background": 0, "enrollment": 0, "query": 0},
        )
        entry[str(record.context["condition"])] += 1
        entry[record.split] += 1
    return dict(sorted(per_identity.items()))
