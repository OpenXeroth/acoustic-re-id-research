"""Deterministic background challenges that never cross the published split.

The original foreground already contains ambient sound. Adding a background is
a challenge, not a claim to have separated or replaced the original background.
The same operation with the caller's own and another caller's backgrounds
separates the cost of adding sound from sensitivity to its recorded location.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from xinyenyana.a2 import ClipRecord

CHALLENGE_DB = (-10.0, 0.0, 10.0)
PAIRING_SALT = "phase-a-background-challenge-20260913"


@dataclass(frozen=True)
class BackgroundPair:
    foreground: ClipRecord
    own: ClipRecord
    other: ClipRecord


def _draw(name: str, options: Sequence[ClipRecord]) -> ClipRecord:
    ordered = sorted(options, key=lambda item: item.filename)
    digest = hashlib.sha256(f"{PAIRING_SALT}:{name}".encode()).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def background_pairs(records: Sequence[ClipRecord]) -> list[BackgroundPair]:
    foreground = [r for r in records if r.context.get("condition", "foreground") == "foreground"]
    backgrounds = [r for r in records if r.context.get("condition") == "background"]
    result = []
    for record in foreground:
        same_split = [r for r in backgrounds if r.split == record.split]
        identities = sorted({r.identity for r in same_split})
        if record.identity not in identities or len(identities) < 2:
            raise ValueError("a background challenge needs own and other identities in each split")
        # Cycle identities rather than drawing from all clips: a prolific donor
        # must not dominate merely because it has more background recordings.
        other_identity = identities[(identities.index(record.identity) + 1) % len(identities)]
        own = _draw(record.filename, [r for r in same_split if r.identity == record.identity])
        other = _draw(record.filename, [r for r in same_split if r.identity == other_identity])
        result.append(BackgroundPair(record, own, other))
    if not result:
        raise ValueError("a background challenge needs foreground clips")
    return result


def mix_background(signal: Any, background: Any, *, foreground_over_background_db: float) -> Any:
    """Mix at a declared whole-waveform RMS ratio, without clipping or trimming.

    Inputs must already share a sample rate. A short background is repeated; callers
    record that fact. This ratio is not the true SNR of the vocalisation, because
    the foreground also contains the original ambient sound.
    """

    import numpy as np

    foreground = np.asarray(signal, dtype=np.float64)
    ambient = np.asarray(background, dtype=np.float64)
    if foreground.ndim != 1 or ambient.ndim != 1 or not foreground.size or not ambient.size:
        raise ValueError("mixing needs two nonempty mono waveforms")
    if not np.isfinite(foreground).all() or not np.isfinite(ambient).all():
        raise ValueError("mixing needs finite samples")
    ambient = np.resize(ambient, foreground.size)
    foreground_rms = float(np.sqrt(np.mean(foreground**2)))
    background_rms = float(np.sqrt(np.mean(ambient**2)))
    if foreground_rms == 0 or background_rms == 0:
        raise ValueError("a silent waveform has no defined mixing ratio")
    amplitude = foreground_rms / background_rms * 10 ** (-foreground_over_background_db / 20)
    mixed = foreground + amplitude * ambient
    # One gain on the sum preserves the requested relative level. Scale every
    # challenged waveform this way, with headroom for int16 output rounding.
    peak = float(np.max(np.abs(mixed)))
    if peak:
        mixed *= 0.95 / peak
    return mixed.astype(np.float32)
