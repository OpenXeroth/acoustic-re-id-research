from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord
from xinyenyana.background_challenge import background_pairs, mix_background


def records() -> list[ClipRecord]:
    return [
        ClipRecord(
            filename=f"{identity}-{split}-{condition}",
            path=Path(f"{identity}-{split}-{condition}.wav"),
            identity=identity,
            split=split,
            context={"condition": condition},
        )
        for identity in ("alpha", "beta", "gamma")
        for split in ("enrollment", "query")
        for condition in ("foreground", "background")
    ]


def test_donors_never_cross_the_split_and_other_donor_is_a_different_identity() -> None:
    pairs = background_pairs(records())
    assert len(pairs) == 6
    for pair in pairs:
        assert pair.foreground.split == pair.own.split == pair.other.split
        assert pair.foreground.identity == pair.own.identity
        assert pair.foreground.identity != pair.other.identity


def test_pairing_is_independent_of_manifest_row_order() -> None:
    def mapping(source: list[ClipRecord]) -> dict[str, tuple[str, str]]:
        return {
            p.foreground.filename: (p.own.filename, p.other.filename)
            for p in background_pairs(source)
        }

    assert mapping(records()) == mapping(list(reversed(records())))


def test_missing_backgrounds_are_refused_not_borrowed_from_the_other_split() -> None:
    incomplete = [
        r for r in records() if not (r.split == "query" and r.context["condition"] == "background")
    ]
    with pytest.raises(ValueError, match="each split"):
        background_pairs(incomplete)


@pytest.mark.parametrize("ratio", [-10.0, 0.0, 10.0])
def test_mixture_retains_the_declared_ratio_and_duration(ratio: float) -> None:
    # Orthogonal signals allow the contributions to be recovered after the
    # common anti-clipping gain, independently of the mixing implementation.
    signal = np.tile([1.0, 0.0], 100)
    background = np.tile([0.0, 1.0], 100)
    mixed = mix_background(signal, background, foreground_over_background_db=ratio)
    measured = 20 * np.log10(np.linalg.norm(mixed[::2]) / np.linalg.norm(mixed[1::2]))
    assert measured == pytest.approx(ratio, abs=1e-5)
    assert len(mixed) == len(signal)
    assert np.max(np.abs(mixed)) <= 0.951


def test_silent_background_is_not_a_successful_challenge() -> None:
    with pytest.raises(ValueError, match="silent"):
        mix_background(np.ones(20), np.zeros(20), foreground_over_background_db=0)
