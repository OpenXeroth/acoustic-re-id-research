"""Reading a representation back from the name a sweep recorded it under.

The sweep writes a name; a later experiment is pointed at that name. These
check the two agree, by round-tripping every representation the sweep can
produce rather than by listing names by hand.
"""

from __future__ import annotations

import pytest

from xinyenyana.a5 import (
    BIRDNET,
    HIDDEN_STATE_MODELS,
    SHORTCUTS,
    SLOWDOWNS,
    SPEAKER_MODELS,
    Representation,
    representation_from_name,
)


def _every_representation() -> list[Representation]:
    produced = [Representation(model=BIRDNET, slowdown=1, layer=None)]
    produced += [Representation(model=name, slowdown=1, layer=None) for name in SHORTCUTS]
    for model in SPEAKER_MODELS:
        produced += [Representation(model=model, slowdown=rate, layer=None) for rate in SLOWDOWNS]
    for model in HIDDEN_STATE_MODELS:
        for rate in SLOWDOWNS:
            produced += [Representation(model=model, slowdown=rate, layer=n) for n in range(0, 25)]
    return produced


def test_every_sweep_name_reads_back_to_the_representation_that_wrote_it() -> None:
    produced = _every_representation()
    assert len(produced) > 900
    for representation in produced:
        assert representation_from_name(representation.name) == representation


def test_the_names_already_on_record_are_accepted_without_a_rate() -> None:
    for name in (BIRDNET, *SHORTCUTS):
        assert representation_from_name(name) == Representation(model=name, slowdown=1, layer=None)


def test_the_bat_sweeps_choice_reads_back_to_wavlm_large_layer_seven() -> None:
    chosen = representation_from_name("wavlm-large-x3-l07")
    assert chosen == Representation(model="microsoft/wavlm-large", slowdown=3, layer=7)


def test_the_chiffchaff_across_year_choice_reads_back_to_xls_r_layer_zero() -> None:
    chosen = representation_from_name("wav2vec2-xls-r-300m-x1-l00")
    assert chosen == Representation(model="facebook/wav2vec2-xls-r-300m", slowdown=1, layer=0)


@pytest.mark.parametrize(
    "name",
    [
        "wavlm-large-x3",
        "spkrec-ecapa-voxceleb-x3-l07",
        "not-a-model-x1-l00",
        "wavlm-large-x9-l07",
        "birdnet-v2.4-x2",
        "birdnet-v2.4-x1-l03",
        "wavlm-large",
    ],
)
def test_a_name_this_project_cannot_produce_is_refused(name: str) -> None:
    with pytest.raises(ValueError):
        representation_from_name(name)
