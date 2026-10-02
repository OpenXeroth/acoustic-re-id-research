"""Tests for reading the Huang release's own splits and its own embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xinyenyana.huang import (
    HUANG_ENDPOINTS,
    HUANG_EXTRA_NIGHTS,
    HUANG_FOLDERS,
    PUBLISHED_EMBEDDINGS,
    PUBLISHED_MODELS,
    PUBLISHED_SPLITS,
    load_huang,
    load_published_vectors,
)

CSV_HEADER = "file_name,class,is_train,is_train-fold_1,is_train-fold_2"


def _release(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    cockatoo = root / HUANG_FOLDERS["cockatoo"]
    penguin = root / HUANG_FOLDERS["penguin"]
    rows = {
        cockatoo: [
            ("AL/2019-20/1.wav", "AL_2019-20", "True", "True"),
            ("AL/2019-20/2.wav", "AL_2019-20", "False", "True"),
            ("W1/2020-21/1.wav", "W1_2020-21", "True", "False"),
            ("W1/2020-21/2.wav", "W1_2020-21", "False", "True"),
        ],
        penguin: [
            ("13B3-1_clipped/a.wav", "13B3-1", "True", "True"),
            ("13B3-1_clipped/b.wav", "13B3-1", "False", "False"),
            ("B_clipped/a.wav", "B", "True", "True"),
            ("B_clipped/b.wav", "B", "False", "True"),
        ],
    }
    for folder, listed in rows.items():
        folder.mkdir(parents=True)
        lines = [CSV_HEADER]
        for name, identity, fold1, fold2 in listed:
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            lines.append(f"{name},{identity},{fold1},{fold1},{fold2}")
        (folder / "metadata.csv").write_text("\n".join(lines) + "\n")
    extra = root / HUANG_EXTRA_NIGHTS
    # The release's own extra-night naming, which is where the recording date
    # of a query clip is published. A fixture inventing a simpler name would
    # let a date parser that cannot read the real files pass.
    for nest, day in (("13B3-1", "250218"), ("17B2-2", "250219")):
        folder = extra / f"{nest}_extra-day"
        folder.mkdir(parents=True)
        (folder / f"sel.01.ch01.{day}.140610.88..wav").write_bytes(b"")
    return root


def _embeddings(root: Path, *, species: str, model: str, entries: list[dict[str, object]]) -> Path:
    folder, filename = PUBLISHED_EMBEDDINGS[species]
    path = root / folder / PUBLISHED_MODELS[model] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")
    return path


# --- the audio endpoints -------------------------------------------------------


def test_the_endpoint_names_carry_the_fold_so_none_is_defaulted() -> None:
    assert "cockatoo-fold1" in HUANG_ENDPOINTS
    assert "penguin-fold5" in HUANG_ENDPOINTS
    assert "penguin-acrossnight" in HUANG_ENDPOINTS
    assert "cockatoo" not in HUANG_ENDPOINTS
    assert "penguin" not in HUANG_ENDPOINTS


def test_a_fold_becomes_enrollment_and_query(tmp_path: Path) -> None:
    root = _release(tmp_path)
    first = load_huang(endpoint="cockatoo-fold1", root=root)
    second = load_huang(endpoint="cockatoo-fold2", root=root)
    assert [record.split for record in first.records] != [record.split for record in second.records]
    assert {record.split for record in first.records} == {"enrollment", "query"}
    assert first.identities == ["AL_2019-20", "W1_2020-21"]


def test_the_across_night_split_enrolls_every_first_night_clip(tmp_path: Path) -> None:
    """The release's README defines it that way, including the nests it leaves unqueried."""

    root = _release(tmp_path)
    endpoint = load_huang(endpoint="penguin-acrossnight", root=root)
    enrolled = {record.identity for record in endpoint.records if record.split == "enrollment"}
    queried = {record.identity for record in endpoint.records if record.split == "query"}
    assert enrolled == {"13B3-1", "B"}
    assert queried == {"13B3-1", "17B2-2"}
    assert {record.context["night"] for record in endpoint.records} == {"first", "later"}


def test_a_clip_named_by_the_list_but_absent_is_refused(tmp_path: Path) -> None:
    root = _release(tmp_path)
    (root / HUANG_FOLDERS["cockatoo"] / "AL/2019-20/1.wav").unlink()
    with pytest.raises(ValueError, match="which is not on disk"):
        load_huang(endpoint="cockatoo-fold1", root=root)


def test_an_unknown_endpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown Huang endpoint"):
        load_huang(endpoint="cockatoo", root=_release(tmp_path))


# --- the published embeddings --------------------------------------------------


def test_a_fold_of_the_published_vectors_comes_back_in_file_order(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _embeddings(
        root,
        species="littleowl",
        model="birdnet",
        entries=[
            {
                "class": "A",
                "dataset_type": "train",
                "is_train-fold_1": True,
                "embedding": [1.0, 0.0],
            },
            {
                "class": "A",
                "dataset_type": "test",
                "is_train-fold_1": False,
                "embedding": [2.0, 0.0],
            },
            {
                "class": "B",
                "dataset_type": "train",
                "is_train-fold_1": True,
                "embedding": [0.0, 1.0],
            },
            {
                "class": "B",
                "dataset_type": "test",
                "is_train-fold_1": False,
                "embedding": [0.0, 2.0],
            },
        ],
    )
    endpoint, vectors = load_published_vectors(
        species="littleowl", model="birdnet", split="fold1", root=root
    )
    assert vectors.shape == (4, 2)
    assert [record.identity for record in endpoint.records] == ["A", "A", "B", "B"]
    assert [record.split for record in endpoint.records] == [
        "enrollment",
        "query",
        "enrollment",
        "query",
    ]
    assert vectors[1][0] == 2.0


def test_the_recordists_split_leaves_out_what_it_does_not_cover(tmp_path: Path) -> None:
    """Tree pipit's file also carries across-year clips; they are a different endpoint."""

    root = tmp_path / "dataset"
    _embeddings(
        root,
        species="pipit",
        model="birdnet",
        entries=[
            {"class": "A", "dataset_type": "train", "is_train-fold_1": True, "embedding": [1.0]},
            {"class": "A", "dataset_type": "test", "is_train-fold_1": False, "embedding": [2.0]},
            {"class": "A", "dataset_type": "more", "is_train-fold_1": True, "embedding": [3.0]},
            {"class": "A", "dataset_type": "years", "is_train-fold_1": False, "embedding": [4.0]},
            {"class": "B", "dataset_type": "train", "is_train-fold_1": True, "embedding": [5.0]},
            {"class": "B", "dataset_type": "test", "is_train-fold_1": False, "embedding": [6.0]},
        ],
    )
    endpoint, vectors = load_published_vectors(
        species="pipit", model="birdnet", split="published", root=root
    )
    assert vectors.shape == (4, 1)
    assert [float(value) for value in vectors[:, 0]] == [1.0, 2.0, 5.0, 6.0]


def test_a_file_without_the_recordists_split_says_so(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _embeddings(
        root,
        species="cockatoo",
        model="birdnet",
        entries=[
            {"class": "A", "is_train-fold_1": True, "embedding": [1.0]},
            {"class": "A", "is_train-fold_1": False, "embedding": [2.0]},
        ],
    )
    with pytest.raises(ValueError, match="carries no dataset_type"):
        load_published_vectors(species="cockatoo", model="birdnet", split="published", root=root)


def test_the_across_night_split_of_the_vectors_takes_its_query_from_the_other_nights(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _embeddings(
        root,
        species="penguin",
        model="birdnet",
        entries=[
            {"class": "A", "is_train-fold_1": True, "embedding": [1.0]},
            {"class": "B", "is_train-fold_1": False, "embedding": [2.0]},
        ],
    )
    extra = root / HUANG_EXTRA_NIGHTS / PUBLISHED_MODELS["birdnet"]
    extra.mkdir(parents=True)
    (extra / "embeddings.json").write_text(json.dumps({"class": "A", "embedding": [3.0]}) + "\n")
    endpoint, vectors = load_published_vectors(
        species="penguin", model="birdnet", split="acrossnight", root=root
    )
    assert [record.split for record in endpoint.records] == [
        "enrollment",
        "enrollment",
        "query",
    ]
    assert vectors.shape == (3, 1)


def test_only_the_penguin_was_recorded_on_other_nights(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _embeddings(
        root,
        species="cockatoo",
        model="birdnet",
        entries=[{"class": "A", "is_train-fold_1": True, "embedding": [1.0]}],
    )
    with pytest.raises(ValueError, match="recorded again on other nights"):
        load_published_vectors(species="cockatoo", model="birdnet", split="acrossnight", root=root)


def test_an_unknown_species_model_or_split_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with pytest.raises(ValueError, match="unknown species"):
        load_published_vectors(species="rook", model="birdnet", split="fold1", root=root)
    with pytest.raises(ValueError, match="unknown model"):
        load_published_vectors(species="pipit", model="wavlm", split="fold1", root=root)
    with pytest.raises(ValueError, match="unknown split"):
        load_published_vectors(species="pipit", model="birdnet", split="fold9", root=root)
    assert set(PUBLISHED_SPLITS) == {
        "fold1",
        "fold2",
        "fold3",
        "fold4",
        "fold5",
        "published",
        "acrossnight",
    }


def test_the_published_vectors_reach_the_compensation_through_one_command(tmp_path) -> None:
    """The authors' vectors must reach E-REID-01 by the same loader that scores them."""

    from typer.testing import CliRunner

    from xinyenyana.cli import app

    result = CliRunner().invoke(
        app,
        [
            "run-published-compensation",
            "--species",
            "not-a-species",
            "--model",
            "birdnet",
            "--split",
            "published",
            "--release-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown species" in result.output


def test_an_extra_night_clip_carries_the_date_in_its_name() -> None:
    """The release states when a query clip was recorded only in its filename,
    and the recording-day probe is registered to run on this endpoint."""

    from xinyenyana.huang import extra_night_date

    assert extra_night_date("sel.01.ch01.250219.140610.88..wav") == "2025-02-19"
    assert extra_night_date("sel.12.ch01.250218.233555.00..wav") == "2025-02-18"


def test_a_name_that_is_not_a_published_extra_night_clip_is_an_error() -> None:
    """The mutation this catches: returning None for an unrecognised name, which
    would date part of the query side and leave the probe measured on whichever
    clips happened to parse."""

    import pytest

    from xinyenyana.huang import extra_night_date

    with pytest.raises(ValueError, match="published extra-night clip name"):
        extra_night_date("13B3-1_exhale-i10_10.wav")


def test_the_across_night_query_records_carry_their_recording_date(tmp_path: Path) -> None:
    """The recording-day probe reads this field and nothing else dates these
    clips, so a loader that drops it leaves the probe with no target."""

    from xinyenyana.huang import load_huang

    endpoint = load_huang(endpoint="penguin-acrossnight", root=_release(tmp_path))
    query = [record for record in endpoint.records if record.split == "query"]
    assert query
    assert {record.context["recorded"] for record in query} == {"2025-02-18", "2025-02-19"}
    assert all(
        "recorded" not in record.context
        for record in endpoint.records
        if record.split == "enrollment"
    )
