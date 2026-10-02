"""Tests for reading the Stowell release's published lists."""

from __future__ import annotations

from pathlib import Path

import pytest

from xinyenyana.a2 import STOWELL_ENDPOINTS, load_stowell

SPECIES = "pipit"
SPAN = "withinyear"


def _write(root: Path, name: str, rows: list[str]) -> None:
    (root / name).write_text("wavfilename,alpha,beta\n" + "\n".join(rows) + "\n")


def _release(tmp_path: Path, *, rows: dict[str, list[str]] | None = None) -> tuple[Path, Path]:
    csv_root = tmp_path / "csv"
    audio_root = tmp_path / "wav"
    csv_root.mkdir()
    default = {
        "fg-trn": ["a_fg_trn.wav,1,", "b_fg_trn.wav,,1"],
        "fg-tst": ["a_fg_tst.wav,1,"],
        "bg-trn": ["a_bg_trn.wav,1,", "b_bg_trn.wav,,1"],
        "bg-tst": ["a_bg_tst.wav,1,"],
    }
    listings = default if rows is None else rows
    for key, lines in listings.items():
        _write(csv_root, f"{SPECIES}-{SPAN}-{key}.csv", lines)
    for condition in ("fg", "bg"):
        folder = audio_root / f"{SPECIES}-{condition}"
        folder.mkdir(parents=True)
        for key, lines in listings.items():
            if not key.startswith(condition):
                continue
            for line in lines:
                (folder / line.split(",")[0]).write_bytes(b"")
    return csv_root, audio_root


# --- what the release says -----------------------------------------------------


def test_training_becomes_enrollment_and_test_becomes_query() -> None:
    """The authors' own split is used rather than one invented here."""

    assert set(STOWELL_ENDPOINTS) == {
        "chiffchaff-withinyear",
        "chiffchaff-acrossyear",
        "littleowl-acrossyear",
        "pipit-withinyear",
        "pipit-acrossyear",
    }


def test_both_conditions_are_carried_in_one_endpoint(tmp_path: Path) -> None:
    """The background probe is a change of condition, not a separate build."""

    csv_root, audio_root = _release(tmp_path)
    endpoint = load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)
    conditions = {record.context["condition"] for record in endpoint.records}
    assert conditions == {"foreground", "background"}
    assert len(endpoint.records) == 6
    assert endpoint.identities == ["alpha", "beta"]


def test_a_record_reports_its_condition_to_the_head(tmp_path: Path) -> None:
    csv_root, audio_root = _release(tmp_path)
    endpoint = load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)
    backgrounds = [
        record.as_evaluation_record()
        for record in endpoint.records
        if record.context["condition"] == "background"
    ]
    assert backgrounds
    assert {record["condition"] for record in backgrounds} == {"background"}


def test_splits_come_from_the_published_lists(tmp_path: Path) -> None:
    csv_root, audio_root = _release(tmp_path)
    endpoint = load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)
    enrollment = {r.filename for r in endpoint.records if r.split == "enrollment"}
    query = {r.filename for r in endpoint.records if r.split == "query"}
    assert enrollment == {"a_fg_trn.wav", "b_fg_trn.wav", "a_bg_trn.wav", "b_bg_trn.wav"}
    assert query == {"a_fg_tst.wav", "a_bg_tst.wav"}


# --- what it refuses -----------------------------------------------------------


def test_a_row_marked_for_two_individuals_is_refused(tmp_path: Path) -> None:
    """Guessing which of two is meant would put a wrong label into a result."""

    csv_root, audio_root = _release(
        tmp_path,
        rows={
            "fg-trn": ["a_fg_trn.wav,1,1", "b_fg_trn.wav,,1"],
            "fg-tst": ["a_fg_tst.wav,1,"],
            "bg-trn": ["a_bg_trn.wav,1,"],
            "bg-tst": ["a_bg_tst.wav,1,"],
        },
    )
    with pytest.raises(ValueError, match="marked for 2 individuals"):
        load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)


def test_a_row_marked_for_nobody_is_refused(tmp_path: Path) -> None:
    csv_root, audio_root = _release(
        tmp_path,
        rows={
            "fg-trn": ["a_fg_trn.wav,,", "b_fg_trn.wav,,1"],
            "fg-tst": ["a_fg_tst.wav,1,"],
            "bg-trn": ["a_bg_trn.wav,1,"],
            "bg-tst": ["a_bg_tst.wav,1,"],
        },
    )
    with pytest.raises(ValueError, match="marked for 0 individuals"):
        load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)


def test_a_listed_clip_that_is_not_on_disk_is_refused(tmp_path: Path) -> None:
    """Dropping it silently would shrink the benchmark without anything saying so."""

    csv_root, audio_root = _release(tmp_path)
    (audio_root / f"{SPECIES}-fg" / "a_fg_trn.wav").unlink()
    with pytest.raises(ValueError, match="not on disk"):
        load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)


def test_a_missing_list_is_refused(tmp_path: Path) -> None:
    csv_root, audio_root = _release(tmp_path)
    (csv_root / f"{SPECIES}-{SPAN}-bg-tst.csv").unlink()
    with pytest.raises(ValueError, match="no list at"):
        load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)


def test_an_unknown_endpoint_names_the_ones_that_exist(tmp_path: Path) -> None:
    csv_root, audio_root = _release(tmp_path)
    with pytest.raises(ValueError, match="littleowl-acrossyear"):
        load_stowell(endpoint="littleowl-withinyear", csv_root=csv_root, audio_root=audio_root)


def test_a_list_without_the_expected_first_column_is_refused(tmp_path: Path) -> None:
    csv_root, audio_root = _release(tmp_path)
    (csv_root / f"{SPECIES}-{SPAN}-fg-trn.csv").write_text("file,alpha\na.wav,1\n")
    with pytest.raises(ValueError, match="wavfilename"):
        load_stowell(endpoint="pipit-withinyear", csv_root=csv_root, audio_root=audio_root)
