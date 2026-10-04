"""PA-V6: the pieces that decide what a number means, tested without a model."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint


def _record(name: str, identity: str, split: str, condition: str = "foreground") -> ClipRecord:
    return ClipRecord(
        filename=name,
        path=Path(f"/nonexistent/{name}.wav"),
        identity=identity,
        split=split,
        context={"condition": condition},
    )


# --- how a clip is cut ---------------------------------------------------------


def test_windows_pad_the_last_window_with_silence_and_keep_every_sample() -> None:
    from xinyenyana.bioacoustic import windows

    signal = np.arange(1, 11, dtype=np.float32)
    cut = windows(signal, 4)
    assert cut.shape == (3, 4)
    assert cut.reshape(-1)[:10].tolist() == signal.tolist()
    assert cut[-1, 2:].tolist() == [0.0, 0.0]
    assert windows(np.ones(2, dtype=np.float32), 4).tolist() == [[1.0, 1.0, 0.0, 0.0]]


def test_pool_tokens_reads_batch_first_and_time_first_blocks_alike() -> None:
    from xinyenyana.bioacoustic import pool_tokens

    tokens = np.random.default_rng(0).normal(size=(7, 5))
    assert np.allclose(pool_tokens(tokens[None]), tokens.mean(axis=0))
    assert np.allclose(pool_tokens(tokens[:, None, :]), tokens.mean(axis=0))
    assert np.allclose(pool_tokens((tokens[None], "attention")), tokens.mean(axis=0))
    assert np.allclose(pool_tokens(tokens.mean(axis=0)[None]), tokens.mean(axis=0))
    feature_map = np.random.default_rng(1).normal(size=(1, 6, 3, 4))
    assert np.allclose(pool_tokens(feature_map), feature_map[0].mean(axis=(1, 2)))


def test_resampling_removes_what_the_new_rate_cannot_carry() -> None:
    from xinyenyana.bioacoustic import resample

    rate = 48_000
    t = np.arange(rate) / rate
    tone = np.sin(2 * np.pi * 12_000 * t)
    out = resample(tone, rate, 16_000)
    assert len(out) == 16_000
    assert np.sqrt(np.mean(out**2)) < 0.01


def test_blocks_are_found_at_the_innermost_level_only() -> None:
    torch = pytest.importorskip("torch")
    from xinyenyana.bioacoustic import transformer_blocks

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

    class Stage(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(), Block()])

    class Net(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = torch.nn.ModuleList([torch.nn.Conv1d(1, 1, 3) for _ in range(3)])
            self.stages = torch.nn.ModuleList([Stage(), Stage(), Stage()])

    found = [path for path, _ in transformer_blocks(Net())]
    assert found == [f"stages.{s}.blocks.{b}" for s in range(3) for b in range(2)]

    class Timm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stem = torch.nn.Sequential(torch.nn.Conv1d(1, 1, 3), torch.nn.ReLU())
            self.blocks = torch.nn.Sequential(Block(), Block(), Block())

    assert [path for path, _ in transformer_blocks(Timm())] == [f"blocks.{b}" for b in range(3)]

    class Cvt(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.ModuleList([Block()])
            self.second = torch.nn.ModuleList([Block(), Block()])

    assert [path for path, _ in transformer_blocks(Cvt())] == ["first.0", "second.0", "second.1"]


# --- the v5 speech path ----------------------------------------------------------


def test_minimum_samples_finds_the_shortest_accepted_input() -> None:
    from xinyenyana.a5 import minimum_samples, pad_to, windows_of

    def forward(samples: np.ndarray) -> None:
        if len(samples) < 400:
            raise RuntimeError("Kernel size can't be greater than actual input size")

    assert minimum_samples(forward) == 400
    assert len(pad_to(np.ones(10, dtype=np.float32), 400)) == 400
    long = np.ones(1000, dtype=np.float32)
    assert pad_to(long, 400) is long
    assert [len(w) for w in windows_of(np.ones(25), 10)] == [10, 10, 5]
    assert len(windows_of(np.ones(25), None)) == 1


# --- score normalisation --------------------------------------------------------


def test_raw_scores_name_the_same_bird_as_the_class_mean_head() -> None:
    from xinyenyana.evaluation import CLASS_MEAN, evaluate_endpoint
    from xinyenyana.score_normalisation import adaptive_s_norm

    rng = np.random.default_rng(3)
    centres = rng.normal(size=(4, 16))
    labels = [str(i) for i in range(4) for _ in range(6)]
    enrol = np.vstack([centres[int(v)] + 0.8 * rng.normal(size=16) for v in labels])
    query = np.vstack([centres[int(v)] + 0.8 * rng.normal(size=16) for v in labels])
    raw, normalised, order = adaptive_s_norm(
        enrolment=enrol,
        enrolment_identities=labels,
        query=query,
        cohort=rng.normal(size=(30, 16)),
    )
    records = [
        {"filename": f"e{i}", "identity": v, "split": "enrollment", "condition": "foreground"}
        for i, v in enumerate(labels)
    ] + [
        {"filename": f"q{i}", "identity": v, "split": "query", "condition": "foreground"}
        for i, v in enumerate(labels)
    ]
    result = evaluate_endpoint(
        records=records,
        vectors=np.vstack((enrol, query)),
        representation="test",
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=1.0,
        seed=1,
        permutations=5,
        bootstrap_replicates=10,
        standardisation="per-clip L2",
        head=CLASS_MEAN,
        extended_nulls=False,
    )
    predicted = [p["predicted_label"] for p in result["predictions"]]
    assert [order[i] for i in np.argmax(raw, axis=1)] == predicted
    assert normalised.shape == raw.shape


# --- open set ---------------------------------------------------------------------


def test_as_norm_retains_correct_and_incorrect_decisions_for_both_conditions(monkeypatch):
    from xinyenyana import v6_analyses

    monkeypatch.setattr(v6_analyses, "REPLICATES", 20)
    records = tuple(
        _record(f"{split}-{condition}-{i}", str(i % 4), split, condition)
        for split in ("enrollment", "query")
        for condition in ("foreground", "background")
        for i in range(12)
    )
    endpoint = Endpoint(
        name="test",
        records=records,
        categorical_targets=(),
        manifest_sha256="test",
        source_document="test",
    )
    vectors = np.random.default_rng(129).normal(size=(len(records), 16))
    result = v6_analyses.as_norm_block(endpoint, vectors)
    for condition, field in (("foreground", "calls_scored"), ("background", "backgrounds_scored")):
        block = result[field]
        predictions = block["predictions"]
        query = [r for r in records if r.split == "query" and r.context["condition"] == condition]
        assert predictions["query_ids"] == [r.filename for r in query]
        labels = predictions["labels"]
        actual = predictions["actual_label_indices"]
        assert [labels[i] for i in actual] == [r.identity for r in query]
        for head in ("raw_class_mean", "as_norm"):
            predicted = predictions[head]
            flags = "".join("1" if a == p else "0" for a, p in zip(actual, predicted, strict=True))
            assert "0" in flags
            assert flags == block[head]["query_correct"]
            assert flags.count("1") / len(flags) == block[head]["accuracy"]


def test_calling_every_animal_new_scores_zero_on_the_geometric_mean() -> None:
    from xinyenyana.open_set import balanced_known_unknown

    observations = [
        {"identity": "a", "is_known": True, "top_identity": "a", "maximum_score": 0.2},
        {"identity": "b", "is_known": True, "top_identity": "b", "maximum_score": 0.3},
        {"identity": "c", "is_known": False, "top_identity": "a", "maximum_score": 0.1},
    ]
    everything_new = balanced_known_unknown(observations, threshold=1.0)
    assert everything_new["balanced_accuracy_known"] == 0.0
    assert everything_new["balanced_accuracy_unknown"] == 1.0
    assert everything_new["geometric_mean_known_unknown"] == 0.0
    sensible = balanced_known_unknown(observations, threshold=0.15)
    assert sensible["geometric_mean_known_unknown"] == pytest.approx(1.0)


# --- merging sweeps ------------------------------------------------------------------


def _sweep(models: dict[str, dict[str, dict[str, object]]]) -> dict[str, object]:
    chosen = {}
    for model, curve in models.items():
        best = max(curve, key=lambda name: float(curve[name]["enrollment_fisher_ratio"]))  # type: ignore[arg-type]
        chosen[model] = {"representation": best, **curve[best]}
    return {
        "endpoint": "e",
        "query_identities": ["a", "b"],
        "curve": models,
        "chosen": chosen,
        "not_computed": [],
    }


def test_merging_unites_candidates_and_chooses_again_by_fisher_ratio() -> None:
    from xinyenyana.v6_analyses import merge_sweeps

    bird = {
        "birdnet-v2.4-x1": {"enrollment_fisher_ratio": 1.0, "accuracy": 0.5, "query_correct": "10"}
    }
    old = _sweep(
        {
            "birdnet-v2.4": bird,
            "m": {
                "m-x1-l00": {"enrollment_fisher_ratio": 0.2, "accuracy": 0.1, "query_correct": "00"}
            },
        }
    )
    new = _sweep(
        {
            "birdnet-v2.4": bird,
            "m": {
                "m-x2-l03": {"enrollment_fisher_ratio": 0.9, "accuracy": 0.4, "query_correct": "01"}
            },
        }
    )
    merged = merge_sweeps([old, new])
    assert merged["chosen"]["m"]["representation"] == "m-x2-l03"
    assert set(merged["curve"]["m"]) == {"m-x1-l00", "m-x2-l03"}
    assert merged["merge"]["models_united"] == ["birdnet-v2.4", "m"]


def test_merging_refuses_files_whose_birdnet_answers_differ() -> None:
    from xinyenyana.v6_analyses import merge_sweeps

    one = _sweep({"birdnet-v2.4": {"b": {"enrollment_fisher_ratio": 1.0, "query_correct": "10"}}})
    two = _sweep({"birdnet-v2.4": {"b": {"enrollment_fisher_ratio": 1.0, "query_correct": "11"}}})
    with pytest.raises(ValueError, match="BirdNET"):
        merge_sweeps([one, two])


# --- the vector store ----------------------------------------------------------------


def test_stored_vectors_come_back_only_for_the_endpoint_they_were_written_for(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from xinyenyana.a6 import load_vectors, store_vectors

    endpoint = Endpoint(
        name="e",
        records=(_record("x", "a", "enrollment"), _record("y", "b", "query")),
        categorical_targets=(),
        manifest_sha256="abc",
        source_document="docs/measured.md",
    )
    vectors = np.arange(6, dtype=np.float32).reshape(2, 3)
    entry = store_vectors(
        store=tmp_path,
        endpoint=endpoint,
        model="m/x",
        representation="r",
        vectors=vectors,
        arms={"own_0db": vectors[:1]},
    )
    back, arms, name = load_vectors(Path(entry["path"]), endpoint)
    assert name == "r" and np.array_equal(back, vectors) and set(arms) == {"own_0db"}
    with pytest.raises(ValueError, match="manifest"):
        load_vectors(Path(entry["path"]), replace(endpoint, manifest_sha256="other"))


# --- the sensitivity floor --------------------------------------------------------------


def test_donors_are_the_best_recorded_birds_and_never_cross_splits() -> None:
    from xinyenyana.sensitivity_floor import pair_donors, plan_planting

    donor_records = [
        _record(f"d{b}-{i}-{s}", f"donor{b}", s)
        for b, count in ((1, 3), (2, 5), (3, 4))
        for s in ("enrollment", "query")
        for i in range(count)
    ]
    target_records = [
        _record(f"t{b}-{i}-{s}", f"target{b}", s, "background")
        for b in (1, 2)
        for s in ("enrollment", "query")
        for i in range(2)
    ]
    donor = Endpoint(
        name="d",
        records=tuple(donor_records),
        categorical_targets=(),
        manifest_sha256="d",
        source_document="x",
    )
    target = Endpoint(
        name="t",
        records=tuple(target_records),
        categorical_targets=(),
        manifest_sha256="t",
        source_document="x",
    )
    assert pair_donors(donor, target) == [("donor2", "target1"), ("donor3", "target2")]
    for call, background in plan_planting(donor, target):
        assert call.split == background.split
        assert background.identity == {"donor2": "target1", "donor3": "target2"}[call.identity]


# --- the great tit at full width -------------------------------------------------------


def test_great_tits_enter_only_with_songs_in_two_years(tmp_path: Path) -> None:
    from xinyenyana.great_tit_full import fathers_by_attempt, male_great_tits, select_songs

    with (tmp_path / "morph.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bto_ring", "species", "sex"])
        writer.writeheader()
        writer.writerows(
            [
                {"bto_ring": "r1", "species": "g", "sex": "m"},
                {"bto_ring": "r2", "species": "g", "sex": "m"},
                {"bto_ring": "r3", "species": "g", "sex": "f"},
            ]
        )
    with (tmp_path / "main.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pnum", "year", "nestbox", "father", "x", "y"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "pnum": "2020A",
                    "year": "2020",
                    "nestbox": "A",
                    "father": "r1",
                    "x": "1",
                    "y": "2",
                },
                {
                    "pnum": "2021B",
                    "year": "2021",
                    "nestbox": "B",
                    "father": "r1",
                    "x": "3",
                    "y": "4",
                },
                {
                    "pnum": "2020C",
                    "year": "2020",
                    "nestbox": "C",
                    "father": "r2",
                    "x": "5",
                    "y": "6",
                },
                {
                    "pnum": "2021D",
                    "year": "2021",
                    "nestbox": "D",
                    "father": "r3",
                    "x": "7",
                    "y": "8",
                },
            ]
        )
    with (tmp_path / "hits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["", "ID"])
        writer.writeheader()
        for attempt in ("2020A", "2021B", "2020C", "2021D"):
            for i in range(12):
                writer.writerow({"": f"{attempt}_20200101_040000_{i}", "ID": attempt})
    males = male_great_tits(tmp_path / "morph.csv")
    assert males == {"r1", "r2"}
    songs = select_songs(tmp_path / "hits.csv", fathers_by_attempt(tmp_path / "main.csv", males))
    assert {s["identity"] for s in songs} == {"r1"}
    assert sum(s["split"] == "enrollment" for s in songs) == 10
    assert {s["year"] for s in songs if s["split"] == "query"} == {2021}


def test_a_fixed_representation_is_read_in_windows_only_if_it_was_measured_so() -> None:
    from xinyenyana.a6 import fixed_from_sweep, windows_from_sweep

    sweep = {
        "chosen": {
            "a": {"representation": "a-x1-l00"},
            "b": {"representation": "b-x3-l02", "read_in_windows_of_seconds": 60.0},
        }
    }
    assert fixed_from_sweep(sweep, ["a", "b"]) == {"a": "a-x1-l00", "b": "b-x3-l02"}
    assert windows_from_sweep(sweep, ["a", "b"]) == {"a": None, "b": 60.0}


def test_every_representation_gets_an_interval_from_the_shared_draws() -> None:
    from xinyenyana.encoder_comparison import paired_bootstrap

    identities = [a for a in "abcdef" for _ in range(4)]
    result = paired_bootstrap(
        identities=identities,
        correct={"birdnet-v2.4": "1" * 12 + "0" * 12, "m": "10" * 12},
        replicates=10_000,
    )
    low, high = result["comparisons"]["m"]["accuracy_95"]
    assert low <= 0.5 <= high
    assert result["replicates"] == 10_000
    assert result["reference_accuracy_95"][0] < result["reference_accuracy_95"][1]


def test_the_open_set_takes_birdnet_and_the_best_of_each_family() -> None:
    from xinyenyana.v6_analyses import open_set_models

    results = [
        {"chosen": {"birdnet-v2.4": {"accuracy": 0.5}, "perch-v2": {"accuracy": 0.6}}},
        {"chosen": {"aves": {"accuracy": 0.6}, "espnet/xeus": {"accuracy": 0.2}}},
        {"chosen": {"microsoft/wavlm-large": {"accuracy": 0.3}}},
    ]
    assert open_set_models(results) == {
        "reference": "birdnet-v2.4",
        "bioacoustic": "aves",
        "speech": "microsoft/wavlm-large",
    }


def test_held_out_sessions_train_on_other_sessions_only() -> None:
    from xinyenyana.layer_selection import choose, held_out_accuracy, session_folds

    identities = [a for a in "ab" for _ in range(6)]
    sessions = [f"{a}{s}" for a in "ab" for s in (1, 1, 2, 2, 3, 3)]
    folds = session_folds(identities, sessions, folds=5)
    for i, j in zip(range(0, 12, 2), range(1, 12, 2), strict=True):
        assert folds[i] == folds[j]
    rng = np.random.default_rng(4)
    separated = np.vstack(
        [rng.normal(size=8) * 0.1 + (3.0 if a == "a" else -3.0) for a in identities]
    )
    noise = rng.normal(size=(12, 8))
    good = held_out_accuracy(separated, identities, sessions)
    assert good["accuracy"] == 1.0 and good["animals_scored"] == 2
    lonely = held_out_accuracy(separated, identities, ["one"] * 12)
    assert lonely["accuracy"] is None and lonely["clips_scored"] == 0
    curve = {
        "x": {"enrolment_held_out": good, "enrollment_fisher_ratio": 1.0},
        "y": {
            "enrolment_held_out": held_out_accuracy(noise, identities, sessions),
            "enrollment_fisher_ratio": 9.0,
        },
    }
    assert choose(curve) == "x"
    assert choose({"only": {"enrollment_fisher_ratio": 1.0}}) == "only"


def test_single_session_animals_remain_competitors_in_every_fold(monkeypatch) -> None:
    from xinyenyana import evaluation
    from xinyenyana.layer_selection import held_out_accuracy

    seen = []
    original = evaluation.prepare_features

    def capture(enrolment, query, *args, **kwargs):
        seen.append((enrolment.copy(), query.copy()))
        return original(enrolment, query, *args, **kwargs)

    monkeypatch.setattr(evaluation, "prepare_features", capture)
    vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    result = held_out_accuracy(vectors, ["a", "a", "singleton"], ["s1", "s2", "only"])
    assert result["clips_scored"] == 2
    assert result["animals_scored"] == 1
    assert len(seen) == 2
    for train, query in seen:
        assert any(np.array_equal(row, vectors[2]) for row in train)
        assert not any(np.array_equal(row, vectors[2]) for row in query)


def test_later_candidate_and_fixed_runs_cannot_replace_stored_evidence(tmp_path) -> None:
    import hashlib

    from xinyenyana.a6 import load_candidate, load_vectors, store_candidates, store_vectors

    endpoint = Endpoint(
        name="e",
        records=(_record("x", "a", "enrollment"), _record("y", "b", "query")),
        categorical_targets=(),
        manifest_sha256="abc",
        source_document="docs/measured.md",
    )
    full = {"first": np.ones((2, 4)), "second": np.zeros((2, 4))}
    before = store_candidates(store=tmp_path, endpoint=endpoint, model="m", candidates=full)
    later = store_candidates(
        store=tmp_path, endpoint=endpoint, model="m", candidates={"first": full["first"] * 2}
    )
    assert before["path"] != later["path"]
    assert hashlib.sha256(Path(before["path"]).read_bytes()).hexdigest() == before["sha256"]
    assert np.array_equal(load_candidate(Path(before["path"]), endpoint, "second"), full["second"])
    chosen = store_vectors(
        store=tmp_path,
        endpoint=endpoint,
        model="m",
        representation="first",
        vectors=full["first"],
        arms={"own_0db": full["first"][:1]},
    )
    store_vectors(
        store=tmp_path,
        endpoint=endpoint,
        model="m",
        representation="first",
        vectors=full["first"] * 2,
        arms=None,
    )
    back, arms, _ = load_vectors(Path(chosen["path"]), endpoint)
    assert np.array_equal(back, full["first"]) and "own_0db" in arms
    assert not list(tmp_path.rglob("*.partial"))


def test_reproduction_cannot_pass_by_skipping_a_missing_candidate() -> None:
    from xinyenyana.a6 import compare_sweep_candidates

    candidate = {"enrollment_fisher_ratio": 1.0, "accuracy": 0.5, "query_correct": "10"}
    old = _sweep({"m": {"one": candidate, "two": candidate}})
    new = _sweep({"m": {"one": candidate}})
    result = compare_sweep_candidates(old, new)
    assert result["candidates_expected"] == 2
    assert result["identical"] == 1
    assert result["missing"] == [{"model": "m", "candidate": "two"}]
    assert not result["passed"]
    assert compare_sweep_candidates(old, old)["passed"]


def test_reproduction_lists_unreported_differences_and_fails_a_reported_one() -> None:
    from xinyenyana.a6 import compare_sweep_candidates

    same = {"enrollment_fisher_ratio": 1.0, "accuracy": 0.5, "query_correct": "10"}
    moved = {"enrollment_fisher_ratio": 0.2, "accuracy": 1.0, "query_correct": "11"}
    old = {**_sweep({"m": {"one": same, "two": same}}), "chosen": {"m": {"representation": "one"}}}
    unreported = compare_sweep_candidates(old, _sweep({"m": {"one": same, "two": moved}}))
    assert unreported["passed"]
    assert unreported["identical"] == 1
    assert unreported["differing"] == [
        {"model": "m", "candidate": "two", "v5_accuracy": 0.5, "v6_accuracy": 1.0}
    ]
    reported = compare_sweep_candidates(old, _sweep({"m": {"one": moved, "two": same}}))
    assert not reported["passed"]
    assert reported["reported_differing"] == [{"model": "m", "candidate": "one"}]


def test_analyses_refuse_a_missing_or_wrong_reported_vector() -> None:
    from xinyenyana.v6_analyses import analyse_endpoint

    endpoint = Endpoint(
        name="e",
        records=(_record("x", "a", "enrollment"), _record("y", "a", "query")),
        categorical_targets=(),
        manifest_sha256="abc",
        source_document="x",
    )
    result = {"chosen": {"m": {"representation": "new"}}}
    with pytest.raises(ValueError, match="no stored vectors"):
        analyse_endpoint(endpoint, [result], sources=[])
    result["stored_vectors"] = {"m": {"representation": "old"}}
    with pytest.raises(ValueError, match="do not match"):
        analyse_endpoint(endpoint, [result], sources=[])


def test_bioacoustic_input_retains_recorded_level_and_dc(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from xinyenyana import bioacoustic

    recorded = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    monkeypatch.setitem(
        sys.modules, "soundfile", SimpleNamespace(read=lambda *a, **k: (recorded, 16000))
    )
    seen = []

    class Model:
        rate = 16000
        length = 4

        def __init__(self, *args):
            pass

        def embed_window(self, window):
            seen.append(window.copy())
            return {"embedding": window}

        def release(self):
            pass

    monkeypatch.setattr(bioacoustic, "Loaded", Model)
    bioacoustic.candidate_vectors("perch-bird", [_record("x", "a", "enrollment")], device="cpu")
    np.testing.assert_array_equal(seen[0], recorded)


def test_missing_sessions_cannot_turn_into_clip_folds() -> None:
    from xinyenyana.a6 import refresh_selection
    from xinyenyana.layer_selection import held_out_accuracy

    assert held_out_accuracy(np.ones((4, 2)), ["a", "a", "b", "b"], [None] * 4)["accuracy"] is None
    endpoint = Endpoint(
        name="e",
        records=tuple(
            _record(f"{a}-{i}", a, split)
            for a in "ab"
            for i, split in enumerate(("enrollment", "enrollment", "query"))
        ),
        categorical_targets=(),
        manifest_sha256="abc",
        source_document="x",
    )
    entry = {
        "enrollment_fisher_ratio": 1.0,
        "accuracy": 0.5,
        "query_correct": "10",
        "enrolment_held_out": {"accuracy": 1.0},
    }
    original = _sweep({"m": {"one": entry, "two": entry}})
    original.update(
        manifest_sha256="abc",
        splits=endpoint.split_digests(),
        chosen_held_out={"m": {"representation": "one"}},
    )
    result = refresh_selection(endpoint, original)
    assert result["chosen_held_out"] == {}
    for key in ("representation", "accuracy", "query_correct", "enrollment_fisher_ratio"):
        assert result["chosen"]["m"][key] == original["chosen"]["m"][key]
    assert result["chosen"]["m"]["enrolment_held_out"]["accuracy"] is None
    assert result["curve"]["m"]["one"]["query_correct"] == "10"
    assert result["curve"]["m"]["one"]["enrolment_held_out"]["accuracy"] is None
    assert original["chosen_held_out"]["m"]["representation"] == "one"


def test_unavailable_merged_holdout_clears_a_fixed_only_choice() -> None:
    from xinyenyana.v6_analyses import _held_out

    fixed = {
        "curve": {"m": {"only": {}}},
        "chosen_held_out": {"m": {"representation": "only"}},
        "candidate_vectors": {"m": {"path": "fixed"}},
    }
    merged = {"curve": {"m": {"one": {}, "two": {}}}, "chosen_held_out": {}}
    assert _held_out([fixed])
    assert _held_out([fixed, merged]) == {}


def test_great_tit_full_is_available_to_the_headline_loader(monkeypatch, tmp_path) -> None:
    from xinyenyana import cli, great_tit_full

    endpoint = object()
    monkeypatch.setattr(great_tit_full, "load_great_tit_full", lambda root: endpoint)
    assert cli._load_endpoint("great-tit-full", tmp_path) is endpoint


def test_sensitivity_floor_uses_the_registered_headline_null_count(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace

    from xinyenyana import a5, identity_run, sensitivity_floor, v6_analyses

    endpoint = SimpleNamespace(name="e", manifest_sha256="m", records=[])
    seen = []
    monkeypatch.setattr(sensitivity_floor, "pair_donors", lambda *args: [])
    monkeypatch.setattr(sensitivity_floor, "planted_endpoint", lambda *a, **k: endpoint)
    monkeypatch.setattr(a5, "birdnet_vectors", lambda records: [])
    monkeypatch.setattr(identity_run, "measure", lambda **kwargs: seen.append(kwargs) or {})
    v6_analyses.sensitivity_floor_run(endpoint, endpoint, scratch=tmp_path)
    assert len(seen) == 3
    assert all(row["permutations"] == 9999 and row["replicates"] == 10000 for row in seen)


def test_stored_open_set_retains_allocation_provenance_and_observations(tmp_path) -> None:
    from xinyenyana.a6 import store_vectors
    from xinyenyana.archive import canonical_sha256
    from xinyenyana.open_set import ALLOCATION_SEED, ALLOCATIONS, ROLES
    from xinyenyana.v6_analyses import open_set_from_store

    endpoint = Endpoint(
        name="synthetic-open",
        records=tuple(
            _record(f"{split}-{i}", str(i), split)
            for split in ("enrollment", "query")
            for i in range(8)
        ),
        categorical_targets=("identity",),
        manifest_sha256="synthetic",
        source_document="test",
    )
    stored = store_vectors(
        store=tmp_path,
        endpoint=endpoint,
        model="m",
        representation="one-hot",
        vectors=np.tile(np.eye(8), (2, 1)),
        arms=None,
    )
    source = tmp_path / "source.json"
    source.write_text("{}")
    result = open_set_from_store(
        endpoint,
        [{"chosen": {"m": {"representation": "one-hot"}}, "stored_vectors": {"m": stored}}],
        models=["m", "clip-duration-only", "clip-level-only"],
        sources=[source],
    )
    assert set(result["representations"]) == {"m"}
    assert set(result["excluded_representations"]) == {"clip-duration-only", "clip-level-only"}
    allocations = result["representations"]["m"]["allocations"]
    assert result["allocation_seed"] == ALLOCATION_SEED
    assert result["allocations"] == len(allocations) == ALLOCATIONS
    assert result["splits"] == endpoint.split_digests()
    assert result["role_digest"] == canonical_sha256(
        [{role: a["roles"][role] for role in ROLES} for a in allocations]
    )
    assert result["representations"]["m"]["vector_source"] == stored
    for allocation in allocations:
        assert len(allocation["observations"]["calibration"]) == 4
        assert len(allocation["observations"]["test"]) == 4
