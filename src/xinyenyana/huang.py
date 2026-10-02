"""The Huang release: two more species, and the embeddings its authors published.

Huang and colleagues published raw audio for red-tailed black cockatoo and
little penguin, a five-fold cross-validation split for each, and BirdNET and
Google-Perch embeddings for those two species and for the three in the Stowell
release. The embeddings make a reproduction check possible: this project's head
can be run on their vectors and their folds, so a disagreement points at the
head or the split rather than staying invisible.

Every split here is the release's own. The fold is named in the endpoint rather
than defaulted, so no run is scored on a fold nobody chose.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.archive import canonical_sha256, sha256_file
from xinyenyana.evaluation import HEADS, KERNEL_RIDGE, PER_CLIP_L2, evaluate_endpoint

#: Folder per species, relative to the unpacked release.
HUANG_FOLDERS: dict[str, str] = {
    "cockatoo": "rtbc-call-types/begging",
    "penguin": "littlepenguin-display_call-exhale",
}

#: The extra nights the authors added so that a method can be enrolled on one
#: night and scored on another. Two nests have no recordings on those nights.
HUANG_EXTRA_NIGHTS = "littlepenguin-display_call-exhale_extra-days"

#: The release publishes five folds. A run names the one it used.
HUANG_FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5)

#: Endpoint name to the species and the fold it scores. ``None`` is the
#: across-night split, which the release defines by folder rather than by fold.
HUANG_ENDPOINTS: dict[str, tuple[str, int | None]] = {
    **{
        f"{species}-fold{fold}": (species, fold)
        for species in sorted(HUANG_FOLDERS)
        for fold in HUANG_FOLDS
    },
    "penguin-acrossnight": ("penguin", None),
}

#: Where each species' published embeddings live, and what the file is called.
#: Chiffchaff's is named for the year span it covers; the rest are not.
PUBLISHED_EMBEDDINGS: dict[str, tuple[str, str]] = {
    "chiffchaff": ("chiffchaff-fg", "within-year-embeddings.json"),
    "littleowl": ("littleowl-fg", "embeddings.json"),
    "pipit": ("pipit-fg", "embeddings.json"),
    "cockatoo": ("rtbc-call-types/begging", "embeddings.json"),
    "penguin": ("littlepenguin-display_call-exhale", "embeddings.json"),
}

#: The two encoders the authors ran, under the folder names they used.
PUBLISHED_MODELS: dict[str, str] = {
    "birdnet": "birdnet-embeddings",
    "google-perch": "google-perch-embeddings",
}

#: How a published embedding file can be split. The five folds are the release's
#: own cross-validation, drawn over clips. ``published`` is the split the people
#: who recorded the birds published with the audio, which the embedding files
#: carry in their ``dataset_type`` field for the three Stowell species. The two
#: cut the same clips differently, so running both on identical vectors
#: separates what the split contributes from what the encoder contributes.
PUBLISHED_SPLITS: tuple[str, ...] = (
    *(f"fold{fold}" for fold in HUANG_FOLDS),
    "published",
    "acrossnight",
)

#: The little penguin's extra nights as embeddings. They carry no fold columns,
#: because they are the query side of the across-night split and nothing else.
PENGUIN_EXTRA_EMBEDDINGS = (HUANG_EXTRA_NIGHTS, "embeddings.json")

#: The release ships two independent fold assignments for the two species whose
#: embedding files name their clips: one in `metadata.csv` and one in the
#: embedding files. Measured on identical filenames, they disagree about which
#: side of the split a clip falls on for roughly a third of clips, on every one
#: of the five folds, for both the cockatoo and the penguin. A fold number
#: therefore means one thing in the audio endpoints, which read the CSV, and a
#: different thing in the published-embedding runs, which read the embedding
#: file. The two are not comparable and no code here compares them.
FOLDS_DIFFER_BETWEEN_CSV_AND_EMBEDDINGS = ("cockatoo", "penguin")

#: The ``dataset_type`` values that mean enrollment and query in the original
#: release. Tree pipit's file also carries its across-year clips, marked
#: ``more`` and ``years``; those are a different endpoint and are left out
#: rather than folded in.
_DATASET_TYPES = {"train": "enrollment", "test": "query"}


def _metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} lists no clips")
    for column in ("file_name", "class"):
        if column not in rows[0]:
            raise ValueError(f"{path} has no {column} column")
    return rows


def _fold_split(row: Mapping[str, Any], fold: int, path: Path) -> str:
    """Enrollment or query for one clip, from the fold the release published.

    The CSV files spell the flag ``True`` and ``False``; the embedding files
    carry it as a JSON boolean. Anything else is a fault in the release rather
    than a split to guess at.
    """

    column = f"is_train-fold_{fold}"
    if column not in row:
        raise ValueError(f"{path} has no {column} column")
    value = row[column]
    if isinstance(value, bool):
        return "enrollment" if value else "query"
    if value not in ("True", "False"):
        raise ValueError(f"{path}: {column} is {value!r}, which is neither True nor False")
    return "enrollment" if value == "True" else "query"


def _folded_records(*, species: str, fold: int, root: Path) -> tuple[list[ClipRecord], list[str]]:
    folder = root / HUANG_FOLDERS[species]
    listing = folder / "metadata.csv"
    if not listing.is_file():
        raise ValueError(f"the release holds no list at {listing}")
    records = []
    for row in _metadata_rows(listing):
        path = folder / row["file_name"]
        if not path.is_file():
            raise ValueError(f"{listing.name} names {row['file_name']}, which is not on disk")
        context: dict[str, Any] = {"condition": "foreground", "species": species, "fold": fold}
        if row.get("call_type"):
            context["call_type"] = row["call_type"]
        records.append(
            ClipRecord(
                filename=row["file_name"],
                path=path,
                identity=row["class"],
                split=_fold_split(row, fold, listing),
                context=context,
            )
        )
    return records, [sha256_file(listing)]


#: The extra nights name each clip `sel.<selection>.<channel>.<YYMMDD>.<HHMMSS>.<hundredths>..wav`,
#: which is the only place the release states when a query clip was recorded.
#: The main folder's filenames carry no date at all, so the enrolment side of
#: the across-night split has none and only the query side can be dated.
EXTRA_NIGHT_FILENAME = re.compile(r"^sel\.\d+\.ch\d+\.(?P<date>\d{6})\.(?P<time>\d{6})\.\d+\.+wav$")


def extra_night_date(filename: str) -> str:
    """The calendar date a penguin extra-night clip was recorded, from its name.

    A filename that does not match the published pattern is an error rather
    than a clip quietly given no date: the recording-day probe would then be
    measured on a subset nobody chose.
    """

    import datetime

    matched = EXTRA_NIGHT_FILENAME.match(filename)
    if matched is None:
        raise ValueError(f"{filename!r} is not a published extra-night clip name")
    return datetime.datetime.strptime(matched.group("date"), "%y%m%d").date().isoformat()


def _across_night_records(root: Path) -> tuple[list[ClipRecord], list[str]]:
    """Every main-folder clip enrolls; every extra-night clip is scored.

    This is the split the release's own README defines. Two nests were not
    recorded on the extra nights, so they are enrolled and never queried, which
    is what that instruction says to do and is left in place rather than
    trimmed.
    """

    enrolled, digests = _folded_records(species="penguin", fold=1, root=root)
    records = [
        ClipRecord(
            filename=record.filename,
            path=record.path,
            identity=record.identity,
            split="enrollment",
            context={**record.context, "night": "first", "fold": None},
        )
        for record in enrolled
    ]
    extra = root / HUANG_EXTRA_NIGHTS
    if not extra.is_dir():
        raise ValueError(f"the release holds no extra nights at {extra}")
    listed = []
    for nest in sorted(path for path in extra.iterdir() if path.is_dir()):
        if not nest.name.endswith("_extra-day"):
            continue
        identity = nest.name[: -len("_extra-day")]
        for clip in sorted(nest.glob("*.wav")):
            filename = f"{nest.name}/{clip.name}"
            listed.append((filename, identity))
            records.append(
                ClipRecord(
                    filename=filename,
                    path=clip,
                    identity=identity,
                    split="query",
                    context={
                        "condition": "foreground",
                        "species": "penguin",
                        "night": "later",
                        "fold": None,
                        "recorded": extra_night_date(clip.name),
                    },
                )
            )
    if not listed:
        raise ValueError(f"{extra} holds no clips")
    # The extra nights ship no list, so the digest is over what is on disk.
    digests.append(canonical_sha256(sorted(listed)))
    return records, digests


def load_huang(*, endpoint: str, root: Path) -> Endpoint:
    """One species and one published split from the Huang release."""

    if endpoint not in HUANG_ENDPOINTS:
        raise ValueError(f"unknown Huang endpoint {endpoint!r}: {sorted(HUANG_ENDPOINTS)}")
    species, fold = HUANG_ENDPOINTS[endpoint]
    if fold is None:
        records, digests = _across_night_records(root)
    else:
        records, digests = _folded_records(species=species, fold=fold, root=root)
    return Endpoint(
        name=endpoint,
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256=canonical_sha256(sorted(digests)),
        source_document="docs/benchmark-data.md",
    )


def _published_split(entry: Mapping[str, Any], split: str, path: Path) -> str | None:
    """Enrollment, query, or out of this split entirely."""

    if split.startswith("fold"):
        return _fold_split(entry, int(split.removeprefix("fold")), path)
    if "dataset_type" not in entry:
        raise ValueError(
            f"{path} carries no dataset_type, so the recordists' own split is not in it"
        )
    return _DATASET_TYPES.get(str(entry["dataset_type"]))


def _embedding_lines(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise ValueError(f"the release holds no embeddings at {path}")
    return [
        (index + 1, json.loads(line))
        for index, line in enumerate(path.read_text().splitlines())
        if line.strip()
    ]


def load_published_vectors(
    *, species: str, model: str, split: str, root: Path
) -> tuple[Endpoint, Any]:
    """The authors' own embeddings under one of their splits.

    Their files carry the identity and the split membership of each vector but
    not always a filename, so the records are named by their line in the file.
    The vectors come back in the same order, and nothing here reads any audio.
    """

    import numpy as np

    if species not in PUBLISHED_EMBEDDINGS:
        raise ValueError(f"unknown species {species!r}: {sorted(PUBLISHED_EMBEDDINGS)}")
    if model not in PUBLISHED_MODELS:
        raise ValueError(f"unknown model {model!r}: {sorted(PUBLISHED_MODELS)}")
    if split not in PUBLISHED_SPLITS:
        raise ValueError(f"unknown split {split!r}: {list(PUBLISHED_SPLITS)}")
    folder, filename = PUBLISHED_EMBEDDINGS[species]
    path = root / folder / PUBLISHED_MODELS[model] / filename
    sources: list[tuple[Path, str | None]] = [(path, None)]
    if split == "acrossnight":
        if species != "penguin":
            raise ValueError("only the penguin was recorded again on other nights")
        extra_folder, extra_filename = PENGUIN_EXTRA_EMBEDDINGS
        sources = [
            (path, "enrollment"),
            (root / extra_folder / PUBLISHED_MODELS[model] / extra_filename, "query"),
        ]

    records = []
    rows = []
    width: int | None = None
    for source, forced in sources:
        for line_number, entry in _embedding_lines(source):
            belongs = forced if forced is not None else _published_split(entry, split, source)
            if belongs is None:
                continue
            vector = entry["embedding"]
            if width is None:
                width = len(vector)
            elif len(vector) != width:
                raise ValueError(f"{source} line {line_number} is {len(vector)} wide, not {width}")
            rows.append(vector)
            records.append(
                ClipRecord(
                    filename=f"{source.parent.parent.name}-line-{line_number}",
                    path=source,
                    identity=str(entry["class"]),
                    split=belongs,
                    context={"condition": "foreground", "species": species, "split": split},
                )
            )
    endpoint = Endpoint(
        name=f"{species}-published-{model}-{split}",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256=canonical_sha256(sorted(sha256_file(source) for source, _ in sources)),
        source_document="docs/benchmark-data.md",
    )
    return endpoint, np.asarray(rows, dtype=np.float64)


#: The head every figure in this project goes through, so a number measured on
#: the authors' vectors is comparable with every number measured on ours.
RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 23


def run_published_embeddings(
    *, species: str, model: str, split: str, root: Path, head: str = KERNEL_RIDGE
) -> dict[str, Any]:
    """This project's head on the authors' own vectors under one of their splits.

    Nothing here recomputes an embedding. The only thing that changes between
    this and a figure measured elsewhere in the project is where the vectors
    came from, so a disagreement with the authors' published accuracy points at
    the head or the split. Running the same vectors under two of their splits
    separates the two, and running each split under all three heads separates
    the split from the classifier.
    """

    if head not in HEADS:
        raise ValueError(f"unknown head: {head}")

    endpoint, vectors = load_published_vectors(species=species, model=model, split=split, root=root)
    evaluated = evaluate_endpoint(
        records=[record.as_evaluation_record() for record in endpoint.records],
        vectors=vectors,
        representation=f"published-{model}",
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=RIDGE_LAMBDA,
        seed=SEED,
        permutations=PERMUTATIONS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        standardisation=PER_CLIP_L2,
        head=head,
    )
    return {
        "endpoint": endpoint.name,
        "head": head,
        "species": species,
        "model": model,
        "split": split,
        "identities": len(endpoint.identities),
        "clips": len(endpoint.records),
        "width": int(vectors.shape[1]),
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "seed": SEED,
        "accuracy": evaluated["classification"]["accuracy"],
        "identity_block_bootstrap_accuracy_95": evaluated["identity_block_bootstrap_accuracy_95"],
        "permutation_p": evaluated["permutation_control"]["p_value_plus_one"],
        "roc_auc": evaluated["verification"]["roc_auc"],
        "enrollment_calls": evaluated["enrollment_calls"],
        "query_calls": evaluated["query_calls"],
    }
