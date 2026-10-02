"""A2: which of the three call layers carries individual identity.

The protocol is [`docs/measurement-protocol.md`](../../docs/measurement-protocol.md),
registered before any layer score was read. This module runs it.

Nothing here selects clips, draws splits or chooses a parameter. Each endpoint
is loaded from the manifest that built it, with the same
identities, the same splits and the same clips, and every extraction parameter
is the default recorded in ``features.py``. The only choice this module makes is
which layer is being evaluated, which is the question.

One clip is read once. All four layers come out of that single read, so the
four evaluations of an endpoint see exactly the same audio.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from xinyenyana.archive import canonical_sha256, sha256_file
from xinyenyana.cleaning import Band, clean
from xinyenyana.evaluation import (
    CONSTANT_TARGET_TOLERANCE,
    PER_DIMENSION,
    evaluate_endpoint,
    leave_one_identity_out,
)
from xinyenyana.features import (
    DEFAULT_F0_MAX,
    DEFAULT_F0_MIN,
    energy_share_below_hz,
    layers_from_signal,
    read_clip,
)

LAYERS = ("source", "filter", "filter_normalised", "motor")

RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 17
BAND_EDGE_TOLERANCE = 0.01


@dataclass(frozen=True)
class ClipRecord:
    """One clip, with the identity, split and context its manifest recorded."""

    filename: str
    path: Path
    identity: str
    split: str
    context: dict[str, Any]

    #: Context keys that name the recording session, in the order they are
    #: preferred. Endpoints call it different things; the within-session
    #: permutation and the hierarchical bootstrap need one name for it.
    SESSION_KEYS = ("source_recording", "recording", "session", "stem", "recorded", "date")

    @property
    def session(self) -> str | None:
        """The recording this clip came out of, or None if the endpoint has no such field.

        A session is the unit that is held constant by a within-session
        permutation, and resampled in the middle stage of the hierarchical
        bootstrap. Where an endpoint carries a timestamp rather than a
        recording name, the calendar day is used and the endpoint's protocol
        says so.
        """

        for key in self.SESSION_KEYS:
            value = self.context.get(key)
            if value is None:
                continue
            text = str(value)
            return text[:10] if key == "recorded" else text
        return None

    def as_evaluation_record(self) -> dict[str, Any]:
        """The shape ``evaluate_endpoint`` reads."""

        record = {
            "filename": self.filename,
            "identity": self.identity,
            "split": self.split,
            "condition": str(self.context.get("condition", "foreground")),
        }
        session = self.session
        if session is not None:
            record["session"] = session
        return record


@dataclass(frozen=True)
class Endpoint:
    """One labelled dataset, its clips, and the context targets it supports."""

    name: str
    records: tuple[ClipRecord, ...]
    categorical_targets: tuple[str, ...]
    manifest_sha256: str
    source_document: str

    def __post_init__(self) -> None:
        splits = {record.split for record in self.records}
        if splits != {"enrollment", "query"}:
            raise ValueError(f"{self.name} has splits {sorted(splits)}")
        if len({record.filename for record in self.records}) != len(self.records):
            raise ValueError(f"{self.name} has two clips with one name")

    @property
    def identities(self) -> list[str]:
        return sorted({record.identity for record in self.records})

    def split_digests(self) -> dict[str, str]:
        return {
            split: canonical_sha256(
                sorted(
                    (record.filename, record.identity)
                    for record in self.records
                    if record.split == split
                )
            )
            for split in ("enrollment", "query")
        }


#: The species and year spans published in the Stowell release, and the number
#: of individuals each carries. A span that does not appear here is not in the
#: release; little owl has no within-year split.
STOWELL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "chiffchaff-withinyear": ("chiffchaff", "withinyear"),
    "chiffchaff-acrossyear": ("chiffchaff", "acrossyear"),
    "littleowl-acrossyear": ("littleowl", "acrossyear"),
    "pipit-withinyear": ("pipit", "withinyear"),
    "pipit-acrossyear": ("pipit", "acrossyear"),
}

#: The authors published a training and a test list per span, and a foreground
#: and a background list for each. Training becomes enrollment and test becomes
#: query, so no split is invented here.
_STOWELL_SPLITS = {"trn": "enrollment", "tst": "query"}
_STOWELL_CONDITIONS = {"fg": "foreground", "bg": "background"}


def _read_stowell_list(path: Path) -> list[tuple[str, str]]:
    """Filename and identity from one published list.

    The identity is a one in exactly one column. A row with none or with more
    than one is a fault in the list rather than a clip to guess at.
    """

    rows = path.read_text().splitlines()
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0].split(",")
    if header[0] != "wavfilename":
        raise ValueError(f"{path} does not start with a wavfilename column")
    identities = header[1:]
    listed = []
    for line in rows[1:]:
        if not line.strip():
            continue
        cells = line.split(",")
        marked = [identities[i] for i, cell in enumerate(cells[1:]) if cell.strip() == "1"]
        if len(marked) != 1:
            raise ValueError(f"{path}: {cells[0]} is marked for {len(marked)} individuals")
        listed.append((cells[0], marked[0]))
    return listed


def load_stowell(*, endpoint: str, csv_root: Path, audio_root: Path) -> Endpoint:
    """One species and year span from the Stowell release, foreground and background.

    Both conditions are carried in one endpoint so that the background probe is
    a change of condition rather than a separate build. A background recording
    is the ambient sound from an individual's location with no call in it; it is
    never a feature, and a method that identifies individuals from it has been
    caught reading the place.
    """

    if endpoint not in STOWELL_ENDPOINTS:
        raise ValueError(f"unknown Stowell endpoint {endpoint!r}: {sorted(STOWELL_ENDPOINTS)}")
    species, span = STOWELL_ENDPOINTS[endpoint]

    records = []
    digests = []
    for condition_key, condition in _STOWELL_CONDITIONS.items():
        for split_key, split in _STOWELL_SPLITS.items():
            listing = csv_root / f"{species}-{span}-{condition_key}-{split_key}.csv"
            if not listing.is_file():
                raise ValueError(f"the release holds no list at {listing}")
            digests.append(sha256_file(listing))
            for filename, identity in _read_stowell_list(listing):
                path = audio_root / f"{species}-{condition_key}" / filename
                if not path.is_file():
                    raise ValueError(f"{listing.name} names {filename}, which is not on disk")
                records.append(
                    ClipRecord(
                        filename=filename,
                        path=path,
                        identity=identity,
                        split=split,
                        context={"condition": condition, "species": species, "span": span},
                    )
                )
    return Endpoint(
        name=endpoint,
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256=canonical_sha256(sorted(digests)),
        source_document="docs/benchmark-data.md",
    )


def load_rookid(*, manifest_path: Path, sample_root: Path) -> Endpoint:
    """The context-matched RookID sample: 5 captive rooks, one call type.

    A sample built on a channel other than the one E01d registered is named
    ``rookid-ch<N>``, so its runs cannot be confused with the channel-0 ones.
    The rule that picks the channel is in
    ``docs/measurement-protocol.md``.
    """

    manifest = json.loads(manifest_path.read_text())
    channel = manifest.get("channel", {})
    name = "rookid"
    if channel.get("overridden"):
        name = f"rookid-ch{int(channel['used'])}"
    records = tuple(
        ClipRecord(
            filename=str(entry["filename"]),
            path=sample_root / str(entry["local_path"]),
            identity=str(entry["identity"]),
            split=str(entry["split"]),
            context={"recording": str(entry["recording"]), "event": str(entry["event"])},
        )
        for entry in manifest["sample"]["files"]
    )
    return Endpoint(
        name=name,
        records=records,
        categorical_targets=(),
        manifest_sha256=sha256_file(manifest_path),
        source_document="docs/measured.md",
    )


def load_zebra_finch(*, manifest_path: Path, sample_root: Path) -> Endpoint:
    """The Zebra Finch sample: 9 captive birds across 24 recording dates."""

    manifest = json.loads(manifest_path.read_text())
    records = tuple(
        ClipRecord(
            filename=str(entry["opaque_id"]),
            path=sample_root / str(entry["local_path"]),
            identity=str(entry["identity"]),
            split=str(entry["split"]),
            context={
                "date": str(entry["date"]),
                "calendar_date_ordinal": float(
                    datetime.strptime(str(entry["date"]), "%y%m%d").toordinal()
                ),
            },
        )
        for entry in manifest["records"]
    )
    return Endpoint(
        name="zebra-finch",
        records=records,
        categorical_targets=(),
        manifest_sha256=sha256_file(manifest_path),
        source_document="docs/measured.md",
    )


def load_great_tit(*, manifest_path: Path, sample_root: Path) -> Endpoint:
    """The Great Tit sample: 16 wild birds at Wytham across three years."""

    manifest = json.loads(manifest_path.read_text())
    records = tuple(
        ClipRecord(
            filename=str(entry["event_token"]),
            path=sample_root / str(entry["clip_path"]),
            identity=str(entry["identity"]),
            split=str(entry["split"]),
            context={
                "year": str(entry["year"]),
                "cohort": str(entry["cohort"]),
                "nestbox": str(entry["nestbox"]),
                "source_recording": str(entry["source_recording"]),
                "song_key": Path(str(entry["member"])).stem,
                "nest_x": float(entry["context"]["nest_x"]),
                "nest_y": float(entry["context"]["nest_y"]),
                "quality": dict(entry["quality"]["metrics"]),
                "annotation": dict(entry["annotation"]),
            },
        )
        for entry in manifest["records"]
    )
    return Endpoint(
        name="great-tit",
        records=records,
        categorical_targets=("year", "cohort"),
        manifest_sha256=sha256_file(manifest_path),
        source_document="docs/measured.md",
    )


def extract_endpoint_layers(
    records: Sequence[ClipRecord],
    *,
    steps: tuple[str, ...] = (),
    band: Band | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[str]]]:
    """Read every clip once and return the four layer matrices and diagnostics.

    Returns the matrices keyed by layer, one diagnostics row per clip, and the
    field names of each layer so a result can say what a dimension was.

    With ``steps`` set, the A3 cleaning runs on the signal before extraction, so
    a cleaned run and an uncleaned one differ in the audio and in nothing else.
    The duration and level recorded per clip are those of the signal the layers
    actually saw.
    """

    import numpy as np

    rows: dict[str, list[Any]] = {layer: [] for layer in LAYERS}
    names: dict[str, list[str]] = {}
    diagnostics: list[dict[str, Any]] = []
    for record in records:
        signal, sample_rate = read_clip(record.path)
        cleaning: dict[str, Any] = {}
        if steps:
            signal, cleaning = clean(signal, sample_rate, steps=steps, band=band)
        layers = layers_from_signal(signal, sample_rate)
        for layer in LAYERS:
            vector = layers[layer]
            rows[layer].append(np.asarray(vector.values, dtype=np.float64))
            names.setdefault(layer, list(vector.names))
        source = layers["source"]
        f0_mean = float(source.values[list(source.names).index("f0_mean")])
        diagnostics.append(
            {
                "filename": record.filename,
                "identity": record.identity,
                "split": record.split,
                "duration_seconds": len(signal) / sample_rate,
                "rms_dbfs": clip_level_dbfs(record.path),
                "f0_mean_hz": f0_mean,
                "f0_at_band_edge": bool(
                    abs(f0_mean - DEFAULT_F0_MIN) <= BAND_EDGE_TOLERANCE * DEFAULT_F0_MIN
                    or abs(f0_mean - DEFAULT_F0_MAX) <= BAND_EDGE_TOLERANCE * DEFAULT_F0_MAX
                ),
                "harmonic_to_noise_db": float(
                    source.values[list(source.names).index("harmonic_to_noise_db")]
                ),
                "voiced_fraction": float(source.diagnostics["voiced_fraction"]),
                "formant_estimation_failed": bool(
                    layers["filter"].diagnostics["formant_estimation_failed"]
                ),
                "notes": int(layers["motor"].diagnostics["notes"]),
                "energy_share_below_50_hz": energy_share_below_hz(signal, sample_rate)[0],
                "cleaning": cleaning,
            }
        )
    matrices = {layer: np.vstack(rows[layer]) for layer in LAYERS}
    return matrices, diagnostics, names


def clip_level_dbfs(path: Path) -> float:
    """The clip's root-mean-square level in decibels, from the stored samples.

    Not from :func:`read_clip`. That divides every clip by its own standard
    deviation, so a level computed after it is 0.0 dBFS for every clip in every
    corpus, to within floating-point dust. The ``rms_dbfs`` context target was
    computed that way, which made it a probe against a constant: on forty little
    owl clips it spanned 0.000002 dB where the stored audio spans 12.58 dB. The
    constant-target guard did not catch it, because its floor is absolute and
    the mean of that target is zero.

    This is the same arithmetic ``shortcut_vectors`` already used for the
    loudness control, which read the stored samples for the same reason.
    """

    import wave
    from array import array

    import numpy as np

    with wave.open(str(path), "rb") as stream:
        raw = array("h")
        raw.frombytes(stream.readframes(stream.getnframes()))
    samples = np.asarray(raw, dtype=np.float64) / 32768.0
    if not len(samples):
        raise ValueError(f"empty WAV: {path}")
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(np.square(samples)))), 1e-12))


def clip_diagnostics(records: Sequence[ClipRecord]) -> list[dict[str, Any]]:
    """The per-clip rows the context probes read, and nothing else.

    ``probe_targets`` uses a clip's filename, split, duration and level. It does
    not use the handcrafted descriptions of the call, so a run that only needs
    the context probes does not pay to compute them. The two numbers are the
    same ones ``extract_endpoint_layers`` records, from the same signal read the
    same way, so a run using this function and one using that one build
    identical targets.

    The difference is what it costs. Those descriptions are framed in samples
    rather than in seconds, so a 250 kHz recording is cut into about six times
    as many frames per second as a 44.1 kHz one, and each frame carries an
    autocorrelation. On the bat endpoint that made the full extraction the
    largest cost in the run by a wide margin.
    """

    rows = []
    for record in records:
        signal, sample_rate = read_clip(record.path)
        rows.append(
            {
                "filename": record.filename,
                "identity": record.identity,
                "split": record.split,
                "duration_seconds": len(signal) / sample_rate,
                "rms_dbfs": clip_level_dbfs(record.path),
            }
        )
    return rows


def estimability(diagnostics: Sequence[dict[str, Any]]) -> dict[str, float]:
    """The diagnostics the protocol requires, reported whatever the scores say."""

    import numpy as np

    return {
        "clips": float(len(diagnostics)),
        "f0_at_band_edge_fraction": float(np.mean([row["f0_at_band_edge"] for row in diagnostics])),
        "median_harmonic_to_noise_db": float(
            np.median([row["harmonic_to_noise_db"] for row in diagnostics])
        ),
        "formant_estimation_failed_fraction": float(
            np.mean([row["formant_estimation_failed"] for row in diagnostics])
        ),
        "median_voiced_fraction": float(np.median([row["voiced_fraction"] for row in diagnostics])),
        "median_notes": float(np.median([row["notes"] for row in diagnostics])),
        "median_energy_share_below_50_hz": float(
            np.median([row["energy_share_below_50_hz"] for row in diagnostics])
        ),
    }


def probe_targets(
    *, endpoint: Endpoint, diagnostics: Sequence[dict[str, Any]], split: str = "query"
) -> tuple[dict[str, Any], frozenset[str], dict[str, str]]:
    """Build the context targets for one split, and say which are categorical.

    Clip duration and level are computed from the audio rather than read from a
    manifest, so every endpoint has the same two, measured the same way.

    A target that carries nothing on this endpoint is dropped here rather than
    scored, and the reason is returned with it so a run record says which
    probes did not run and why. E01d pads every RookID clip to exactly three
    seconds, so its duration target is constant and is dropped for that reason.
    """

    import numpy as np

    rows = [row for row in diagnostics if row["split"] == split]
    by_name = {record.filename: record for record in endpoint.records}
    targets: dict[str, Any] = {
        "log_duration_seconds": np.asarray(
            [math.log(row["duration_seconds"]) for row in rows], dtype=np.float64
        ),
        "rms_dbfs": np.asarray([row["rms_dbfs"] for row in rows], dtype=np.float64),
    }
    context_names = sorted(
        {
            name
            for record in endpoint.records
            for name in record.context
            if name in {"calendar_date_ordinal", *endpoint.categorical_targets}
        }
    )
    for name in context_names:
        values = [by_name[row["filename"]].context[name] for row in rows]
        targets[name] = np.asarray(
            values, dtype=object if name in endpoint.categorical_targets else np.float64
        )
    categorical = frozenset(endpoint.categorical_targets)
    not_defined: dict[str, str] = {}
    for name in sorted(targets):
        values = targets[name]
        if name in categorical:
            if len(set(values.tolist())) < 2:
                not_defined[name] = "takes one value on this endpoint"
        else:
            numeric = np.asarray(values, dtype=np.float64)
            if float(numeric.std()) <= CONSTANT_TARGET_TOLERANCE * max(
                1.0, float(np.abs(numeric.mean()))
            ):
                not_defined[name] = "constant on this endpoint"
    for name in not_defined:
        del targets[name]
    return targets, categorical, not_defined


def evaluate_layer(
    *,
    endpoint: Endpoint,
    layer: str,
    vectors: Any,
    diagnostics: Sequence[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Run one layer through the E01d head and the context probes."""

    import numpy as np

    records = [record.as_evaluation_record() for record in endpoint.records]
    result = evaluate_endpoint(
        records=records,
        vectors=vectors,
        representation=layer,
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=RIDGE_LAMBDA,
        seed=seed,
        permutations=PERMUTATIONS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        standardisation=PER_DIMENSION,
    )
    query_rows = [index for index, record in enumerate(endpoint.records) if record.split == "query"]
    targets, categorical, not_defined = probe_targets(endpoint=endpoint, diagnostics=diagnostics)
    result["context_probes"] = leave_one_identity_out(
        features=np.asarray(vectors)[query_rows],
        identities=np.asarray([endpoint.records[index].identity for index in query_rows]),
        targets=targets,
        categorical=categorical,
        standardisation=PER_DIMENSION,
        ridge_lambda=RIDGE_LAMBDA,
    )
    result["context_probes_not_defined"] = not_defined
    return result


def run_a2_endpoint(*, endpoint: Endpoint) -> dict[str, Any]:
    """Evaluate all four layers on one endpoint and return the measurement."""

    matrices, diagnostics, names = extract_endpoint_layers(endpoint.records)
    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "clips": len(endpoint.records),
        "identities": len(endpoint.identities),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "seed": SEED,
        "ridge_lambda": RIDGE_LAMBDA,
        "permutations": PERMUTATIONS,
        "f0_band_hz": [DEFAULT_F0_MIN, DEFAULT_F0_MAX],
        "estimability": estimability(diagnostics),
        "dimensions": names,
        "layers": {},
    }
    for index, layer in enumerate(LAYERS):
        result = evaluate_layer(
            endpoint=endpoint,
            layer=layer,
            vectors=matrices[layer],
            diagnostics=diagnostics,
            seed=SEED + index * 1000,
        )
        summary["layers"][layer] = result
    return summary
