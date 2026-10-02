"""A5: which published audio model carries individual identity for a bird?

BirdNET is trained to tell species apart, and that objective actively discards
what separates two birds of one species. Three speaker-verification encoders are
trained on exactly the task this project needs, told on human voices. Thirteen
self-supervised speech encoders learn speech without labels, differing in
objective (contrastive, unit prediction, latent regression, speaker contrast),
in architecture, in pretraining data from English read speech to a thousand
languages, and in size. This module runs every one of them on the same clips,
through the same head, and ranks them against the BirdNET numbers already
recorded.

Nothing here has to reach a number. The output is the comparison.

The rules are registered in
[`docs/measurement-protocol.md`](../../docs/measurement-protocol.md). Two of them
decide what this module may report:

* three playback rates, because these models stop at 8 kHz and bird calls do
  not, and slowing a call moves it into the band the model was trained on;
* the layer and rate reported as a model's result are chosen by a Fisher ratio
  computed on the enrollment clips alone. The full curve is reported beside it,
  because taking the best of hundreds of accuracies would be a threshold
  measured on the sample that produced it.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord, Endpoint
from xinyenyana.evaluation import (
    evaluate_endpoint,
    leave_one_identity_out,
    standardisation_for,
)
from xinyenyana.features import read_clip

MODEL_SAMPLE_RATE = 16_000
SLOWDOWNS: tuple[int, ...] = (1, 2, 3)
BIRDNET = "birdnet-v2.4"
DURATION_ONLY = "clip-duration-only"
LEVEL_ONLY = "clip-level-only"
SHORTCUTS: tuple[str, ...] = (DURATION_ONLY, LEVEL_ONLY)
#: Every architecture whose layers are reachable through ``AutoModel`` with
#: ``output_hidden_states=True``. Each was loaded on this machine and made to
#: embed a clip before it was listed here; nothing is listed on the strength of
#: its model card. They differ in pretraining objective (contrastive for
#: wav2vec2, masked-unit prediction for HuBERT, denoising plus speaker mixing
#: for WavLM, utterance-level speaker contrast for UniSpeech-SAT), in
#: pretraining data (English read speech, multilingual speech for XLS-R), and
#: in size. The point of the list is that difference: which of them, if any,
#: carries individual identity for a bird.
HIDDEN_STATE_MODELS: tuple[str, ...] = (
    "facebook/wav2vec2-base",
    "facebook/wav2vec2-large-robust",
    "facebook/wav2vec2-conformer-rope-large",
    "facebook/wav2vec2-xls-r-300m",
    "facebook/mms-300m",
    "facebook/hubert-base-ls960",
    "facebook/hubert-large-ll60k",
    "facebook/data2vec-audio-base-100h",
    "facebook/data2vec-audio-base-960h",
    "microsoft/wavlm-base-plus",
    "microsoft/wavlm-large",
    "microsoft/wavlm-base-plus-sv",
    "microsoft/unispeech-sat-base-plus",
)

#: Speaker-verification encoders published through SpeechBrain. These are the
#: closest published objective to what this project needs: tell two individuals
#: of one species apart. Each returns one vector per clip, not a stack of
#: layers.
SPEAKER_MODELS: tuple[str, ...] = (
    "speechbrain/spkrec-ecapa-voxceleb",
    "speechbrain/spkrec-xvect-voxceleb",
    "speechbrain/spkrec-resnet-voxceleb",
)

#: Everything compared, in the order the run walks them. BirdNET first because
#: it is what the rest are measured against, then the two shortcut controls that
#: any real result has to beat, then the pretrained encoders.
MODELS: tuple[str, ...] = (BIRDNET, *SHORTCUTS, *SPEAKER_MODELS, *HIDDEN_STATE_MODELS)

RIDGE_LAMBDA = 1.0
PERMUTATIONS = 999
BOOTSTRAP_REPLICATES = 2000
SEED = 17


@dataclass(frozen=True)
class Representation:
    """One model, one playback rate, and one layer of it."""

    model: str
    slowdown: int
    layer: int | None

    @property
    def name(self) -> str:
        layer = "" if self.layer is None else f"-l{self.layer:02d}"
        return f"{self.model.split('/')[-1]}-x{self.slowdown}{layer}"


def resample_for_model(signal: Any, sample_rate: int, slowdown: int) -> Any:
    """Bring a clip to 16 kHz, optionally slowed by an integer factor.

    Slowing is done by claiming the clip was recorded at a lower rate, which
    divides every frequency in it by the factor and multiplies its duration.

    Resampling is polyphase with an anti-aliasing low-pass filter. Until
    2026-09-22 it was linear interpolation, on the reasoning that everything
    above 8 kHz is discarded by the model regardless. That reasoning was wrong:
    interpolation does not discard energy above the new Nyquist frequency, it
    folds it back into the band, so a 12 kHz component in a 48 kHz recording
    arrived at the model as 4 kHz. Every speech-encoder figure measured before
    that date carries the fold-back; BirdNET does not, because its own reader
    resamples by the Fourier method.
    """

    import numpy as np

    resampled = _resample_unscaled(signal, sample_rate, slowdown)
    peak = float(np.abs(resampled).max()) if len(resampled) else 0.0
    return (resampled / peak if peak > 0 else resampled).astype(np.float32)


def _resample_unscaled(signal: Any, sample_rate: int, slowdown: int) -> Any:
    """The anti-aliased 16 kHz signal before peak scaling, for testing the filter."""

    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    samples = np.asarray(signal, dtype=np.float64)
    # Read at sample_rate / slowdown, written at MODEL_SAMPLE_RATE: the ratio is
    # MODEL_SAMPLE_RATE * slowdown / sample_rate, which is rational for every
    # integer rate a WAV header can carry.
    up = MODEL_SAMPLE_RATE * slowdown
    down = int(sample_rate)
    common = gcd(up, down)
    return resample_poly(samples, up // common, down // common)


def _load_batch(records: Sequence[ClipRecord], slowdown: int) -> Iterator[Any]:
    """One clip at a time, read and brought to the model's rate.

    This yields rather than returning a list because the caller encodes each
    clip and keeps only the encoding. Holding the whole corpus as audio costs
    ``total seconds x 16000 x slowdown x 4`` bytes, which for the 12,349-clip
    chiffchaff within-year endpoint is 12.6 GB at the slowest rate, enough to
    have the run killed for memory on a shared machine. Yielding makes the
    audio cost one clip. Every sample handed to the model is identical either
    way: `resample_for_model` reads one clip and nothing else.
    """

    for record in records:
        signal, sample_rate = read_clip(record.path)
        yield resample_for_model(signal, sample_rate, slowdown)


def birdnet_vectors(records: Sequence[ClipRecord]) -> Any:
    """One 1024-dimensional BirdNET v2.4 embedding per clip.

    BirdNET is the representation every other one is compared against, so it is
    produced here, by the same function, from the same clips, rather than by a
    second path whose agreement would have to be argued.
    """

    import numpy as np

    from xinyenyana.representations import birdnet_v2_4_vectors

    vectors, _ = birdnet_v2_4_vectors([record.path for record in records])
    return np.asarray(vectors)


def shortcut_vectors(records: Sequence[ClipRecord], which: str) -> Any:
    """The controls a representation has to beat before its result means anything.

    Both are measured on the **call**, not on the clip that holds it: the span
    from the first non-zero sample to the last. E01d pads every clip to a fixed
    three seconds, so a duration measured on the clip is a constant and the
    control would sit at chance for a reason that has nothing to do with the
    bird, and a level measured on the clip would be diluted by the padding.

    Duration is one number. Level is three: RMS in decibels, peak in decibels,
    and the ratio of peak to RMS.

    A method that scores well on either has found how long the bird called for,
    or how close it was to the microphone, not which bird it was.
    """

    import wave
    from array import array

    import numpy as np

    values: list[list[float]] = []
    for record in records:
        # Read the raw samples, not read_clip's. That subtracts the mean, which
        # turns a clip's zero padding into a small non-zero value and hides
        # exactly the boundary this function is looking for.
        with wave.open(str(record.path), "rb") as stream:
            sample_rate = stream.getframerate()
            raw = array("h")
            raw.frombytes(stream.readframes(stream.getnframes()))
        samples = np.asarray(raw, dtype=np.float64) / 32768.0
        sounding = np.flatnonzero(samples)
        window = samples[sounding[0] : sounding[-1] + 1] if len(sounding) else samples
        if which == DURATION_ONLY:
            values.append([len(window) / sample_rate])
            continue
        rms = float(np.sqrt(np.mean(np.square(window))))
        peak = float(np.max(np.abs(window)))
        floor = 1e-12
        values.append(
            [
                20 * math.log10(max(rms, floor)),
                20 * math.log10(max(peak, floor)),
                peak / max(rms, floor),
            ]
        )
    return np.asarray(values, dtype=np.float64)


#: xen1's graphics card is shared with other work on the same machine, so a
#: forward pass can fail for memory another process holds rather than for
#: anything about the clip. Waiting and asking again for the same clip returns
#: the same vector, so the pass is retried before the run is lost. Nothing about
#: what is measured changes; only when it is computed.
#: How long a pass waits in total is an operating choice, not a property of the
#: measurement, so it is settable per run. The default parks a sweep for six
#: minutes, which covers a short burst of other work; a run started while the
#: card is known to be busy for longer is given a longer wait.
SHARED_DEVICE_ATTEMPTS = int(os.environ.get("XINYENYANA_DEVICE_ATTEMPTS", "12"))
SHARED_DEVICE_WAIT_SECONDS = float(os.environ.get("XINYENYANA_DEVICE_WAIT_SECONDS", "30"))


def on_device(compute: Any, *, what: str, seconds: float) -> Any:
    """Run one forward pass, waiting out memory another process is holding.

    ``compute`` takes no arguments and returns whatever the pass returns.
    ``seconds`` is the length the model is asked to encode, after the slowdown,
    at 16 kHz.

    If the card is still full after every attempt, the error states the clip,
    that length, and the free memory at the last attempt. It does not name a
    cause. The first version of this message said another process was holding
    the rest, and five of the six failures on record were the clip exceeding
    what the model fits on this card at any moment, which no amount of waiting
    changes. The two are told apart by the length, not by the free memory.
    """

    import time

    import torch

    for attempt in range(1, SHARED_DEVICE_ATTEMPTS + 1):
        try:
            return compute()
        except torch.cuda.OutOfMemoryError:
            pass
        # The cache is emptied only once the except block has ended. Inside it
        # the traceback still holds the failed pass's frames, and with them its
        # activations, so the memory could not be released: on 2026-09-23 the
        # retries of wavlm-large on a 163-second tree pipit clip each found
        # 3.76 GiB free of 15.48 because this process was holding the rest.
        torch.cuda.empty_cache()
        if attempt == SHARED_DEVICE_ATTEMPTS:
            free, total = torch.cuda.mem_get_info()
            raise RuntimeError(
                f"{what} did not fit after {SHARED_DEVICE_ATTEMPTS} attempts over "
                f"{SHARED_DEVICE_ATTEMPTS * SHARED_DEVICE_WAIT_SECONDS / 60:.0f} minutes; "
                f"{seconds:.1f} seconds at 16 kHz, "
                f"{free / 2**30:.2f} GiB of {total / 2**30:.2f} GiB free on the card "
                "at the last attempt"
            )
        time.sleep(SHARED_DEVICE_WAIT_SECONDS)
    raise AssertionError("unreachable")


def minimum_samples(forward: Any, *, ceiling: int = 16_000) -> int:
    """The shortest input, in samples, that a model's forward pass accepts.

    A clip shorter than a model's convolutional receptive field makes the
    pass fail rather than return a vector: on the group of eight zebra finches
    fifteen of the sixteen encoders failed at normal speed for that reason in
    v5. The shortest accepted length is a property of the architecture, found
    here by bisection on silence, so it is the same on every run and does not
    depend on any clip. ``forward`` takes a 1-D float32 array.
    """

    import numpy as np

    def accepts(length: int) -> bool:
        try:
            forward(np.zeros(length, dtype=np.float32))
        except RuntimeError:
            return False
        return True

    low, high = 1, ceiling
    if not accepts(high):
        raise RuntimeError(f"no input up to {ceiling} samples is accepted")
    while low < high:
        middle = (low + high) // 2
        if accepts(middle):
            high = middle
        else:
            low = middle + 1
    return low


def pad_to(clip: Any, length: int) -> Any:
    """A clip shorter than ``length`` samples, extended with silence to it."""

    import numpy as np

    if len(clip) >= length:
        return clip
    return np.pad(clip, (0, length - len(clip))).astype(np.float32)


def windows_of(clip: Any, samples: int | None) -> list[Any]:
    """Consecutive windows of at most ``samples``; the whole clip when ``samples`` is None."""

    if samples is None or len(clip) <= samples:
        return [clip]
    return [clip[start : start + samples] for start in range(0, len(clip), samples)]


def speaker_vectors(
    source: str,
    records: Sequence[ClipRecord],
    *,
    slowdown: int,
    device: str = "cuda",
    window_seconds: float | None = None,
) -> Any:
    """One speaker embedding per clip, from a published verification encoder.

    A clip shorter than the encoder accepts is padded with silence to that
    length (PA-V6). With ``window_seconds`` a longer clip is embedded in windows
    and the embeddings are averaged, weighted by window length; this is used
    only where the whole clip did not fit on the card, and is recorded.
    """

    import numpy as np
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    encoder = EncoderClassifier.from_hparams(
        source=source,
        savedir=f"/mnt/data/xinyenyana/models/{source.split('/')[-1]}",
        run_opts={"device": device},
    )

    def embed(samples: Any) -> Any:
        tensor = torch.from_numpy(np.ascontiguousarray(samples))[None, :].to(device)
        with torch.no_grad():
            return encoder.encode_batch(tensor).squeeze().detach().cpu().numpy()

    shortest = minimum_samples(embed)
    window = None if window_seconds is None else int(window_seconds * MODEL_SAMPLE_RATE)
    vectors = []
    for record, clip in zip(records, _load_batch(records, slowdown), strict=True):

        def encode(clip: Any = clip) -> Any:
            parts = [pad_to(part, shortest) for part in windows_of(clip, window)]
            if len(parts) == 1:
                # Exactly the v5 computation whenever the clip is read whole.
                return embed(parts[0])
            weights = np.asarray([len(part) for part in parts], dtype=np.float64)
            embedded = np.vstack([embed(part) for part in parts])
            return (embedded * weights[:, None]).sum(axis=0) / weights.sum()

        vectors.append(
            on_device(
                encode,
                what=f"{source} on {record.filename}",
                seconds=len(clip) / MODEL_SAMPLE_RATE,
            )
        )
    return np.vstack(vectors)


def speech_layer_vectors(
    model_name: str,
    records: Sequence[ClipRecord],
    *,
    slowdown: int,
    device: str = "cuda",
    window_seconds: float | None = None,
) -> dict[int, Any]:
    """Mean-pooled vectors for every transformer layer, in one pass per clip.

    A clip shorter than the model accepts is padded with silence to that length
    (PA-V6). With ``window_seconds`` a longer clip is read in windows and each
    layer is averaged over every frame of every window; used only where the
    whole clip did not fit on the card, and recorded.
    """

    import numpy as np
    import torch
    from transformers import AutoModel

    model = (
        AutoModel.from_pretrained(model_name, cache_dir="/mnt/data/xinyenyana/models/hf")
        .to(device)
        .eval()
    )

    def hidden_of(samples: Any, encoder: Any = model) -> Any:
        tensor = torch.from_numpy(np.ascontiguousarray(samples))[None, :].to(device)
        with torch.no_grad():
            return encoder(tensor, output_hidden_states=True).hidden_states

    shortest = minimum_samples(hidden_of)
    window = None if window_seconds is None else int(window_seconds * MODEL_SAMPLE_RATE)
    collected: dict[int, list[Any]] = {}
    for record, clip in zip(records, _load_batch(records, slowdown), strict=True):

        def pooled(clip: Any = clip) -> list[Any]:
            parts = windows_of(clip, window)
            if len(parts) == 1:
                # Exactly the v5 computation whenever the clip is read whole.
                hidden = hidden_of(pad_to(parts[0], shortest))
                return [layer.squeeze(0).mean(dim=0).detach().cpu().numpy() for layer in hidden]
            sums: list[Any] = []
            frames = 0
            for part in parts:
                hidden = hidden_of(pad_to(part, shortest))
                layers = [layer.squeeze(0).double().sum(dim=0).cpu().numpy() for layer in hidden]
                frames += int(hidden[0].shape[1])
                sums = layers if not sums else [a + b for a, b in zip(sums, layers, strict=True)]
            return [(total / frames).astype(np.float32) for total in sums]

        encoded = on_device(
            pooled,
            what=f"{model_name} on {record.filename}",
            seconds=len(clip) / MODEL_SAMPLE_RATE,
        )
        for index, layer in enumerate(encoded):
            collected.setdefault(index, []).append(layer)
    del model
    torch.cuda.empty_cache()
    return {index: np.vstack(rows) for index, rows in collected.items()}


def representations_of(
    model: str,
    records: Sequence[ClipRecord],
    *,
    device: str = "cuda",
    not_computed: list[dict[str, Any]] | None = None,
    slowdowns: Sequence[int] | None = None,
    window_seconds: float | None = None,
) -> Iterator[tuple[Representation, Any]]:
    """Every rate and layer this model offers, as vectors over the same clips.

    One place decides what a model is asked for, so adding an architecture is
    adding it to a table above rather than adding a branch to the run loop. A
    model that reaches here without a loader raises, rather than being silently
    skipped and leaving a gap nobody sees in the ranking.

    A rate whose longest clip does not fit on the machine is recorded in
    ``not_computed`` and skipped, rather than ending the run. The remaining
    rates for that model are still measured and the reason is carried into the
    result, so a gap in the curve says what caused it.
    """

    if model == BIRDNET:
        # BirdNET reads the clip at its own rate; slowing it is not defined for
        # a model trained on bird calls at their own speed.
        yield Representation(model=model, slowdown=1, layer=None), birdnet_vectors(records)
        return
    if model in SHORTCUTS:
        yield Representation(model=model, slowdown=1, layer=None), shortcut_vectors(records, model)
        return
    if model not in SPEAKER_MODELS and model not in HIDDEN_STATE_MODELS:
        raise ValueError(f"no loader for {model!r}")
    for slowdown in SLOWDOWNS if slowdowns is None else slowdowns:
        try:
            if model in SPEAKER_MODELS:
                produced = [
                    (
                        Representation(model=model, slowdown=slowdown, layer=None),
                        speaker_vectors(
                            model,
                            records,
                            slowdown=slowdown,
                            device=device,
                            window_seconds=window_seconds,
                        ),
                    )
                ]
            else:
                produced = [
                    (Representation(model=model, slowdown=slowdown, layer=index), vectors)
                    for index, vectors in speech_layer_vectors(
                        model,
                        records,
                        slowdown=slowdown,
                        device=device,
                        window_seconds=window_seconds,
                    ).items()
                ]
        except RuntimeError as error:
            if not_computed is None:
                raise
            not_computed.append({"model": model, "slowdown": slowdown, "reason": str(error)})
            continue
        yield from produced


def fisher_ratio(vectors: Any, identities: Sequence[str]) -> float:
    """Between-identity variance over within-identity variance, averaged.

    Computed on enrollment vectors only. It says how far apart the identities
    sit relative to how much one identity varies, without looking at a single
    query clip, which is what makes it usable to choose a layer.
    """

    import numpy as np

    values = np.asarray(vectors, dtype=np.float64)
    labels = np.asarray(identities)
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        raise ValueError("a Fisher ratio needs at least two identities")
    means = np.vstack([values[labels == identity].mean(axis=0) for identity in unique])
    within = np.mean(
        [values[labels == identity].var(axis=0, ddof=0) for identity in unique], axis=0
    )
    between = means.var(axis=0, ddof=0)
    usable = within > 1e-12
    if not usable.any():
        return 0.0
    return float(np.mean(between[usable] / within[usable]))


def evaluate_representation(
    *,
    endpoint: Endpoint,
    vectors: Any,
    name: str,
    seed: int,
    diagnostics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """One representation through the identical head and context probes.

    ``diagnostics`` is the per-clip measurement the context targets are built
    from. It depends on the clips and not on the representation, so the caller
    computes it once: taken here it would re-read and re-analyse every clip for
    every representation in the run, which is most of it.
    """

    import numpy as np

    from xinyenyana.a2 import probe_targets

    rule = standardisation_for(int(np.asarray(vectors).shape[1]))
    result = evaluate_endpoint(
        records=[record.as_evaluation_record() for record in endpoint.records],
        vectors=vectors,
        representation=name,
        enrollment_condition="foreground",
        query_condition="foreground",
        ridge_lambda=RIDGE_LAMBDA,
        seed=seed,
        permutations=PERMUTATIONS,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        standardisation=rule,
        extended_nulls=False,
    )
    # Which query clips this candidate named correctly, in query order, so a
    # paired comparison between two representations can resample the same
    # animals for both without re-extracting anything.
    result["query_correct"] = "".join(
        "1" if entry["actual_label"] == entry["predicted_label"] else "0"
        for entry in result["predictions"]
    )
    query_rows = [i for i, record in enumerate(endpoint.records) if record.split == "query"]
    targets, categorical, not_defined = probe_targets(endpoint=endpoint, diagnostics=diagnostics)
    result["context_probes"] = leave_one_identity_out(
        features=np.asarray(vectors)[query_rows],
        identities=np.asarray([endpoint.records[i].identity for i in query_rows]),
        targets=targets,
        categorical=categorical,
        standardisation=rule,
        ridge_lambda=RIDGE_LAMBDA,
    )
    result["context_probes_not_defined"] = not_defined
    # One field for both, because one rule was used for both. When they were
    # chosen separately the probe read features the head never saw, and it
    # reported a floor score on a target the features were.
    result["standardisation"] = rule
    result.pop("predictions", None)
    return result


def run_a5_endpoint(
    *, endpoint: Endpoint, device: str = "cuda", checkpoint_dir: Path | None = None
) -> dict[str, Any]:
    """Every model, every rate, every layer, and the enrollment-chosen result.

    With ``checkpoint_dir`` each finished model's curve is written there, keyed
    by the endpoint's manifest and split digests and by the digest of this
    package's source, and a later run with the same three digests reads it back
    instead of recomputing it. A lease displaced by higher-priority work then
    loses only the model it was on. A checkpoint from different code or
    different data is ignored, so a resumed result is exactly what one
    uninterrupted run of the same code would have written.
    """

    import json

    import numpy as np

    from xinyenyana.archive import canonical_sha256, source_digest

    fingerprint = canonical_sha256(
        {
            "manifest": endpoint.manifest_sha256,
            "splits": endpoint.split_digests(),
            "source": source_digest(Path(__file__).resolve().parent),
        }
    )

    from xinyenyana.a2 import clip_diagnostics

    diagnostics = clip_diagnostics(endpoint.records)
    enrollment = [i for i, record in enumerate(endpoint.records) if record.split == "enrollment"]
    enrollment_identities = [endpoint.records[i].identity for i in enrollment]
    summary: dict[str, Any] = {
        "endpoint": endpoint.name,
        "chance_accuracy": 1.0 / len(endpoint.identities),
        "clips": len(endpoint.records),
        "identities": len(endpoint.identities),
        "manifest_sha256": endpoint.manifest_sha256,
        "splits": endpoint.split_digests(),
        "seed": SEED,
        "device": device,
        "slowdowns": list(SLOWDOWNS),
        "models": list(MODELS),
        "curve": {},
        "chosen": {},
        # The identity of every query clip, in the order each candidate's
        # `query_correct` string is written in.
        "query_identities": [
            record.identity
            for record in endpoint.records
            if record.split == "query"
            and str(record.context.get("condition", "foreground")) == "foreground"
        ],
        "resampler": "polyphase with anti-aliasing filter (scipy.signal.resample_poly)",
    }
    seed = SEED
    not_computed: list[dict[str, Any]] = []
    summary["not_computed"] = not_computed
    for model in MODELS:
        stored = (
            checkpoint_dir / f"{fingerprint[:16]}-{model.replace('/', '_')}.json"
            if checkpoint_dir is not None
            else None
        )
        if stored is not None and stored.exists():
            saved = json.loads(stored.read_text())
            if saved.get("fingerprint") == fingerprint:
                summary["curve"][model] = saved["results"]
                not_computed.extend(saved["not_computed"])
                seed = int(saved["seed_after"])
                if saved["chosen"] is not None:
                    summary["chosen"][model] = saved["chosen"]
                continue
        dropped_before = len(not_computed)
        candidates: list[tuple[float, Representation, Any]] = []
        for representation, vectors in representations_of(
            model, endpoint.records, device=device, not_computed=not_computed
        ):
            ratio = fisher_ratio(np.asarray(vectors)[enrollment], enrollment_identities)
            candidates.append((ratio, representation, vectors))
        results: dict[str, Any] = {}
        for ratio, representation, vectors in candidates:
            seed += 1
            evaluated = evaluate_representation(
                endpoint=endpoint,
                vectors=vectors,
                name=representation.name,
                seed=seed,
                diagnostics=diagnostics,
            )
            results[representation.name] = {
                "slowdown": representation.slowdown,
                "layer": representation.layer,
                "enrollment_fisher_ratio": ratio,
                "accuracy": evaluated["classification"]["accuracy"],
                "identity_block_bootstrap_accuracy_95": evaluated[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation_p": evaluated["permutation_control"]["p_value_plus_one"],
                "roc_auc": evaluated["verification"]["roc_auc"],
                "context_probes": evaluated["context_probes"],
                "standardisation": evaluated["standardisation"],
                "query_correct": evaluated["query_correct"],
            }
        summary["curve"][model] = results
        if not candidates:
            # Every rate this model offers failed for want of memory. Its curve
            # is recorded empty and it is left out of the ranking; the reasons
            # are in `not_computed`, so the absence is visible rather than a gap
            # nobody sees.
            continue
        best = max(candidates, key=lambda entry: entry[0])[1]
        summary["chosen"][model] = {"representation": best.name, **results[best.name]}
        if stored is not None:
            stored.parent.mkdir(parents=True, exist_ok=True)
            stored.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "results": results,
                        "chosen": summary["chosen"][model],
                        "not_computed": not_computed[dropped_before:],
                        "seed_after": seed,
                    }
                )
            )
    summary["ranking"] = sorted(
        (
            {
                "model": model,
                "representation": chosen["representation"],
                "accuracy": chosen["accuracy"],
                "identity_block_bootstrap_accuracy_95": chosen[
                    "identity_block_bootstrap_accuracy_95"
                ],
                "permutation_p": chosen["permutation_p"],
                "roc_auc": chosen["roc_auc"],
            }
            for model, chosen in summary["chosen"].items()
        ),
        key=lambda entry: (-float(entry["accuracy"]), str(entry["model"])),
    )
    return summary


#: The short name each model is recorded under in a sweep result, back to the
#: identifier its loader needs. Built from ``MODELS`` so that a model added to
#: the tables above is reachable by name without a second list to maintain.
_MODEL_BY_SHORT_NAME: dict[str, str] = {model.split("/")[-1]: model for model in MODELS}


def representation_from_name(name: str) -> Representation:
    """One representation from the name a sweep recorded it under.

    A sweep writes ``wavlm-large-x3-l07`` or ``birdnet-v2.4-x1``; this reads
    that back. It exists so that a later experiment can be pointed at the
    representation an earlier one chose, by the name in the archived result,
    rather than by a second spelling that has to be kept in step with it.

    ``birdnet-v2.4``, ``clip-duration-only`` and ``clip-level-only`` are also
    accepted without the rate suffix, because that is how every compensation
    result already on record names them.
    """

    if name in (BIRDNET, *SHORTCUTS):
        return Representation(model=name, slowdown=1, layer=None)

    stem = name
    layer: int | None = None
    head, separator, tail = stem.rpartition("-l")
    if separator and tail.isdigit():
        stem, layer = head, int(tail)
    head, separator, tail = stem.rpartition("-x")
    if not separator or not tail.isdigit():
        raise ValueError(f"{name!r} carries no playback rate; expected a -x<rate> suffix")
    stem, slowdown = head, int(tail)

    if stem in (BIRDNET, *SHORTCUTS):
        if layer is not None:
            raise ValueError(f"{stem} has no layers to choose between: {name!r}")
        if slowdown != 1:
            raise ValueError(f"{stem} is read at the recording's own rate: {name!r}")
        return Representation(model=stem, slowdown=1, layer=None)

    model = _MODEL_BY_SHORT_NAME.get(stem)
    if model is None:
        raise ValueError(f"no model in this project is named {stem!r}: {name!r}")
    if slowdown not in SLOWDOWNS:
        raise ValueError(f"{slowdown} is not one of the registered rates {SLOWDOWNS}: {name!r}")
    if model in SPEAKER_MODELS and layer is not None:
        raise ValueError(f"{model} returns one vector per clip, not a stack of layers: {name!r}")
    if model in HIDDEN_STATE_MODELS and layer is None:
        raise ValueError(f"{model} needs a layer; expected a -l<layer> suffix: {name!r}")
    return Representation(model=model, slowdown=slowdown, layer=layer)


def vectors_for(
    representation: Representation,
    records: Sequence[ClipRecord],
    *,
    device: str = "cuda",
    window_seconds: float | None = None,
) -> Any:
    """The vectors of one named representation over these clips.

    ``representations_of`` walks every rate and layer a model offers, which is
    what the sweep needs. This returns one of them, which is what an experiment
    downstream of the sweep needs, through the same loaders, so the two cannot
    drift apart.
    """

    model = representation.model
    # Windows only where the representation was measured in windows (PA-V6).
    windowed: dict[str, Any] = {} if window_seconds is None else {"window_seconds": window_seconds}
    if model == BIRDNET:
        return birdnet_vectors(records)
    if model in SHORTCUTS:
        return shortcut_vectors(records, model)
    if model in SPEAKER_MODELS:
        return speaker_vectors(
            model, records, slowdown=representation.slowdown, device=device, **windowed
        )
    if model in HIDDEN_STATE_MODELS:
        layers = speech_layer_vectors(
            model, records, slowdown=representation.slowdown, device=device, **windowed
        )
        if representation.layer not in layers:
            raise ValueError(
                f"{model} returned layers {sorted(layers)}, not {representation.layer}"
            )
        return layers[representation.layer]
    raise ValueError(f"no loader for {model!r}")
