"""XEUS, a multilingual speech encoder, read like the other speech encoders.

XEUS (espnet/xeus on Hugging Face; E-Branchformer, 577M parameters, trained
on over a million hours of speech in over 4,000 languages, CC BY-NC-SA 4.0)
is added because Cauzinille et al. (2025, arXiv 2509.04166) used it for
animal sounds. It is read exactly as A5 reads a speech encoder: 16 kHz, three
playback rates, every layer mean-pooled over frames, the layer and rate chosen
by the enrolment Fisher ratio. Two things differ and are registered:

* it runs in its own Python environment on xen1, with the ESPnet release
  (202610) whose model code matches the published configuration;
* every clip longer than 60 seconds at the model rate is read in 60-second
  windows, each layer averaged over every frame of every window, at every
  rate. A model of this size does not hold a three-minute clip on the card,
  and applying the rule from the start keeps one rule for every clip.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord

MODEL = "espnet/xeus"
SHORT = "xeus"
CHECKPOINT = Path("/mnt/data/xinyenyana/models/xeus/model/xeus_checkpoint_new.pth")
WINDOW_SECONDS = 60.0


def _load(device: str) -> Any:
    from espnet2.tasks.ssl import SSLTask

    model, _ = SSLTask.build_model_from_file(None, str(CHECKPOINT), device)
    return model.eval()


def layer_vectors(
    records: Sequence[ClipRecord], *, slowdown: int, device: str, model: Any
) -> dict[int, Any]:
    """Every layer's frame-mean for every clip at one playback rate."""

    import numpy as np
    import torch

    from xinyenyana.a5 import (
        MODEL_SAMPLE_RATE,
        _load_batch,
        minimum_samples,
        on_device,
        pad_to,
        windows_of,
    )

    def hidden_of(samples: Any) -> Any:
        wav = torch.from_numpy(np.ascontiguousarray(samples))[None, :].to(device)
        lengths = torch.LongTensor([wav.shape[1]]).to(device)
        with torch.no_grad():
            # ESPnet 202610: (final output, every block's output, lengths). The
            # blocks' outputs are read, as A5 reads every layer of a speech model.
            return model.inference_encode(wav, lengths, use_mask=False, use_final_output=False)[1]

    shortest = minimum_samples(hidden_of)
    window = int(WINDOW_SECONDS * MODEL_SAMPLE_RATE)
    collected: dict[int, list[Any]] = {}
    for record, clip in zip(records, _load_batch(records, slowdown), strict=True):

        def pooled(clip: Any = clip) -> list[Any]:
            sums: list[Any] = []
            frames = 0
            for part in windows_of(clip, window):
                hidden = hidden_of(pad_to(part, shortest))
                layers = [layer.squeeze(0).double().sum(dim=0).cpu().numpy() for layer in hidden]
                frames += int(hidden[0].shape[1])
                sums = layers if not sums else [a + b for a, b in zip(sums, layers, strict=True)]
            return [(total / frames).astype(np.float32) for total in sums]

        encoded = on_device(
            pooled, what=f"{MODEL} on {record.filename}", seconds=len(clip) / MODEL_SAMPLE_RATE
        )
        for index, layer in enumerate(encoded):
            collected.setdefault(index, []).append(layer)
    return {index: np.vstack(rows) for index, rows in collected.items()}


def candidate_vectors(
    records: Sequence[ClipRecord],
    *,
    device: str,
    not_computed: list[dict[str, Any]],
    fixed: str | None,
) -> Iterator[tuple[str, dict[str, Any], Any]]:
    """(name, description, vectors) for every rate and layer, or the one fixed."""

    import torch

    from xinyenyana.a5 import SLOWDOWNS

    wanted_rate = wanted_layer = None
    if fixed is not None:
        stem, _, layer = fixed.rpartition("-l")
        wanted_layer = int(layer)
        wanted_rate = int(stem.rpartition("-x")[2])
    model = _load(device)
    try:
        for slowdown in SLOWDOWNS:
            if wanted_rate is not None and slowdown != wanted_rate:
                continue
            try:
                layers = layer_vectors(records, slowdown=slowdown, device=device, model=model)
            except RuntimeError as error:
                not_computed.append({"model": MODEL, "slowdown": slowdown, "reason": str(error)})
                continue
            for index, vectors in layers.items():
                if wanted_layer is not None and index != wanted_layer:
                    continue
                yield (
                    f"{SHORT}-x{slowdown}-l{index:02d}",
                    {
                        "slowdown": slowdown,
                        "layer": index,
                        "read_in_windows_of_seconds": WINDOW_SECONDS,
                    },
                    vectors,
                )
    finally:
        del model
        torch.cuda.empty_cache()
