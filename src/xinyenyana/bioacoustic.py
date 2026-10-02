"""The bioacoustic and general audio models of PA-V6, and how a clip is read by them.

Each model is loaded through the class Bacpipe 1.3.5 publishes for it (Kather
et al. 2026, Methods in Ecology and Evolution, doi 10.1111/2041-210x.70406),
with the preprocessing and weights that class uses, except the two AVEX
encoders, which Earth Species Project publishes through its ``avex`` library.
Bacpipe's own pipeline is not used: it pads a short clip by repeating it and
its probes divide clips at random. Only its model classes are.

**How a clip is read.** Brought to the model's rate by polyphase resampling
with an anti-aliasing filter; cut into consecutive windows of the model's
input length; the last window, and a clip shorter than one window, padded with
silence; every window passed through the model's preprocessing and network one
at a time; the clip's vector is the mean over its windows. One window at a time
because two of the published preprocessors normalise over whatever batch they
are given, which would make a clip's vector depend on its neighbours.

**Candidates.** ``embedding`` is the model's own output as its publishers pool
it; a token sequence, if that is what comes back, is averaged over its tokens.
For a transformer, ``block NN`` is the output of its NN-th repeated block,
found by :func:`transformer_blocks`, averaged over tokens. Sweeps retain ordinal
block names. :func:`describe` supplies named module paths for a separate load
audit; historical sweeps did not retain those paths in their result objects.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xinyenyana.a2 import ClipRecord

MODEL_ROOT = Path("/mnt/data/xinyenyana/models/bacpipe")
#: The AVEX weights, fetched once from EarthSpeciesProject on Hugging Face and
#: read from here, so a run never streams them from the Hub.
AVEX_ROOT = Path("/mnt/data/xinyenyana/models/avex")
EMBEDDING = "embedding"


@dataclass(frozen=True)
class Spec:
    """One model: where it comes from and whether its blocks are read."""

    source: str
    transformer: bool
    note: str


#: The nineteen models of PA-V6, by the short name results carry.
MODELS: dict[str, Spec] = {
    "perch-v2": Spec("bacpipe:perch_v2", False, "Perch 2.0, EfficientNet-B3, ONNX graph"),
    "perch-bird": Spec("bacpipe:perch_bird", False, "Perch v1 (bird), EfficientNet-B1, TF"),
    "birdnet-v3-preview": Spec(
        "bacpipe:birdnet_v3", False, "BirdNET v3 developer preview as Bacpipe distributes it, ONNX"
    ),
    "esp-aves2-sl-beats-all": Spec("avex:esp_aves2_sl_beats_all", True, "AVEX, BEATs"),
    "esp-aves2-effnetb0-all": Spec("avex:esp_aves2_effnetb0_all", False, "AVEX, EfficientNet-B0"),
    "surfperch": Spec("bacpipe:surfperch", False, "SurfPerch, EfficientNet-B0, TF"),
    "avesecho-passt": Spec("bacpipe:avesecho_passt", True, "AvesEcho, PaSST"),
    "audioprotopnet": Spec("bacpipe:audioprotopnet", False, "AudioProtoPNet, ConvNeXt"),
    "convnext-birdset": Spec("bacpipe:convnext_birdset", False, "ConvNeXt trained on BirdSet"),
    "birdmae": Spec("bacpipe:birdmae", True, "Bird-MAE-Huge, ViT"),
    "protoclr": Spec("bacpipe:protoclr", True, "ProtoCLR, CvT-13"),
    "rcl-fs-bsed": Spec("bacpipe:rcl_fs_bsed", False, "RCL few-shot SED, ResNet9"),
    "birdaves": Spec("bacpipe:birdaves_especies", True, "BirdAVES, HuBERT large"),
    "aves": Spec("bacpipe:aves_especies", True, "AVES bio, HuBERT base"),
    "naturebeats": Spec("bacpipe:naturebeats", True, "NatureLM-audio's BEATs encoder"),
    "biolingual": Spec("bacpipe:biolingual", True, "BioLingual, CLAP audio tower (HTS-AT)"),
    "beats": Spec("bacpipe:beats", True, "BEATs, AudioSet"),
    "audiomae": Spec("bacpipe:audiomae", True, "Audio-MAE, ViT, AudioSet"),
    "vggish": Spec("bacpipe:vggish", False, "VGGish, AudioSet, TF"),
}

TENSORFLOW = frozenset({"perch-bird", "surfperch", "vggish"})

#: Where, below a Bacpipe class's ``model`` attribute, the network whose
#: blocks are read sits, where it is not ``model`` itself: Bacpipe's BEATs
#: wraps the network in a plain object, and BioLingual's CLAP model carries a
#: text tower that an audio clip never passes through.
TORCH_ROOT: dict[str, str] = {"beats": "model", "biolingual": "audio_model"}


def layer_of(candidate: str) -> int | None:
    return None if candidate == EMBEDDING else int(candidate.split()[-1])


# --- reading a clip ----------------------------------------------------------


def resample(signal: Any, rate: int, target: int) -> Any:
    """Polyphase resampling with an anti-aliasing filter, to ``target`` Hz."""

    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    samples = np.asarray(signal, dtype=np.float64)
    if rate == target:
        return samples.astype(np.float32)
    common = gcd(int(target), int(rate))
    return resample_poly(samples, target // common, rate // common).astype(np.float32)


def windows(signal: Any, length: int) -> Any:
    """Consecutive windows of ``length`` samples, the last one padded with silence."""

    import numpy as np

    samples = np.asarray(signal, dtype=np.float32)
    count = max(1, int(np.ceil(len(samples) / length)))
    padded = np.zeros(count * length, dtype=np.float32)
    padded[: len(samples)] = samples
    return padded.reshape(count, length)


def pool_tokens(output: Any) -> Any:
    """One window's output as one vector.

    A (1, T, D) or (T, 1, D) token sequence is averaged over its tokens; a
    (1, C, H, W) feature map over its height and width.
    """

    import numpy as np

    value = output[0] if isinstance(output, tuple | list) else output
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        return array
    if array.ndim == 4 and array.shape[0] == 1:
        # A convolutional feature map (1, C, H, W), averaged over H and W as
        # the network's own average pooling does before its classifier.
        return array[0].mean(axis=(1, 2))
    if array.shape[0] == 1:
        array = array[0]
    elif array.ndim >= 3 and array.shape[1] == 1:
        array = array[:, 0]
    return array.reshape(-1, array.shape[-1]).mean(axis=0)


# --- finding the blocks ------------------------------------------------------

_NOT_BLOCKS = ("conv", "norm", "linear", "dropout", "embed", "pool", "activation", "gelu")


def transformer_blocks(module: Any) -> list[tuple[str, Any]]:
    """Every repeated block of a transformer, innermost level, in forward order.

    A block list is a ``ModuleList`` or ``Sequential`` of modules of one class
    whose name is not a convolution, normalisation, projection, dropout,
    embedding, pooling or activation, where that class appears at least twice
    in one such list somewhere in the network (timm's vision transformers, and
    so PaSST, keep their blocks in a ``Sequential``; the first stage of a CvT
    holds a single block). Where one block list sits inside another, only the
    inner one is read, so a Swin stage and the blocks inside it are not both
    counted.
    """

    import torch

    candidates: list[tuple[str, Any, str]] = []
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.ModuleList | torch.nn.Sequential) or len(child) < 1:
            continue
        kinds = {type(member).__name__ for member in child}
        if len(kinds) != 1:
            continue
        kind = next(iter(kinds))
        if any(word in kind.lower() for word in _NOT_BLOCKS):
            continue
        candidates.append((name, child, kind))
    # A list of one block counts when the same block class repeats elsewhere in
    # the network, as in the first stage of a CvT, which holds a single block.
    repeated = {kind for _, child, kind in candidates if len(child) >= 2}
    lists = [(name, child) for name, child, kind in candidates if kind in repeated]
    inner = [
        (name, child)
        for name, child in lists
        if not any(other != name and other.startswith(f"{name}.") for other, _ in lists)
    ]
    return [
        (f"{name}.{index}", block) for name, child in inner for index, block in enumerate(child)
    ]


# --- loading -----------------------------------------------------------------


def _evaluation_mode(module: Any) -> None:
    """Put a torch network in evaluation mode, so dropout and layer drop are off.

    Bacpipe's ``prepare_inference`` calls ``eval`` on the object it holds, which
    for BEATs is a plain wrapper, so the network inside is put in evaluation
    mode here as well. A TensorFlow model is left alone.
    """

    import torch

    if isinstance(module, torch.nn.Module):
        module.eval()
        if any(child.training for child in module.modules()):
            raise RuntimeError("a module stayed in training mode")


class Loaded:
    """A model ready to embed one window, with its blocks hooked."""

    def __init__(self, name: str, device: str) -> None:
        self.name = name
        self.spec = MODELS[name]
        self.device = device
        self.captured: list[Any] = []
        self.block_paths: list[str] = []
        kind, _, target = self.spec.source.partition(":")
        if kind == "bacpipe":
            self._load_bacpipe(target)
        elif kind == "avex":
            self._load_avex(target)
        else:
            raise ValueError(f"unknown source {self.spec.source}")
        if self.spec.transformer:
            self._hook()

    def _load_bacpipe(self, target: str) -> None:
        import bacpipe

        if target in bacpipe.NEEDS_CHECKPOINT:
            bacpipe.ensure_models_exist(model_base_path=MODEL_ROOT, model_names=[target])
        module = importlib.import_module(f"bacpipe.model_pipelines.feature_extractors.{target}")
        self.model = module.Model(
            model_name=target,
            device=self.device,
            model_base_path=MODEL_ROOT,
            global_batch_size=4,
            run_pretrained_classifier=False,
        )
        if hasattr(self.model, "prepare_inference"):
            self.model.prepare_inference()
        # The rate and window the class itself was built with; a subclass
        # such as BirdAVES does not repeat its parent's module constants.
        self.rate = int(self.model.sr)
        self.length = int(self.model.segment_length)
        root = getattr(self.model, "model", None)
        for attribute in TORCH_ROOT.get(self.name, "").split("."):
            if attribute:
                root = getattr(root, attribute)
        self.torch_root = root
        _evaluation_mode(self.torch_root)

    def _load_avex(self, target: str) -> None:
        from avex import load_model

        repository = target.replace("_", "-")
        weights = AVEX_ROOT / repository / f"{repository}.safetensors"
        self.model = load_model(
            target,
            device=self.device,
            checkpoint_path=str(weights),
            return_features_only=True,
        )
        _evaluation_mode(self.model)
        self.rate = 16_000
        self.length = 16_000 * 5
        self.torch_root = self.model

    def _hook(self) -> None:
        import torch

        if not isinstance(self.torch_root, torch.nn.Module):
            raise RuntimeError(f"{self.name} exposes no torch module whose blocks can be read")
        blocks = transformer_blocks(self.torch_root)
        if not blocks:
            raise RuntimeError(f"{self.name}: no transformer blocks found")
        self.block_paths = [path for path, _ in blocks]
        for _, block in blocks:
            block.register_forward_hook(
                lambda _module, _inputs, output: self.captured.append(pool_tokens(output))
            )

    def embed_window(self, window: Any) -> dict[str, Any]:
        """Every candidate of one window."""

        import numpy as np
        import torch

        self.captured = []
        batch = torch.from_numpy(np.asarray(window, dtype=np.float32)[None, :])
        with torch.no_grad():
            if self.spec.source.startswith("avex:"):
                output = self.model(batch.to(self.device))
            else:
                # As Bacpipe's own runner does: the window goes to the model's
                # device before preprocessing, and the preprocessed batch after.
                device = str(self.model.device)
                prepared = self.model.preprocess(batch.to(device))
                if device != "cpu" and hasattr(prepared, "to"):
                    prepared = prepared.to(device)
                output = self.model(prepared)
        result = {EMBEDDING: pool_tokens(output)}
        if self.spec.transformer:
            if len(self.captured) != len(self.block_paths):
                raise RuntimeError(
                    f"{self.name}: {len(self.captured)} block outputs for "
                    f"{len(self.block_paths)} blocks"
                )
            for index, vector in enumerate(self.captured):
                result[f"block {index:02d}"] = vector
        return result

    def release(self) -> None:
        import torch

        del self.model
        self.torch_root = None
        torch.cuda.empty_cache()


def candidate_vectors(name: str, records: Sequence[ClipRecord], *, device: str) -> dict[str, Any]:
    """Every candidate of one model over these clips, in record order."""

    import numpy as np
    import soundfile

    loaded = Loaded(name, device)
    try:
        rows: dict[str, list[Any]] = {}
        for record in records:
            # The shared speech reader centres and standardises waveforms.
            # PA-V6 explicitly preserves the recorded level for these models.
            signal, rate = soundfile.read(record.path, dtype="float32")
            if signal.ndim == 2:
                signal = signal.mean(axis=1)
            per_window = [
                loaded.embed_window(window)
                for window in windows(resample(signal, rate, loaded.rate), loaded.length)
            ]
            for key in per_window[0]:
                rows.setdefault(key, []).append(
                    np.mean([entry[key] for entry in per_window], axis=0)
                )
        return {key: np.vstack(values).astype(np.float32) for key, values in rows.items()}
    finally:
        loaded.release()


def describe(name: str, device: str = "cpu") -> dict[str, Any]:
    """What a model reads: rate, window, blocks. Used to register and to audit."""

    loaded = Loaded(name, device)
    try:
        return {
            "model": name,
            "source": loaded.spec.source,
            "note": loaded.spec.note,
            "sample_rate": loaded.rate,
            "window_samples": loaded.length,
            "blocks": loaded.block_paths,
        }
    finally:
        loaded.release()
