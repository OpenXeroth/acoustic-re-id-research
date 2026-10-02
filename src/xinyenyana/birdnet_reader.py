"""Read a frozen BirdNET checkpoint once and retain its spatial/time structure.

The mean and mean/spread outputs preserve the registered three-second audio
path. Sequences average frequency bins and retain time positions, omitting only
trailing positions in proportion to the last segment's unpadded duration.
This is a uniform-grid crop, not a receptive-field boundary calculation.
No identity labels or species scores enter extraction.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from xinyenyana.archive import sha256_file
from xinyenyana.extraction_cache import ExtractionCache
from xinyenyana.representations import (
    BIRDNET_LAYERS,
    BIRDNET_SAMPLE_RATE,
    BIRDNET_SEGMENT_SAMPLES,
    birdnet_pad_segment,
    birdnet_segment_bounds,
    pool_layer,
)


def time_positions(activation: Any, valid_fraction: float) -> Any:
    """Average frequency and crop a uniform time grid by the real-audio fraction."""

    import numpy as np

    grid = np.asarray(activation, dtype=np.float32)
    if grid.ndim == 2 and grid.shape[0] == 1:
        return grid.copy()
    if grid.ndim != 4 or grid.shape[0] != 1:
        raise ValueError("expected a single BirdNET spatial activation")
    if not 0 < valid_fraction <= 1:
        raise ValueError("valid fraction must lie in (0, 1]")
    usable = max(1, int(np.ceil(grid.shape[1] * valid_fraction)))
    return grid[0, :usable].mean(axis=1)


class BirdNETReader:
    def __init__(self, cache_root: Path, *, threads: int = 1) -> None:
        import ai_edge_litert.interpreter as litert
        import birdnet

        model = birdnet.load("acoustic", "2.4", "tf", library="litert")
        model_path = Path(model.model_path)
        self.interpreter = litert.Interpreter(
            model_path=str(model_path),
            num_threads=threads,
            experimental_op_resolver_type=litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            experimental_preserve_all_tensors=True,
        )
        self.interpreter.allocate_tensors()
        tensors = self.interpreter.get_tensor_details()
        indices = {row["name"]: row["index"] for row in tensors}
        self.indices = {short: indices[name] for short, name in BIRDNET_LAYERS}
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.specification = {
            "extractor_sha256": sha256_file(Path(__file__)),
            "model_sha256": sha256_file(model_path),
            "parameters": {
                "preprocessing_source_sha256": sha256_file(
                    Path(__file__).with_name("representations.py")
                ),
                "layers": list(BIRDNET_LAYERS),
                "threads": threads,
                "sample_rate": BIRDNET_SAMPLE_RATE,
                "segment_samples": BIRDNET_SEGMENT_SAMPLES,
                "audio": (
                    "int16 / 32768; mono; source-rate slicing; Fourier resampling; zero padding"
                ),
                "pooling": "mean and population spread per segment, then mean across segments",
                "sequence": "frequency mean; first ceil(time positions * real-audio fraction)",
            },
            "environment": {
                name: importlib.metadata.version(name)
                for name in ("birdnet", "numpy", "scipy", "soundfile", "ai-edge-litert")
            },
        }
        self.cache = ExtractionCache(cache_root, self.specification)

    def read(self, path: Path) -> dict[str, Any]:
        return self.cache.get(path, lambda: self._extract(path))

    def _extract(self, path: Path) -> dict[str, Any]:
        import numpy as np
        import soundfile
        from scipy.signal import resample

        audio, rate = soundfile.read(str(path), dtype="int16")
        signal = np.asarray(audio, dtype=np.float32) / 32768.0
        if signal.ndim == 2:
            signal = signal.mean(axis=1)
        collected: dict[str, list[Any]] = {}
        for start, stop in birdnet_segment_bounds(len(signal), rate):
            span = signal[start:stop]
            fraction = len(span) / (3.0 * rate)
            if rate != BIRDNET_SAMPLE_RATE:
                span = resample(span, round(len(span) / rate * BIRDNET_SAMPLE_RATE)).astype(
                    np.float32
                )
            self.interpreter.set_tensor(self.input_index, birdnet_pad_segment(span)[None, :])
            self.interpreter.invoke()
            for layer, index in self.indices.items():
                activation = self.interpreter.get_tensor(index)
                for spread in (False, True):
                    key = f"{layer}.{'mean_std' if spread else 'mean'}"
                    collected.setdefault(key, []).append(
                        pool_layer(activation, standard_deviation=spread)
                    )
                collected.setdefault(f"{layer}.sequence", []).append(
                    time_positions(activation, fraction)
                )
        return {
            key: (
                np.concatenate(values, axis=0)
                if key.endswith(".sequence")
                else np.mean(values, axis=0)
            )
            for key, values in collected.items()
        }
