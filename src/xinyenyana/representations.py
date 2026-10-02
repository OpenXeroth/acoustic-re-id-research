"""Reading a clip, and turning it into each of the fixed representations.

These four were the representations every recorded experiment used, and they are
what a new representation is compared against: BirdNET v2.4 embeddings, a
low-capacity log-mel summary, and the two nuisance controls a result has to beat
before it means anything.

The code is unchanged from where it was written, in modules named after
experiments that have been deleted. It is here because it is not about those
experiments: it is how a WAV becomes a vector.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

# --- reading a clip, and the fixed representations ---


def read_pcm_wav(path: Path) -> tuple[Any, int]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - CLI environment guard
        raise RuntimeError("control evaluation requires `uv sync --extra evaluation`") from exc

    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frames = audio.readframes(audio.getnframes())
    if sample_width == 1:
        signal = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0
    elif sample_width == 2:
        signal = np.frombuffer(frames, dtype="<i2").astype(np.float32)
    elif sample_width == 4:
        signal = np.frombuffer(frames, dtype="<i4").astype(np.float32)
    else:
        raise ValueError(f"unsupported PCM sample width {sample_width}: {path}")
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
    if not len(signal):
        raise ValueError(f"empty WAV: {path}")
    signal -= signal.mean()
    scale = float(signal.std())
    if scale > 0:
        signal /= scale
    return signal, sample_rate


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> Any:
    import numpy as np

    def hz_to_mel(value: Any) -> Any:
        return 2595.0 * np.log10(1.0 + value / 700.0)

    def mel_to_hz(value: Any) -> Any:
        return 700.0 * (10 ** (value / 2595.0) - 1.0)

    frequencies = np.linspace(0.0, sample_rate / 2, n_fft // 2 + 1)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / max(center - left, 1e-12),
                (right - frequencies) / max(right - center, 1e-12),
            ),
        )
    return filters


def log_mel_summary(path: Path, *, n_fft: int = 1024, hop: int = 512, n_mels: int = 40) -> Any:
    """Return mean/std log-mel summaries as a deliberately simple nuisance probe."""

    import numpy as np

    signal, sample_rate = read_pcm_wav(path)
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[::hop]
    power = np.abs(np.fft.rfft(frames * np.hamming(n_fft), axis=1)) ** 2
    fmax = min(15_000.0, sample_rate / 2)
    filters = mel_filterbank(sample_rate, n_fft, n_mels, 500.0, fmax)
    log_mel = np.log1p(power @ filters.T)
    return np.concatenate((log_mel.mean(axis=0), log_mel.std(axis=0))).astype(np.float32)


def birdnet_v2_4_vectors(paths: list[Path]) -> tuple[Any, dict[str, Any]]:
    """Encode files with the official BirdNET v2.4 model and mean-pool segments."""

    try:
        import birdnet
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - CLI environment guard
        raise RuntimeError("BirdNET control requires `uv sync --extra birdnet-evaluation`") from exc

    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    # n_workers=4 was checked against n_workers=1 on 96 rook clips and returned
    # bit-identical embeddings, at 2.6 times the rate. n_workers=10 is faster
    # again and is NOT identical: it differs in the sixth decimal place, which
    # is enough to make a run unreproducible, so it is not used.
    result = model.encode(paths, n_workers=4, batch_size=8)
    structured = result.to_structured_array()
    grouped: dict[str, list[Any]] = defaultdict(list)
    for row in structured:
        raw_input = row["input"]
        key = raw_input.decode() if isinstance(raw_input, bytes) else str(raw_input)
        grouped[str(Path(key).resolve())].append(np.asarray(row["embedding"], dtype=np.float32))
    pooled: list[Any] = []
    segment_counts: list[int] = []
    for path in paths:
        segments = grouped.get(str(path.resolve()), [])
        if not segments:
            raise ValueError(f"BirdNET produced no embedding for {path}")
        pooled.append(np.mean(segments, axis=0))
        segment_counts.append(len(segments))
    model_path = Path(model.model_path)
    metadata = {
        "type": "birdnet_v2_4_mean_pooled_segments",
        "dimension": int(np.asarray(pooled).shape[1]),
        "package": "birdnet",
        "package_version": importlib.metadata.version("birdnet"),
        "backend": "LiteRT fp32",
        "model_version": "2.4",
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "model_sample_rate_hz": 48_000,
        "segment_seconds": 3.0,
        "segment_pooling": "arithmetic mean per source clip before L2 normalization",
        "segments_per_clip": {
            "minimum": min(segment_counts),
            "maximum": max(segment_counts),
            "mean": sum(segment_counts) / len(segment_counts),
        },
        "model_license_spdx": "CC-BY-NC-SA-4.0",
        "reproduction_limit": (
            "the 2025 benchmark release does not publish its embedding-generation code, "
            "BirdNET package/checkpoint version, or clip padding rule; this uses the pinned "
            "current official implementation and is not claimed byte-equivalent"
        ),
    }
    return np.asarray(pooled, dtype=np.float32), metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- reading BirdNET's inner layers -----------------------------------------

#: BirdNET v2.4 takes three seconds at 48 kHz.
BIRDNET_SAMPLE_RATE = 48_000
BIRDNET_SEGMENT_SECONDS = 3.0
BIRDNET_SEGMENT_SAMPLES = 144_000

#: The layers this project can read, shallowest first, by the name they carry
#: in the checkpoint rather than by index, which a re-export would change.
#: Time resolution falls from 24 steps to one along the ladder, and the last
#: entry is the 1,024-number embedding every recorded BirdNET figure comes from.
BIRDNET_LAYERS: tuple[tuple[str, str], ...] = (
    ("stage1", "model/BLOCK_1-3_ADD/add"),
    ("stage2", "model/BLOCK_2-4_ADD/add"),
    ("stage3", "model/BLOCK_3-5_ADD/add"),
    ("stage4", "model/BLOCK_4-4_ADD/add"),
    ("post_activation", "model/ACT_POST/Relu;model/BNORM_POST_NOQUANT/FusedBatchNormV3"),
    (
        "post_convolution",
        "model/POST_ACT_1/Relu;model/POST_BN_1/FusedBatchNormV3;model/POST_CONV_1/Conv2D",
    ),
    ("embedding", "model/GLOBAL_AVG_POOL/Mean"),
)


def birdnet_segment_bounds(sample_count: int, sample_rate: int) -> list[tuple[int, int]]:
    """Three-second spans, cut in the recording own sample rate.

    The order matters and was measured rather than assumed. Cutting in the
    source rate and resampling each span reproduces the published package own
    embedding to 2e-6. Resampling the whole signal first and cutting afterwards
    reaches only a cosine of 0.999 on clips longer than one span, because the
    Fourier resampler does not commute with slicing.
    """

    if sample_rate <= 0:
        raise ValueError("a sample rate must be positive")
    if sample_count <= 0:
        raise ValueError("an empty signal has no segments")
    step = int(round(BIRDNET_SEGMENT_SECONDS * sample_rate))
    return [(start, min(start + step, sample_count)) for start in range(0, sample_count, step)]


def birdnet_pad_segment(samples: Any) -> Any:
    """One span brought to the exact length BirdNET reads, padded with silence.

    A span shorter than three seconds is padded rather than stretched or tiled.
    Measured against the package own embedding, padding agrees to 2e-6 while
    tiling disagrees at 0.74 and stretching at 1.87.
    """

    import numpy as np

    signal = np.asarray(samples, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("a segment is cut from one channel, not from a stereo array")
    if signal.size == 0:
        raise ValueError("an empty signal has no segments")
    padded = np.zeros(BIRDNET_SEGMENT_SAMPLES, dtype=np.float32)
    usable = min(signal.size, BIRDNET_SEGMENT_SAMPLES)
    padded[:usable] = signal[:usable]
    return padded


def pool_layer(activation: Any, *, standard_deviation: bool) -> Any:
    """One vector per segment from a layer's time and frequency grid.

    Averaging over time is what every published comparison of these encoders
    does. Carrying the spread beside the average is what speaker verification
    does instead, and it is the cheaper of the two changes this project can
    make to how a layer is read.
    """

    import numpy as np

    values = np.asarray(activation, dtype=np.float32)
    if values.ndim == 2:  # already pooled by the network
        vector = values.reshape(-1)
        return np.concatenate([vector, np.zeros_like(vector)]) if standard_deviation else vector
    if values.ndim != 4:
        raise ValueError(f"a layer is pooled from a 4-axis grid, not {values.ndim} axes")
    flat = values.reshape(values.shape[0], -1, values.shape[-1])[0]
    mean = flat.mean(axis=0)
    if not standard_deviation:
        return mean
    return np.concatenate([mean, flat.std(axis=0)])


def birdnet_layer_vectors(
    paths: list[Path],
    *,
    layers: tuple[tuple[str, str], ...] = BIRDNET_LAYERS,
    standard_deviation: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One vector per clip per layer, through the published package's own audio path.

    Every step of the audio path was checked against the package own embedding
    rather than assumed: read as integers and scaled by 32,768, mixed to one
    channel, cut into three-second spans in the recording own sample rate, each
    span resampled to 48 kHz by the Fourier method, and the last one padded with
    silence.

    Two wrong turns were measured on the way. Linear interpolation in place of
    the Fourier resampler agrees only to a cosine of 0.968. Resampling the whole
    signal before cutting it agrees to 0.999 on clips longer than one span.
    Either would have made every layer read here quietly incomparable with the
    BirdNET figures already on record.
    """

    try:
        import ai_edge_litert.interpreter as litert
        import birdnet
        import numpy as np
        import soundfile
        from scipy.signal import resample
    except ModuleNotFoundError as exc:  # pragma: no cover - CLI environment guard
        raise RuntimeError(
            "reading BirdNET's inner layers requires `uv sync --extra birdnet-evaluation`"
        ) from exc

    model = birdnet.load("acoustic", "2.4", "tf", library="litert")
    interpreter = litert.Interpreter(
        model_path=str(model.model_path),
        num_threads=1,
        experimental_op_resolver_type=litert.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    index_of = {detail["name"]: detail["index"] for detail in interpreter.get_tensor_details()}
    missing = [name for _, name in layers if name not in index_of]
    if missing:
        raise ValueError(f"this checkpoint holds no layer named {missing}")
    input_index = interpreter.get_input_details()[0]["index"]

    collected: dict[str, list[Any]] = {short: [] for short, _ in layers}
    segments_per_clip = []
    for path in paths:
        audio, rate = soundfile.read(str(path), dtype="int16")
        signal = np.asarray(audio, dtype=np.float32) / 32768.0
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        segments = []
        for start, stop in birdnet_segment_bounds(signal.size, rate):
            span = signal[start:stop]
            if rate != BIRDNET_SAMPLE_RATE:
                span = resample(span, round(span.size / rate * BIRDNET_SAMPLE_RATE)).astype(
                    np.float32
                )
            segments.append(birdnet_pad_segment(span))
        segments_per_clip.append(len(segments))
        pooled: dict[str, list[Any]] = {short: [] for short, _ in layers}
        for segment in segments:
            interpreter.set_tensor(input_index, segment[None, :])
            interpreter.invoke()
            for short, name in layers:
                pooled[short].append(
                    pool_layer(
                        interpreter.get_tensor(index_of[name]),
                        standard_deviation=standard_deviation,
                    )
                )
        for short in pooled:
            collected[short].append(np.mean(pooled[short], axis=0))

    vectors = {short: np.asarray(rows, dtype=np.float32) for short, rows in collected.items()}
    metadata = {
        "type": "birdnet_v2_4_inner_layers",
        "model_version": "2.4",
        "model_path": str(model.model_path),
        "model_sha256": _sha256(Path(model.model_path)),
        "sample_rate_hz": BIRDNET_SAMPLE_RATE,
        "segment_samples": BIRDNET_SEGMENT_SAMPLES,
        "segmentation": "three-second spans cut in the recording own sample rate",
        "resampling": "scipy.signal.resample per span, Fourier method",
        "short_span_handling": "zero padded to one segment",
        "pooling": "mean and standard deviation" if standard_deviation else "mean",
        "layers": {short: int(vectors[short].shape[1]) for short, _ in layers},
        "segments_per_clip": {
            "minimum": min(segments_per_clip),
            "maximum": max(segments_per_clip),
            "mean": sum(segments_per_clip) / len(segments_per_clip),
        },
    }
    return vectors, metadata
