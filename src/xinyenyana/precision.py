"""Explicit CUDA arithmetic for replayable speech-transformer extraction."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def speech_matmul(precision: str) -> Iterator[dict[str, bool | str]]:
    """Ignore inherited SpeechBrain matmul flags; restore them on every exit.

    The two forensic contexts differ only in CUDA matrix TF32. Convolutions
    retain TF32 permission in both, as in the archived diagnostic.
    """
    if precision not in {"fp32", "tf32"}:
        raise ValueError("speech matrix precision must be fp32 or tf32")
    import torch

    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    previous = matmul.allow_tf32, cudnn.allow_tf32
    try:
        matmul.allow_tf32 = precision == "tf32"
        cudnn.allow_tf32 = True
        yield {
            "speech_matmul": precision,
            "cuda_matmul_allow_tf32": bool(matmul.allow_tf32),
            "cudnn_allow_tf32": bool(cudnn.allow_tf32),
        }
        if matmul.allow_tf32 != (precision == "tf32") or not cudnn.allow_tf32:
            raise RuntimeError("speech extraction changed its declared CUDA arithmetic")
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = previous
