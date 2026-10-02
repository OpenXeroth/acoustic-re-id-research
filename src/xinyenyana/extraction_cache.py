"""Content-addressed, resumable extraction on the benchmark host's own disk.

Only complete entries are visible. Identity labels are deliberately absent:
changing a split can reuse frozen audio features, but changing audio, the model,
the numerical environment or extraction parameters cannot reuse an old entry.
These arrays are working data, never repository files or publication artifacts.
"""

from __future__ import annotations

import json
import os
import signal
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from xinyenyana.archive import canonical_sha256, sha256_file


class ExtractionCache:
    def __init__(self, root: Path, specification: dict[str, Any]) -> None:
        required = {"extractor_sha256", "model_sha256", "parameters", "environment"}
        if missing := required - specification.keys():
            raise ValueError(f"cache specification is missing {sorted(missing)}")
        self.specification = specification
        self.digest = canonical_sha256(specification)
        self.root = root / self.digest
        self.root.mkdir(parents=True, exist_ok=True)
        description = self.root / "specification.json"
        encoded = json.dumps(specification, sort_keys=True, indent=2) + "\n"
        if description.exists() and description.read_text() != encoded:
            raise ValueError("cache specification does not match its directory digest")
        if not description.exists():
            _atomic_text(description, encoded)

    def entry(self, audio: Path) -> Path:
        return self.root / f"{sha256_file(audio)}.npz"

    def get(self, audio: Path, extract: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        import numpy as np

        target = self.entry(audio)
        if target.exists():
            with np.load(target, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            self._validate(arrays)
            return arrays
        arrays = {key: np.asarray(value) for key, value in extract().items()}
        self._validate(arrays)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return arrays

    @staticmethod
    def _validate(arrays: dict[str, Any]) -> None:
        import numpy as np

        if not arrays:
            raise ValueError("an extraction cache entry cannot be empty")
        for key, array in arrays.items():
            if array.dtype.kind not in "fiu" or not array.size or not np.isfinite(array).all():
                raise ValueError(f"invalid numeric extraction array {key!r}")


def _atomic_text(target: Path, text: str) -> None:
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        with temporary.open("x") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def checkpointed_interrupts() -> Iterator[None]:
    """A warden stop terminates the job; completed cache entries remain reusable."""

    def stop(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        for signum in previous:
            signal.signal(signum, stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
