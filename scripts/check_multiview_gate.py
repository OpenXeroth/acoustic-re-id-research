"""Bounded xen1 integration check using generated audio, never benchmark labels."""

from __future__ import annotations

import json
import tempfile
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from xinyenyana.multiview_gate import run_multiview_gate


def main() -> None:
    rate = 48000
    rng = np.random.default_rng(19)
    signal = np.zeros((rate * 13, 2), dtype=np.int16)
    annotations = ["Source\tEvent\tComment\tStart\tEnd"]
    for index in range(24):
        start = 0.2 + index * 0.5
        samples = rng.integers(-3000, 3000, size=9600, dtype=np.int16)
        first = round(start * rate)
        signal[first : first + len(samples), 0] = samples
        signal[first : first + len(samples), 1] = samples // 2
        annotations.append(f"Inc\tfixture\tgenerated\t{start}\t{start + 0.2}")
    with tempfile.TemporaryDirectory(prefix="xyy-multiview-fixture-") as work:
        root = Path(work)
        wav = root / "fixture.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(signal.tobytes())
        archive = root / "fixture.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.write(wav, "RookID/fixture.wav")
            bundle.writestr("RookID/fixture.tsv", "\n".join(annotations) + "\n")
        first_result = run_multiview_gate(archive, root / "cache")
        assert first_result["counts"]["annotations"] == 24
        assert first_result["counts"]["usable_events"] == 24
        assert first_result["counts"]["recordings_with_cross_group_events"] == 0
        assert not first_result["registered_count_gate_met_for_inferred_groups"]
        assert first_result["negative_associations"]["accepted"] == 0
        assert not first_result["hardware_mapping_verified"]
        with patch(
            "xinyenyana.multiview_gate.audit_recording",
            side_effect=AssertionError("cache was not reused"),
        ):
            replay = run_multiview_gate(archive, root / "cache")
        assert replay == first_result
        print(
            json.dumps(
                {
                    "kind": "generated integration fixture",
                    "scientific_result": False,
                    "checks": (
                        "24 same-event pairs retained; distinct events rejected; "
                        "checkpoint replay identical"
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
