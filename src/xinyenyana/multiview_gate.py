"""Prospective acoustic applicability checks for the RookID multiview study.

Correlation groups are inferred, not a recovered hardware wiring diagram.
Automatic onset and association screens do not supply human event validation.
The count gate can reject a corpus under its registered criteria; a pass alone
does not establish the intended physical interpretation.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from xinyenyana.archive import canonical_sha256, sha256_file
from xinyenyana.extraction_cache import _atomic_text, checkpointed_interrupts
from xinyenyana.frozen_probe import runtime_record
from xinyenyana.open_set import wilson_interval

CONFIG: dict[str, int | float | str] = {
    "overlap_margin_seconds": 0.02,
    "geometry_windows": 100,
    "geometry_minimum_windows": 20,
    "geometry_pair_median_minimum": 0.35,
    "geometry_pairing_margin": 0.05,
    "maximum_lag_seconds": 0.075,
    "minimum_duration_seconds": 0.1,
    "minimum_rms": 1e-5,
    "maximum_clipped_share": 0.01,
    "onset_bin_seconds": 0.01,
    "onset_rise_fraction": 0.30,
    "onset_merge_seconds": 0.02,
    "pair_minimum_correlation": 0.60,
    "pair_peak_margin": 0.10,
    "peak_exclusion_seconds": 0.001,
    "half_minimum_correlation": 0.40,
    "half_delay_tolerance_seconds": 0.001,
    "minimum_events": 500,
    "minimum_cross_group_recordings": 5,
    "selection_salt": "xyy-reid02-geometry-20260913",
}


@dataclass(frozen=True)
class Window:
    row: int
    start: float
    end: float
    multiple_emitters: bool


def eligible_windows(windows: list[Window]) -> tuple[list[Window], dict[int, str]]:
    rejected: dict[int, str] = {}
    margin = float(CONFIG["overlap_margin_seconds"])
    for window in windows:
        if not all(math.isfinite(v) for v in (window.start, window.end)) or not (
            0 <= window.start < window.end
        ):
            raise ValueError("invalid published annotation interval")
        if window.multiple_emitters:
            rejected[window.row] = "explicit multiple emitters"
        elif any(
            other.row != window.row
            and other.start - margin < window.end + margin
            and other.end + margin > window.start - margin
            for other in windows
        ):
            rejected[window.row] = "overlapping annotation with 20 ms margins"
    return [window for window in windows if window.row not in rejected], rejected


def correlation_peak(first: Any, second: Any, rate: int) -> dict[str, float]:
    """DC-removed absolute FFT correlation, normalised by full-window energy."""

    import numpy as np

    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 2:
        raise ValueError("a correlation pair needs equal nontrivial mono windows")
    if rate <= 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid correlation input")
    a, b = a - a.mean(), b - b.mean()
    energy = float(np.linalg.norm(a) * np.linalg.norm(b))
    if energy <= 1e-30:
        return {"correlation": 0.0, "lag_seconds": 0.0, "peak_margin": 0.0}
    nfft = 1 << (2 * len(a) - 1).bit_length()
    correlation = (
        np.abs(np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)) / energy
    )
    lag_limit = min(len(a) - 1, round(rate * float(CONFIG["maximum_lag_seconds"])))
    lags = np.arange(-lag_limit, lag_limit + 1)
    values = correlation[lags % nfft]
    best = int(np.argmax(values))
    outside = np.abs(lags - lags[best]) > round(rate * float(CONFIG["peak_exclusion_seconds"]))
    runner_up = float(values[outside].max()) if outside.any() else float(values[best])
    return {
        "correlation": float(values[best]),
        "lag_seconds": float(lags[best] / rate),
        "peak_margin": float(values[best] - runner_up),
    }


def pair_association(first: Any, second: Any, rate: int) -> dict[str, Any]:
    import numpy as np

    if len(first) != len(second):
        raise ValueError("an association requires aligned window lengths")
    if len(first) < math.ceil(rate * float(CONFIG["minimum_duration_seconds"])):
        return {"accepted": False, "reason": "shorter than 100 ms"}
    full = correlation_peak(first, second, rate)
    middle = len(first) // 2
    halves = [
        correlation_peak(first[a:b], second[a:b], rate)
        for a, b in ((0, middle), (middle, len(first)))
    ]
    stable = abs(halves[0]["lag_seconds"] - halves[1]["lag_seconds"]) <= float(
        CONFIG["half_delay_tolerance_seconds"]
    )
    accepted = (
        full["correlation"] >= float(CONFIG["pair_minimum_correlation"])
        and full["peak_margin"] >= float(CONFIG["pair_peak_margin"])
        and all(half["correlation"] >= float(CONFIG["half_minimum_correlation"]) for half in halves)
        and stable
    )
    energy_a, energy_b = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    level_ratio = 20 * math.log10(energy_b / energy_a) if min(energy_a, energy_b) > 0 else None
    return {
        "accepted": accepted,
        **full,
        "half_peaks": halves,
        "half_delay_consistent": stable,
        "second_over_first_rms_db": level_ratio,
    }


def single_onset_channel(signal: Any, rate: int) -> dict[str, Any]:
    import numpy as np

    signal = np.asarray(signal, dtype=float)
    if not np.isfinite(signal).all():
        raise ValueError("nonfinite source audio")
    width = max(1, round(rate * float(CONFIG["onset_bin_seconds"])))
    if len(signal) < math.ceil(rate * float(CONFIG["minimum_duration_seconds"])):
        return {"accepted": False, "reason": "shorter than 100 ms"}
    rms = float(np.sqrt(np.mean(signal * signal)))
    clipping = float(np.mean(np.abs(signal) >= 0.999))
    count = len(signal) // width
    envelope = np.sqrt(np.mean(signal[: count * width].reshape(count, width) ** 2, axis=1))
    smooth = np.convolve(envelope, np.ones(3) / 3, mode="same")
    rises = np.maximum(np.diff(smooth, prepend=0.0), 0.0)
    active = np.flatnonzero(rises > float(CONFIG["onset_rise_fraction"]) * rises.max())
    allowed_gap = round(float(CONFIG["onset_merge_seconds"]) * rate / width)
    runs = 0 if len(active) == 0 else 1 + int(np.sum(np.diff(active) > allowed_gap + 1))
    return {
        "accepted": rms >= float(CONFIG["minimum_rms"])
        and clipping <= float(CONFIG["maximum_clipped_share"])
        and runs == 1,
        "rms": rms,
        "clipped_share": clipping,
        "dominant_onset_runs": runs,
    }


def _pairings(channels: tuple[int, ...]) -> list[tuple[tuple[int, int], ...]]:
    if not channels:
        return [()]
    first = channels[0]
    return [
        ((first, second), *rest)
        for second in channels[1:]
        for rest in _pairings(tuple(c for c in channels if c not in (first, second)))
    ]


def infer_groups(medians: Any) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(medians, dtype=float)
    if values.shape[0] not in (2, 4, 6) or values.shape != (values.shape[0], values.shape[0]):
        raise ValueError("RookID grouping expects 2, 4 or 6 channels")
    candidates = sorted(
        [
            (float(np.mean([values[a, b] for a, b in pairs])), pairs)
            for pairs in _pairings(tuple(range(len(values))))
        ],
        key=lambda item: (-item[0], item[1]),
    )
    score, pairs = candidates[0]
    margin = score - candidates[1][0] if len(candidates) > 1 else None
    minimum = float(CONFIG["geometry_pair_median_minimum"])
    required_margin = float(CONFIG["geometry_pairing_margin"])
    weakest = min(values[a, b] for a, b in pairs)
    reasons = []
    if weakest < minimum:
        reasons.append(
            f"the weakest inferred pair correlates at {weakest:.3f}, below {minimum:.2f}"
        )
    if margin is not None and margin < required_margin:
        reasons.append(
            f"the best pairing leads the next by {margin:.3f}, below {required_margin:.2f}"
        )
    return {
        # A rejection says which registered number it missed and by how much.
        # It used to carry no reason at all here, and sixteen of the eighty-one
        # recordings in the first full run were rejected by a null.
        "accepted": not reasons,
        "reason": "; ".join(reasons) or None,
        "inferred_pairs": [list(pair) for pair in pairs],
        "mean_pair_score": score,
        "margin": margin,
        "weakest_pair_correlation": float(weakest),
        "pairwise_median_correlations": values.tolist(),
        "hardware_mapping_verified": False,
    }


def _window_audio(stream: Any, window: Window) -> Any:
    rate = stream.samplerate
    start, end = int(window.start * rate), int(window.end * rate)
    if end > len(stream):
        raise ValueError("annotation extends beyond the source recording")
    stream.seek(start)
    signal = stream.read(end - start, dtype="float64", always_2d=True)
    if len(signal) != end - start:
        raise ValueError("incomplete local waveform read")
    return signal


def audit_recording(path: Path, windows: list[Window], recording: str) -> dict[str, Any]:
    import numpy as np
    import soundfile

    eligible, rejected = eligible_windows(windows)
    ordered = sorted(eligible, key=lambda w: (w.start, w.row))
    rows: dict[int, dict[str, Any]] = {
        w.row: {
            "row": w.row,
            "start": w.start,
            "end": w.end,
            "annotation_eligible": w.row not in rejected,
            "reason": rejected.get(w.row),
        }
        for w in windows
    }
    output: dict[str, Any] = {
        "recording": recording,
        "annotations": len(windows),
        "eligible_annotations": len(eligible),
    }
    if len(eligible) < int(CONFIG["geometry_minimum_windows"]):
        return {
            **output,
            "geometry": {"accepted": False, "reason": "fewer than 20 eligible windows"},
            "events": list(rows.values()),
            "usable_events": 0,
            "cross_group_events": 0,
            "negative_associations": {"trials": 0, "accepted": 0},
        }
    with soundfile.SoundFile(path) as stream:
        rate, channels = stream.samplerate, stream.channels
        if rate != 48000 or channels not in (2, 4, 6):
            raise ValueError("unexpected published RookID audio format")
        selection = sorted(
            eligible,
            key=lambda w: hashlib.sha256(
                f"{CONFIG['selection_salt']}:{recording}:{w.row}".encode()
            ).hexdigest(),
        )[: int(CONFIG["geometry_windows"])]
        correlations: dict[tuple[int, int], list[float]] = {
            pair: [] for pair in combinations(range(channels), 2)
        }
        level_ratios: dict[tuple[int, int], list[float]] = {pair: [] for pair in correlations}
        peak_delays: dict[tuple[int, int], list[float]] = {pair: [] for pair in correlations}
        for window in selection:
            signal = _window_audio(stream, window)
            for pair, values in correlations.items():
                peak = correlation_peak(signal[:, pair[0]], signal[:, pair[1]], rate)
                values.append(peak["correlation"])
                peak_delays[pair].append(peak["lag_seconds"])
                norm_a, norm_b = (
                    np.linalg.norm(signal[:, pair[0]]),
                    np.linalg.norm(signal[:, pair[1]]),
                )
                if min(norm_a, norm_b) > 0:
                    level_ratios[pair].append(float(20 * np.log10(norm_b / norm_a)))
        matrix = np.eye(channels)
        for (a, b), values in correlations.items():
            matrix[a, b] = matrix[b, a] = np.median(values)
        geometry = {
            **infer_groups(matrix),
            "selected_annotation_rows": [w.row for w in selection],
            "channel_pair_diagnostics": [
                {
                    "channels": list(pair),
                    "median_lag_seconds": float(np.median(peak_delays[pair])),
                    "median_second_over_first_rms_db": float(np.median(level_ratios[pair]))
                    if level_ratios[pair]
                    else None,
                }
                for pair in correlations
            ],
        }
        output["geometry"] = geometry
        negative_trials = negative_accepted = usable = cross_group = 0
        previous = None
        group = {
            channel: i for i, pair in enumerate(geometry["inferred_pairs"]) for channel in pair
        }
        for window in ordered:
            signal = _window_audio(stream, window)
            row = rows[window.row]
            if previous is not None:
                count = min(len(previous), len(signal))
                negative = pair_association(previous[:count, 0], signal[:count, 1], rate)
                negative_trials += 1
                negative_accepted += int(negative["accepted"])
                row["negative_previous_event"] = negative
            previous = signal
            if not geometry["accepted"]:
                row["reason"] = "inferred geometry ambiguous or weak"
                continue
            checks = [single_onset_channel(signal[:, channel], rate) for channel in range(channels)]
            row["channel_checks"] = checks
            pairs: list[dict[str, Any]] = []
            for a, b in combinations(range(channels), 2):
                if checks[a]["accepted"] and checks[b]["accepted"]:
                    association = pair_association(signal[:, a], signal[:, b], rate)
                    pairs.append(
                        {
                            "channels": [a, b],
                            "across_inferred_groups": group[a] != group[b],
                            **association,
                        }
                    )
            accepted = [pair for pair in pairs if pair["accepted"]]
            row["pair_checks"] = pairs
            row["usable_channels"] = sorted({c for pair in accepted for c in pair["channels"]})
            row["usable"] = len(row["usable_channels"]) >= 2
            row["across_inferred_groups"] = any(pair["across_inferred_groups"] for pair in accepted)
            usable += int(row["usable"])
            cross_group += int(row["across_inferred_groups"])
        return {
            **output,
            "events": list(rows.values()),
            "usable_events": usable,
            "cross_group_events": cross_group,
            "negative_associations": {"trials": negative_trials, "accepted": negative_accepted},
        }


def run_multiview_gate(archive: Path, cache_root: Path) -> dict[str, Any]:
    specification = {
        "archive_sha256": sha256_file(archive),
        "code_sha256": sha256_file(Path(__file__)),
        "config": CONFIG,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "soundfile")},
    }
    root = cache_root / canonical_sha256(specification)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_text(root / "specification.json", json.dumps(specification, sort_keys=True, indent=2))
    recordings = []
    with checkpointed_interrupts(), zipfile.ZipFile(archive) as bundle:
        annotations = sorted(name for name in bundle.namelist() if name.endswith(".tsv"))
        for index, annotation in enumerate(annotations):
            stem = Path(annotation).stem
            target = root / (hashlib.sha256(annotation.encode()).hexdigest() + ".json")
            if target.exists():
                result = json.loads(target.read_text())
            else:
                table = csv.DictReader(
                    io.StringIO(bundle.read(annotation).decode("utf-8-sig")), delimiter="\t"
                )
                windows = [
                    Window(
                        i, float(row["Start"]), float(row["End"]), row["Source"].strip() == "Pls"
                    )
                    for i, row in enumerate(table, 2)
                ]
                with tempfile.TemporaryDirectory(prefix="multiview-source-", dir=root) as temporary:
                    wav = Path(temporary) / "recording.wav"
                    with (
                        bundle.open(annotation[:-4] + ".wav") as source,
                        wav.open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination, 1 << 20)
                    result = audit_recording(wav, windows, stem)
                _atomic_text(target, json.dumps(result, sort_keys=True, indent=2))
            recordings.append(result)
            print(
                f"multiview geometry {index + 1}/{len(annotations)} recordings",
                file=sys.stderr,
                flush=True,
            )
    usable = sum(r["usable_events"] for r in recordings)
    crossed = sum(r["cross_group_events"] > 0 for r in recordings)
    trials = sum(r["negative_associations"]["trials"] for r in recordings)
    false_matches = sum(r["negative_associations"]["accepted"] for r in recordings)
    return {
        "experiment": "E-REID-02-stage-one",
        "specification": specification,
        "environment": runtime_record(),
        "recordings": recordings,
        "counts": {
            "recordings": len(recordings),
            "annotations": sum(r["annotations"] for r in recordings),
            "annotation_eligible": sum(r["eligible_annotations"] for r in recordings),
            "accepted_inferred_geometries": sum(r["geometry"]["accepted"] for r in recordings),
            "usable_events": usable,
            "recordings_with_cross_group_events": crossed,
        },
        "registered_count_gate_met_for_inferred_groups": usable >= int(CONFIG["minimum_events"])
        and crossed >= int(CONFIG["minimum_cross_group_recordings"]),
        "negative_associations": {
            "accepted": false_matches,
            "trials": trials,
            "wilson_95": wilson_interval(false_matches, trials) if trials else None,
        },
        "hardware_mapping_verified": False,
        "claim_boundary": (
            "Automatic acoustic eligibility under fixed criteria; inferred correlation groups "
            "do not establish recorder hardware, source position or validated same-emission "
            "labels. No representation distance or identity accuracy was measured."
        ),
    }
