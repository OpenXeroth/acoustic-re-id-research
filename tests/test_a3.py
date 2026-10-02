"""Tests for the A3 runner: the band source, the trim exclusion, the rule."""

from __future__ import annotations

import wave
from array import array
from pathlib import Path

import numpy as np
import pytest

from xinyenyana.a2 import ClipRecord, Endpoint, extract_endpoint_layers
from xinyenyana.a3 import (
    CONFIGURATIONS,
    LAYERS_EXCLUDED_BY_TRIM,
    adopt,
    endpoint_band,
    run_a3_endpoint,
)
from xinyenyana.cleaning import Band

SAMPLE_RATE = 22_050


def _write(path: Path, samples: np.ndarray) -> Path:
    scaled = np.clip(samples / max(float(np.abs(samples).max()), 1e-9) * 0.8, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(array("h", (int(v * 32_000) for v in scaled)).tobytes())
    return path


def _tone(frequency: float, seconds: float = 0.6) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return np.sin(2 * np.pi * frequency * t)


def _endpoint(tmp_path: Path, *, enrollment_hz: float, query_hz: float) -> Endpoint:
    records = []
    for index in range(8):
        split = "enrollment" if index < 4 else "query"
        frequency = enrollment_hz if split == "enrollment" else query_hz
        path = _write(tmp_path / f"{split}-{index}.wav", _tone(frequency + 40.0 * (index % 2)))
        records.append(
            ClipRecord(
                filename=f"clip-{index}",
                path=path,
                identity=f"bird{index % 2}",
                split=split,
                context={},
            )
        )
    return Endpoint(
        name="test-endpoint",
        records=tuple(records),
        categorical_targets=(),
        manifest_sha256="0" * 64,
        source_document="docs/measured.md",
    )


def test_the_band_comes_from_enrollment_clips_and_not_from_query_clips(
    tmp_path: Path,
) -> None:
    """The query clips hold 7 kHz; the band must not reach them."""

    endpoint = _endpoint(tmp_path, enrollment_hz=1_000.0, query_hz=7_000.0)
    band = endpoint_band(endpoint)
    assert band.clips == 4
    assert band.low_hz <= 1_000.0 <= band.high_hz
    assert band.high_hz < 7_000.0


def test_cleaning_changes_the_signal_the_layers_see(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path, enrollment_hz=1_000.0, query_hz=1_000.0)
    plain, _, _ = extract_endpoint_layers(endpoint.records)
    limited, _, _ = extract_endpoint_layers(
        endpoint.records, steps=("band",), band=Band(low_hz=3_000.0, high_hz=5_000.0, clips=4)
    )
    assert not np.allclose(plain["filter"], limited["filter"])


def test_band_limiting_without_a_band_is_refused(tmp_path: Path) -> None:
    endpoint = _endpoint(tmp_path, enrollment_hz=1_000.0, query_hz=1_000.0)
    with pytest.raises(ValueError, match="needs a band"):
        extract_endpoint_layers(endpoint.records, steps=("band",), band=None)


def test_every_configuration_that_trims_excludes_the_motor_layer() -> None:
    trimming = [name for name, steps in CONFIGURATIONS if "trim" in steps]
    assert trimming == ["trim", "all"]
    assert LAYERS_EXCLUDED_BY_TRIM == ("motor",)


def test_a_step_that_raises_accuracy_and_leaves_the_probes_alone_is_adopted() -> None:
    summary = {
        "configurations": {
            "none": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.40},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.1}},
                    }
                }
            },
            "band": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.55},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.1}},
                    }
                }
            },
        }
    }
    verdict = adopt(summary, layer="filter")["band"]
    assert verdict["raised_accuracy"] is True
    assert verdict["raised_a_context_probe"] is False
    assert verdict["adopted"] is True


def test_a_step_that_raises_accuracy_and_a_probe_is_not_adopted() -> None:
    summary = {
        "configurations": {
            "none": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.40},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.1}},
                    }
                }
            },
            "denoise": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.55},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.3}},
                    }
                }
            },
        }
    }
    assert adopt(summary, layer="filter")["denoise"]["adopted"] is False


def test_a_step_that_lowers_accuracy_is_not_adopted() -> None:
    summary = {
        "configurations": {
            "none": {
                "layers": {"filter": {"classification": {"accuracy": 0.40}, "context_probes": {}}}
            },
            "trim": {
                "layers": {"filter": {"classification": {"accuracy": 0.20}, "context_probes": {}}}
            },
        }
    }
    assert adopt(summary, layer="filter")["trim"]["adopted"] is False


def test_a_probe_at_its_floor_does_not_block_adoption() -> None:
    """The reason the rule says 'does not raise' rather than 'lowers'."""

    summary = {
        "configurations": {
            "none": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.40},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.0}},
                    }
                }
            },
            "band": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.55},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.0}},
                    }
                }
            },
        }
    }
    assert adopt(summary, layer="filter")["band"]["adopted"] is True


def test_a_categorical_probe_is_compared_on_accuracy() -> None:
    summary = {
        "configurations": {
            "none": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.40},
                        "context_probes": {"year": {"accuracy": 0.5}},
                    }
                }
            },
            "band": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.55},
                        "context_probes": {"year": {"accuracy": 0.7}},
                    }
                }
            },
        }
    }
    assert adopt(summary, layer="filter")["band"]["adopted"] is False


def test_a_layer_absent_from_a_configuration_is_skipped_by_the_rule() -> None:
    summary = {
        "configurations": {
            "none": {
                "layers": {"motor": {"classification": {"accuracy": 0.4}, "context_probes": {}}}
            },
            "trim": {"layers": {}},
        }
    }
    assert adopt(summary, layer="motor") == {}


def test_the_rule_reads_the_shape_the_runner_actually_produces(tmp_path: Path) -> None:
    """The synthetic summaries above would pass against a shape nothing produces.

    This runs the real runner on a real endpoint and applies the rule to its
    output, so a change to what ``evaluate_layer`` returns fails here.
    """

    endpoint = _endpoint(tmp_path, enrollment_hz=1_000.0, query_hz=1_000.0)
    summary = run_a3_endpoint(endpoint=endpoint)
    assert set(summary["configurations"]) == {name for name, _ in CONFIGURATIONS}
    verdicts = adopt(summary, layer="filter")
    assert set(verdicts) == {"trim", "band", "denoise", "all"}
    for verdict in verdicts.values():
        assert isinstance(verdict["adopted"], bool)
        assert 0.0 <= verdict["accuracy_before"] <= 1.0
        assert 0.0 <= verdict["accuracy_after"] <= 1.0


def test_a_probe_with_no_baseline_reading_blocks_adoption() -> None:
    """The RookID case: every clip is the same length until a step trims it.

    ``log_duration_seconds`` is not measurable on the untreated clips, so the
    baseline carries no reading for it. Trimming makes durations vary, and the
    probe then recovers identity from duration alone. Skipping the probe for
    want of a baseline would adopt the step for producing that signal.
    """

    summary = {
        "configurations": {
            "none": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.50},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.0}},
                    }
                }
            },
            "all": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.70},
                        "context_probes": {
                            "rms_dbfs": {"r_squared": 0.0},
                            "log_duration_seconds": {"r_squared": 0.57},
                        },
                    }
                }
            },
        }
    }
    verdict = adopt(summary, layer="filter")["all"]
    assert verdict["raised_accuracy"] is True
    assert verdict["raised_a_context_probe"] is True
    assert verdict["adopted"] is False


def test_a_probe_with_no_baseline_reading_at_its_floor_does_not_block() -> None:
    summary = {
        "configurations": {
            "none": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.50},
                        "context_probes": {"rms_dbfs": {"r_squared": 0.0}},
                    }
                }
            },
            "trim": {
                "layers": {
                    "filter": {
                        "classification": {"accuracy": 0.73},
                        "context_probes": {
                            "rms_dbfs": {"r_squared": 0.0},
                            "log_duration_seconds": {"r_squared": 0.0},
                        },
                    }
                }
            },
        }
    }
    assert adopt(summary, layer="filter")["trim"]["adopted"] is True


def test_a_categorical_probe_with_no_baseline_is_compared_to_its_majority_rate() -> None:
    def summary_with(accuracy: float) -> dict[str, object]:
        return {
            "configurations": {
                "none": {
                    "layers": {
                        "filter": {
                            "classification": {"accuracy": 0.50},
                            "context_probes": {"rms_dbfs": {"r_squared": 0.0}},
                        }
                    }
                },
                "trim": {
                    "layers": {
                        "filter": {
                            "classification": {"accuracy": 0.70},
                            "context_probes": {
                                "rms_dbfs": {"r_squared": 0.0},
                                "year": {
                                    "accuracy": accuracy,
                                    "classes": 2.0,
                                    "majority_class_rate": 0.6875,
                                },
                            },
                        }
                    }
                },
            }
        }

    assert adopt(summary_with(0.70), layer="filter")["trim"]["adopted"] is False
    assert adopt(summary_with(0.60), layer="filter")["trim"]["adopted"] is True
