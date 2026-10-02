"""Tests for waiting out a graphics card another process on xen1 is using."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from xinyenyana import a5


class _OutOfMemory(Exception):
    """Stands in for torch's own out-of-memory error."""


def _fake_torch(*, free: int = 1 << 30, total: int = 16 << 30) -> types.ModuleType:
    module = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        OutOfMemoryError=_OutOfMemory,
        empty_cache=lambda: emptied.append(True),
        mem_get_info=lambda: (free, total),
    )
    module.cuda = cuda  # type: ignore[attr-defined]
    return module


emptied: list[bool] = []


@pytest.fixture(autouse=True)
def _installed(monkeypatch: pytest.MonkeyPatch) -> None:
    emptied.clear()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setattr(a5, "SHARED_DEVICE_WAIT_SECONDS", 0.0)


def test_a_pass_that_fits_is_run_once() -> None:
    calls: list[int] = []

    def compute() -> str:
        calls.append(1)
        return "vector"

    assert a5.on_device(compute, what="a clip", seconds=1.0) == "vector"
    assert len(calls) == 1
    assert emptied == []


def test_a_pass_that_fits_on_a_later_attempt_returns_the_same_thing() -> None:
    calls: list[int] = []

    def compute() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _OutOfMemory("no room")
        return "vector"

    assert a5.on_device(compute, what="a clip", seconds=1.0) == "vector"
    assert len(calls) == 3
    assert len(emptied) == 2


def test_a_pass_that_never_fits_names_the_clip_its_length_and_the_card() -> None:
    """The message states what was measured and does not name a cause.

    It used to end "so another process on this machine is holding the rest".
    Five of the six failures on record were the clip exceeding what the model
    fits on this card at any moment, which waiting cannot change, and the
    message said the opposite. The length is what tells the two apart.
    """

    def compute() -> Any:
        raise _OutOfMemory("no room")

    with pytest.raises(RuntimeError) as raised:
        a5.on_device(compute, what="wavlm-large on owl-0004.wav", seconds=163.4)
    message = str(raised.value)
    assert "wavlm-large on owl-0004.wav" in message
    assert "163.4 seconds at 16 kHz" in message
    assert "1.00 GiB of 16.00 GiB free" in message
    assert "another process" not in message


def test_every_attempt_is_used_before_giving_up() -> None:
    calls: list[int] = []

    def compute() -> Any:
        calls.append(1)
        raise _OutOfMemory("no room")

    with pytest.raises(RuntimeError):
        a5.on_device(compute, what="a clip", seconds=1.0)
    assert len(calls) == a5.SHARED_DEVICE_ATTEMPTS


def test_an_error_that_is_not_about_memory_is_not_retried() -> None:
    calls: list[int] = []

    def compute() -> Any:
        calls.append(1)
        raise ValueError("the clip is empty")

    with pytest.raises(ValueError, match="the clip is empty"):
        a5.on_device(compute, what="a clip", seconds=1.0)
    assert len(calls) == 1


def test_the_failed_pass_is_released_before_the_cache_is_emptied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emptying the cache inside the except block cannot free what the traceback holds."""

    import gc
    import weakref

    class Activations:
        pass

    alive_at_empty: list[bool] = []
    handle: list[Any] = []

    def empty() -> None:
        gc.collect()
        alive_at_empty.append(handle[-1]() is not None)

    torch = _fake_torch()
    torch.cuda.empty_cache = empty  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls: list[int] = []

    def compute() -> str:
        calls.append(1)
        held = Activations()
        handle.append(weakref.ref(held))
        if len(calls) < 2:
            raise _OutOfMemory("no room")
        return "vector"

    assert a5.on_device(compute, what="a clip", seconds=1.0) == "vector"
    assert alive_at_empty == [False]
