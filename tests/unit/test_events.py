from datetime import datetime, timedelta

import pytest
import torch
from torch import Tensor

from lwfm.results.events import EventKind, RunEvent


def test_timestamp_is_set_in_utc() -> None:
    stamp = datetime.fromisoformat(RunEvent("message", "hello", "hi").timestamp_utc)
    assert stamp.utcoffset() == timedelta(0)


def test_one_element_tensor_scalar_becomes_a_number() -> None:
    event = RunEvent("scalar", "train/loss/ce", torch.tensor(0.25))
    assert event.value == 0.25
    assert type(event.value) is float


@pytest.mark.parametrize(
    ("kind", "value"),
    [("scalar", torch.ones(2)), ("table", torch.tensor(1.0))],
    ids=["two_element_scalar", "tensor_in_table"],
)
def test_other_tensors_are_refused(kind: EventKind, value: Tensor) -> None:
    with pytest.raises(TypeError, match="logits.*artifact event"):
        RunEvent(kind, "logits", value)


def test_unknown_kind_fails() -> None:
    with pytest.raises(ValueError, match="kind 'metric'"):
        RunEvent("metric", "top1", 0.5)  # type: ignore[arg-type]


def test_unknown_status_fails() -> None:
    with pytest.raises(ValueError, match="status 'done'"):
        RunEvent("status", "status", "done")
