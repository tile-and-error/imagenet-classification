from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, get_args

from torch import Tensor

# What each kind of event carries. Only `scalar` and `status` values are checked in
# `RunEvent.__post_init__`; the others are stored as given.
#   scalar    one number per step: `train/loss/total`, `eval/val/top1`
#   table     named columns of values, or a curve from `lwfm.results.curves`
#             such as `ROCCurve` (stored as its fields: `{"fpr": [...], ...}`)
#   artifact  the path of a file the run saved, e.g. a checkpoint `student.pt`
#   status    a run state change (see `RunStatus`), sent by `OutputRouter` on exit
#   message   free text such as a note or the traceback of a failed run
type EventKind = Literal["scalar", "table", "artifact", "status", "message"]
type RunStatus = Literal["running", "completed", "failed", "interrupted"]

KINDS: tuple[str, ...] = get_args(EventKind.__value__)
STATUSES: tuple[str, ...] = get_args(RunStatus.__value__)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class RunEvent:
    """One result record: a scalar, a table, an artifact, a status or a message.

    Names follow `train/loss/<binding>`, `train/loss/total` and
    `eval/<split>/<metric>`. A scalar may be given as a one-element tensor (as
    `LossComposer` returns); it is stored as a Python number. Other tensors are
    refused: save them to a file and emit an `artifact` event with its path.
    """

    kind: EventKind
    name: str
    value: object
    stage: str | None = None
    step: int | None = None
    timestamp_utc: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        # Check correct kind (datatype)
        if self.kind not in KINDS:
            raise ValueError(f"event {self.name!r}: kind {self.kind!r} not in {KINDS}")
        # Prevent saving Tensors through RunEvents
        if isinstance(self.value, Tensor):
            if self.kind != "scalar" or self.value.numel() != 1:
                raise TypeError(
                    f"event {self.name!r} holds a tensor of shape "
                    f"{tuple(self.value.shape)}; save it to a file and emit an "
                    "artifact event instead"
                )
            self.value = self.value.item()
        # Validate Status
        if self.kind == "status" and self.value not in STATUSES:
            raise ValueError(f"status {self.value!r} not in {STATUSES}")
