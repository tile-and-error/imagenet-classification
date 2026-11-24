import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torchmetrics import Metric

from lwfm.core.names import as_list, describe_values, resolve_all
from lwfm.core.types import Batch, Name


@dataclass
class MetricBinding:
    """Pairs a metric with the values it reads.

    `metric` is a torchmetrics metric.
    `inputs` lists the values passed to `update`, in argument order, e.g.
    `["student.logits", "targets.class"]` for `update(preds, target)`. Call
    `update(batch, node_outputs)` once per batch; `Evaluator` resets and computes.
    """

    name: str
    metric: Metric
    inputs: Name | Sequence[Name]

    def __post_init__(self) -> None:
        self.inputs = as_list(self.inputs)

    def update(self, batch: Batch, node_outputs: dict[Name, Tensor]) -> None:
        input_values = resolve_all(self.inputs, batch, node_outputs)
        try:
            self.metric.update(*input_values)
        except Exception as error:
            error.add_note(
                f"in metric {self.name!r}, reading "
                f"{describe_values(self.inputs, input_values)}"
            )
            raise


@dataclass
class CurveBinding:
    """Pairs a table builder with the values it reads, passed in order: e.g.
    `CurveBinding("roc", ROCCurve.from_scores, ["student.logits", "targets.class"])`
    calls `ROCCurve.from_scores(logits, classes)`.

    `build` returns a table, such as a curve from `lwfm.results.curves`; for named
    classes, pass `lambda s, y: ConfusionMatrix.from_scores(s, y, names)`. Unlike a
    metric, which keeps running totals, every batch's values are kept (on the CPU)
    and joined along the batch dimension before `build` runs once per split, so
    memory grows with the split: 50k ImageNet logits take 200 MB.
    """

    name: str
    build: Callable[..., object]
    inputs: Name | Sequence[Name]
    _batches: list[list[Tensor]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.inputs = as_list(self.inputs)
        try:  # a wrong number of inputs fails here, not after the first validation
            inspect.signature(self.build).bind(*self.inputs)
        except TypeError as error:
            raise ValueError(
                f"curve {self.name!r} reads {len(self.inputs)} values "
                f"{self.inputs}, which its build function can't take: {error}"
            ) from None

    def reset(self) -> None:
        self._batches = []

    def update(self, batch: Batch, node_outputs: dict[Name, Tensor]) -> None:
        values = resolve_all(self.inputs, batch, node_outputs)
        self._batches.append([value.detach().cpu() for value in values])

    def compute(self) -> object:
        values = [torch.cat(column) for column in zip(*self._batches, strict=True)]
        try:
            return self.build(*values)
        except Exception as error:
            error.add_note(
                f"in curve {self.name!r}, reading "
                f"{describe_values(self.inputs, values)}"
            )
            raise
