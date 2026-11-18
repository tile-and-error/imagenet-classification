from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor
from torchmetrics import Metric

from lwfm.core.names import as_list, resolve_all
from lwfm.core.types import Batch, Name


@dataclass
class MetricBinding:
    """Pairs a metric with the values it reads.

    `metric` is a torchmetrics metric.
    `reads` lists the values passed to `update`, in argument order, e.g.
    `["student.logits", "targets.class"]` for `update(preds, target)`. Call
    `update(batch, outputs)` once per batch; `Evaluator` resets and computes.
    """

    name: str
    metric: Metric
    reads: Name | Sequence[Name]

    def __post_init__(self) -> None:
        self.reads = as_list(self.reads)

    def update(self, batch: Batch, outputs: dict[Name, Tensor]) -> None:
        self.metric.update(*resolve_all(self.reads, batch, outputs))
