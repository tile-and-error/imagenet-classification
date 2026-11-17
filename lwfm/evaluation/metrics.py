from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from lwfm.core.names import as_list, resolve_all
from lwfm.core.types import Batch, Name

if TYPE_CHECKING:  # torchmetrics is not a dependency yet; any .update(...) works
    from torchmetrics import Metric  # pyright: ignore[reportMissingImports]


@dataclass
class MetricBinding:
    """Pairs a metric with the values it reads.

    `metric` is anything with an `update` method, normally a torchmetrics metric.
    `reads` lists the values passed to `update`, in argument order, e.g.
    `["student.logits", "targets.class"]` for `update(preds, target)`. Call
    `update(batch, outputs)` once per batch; computing and resetting the metric
    is left to the caller.
    """

    name: str
    metric: Metric
    reads: Name | Sequence[Name]

    def __post_init__(self) -> None:
        self.reads = as_list(self.reads)

    def update(self, batch: Batch, outputs: dict[Name, Tensor]) -> None:
        self.metric.update(*resolve_all(self.reads, batch, outputs))
