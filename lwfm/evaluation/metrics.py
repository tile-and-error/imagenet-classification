from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from lwfm.core.ports import resolve_all
from lwfm.core.types import Batch, Port

if TYPE_CHECKING:  # torchmetrics is not a dependency yet; any .update(**kw) works
    from torchmetrics import Metric


@dataclass
class MetricBinding:
    name: str
    metric: Metric
    inputs: dict[str, Port]  # metric.update argument -> port

    def update(self, batch: Batch, outputs: dict[Port, Tensor]) -> None:
        self.metric.update(**resolve_all(self.inputs, batch, outputs))
