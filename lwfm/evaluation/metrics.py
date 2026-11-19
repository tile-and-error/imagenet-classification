from collections.abc import Sequence
from dataclasses import dataclass

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
