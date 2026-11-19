from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph
from lwfm.core.types import Batch
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink


@dataclass(frozen=True)
class EvaluationResult:
    metric_values: dict[str, float]


class Evaluator:
    """Scores a graph on a loader: reset each metric, run the graph in eval mode
    under `torch.inference_mode()`, update every metric per batch, compute once at
    the end. The graph's train/eval mode is restored afterwards.

    With an `output_sink`, each value is emitted as the scalar
    `eval/<split_name>/<metric name>` at `training_step`.
    """

    def evaluate(
        self,
        graph: ExecutionGraph,
        data_loader: Iterable[Batch],
        metric_bindings: Sequence[MetricBinding],
        *,
        device: torch.device,
        split_name: str = "val",
        output_sink: OutputSink | None = None,
        training_step: int | None = None,
    ) -> EvaluationResult:
        metric_names = [binding.name for binding in metric_bindings]
        if not metric_names:
            raise ValueError("evaluate needs at least one MetricBinding")
        if len(set(metric_names)) != len(metric_names):
            raise ValueError(f"metric names must be unique, got {metric_names}")

        for binding in metric_bindings:
            binding.metric.reset()
            binding.metric.to(device)

        # Save model training state
        was_training = graph.training
        graph.eval()
        batch_count = 0
        try:
            with torch.inference_mode():
                for batch in data_loader:
                    batch = to_device(batch, device)
                    node_outputs = graph(batch)
                    for binding in metric_bindings:
                        binding.update(batch, node_outputs)
                    batch_count += 1
        finally:
            # Set the model training mode (this does not actually commence training)
            graph.train(was_training)
        if batch_count == 0:
            raise ValueError("data_loader yielded no batches")

        metric_values = {
            binding.name: compute_single_value(binding) for binding in metric_bindings
        }

        # Write to Output Sinks
        if output_sink is not None:
            for metric_name, value in metric_values.items():
                output_sink.write(
                    RunEvent(
                        "scalar",
                        f"eval/{split_name}/{metric_name}",
                        value,
                        step=training_step,
                    )
                )
        return EvaluationResult(metric_values)


def compute_single_value(binding: MetricBinding) -> float:
    computed = binding.metric.compute()
    if computed.numel() != 1:
        raise TypeError(
            f"metric {binding.name!r} computed shape {tuple(computed.shape)}, "
            "expected one number"
        )
    return computed.item()
