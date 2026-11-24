from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph
from lwfm.core.types import Batch
from lwfm.evaluation.metrics import CurveBinding, MetricBinding
from lwfm.results.curves import Curve
from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink


@dataclass(frozen=True)
class EvaluationResult:
    metric_values: dict[str, float]
    curves: dict[str, Curve] = field(default_factory=dict)


class Evaluator:
    """Scores a graph on a loader: reset each metric, run the graph in eval mode
    under `torch.inference_mode()`, update every metric per batch, compute once at
    the end. The graph's train/eval mode is restored afterwards.

    With an `output_sink`, each value is emitted as the scalar
    `eval/<split_name>/<metric name>` at `training_step`, and each curve, built
    from the whole split, as the table `eval/<split_name>/<curve name>`.
    """

    def evaluate(
        self,
        graph: ExecutionGraph,
        data_loader: Iterable[Batch],
        metric_bindings: Sequence[MetricBinding],
        *,
        curve_bindings: Sequence[CurveBinding] = (),
        device: torch.device,
        split_name: str = "val",
        output_sink: OutputSink | None = None,
        training_step: int | None = None,
    ) -> EvaluationResult:
        if not metric_bindings and not curve_bindings:
            raise ValueError(
                "evaluate needs at least one MetricBinding or CurveBinding"
            )
        bindings = [*metric_bindings, *curve_bindings]
        names = [binding.name for binding in bindings]
        if len(set(names)) != len(names):
            raise ValueError(f"metric and curve names must be unique, got {names}")
        graph.check_reads(name for binding in bindings for name in binding.inputs)

        for binding in metric_bindings:
            binding.metric.reset()
            binding.metric.to(device)
        for binding in curve_bindings:
            binding.reset()

        # Save model training state
        was_training = graph.training
        graph.eval()
        batch_count = 0
        try:
            with torch.inference_mode():
                for batch in data_loader:
                    batch = to_device(batch, device)
                    node_outputs = graph(batch)
                    for binding in bindings:
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
        curves = {binding.name: binding.compute() for binding in curve_bindings}

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
            for curve_name, table in curves.items():
                output_sink.write(
                    RunEvent(
                        "table",
                        f"eval/{split_name}/{curve_name}",
                        table,
                        step=training_step,
                    )
                )
        return EvaluationResult(metric_values, curves)


def compute_single_value(binding: MetricBinding) -> float:
    computed = binding.metric.compute()
    if computed.numel() != 1:
        raise TypeError(
            f"metric {binding.name!r} computed shape {tuple(computed.shape)}, "
            "expected one number"
        )
    return computed.item()
