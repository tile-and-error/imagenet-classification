from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph
from lwfm.core.types import Batch
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.events import EventSink, RunEvent


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]


class Evaluator:
    """Scores a graph on a loader: reset each metric, run the graph in eval mode
    under `torch.inference_mode()`, update every metric per batch, compute once at
    the end. The graph's train/eval mode is restored afterwards.

    With an `event_sink`, each value is emitted as the scalar `eval/<split>/<name>`.
    """

    def evaluate(
        self,
        graph: ExecutionGraph,
        loader: Iterable[Batch],
        metrics: Sequence[MetricBinding],
        *,
        device: torch.device,
        split: str = "val",
        event_sink: EventSink | None = None,
        step: int | None = None,
    ) -> EvaluationResult:
        names = [m.name for m in metrics]
        if not names:
            raise ValueError("evaluate needs at least one MetricBinding")
        if len(set(names)) != len(names):
            raise ValueError(f"metric names must be unique, got {names}")

        for binding in metrics:
            binding.metric.reset()
            binding.metric.to(device)

        was_training = graph.training
        graph.eval()
        batches = 0
        try:
            with torch.inference_mode():
                for batch in loader:
                    batch = to_device(batch, device)
                    outputs = graph(batch)
                    for binding in metrics:
                        binding.update(batch, outputs)
                    batches += 1
        finally:
            graph.train(was_training)
        if batches == 0:
            raise ValueError("loader yielded no batches")

        values = {m.name: scalar(m) for m in metrics}
        if event_sink is not None:
            for name, value in values.items():
                event_sink.emit(
                    RunEvent("scalar", f"eval/{split}/{name}", value, step=step)
                )
        return EvaluationResult(values)


def scalar(binding: MetricBinding) -> float:
    value = binding.metric.compute()
    if value.numel() != 1:
        raise TypeError(
            f"metric {binding.name!r} computed shape {tuple(value.shape)}, "
            "expected one number"
        )
    return value.item()
