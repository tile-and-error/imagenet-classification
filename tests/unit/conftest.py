"""Shared test helpers.

pytest finds the fixtures here by itself. Import the rest by name:
`from conftest import CE_READS, ListSink, setup`.
"""

import pytest
import torch
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from lwfm.core.graph import ExecutionGraph, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.sinks import OutputSink

K = 3  # classes in the training tests
CE_READS = ["student.logits", "targets.class"]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ListSink(OutputSink):
    """Keeps every event in memory. With `log`, also appends `(sink, name)` to it, to
    check the order events reach several sinks in."""

    def __init__(self, log=None):
        self.events = []
        self.closed = False
        self.log = log

    def write(self, event):
        self.events.append(event)
        if self.log is not None:
            self.log.append((self, event.name))

    def close(self):
        self.closed = True

    def steps_of(self, name):
        return [event.step for event in self.events if event.name == name]


@pytest.fixture
def sink():
    return ListSink()


def batches(count, batch_size=4, seed=0):
    """`count` seeded batches of random 3x2x2 images with labels in [0, K), plus a
    constant `targets.one` for a metric that never changes."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "inputs": {"image": torch.rand(batch_size, 3, 2, 2, generator=generator)},
            "targets": {
                "class": torch.randint(0, K, (batch_size,), generator=generator),
                "one": torch.ones(batch_size),
            },
        }
        for _ in range(count)
    ]


def classifier():
    """The student's architecture: a linear classifier on flattened 3x2x2 images."""
    return nn.Sequential(nn.Flatten(), nn.Linear(12, K))


def student(module=None):
    """A student node (`classifier()` unless `module` is given): reads
    `inputs.image`, writes `student.logits`."""
    return ModelNode(
        "student", module or classifier(), inputs="inputs.image", outputs="logits"
    )


def setup(lr=0.1, weight=1.0, seed=0, module=None):
    """A seeded student, its one-node graph, a CE loss with `weight`, and SGD over
    the student."""
    torch.manual_seed(seed)
    node = student(module)
    loss = LossComposer(
        [LossBinding("ce", nn.CrossEntropyLoss(), CE_READS, weight=weight)]
    )
    return node, ExecutionGraph([node]), loss, torch.optim.SGD(node.parameters(), lr=lr)


def accuracy():
    """Micro top-1 accuracy on the student's logits, as a one-metric list."""
    return [MetricBinding("top1", MulticlassAccuracy(K, average="micro"), CE_READS)]


def constant_metric():
    """A validation metric that is always 1.0, so it never improves after the first."""
    return [MetricBinding("const", MeanMetric(), ["targets.one"])]


def is_png(path):
    return path.read_bytes().startswith(PNG_SIGNATURE)
