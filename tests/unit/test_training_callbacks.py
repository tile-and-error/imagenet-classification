import copy
import json

import pytest
import torch
from torch import nn
from torchmetrics import MeanMetric

from lwfm.core.graph import ExecutionGraph, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import LocalFileSink, OutputSink
from lwfm.training import Callback, Trainer, TrainingResult, TrainingState
from lwfm.training.callbacks import (
    EarlyStopping,
    EMAUpdate,
    LossWeightSchedule,
    ModelCheckpoint,
)

K = 3
CE_READS = ["student.logits", "targets.class"]


class ListSink(OutputSink):
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def close(self):
        pass


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


def make_module():
    """The student's architecture: a linear classifier on flattened 3x2x2 images."""
    return nn.Sequential(nn.Flatten(), nn.Linear(12, K))


def setup(weight=1.0):
    """A seeded student, its graph, a CE loss with this `weight`, and SGD."""
    torch.manual_seed(0)
    node = ModelNode("student", make_module(), inputs="inputs.image", outputs="logits")
    loss = LossComposer(
        [LossBinding("ce", nn.CrossEntropyLoss(), CE_READS, weight=weight)]
    )
    optimizer = torch.optim.SGD(node.parameters(), lr=0.1)
    return node, ExecutionGraph([node]), loss, optimizer


def constant_metric():
    """A validation metric that is always 1.0, so it never improves after the first."""
    return [MetricBinding("const", MeanMetric(), ["targets.one"])]


def bare_state(graph, loss, optimizer, sink=None, metric_names=("top1",)):
    """A TrainingState for calling a callback's hooks directly, without a training
    loop."""
    return TrainingState(
        graph=graph,
        loss=loss,
        optimizer=optimizer,
        device=torch.device("cpu"),
        output_sink=sink,
        metric_names=metric_names,
    )


# --- LossWeightSchedule ---------------------------------------------------------


def test_loss_weight_schedule_sets_the_weight_every_step():
    node, graph, loss, optimizer = setup(weight=0.0)
    seen = []

    class Watch(Callback):  # runs after the schedule, so it sees the new weight
        def on_step_start(self, state, batch):
            seen.append(state.loss.loss_bindings[0].weight)

    ramp = LossWeightSchedule("ce", lambda step: min(1.0, step / 4))
    Trainer(device="cpu", max_epochs=1).fit(
        graph, batches(5), loss, optimizer, callbacks=[ramp, Watch()]
    )
    assert seen == [0.0, 0.25, 0.5, 0.75, 1.0]


def test_loss_weight_schedule_rejects_an_unknown_binding():
    _, graph, loss, optimizer = setup()
    with pytest.raises(ValueError, match="no loss binding named 'kd'"):
        Trainer(device="cpu", max_steps=1).fit(
            graph,
            batches(1),
            loss,
            optimizer,
            callbacks=[LossWeightSchedule("kd", lambda step: 1.0)],
        )


# --- EarlyStopping --------------------------------------------------------------


def test_early_stopping_after_patience_validations_without_improvement():
    # The metric never changes: validation 1 sets the best, 2 and 3 don't improve
    _, graph, loss, optimizer = setup()
    result = Trainer(device="cpu", max_epochs=10).fit(
        graph,
        batches(2),
        loss,
        optimizer,
        val_loader=batches(1),
        metrics=constant_metric(),
        callbacks=[EarlyStopping("const", patience=2)],
    )
    assert (result.stop_reason, result.epochs_completed) == ("stopped", 3)


def test_early_stopping_counts_from_the_last_improvement():
    _, graph, loss, optimizer = setup()
    stopper = EarlyStopping("top1", patience=2, min_delta=0.05)
    state = bare_state(graph, loss, optimizer)
    stopper.on_fit_start(state)
    for value in (0.5, 0.52, 0.6, 0.61):  # 0.52 and 0.61 are within min_delta
        stopper.on_validation_end(state, {"top1": value})
    assert not state.should_stop
    stopper.on_validation_end(state, {"top1": 0.6})
    assert state.should_stop


@pytest.mark.parametrize("with_validation", [True, False])
def test_monitor_must_be_a_validation_metric(with_validation):
    _, graph, loss, optimizer = setup()
    validation = (
        {"val_loader": batches(1), "metrics": constant_metric()}
        if with_validation
        else {}
    )
    with pytest.raises(ValueError, match="monitors 'top1'"):
        Trainer(device="cpu", max_steps=1).fit(
            graph,
            batches(1),
            loss,
            optimizer,
            callbacks=[EarlyStopping("top1")],
            **validation,
        )


# --- ModelCheckpoint ------------------------------------------------------------


def test_checkpoint_saves_best_only_on_improvement(tmp_path):
    node, graph, loss, optimizer = setup()
    sink = ListSink()
    checkpoint = ModelCheckpoint(tmp_path, node="student", monitor="top1")
    state = bare_state(graph, loss, optimizer, sink)
    checkpoint.on_fit_start(state)
    for step, value in [(1, 0.5), (2, 0.4), (3, 0.6)]:
        state.global_step = step
        checkpoint.on_validation_end(state, {"top1": value})

    saves = [(e.name, str(e.value), e.step) for e in sink.events]
    assert saves == [
        ("checkpoint/best", "checkpoints/best.pt", 1),  # relative to the run dir
        ("checkpoint/best", "checkpoints/best.pt", 3),
    ]
    # best.pt is the deployable module alone, and loads strictly into a fresh one
    weights = torch.load(tmp_path / "checkpoints/best.pt", weights_only=True)
    make_module().load_state_dict(weights)


def test_checkpoint_last_holds_the_full_training_state(tmp_path):
    _, graph, loss, optimizer = setup()
    checkpoint = ModelCheckpoint(tmp_path)  # no monitor: last.pt only
    state = bare_state(graph, loss, optimizer)
    state.global_step, state.epoch = 7, 2
    checkpoint.on_fit_start(state)
    checkpoint.on_validation_end(state, {"top1": 1.0})
    checkpoint.on_fit_end(state, TrainingResult(7, 2, "max_epochs", []))

    assert not (tmp_path / "checkpoints/best.pt").exists()
    last = torch.load(tmp_path / "checkpoints/last.pt", weights_only=True)
    assert set(last) == {
        "graph",
        "loss",
        "optimizer",
        "scheduler",
        "scaler",
        "global_step",
        "epoch",
    }
    assert (last["global_step"], last["epoch"]) == (7, 2)


def test_checkpoint_in_a_real_run_records_artifacts(tmp_path):
    _, graph, loss, optimizer = setup()
    with OutputRouter(LocalFileSink(tmp_path)) as sink:
        Trainer(device="cpu", max_epochs=2).fit(
            graph,
            batches(2),
            loss,
            optimizer,
            val_loader=batches(1),
            metrics=constant_metric(),
            callbacks=[ModelCheckpoint(tmp_path, monitor="const")],
            output_sink=sink,
        )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").open()]
    artifacts = [(e["name"], e["value"]) for e in events if e["kind"] == "artifact"]
    # The constant metric improves once (the first validation), then never again
    assert artifacts == [
        ("checkpoint/best", "checkpoints/best.pt"),
        ("checkpoint/last", "checkpoints/last.pt"),
    ]


def test_checkpoint_rejects_an_unknown_node(tmp_path):
    _, graph, loss, optimizer = setup()
    with pytest.raises(ValueError, match="'teacher'"):
        ModelCheckpoint(tmp_path, node="teacher").on_fit_start(
            bare_state(graph, loss, optimizer)
        )


# --- EMAUpdate ------------------------------------------------------------------


def ema_graph():
    """A student and a frozen copy of it as teacher, with the teacher's weights set
    to 1 so an EMA update visibly moves them."""
    torch.manual_seed(0)
    module = nn.Sequential(nn.Flatten(), nn.BatchNorm1d(12), nn.Linear(12, K))
    student = ModelNode("student", module, inputs="inputs.image", outputs="logits")
    teacher = ModelNode(
        "teacher",
        copy.deepcopy(module),
        inputs="inputs.image",
        outputs="logits",
        frozen=True,
    )
    with torch.no_grad():  # start the teacher somewhere else, to see it move
        for p in teacher.parameters():
            p.fill_(1.0)
    return student, teacher, ExecutionGraph([student, teacher])


def test_ema_moves_the_target_towards_the_source():
    student, teacher, graph = ema_graph()
    loss = LossComposer([LossBinding("ce", nn.CrossEntropyLoss(), CE_READS)])
    expected = []

    class Expect(Callback):  # registered first, so it sees the weights before EMA
        def on_after_optimizer_step(self, state):
            expected.append(
                [
                    0.9 * t + 0.1 * s
                    for t, s in zip(
                        teacher.parameters(), student.parameters(), strict=True
                    )
                ]
            )

    Trainer(device="cpu", max_steps=1).fit(
        graph,
        batches(1),
        loss,
        torch.optim.SGD(student.parameters(), lr=0.1),
        callbacks=[Expect(), EMAUpdate("student", "teacher", momentum=0.9)],
    )
    for want, got in zip(expected[0], teacher.parameters(), strict=True):
        torch.testing.assert_close(got, want)
    for b_teacher, b_student in zip(teacher.buffers(), student.buffers(), strict=True):
        assert torch.equal(b_teacher, b_student)  # BatchNorm statistics copied


def test_ema_rejects_different_architectures():
    student = ModelNode("student", make_module(), inputs="inputs.image", outputs="z")
    other = ModelNode(
        "teacher",
        nn.Sequential(nn.Flatten(), nn.Linear(12, 5)),
        inputs="inputs.image",
        outputs="z",
        frozen=True,
    )
    graph = ExecutionGraph([student, other])
    state = bare_state(graph, None, None)
    with pytest.raises(ValueError, match="same architecture"):
        EMAUpdate("student", "teacher").on_fit_start(state)


def test_ema_rejects_a_bad_momentum():
    with pytest.raises(ValueError, match="momentum"):
        EMAUpdate("student", "teacher", momentum=1.5)
