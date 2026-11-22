import pytest
import torch
from torch import nn
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy

from lwfm.core.graph import ExecutionGraph, GraphError, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.sinks import OutputSink
from lwfm.training import Callback, Trainer, TrainingResult, seed_everything
from lwfm.training.callbacks import HOOKS

K = 3
CE_READS = ["student.logits", "targets.class"]


class ListSink(OutputSink):
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)

    def close(self):
        pass

    def steps_of(self, name):
        return [event.step for event in self.events if event.name == name]


class HookRecorder(Callback):
    """Records every hook call, in order."""

    def __init__(self):
        self.calls = []

    def __getattribute__(self, name):
        if name in HOOKS:
            return lambda state, *args: self.calls.append(name)
        return super().__getattribute__(name)


def batches(count, batch_size=4, seed=0):
    """`count` seeded batches of random 3x2x2 images with labels in [0, K), plus a
    constant `targets.one` for a metric that never changes."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "inputs": {"image": torch.rand(batch_size, 3, 2, 2, generator=generator)},
            "targets": {
                "class": torch.randint(0, K, (batch_size,), generator=generator),
                "one": torch.ones(
                    batch_size
                ),  # a constant, for a metric that never moves
            },
        }
        for _ in range(count)
    ]


def student():
    """A fresh linear student node: reads `inputs.image`, writes `student.logits`."""
    return ModelNode(
        "student",
        nn.Sequential(nn.Flatten(), nn.Linear(12, K)),
        inputs="inputs.image",
        outputs="logits",
    )


def setup(lr=0.1, seed=0):
    """A seeded student, its one-node graph, a CE loss and SGD over the student."""
    torch.manual_seed(seed)
    node = student()
    graph = ExecutionGraph([node])
    loss = LossComposer([LossBinding("ce", nn.CrossEntropyLoss(), CE_READS)])
    return node, graph, loss, torch.optim.SGD(node.parameters(), lr=lr)


def accuracy():
    """Micro top-1 accuracy on the student's logits, as a one-metric list."""
    return [MetricBinding("top1", MulticlassAccuracy(K, average="micro"), CE_READS)]


def constant_metric():
    """A validation metric that is always 1.0, so it never improves after the first."""
    return [MetricBinding("const", MeanMetric(), ["targets.one"])]


# --- training ------------------------------------------------------------------


def test_overfits_one_batch():
    # PLAN-style sanity check: the loop must be able to drive the loss towards 0
    _, graph, loss, optimizer = setup(lr=0.5)
    one = batches(1) * 200
    first = loss(one[0], graph(one[0]))[0].item()
    result = Trainer(device="cpu", max_steps=200).fit(graph, one, loss, optimizer)
    assert result.stop_reason == "max_steps"
    assert loss(one[0], graph(one[0]))[0].item() < 0.1 * first


def test_matches_hand_written_loop_exactly():
    # The trainer adds bookkeeping, never different maths
    data = batches(6)
    node_a, graph_a, loss_a, opt_a = setup()
    Trainer(device="cpu", max_epochs=2).fit(graph_a, data, loss_a, opt_a)

    node_b, graph_b, loss_b, opt_b = setup()
    graph_b.train()
    for _ in range(2):
        for batch in data:
            total, _ = loss_b(batch, graph_b(batch))
            opt_b.zero_grad()
            total.backward()
            opt_b.step()

    for a, b in zip(node_a.parameters(), node_b.parameters(), strict=True):
        assert torch.equal(a, b)


def test_counts_epochs_validates_each_epoch_and_logs():
    _, graph, loss, optimizer = setup()
    sink = ListSink()
    result = Trainer(device="cpu", max_epochs=3, log_every_n_steps=5).fit(
        graph,
        batches(5),
        loss,
        optimizer,
        val_loader=batches(2, seed=1),
        metrics=accuracy(),
        output_sink=sink,
    )
    assert (result.global_step, result.epochs_completed) == (15, 3)
    assert result.stop_reason == "max_epochs"
    assert [step for step, _ in result.history] == [5, 10, 15]
    for name in ("train/loss/ce", "train/loss/total", "train/lr"):
        assert sink.steps_of(name) == [5, 10, 15]
    assert sink.steps_of("train/epoch_seconds") == [5, 10, 15]
    assert sink.steps_of("eval/val/top1") == [5, 10, 15]


def test_hooks_fire_in_order():
    _, graph, loss, optimizer = setup()
    recorder = HookRecorder()
    Trainer(device="cpu", max_epochs=1).fit(
        graph,
        batches(2),
        loss,
        optimizer,
        val_loader=batches(1),
        metrics=accuracy(),
        callbacks=[recorder],
    )
    step = [
        "on_step_start",
        "on_after_backward",
        "on_after_optimizer_step",
        "on_step_end",
    ]
    assert recorder.calls == [
        "on_fit_start",
        "on_epoch_start",
        *step,
        *step,
        "on_epoch_end",
        "on_validation_end",  # after on_epoch_end
        "on_fit_end",
    ]


def test_max_steps_on_an_epoch_boundary_counts_the_epoch():
    # 10 steps over 5-batch epochs is exactly two full epochs
    _, graph, loss, optimizer = setup()
    recorder = HookRecorder()
    result = Trainer(device="cpu", max_steps=10).fit(
        graph, batches(5), loss, optimizer, callbacks=[recorder]
    )
    assert (result.global_step, result.epochs_completed) == (10, 2)
    assert recorder.calls.count("on_epoch_end") == 2


def test_stopping_mid_epoch_skips_the_epoch_but_still_validates():
    _, graph, loss, optimizer = setup()
    sink = ListSink()
    result = Trainer(device="cpu", max_steps=7).fit(
        graph,
        batches(5),
        loss,
        optimizer,
        val_loader=batches(1),
        metrics=accuracy(),
        output_sink=sink,
    )
    assert (result.global_step, result.epochs_completed) == (7, 1)
    assert [step for step, _ in result.history] == [5, 7]  # epoch 1, then the stop
    assert sink.steps_of("train/epoch_seconds") == [5]  # the partial epoch isn't timed


def test_val_every_n_steps_and_a_final_validation():
    _, graph, loss, optimizer = setup()
    result = Trainer(device="cpu", max_epochs=1, val_every_n_steps=2).fit(
        graph, batches(5), loss, optimizer, val_loader=batches(1), metrics=accuracy()
    )
    assert [step for step, _ in result.history] == [2, 4, 5]


def test_frozen_teacher_stays_in_eval_and_unchanged():
    node = student()
    teacher = ModelNode(
        "teacher",
        nn.Sequential(nn.Flatten(), nn.Dropout(0.5), nn.Linear(12, K)),
        inputs="inputs.image",
        outputs="logits",
        frozen=True,
    )
    graph = ExecutionGraph([node, teacher])
    loss = LossComposer(
        [
            LossBinding("ce", nn.CrossEntropyLoss(), CE_READS),
            LossBinding("kd", nn.MSELoss(), ["student.logits", "teacher.logits"]),
        ]
    )
    before = [p.clone() for p in teacher.parameters()]

    class CheckModes(Callback):
        def on_step_start(self, state, batch):
            assert state.graph.nodes["student"].training
            assert not state.graph.nodes["teacher"].training

    Trainer(device="cpu", max_steps=3).fit(
        graph,
        batches(3),
        loss,
        torch.optim.SGD(node.parameters(), lr=0.1),
        callbacks=[CheckModes()],
    )
    for a, b in zip(before, teacher.parameters(), strict=True):
        assert torch.equal(a, b)
        assert b.grad is None


def test_lr_is_logged_per_param_group():
    node, graph, loss, _ = setup()
    weight, bias = node.parameters()
    optimizer = torch.optim.SGD(
        [{"params": [weight]}, {"params": [bias], "lr": 0.01}], lr=0.1
    )
    sink = ListSink()
    Trainer(device="cpu", max_steps=1, log_every_n_steps=1).fit(
        graph, batches(1), loss, optimizer, output_sink=sink
    )
    assert {e.name: e.value for e in sink.events if "lr" in e.name} == {
        "train/lr/0": 0.1,
        "train/lr/1": 0.01,
    }


# --- schedulers -----------------------------------------------------------------


def test_scheduler_stepped_per_step():
    _, graph, loss, optimizer = setup()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    Trainer(device="cpu", max_epochs=2).fit(
        graph,
        batches(5),
        loss,
        optimizer,
        scheduler=scheduler,
        scheduler_interval="step",
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_scheduler_stepped_per_epoch():
    _, graph, loss, optimizer = setup(lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    Trainer(device="cpu", max_epochs=2).fit(
        graph,
        batches(3),
        loss,
        optimizer,
        scheduler=scheduler,
        scheduler_interval="epoch",
    )
    assert optimizer.param_groups[0]["lr"] == 0.25


def test_plateau_scheduler_steps_from_a_callback():
    # ReduceLROnPlateau needs a metric, so it is stepped from a callback rather than
    # by the trainer. The metric never changes, so with patience 0 every validation
    # after the first halves the learning rate.
    class Plateau(Callback):
        def __init__(self, scheduler, monitored_metric):
            self.scheduler, self.monitored_metric = scheduler, monitored_metric

        def on_validation_end(self, state, metrics):
            self.scheduler.step(metrics[self.monitored_metric])

    _, graph, loss, optimizer = setup(lr=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=0
    )
    Trainer(device="cpu", max_epochs=3).fit(
        graph,
        batches(2),
        loss,
        optimizer,
        val_loader=batches(1),
        metrics=constant_metric(),
        callbacks=[Plateau(scheduler, "const")],
    )
    assert optimizer.param_groups[0]["lr"] == 0.25


@pytest.mark.parametrize(
    ("make_scheduler", "kwargs", "message"),
    [
        (lambda o: torch.optim.lr_scheduler.StepLR(o, 1), {}, "scheduler_interval"),
        (
            lambda o: torch.optim.lr_scheduler.ReduceLROnPlateau(o),
            {"scheduler_interval": "epoch"},
            "on_validation_end",
        ),
        (
            lambda o: torch.optim.lr_scheduler.StepLR(
                torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1), 1
            ),
            {"scheduler_interval": "step"},
            "different optimizer",
        ),
    ],
)
def test_scheduler_misuse_fails_before_training(make_scheduler, kwargs, message):
    _, graph, loss, optimizer = setup()
    with pytest.raises(ValueError, match=message):
        Trainer(device="cpu", max_steps=1).fit(
            graph,
            batches(1),
            loss,
            optimizer,
            val_loader=batches(1),
            metrics=constant_metric(),
            scheduler=make_scheduler(optimizer),
            **kwargs,
        )


def test_scheduler_interval_without_a_scheduler_fails():
    _, graph, loss, optimizer = setup()
    with pytest.raises(ValueError, match="without a scheduler"):
        Trainer(device="cpu", max_steps=1).fit(
            graph, batches(1), loss, optimizer, scheduler_interval="step"
        )


# --- checks before training -----------------------------------------------------


def test_misspelled_binding_name_fails_before_training():
    node, graph, _, optimizer = setup()
    bad = LossComposer(
        [LossBinding("ce", nn.CrossEntropyLoss(), ["student.logit", "targets.class"])]
    )
    with pytest.raises(GraphError, match="student.logit"):
        Trainer(device="cpu", max_steps=1).fit(graph, batches(1), bad, optimizer)


def test_misspelled_metric_name_fails_before_training():
    _, graph, loss, optimizer = setup()
    # "student.logit" is a deliberate typo for "student.logits"
    typo = [
        MetricBinding(
            "top1",
            MulticlassAccuracy(K, average="micro"),
            ["student.logit", "targets.class"],
        )
    ]
    recorder = HookRecorder()
    with pytest.raises(GraphError, match="student.logit"):
        Trainer(device="cpu", max_epochs=1).fit(
            graph,
            batches(1),
            loss,
            optimizer,
            val_loader=batches(1),
            metrics=typo,
            callbacks=[recorder],
        )
    # Not even on_fit_start ran, so it failed before the first batch, not at the
    # first validation an epoch later
    assert recorder.calls == []


def test_trainable_node_missing_from_optimizer_is_named():
    node = student()
    projector = ModelNode(
        "projector", nn.Linear(K, K), inputs="student.logits", outputs="z"
    )
    graph = ExecutionGraph([node, projector])
    loss = LossComposer(
        [LossBinding("ce", nn.CrossEntropyLoss(), ["projector.z", "targets.class"])]
    )
    with pytest.raises(ValueError, match="projector"):
        Trainer(device="cpu", max_steps=1).fit(
            graph, batches(1), loss, torch.optim.SGD(node.parameters(), lr=0.1)
        )


def test_trainable_loss_parameter_missing_from_optimizer_is_named():
    class TemperatureCE(nn.Module):  # a loss with its own learnable parameter
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(()))

        def forward(self, logits, target):
            return nn.functional.cross_entropy(logits / self.temperature, target)

    node, graph, _, optimizer = setup()  # the optimizer holds the student only
    loss = LossComposer([LossBinding("ce", TemperatureCE(), CE_READS)])
    with pytest.raises(ValueError, match="temperature"):
        Trainer(device="cpu", max_steps=1).fit(graph, batches(1), loss, optimizer)


def test_val_loader_and_metrics_go_together():
    _, graph, loss, optimizer = setup()
    trainer = Trainer(device="cpu", max_steps=1)
    with pytest.raises(ValueError, match="together"):
        trainer.fit(graph, batches(1), loss, optimizer, val_loader=batches(1))
    with pytest.raises(ValueError, match="together"):
        trainer.fit(graph, batches(1), loss, optimizer, metrics=accuracy())


def test_empty_loader_fails():
    _, graph, loss, optimizer = setup()
    with pytest.raises(ValueError, match="no batches"):
        Trainer(device="cpu", max_epochs=1).fit(graph, [], loss, optimizer)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "max_epochs, max_steps"),
        ({"max_epochs": 0}, "max_epochs"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": 1, "val_every_n_steps": 0}, "val_every_n_steps"),
        ({"max_steps": 1, "log_every_n_steps": 0}, "log_every_n_steps"),
        ({"max_steps": 1, "accumulate_grad_batches": 0}, "accumulate_grad_batches"),
        ({"max_steps": 1, "precision": "fp8"}, "precision"),
        ({"max_steps": 1, "gradient_clip_norm": 0.0}, "gradient_clip_norm"),
        ({"max_steps": 1, "gradient_clip_norm": -1.0}, "gradient_clip_norm"),
    ],
)
def test_invalid_trainer_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Trainer(device="cpu", **kwargs)


def test_misspelled_hook_fails_before_training():
    class Typo(Callback):
        def on_validation_ended(self, state, metrics): ...

    _, graph, loss, optimizer = setup()
    with pytest.raises(ValueError, match="on_validation_ended"):
        Trainer(device="cpu", max_steps=1).fit(
            graph, batches(1), loss, optimizer, callbacks=[Typo()]
        )


def test_non_callback_is_rejected():
    _, graph, loss, optimizer = setup()
    with pytest.raises(TypeError, match="not a Callback"):
        Trainer(device="cpu", max_steps=1).fit(
            graph,
            batches(1),
            loss,
            optimizer,
            callbacks=[object()],  # type: ignore[list-item]  # deliberately wrong
        )


# --- stopping -------------------------------------------------------------------


def test_callback_can_stop_training():
    class StopAfterFirstValidation(Callback):
        def on_validation_end(self, state, metrics):
            state.should_stop = True

    _, graph, loss, optimizer = setup()
    result = Trainer(device="cpu", max_epochs=10).fit(
        graph,
        batches(3),
        loss,
        optimizer,
        val_loader=batches(1),
        metrics=accuracy(),
        callbacks=[StopAfterFirstValidation()],
    )
    assert (result.stop_reason, result.epochs_completed) == ("stopped", 1)


@pytest.mark.parametrize(
    ("error", "reason"),
    [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")],
    ids=["ctrl_c", "crash"],
)
def test_early_exit_runs_on_fit_end_then_reraises(error, reason):
    ended = []

    class Exit(Callback):
        def on_step_end(self, state, output):
            if state.global_step == 2:
                raise error

        def on_fit_end(self, state, result):
            ended.append(result.stop_reason)

    _, graph, loss, optimizer = setup()
    with pytest.raises(error):
        Trainer(device="cpu", max_epochs=1).fit(
            graph, batches(5), loss, optimizer, callbacks=[Exit()]
        )
    assert ended == [reason]


# --- result and seeding ---------------------------------------------------------


def test_best_picks_the_best_validation():
    history = [(1, {"top1": 0.2, "loss": 3.0}), (2, {"top1": 0.6, "loss": 1.0})]
    result = TrainingResult(2, 2, "max_epochs", history)
    assert result.best("top1") == {"top1": 0.6, "loss": 1.0, "step": 2}
    assert result.best("loss", goal="min")["step"] == 2
    with pytest.raises(ValueError, match="no validation"):
        TrainingResult(0, 0, "max_steps", []).best("top1")


def test_seed_everything_is_reproducible():
    seed_everything(3)
    first = torch.randn(4)
    seed_everything(3)
    assert torch.equal(first, torch.randn(4))
