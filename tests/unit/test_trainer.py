import json

import pytest
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from lwfm.core.graph import ExecutionGraph, GraphError, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import CurveBinding, MetricBinding
from lwfm.results.curves import ConfusionMatrix
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import LocalFileSink
from lwfm.training import Callback, Trainer, TrainingResult, seed_everything
from lwfm.training.callbacks import HOOKS
from lwfm.training.trainer import json_safe_values

from conftest import (
    CE_READS,
    K,
    accuracy,
    batches,
    constant_metric,
    setup,
    student,
)


class HookRecorder(Callback):
    """Records every hook call, in order."""

    def __init__(self):
        self.calls = []

    def __getattribute__(self, name):
        if name in HOOKS:
            return lambda state, *args: self.calls.append(name)
        return super().__getattribute__(name)


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


def test_counts_epochs_validates_each_epoch_and_logs(sink):
    _, graph, loss, optimizer = setup()
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


def test_stopping_mid_epoch_skips_the_epoch_but_still_validates(sink):
    _, graph, loss, optimizer = setup()
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


def test_curves_are_drawn_at_every_validation(sink):
    _, graph, loss, optimizer = setup()
    confusion = CurveBinding("confusion", ConfusionMatrix.from_scores, CE_READS)
    Trainer(device="cpu", max_epochs=1, val_every_n_steps=2).fit(
        graph,
        batches(5),
        loss,
        optimizer,
        val_loader=batches(2),
        metrics=accuracy(),
        curves=[confusion],
        output_sink=sink,
    )
    assert sink.steps_of("eval/val/confusion") == [2, 4, 5]
    assert all(
        sum(map(sum, e.value.matrix)) == 8 for e in sink.events if e.kind == "table"
    )  # all 2 * 4 validation samples, each time


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


def test_lr_is_logged_per_param_group(sink):
    node, graph, loss, _ = setup()
    weight, bias = node.parameters()
    optimizer = torch.optim.SGD(
        [{"params": [weight]}, {"params": [bias], "lr": 0.01}], lr=0.1
    )
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
    confusion = CurveBinding("confusion", ConfusionMatrix.from_scores, CE_READS)
    with pytest.raises(ValueError, match="curves"):
        trainer.fit(graph, batches(1), loss, optimizer, curves=[confusion])


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


# --- recorded settings ---------------------------------------------------------


def test_settings_are_recorded_once_before_training(sink):
    _, graph, loss, optimizer = setup(seed=3)
    Trainer(
        device="cpu",
        max_steps=2,
        precision="bf16",
        accumulate_grad_batches=2,
        gradient_clip_norm=1.0,
    ).fit(
        graph, batches(4), loss, optimizer, callbacks=[HookRecorder()], output_sink=sink
    )

    first = sink.events[0]
    assert (first.kind, first.name, first.step) == ("message", "trainer/settings", 0)
    assert [e.name for e in sink.events].count("trainer/settings") == 1
    settings = first.value
    assert settings["precision"] == "bf16"
    assert settings["accumulate_grad_batches"] == 2
    assert settings["gradient_clip_norm"] == 1.0
    assert (settings["max_steps"], settings["max_epochs"]) == (2, None)
    assert settings["torch_seed"] == 3  # setup seeds torch with 3
    assert settings["step_fn"] == "lwfm.training.steps.default_step"
    assert settings["optimizer"]["type"] == "SGD"
    assert settings["optimizer"]["lr"] == 0.1
    assert settings["callbacks"] == ["HookRecorder"]


def test_settings_survive_a_file_sink(tmp_path):
    node, graph, loss, _ = setup()
    optimizer = torch.optim.Adam(node.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    with OutputRouter(LocalFileSink(tmp_path)) as sink:
        Trainer(device="cpu", max_steps=1).fit(
            graph,
            batches(1),
            loss,
            optimizer,
            scheduler=scheduler,
            scheduler_interval="step",
            output_sink=sink,
        )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").open()]
    settings = next(e["value"] for e in events if e["name"] == "trainer/settings")
    assert settings["optimizer"]["betas"] == [0.9, 0.999]  # a tuple, stored as a list
    assert (settings["scheduler"], settings["scheduler_interval"]) == ("StepLR", "step")


def test_json_safe_values_leaves_out_what_json_cannot_hold():
    settings = {
        "lr": torch.tensor(0.1),
        "betas": (0.9, 0.99),
        "foreach": None,
        "odd": [object()],
    }
    assert json_safe_values(settings) == {"betas": [0.9, 0.99], "foreach": None}


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
