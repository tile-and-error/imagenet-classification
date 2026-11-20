import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from lwfm.core.losses import LossBinding
from lwfm.core.types import Batch
from lwfm.results.events import RunEvent

if TYPE_CHECKING:
    from lwfm.training.state import StepOutput, TrainingResult, TrainingState

# In the order they first fire within a fit
HOOKS = (
    "on_fit_start",
    "on_epoch_start",
    "on_step_start",
    "on_after_backward",
    "on_after_optimizer_step",
    "on_step_end",
    "on_epoch_end",
    "on_validation_end",
    "on_fit_end",
)


class Callback:
    """Code run at fixed points in training.

    Every hook does nothing by default, so a callback overrides only the hooks it
    needs, and adding a hook later breaks no callback. `Trainer.fit` rejects any
    other `on_*` method, so a misspelled hook fails instead of silently never running.
    """

    def on_fit_start(self, state: "TrainingState") -> None: ...
    def on_epoch_start(self, state: "TrainingState") -> None: ...
    def on_step_start(self, state: "TrainingState", batch: Batch) -> None: ...
    def on_after_backward(self, state: "TrainingState") -> None: ...
    def on_after_optimizer_step(self, state: "TrainingState") -> None: ...
    def on_step_end(self, state: "TrainingState", output: "StepOutput") -> None: ...
    def on_epoch_end(self, state: "TrainingState") -> None: ...
    def on_validation_end(
        self, state: "TrainingState", metrics: dict[str, float]
    ) -> None: ...
    def on_fit_end(self, state: "TrainingState", result: "TrainingResult") -> None: ...


def check_callbacks(callbacks: Sequence[Callback]) -> None:
    for callback in callbacks:
        if not isinstance(callback, Callback):
            raise TypeError(f"{callback!r} is not a Callback")
        unknown = sorted(
            name
            for name in dir(callback)
            if name.startswith("on_") and name not in HOOKS
        )
        if unknown:
            raise ValueError(
                f"{type(callback).__name__} defines {unknown}, which are not hooks; "
                f"the hooks are {list(HOOKS)}"
            )


def improved(
    value: float,
    best: float | None,
    mode: Literal["max", "min"],
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True
    return value > best + min_delta if mode == "max" else value < best - min_delta


def require_metric(callback: Callback, metric: str, state: "TrainingState") -> None:
    # Fails at fit start, not at the first validation an epoch later
    if metric not in state.metric_names:
        raise ValueError(
            f"{type(callback).__name__} monitors {metric!r}, but validation computes "
            f"{list(state.metric_names) or 'nothing (no val_loader)'}"
        )


def check_mode(mode: str) -> None:
    if mode not in ("max", "min"):
        raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")


class LossWeightSchedule(Callback):
    """Sets a loss binding's weight to `schedule(global_step)` before every batch,
    e.g. to ramp a distillation term in from 0."""

    def __init__(self, binding_name: str, schedule: Callable[[int], float]):
        self.binding_name, self.schedule = binding_name, schedule
        self.binding: LossBinding | None = None

    def on_fit_start(self, state: "TrainingState") -> None:
        names = [binding.name for binding in state.loss.loss_bindings]
        if self.binding_name not in names:
            raise ValueError(f"no loss binding named {self.binding_name!r}: {names}")
        self.binding = state.loss.loss_bindings[names.index(self.binding_name)]

    def on_step_start(self, state: "TrainingState", batch: Batch) -> None:
        # Changing a binding's float weight is safe: LossComposer reads it on every
        # call. (Swapping its loss module is not; see lwfm/core/losses.py.)
        assert self.binding is not None
        self.binding.weight = self.schedule(state.global_step)


class EarlyStopping(Callback):
    """Stops training once `monitor` has failed to improve by more than `min_delta`
    for `patience` validations in a row."""

    def __init__(
        self,
        monitor: str,
        mode: Literal["max", "min"] = "max",
        patience: int = 5,
        min_delta: float = 0.0,
    ):
        check_mode(mode)
        self.monitor = monitor
        self.mode: Literal["max", "min"] = mode
        self.patience, self.min_delta = patience, min_delta
        self.best: float | None = None
        self.validations_without_improvement = 0

    def on_fit_start(self, state: "TrainingState") -> None:
        require_metric(self, self.monitor, state)
        self.best, self.validations_without_improvement = None, 0

    def on_validation_end(
        self, state: "TrainingState", metrics: dict[str, float]
    ) -> None:
        value = metrics[self.monitor]
        if improved(value, self.best, self.mode, self.min_delta):
            self.best, self.validations_without_improvement = value, 0
            return
        self.validations_without_improvement += 1
        if self.validations_without_improvement >= self.patience:
            state.should_stop = True


class ModelCheckpoint(Callback):
    """Saves one node's weights as `best.pt` whenever `monitor` improves (only if a
    monitor is given), and the full training state as `last.pt` when fit ends,
    including after Ctrl-C or a crash (check `result.stop_reason` to tell).

    Files go in `run_dir/subdir`. Each save is written atomically, then announced as
    an `artifact` event whose path is relative to `run_dir`, so the run directory can
    be moved without breaking its records.
    """

    def __init__(
        self,
        run_dir: str | Path,
        node: str = "student",
        monitor: str | None = None,
        mode: Literal["max", "min"] = "max",
        subdir: str = "checkpoints",
        save_last: bool = True,
    ):
        check_mode(mode)
        self.run_dir = Path(run_dir)
        self.node, self.monitor = node, monitor
        self.mode: Literal["max", "min"] = mode
        self.save_last = save_last
        self.best_path = self.run_dir / subdir / "best.pt"
        self.last_path = self.run_dir / subdir / "last.pt"
        self.best_value: float | None = None

    def on_fit_start(self, state: "TrainingState") -> None:
        if self.node not in state.graph.nodes:
            raise ValueError(
                f"ModelCheckpoint saves node {self.node!r}, but the graph has "
                f"{list(state.graph.nodes)}"
            )
        if self.monitor is not None:
            require_metric(self, self.monitor, state)
        self.best_path.parent.mkdir(parents=True, exist_ok=True)
        self.best_value = None

    def on_validation_end(
        self, state: "TrainingState", metrics: dict[str, float]
    ) -> None:
        if self.monitor is None:
            return
        value = metrics[self.monitor]
        if improved(value, self.best_value, self.mode):
            self.best_value = value
            # The deployable module only: no teachers, no training-only projectors
            weights = state.graph.deployable_model(self.node).state_dict()
            self._save(state, weights, self.best_path, "checkpoint/best")

    def on_fit_end(self, state: "TrainingState", result: "TrainingResult") -> None:
        if self.save_last:
            self._save(state, state.state_dict(), self.last_path, "checkpoint/last")

    def _save(
        self, state: "TrainingState", contents: dict, path: Path, event_name: str
    ) -> None:
        # File first, then the event, so the record never points at a missing file.
        # os.replace is atomic: a job killed mid-save leaves the previous file intact.
        temporary = path.with_name(path.name + ".tmp")
        torch.save(contents, temporary)
        os.replace(temporary, path)
        if state.output_sink is not None:
            state.output_sink.write(
                RunEvent(
                    "artifact",
                    event_name,
                    path.relative_to(self.run_dir),
                    step=state.global_step,
                )
            )


class EMAUpdate(Callback):
    """After every optimizer step, moves the `target` node's weights towards the
    `source` node's: target ← m·target + (1 − m)·source.

    For mean-teacher or DINO-style self-distillation, where the target is a frozen
    copy of the source. Buffers (e.g. BatchNorm statistics) are copied, not averaged.
    """

    def __init__(self, source: str, target: str, momentum: float = 0.996):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.source, self.target, self.momentum = source, target, momentum

    def on_fit_start(self, state: "TrainingState") -> None:
        source, target = self._modules(state)
        source_shapes = [p.shape for p in source.parameters()]
        if source_shapes != [p.shape for p in target.parameters()]:
            raise ValueError(
                f"EMAUpdate needs {self.source!r} and {self.target!r} to have the same "
                "architecture, but their parameter shapes differ"
            )

    @torch.no_grad()
    def on_after_optimizer_step(self, state: "TrainingState") -> None:
        source, target = self._modules(state)
        for p_target, p_source in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            # lerp_: target + (1 − m)(source − target) = m·target + (1 − m)·source
            p_target.lerp_(p_source, 1.0 - self.momentum)
        for b_target, b_source in zip(target.buffers(), source.buffers(), strict=True):
            b_target.copy_(b_source)

    def _modules(
        self, state: "TrainingState"
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        for name in (self.source, self.target):
            if name not in state.graph.nodes:
                raise ValueError(f"EMAUpdate: the graph has no node {name!r}")
        graph = state.graph
        return graph.deployable_model(self.source), graph.deployable_model(self.target)
