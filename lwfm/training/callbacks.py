from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

from lwfm.core.losses import LossBinding
from lwfm.core.types import Batch
from lwfm.results.artifacts import save_artifact

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

    +----------------------------------------------------------------------------------+
    | FIT   one call to trainer.fit(...)                                               |
    |   - on_fit_start                                                                 |
    |   +----------------------------------------------------------------------------+ |
    |   | EPOCH   repeats until max_epochs, max_steps or state.should_stop           | |
    |   |   - on_epoch_start                                                         | |
    |   |   +----------------------------------------------------------------------+ | |
    |   |   | BATCH   repeats for each batch of train_loader                       | | |
    |   |   |   - on_step_start(batch)                                             | | |
    |   |   |   - on_after_backward                                                | | |
    |   |   |   +----------------------------------------------------------------+ | | |
    |   |   |   | OPTIMIZER STEP   only at an accumulation boundary              | | | |
    |   |   |   |   - on_after_optimizer_step                                    | | | |
    |   |   |   +----------------------------------------------------------------+ | | |
    |   |   |   - on_step_end(output)                                              | | |
    |   |   |   +----------------------------------------------------------------+ | | |
    |   |   |   | VALIDATION   at a boundary, every val_every_n_steps (if set)   | | | |
    |   |   |   |   - on_validation_end(metrics)                                 | | | |
    |   |   |   +----------------------------------------------------------------+ | | |
    |   |   +----------------------------------------------------------------------+ | |
    |   |   - on_epoch_end            (skipped if training stopped mid-epoch)        | |
    |   |   +----------------------------------------------------------------------+ | |
    |   |   | VALIDATION   after the epoch, if val_every_n_steps is unset          | | |
    |   |   |   - on_validation_end(metrics)                                       | | |
    |   |   +----------------------------------------------------------------------+ | |
    |   +----------------------------------------------------------------------------+ |
    |   +----------------------------------------------------------------------------+ |
    |   | FINAL VALIDATION   unless the last step was just validated                 | |
    |   |   - on_validation_end(metrics)                                             | |
    |   +----------------------------------------------------------------------------+ |
    |   - on_fit_end(result)      (also on Ctrl-C or a crash, before re-raising)       |
    +----------------------------------------------------------------------------------+
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
    """Rejects anything that isn't a Callback, and any on_* method that isn't a hook
    (e.g. a misspelled one, which would otherwise never run)."""
    for callback in callbacks:
        if not isinstance(callback, Callback):
            raise TypeError(f"{callback!r} is not a Callback")
        unknown = [
            name
            for name in dir(callback)  # already sorted
            if name.startswith("on_") and name not in HOOKS
        ]
        if unknown:
            raise ValueError(
                f"{type(callback).__name__} defines {unknown}, which are not hooks; "
                f"the hooks are {list(HOOKS)}"
            )


def improved(
    value: float,
    best: float | None,
    goal: Literal["max", "min"],
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True
    return value > best + min_delta if goal == "max" else value < best - min_delta


def require_metric(callback: Callback, metric: str, state: "TrainingState") -> None:
    # Fails at fit start, not at the first validation an epoch later
    if metric not in state.metric_names:
        raise ValueError(
            f"{type(callback).__name__} monitors {metric!r}, but validation computes "
            f"{list(state.metric_names) or 'nothing (no val_loader)'}"
        )


def check_goal(goal: str) -> None:
    if goal not in ("max", "min"):
        raise ValueError(f"goal must be 'max' or 'min', got {goal!r}")


class LossWeightSchedule(Callback):
    """Sets a loss binding's weight to `weight_at_step(global_step)` before every batch,
    e.g. to ramp a distillation term in from 0."""

    def __init__(self, binding_name: str, weight_at_step: Callable[[int], float]):
        self.binding_name, self.weight_at_step = binding_name, weight_at_step
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
        self.binding.weight = self.weight_at_step(state.global_step)


class EarlyStopping(Callback):
    """Stops training once `monitored_metric` has failed to improve by more than
    `min_delta` for `max_validations_without_improvement` validations in a row."""

    def __init__(
        self,
        monitored_metric: str,
        goal: Literal["max", "min"] = "max",
        max_validations_without_improvement: int = 5,
        min_delta: float = 0.0,
    ):
        check_goal(goal)
        self.monitored_metric = monitored_metric
        self.goal: Literal["max", "min"] = goal
        self.max_validations_without_improvement = max_validations_without_improvement
        self.min_delta = min_delta
        self.best: float | None = None
        self.validations_without_improvement = 0

    def on_fit_start(self, state: "TrainingState") -> None:
        require_metric(self, self.monitored_metric, state)
        self.best, self.validations_without_improvement = None, 0

    def on_validation_end(
        self, state: "TrainingState", metrics: dict[str, float]
    ) -> None:
        value = metrics[self.monitored_metric]
        if improved(value, self.best, self.goal, self.min_delta):
            self.best, self.validations_without_improvement = value, 0
            return
        self.validations_without_improvement += 1
        limit = self.max_validations_without_improvement
        if self.validations_without_improvement >= limit:
            state.should_stop = True


class ModelCheckpoint(Callback):
    """Saves one node's weights as `best.pt` whenever `monitored_metric` improves
    (only if one is given). When fit ends, including after Ctrl-C or a crash (check
    `result.stop_reason` to tell), saves the full training state as `last.pt` and
    the node's final weights as `last_weights.pt`.

    `best.pt` and `last_weights.pt` hold `deployable_model(node).state_dict()`
    alone: no teachers or training-only nodes, so they load into a fresh module and
    their size is the model's size. `last.pt` is for resuming and inspection.

    Files go in `run_dir/subdir`. Each save is written atomically, then announced as
    an `artifact` event whose path is relative to `run_dir`, so the run directory can
    be moved without breaking its records.
    """

    def __init__(
        self,
        run_dir: str | Path,
        node: str,
        monitored_metric: str | None = None,
        goal: Literal["max", "min"] = "max",
        subdir: str = "checkpoints",
        save_last: bool = True,
    ):
        check_goal(goal)
        self.run_dir = Path(run_dir)
        self.node, self.monitored_metric = node, monitored_metric
        self.goal: Literal["max", "min"] = goal
        self.save_last = save_last
        self.best_path = self.run_dir / subdir / "best.pt"
        self.last_path = self.run_dir / subdir / "last.pt"
        self.last_weights_path = self.run_dir / subdir / "last_weights.pt"
        self.best_value: float | None = None

    def on_fit_start(self, state: "TrainingState") -> None:
        if self.node not in state.graph.nodes:
            raise ValueError(
                f"ModelCheckpoint saves node {self.node!r}, but the graph has "
                f"{list(state.graph.nodes)}"
            )
        if self.monitored_metric is not None:
            require_metric(self, self.monitored_metric, state)
        self.best_value = None

    def on_validation_end(
        self, state: "TrainingState", metrics: dict[str, float]
    ) -> None:
        if self.monitored_metric is None:
            return
        value = metrics[self.monitored_metric]
        if improved(value, self.best_value, self.goal):
            self.best_value = value
            self._save(
                state, self._node_weights(state), self.best_path, "checkpoint/best"
            )

    def on_fit_end(self, state: "TrainingState", result: "TrainingResult") -> None:
        if self.save_last:
            self._save(state, state.state_dict(), self.last_path, "checkpoint/last")
            self._save(
                state,
                self._node_weights(state),
                self.last_weights_path,
                "checkpoint/last_weights",
            )

    def _node_weights(self, state: "TrainingState") -> dict:
        # The deployable module only: no teachers, no training-only projectors
        return state.graph.deployable_model(self.node).state_dict()

    def _save(
        self, state: "TrainingState", contents: dict, path: Path, event_name: str
    ) -> None:
        save_artifact(
            self.run_dir,
            path.relative_to(self.run_dir),
            lambda temporary: torch.save(contents, temporary),
            state.output_sink,
            name=event_name,
            step=state.global_step,
        )


class RestoreBestWeights(Callback):
    """Trains for the full budget, then gives `node` back its weights from the
    validation with the best `monitored_metric`. The copy is kept in memory, and is
    restored even after Ctrl-C or a crash."""

    def __init__(
        self, node: str, monitored_metric: str, goal: Literal["max", "min"] = "max"
    ):
        check_goal(goal)
        self.node, self.monitored_metric = node, monitored_metric
        self.goal: Literal["max", "min"] = goal
        self.best_value: float | None = None
        self.best_weights: dict | None = None

    def on_fit_start(self, state: "TrainingState") -> None:
        if self.node not in state.graph.nodes:
            raise ValueError(
                f"RestoreBestWeights keeps node {self.node!r}, but the graph has "
                f"{list(state.graph.nodes)}"
            )
        require_metric(self, self.monitored_metric, state)
        self.best_value, self.best_weights = None, None

    def on_validation_end(
        self, state: "TrainingState", metrics: dict[str, float]
    ) -> None:
        value = metrics[self.monitored_metric]
        if improved(value, self.best_value, self.goal):
            weights = state.graph.deployable_model(self.node).state_dict()
            self.best_value = value
            # clone: state_dict tensors share memory with the weights still training
            self.best_weights = {
                name: t.detach().clone() for name, t in weights.items()
            }

    def on_fit_end(self, state: "TrainingState", result: "TrainingResult") -> None:
        if self.best_weights is not None:
            state.graph.deployable_model(self.node).load_state_dict(self.best_weights)


class EMAUpdate(Callback):
    """After every optimizer step, moves the `target` node's weights towards the
    `source` node's: target <- m * target + (1 - m) * source.

    For mean-teacher or DINO-style self-distillation, where the target is a frozen
    copy of the source. Buffers (e.g. BatchNorm statistics) are copied, not averaged.
    """

    def __init__(self, student_name: str, teacher_name: str, momentum: float = 0.996):
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")
        self.student_name, self.teacher_name, self.momentum = (
            student_name,
            teacher_name,
            momentum,
        )

    def on_fit_start(self, state: "TrainingState") -> None:
        source, target = self._modules(state)
        source_shapes = [p.shape for p in source.parameters()]
        if source_shapes != [p.shape for p in target.parameters()]:
            raise ValueError(
                f"EMAUpdate needs {self.student_name!r} and {self.teacher_name!r} to "
                + "have the same architecture, but their parameter shapes differ"
            )

    @torch.no_grad()
    def on_after_optimizer_step(self, state: "TrainingState") -> None:
        student, teacher = self._modules(state)
        # Parameters (learned weights): averaged, so the teacher changes smoothly
        for p_target, p_source in zip(
            teacher.parameters(), student.parameters(), strict=True
        ):
            # lerp_: t + (1 - m) * (s - t) = m * t + (1 - m) * s
            p_target.lerp_(p_source, 1.0 - self.momentum)
        # Buffers (e.g. BatchNorm running stats): copied. The frozen teacher never
        # updates its own, and integer ones (num_batches_tracked) can't be averaged
        for b_target, b_source in zip(
            teacher.buffers(), student.buffers(), strict=True
        ):
            b_target.copy_(b_source)

    def _modules(
        self, state: "TrainingState"
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        for name in (self.student_name, self.teacher_name):
            if name not in state.graph.nodes:
                raise ValueError(f"EMAUpdate: the graph has no node {name!r}")
        graph = state.graph
        return graph.deployable_model(self.student_name), graph.deployable_model(
            self.teacher_name
        )
