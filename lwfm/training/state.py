from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from lwfm.core.graph import ExecutionGraph
from lwfm.core.losses import LossComposer
from lwfm.results.sinks import OutputSink

if TYPE_CHECKING:
    from lwfm.training.callbacks import Callback

type Precision = Literal["fp32", "bf16", "fp16"]
# When the trainer calls scheduler.step(): after each optimizer step, or after each
# full epoch
type SchedulerInterval = Literal["step", "epoch"]
type StopReason = Literal["max_epochs", "max_steps", "stopped", "interrupted", "failed"]
# (global_step, validation metrics) for every validation, in order
type History = list[tuple[int, dict[str, float]]]


class StepOutput(NamedTuple):
    total_loss: Tensor  # detached
    loss_terms: dict[str, Tensor]  # detached, by LossBinding name


@dataclass
class TrainingState:
    """Everything about one `fit` call that a step or a callback may need.

    `Trainer.fit` creates one per call and passes it to `step_fn` and to every
    callback hook. Counters: `global_step` counts optimizer steps; `epoch` counts
    complete passes over the train loader; `accumulated` counts the batches whose
    gradients are waiting for the next optimizer step.
    """

    graph: ExecutionGraph
    loss: LossComposer
    optimizer: torch.optim.Optimizer
    device: torch.device
    precision: Precision = "fp32"
    scaler: GradScaler | None = None  # fp16 only
    accumulate_grad_batches: int = 1
    gradient_clip_norm: float | None = None
    scheduler: LRScheduler | None = None
    scheduler_interval: SchedulerInterval | None = None
    output_sink: OutputSink | None = None
    callbacks: Sequence["Callback"] = ()
    # Names of the validation metrics, so callbacks can check a monitor before training
    metric_names: tuple[str, ...] = ()
    global_step: int = 0
    epoch: int = 0
    accumulated: int = 0
    is_last_batch: bool = False
    last_val_metrics: dict[str, float] = field(default_factory=dict)
    # Set by a callback to stop training at the next optimizer step
    should_stop: bool = False
    # Set by steps.backward, so the trainer can tell whether a custom step used it
    used_backward_helper: bool = False

    @property
    def is_accumulation_boundary(self) -> bool:
        # An optimizer step is due after k batches, and at the end of every epoch so
        # gradients never carry over into the next one
        return self.accumulated >= self.accumulate_grad_batches or self.is_last_batch

    def emit(self, hook: str, *args) -> None:
        for callback in self.callbacks:
            getattr(callback, hook)(self, *args)

    def state_dict(self) -> dict:
        """Tensors and plain values only, so `torch.load(..., weights_only=True)` can
        read it back. The whole graph is saved, teachers included: an EMA teacher
        changes during training."""
        return {
            "graph": self.graph.state_dict(),
            "loss": self.loss.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class TrainingResult:
    global_step: int
    epochs_completed: int
    stop_reason: StopReason
    history: History

    def best(
        self, metric: str, mode: Literal["max", "min"] = "max"
    ) -> dict[str, float]:
        """The validation with the best `metric`, plus the step it ran at."""
        if not self.history:
            raise ValueError("no validation ran, so there is no best result")
        pick = max if mode == "max" else min
        step, values = pick(self.history, key=lambda entry: entry[1][metric])
        return values | {"step": step}
