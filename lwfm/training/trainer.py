import time
from collections.abc import Callable, Iterable, Iterator, Sequence

import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph
from lwfm.core.losses import LossComposer
from lwfm.core.types import Batch
from lwfm.evaluation.evaluator import Evaluator
from lwfm.evaluation.metrics import MetricBinding
from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink
from lwfm.training.callbacks import Callback, check_callbacks
from lwfm.training.state import (
    History,
    Precision,
    SchedulerInterval,
    StepOutput,
    StopReason,
    TrainingResult,
    TrainingState,
)
from lwfm.training.steps import AUTOCAST_DTYPES, default_step

type StepFn = Callable[[TrainingState, Batch], StepOutput]


class Trainer:
    """Trains an `ExecutionGraph` with a `LossComposer` and any torch optimizer.

    The constructor holds the loop settings, and keeps no state between fits, so one
    trainer can run many experiments. `fit` takes the objects for one run.

    `max_steps` and `val_every_n_steps` count optimizer steps; with
    `accumulate_grad_batches=k`, one step is k batches. Without `val_every_n_steps`,
    validation runs after every epoch. Either way, a run that stops part-way through
    an epoch validates once more, so it always ends with a validation result.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        max_epochs: int | None = None,
        max_steps: int | None = None,
        precision: Precision = "fp32",
        accumulate_grad_batches: int = 1,
        gradient_clip_norm: float | None = None,
        val_every_n_steps: int | None = None,
        log_every_n_steps: int = 50,
        step_fn: StepFn = default_step,
    ):
        if max_epochs is None and max_steps is None:
            raise ValueError("set max_epochs, max_steps or both")
        counts = {
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "val_every_n_steps": val_every_n_steps,
            "accumulate_grad_batches": accumulate_grad_batches,
            "log_every_n_steps": log_every_n_steps,
        }
        for name, value in counts.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
        if precision not in AUTOCAST_DTYPES:
            raise ValueError(f"precision must be one of {list(AUTOCAST_DTYPES)}")
        if gradient_clip_norm is not None and gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive")

        self.device = torch.device(device)
        self.max_epochs, self.max_steps = max_epochs, max_steps
        self.precision: Precision = precision
        self.accumulate_grad_batches = accumulate_grad_batches
        self.gradient_clip_norm = gradient_clip_norm
        self.val_every_n_steps = val_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.step_fn = step_fn

    def fit(
        self,
        graph: ExecutionGraph,
        train_loader: Iterable[Batch],
        loss: LossComposer,
        optimizer: torch.optim.Optimizer,
        *,
        val_loader: Iterable[Batch] | None = None,
        metrics: Sequence[MetricBinding] = (),
        scheduler: LRScheduler | None = None,
        scheduler_interval: SchedulerInterval | None = None,
        callbacks: Sequence[Callback] = (),
        output_sink: OutputSink | None = None,
    ) -> TrainingResult:
        # Every check runs before the first batch, so a mistake costs seconds, not a run
        check_reads(graph, loss, metrics)
        check_validation(val_loader, metrics)
        check_optimizer(graph, loss, optimizer)
        check_scheduler(scheduler, scheduler_interval, optimizer)
        check_callbacks(callbacks)

        graph.to(self.device)
        loss.to(self.device)  # loss buffers too, e.g. CrossEntropyLoss class weights
        optimizer.zero_grad(set_to_none=True)  # accumulate from clean gradients
        state = TrainingState(
            graph=graph,
            loss=loss,
            optimizer=optimizer,
            device=self.device,
            precision=self.precision,
            scaler=GradScaler(self.device.type) if self.precision == "fp16" else None,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_norm=self.gradient_clip_norm,
            scheduler=scheduler,
            scheduler_interval=scheduler_interval,
            output_sink=output_sink,
            callbacks=callbacks,
            metric_names=tuple(binding.name for binding in metrics),
        )
        history: History = []

        state.emit("on_fit_start")
        try:
            while not self._should_stop(state):
                completed = self._run_epoch(
                    state, train_loader, val_loader, metrics, history
                )
                if completed and val_loader is not None and not self.val_every_n_steps:
                    self._validate(state, val_loader, metrics, history)
            # End on a validation result, unless the last step was just validated
            if val_loader is not None and (
                not history or history[-1][0] != state.global_step
            ):
                self._validate(state, val_loader, metrics, history)
        except (KeyboardInterrupt, Exception) as error:
            # Ctrl-C or a crash: callbacks still get on_fit_end (so "last" is saved),
            # then the original error propagates, for OutputRouter to record. A
            # callback failing here is attached as a note, never hiding the error.
            reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            try:
                state.emit("on_fit_end", self._result(state, history, reason))
            except Exception as hook_error:
                error.add_note(f"on_fit_end also failed: {hook_error!r}")
            raise
        result = self._result(state, history, self._stop_reason(state))
        state.emit("on_fit_end", result)
        return result

    def _run_epoch(
        self,
        state: TrainingState,
        train_loader: Iterable[Batch],
        val_loader: Iterable[Batch] | None,
        metrics: Sequence[MetricBinding],
        history: History,
    ) -> bool:
        """One pass over the train loader. Returns False if training stopped part-way,
        in which case the epoch isn't counted."""
        state.emit("on_epoch_start")
        state.graph.train()  # frozen nodes (teachers) stay in eval mode
        started, batches = time.perf_counter(), 0
        for batch, is_last in mark_last(train_loader):
            batches += 1
            state.is_last_batch = is_last
            state.accumulated += 1
            batch = to_device(batch, self.device)

            state.emit("on_step_start", batch)
            output = self.step_fn(state, batch)
            self._check_step_used_helpers(state)
            stepped = state.is_accumulation_boundary
            if stepped:
                state.accumulated = 0
                if state.scheduler is not None and state.scheduler_interval == "step":
                    state.scheduler.step()
                state.global_step += 1
            state.emit("on_step_end", output)
            if not stepped:
                continue

            # Logging, validation and stopping happen only at optimizer steps, so no
            # half-accumulated gradient is ever dropped
            if state.global_step % self.log_every_n_steps == 0:
                self._log_step(state, output)
            if (
                val_loader is not None
                and self.val_every_n_steps
                and state.global_step % self.val_every_n_steps == 0
            ):
                self._validate(state, val_loader, metrics, history)
            if self._should_stop(state) and not is_last:
                return False
        if batches == 0:
            raise ValueError("train_loader yielded no batches")

        if state.scheduler is not None and state.scheduler_interval == "epoch":
            state.scheduler.step()
        state.epoch += 1
        self._write(state, "train/epoch_seconds", time.perf_counter() - started)
        state.emit("on_epoch_end")
        return True

    def _validate(
        self,
        state: TrainingState,
        val_loader: Iterable[Batch],
        metrics: Sequence[MetricBinding],
        history: History,
    ) -> None:
        # The Evaluator logs eval/val/<metric> and restores the graph's train mode
        result = Evaluator().evaluate(
            state.graph,
            val_loader,
            metrics,
            device=self.device,
            output_sink=state.output_sink,
            training_step=state.global_step,
        )
        values = result.metric_values
        state.last_val_metrics = values
        history.append((state.global_step, values))
        state.emit("on_validation_end", values)

    def _check_step_used_helpers(self, state: TrainingState) -> None:
        # A step that calls .backward() itself skips fp16 loss scaling and the 1/k
        # accumulation scaling, so its gradients would be silently wrong
        needs_helpers = self.precision == "fp16" or self.accumulate_grad_batches > 1
        if needs_helpers and not state.used_backward_helper:
            raise RuntimeError(
                "with precision='fp16' or accumulate_grad_batches > 1, a custom "
                "step_fn must call lwfm.training.steps.backward(state, loss) and, "
                "when state.is_accumulation_boundary, optimizer_step(state)"
            )

    def _log_step(self, state: TrainingState, output: StepOutput) -> None:
        # The losses of the batch that completed this optimizer step
        for name, value in output.loss_terms.items():
            self._write(state, f"train/loss/{name}", value)
        self._write(state, "train/loss/total", output.total_loss)
        groups = state.optimizer.param_groups
        for index, group in enumerate(groups):
            name = "train/lr" if len(groups) == 1 else f"train/lr/{index}"
            self._write(state, name, group["lr"])

    def _write(self, state: TrainingState, name: str, value: float | Tensor) -> None:
        if state.output_sink is not None:
            state.output_sink.write(
                RunEvent("scalar", name, value, step=state.global_step)
            )

    def _should_stop(self, state: TrainingState) -> bool:
        return self._stop_reason(state) != "max_epochs" or (
            self.max_epochs is not None and state.epoch >= self.max_epochs
        )

    def _stop_reason(self, state: TrainingState) -> StopReason:
        # "max_epochs" doubles as "keep going" while the epoch limit isn't reached
        if state.should_stop:
            return "stopped"
        if self.max_steps is not None and state.global_step >= self.max_steps:
            return "max_steps"
        return "max_epochs"

    def _result(
        self, state: TrainingState, history: History, stop_reason: StopReason
    ) -> TrainingResult:
        return TrainingResult(state.global_step, state.epoch, stop_reason, history)


def mark_last(batches: Iterable[Batch]) -> Iterator[tuple[Batch, bool]]:
    """Yields (batch, is_last). Looking one batch ahead is how the loop knows an epoch
    is ending, which works for loaders without len()."""
    iterator = iter(batches)
    try:
        current = next(iterator)
    except StopIteration:
        return
    for upcoming in iterator:
        yield current, False
        current = upcoming
    yield current, True


def check_reads(
    graph: ExecutionGraph, loss: LossComposer, metrics: Sequence[MetricBinding]
) -> None:
    # A typo in a loss or metric name fails here, listing the outputs that do exist
    graph.check_reads(name for binding in loss.loss_bindings for name in binding.inputs)
    graph.check_reads(name for binding in metrics for name in binding.inputs)


def check_validation(
    val_loader: Iterable[Batch] | None, metrics: Sequence[MetricBinding]
) -> None:
    if (val_loader is None) != (len(metrics) == 0):
        raise ValueError(
            "pass val_loader and metrics together: one is useless without the other"
        )


def check_optimizer(
    graph: ExecutionGraph, loss: LossComposer, optimizer: torch.optim.Optimizer
) -> None:
    # Every trainable parameter must be in the optimizer, e.g. a distillation
    # projector, which would otherwise silently stay at its random init
    in_optimizer = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [
        name
        for module in (graph, loss)
        for name, p in module.named_parameters()
        if p.requires_grad and id(p) not in in_optimizer
    ]
    if missing:
        raise ValueError(f"trainable parameters missing from the optimizer: {missing}")


def check_scheduler(
    scheduler: LRScheduler | None,
    interval: SchedulerInterval | None,
    optimizer: torch.optim.Optimizer,
) -> None:
    if scheduler is None:
        if interval is not None:
            raise ValueError("scheduler_interval given without a scheduler")
        return
    # No default: per step and per epoch differ by a factor of the epoch length
    if interval not in ("step", "epoch"):
        raise ValueError("pass scheduler_interval='step' or 'epoch' with a scheduler")
    if scheduler.optimizer is not optimizer:
        raise ValueError("the scheduler was built for a different optimizer")
    # The trainer only calls scheduler.step(); ReduceLROnPlateau needs a metric, which
    # would fail with a TypeError after the first epoch. Refuse it now instead.
    if isinstance(scheduler, ReduceLROnPlateau):
        raise ValueError(
            "ReduceLROnPlateau steps on a validation metric: call "
            "scheduler.step(metrics[name]) from a Callback's on_validation_end"
        )
