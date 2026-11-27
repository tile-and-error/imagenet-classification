import random
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial

import numpy as np  # a torchvision dependency, so always installed
import torch
from torch import Tensor
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph
from lwfm.core.types import Batch
from lwfm.evaluation.evaluator import EvaluationResult, Evaluator
from lwfm.evaluation.metrics import CurveBinding, MetricBinding
from lwfm.loss.losses import LossComposer
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
    Curves passed to `fit` are built at the last validation only, or also at every
    `curves_every_n_validations`-th one.

    With an output sink, `fit` first writes a `trainer/settings` message holding the
    loop settings, the torch seed and the optimizer, scheduler and callbacks, so
    every run records what it needs to be reproduced and compared.
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
        curves_every_n_validations: int | None = None,
        log_every_n_steps: int = 50,
        step_fn: StepFn = default_step,
    ):
        if max_epochs is None and max_steps is None:
            raise ValueError("set max_epochs, max_steps or both")
        counting_settings = {
            "max_epochs": max_epochs,
            "max_steps": max_steps,
            "val_every_n_steps": val_every_n_steps,
            "curves_every_n_validations": curves_every_n_validations,
            "accumulate_grad_batches": accumulate_grad_batches,
            "log_every_n_steps": log_every_n_steps,
        }
        for setting_name, setting_value in counting_settings.items():
            if setting_value is not None and setting_value < 1:
                raise ValueError(
                    f"{setting_name} must be at least 1, got {setting_value}"
                )
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
        self.validate_per_epoch = self.val_every_n_steps is None
        self.curves_every_n_validations = curves_every_n_validations
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
        curves: Sequence[CurveBinding] = (),
        scheduler: LRScheduler | None = None,
        scheduler_interval: SchedulerInterval | None = None,
        callbacks: Sequence[Callback] = (),
        output_sink: OutputSink | None = None,
    ) -> TrainingResult:
        """Trains until a stopping condition is met, then returns the result."""
        # Every check runs before the first batch, so a mistake costs seconds, not a run
        check_reads(graph, loss, [*metrics, *curves])
        check_validation(val_loader, metrics, curves)
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
        curve_steps: list[int] = []  # the steps whose validation built the curves
        # Validation with this run's loader, metrics and curves; None without a loader
        validate = (
            None
            if val_loader is None
            else partial(
                self._validate, state, val_loader, metrics, curves, history, curve_steps
            )
        )

        if output_sink is not None:
            settings = self._settings(
                optimizer, scheduler, scheduler_interval, callbacks
            )
            output_sink.write(RunEvent("message", "trainer/settings", settings, step=0))
        state.emit("on_fit_start")
        try:
            while self._stop_reason(state) is None:
                epoch_completed = self._run_epoch(state, train_loader, validate)
                if epoch_completed and validate is not None and self.validate_per_epoch:
                    validate()
            # End on a validation result, unless the last step was just validated
            last_step_validated = bool(history) and history[-1][0] == state.global_step
            if validate is not None and not last_step_validated:
                validate()
            if (
                val_loader is not None
                and curves
                and curve_steps[-1:] != [state.global_step]
            ):
                # A callback ended the run at a validation that had no curves
                self._evaluate(state, val_loader, [], curves)
        except (KeyboardInterrupt, Exception) as error:
            # Ctrl-C or a crash: callbacks still get on_fit_end (so "last" is saved),
            # then the original error propagates, for OutputRouter to record. A
            # callback failing here is attached as a note, never hiding the error.
            stop_reason = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            result = TrainingResult(
                state.global_step, state.epoch, stop_reason, history
            )
            try:
                state.emit("on_fit_end", result)
            except Exception as hook_error:
                error.add_note(f"on_fit_end also failed: {hook_error!r}")
            raise
        stop_reason = self._stop_reason(state)
        assert stop_reason is not None  # the loop only ends once a limit is reached
        result = TrainingResult(state.global_step, state.epoch, stop_reason, history)
        state.emit("on_fit_end", result)
        return result

    def _run_epoch(
        self,
        state: TrainingState,
        train_loader: Iterable[Batch],
        validate: Callable[[], None] | None,
    ) -> bool:
        """One pass over the train loader. Returns False if training stopped part-way,
        in which case the epoch isn't counted."""
        state.emit("on_epoch_start")
        state.graph.train()  # frozen nodes (teachers) stay in eval mode
        epoch_start_time, batch_count = time.perf_counter(), 0
        for batch, is_last_batch in mark_last_batch(train_loader):
            batch_count += 1
            state.is_last_batch = is_last_batch
            state.accumulated += 1
            batch = to_device(batch, self.device)

            state.emit("on_step_start", batch)
            state.used_backward_helper = False  # so every step is checked, not just one
            step_output = self.step_fn(state, batch)
            self._check_step_used_helpers(state)
            optimizer_stepped = state.is_accumulation_boundary
            if optimizer_stepped:
                state.accumulated = 0
                if state.scheduler is not None and state.scheduler_interval == "step":
                    state.scheduler.step()
                state.global_step += 1
            state.emit("on_step_end", step_output)
            if not optimizer_stepped:
                continue

            # Logging, validation and stopping happen only at optimizer steps, so no
            # half-accumulated gradient is ever dropped
            if state.global_step % self.log_every_n_steps == 0:
                self._log_step(state, step_output)
            validation_due = (
                self.val_every_n_steps is not None
                and state.global_step % self.val_every_n_steps == 0
            )
            if validate is not None and validation_due:
                validate()
            if self._stop_reason(state) is not None and not is_last_batch:
                return False
        if batch_count == 0:
            raise ValueError("train_loader yielded no batches")

        if state.scheduler is not None and state.scheduler_interval == "epoch":
            state.scheduler.step()
        state.epoch += 1
        epoch_seconds = time.perf_counter() - epoch_start_time
        self._write_scalar(state, "train/epoch_seconds", epoch_seconds)
        state.emit("on_epoch_end")
        return True

    def _validate(
        self,
        state: TrainingState,
        val_loader: Iterable[Batch],
        metrics: Sequence[MetricBinding],
        curves: Sequence[CurveBinding],
        history: History,
        curve_steps: list[int],
    ) -> None:
        """Runs the metrics over `val_loader`, records them, and tells the callbacks."""
        # Curves at the last validation (if training is about to stop) and at every
        # n-th; the rest skip them
        every = self.curves_every_n_validations
        if self._stop_reason(state) is not None or (
            every is not None and (len(history) + 1) % every == 0
        ):
            curve_steps.append(state.global_step)
        else:
            curves = ()
        val_metrics = self._evaluate(state, val_loader, metrics, curves).metric_values
        state.last_val_metrics = val_metrics
        history.append((state.global_step, val_metrics))
        state.emit("on_validation_end", val_metrics)

    def _evaluate(
        self,
        state: TrainingState,
        val_loader: Iterable[Batch],
        metrics: Sequence[MetricBinding],
        curves: Sequence[CurveBinding],
    ) -> EvaluationResult:
        # The Evaluator logs eval/val/<metric> and <curve>, and restores the graph's
        # train mode
        return Evaluator().evaluate(
            state.graph,
            val_loader,
            metrics,
            curve_bindings=curves,
            device=self.device,
            output_sink=state.output_sink,
            training_step=state.global_step,
        )

    def _check_step_used_helpers(self, state: TrainingState) -> None:
        """Refuses a custom step that bypassed `steps.backward` when it matters."""
        # A step that calls .backward() itself skips fp16 loss scaling and the 1/k
        # accumulation scaling, so its gradients would be silently wrong
        helpers_required = self.precision == "fp16" or self.accumulate_grad_batches > 1
        if helpers_required and not state.used_backward_helper:
            raise RuntimeError(
                "with precision='fp16' or accumulate_grad_batches > 1, a custom "
                "step_fn must call lwfm.training.steps.backward(state, loss) and, "
                "when state.is_accumulation_boundary, optimizer_step(state)"
            )

    def _log_step(self, state: TrainingState, step_output: StepOutput) -> None:
        """Writes each loss term, the total loss and the learning rate(s)."""
        # The losses of the batch that completed this optimizer step
        for term_name, term_value in step_output.loss_terms.items():
            self._write_scalar(state, f"train/loss/{term_name}", term_value)
        self._write_scalar(state, "train/loss/total", step_output.total_loss)
        param_groups = state.optimizer.param_groups
        for group_index, param_group in enumerate(param_groups):
            event_name = (
                "train/lr" if len(param_groups) == 1 else f"train/lr/{group_index}"
            )
            self._write_scalar(state, event_name, param_group["lr"])

    def _write_scalar(
        self, state: TrainingState, name: str, value: float | Tensor
    ) -> None:
        """Sends one scalar to the output sink, if there is one."""
        if state.output_sink is not None:
            state.output_sink.write(
                RunEvent("scalar", name, value, step=state.global_step)
            )

    def _settings(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None,
        scheduler_interval: SchedulerInterval | None,
        callbacks: Sequence[Callback],
    ):
        """Everything needed to reproduce or compare this run, as JSON-safe values."""
        return {
            "device": str(self.device),
            "precision": self.precision,
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "gradient_clip_norm": self.gradient_clip_norm,
            "val_every_n_steps": self.val_every_n_steps,
            "curves_every_n_validations": self.curves_every_n_validations,
            "log_every_n_steps": self.log_every_n_steps,
            "step_fn": qualified_name(self.step_fn),
            # The seed torch's default generator started from: the one given to
            # seed_everything, or the random one torch picked if nothing seeded it
            "torch_seed": torch.initial_seed(),
            "optimizer": {
                "type": type(optimizer).__name__,
                **json_safe_values(optimizer.defaults),
            },
            "scheduler": None if scheduler is None else type(scheduler).__name__,
            "scheduler_interval": scheduler_interval,
            "callbacks": [type(callback).__name__ for callback in callbacks],
        }

    def _stop_reason(self, state: TrainingState) -> StopReason | None:
        """Why training should stop now, or None to keep going."""
        if state.should_stop:
            return "stopped"
        if self.max_steps is not None and state.global_step >= self.max_steps:
            return "max_steps"
        if self.max_epochs is not None and state.epoch >= self.max_epochs:
            return "max_epochs"
        return None


def seed_everything(seed: int) -> None:
    """Seeds Python's, NumPy's and torch's generators (torch's covers every CUDA
    device). Call it before building the model: weight initialisation draws from them.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def qualified_name(function: StepFn) -> str:
    """e.g. `lwfm.training.steps.default_step`."""
    module = getattr(function, "__module__", None)
    name = getattr(function, "__qualname__", type(function).__name__)
    return f"{module}.{name}" if module else name


type JsonScalar = bool | int | float | str | None


def json_safe_values(settings) -> dict[str, JsonScalar | list[JsonScalar]]:
    """The entries whose values are plain numbers, strings, booleans or None, or
    lists or tuples of them; anything else (e.g. a tensor learning rate) is left out."""
    plain_types = (bool, int, float, str, type(None))
    kept: dict[str, JsonScalar | list[JsonScalar]] = {}
    for key, value in settings.items():
        if isinstance(value, plain_types):
            kept[key] = value
        elif isinstance(value, list | tuple) and all(
            isinstance(item, plain_types) for item in value
        ):
            kept[key] = list(value)
    return kept


def mark_last_batch(batches: Iterable[Batch]) -> Iterator[tuple[Batch, bool]]:
    """Yields (batch, is_last_batch). Looking one batch ahead is how the loop knows an
    epoch is ending, which works for loaders without len()."""
    batch_iterator = iter(batches)
    try:
        current_batch = next(batch_iterator)
    except StopIteration:
        return
    for next_batch in batch_iterator:
        yield current_batch, False
        current_batch = next_batch
    yield current_batch, True


def check_reads(
    graph: ExecutionGraph,
    loss: LossComposer,
    bindings: Sequence[MetricBinding | CurveBinding],
) -> None:
    """Checks that every name a loss, metric or curve reads is produced by the graph."""
    # A typo in a loss or metric name fails here, listing the outputs that do exist
    graph.check_reads(name for binding in loss.loss_bindings for name in binding.inputs)
    graph.check_reads(name for binding in bindings for name in binding.inputs)


def check_validation(
    val_loader: Iterable[Batch] | None,
    metrics: Sequence[MetricBinding],
    curves: Sequence[CurveBinding],
) -> None:
    """Checks that `val_loader` and `metrics` are given together or not at all, and
    that `curves` come with them."""
    if (val_loader is None) != (len(metrics) == 0):
        raise ValueError(
            "pass val_loader and metrics together: one is useless without the other"
        )
    if curves and val_loader is None:
        raise ValueError("curves are drawn at validation: pass val_loader and metrics")


def check_optimizer(
    graph: ExecutionGraph, loss: LossComposer, optimizer: torch.optim.Optimizer
) -> None:
    """Checks that every trainable parameter of the graph and loss is optimised."""
    # e.g. a distillation projector, which would otherwise silently stay at its
    # random init
    optimised_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    missing_param_names = [
        param_name
        for module in (graph, loss)
        for param_name, param in module.named_parameters()
        if param.requires_grad and id(param) not in optimised_param_ids
    ]
    if missing_param_names:
        raise ValueError(
            f"trainable parameters missing from the optimizer: {missing_param_names}"
        )


def check_scheduler(
    scheduler: LRScheduler | None,
    scheduler_interval: SchedulerInterval | None,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Checks that a scheduler has an interval, matches the optimizer, and can be
    stepped without a metric."""
    if scheduler is None:
        if scheduler_interval is not None:
            raise ValueError("scheduler_interval given without a scheduler")
        return
    # No default: per step and per epoch differ by a factor of the epoch length
    if scheduler_interval not in ("step", "epoch"):
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
