"""One training step, built from three public helpers.

A custom `step_fn` that calls `backward` and `optimizer_step` keeps mixed precision,
gradient accumulation, clipping and the `on_after_*` hooks. One that calls
`.backward()` itself opts out of them; the trainer refuses that combination with
fp16 or accumulation, where the gradients would silently be wrong.
"""

import torch
from torch import Tensor

from lwfm.core.types import Batch
from lwfm.training.state import StepOutput, TrainingState

AUTOCAST_DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def forward_and_loss(
    state: TrainingState, batch: Batch
) -> tuple[Tensor, dict[str, Tensor]]:
    """Runs the graph and the loss, under autocast unless the precision is fp32."""
    dtype = AUTOCAST_DTYPES[state.precision]
    with torch.autocast(state.device.type, dtype=dtype, enabled=dtype is not None):
        return state.loss(batch, state.graph(batch))


def backward(state: TrainingState, total_loss: Tensor) -> None:
    # Divided by k, so k accumulated batches add up to one averaged gradient. A
    # partial group at the end of an epoch is divided by k too, so it counts slightly
    # less.
    loss = total_loss / state.accumulate_grad_batches
    if state.scaler is not None:  # fp16: scale up so small gradients don't underflow
        loss = state.scaler.scale(loss)
    loss.backward()
    state.used_backward_helper = True
    state.emit("on_after_backward")


def optimizer_step(state: TrainingState) -> None:
    """One optimizer update from the accumulated gradients, which it then clears."""
    if state.scaler is not None:
        state.scaler.unscale_(state.optimizer)  # clip the true gradients
    if state.gradient_clip_norm is not None:
        params = [p for group in state.optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, state.gradient_clip_norm)
    if state.scaler is not None:
        state.scaler.step(state.optimizer)  # skipped if any gradient overflowed
        state.scaler.update()
    else:
        state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    state.emit("on_after_optimizer_step")


def default_step(state: TrainingState, batch: Batch) -> StepOutput:
    # The maths of one batch. The trainer's loop does the counting, scheduling,
    # logging and validation around it, so a replacement step can't break those.
    total, terms = forward_and_loss(state, batch)
    backward(state, total)
    if state.is_accumulation_boundary:
        optimizer_step(state)
    return StepOutput(total.detach(), terms)
