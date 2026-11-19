from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lwfm.core.names import as_list, describe_values, resolve_all
from lwfm.core.types import Batch, Name


@dataclass
class LossBinding:
    """Pairs a loss with the values it reads and its weight in the total.

    `loss` is any callable returning a scalar tensor, e.g. `nn.CrossEntropyLoss()`.
    `inputs` lists the values passed to it, in argument order, e.g.
    `["student.logits", "targets.class"]` for `CrossEntropyLoss(input, target)`.
    """

    name: str
    loss: Callable[..., Tensor]
    inputs: Name | Sequence[Name]
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.inputs = as_list(self.inputs)


class LossComposer(nn.Module):
    """Computes the weighted sum of several losses: total = sum(weight * loss).

    Calling `composer(batch, node_outputs)` returns `(total_loss, loss_terms)`:
    `total_loss` for `backward()`, and `loss_terms`, each loss by name and
    detached, for logging. Raises `ValueError` if a loss isn't a finite scalar.
    Losses that are `nn.Module`s are registered, so any weights they have are
    trained and moved with `.to()`.
    """

    def __init__(self, loss_bindings: list[LossBinding]):
        super().__init__()
        if not loss_bindings:
            raise ValueError("LossComposer needs at least one LossBinding")
        self.loss_bindings = loss_bindings

        # Ensure loss names are unique
        loss_names = [b.name for b in self.loss_bindings]
        if len(set(loss_names)) < len(loss_names):
            raise ValueError(f"loss names must be unique, got {loss_names}")

        # Register losses with weights, to appear in .parameters() and move with .to()
        # Pitfall: Assigning an nn.Module to an attribute of a module registers it,
        # but mutating it inside a plain list or dataclass doesn't. After
        # composer.loss_bindings[0].loss = other, other isn't registered. To swap a
        # loss, build a new LossComposer.
        self.loss_modules = nn.ModuleList(
            binding.loss
            for binding in loss_bindings
            if isinstance(binding.loss, nn.Module)
        )

    def forward(
        self, batch: Batch, node_outputs: dict[Name, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        loss_terms = {}
        for binding in self.loss_bindings:
            input_values = resolve_all(binding.inputs, batch, node_outputs)
            try:
                loss_terms[binding.name] = binding.loss(*input_values)
            except Exception as error:
                error.add_note(
                    f"in loss {binding.name!r}, reading "
                    f"{describe_values(binding.inputs, input_values)}"
                )
                raise
        for name, term in loss_terms.items():
            if term.ndim != 0 or not torch.isfinite(term):
                raise ValueError(f"loss {name!r} is not a finite scalar: {term}")
        total_loss = torch.stack(
            [
                binding.weight * loss_terms[binding.name]
                for binding in self.loss_bindings
            ]
        ).sum()
        return total_loss, {name: term.detach() for name, term in loss_terms.items()}
