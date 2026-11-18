from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from lwfm.core.names import as_list, describe, resolve_all
from lwfm.core.types import Batch, Name


@dataclass
class LossBinding:
    """Pairs a loss with the values it reads and its weight in the total.

    `loss` is any callable returning a scalar tensor, e.g. `nn.CrossEntropyLoss()`.
    `reads` lists the values passed to it, in argument order, e.g.
    `["student.logits", "targets.class"]` for `CrossEntropyLoss(input, target)`.
    """

    name: str
    loss: Callable[..., Tensor]
    reads: Name | Sequence[Name]
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.reads = as_list(self.reads)


class LossComposer(nn.Module):
    """Computes the weighted sum of several losses: total = sum(weight * loss).

    Calling `composer(batch, outputs)` returns `(total, terms)`: `total` for
    `backward()`, and `terms`, each loss by name and detached, for logging. Raises
    `ValueError` if a loss isn't a finite scalar. Losses that are `nn.Module`s are
    registered, so any weights they have are trained and moved with `.to()`.
    """

    def __init__(self, bindings: list[LossBinding]):
        super().__init__()
        if not bindings:
            raise ValueError("LossComposer needs at least one LossBinding")
        self.bindings = bindings
        # Registered: a loss with weights appears in .parameters() and moves with .to().
        self.losses = nn.ModuleList(
            b.loss for b in bindings if isinstance(b.loss, nn.Module)
        )

    def forward(
        self, batch: Batch, outputs: dict[Name, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        terms = {}
        for b in self.bindings:
            values = resolve_all(b.reads, batch, outputs)
            try:
                terms[b.name] = b.loss(*values)
            except Exception as error:
                error.add_note(
                    f"in loss {b.name!r}, reading {describe(b.reads, values)}"
                )
                raise
        for name, term in terms.items():
            if term.ndim != 0 or not torch.isfinite(term):
                raise ValueError(f"loss {name!r} is not a finite scalar: {term}")
        total = cast(Tensor, sum(b.weight * terms[b.name] for b in self.bindings))
        return total, {name: term.detach() for name, term in terms.items()}
