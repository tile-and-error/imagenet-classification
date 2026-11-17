from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lwfm.core.ports import resolve_all
from lwfm.core.types import Batch, Port


@dataclass
class LossBinding:
    name: str
    loss: Callable[..., Tensor]
    inputs: dict[str, Port]  # loss argument -> port
    weight: float = 1.0


class LossComposer(nn.Module):
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
        self, batch: Batch, outputs: dict[Port, Tensor]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        terms = {
            b.name: b.loss(**resolve_all(b.inputs, batch, outputs))
            for b in self.bindings
        }
        for name, term in terms.items():
            if term.ndim != 0 or not torch.isfinite(term):
                raise ValueError(f"loss {name!r} is not a finite scalar: {term}")
        total = sum(b.weight * terms[b.name] for b in self.bindings)
        return total, {name: term.detach() for name, term in terms.items()}
