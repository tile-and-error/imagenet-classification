import torch
from torch import Tensor

from lwfm.core.types import Batch


def to_device(batch: Batch, device: torch.device) -> Batch:
    """Moves every tensor in the batch's groups to `device`; other values (ids,
    paths) are kept as they are."""
    return {
        group: {
            k: v.to(device) if isinstance(v, Tensor) else v for k, v in values.items()
        }
        if isinstance(values, dict)
        else values
        for group, values in batch.items()
    }
