import torch
from torch import Tensor

from lwfm.core.types import Batch


def to_device(batch: Batch, device: torch.device) -> Batch:
    """Moves every tensor in the batch's groups to `device`; other values (ids,
    paths) are kept as they are."""
    return {
        group: {
            key: value.to(device) if isinstance(value, Tensor) else value
            for key, value in group_values.items()
        }
        if isinstance(group_values, dict)
        else group_values
        for group, group_values in batch.items()
    }
