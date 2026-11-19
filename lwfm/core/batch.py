import torch
from torch import Tensor

from lwfm.core.types import Batch


def to_device(batch: Batch, device: torch.device) -> Batch:
    """Moves every tensor in the batch's groups to `device`; other values (ids,
    paths) are kept as they are. Copies to an accelerator don't block the CPU when
    the batch is in pinned memory (`DataLoader(pin_memory=True)`)."""
    # A non-blocking copy to the CPU could be read before it lands, so only
    # copies to an accelerator are non-blocking; later GPU work waits for them.
    non_blocking = torch.device(device).type != "cpu"
    return {
        group: {
            key: value.to(device, non_blocking=non_blocking)
            if isinstance(value, Tensor)
            else value
            for key, value in group_values.items()
        }
        if isinstance(group_values, dict)
        else group_values
        for group, group_values in batch.items()
    }
