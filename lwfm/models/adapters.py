import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lwfm.core.types import InputSpec


class InputAdapter(nn.Module):
    """Prepares images in `[0, 1]` for one model, as its `InputSpec` describes.

    Resizes `[B, C, H, W]` images to `spec.image_size` only if their size differs,
    then normalises with `(x - mean) / std`. Mean and std are buffers that move
    with `.to()` but are not saved in `state_dict()`; they are rebuilt from the spec.
    """

    mean: Tensor
    std: Tensor

    def __init__(self, spec: InputSpec):
        super().__init__()
        self.spec = spec
        mean = torch.tensor(spec.mean).view(-1, 1, 1)
        std = torch.tensor(spec.std).view(-1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, image: Tensor) -> Tensor:
        if tuple(image.shape[-2:]) != self.spec.image_size:
            image = F.interpolate(
                image, size=self.spec.image_size, mode="bilinear", antialias=True
            )
        return (image - self.mean) / self.std
