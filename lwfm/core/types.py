from dataclasses import dataclass
from typing import Any

type Name = str  # "inputs.image" (from the batch) or "student.logits" (from a node)
type Sample = dict[str, Any]  # {"inputs": {...}, "targets": {...}, "meta": {...}}
type Batch = dict[str, Any]


@dataclass(frozen=True)
class InputSpec:
    """What a model expects as input: channel count, image size, and normalisation.

    Each model states its own spec, so a teacher and a student can expect different
    sizes. `InputAdapter` applies it; runs log it so the input is always recorded.
    """

    channels: int = 3
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
