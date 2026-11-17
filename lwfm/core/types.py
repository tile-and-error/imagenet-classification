from dataclasses import dataclass
from typing import Any

type Port = str  # "batch.inputs.image", "student.logits"
type Sample = dict[str, Any]  # {"inputs": {...}, "targets": {...}, "meta": {...}}
type Batch = dict[str, Any]


@dataclass(frozen=True)
class InputSpec:
    channels: int = 3
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
