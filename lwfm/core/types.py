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


@dataclass(frozen=True)
class ArtifactRef:
    """A reference to a file a stage saved, such as a checkpoint or a metrics table.

    `path` is inside the run directory unless an external path was approved.
    Events and stage results pass artifacts by reference, never by content.
    """

    kind: str  # "torch_checkpoint", "metrics", "model_export"
    path: str
    producer_stage: str
    sha256: str | None = None
