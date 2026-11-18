from dataclasses import dataclass
from typing import Any

type Name = str  # "inputs.image" (from the batch) or "student.logits" (from a node)
type Batch = dict[str, Any]


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
