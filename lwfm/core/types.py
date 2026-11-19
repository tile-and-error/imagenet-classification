from typing import Any

type Name = str  # "inputs.image" (from the batch) or "student.logits" (from a node)
type Batch = dict[str, Any]
