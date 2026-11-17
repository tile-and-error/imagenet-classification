from collections.abc import Sequence
from typing import Any

from torch import Tensor

from lwfm.core.types import Batch, Name

BATCH_GROUPS = ("inputs", "targets", "meta")


class UnknownNameError(KeyError):
    """A name matches no batch value and no node output.

    The message gives the name and lists the node outputs that do exist.
    """


def resolve(name: Name, batch: Batch, outputs: dict[Name, Tensor]) -> Any:
    if name in outputs:
        return outputs[name]
    group, _, key = name.partition(".")
    if group in BATCH_GROUPS and key in batch.get(group, {}):
        return batch[group][key]
    raise UnknownNameError(f"no value named {name!r}; node outputs: {sorted(outputs)}")


def resolve_all(
    names: Sequence[Name], batch: Batch, outputs: dict[Name, Tensor]
) -> list[Any]:
    return [resolve(name, batch, outputs) for name in names]


def as_list(names: Name | Sequence[Name]) -> list[Name]:
    return [names] if isinstance(names, str) else list(names)
