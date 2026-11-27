from collections.abc import Sequence

from torch import Tensor

from lwfm.core.types import Batch, Name

BATCH_GROUPS = ("inputs", "targets", "meta")


class UnknownNameError(KeyError):
    """A name matches no batch value and no node output.

    The message gives the name and lists the node outputs that do exist.
    """


def resolve(value_name: Name, batch: Batch, node_outputs: dict[Name, Tensor]):
    """Look up a name: a node output (`student.logits`) first, else a batch value
    (`targets.class`, where the part before the first `.` is one of `BATCH_GROUPS`).
    Raises `UnknownNameError` if it is neither.
    """
    if value_name in node_outputs:
        return node_outputs[value_name]
    group, _, key = value_name.partition(".")
    if group in BATCH_GROUPS and key in batch.get(group, {}):
        return batch[group][key]
    raise UnknownNameError(
        f"no value named {value_name!r}; node outputs: {sorted(node_outputs)}"
    )


def resolve_all(
    value_names: Sequence[Name], batch: Batch, node_outputs: dict[Name, Tensor]
):
    """Resolves a list of names, see docs on resolve()."""
    return [resolve(name, batch, node_outputs) for name in value_names]


def as_list(one_or_many_names: Name | Sequence[Name]) -> list[Name]:
    """Converts the given input `one_or_many_names` to a list.
    For example as_list('a')=['a'], as_list(['a','b'])=['a','b']
    """
    if isinstance(one_or_many_names, str):
        return [one_or_many_names]
    return list(one_or_many_names)


def describe_values(value_names: Sequence[Name], values) -> str:
    """Describes values for use in error notes, for example:
    `"student.logits (8, 10), targets.class (8,)"`
    """
    return ", ".join(
        f"{name} "
        f"{tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__}"
        for name, value in zip(value_names, values, strict=True)
    )
