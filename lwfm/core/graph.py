from collections.abc import Callable, Sequence
from itertools import chain
from typing import Literal, cast

import torch
from torch import Tensor, nn

from lwfm.core.batch import to_device
from lwfm.core.names import BATCH_GROUPS, as_list, describe, resolve_all
from lwfm.core.types import Batch, InputSpec, Name
from lwfm.models.adapters import InputAdapter

DEFAULT_BATCH_SIZE = 2  # > 1, so a squeeze() that drops the batch dimension is caught
DEFAULT_TARGETS: dict[str, Callable[[int], Tensor]] = {
    "class": lambda n: torch.zeros(n, dtype=torch.long),
}


class GraphError(ValueError):
    """The graph's wiring is invalid, found while building or running it.

    Raised for a bad or duplicate node name, a node reading a name that nothing
    writes, a cycle, a module returning a different number of outputs than its
    node declares, or an output that loses the batch dimension.
    """


class ModelNode(nn.Module):
    """Wraps an unchanged `nn.Module` so the graph can feed it and read its outputs.

    `reads` lists the values passed to `module.forward`, in argument order, e.g.
    `"inputs.image"` or `["a.y", "b.y"]`. `writes` names what the module returns:
    a Tensor gets one name, a tuple one name per item in order, and a dict keeps
    only the listed fields. The node stores each under its own name, so
    `ModelNode("student", ..., writes="logits")` produces `"student.logits"`.
    Build straight chains, such as preprocessing then a model, with plain torch:
    `nn.Sequential(InputAdapter(spec), resnet18())`.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        reads: Name | Sequence[Name],
        writes: str | Sequence[str],
    ):
        super().__init__()
        if "." in name or name in BATCH_GROUPS:
            raise GraphError(
                f"node name {name!r} can't contain '.' or be a batch group"
            )
        self.name = name
        self.module = module
        self.reads = as_list(reads)
        self.writes = as_list(writes)
        if len(set(self.writes)) < len(self.writes):
            raise GraphError(f"{name!r} writes the same name twice: {self.writes}")

    @property
    def outputs(self) -> list[Name]:
        return [f"{self.name}.{field}" for field in self.writes]

    def forward(self, *inputs: Tensor) -> dict[Name, Tensor]:
        result = self.module(*inputs)
        if isinstance(result, Tensor):
            result = (result,)
        values = (
            [result[f] for f in self.writes] if isinstance(result, dict) else result
        )
        if len(values) != len(self.writes):
            raise GraphError(
                f"{self.name!r} returned {len(values)} values but writes {self.writes}"
            )
        return dict(zip(self.outputs, values, strict=True))


class ExecutionGraph(nn.Module):
    """Runs `ModelNode`s in dependency order and returns their outputs by name.

    Calling `graph(batch)` returns a flat `{name: Tensor}` dict with every node's
    outputs; the batch itself is not copied in. A single model is a one-node
    graph. Nodes are stored in an `nn.ModuleDict`, so parameters, `.to()`,
    `.train()`/`.eval()` and `state_dict()` work as for any module.

    Building the graph checks the wiring (see `GraphError`), fixes the run order,
    then runs `preflight(sample)` to check shapes: `sample` is a batch,
    `"default"` (zeros, see `default_batch`) or `None` (skip). The output shapes
    are kept in `graph.shapes`.
    """

    def __init__(
        self,
        nodes: list[ModelNode],
        sample: Batch | Literal["default"] | None = "default",
    ):
        super().__init__()
        self.order = run_order(nodes)
        self.nodes = nn.ModuleDict({node.name: node for node in nodes})
        self.shapes = self.preflight(sample)

    def forward(self, batch: Batch) -> dict[Name, Tensor]:
        outputs: dict[Name, Tensor] = {}
        for name in self.order:
            node = cast(ModelNode, self.nodes[name])
            inputs = resolve_all(node.reads, batch, outputs)
            try:
                outputs |= node(*inputs)
            except Exception as error:
                error.add_note(
                    f"in node {name!r}, reading {describe(node.reads, inputs)}"
                )
                raise
        return outputs

    def preflight(
        self, sample: Batch | Literal["default"] | None = "default"
    ) -> dict[Name, tuple[int, ...]] | None:
        """Runs the graph once on `sample`, so a shape mismatch fails now, with a
        note naming the node, rather than mid-training. Also checks every output
        keeps the batch size as its first dimension. Runs in eval mode without
        gradients and restores the mode. Returns `{output name: shape}`, or
        `None` if `sample` is `None`.
        """
        if sample is None:
            return None
        if isinstance(sample, str):
            if sample != "default":
                raise ValueError(
                    f"sample must be a batch, 'default' or None, not {sample!r}"
                )
            sample = default_batch(self)
        tensor = next(chain(self.parameters(), self.buffers()), None)
        batch = to_device(
            sample, tensor.device if tensor is not None else torch.device("cpu")
        )
        size = batch_size(batch)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                outputs = self(batch)
        finally:
            self.train(was_training)
        for name, value in outputs.items():
            if value.ndim == 0 or value.shape[0] != size:
                raise GraphError(
                    f"{name!r} has shape {tuple(value.shape)}; its first dimension "
                    f"should be the batch size, {size}"
                )
        return {name: tuple(value.shape) for name, value in outputs.items()}

    def deployable_model(self, node: str) -> nn.Module:
        """Return the node's module: a plain `nn.Module`, sharing the graph's weights.

        It has no names, graph or losses, so it can be saved, exported or wrapped
        for TIAToolbox on its own.
        """
        return cast(ModelNode, self.nodes[node]).module


def run_order(nodes: list[ModelNode]) -> list[str]:
    # Each node runs after every node it reads from (a topological sort).
    names = [n.name for n in nodes]
    if len(set(names)) < len(names):
        raise GraphError(f"node names must be unique: {names}")
    producer = {out: n.name for n in nodes for out in n.outputs}
    needs: dict[str, set[str]] = {}
    for n in nodes:
        needs[n.name] = set()
        for name in n.reads:
            if name in producer:
                needs[n.name].add(producer[name])
            elif name.partition(".")[0] not in BATCH_GROUPS:
                raise GraphError(f"{n.name!r} reads {name!r}, which no node writes")
    order: list[str] = []
    while needs:
        ready = sorted(n for n, deps in needs.items() if deps <= set(order))
        if not ready:
            raise GraphError(f"cycle among nodes {sorted(needs)}")
        order += ready
        for n in ready:
            del needs[n]
    return order


def default_batch(graph: ExecutionGraph, size: int = DEFAULT_BATCH_SIZE) -> Batch:
    """Zeros for every batch value the graph's nodes read.

    An `inputs.<key>` is shaped `[size, C, H, W]` from the `InputSpec` of the
    `InputAdapter` in the first node that takes it as its first argument. A
    `targets.<key>` comes from `DEFAULT_TARGETS` (class labels of 0). Anything
    else needs a real sample batch.
    """
    nodes = [cast(ModelNode, graph.nodes[name]) for name in graph.order]
    batch: Batch = {group: {} for group in BATCH_GROUPS}
    for name in dict.fromkeys(r for n in nodes for r in n.reads):
        group, _, key = name.partition(".")
        if group not in BATCH_GROUPS:
            continue  # a node output
        spec = input_spec(nodes, name) if group == "inputs" else None
        if spec is not None:
            batch[group][key] = torch.zeros(size, spec.channels, *spec.image_size)
        elif group == "targets" and key in DEFAULT_TARGETS:
            batch[group][key] = DEFAULT_TARGETS[key](size)
        else:
            raise ValueError(
                f"no default value for {name!r}; pass a sample batch, or "
                "sample=None to skip the shape check"
            )
    return batch


def input_spec(nodes: Sequence[ModelNode], name: Name) -> InputSpec | None:
    for node in nodes:
        if node.reads[:1] == [name]:
            for module in node.module.modules():
                if isinstance(module, InputAdapter):
                    return module.spec
    return None


def batch_size(batch: Batch) -> int:
    for value in batch.get("inputs", {}).values():
        if isinstance(value, Tensor) and value.ndim > 0:
            return value.shape[0]
    raise ValueError("the sample batch has no input tensors")
