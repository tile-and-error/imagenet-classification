from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn

from lwfm.core.names import BATCH_GROUPS, as_list, resolve_all
from lwfm.core.types import Batch, Name


class GraphError(ValueError):
    """The graph's wiring is invalid, found while building or running it.

    Raised for a bad or duplicate node name, a node reading a name that nothing
    writes, a cycle, or a module returning a different number of outputs than
    its node declares.
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
    outputs; the batch itself is not copied in. The run order is fixed and the
    wiring is checked when the graph is built (see `GraphError`). A single model
    is a one-node graph. Nodes are stored in an `nn.ModuleDict`, so parameters,
    `.to()`, `.train()`/`.eval()` and `state_dict()` work as for any module.
    """

    def __init__(self, nodes: list[ModelNode]):
        super().__init__()
        self.order = run_order(nodes)
        self.nodes = nn.ModuleDict({node.name: node for node in nodes})

    def forward(self, batch: Batch) -> dict[Name, Tensor]:
        outputs: dict[Name, Tensor] = {}
        for name in self.order:
            node = cast(ModelNode, self.nodes[name])
            outputs |= node(*resolve_all(node.reads, batch, outputs))
        return outputs

    def deployable_model(self, node: str = "student") -> nn.Module:
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
