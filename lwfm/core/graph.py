from torch import Tensor, nn

from lwfm.core.ports import resolve_all
from lwfm.core.types import Batch, Port


class GraphError(ValueError):
    pass


class ModelNode(nn.Module):
    def __init__(
        self,
        name: str,
        module: nn.Module,
        inputs: dict[str, Port],  # module argument -> port it reads
        outputs: dict[str, Port],  # module return field -> port it writes
        input_adapter: nn.Module | None = None,
    ):
        super().__init__()
        self.name = name
        self.module = module
        self.inputs = inputs
        self.outputs = outputs
        self.input_adapter = nn.Identity() if input_adapter is None else input_adapter

    def forward(self, **inputs: Tensor) -> dict[Port, Tensor]:
        result = self.module(**{k: self.input_adapter(v) for k, v in inputs.items()})
        if isinstance(result, Tensor):  # a plain Tensor is the one declared output
            (field,) = self.outputs
            result = {field: result}
        return {port: result[field] for field, port in self.outputs.items()}


class ExecutionGraph(nn.Module):
    def __init__(self, nodes: list[ModelNode]):
        super().__init__()
        self.order = run_order(nodes)
        self.nodes = nn.ModuleDict({node.name: node for node in nodes})

    def forward(self, batch: Batch) -> dict[Port, Tensor]:
        outputs: dict[Port, Tensor] = {}
        for name in self.order:
            node = self.nodes[name]
            outputs |= node(**resolve_all(node.inputs, batch, outputs))
        return outputs

    def deployable_model(self, node: str = "student") -> nn.Module:
        chosen = self.nodes[node]
        return nn.Sequential(chosen.input_adapter, chosen.module)  # shares weights


def run_order(nodes: list[ModelNode]) -> list[str]:
    # Each node runs after every node it reads from (a topological sort).
    producer = {port: n.name for n in nodes for port in n.outputs.values()}
    names = [n.name for n in nodes]
    n_outputs = sum(len(n.outputs) for n in nodes)
    if len(set(names)) < len(names) or len(producer) < n_outputs:
        raise GraphError("node names and output ports must be unique")
    needs: dict[str, set[str]] = {}
    for n in nodes:
        needs[n.name] = set()
        for port in n.inputs.values():
            if port in producer:
                needs[n.name].add(producer[port])
            elif not port.startswith("batch."):
                raise GraphError(f"{n.name!r} reads {port!r}, which no node writes")
    order: list[str] = []
    while needs:
        ready = sorted(n for n, deps in needs.items() if deps <= set(order))
        if not ready:
            raise GraphError(f"cycle among nodes {sorted(needs)}")
        order += ready
        for n in ready:
            del needs[n]
    return order
