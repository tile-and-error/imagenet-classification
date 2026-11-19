from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from lwfm.core.names import BATCH_GROUPS, as_list, describe_values, resolve_all
from lwfm.core.types import Batch, Name


class GraphError(ValueError):
    """The graph's wiring is invalid, found while building or running it.

    Raised for a bad or duplicate node name, a node reading a name that nothing
    writes, a cycle, a module returning a different number of outputs than its
    node declares, or an output that loses the batch dimension.
    """


class ModelNode(nn.Module):
    """Wraps an unchanged `nn.Module` so the graph can feed it and read its outputs.

    `inputs` names the values passed to the module's `forward`, in order.
    `outputs` names what it returns, one name per returned item, published as
    `<node name>.<output>`:

        ModelNode("student", resnet18(), inputs="inputs.image", outputs="logits")

    reads `batch["inputs"]["image"]` and produces `"student.logits"`.
    """

    def __init__(
        self,
        name: str,
        module: nn.Module,
        inputs: Name | Sequence[Name],
        outputs: str | Sequence[str],
    ):
        super().__init__()
        if "." in name or name in BATCH_GROUPS:
            raise GraphError(
                f"node name {name!r} can't contain '.' or be a batch group"
            )
        self.name = name
        self.module = module
        self.inputs = as_list(inputs)
        self.outputs = as_list(outputs)
        if len(set(self.outputs)) < len(self.outputs):
            raise GraphError(
                f"{name!r} writes the same output field twice: {self.outputs}"
            )

    @property
    def output_names(self) -> list[Name]:
        return [f"{self.name}.{field}" for field in self.outputs]

    def forward(self, *input_values: Tensor) -> dict[Name, Tensor]:
        module_result = self.module(*input_values)
        if isinstance(module_result, Tensor):
            module_result = (module_result,)
        output_values = (
            [module_result[field] for field in self.outputs]
            if isinstance(module_result, dict)
            else module_result
        )
        if len(output_values) != len(self.outputs):
            raise GraphError(
                f"{self.name!r} returned {len(output_values)} values but writes "
                f"{self.outputs}"
            )
        return dict(zip(self.output_names, output_values, strict=True))


class ExecutionGraph(nn.Module):
    """Runs `ModelNode`s in dependency order and returns their outputs by name.

    Calling `graph(batch)` returns a flat `{name: Tensor}` dict with every node's
    outputs; the batch itself is not copied in. A single model is a one-node
    graph. Nodes are stored in an `nn.ModuleDict`, so parameters, `.to()`,
    `.train()`/`.eval()` and `state_dict()` work as for any module.

    Building the graph checks the wiring (see `GraphError`) and fixes
    `node_run_order`. Given a `preflight_sample` batch, it also runs
    `preflight(preflight_sample)` and keeps `preflight_output_shapes`; otherwise
    that is `None`.
    """

    def __init__(self, nodes: list[ModelNode], preflight_sample: Batch | None = None):
        super().__init__()
        self.node_run_order = compute_node_run_order(nodes)
        self.nodes = nn.ModuleDict({node.name: node for node in nodes})
        self.preflight_output_shapes = (
            None if preflight_sample is None else self.preflight(preflight_sample)
        )

    def forward(self, batch: Batch) -> dict[Name, Tensor]:
        node_outputs: dict[Name, Tensor] = {}
        for node_name in self.node_run_order:
            # We can assume all nodes in our graph are ModelNode by construction
            node = cast(ModelNode, self.nodes[node_name])
            input_values = resolve_all(node.inputs, batch, node_outputs)
            try:
                node_outputs |= node(*input_values)
            except Exception as error:
                error.add_note(
                    f"in node {node_name!r}, reading "
                    f"{describe_values(node.inputs, input_values)}"
                )
                raise
        return node_outputs

    def preflight(self, preflight_sample: Batch) -> dict[Name, tuple[int, ...]]:
        """Runs the graph once on `preflight_sample`, so a shape mismatch fails
        now, with a note naming the node, rather than mid-training. Also checks
        every output keeps the batch size as its first dimension. Runs in eval mode
        without gradients and restores the mode. The sample must be on the graph's
        device. Returns `{output name: shape}`.
        """
        sample_batch_size = batch_size(preflight_sample)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                node_outputs = self(preflight_sample)
        finally:
            self.train(was_training)
        for output_name, value in node_outputs.items():
            if value.ndim == 0 or value.shape[0] != sample_batch_size:
                raise GraphError(
                    f"{output_name!r} has shape {tuple(value.shape)}; its first "
                    f"dimension should be the batch size, {sample_batch_size}"
                )
        return {name: tuple(value.shape) for name, value in node_outputs.items()}

    def deployable_model(self, node_name: str) -> nn.Module:
        """Return the node's module: a plain `nn.Module`, sharing the graph's weights.

        It has no names, graph or losses, so it can be saved, exported or wrapped
        for TIAToolbox on its own.
        """
        return cast(ModelNode, self.nodes[node_name]).module


def compute_node_run_order(nodes: list[ModelNode]) -> list[str]:
    # Each node runs after every node it reads from (a topological sort).
    node_names = [node.name for node in nodes]
    if len(set(node_names)) < len(node_names):
        raise GraphError(f"node names must be unique: {node_names}")
    producer_of_output = {
        output_name: node.name for node in nodes for output_name in node.output_names
    }
    upstream_nodes: dict[str, set[str]] = {}
    for node in nodes:
        upstream_nodes[node.name] = set()
        for input_name in node.inputs:
            if input_name in producer_of_output:
                upstream_nodes[node.name].add(producer_of_output[input_name])
            elif input_name.partition(".")[0] not in BATCH_GROUPS:
                raise GraphError(
                    f"{node.name!r} reads {input_name!r}, which no node writes"
                )
    run_order: list[str] = []
    while upstream_nodes:
        ready_nodes = sorted(
            name
            for name, upstream in upstream_nodes.items()
            if upstream <= set(run_order)
        )
        if not ready_nodes:
            raise GraphError(f"cycle among nodes {sorted(upstream_nodes)}")
        run_order += ready_nodes
        for name in ready_nodes:
            del upstream_nodes[name]
    return run_order


def batch_size(batch: Batch) -> int:
    for value in batch.get("inputs", {}).values():
        if isinstance(value, Tensor) and value.ndim > 0:
            return value.shape[0]
    raise ValueError("the sample batch has no input tensors")
