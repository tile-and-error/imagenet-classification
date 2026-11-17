import pytest
import torch
from torch import Tensor, nn

from lwfm.core.graph import ExecutionGraph, GraphError, ModelNode


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: Tensor) -> Tensor:
        return self.w * x


class TwoHeads(nn.Module):
    def forward(self, pixels: Tensor) -> dict[str, Tensor]:
        return {"logits": pixels.sum(dim=1), "features": pixels, "unused": pixels}


class Add(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a + b


class AddOne(nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        return image + 1


@pytest.fixture
def batch():
    return {
        "inputs": {"image": torch.ones(2, 3)},
        "targets": {"class": torch.tensor([0, 1])},
        "meta": {"sample_id": ["a", "b"]},
    }


def test_node_maps_tensor_return_to_its_port():
    n = ModelNode("student", Scale(), {"x": "batch.inputs.image"}, {"y": "student.y"})
    out = n(x=torch.ones(2))
    assert set(out) == {"student.y"}
    torch.testing.assert_close(out["student.y"], torch.full((2,), 2.0))


def test_node_maps_argument_names_and_dict_fields():
    n = ModelNode(
        "student",
        TwoHeads(),
        {"pixels": "batch.inputs.image"},
        {"logits": "student.logits", "features": "student.features"},
    )
    out = n(pixels=torch.ones(2, 3))
    assert set(out) == {"student.logits", "student.features"}  # "unused" dropped
    assert out["student.logits"].shape == (2,)


def test_node_applies_input_adapter_before_module():
    n = ModelNode(
        "s",
        Scale(),
        {"x": "batch.inputs.image"},
        {"y": "s.y"},
        input_adapter=AddOne(),
    )
    torch.testing.assert_close(n(x=torch.zeros(1))["s.y"], torch.tensor([2.0]))


def test_one_node_graph_reads_batch_and_returns_flat_ports(batch):
    graph = ExecutionGraph(
        [ModelNode("student", Scale(), {"x": "batch.inputs.image"}, {"y": "student.y"})]
    )
    out = graph(batch)
    assert set(out) == {"student.y"}
    torch.testing.assert_close(out["student.y"], 2 * batch["inputs"]["image"])


def test_nodes_run_in_dependency_order_not_list_order(batch):
    graph = ExecutionGraph(
        [
            ModelNode("c", Add(), {"a": "a.y", "b": "b.y"}, {"y": "c.y"}),
            ModelNode("b", Scale(), {"x": "a.y"}, {"y": "b.y"}),
            ModelNode("a", Scale(), {"x": "batch.inputs.image"}, {"y": "a.y"}),
        ]
    )
    assert graph.order == ["a", "b", "c"]
    x = batch["inputs"]["image"]
    torch.testing.assert_close(graph(batch)["c.y"], 2 * x + 4 * x)


def test_graph_is_a_normal_module(batch):
    graph = ExecutionGraph(
        [ModelNode("student", Scale(), {"x": "batch.inputs.image"}, {"y": "student.y"})]
    )
    assert isinstance(graph, nn.Module)
    assert len(list(graph.parameters())) == 1
    assert list(graph.state_dict()) == ["nodes.student.module.w"]
    graph.eval()
    assert not graph.nodes["student"].module.training
    graph.train()
    assert graph.nodes["student"].module.training

    clone = ExecutionGraph(
        [ModelNode("student", Scale(), {"x": "batch.inputs.image"}, {"y": "student.y"})]
    )
    with torch.no_grad():
        graph.nodes["student"].module.w.fill_(3.0)
    clone.load_state_dict(graph.state_dict())
    torch.testing.assert_close(clone(batch)["student.y"], graph(batch)["student.y"])


def test_gradients_reach_node_parameters(batch):
    graph = ExecutionGraph(
        [ModelNode("student", Scale(), {"x": "batch.inputs.image"}, {"y": "student.y"})]
    )
    graph(batch)["student.y"].sum().backward()
    assert graph.nodes["student"].module.w.grad is not None


def test_deployable_model_matches_node_and_shares_weights():
    student = ModelNode(
        "student",
        Scale(),
        {"x": "batch.inputs.image"},
        {"y": "student.y"},
        input_adapter=AddOne(),
    )
    graph = ExecutionGraph([student])
    model = graph.deployable_model("student")
    x = torch.zeros(3)
    torch.testing.assert_close(model(x), student(x=x)["student.y"])
    assert model[1].w is student.module.w


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            [
                ModelNode("a", Scale(), {"x": "batch.inputs.image"}, {"y": "a.y"}),
                ModelNode("a", Scale(), {"x": "batch.inputs.image"}, {"y": "a.z"}),
            ],
            "unique",
        ),
        (
            [
                ModelNode("a", Scale(), {"x": "batch.inputs.image"}, {"y": "a.y"}),
                ModelNode("b", Scale(), {"x": "batch.inputs.image"}, {"y": "a.y"}),
            ],
            "unique",
        ),
        (
            [ModelNode("a", Scale(), {"x": "studnet.y"}, {"y": "a.y"})],
            "studnet.y",
        ),
        (
            [
                ModelNode("a", Scale(), {"x": "b.y"}, {"y": "a.y"}),
                ModelNode("b", Scale(), {"x": "a.y"}, {"y": "b.y"}),
            ],
            "cycle",
        ),
    ],
    ids=["duplicate_name", "duplicate_port", "unknown_port", "cycle"],
)
def test_invalid_graphs_fail_at_build_time(nodes, message):
    with pytest.raises(GraphError, match=message):
        ExecutionGraph(nodes)
