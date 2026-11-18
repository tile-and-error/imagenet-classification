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


class Pair(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x + 1, x - 1


class Sub(nn.Module):
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return a - b


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


def student() -> ModelNode:
    return ModelNode("student", Scale(), reads="inputs.image", writes="y")


def test_node_writes_tensor_under_its_own_name():
    node = student()
    assert node.reads == ["inputs.image"]
    assert node.outputs == ["student.y"]
    out = node(torch.ones(2))
    assert set(out) == {"student.y"}
    torch.testing.assert_close(out["student.y"], torch.full((2,), 2.0))


def test_node_keeps_only_listed_dict_fields():
    node = ModelNode(
        "student", TwoHeads(), reads="inputs.image", writes=["logits", "features"]
    )
    out = node(torch.ones(2, 3))
    assert set(out) == {"student.logits", "student.features"}  # "unused" dropped
    assert out["student.logits"].shape == (2,)


def test_node_names_tuple_items_in_order():
    out = ModelNode("p", Pair(), reads="inputs.image", writes=["up", "down"])(
        torch.zeros(1)
    )
    torch.testing.assert_close(out["p.up"], torch.tensor([1.0]))
    torch.testing.assert_close(out["p.down"], torch.tensor([-1.0]))


def test_node_passes_reads_by_position():
    node = ModelNode("d", Sub(), reads=["a.y", "b.y"], writes="y")
    out = node(torch.tensor(5.0), torch.tensor(3.0))  # forward(a=5, b=3)
    torch.testing.assert_close(out["d.y"], torch.tensor(2.0))


def test_node_runs_a_sequential_chain_in_order():
    node = ModelNode(
        "s", nn.Sequential(AddOne(), Scale()), reads="inputs.image", writes="y"
    )
    torch.testing.assert_close(node(torch.zeros(1))["s.y"], torch.tensor([2.0]))


def test_wrong_number_of_outputs_fails():
    node = ModelNode("p", Pair(), reads="inputs.image", writes="only_one")
    with pytest.raises(GraphError, match="returned 2 values"):
        node(torch.zeros(1))


@pytest.mark.parametrize("name", ["a.b", "inputs", "targets", "meta"])
def test_bad_node_names_fail(name):
    with pytest.raises(GraphError, match="node name"):
        ModelNode(name, Scale(), reads="inputs.image", writes="y")


def test_duplicate_writes_fail():
    with pytest.raises(GraphError, match="twice"):
        ModelNode("s", Pair(), reads="inputs.image", writes=["y", "y"])


def test_one_node_graph_reads_batch_and_returns_flat_names(batch):
    graph = ExecutionGraph([student()], sample=batch)
    out = graph(batch)
    assert set(out) == {"student.y"}
    torch.testing.assert_close(out["student.y"], 2 * batch["inputs"]["image"])


def test_nodes_run_in_dependency_order_not_list_order(batch):
    graph = ExecutionGraph(
        [
            ModelNode("c", Sub(), reads=["b.y", "a.y"], writes="y"),
            ModelNode("b", Scale(), reads="a.y", writes="y"),
            ModelNode("a", Scale(), reads="inputs.image", writes="y"),
        ],
        sample=batch,
    )
    assert graph.order == ["a", "b", "c"]
    x = batch["inputs"]["image"]
    torch.testing.assert_close(graph(batch)["c.y"], 4 * x - 2 * x)


def test_graph_is_a_normal_module(batch):
    graph = ExecutionGraph([student()], sample=batch)
    assert isinstance(graph, nn.Module)
    assert len(list(graph.parameters())) == 1
    assert list(graph.state_dict()) == ["nodes.student.module.w"]
    graph.eval()
    assert not graph.nodes["student"].module.training
    graph.train()
    assert graph.nodes["student"].module.training

    clone = ExecutionGraph([student()], sample=batch)
    with torch.no_grad():
        graph.nodes["student"].module.w.fill_(3.0)
    clone.load_state_dict(graph.state_dict())
    torch.testing.assert_close(clone(batch)["student.y"], graph(batch)["student.y"])


def test_gradients_reach_node_parameters(batch):
    graph = ExecutionGraph([student()], sample=batch)
    graph(batch)["student.y"].sum().backward()
    assert graph.nodes["student"].module.w.grad is not None


def test_deployable_model_is_the_nodes_module():
    chain = nn.Sequential(AddOne(), Scale())
    node = ModelNode("student", chain, reads="inputs.image", writes="y")
    model = ExecutionGraph([node], sample=None).deployable_model("student")
    assert model is chain  # the same object, so the same weights
    x = torch.zeros(3)
    torch.testing.assert_close(model(x), node(x)["student.y"])


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            [
                ModelNode("a", Scale(), reads="inputs.image", writes="y"),
                ModelNode("a", Scale(), reads="inputs.image", writes="z"),
            ],
            "unique",
        ),
        (
            [ModelNode("a", Scale(), reads="studnet.y", writes="y")],
            "studnet.y",
        ),
        (
            [
                ModelNode("a", Scale(), reads="b.y", writes="y"),
                ModelNode("b", Scale(), reads="a.y", writes="y"),
            ],
            "cycle",
        ),
    ],
    ids=["duplicate_name", "unknown_name", "cycle"],
)
def test_invalid_graphs_fail_at_build_time(nodes, message):
    with pytest.raises(GraphError, match=message):
        ExecutionGraph(nodes)
