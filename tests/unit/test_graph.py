import pytest
import torch
from torch import Tensor, nn

from lwfm.core.graph import ExecutionGraph, GraphError, ModelNode
from lwfm.core.types import Batch
from lwfm.loss.losses import LossBinding


class Scale(nn.Module):
    def __init__(self) -> None:
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
def batch() -> Batch:
    return {
        "inputs": {"image": torch.ones(2, 3)},
        "targets": {"class": torch.tensor([0, 1])},
        "meta": {"sample_id": ["a", "b"]},
    }


def student(module: nn.Module | None = None) -> ModelNode:
    module = Scale() if module is None else module
    return ModelNode("student", module, inputs="inputs.image", outputs="y")


def test_node_writes_tensor_under_its_own_name() -> None:
    node = student()
    assert node.inputs == ["inputs.image"]
    assert node.output_names == ["student.y"]
    out = node(torch.ones(2))
    assert set(out) == {"student.y"}
    torch.testing.assert_close(out["student.y"], torch.full((2,), 2.0))


def test_node_keeps_only_listed_dict_fields() -> None:
    node = ModelNode(
        "student",
        TwoHeads(),
        inputs="inputs.image",
        outputs=["logits", "features"],
    )
    out = node(torch.ones(2, 3))
    assert set(out) == {"student.logits", "student.features"}  # "unused" dropped
    assert out["student.logits"].shape == (2,)


def test_node_names_tuple_items_in_order() -> None:
    out = ModelNode("p", Pair(), inputs="inputs.image", outputs=["up", "down"])(
        torch.zeros(1)
    )
    torch.testing.assert_close(out["p.up"], torch.tensor([1.0]))
    torch.testing.assert_close(out["p.down"], torch.tensor([-1.0]))


def test_node_passes_reads_by_position() -> None:
    node = ModelNode("d", Sub(), inputs=["a.y", "b.y"], outputs="y")
    out = node(torch.tensor(5.0), torch.tensor(3.0))  # forward(a=5, b=3)
    torch.testing.assert_close(out["d.y"], torch.tensor(2.0))


def test_node_runs_a_sequential_chain_in_order() -> None:
    node = ModelNode(
        "s",
        nn.Sequential(AddOne(), Scale()),
        inputs="inputs.image",
        outputs="y",
    )
    torch.testing.assert_close(node(torch.zeros(1))["s.y"], torch.tensor([2.0]))


def test_wrong_number_of_outputs_fails() -> None:
    node = ModelNode("p", Pair(), inputs="inputs.image", outputs="only_one")
    with pytest.raises(GraphError, match="returned 2 values"):
        node(torch.zeros(1))


@pytest.mark.parametrize("name", ["a.b", "inputs", "targets", "meta"])
def test_bad_node_names_fail(name: str) -> None:
    with pytest.raises(GraphError, match="node name"):
        ModelNode(name, Scale(), inputs="inputs.image", outputs="y")


def test_duplicate_writes_fail() -> None:
    with pytest.raises(GraphError, match="twice"):
        ModelNode("s", Pair(), inputs="inputs.image", outputs=["y", "y"])


def test_one_node_graph_reads_batch_and_returns_flat_names(batch: Batch) -> None:
    graph = ExecutionGraph([student()], preflight_sample=batch)
    out = graph(batch)
    assert set(out) == {"student.y"}
    torch.testing.assert_close(out["student.y"], 2 * batch["inputs"]["image"])


def test_nodes_run_in_dependency_order_not_list_order(batch: Batch) -> None:
    graph = ExecutionGraph(
        [
            ModelNode("c", Sub(), inputs=["b.y", "a.y"], outputs="y"),
            ModelNode("b", Scale(), inputs="a.y", outputs="y"),
            ModelNode("a", Scale(), inputs="inputs.image", outputs="y"),
        ],
        preflight_sample=batch,
    )
    assert graph.node_run_order == ["a", "b", "c"]
    x = batch["inputs"]["image"]
    torch.testing.assert_close(graph(batch)["c.y"], 4 * x - 2 * x)


def test_graph_is_a_normal_module(batch: Batch) -> None:
    scale = Scale()
    graph = ExecutionGraph([student(scale)], preflight_sample=batch)
    assert isinstance(graph, nn.Module)
    assert len(list(graph.parameters())) == 1
    assert list(graph.state_dict()) == ["nodes.student.module.w"]
    graph.eval()
    assert not scale.training
    graph.train()
    assert scale.training

    clone = ExecutionGraph([student()], preflight_sample=batch)
    with torch.no_grad():
        scale.w.fill_(3.0)
    clone.load_state_dict(graph.state_dict())
    torch.testing.assert_close(clone(batch)["student.y"], graph(batch)["student.y"])


def test_gradients_reach_node_parameters(batch: Batch) -> None:
    scale = Scale()
    graph = ExecutionGraph([student(scale)], preflight_sample=batch)
    graph(batch)["student.y"].sum().backward()
    assert scale.w.grad is not None


def test_deployable_model_is_the_nodes_module() -> None:
    chain = nn.Sequential(AddOne(), Scale())
    node = ModelNode("student", chain, inputs="inputs.image", outputs="y")
    model = ExecutionGraph([node]).deployable_model("student")
    assert model is chain  # the same object, so the same weights
    x = torch.zeros(3)
    torch.testing.assert_close(model(x), node(x)["student.y"])


@pytest.mark.parametrize(
    ("nodes", "message"),
    [
        (
            [
                ModelNode("a", Scale(), inputs="inputs.image", outputs="y"),
                ModelNode("a", Scale(), inputs="inputs.image", outputs="z"),
            ],
            "unique",
        ),
        (  # "studnet" is a deliberate typo: no node writes it
            [ModelNode("a", Scale(), inputs="studnet.y", outputs="y")],
            "studnet.y",
        ),
        (
            [
                ModelNode("a", Scale(), inputs="b.y", outputs="y"),
                ModelNode("b", Scale(), inputs="a.y", outputs="y"),
            ],
            "cycle",
        ),
    ],
    ids=["duplicate_name", "unknown_name", "cycle"],
)
def test_invalid_graphs_fail_at_build_time(
    nodes: list[ModelNode], message: str
) -> None:
    with pytest.raises(GraphError, match=message):
        ExecutionGraph(nodes)


def test_check_reads_rejects_names_nothing_writes() -> None:
    student = ModelNode("student", nn.Identity(), "inputs.image", "logits")
    pipeline = ExecutionGraph([student])
    pipeline.check_reads(["student.logits", "targets.class"])
    # "studnet" is a deliberate typo: nothing writes it, so it must be rejected
    typo = LossBinding("ce", nn.CrossEntropyLoss(), ["studnet.logits", "targets.class"])
    with pytest.raises(GraphError, match="studnet.logits"):
        pipeline.check_reads(typo.inputs)
