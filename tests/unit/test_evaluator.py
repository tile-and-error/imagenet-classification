from collections.abc import Iterator
from dataclasses import dataclass
from typing import NoReturn

import pytest
import torch
from matplotlib.axes import Axes
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph, GraphError, ModelNode
from lwfm.core.types import Batch
from lwfm.evaluation.evaluator import Evaluator
from lwfm.evaluation.metrics import CurveBinding, MetricBinding
from lwfm.results.curves import ConfusionMatrix

from conftest import CE_READS, ListSink

CPU = torch.device("cpu")

# Hand-worked, K = 6. Rank of the true class: 1, 1, 6, 2.
# top-1 = 2/4 = 0.5; top-5 = 3/4 = 0.75 (only the third sample misses).
LOGITS = torch.tensor(
    [
        [5.0, 1, 0, 0, 0, 0],
        [0.0, 5, 1, 0, 0, 0],
        [5.0, 4, 3, 2, 1, 0],
        [5.0, 0, 4, 0, 0, 0],
    ]
)

CLASSES = torch.tensor([0, 1, 5, 2])


def batch(rows: list[int]) -> Batch:
    return {
        "inputs": {"logits": LOGITS[rows]},
        "targets": {"class": CLASSES[rows]},
        "meta": {"sample_id": [f"s{i}" for i in rows]},
    }


# Unequal batches: the mean of per-batch top-1 is (2/3 + 0)/2 = 1/3, not 0.5.
LOADER = [batch([0, 1, 2]), batch([3])]


def logits_graph(module: nn.Module | None = None) -> ExecutionGraph:
    node = ModelNode("student", module or nn.Identity(), "inputs.logits", "logits")
    return ExecutionGraph([node], preflight_sample=LOADER[0])


def top_k(k: int) -> MetricBinding:
    return MetricBinding(
        f"top{k}", MulticlassAccuracy(num_classes=6, top_k=k, average="micro"), CE_READS
    )


class Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(6, 6)
        self.seen_modes = []

    def forward(self, x: Tensor) -> Tensor:
        self.seen_modes.append((self.training, torch.is_inference_mode_enabled()))
        return self.linear(x)


def test_micro_top1_top5_match_hand_worked_example() -> None:
    result = Evaluator().evaluate(
        logits_graph(), LOADER, [top_k(1), top_k(5)], device=CPU
    )
    assert result.metric_values == pytest.approx({"top1": 0.5, "top5": 0.75})


def test_metric_state_resets_between_runs() -> None:
    metrics = [top_k(1)]
    first = Evaluator().evaluate(logits_graph(), LOADER, metrics, device=CPU)
    second = Evaluator().evaluate(logits_graph(), LOADER[1:], metrics, device=CPU)
    assert first.metric_values["top1"] == pytest.approx(0.5)
    assert second.metric_values["top1"] == 0.0


def test_runs_in_eval_and_inference_mode_and_changes_no_weights() -> None:
    probe = Probe()
    graph = logits_graph(probe)
    probe.seen_modes.clear()  # drop the build-time shape check
    before = {k: v.clone() for k, v in graph.state_dict().items()}
    first = Evaluator().evaluate(graph, LOADER, [top_k(1)], device=CPU)
    second = Evaluator().evaluate(graph, LOADER, [top_k(1)], device=CPU)
    assert set(probe.seen_modes) == {(False, True)}
    assert graph.training  # mode restored
    assert first == second
    for k, v in graph.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_eval_mode_is_kept_if_graph_was_in_eval() -> None:
    graph = logits_graph().eval()
    Evaluator().evaluate(graph, LOADER, [top_k(1)], device=CPU)
    assert not graph.training


@pytest.mark.parametrize("was_training", [True, False], ids=["train", "eval"])
def test_mode_is_restored_when_a_metric_fails(was_training: bool) -> None:
    # .train(False) is equivalent to .eval()
    graph = logits_graph().train(was_training)
    bad = MetricBinding("bad", MulticlassAccuracy(num_classes=6), ["targets.missing"])
    with pytest.raises(KeyError, match="targets.missing"):
        Evaluator().evaluate(graph, LOADER, [bad], device=CPU)
    assert graph.training == was_training


def test_values_are_emitted_as_eval_split_name(sink: ListSink) -> None:
    Evaluator().evaluate(
        logits_graph(),
        LOADER,
        [top_k(1), top_k(5)],
        device=CPU,
        split_name="test",
        output_sink=sink,
        training_step=7,
    )
    assert [(e.kind, e.name, e.value, e.step) for e in sink.events] == [
        ("scalar", "eval/test/top1", pytest.approx(0.5), 7),
        ("scalar", "eval/test/top5", pytest.approx(0.75), 7),
    ]


@pytest.mark.parametrize(
    ("loader", "metrics", "error", "match"),
    [
        ([], [top_k(1)], ValueError, "no batches"),
        (LOADER, [], ValueError, "at least one"),
        (LOADER, [top_k(1), top_k(1)], ValueError, "unique"),
        (
            LOADER,
            [
                MetricBinding(
                    "per_class", MulticlassAccuracy(6, average="none"), CE_READS
                )
            ],
            TypeError,
            "'per_class' computed shape \\(6,\\)",
        ),
    ],
    ids=["empty_loader", "no_metrics", "duplicate_names", "non_scalar_metric"],
)
def test_bad_inputs_fail_clearly(
    loader: list[Batch],
    metrics: list[MetricBinding],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        Evaluator().evaluate(logits_graph(), loader, metrics, device=CPU)


def confusion() -> CurveBinding:
    return CurveBinding("confusion", ConfusionMatrix.from_scores, CE_READS)


def test_a_curve_is_built_from_the_whole_split_and_emitted(sink: ListSink) -> None:
    result = Evaluator().evaluate(
        logits_graph(),
        LOADER,  # two unequal batches
        [top_k(1)],
        curve_bindings=[confusion()],
        device=CPU,
        output_sink=sink,
        training_step=7,
    )
    whole_split = ConfusionMatrix.from_scores(LOGITS, CLASSES)
    assert result.curves == {"confusion": whole_split}
    assert [(e.kind, e.name, e.value, e.step) for e in sink.events] == [
        ("scalar", "eval/val/top1", pytest.approx(0.5), 7),
        ("table", "eval/val/confusion", whole_split, 7),
    ]


def test_curve_values_reset_between_runs() -> None:
    curves = [confusion()]
    Evaluator().evaluate(
        logits_graph(), LOADER, [top_k(1)], curve_bindings=curves, device=CPU
    )
    second = Evaluator().evaluate(
        logits_graph(), LOADER[1:], [top_k(1)], curve_bindings=curves, device=CPU
    )
    confusion_matrix = second.curves["confusion"]
    assert isinstance(confusion_matrix, ConfusionMatrix)
    assert sum(map(sum, confusion_matrix.matrix)) == 1  # one sample


def test_curves_alone_need_no_metric() -> None:
    result = Evaluator().evaluate(
        logits_graph(), LOADER, [], curve_bindings=[confusion()], device=CPU
    )
    assert result.metric_values == {} and list(result.curves) == ["confusion"]


def test_a_curve_and_a_metric_cannot_share_a_name() -> None:
    clash = CurveBinding("top1", ConfusionMatrix.from_scores, CE_READS)
    with pytest.raises(ValueError, match="unique"):
        Evaluator().evaluate(
            logits_graph(), LOADER, [top_k(1)], curve_bindings=[clash], device=CPU
        )


@dataclass(frozen=True)
class Shapes:
    shapes: list[list[int]]

    def draw(self, axes: Axes) -> None:
        pass


def test_a_curve_reads_any_number_of_values_in_order() -> None:
    shapes = CurveBinding(
        "shapes",
        lambda *values: Shapes([list(value.shape) for value in values]),
        ["student.logits", "targets.class", "inputs.logits"],
    )
    result = Evaluator().evaluate(
        logits_graph(), LOADER, [top_k(1)], curve_bindings=[shapes], device=CPU
    )
    built = result.curves["shapes"]
    assert isinstance(built, Shapes)
    assert built.shapes == [[4, 6], [4], [4, 6]]  # batches joined


def test_a_failing_curve_names_itself_and_what_it_read() -> None:
    def broken(scores: Tensor, labels: Tensor) -> NoReturn:
        raise RuntimeError("bad curve")

    curve = CurveBinding("broken", broken, CE_READS)
    with pytest.raises(RuntimeError, match="bad curve") as caught:
        Evaluator().evaluate(
            logits_graph(), LOADER, [top_k(1)], curve_bindings=[curve], device=CPU
        )
    # Shapes of the whole split, both batches joined
    note = "in curve 'broken', reading student.logits (4, 6), targets.class (4,)"
    assert caught.value.__notes__ == [note]


def test_a_curve_must_build_something_drawable() -> None:
    def plain_dict(scores: Tensor, labels: Tensor) -> dict[str, int]:
        return {"n": len(scores)}

    plain = CurveBinding("plain", plain_dict, CE_READS)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="'plain' built a dict, which has no draw"):
        Evaluator().evaluate(
            logits_graph(), LOADER, [top_k(1)], curve_bindings=[plain], device=CPU
        )


def test_a_curve_must_read_what_its_build_function_takes() -> None:
    with pytest.raises(ValueError, match="reads 1 values"):
        CurveBinding("roc", ConfusionMatrix.from_scores, ["student.logits"])


def test_to_device_moves_tensors_and_keeps_other_values() -> None:
    moved = to_device(batch([0, 1]), torch.device("meta"))
    assert moved["inputs"]["logits"].device.type == "meta"
    assert moved["targets"]["class"].device.type == "meta"
    assert moved["meta"]["sample_id"] == ["s0", "s1"]


def test_unknown_metric_input_fails_before_any_batch() -> None:
    def loader() -> Iterator[Batch]:
        raise AssertionError("no batch should be read")
        yield

    # "studnet" is a deliberate typo: it must be rejected before any batch is read
    typo = MetricBinding(
        "top1", MulticlassAccuracy(num_classes=6), ["studnet.logits", "targets.class"]
    )
    with pytest.raises(GraphError, match="studnet.logits"):
        Evaluator().evaluate(
            logits_graph(), loader(), [typo], device=torch.device("cpu")
        )
    curve_typo = CurveBinding("roc", ConfusionMatrix.from_scores, typo.inputs)
    with pytest.raises(GraphError, match="studnet.logits"):
        Evaluator().evaluate(
            logits_graph(),
            loader(),
            [top_k(1)],
            curve_bindings=[curve_typo],
            device=CPU,
        )
