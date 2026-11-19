import pytest
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import v2

from lwfm.core.graph import ExecutionGraph, GraphError, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import MetricBinding

READS = ["student.logits", "targets.class"]


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append((self.training, torch.is_grad_enabled()))
        return x


def classifier(*tail):
    return nn.Sequential(
        v2.Resize((16, 16), antialias=True),
        nn.Flatten(),
        nn.Linear(3 * 16 * 16, 10),
        *tail,
    )


def student(*tail):
    return ModelNode("student", classifier(*tail), "inputs.image", "logits")


def probe():  # wrong input size: student.logits has 10 features, not 512
    return ModelNode("probe", nn.Linear(512, 2), "student.logits", "logits")


def batch(size=4, height=16, width=16):
    return {
        "inputs": {"image": torch.rand(size, 3, height, width)},
        "targets": {"class": torch.randint(0, 10, (size,))},
        "meta": {"sample_id": [f"s{i}" for i in range(size)]},
    }


def test_sample_checks_shapes_at_build_time():
    graph = ExecutionGraph([student()], preflight_sample=batch())
    assert graph.preflight_output_shapes == {"student.logits": (4, 10)}


def test_sample_of_another_size_is_resized_inside():
    graph = ExecutionGraph([student()], preflight_sample=batch(5, 40, 30))
    assert graph.preflight_output_shapes == {"student.logits": (5, 10)}


def test_no_sample_skips_the_check():
    def boom(x):
        raise AssertionError("must not run")

    graph = ExecutionGraph([student(Lambda(boom))])
    assert graph.preflight_output_shapes is None


def test_shape_mismatch_fails_at_build_time_naming_the_node():
    with pytest.raises(RuntimeError) as info:
        ExecutionGraph([student(), probe()], preflight_sample=batch())
    assert info.value.__notes__ == ["in node 'probe', reading student.logits (4, 10)"]


def test_note_is_added_in_ordinary_runs_too():
    graph = ExecutionGraph([student(), probe()])
    with pytest.raises(RuntimeError) as info:
        graph(batch())
    assert info.value.__notes__ == ["in node 'probe', reading student.logits (4, 10)"]


def test_preflight_can_be_rerun_with_a_real_batch():
    graph = ExecutionGraph([student()], preflight_sample=batch())
    assert graph.preflight(batch(3)) == {"student.logits": (3, 10)}


def test_dropped_batch_dimension_fails():
    with pytest.raises(GraphError, match=r"'student.logits' has shape \(10,\).*, 4"):
        ExecutionGraph(
            [student(Lambda(lambda x: x.sum(dim=0)))], preflight_sample=batch()
        )


def test_runs_in_eval_mode_without_gradients_and_leaves_train_mode():
    spy = Probe()
    graph = ExecutionGraph([student(spy)], preflight_sample=batch())
    assert spy.seen == [(False, False)]
    assert graph.training


def test_mode_is_restored_after_a_failed_rerun():
    graph = ExecutionGraph([student(), probe()])
    with pytest.raises(RuntimeError):
        graph.preflight(batch())
    assert graph.training


def test_sample_without_input_tensors_fails():
    with pytest.raises(ValueError, match="no input tensors"):
        ExecutionGraph([student()], preflight_sample={"inputs": {}, "targets": {}})


def test_inputs_and_targets_with_different_batch_sizes_fail():
    sample = batch(size=4)
    sample["targets"]["class"] = torch.zeros(3, dtype=torch.long)
    with pytest.raises(ValueError, match="inputs.image 4, targets.class 3"):
        ExecutionGraph([student()], preflight_sample=sample)


def test_loss_failure_names_the_loss():
    graph = ExecutionGraph([student()])
    bad = LossComposer([LossBinding("bad", lambda x, y: x @ y.float(), READS)])
    sample = batch()
    with pytest.raises(RuntimeError) as info:
        bad(sample, graph(sample))
    assert info.value.__notes__ == [
        "in loss 'bad', reading student.logits (4, 10), targets.class (4,)"
    ]


def test_metric_failure_names_the_metric():
    graph = ExecutionGraph([student()])
    swapped = MetricBinding(
        "swapped", MulticlassAccuracy(num_classes=10), list(reversed(READS))
    )
    sample = batch()
    with pytest.raises(Exception) as info:  # noqa: B017 (torchmetrics' own error)
        swapped.update(sample, graph(sample))
    assert info.value.__notes__ == [
        "in metric 'swapped', reading targets.class (4,), student.logits (4, 10)"
    ]
