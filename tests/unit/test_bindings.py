import pytest
import torch
from torch import nn

from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import MetricBinding

CE_READS = ["student.logits", "targets.class"]


@pytest.fixture
def batch():
    return {"inputs": {}, "targets": {"class": torch.tensor([0, 2])}, "meta": {}}


@pytest.fixture
def outputs():
    return {"student.logits": torch.randn(2, 3, requires_grad=True)}


def test_composer_is_weighted_sum_with_detached_terms(batch, outputs):
    composer = LossComposer(
        [
            LossBinding("ce", nn.CrossEntropyLoss(), CE_READS, weight=0.7),
            LossBinding("margin", nn.MultiMarginLoss(), CE_READS, weight=0.3),
        ]
    )
    total, terms = composer(batch, outputs)
    assert set(terms) == {"ce", "margin"}
    assert not any(t.requires_grad for t in terms.values())
    torch.testing.assert_close(
        total.detach(), 0.7 * terms["ce"] + 0.3 * terms["margin"]
    )
    total.backward()
    assert outputs["student.logits"].grad is not None


def test_plain_callable_loss_is_accepted(batch, outputs):
    composer = LossComposer(
        [LossBinding("sq", lambda x: (x**2).mean(), "student.logits")]
    )
    total, _ = composer(batch, outputs)
    assert total.ndim == 0


def test_loss_with_parameters_is_registered():
    learned = nn.Linear(3, 1)
    composer = LossComposer(
        [
            LossBinding("l", lambda x: x.sum(), "a.y"),
            LossBinding("p", learned, []),
        ]
    )
    assert {id(p) for p in composer.parameters()} == {
        id(p) for p in learned.parameters()
    }


@pytest.mark.parametrize(
    "value", [torch.tensor(float("nan")), torch.ones(2)], ids=["nan", "not_scalar"]
)
def test_non_finite_or_non_scalar_loss_fails(batch, outputs, value):
    composer = LossComposer([LossBinding("bad", lambda: value, [])])
    with pytest.raises(ValueError, match="bad"):
        composer(batch, outputs)


def test_binding_accepts_one_name():
    assert LossBinding("sq", lambda x: x, "student.logits").reads == ["student.logits"]


def test_composer_needs_a_binding():
    with pytest.raises(ValueError):
        LossComposer([])


def test_metric_binding_passes_values_in_order(batch, outputs):
    class Recorder:
        def __init__(self):
            self.calls = []

        def update(self, preds, target):
            self.calls.append((preds, target))

    recorder = Recorder()
    binding = MetricBinding("rec", recorder, ["student.logits", "targets.class"])
    binding.update(batch, outputs)
    binding.update(batch, outputs)
    assert len(recorder.calls) == 2
    preds, target = recorder.calls[0]
    assert preds is outputs["student.logits"]
    assert target is batch["targets"]["class"]
