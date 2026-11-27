import pytest
import torch
from torch import nn

from lwfm.loss.losses import LossBinding, LossComposer

from conftest import CE_READS


@pytest.fixture
def batch():
    return {"inputs": {}, "targets": {"class": torch.tensor([0, 2])}, "meta": {}}


@pytest.fixture
def outputs():
    logits = torch.tensor([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]], requires_grad=True)
    return {"student.logits": logits}


@pytest.fixture
def composer():
    return LossComposer(
        [
            LossBinding("ce", nn.CrossEntropyLoss(), CE_READS, weight=0.7),
            LossBinding("margin", nn.MultiMarginLoss(), CE_READS, weight=0.3),
        ]
    )


def test_composer_total_is_weighted_sum_of_terms(composer, batch, outputs):
    # Both rows of the logits give the same loss, so the means are closed-form:
    # CE = log(1 + e^-0.5 + e^-1); margin = (max(0, 1 - 1 + 0.5) + 0) / 3 = 1/6.
    ce, margin = 0.6802696585655212, 1 / 6
    total, terms = composer(batch, outputs)
    assert set(terms) == {"ce", "margin"}
    torch.testing.assert_close(terms["ce"], torch.tensor(ce))
    torch.testing.assert_close(terms["margin"], torch.tensor(margin))
    torch.testing.assert_close(total.detach(), torch.tensor(0.7 * ce + 0.3 * margin))
    total.backward()
    assert outputs["student.logits"].grad is not None


def test_composer_terms_are_detached(composer, batch, outputs):
    total, terms = composer(batch, outputs)
    assert total.requires_grad
    assert not any(term.requires_grad for term in terms.values())


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


def test_binding_converts_one_name_to_list():
    assert LossBinding("sq", lambda x: x, "student.logits").inputs == ["student.logits"]


def test_composer_needs_a_binding():
    with pytest.raises(ValueError):
        LossComposer([])


def test_non_unique_loss_composer_fails():
    repeated_binding = LossBinding("repeated", lambda: torch.tensor(5), [])
    with pytest.raises(ValueError):
        LossComposer([repeated_binding, repeated_binding])
