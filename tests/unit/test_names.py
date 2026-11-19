import pytest
import torch

from lwfm.core.names import UnknownNameError, as_list, resolve, resolve_all


@pytest.fixture
def batch():
    return {
        "inputs": {"image": torch.rand(2, 3, 4, 4)},
        "targets": {"class": torch.tensor([0, 1])},
        "meta": {"sample_id": ["a", "b"]},
    }


def test_resolves_batch_values(batch):
    assert resolve("inputs.image", batch, {}) is batch["inputs"]["image"]
    assert resolve("targets.class", batch, {}) is batch["targets"]["class"]
    assert resolve("meta.sample_id", batch, {}) == ["a", "b"]


def test_resolves_node_outputs(batch):
    logits = torch.zeros(2, 5)
    assert resolve("student.logits", batch, {"student.logits": logits}) is logits


# Deliberate typos ("imgae", "studnet"): each must appear in the error message.
@pytest.mark.parametrize(
    "name", ["inputs.imgae", "studnet.logits", "inputs", "batch.inputs.image"]
)
def test_unknown_name_is_in_the_error(batch, name):
    with pytest.raises(UnknownNameError, match=name.replace(".", r"\.")):
        resolve(name, batch, {"student.logits": torch.zeros(2, 5)})


def test_resolve_all_keeps_order(batch):
    logits = torch.zeros(2, 5)
    values = resolve_all(
        ["student.logits", "targets.class"], batch, {"student.logits": logits}
    )
    assert values[0] is logits
    assert values[1] is batch["targets"]["class"]


def test_as_list_accepts_one_name_or_many():
    assert as_list("inputs.image") == ["inputs.image"]
    assert as_list(("a.y", "b.y")) == ["a.y", "b.y"]
