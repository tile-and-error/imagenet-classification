import pytest
import torch

from lwfm.core.ports import PortError, resolve, resolve_all


@pytest.fixture
def batch():
    return {
        "inputs": {"image": torch.rand(2, 3, 4, 4)},
        "targets": {"class": torch.tensor([0, 1])},
        "meta": {"sample_id": ["a", "b"]},
    }


def test_resolves_batch_ports(batch):
    assert resolve("batch.inputs.image", batch, {}) is batch["inputs"]["image"]
    assert resolve("batch.meta.sample_id", batch, {}) == ["a", "b"]


def test_resolves_node_outputs(batch):
    logits = torch.zeros(2, 5)
    assert resolve("student.logits", batch, {"student.logits": logits}) is logits


@pytest.mark.parametrize("port", ["batch.inputs.imgae", "studnet.logits", "batch"])
def test_unknown_port_names_the_port(batch, port):
    with pytest.raises(PortError, match=port.replace(".", r"\.")):
        resolve(port, batch, {"student.logits": torch.zeros(2, 5)})


def test_resolve_all_maps_argument_names(batch):
    resolved = resolve_all(
        {"input": "student.logits", "target": "batch.targets.class"},
        batch,
        {"student.logits": torch.zeros(2, 5)},
    )
    assert set(resolved) == {"input", "target"}
    assert resolved["target"] is batch["targets"]["class"]
