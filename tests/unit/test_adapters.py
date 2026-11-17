import torch

from lwfm.core.types import InputSpec
from lwfm.models.adapters import InputAdapter


def test_normalises_without_resizing_when_size_matches():
    spec = InputSpec(image_size=(4, 4), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    x = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(InputAdapter(spec)(x), (x - 0.5) / 0.5)


def test_resizes_to_spec():
    out = InputAdapter(InputSpec(image_size=(8, 6)))(torch.rand(2, 3, 16, 12))
    assert out.shape == (2, 3, 8, 6)


def test_statistics_are_buffers_not_saved_weights():
    adapter = InputAdapter(InputSpec())
    assert list(adapter.parameters()) == []
    assert adapter.state_dict() == {}
    assert {name for name, _ in adapter.named_buffers()} == {"mean", "std"}
