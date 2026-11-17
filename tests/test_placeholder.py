import torch


def test_placeholder():
    assert torch.ones(2).sum() == 2
