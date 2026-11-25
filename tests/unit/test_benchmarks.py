import pytest
import torch
from torch import nn

from lwfm.evaluation.benchmarks import inference_cost


def test_inference_cost_of_a_linear_layer():
    model = nn.Linear(8, 4)  # 8 * 4 weights + 4 biases
    cost = inference_cost(model, torch.rand(1, 8), warmup=1, repeats=5)
    assert cost["params_m"] == pytest.approx(36 / 1e6)
    assert cost["gflops"] == pytest.approx(2 * 8 * 4 / 1e9)  # 2 FLOPs per multiply-add
    assert cost["size_mb"] > 0
    assert 0 < cost["latency_ms_median"] <= cost["latency_ms_p95"]


def test_inference_cost_restores_the_train_mode():
    model = nn.Sequential(nn.Linear(8, 4), nn.Dropout(0.5))
    inference_cost(model, torch.rand(1, 8), warmup=0, repeats=1)
    assert model.training
