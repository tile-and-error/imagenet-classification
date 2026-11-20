import pytest
import torch
from torch import nn

from lwfm.core.graph import ExecutionGraph, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.training import Callback, StepOutput, Trainer
from lwfm.training.steps import forward_and_loss

K = 3
CE_READS = ["student.logits", "targets.class"]


def batches(count, batch_size=4, seed=0):
    """`count` seeded batches of random 3x2x2 images with labels in [0, K)."""
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "inputs": {"image": torch.rand(batch_size, 3, 2, 2, generator=generator)},
            "targets": {
                "class": torch.randint(0, K, (batch_size,), generator=generator)
            },
        }
        for _ in range(count)
    ]


class RecordsDtype(nn.Module):
    """A linear model that remembers the dtype its output was computed in."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, K)
        self.dtypes = []

    def forward(self, image):
        out = self.linear(image.flatten(1))
        self.dtypes.append(out.dtype)
        return out


def setup(seed=0, module=None):
    """A seeded student (a linear classifier, or `module`), its graph and a CE loss."""
    torch.manual_seed(seed)
    module = module or nn.Sequential(nn.Flatten(), nn.Linear(12, K))
    node = ModelNode("student", module, inputs="inputs.image", outputs="logits")
    loss = LossComposer([LossBinding("ce", nn.CrossEntropyLoss(), CE_READS)])
    return node, ExecutionGraph([node]), loss


class CountSteps(Callback):
    def __init__(self):
        self.optimizer_steps = 0

    def on_after_optimizer_step(self, state):
        self.optimizer_steps += 1


# --- gradient accumulation ------------------------------------------------------


def test_accumulation_steps_every_k_batches_and_at_epoch_end():
    # 5 batches with k = 2: groups of 2, 2 and a final partial group of 1
    node, graph, loss = setup()
    counter = CountSteps()
    result = Trainer(device="cpu", max_epochs=2, accumulate_grad_batches=2).fit(
        graph,
        batches(5),
        loss,
        torch.optim.SGD(node.parameters(), lr=0.1),
        callbacks=[counter],
    )
    assert result.global_step == counter.optimizer_steps == 6


def test_max_steps_counts_optimizer_steps_not_batches():
    # k = 2, so 3 optimizer steps take 6 of the 10 batches
    class CountBatches(CountSteps):
        def __init__(self):
            super().__init__()
            self.batches = 0

        def on_step_start(self, state, batch):
            self.batches += 1

    node, graph, loss = setup()
    counter = CountBatches()
    result = Trainer(device="cpu", max_steps=3, accumulate_grad_batches=2).fit(
        graph,
        batches(10),
        loss,
        torch.optim.SGD(node.parameters(), lr=0.1),
        callbacks=[counter],
    )
    assert result.global_step == counter.optimizer_steps == 3
    assert counter.batches == 6


def test_accumulating_two_halves_equals_one_full_batch():
    # Mean loss over 8 samples = average of the mean losses of its two halves of 4
    data = batches(2, batch_size=4)
    full = {
        "inputs": {"image": torch.cat([b["inputs"]["image"] for b in data])},
        "targets": {"class": torch.cat([b["targets"]["class"] for b in data])},
    }
    node_a, graph_a, loss_a = setup()
    Trainer(device="cpu", max_epochs=1, accumulate_grad_batches=2).fit(
        graph_a, data, loss_a, torch.optim.SGD(node_a.parameters(), lr=0.5)
    )
    node_b, graph_b, loss_b = setup()
    Trainer(device="cpu", max_epochs=1).fit(
        graph_b, [full], loss_b, torch.optim.SGD(node_b.parameters(), lr=0.5)
    )
    for a, b in zip(node_a.parameters(), node_b.parameters(), strict=True):
        torch.testing.assert_close(a, b)


# --- clipping and precision -----------------------------------------------------


def test_gradient_clipping_bounds_the_update():
    # Plain SGD with lr = 1 moves the weights by exactly the clipped gradient
    node, graph, loss = setup()
    before = torch.cat([p.detach().flatten().clone() for p in node.parameters()])
    Trainer(device="cpu", max_steps=1, gradient_clip_norm=1e-3).fit(
        graph, batches(1), loss, torch.optim.SGD(node.parameters(), lr=1.0)
    )
    after = torch.cat([p.detach().flatten() for p in node.parameters()])
    assert (after - before).norm() <= 1e-3 * (1 + 1e-4)


def test_bf16_runs_the_forward_pass_under_autocast():
    module = RecordsDtype()
    node, graph, loss = setup(module=module)
    Trainer(device="cpu", max_steps=2, precision="bf16").fit(
        graph, batches(2), loss, torch.optim.SGD(node.parameters(), lr=0.1)
    )
    assert module.dtypes == [torch.bfloat16, torch.bfloat16]
    assert all(p.dtype == torch.float32 for p in node.parameters())  # weights stay fp32


def test_fp16_uses_a_grad_scaler_and_trains():
    scalers = []

    class RecordScaler(Callback):
        def on_fit_start(self, state):
            scalers.append(state.scaler)

    node, graph, loss = setup()
    before = [p.detach().clone() for p in node.parameters()]
    Trainer(device="cpu", max_steps=3, precision="fp16").fit(
        graph,
        batches(3),
        loss,
        torch.optim.SGD(node.parameters(), lr=0.1),
        callbacks=[RecordScaler()],
    )
    assert scalers[0] is not None
    assert any(
        not torch.equal(a, b) for a, b in zip(before, node.parameters(), strict=True)
    )


# --- custom steps ---------------------------------------------------------------


def manual_step(state, batch):
    """A custom step that calls .backward() itself instead of the helpers."""
    total, terms = forward_and_loss(state, batch)
    total.backward()
    state.optimizer.step()
    state.optimizer.zero_grad()
    return StepOutput(total.detach(), terms)


def test_custom_step_works_in_plain_fp32():
    node, graph, loss = setup()
    result = Trainer(device="cpu", max_steps=3, step_fn=manual_step).fit(
        graph, batches(3), loss, torch.optim.SGD(node.parameters(), lr=0.1)
    )
    assert result.global_step == 3


@pytest.mark.parametrize(
    "settings", [{"precision": "fp16"}, {"accumulate_grad_batches": 2}]
)
def test_custom_step_without_helpers_is_refused_when_they_matter(settings):
    node, graph, loss = setup()
    trainer = Trainer(device="cpu", max_steps=3, step_fn=manual_step, **settings)
    with pytest.raises(RuntimeError, match="steps.backward"):
        trainer.fit(graph, batches(4), loss, torch.optim.SGD(node.parameters(), lr=0.1))
