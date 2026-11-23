import pytest
import torch
from torch import nn

from lwfm.core.graph import ExecutionGraph, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.results.events import RunEvent
from lwfm.results.plots import PlotSink, is_columns
from lwfm.results.router import OutputRouter
from lwfm.training import Trainer

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_names(plot_dir):
    paths = sorted(plot_dir.glob("*.png"))
    assert all(path.read_bytes().startswith(PNG_SIGNATURE) for path in paths)
    return [path.name for path in paths]


def test_scalars_and_curves_become_one_png_each(tmp_path):
    sink = PlotSink(tmp_path / "plots")
    for step, loss in enumerate([1.0, 0.5, 0.25], start=1):
        sink.write(
            RunEvent("scalar", "train/loss/total", torch.tensor(loss), step=step)
        )
    roc = {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.9, 1.0]}
    sink.write(RunEvent("table", "eval/val/roc", roc))
    assert not (tmp_path / "plots").exists()  # nothing is drawn before close
    sink.close()
    assert png_names(tmp_path / "plots") == ["eval_val_roc.png", "train_loss_total.png"]
    assert sink.scalar_series["train/loss/total"] == [(1, 1.0), (2, 0.5), (3, 0.25)]


def test_events_that_cannot_be_plotted_are_ignored(tmp_path):
    sink = PlotSink(tmp_path)
    sink.write(RunEvent("scalar", "eval/test/top1", 0.9))  # no step
    sink.write(
        RunEvent("table", "per_class", [[0, 0.9], [1, 0.8]])
    )  # rows, not columns
    sink.write(RunEvent("message", "note", "hello"))
    sink.close()
    assert png_names(tmp_path) == []


def test_a_later_table_replaces_an_earlier_one(tmp_path):
    sink = PlotSink(tmp_path)
    sink.write(RunEvent("table", "roc", {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}))
    sink.write(RunEvent("table", "roc", {"fpr": [0.0, 1.0], "tpr": [1.0, 1.0]}))
    assert sink.curves["roc"]["tpr"] == [1.0, 1.0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"x": [0.0, 1.0], "y": [1, 2]}, True),
        ({"x": [0.0, 1.0], "y": [1.0, 2.0], "z": [3.0, 4.0]}, True),  # two y lines
        ({"x": [0.0, 1.0]}, False),  # nothing to plot against
        ({"x": [0.0, 1.0], "y": [1.0]}, False),  # unequal lengths
        ({"x": [], "y": []}, False),
        ({"x": [0.0], "y": ["a"]}, False),
        ([[0.0, 1.0]], False),
    ],
)
def test_is_columns(value, expected):
    assert is_columns(value) is expected


def test_close_is_safe_to_call_twice(tmp_path):
    sink = PlotSink(tmp_path)
    sink.close()
    sink.close()


def test_a_training_run_plots_its_losses(tmp_path):
    torch.manual_seed(0)
    node = ModelNode("student", nn.Linear(2, 2), inputs="inputs.x", outputs="logits")
    batch = {
        "inputs": {"x": torch.rand(4, 2)},
        "targets": {"class": torch.tensor([0, 1, 0, 1])},
    }
    loss = LossComposer(
        [LossBinding("ce", nn.CrossEntropyLoss(), ["student.logits", "targets.class"])]
    )
    with OutputRouter(PlotSink(tmp_path)) as sink:
        Trainer(device="cpu", max_steps=3, log_every_n_steps=1).fit(
            ExecutionGraph([node]),
            [batch],
            loss,
            torch.optim.SGD(node.parameters(), lr=0.1),
            output_sink=sink,
        )
    assert "train_loss_total.png" in png_names(tmp_path)
