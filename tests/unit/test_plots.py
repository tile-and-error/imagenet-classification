import pytest
import torch
from torch import nn

from lwfm.core.graph import ExecutionGraph, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.evaluation.metrics import CurveBinding
from lwfm.results.curves import ConfusionMatrix, PRCurve, ROCCurve
from lwfm.results.events import RunEvent
from lwfm.results.plots import PlotSink, is_columns
from lwfm.results.router import OutputRouter
from lwfm.training import Trainer

from conftest import CE_READS, accuracy, batches, is_png, setup


def png_names(plot_dir):
    paths = sorted(plot_dir.glob("*.png"))
    assert all(is_png(path) for path in paths)
    return [path.name for path in paths]


def test_scalars_and_curves_become_one_png_each(tmp_path):
    sink = PlotSink(tmp_path / "plots")
    for step, loss in enumerate([1.0, 0.5, 0.25], start=1):
        sink.write(
            RunEvent("scalar", "train/loss/total", torch.tensor(loss), step=step)
        )
    roc = {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.9, 1.0]}
    sink.write(RunEvent("table", "eval/val/roc", roc))
    # A table is drawn when it arrives; the scalars only once they are all in
    assert png_names(tmp_path / "plots") == ["eval_val_roc.png"]
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


def test_curve_types_are_drawn(tmp_path):
    sink = PlotSink(tmp_path)
    sink.write(
        RunEvent("table", "roc", ROCCurve([0.0, 0.2, 1.0], [0.0, 0.8, 1.0], 0.8))
    )
    sink.write(RunEvent("table", "pr", PRCurve([0.5, 1.0], [1.0, 0.0], 0.5)))
    confusion = ConfusionMatrix([[5, 1], [2, 4]], ["cat", "dog"])
    sink.write(RunEvent("table", "confusion", confusion))
    sink.close()
    assert png_names(tmp_path) == ["confusion.png", "pr.png", "roc.png"]


def test_each_validation_step_keeps_its_own_table_plot(tmp_path):
    sink = PlotSink(tmp_path)
    for step in [100, 200]:
        roc = ROCCurve([0.0, 1.0], [0.0, 1.0], 0.5)
        sink.write(RunEvent("table", "eval/val/roc", roc, step=step))
    assert png_names(tmp_path) == [
        "eval_val_roc_step100.png",
        "eval_val_roc_step200.png",
    ]


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


def test_a_training_run_plots_its_curves_at_each_validation(tmp_path):
    _, graph, loss, optimizer = setup()
    confusion = CurveBinding("confusion", ConfusionMatrix.from_scores, CE_READS)
    with OutputRouter(PlotSink(tmp_path)) as sink:
        Trainer(device="cpu", max_steps=4, val_every_n_steps=2).fit(
            graph,
            batches(4),
            loss,
            optimizer,
            val_loader=batches(1),
            metrics=accuracy(),
            curves=[confusion],
            output_sink=sink,
        )
    assert {"eval_val_confusion_step2.png", "eval_val_confusion_step4.png"} <= set(
        png_names(tmp_path)
    )
