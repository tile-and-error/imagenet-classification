from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeGuard

from matplotlib.axes import Axes
from matplotlib.figure import Figure  # no pyplot: no GUI backend or global state

from lwfm.results.curves import ConfusionMatrix, PRCurve, ROCCurve
from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink

type Columns = dict[str, list[float]]
type Table = Columns | ROCCurve | PRCurve | ConfusionMatrix


class PlotSink(OutputSink):
    """Draws the run's events as PNG files in `plot_dir`. Two kinds of event are
    drawn; the rest are ignored:

    - Scalars over training, when the sink closes: every `scalar` event named e.g.
      `train/loss/total` gives one point (step, value), and the points are joined
      into one line plot of that value against the training step, saved as
      `train_loss_total.png`. Scalars without a step are skipped.
    - Tables, as soon as they arrive: a `table` event whose value is a `ROCCurve`,
      `PRCurve` or `ConfusionMatrix` (from `lwfm.results.curves`) is drawn in that
      graph's standard form. Any other named columns of numbers, e.g.
      `{"epoch": [...], "top1": [...]}`, are drawn with the first column on the x
      axis and each other column as a line. `eval/val/roc` at step 120 is saved as
      `eval_val_roc_step120.png`, so each validation keeps its own plot; without a
      step, as `eval_val_roc.png`.

    For any other plot, draw the figure yourself and save it with
    `save_artifact(..., figure.savefig, ...)`.
    """

    def __init__(self, plot_dir: str | Path):
        self.plot_dir = Path(plot_dir)
        self.scalar_series: dict[str, list[tuple[int, float]]] = {}
        self._closed = False

    def write(self, event: RunEvent) -> None:
        if event.kind == "scalar" and event.step is not None:
            value = float(event.value)  # type: ignore[arg-type]  # numbers only
            self.scalar_series.setdefault(event.name, []).append((event.step, value))
        elif event.kind == "table" and (
            type(event.value) in DRAWERS or is_columns(event.value)
        ):
            self._save(event.name, event.value, event.step)  # type: ignore[arg-type]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, points in self.scalar_series.items():
            steps, values = zip(*points, strict=True)
            self._save(name, {"step": list(steps), name: list(values)})

    def _save(self, name: str, table: Table, step: int | None = None) -> None:
        figure = Figure(figsize=(5, 4), layout="constrained")
        axes = figure.subplots()
        DRAWERS.get(type(table), draw_columns)(axes, table)
        axes.set_title(name if step is None else f"{name} at step {step}")
        file_name = name.replace("/", "_") + ("" if step is None else f"_step{step}")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.plot_dir / f"{file_name}.png", dpi=120)


def draw_columns(axes: Axes, columns: Columns) -> None:
    (x_name, x), *y_columns = columns.items()
    for y_name, y in y_columns:
        axes.plot(x, y, marker="." if len(x) <= 50 else None, label=y_name)
    axes.set_xlabel(x_name)
    if len(y_columns) == 1:
        axes.set_ylabel(y_columns[0][0])
    else:
        axes.legend()
    axes.grid(alpha=0.3)


def draw_roc(axes: Axes, roc: ROCCurve) -> None:
    axes.plot(roc.fpr, roc.tpr, label=f"AUC = {roc.auc:.3f}")
    axes.plot([0, 1], [0, 1], "k:", label="chance")
    axes.set(xlabel="false positive rate", ylabel="true positive rate")
    unit_square(axes)


def draw_pr(axes: Axes, pr: PRCurve) -> None:
    axes.plot(pr.recall, pr.precision, label=f"AP = {pr.average_precision:.3f}")
    axes.set(xlabel="recall", ylabel="precision")
    unit_square(axes)


def unit_square(axes: Axes) -> None:
    axes.set(xlim=(0, 1), ylim=(0, 1.02), aspect="equal")
    axes.legend(loc="lower right")
    axes.grid(alpha=0.3)


def draw_confusion(axes: Axes, confusion: ConfusionMatrix) -> None:
    image = axes.imshow(confusion.matrix, cmap="Blues")
    axes.figure.colorbar(image, ax=axes)
    ticks = range(len(confusion.class_names))
    axes.set(xlabel="predicted", ylabel="true", xticks=ticks, yticks=ticks)
    axes.set_xticklabels(confusion.class_names, rotation=45, ha="right")
    axes.set_yticklabels(confusion.class_names)
    if len(confusion.class_names) <= 20:  # counts stay readable
        peak = max(max(row) for row in confusion.matrix)
        for true, row in enumerate(confusion.matrix):
            for predicted, count in enumerate(row):
                colour = "white" if count > peak / 2 else "black"
                axes.text(
                    predicted, true, str(count), ha="center", va="center", c=colour
                )


DRAWERS: dict[type, Callable[[Axes, Any], None]] = {
    ROCCurve: draw_roc,
    PRCurve: draw_pr,
    ConfusionMatrix: draw_confusion,
}


def is_columns(value: object) -> TypeGuard[Columns]:
    """True for a dict of at least two equal-length lists of numbers."""
    if not isinstance(value, dict) or len(value) < 2:
        return False
    columns = list(value.values())
    return all(
        isinstance(column, list)
        and len(column) == len(columns[0]) > 0
        and all(isinstance(item, int | float) for item in column)
        for column in columns
    )
