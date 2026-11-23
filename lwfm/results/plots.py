from pathlib import Path
from typing import TypeGuard

from matplotlib.figure import Figure  # no pyplot: no GUI backend or global state

from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink

type Columns = dict[str, list[float]]


class PlotSink(OutputSink):
    """Saves one PNG per plottable event to `plot_dir` when closed.

    - A scalar with a step becomes a line against the step, e.g. `train/loss/total`.
    - A `table` whose value is a dict of equal-length numeric columns becomes a
      curve: the first column on x, each other column on y. A ROC curve is
      `{"fpr": [...], "tpr": [...]}`. A later table of the same name replaces it.

    Other events are ignored. Files are named after the event, `/` as `_`:
    `eval/val/roc` is saved as `eval_val_roc.png`.
    """

    def __init__(self, plot_dir: str | Path):
        self.plot_dir = Path(plot_dir)
        self.scalar_series: dict[str, list[tuple[int, float]]] = {}
        self.curves: dict[str, Columns] = {}
        self._closed = False

    def write(self, event: RunEvent) -> None:
        if event.kind == "scalar" and event.step is not None:
            value = float(event.value)  # type: ignore[arg-type]  # numbers only
            self.scalar_series.setdefault(event.name, []).append((event.step, value))
        elif event.kind == "table" and is_columns(event.value):
            self.curves[event.name] = event.value

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        for name, points in self.scalar_series.items():
            steps, values = zip(*points, strict=True)
            self._save(name, {"step": list(steps), name: list(values)})
        for name, columns in self.curves.items():
            self._save(name, columns)

    def _save(self, name: str, columns: Columns) -> None:
        (x_name, x), *y_columns = columns.items()
        figure = Figure(figsize=(5, 4), layout="constrained")
        axes = figure.subplots()
        for y_name, y in y_columns:
            axes.plot(x, y, marker="." if len(x) <= 50 else None, label=y_name)
        axes.set(title=name, xlabel=x_name)
        if len(y_columns) == 1:
            axes.set_ylabel(y_columns[0][0])
        else:
            axes.legend()
        axes.grid(alpha=0.3)
        figure.savefig(self.plot_dir / f"{name.replace('/', '_')}.png", dpi=120)


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
