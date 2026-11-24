from pathlib import Path

from matplotlib.figure import Figure  # no pyplot: no GUI backend or global state

from lwfm.results.curves import Curve, Lines
from lwfm.results.events import RunEvent
from lwfm.results.sinks import OutputSink


class PlotSink(OutputSink):
    """Draws the run's events as PNG files in `plot_dir`. Two kinds of event are
    drawn; the rest are ignored:

    - Scalars over training, when the sink closes: every `scalar` event named e.g.
      `train/loss/total` gives one point (step, value), and the points are joined
      into one line plot of that value against the training step, saved as
      `train_loss_total.png`. Scalars without a step are skipped.
    - Curves, as soon as they arrive: a `table` event whose value is a `Curve` (from
      `lwfm.results.curves`, or your own) is drawn by its `draw(axes)`.
      `eval/val/roc` at step 120 is saved as `eval_val_roc_step120.png`, so each
      validation keeps its own plot; without a step, as `eval_val_roc.png`.

    For a plot that isn't a curve, e.g. one drawn after training, draw the figure
    yourself and save it with `save_artifact(..., figure.savefig, ...)`.
    """

    def __init__(self, plot_dir: str | Path):
        self.plot_dir = Path(plot_dir)
        self.scalar_series: dict[str, list[tuple[int, float]]] = {}
        self._closed = False

    def write(self, event: RunEvent) -> None:
        if event.kind == "scalar" and event.step is not None:
            value = float(event.value)  # type: ignore[arg-type]  # numbers only
            self.scalar_series.setdefault(event.name, []).append((event.step, value))
        elif event.kind == "table" and isinstance(event.value, Curve):
            self._save(event.name, event.value, event.step)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, points in self.scalar_series.items():
            steps, values = zip(*points, strict=True)
            self._save(name, Lines({"step": list(steps), name: list(values)}))

    def _save(self, name: str, curve: Curve, step: int | None = None) -> None:
        figure = Figure(figsize=(5, 4), layout="constrained")
        axes = figure.subplots()
        curve.draw(axes)
        axes.set_title(name if step is None else f"{name} at step {step}")
        file_name = name.replace("/", "_") + ("" if step is None else f"_step{step}")
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.plot_dir / f"{file_name}.png", dpi=120)
