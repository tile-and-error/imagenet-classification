"""Plots as plain data. A `Curve` is a dataclass with a `draw(axes)` method, sent as
the value of a `table` event. During training, at the last validation (or every n-th,
see `Trainer`):

    Trainer().fit(..., curves=[CurveBinding("roc", ROCCurve.from_scores, reads)])

or once, by hand:

    sink.write(RunEvent("table", "eval/test/roc", ROCCurve.from_scores(s, y)))

`LocalFileSink` stores each as its fields (`{"fpr": [...], "tpr": [...], "auc": 0.9}`),
and `PlotSink` calls its `draw`. A plot of your own is one more such dataclass.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryROC,
    MulticlassConfusionMatrix,
)

if TYPE_CHECKING:  # drawing only calls methods on the axes it is given
    from matplotlib.axes import Axes


@runtime_checkable
class Curve(Protocol):
    """Anything `PlotSink` can draw: a table that draws itself on the given axes."""

    def draw(self, axes: "Axes") -> None: ...


def check_same_length(columns: dict[str, list[float]]) -> None:
    lengths = {name: len(column) for name, column in columns.items()}
    if len(set(lengths.values())) != 1 or 0 in lengths.values():
        raise ValueError(f"columns must be non-empty and equally long, got {lengths}")


@dataclass(frozen=True)
class Lines:
    """Each column against the first, e.g. `{"epoch": [...], "top1": [...]}`."""

    columns: dict[str, list[float]]

    def __post_init__(self) -> None:
        if len(self.columns) < 2:
            raise ValueError(
                f"need an x column and a y column, got {list(self.columns)}"
            )
        check_same_length(self.columns)

    def draw(self, axes: "Axes") -> None:
        (x_name, x), *y_columns = self.columns.items()
        for y_name, y in y_columns:
            axes.plot(x, y, marker="." if len(x) <= 50 else None, label=y_name)
        axes.set_xlabel(x_name)
        if len(y_columns) == 1:
            axes.set_ylabel(y_columns[0][0])
        else:
            axes.legend()
        axes.grid(alpha=0.3)


def unit_square(axes: "Axes") -> None:
    axes.set(xlim=(0, 1), ylim=(0, 1.02), aspect="equal")
    axes.legend(loc="lower right")
    axes.grid(alpha=0.3)


@dataclass(frozen=True)
class ROCCurve:
    """True against false positive rate, at every threshold of a binary score, and
    the area under the curve."""

    fpr: list[float]
    tpr: list[float]
    auc: float

    def __post_init__(self) -> None:
        check_same_length({"fpr": self.fpr, "tpr": self.tpr})

    @classmethod
    def from_scores(cls, scores: Tensor, labels: Tensor) -> "ROCCurve":
        """From scores (logits or probabilities) and 0/1 labels."""
        fpr, tpr, _ = BinaryROC()(scores, labels)
        auc = BinaryAUROC()(scores, labels).item()
        return cls(fpr.tolist(), tpr.tolist(), auc)

    def draw(self, axes: "Axes") -> None:
        axes.plot(self.fpr, self.tpr, label=f"AUC = {self.auc:.3f}")
        axes.plot([0, 1], [0, 1], "k:", label="chance")
        axes.set(xlabel="false positive rate", ylabel="true positive rate")
        unit_square(axes)


@dataclass(frozen=True)
class PRCurve:
    """Precision against recall, at every threshold of a binary score, and the
    average precision (the area under the curve, as torchmetrics defines it)."""

    precision: list[float]
    recall: list[float]
    average_precision: float

    def __post_init__(self) -> None:
        check_same_length({"precision": self.precision, "recall": self.recall})

    @classmethod
    def from_scores(cls, scores: Tensor, labels: Tensor) -> "PRCurve":
        """From scores (logits or probabilities) and 0/1 labels."""
        precision, recall, _ = BinaryPrecisionRecallCurve()(scores, labels)
        average_precision = BinaryAveragePrecision()(scores, labels).item()
        return cls(precision.tolist(), recall.tolist(), average_precision)

    def draw(self, axes: "Axes") -> None:
        axes.plot(
            self.recall, self.precision, label=f"AP = {self.average_precision:.3f}"
        )
        axes.set(xlabel="recall", ylabel="precision")
        unit_square(axes)


@dataclass(frozen=True)
class ConfusionMatrix:
    """`matrix[true][predicted]` counts, one row and column per class."""

    matrix: list[list[int]]
    class_names: list[str]

    def __post_init__(self) -> None:
        k = len(self.class_names)
        if len(self.matrix) != k or any(len(row) != k for row in self.matrix):
            raise ValueError(f"matrix must be {k}x{k}, one row per class name")

    @classmethod
    def from_scores(
        cls, scores: Tensor, labels: Tensor, class_names: list[str] | None = None
    ) -> "ConfusionMatrix":
        """From a binary score `[B]` (positive above probability 0.5) or class scores
        `[B, K]` (the highest is predicted), and true classes `[B]`. Classes are
        named 0, 1, ... unless `class_names` is given."""
        binary = scores.ndim == 1
        k = 2 if binary else scores.shape[-1]
        count = BinaryConfusionMatrix() if binary else MulticlassConfusionMatrix(k)
        names = list(class_names) if class_names else [str(i) for i in range(k)]
        return cls(count(scores, labels).tolist(), names)

    def draw(self, axes: "Axes") -> None:
        image = axes.imshow(self.matrix, cmap="Blues")
        axes.figure.colorbar(image, ax=axes)
        ticks = range(len(self.class_names))
        axes.set(xlabel="predicted", ylabel="true", xticks=ticks, yticks=ticks)
        axes.set_xticklabels(self.class_names, rotation=45, ha="right")
        axes.set_yticklabels(self.class_names)
        if len(self.class_names) <= 20:  # counts stay readable
            peak = max(max(row) for row in self.matrix)
            for true, row in enumerate(self.matrix):
                for predicted, count in enumerate(row):
                    colour = "white" if count > peak / 2 else "black"
                    axes.text(
                        predicted, true, str(count), ha="center", va="center", c=colour
                    )
