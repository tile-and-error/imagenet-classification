"""Common evaluation graphs as plain data, sent as the value of a `table` event. During
training, at the last validation (or every n-th, see `Trainer`):

    Trainer().fit(..., curves=[CurveBinding("roc", ROCCurve.from_scores, reads)])

or once, by hand:

    sink.write(RunEvent("table", "eval/test/roc", ROCCurve.from_scores(s, y)))

`LocalFileSink` stores each as its fields (`{"fpr": [...], "tpr": [...], "auc": 0.9}`),
and `PlotSink` draws each in its standard form.
"""

from dataclasses import dataclass

from torch import Tensor
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryConfusionMatrix,
    BinaryPrecisionRecallCurve,
    BinaryROC,
    MulticlassConfusionMatrix,
)


def check_same_length(columns: dict[str, list[float]]) -> None:
    lengths = {name: len(column) for name, column in columns.items()}
    if len(set(lengths.values())) != 1 or 0 in lengths.values():
        raise ValueError(f"columns must be non-empty and equally long, got {lengths}")


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
