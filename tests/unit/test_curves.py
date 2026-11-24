import json

import pytest
import torch

from lwfm.results.curves import ConfusionMatrix, Lines, PRCurve, ROCCurve
from lwfm.results.events import RunEvent
from lwfm.results.router import OutputRouter
from lwfm.results.sinks import LocalFileSink

# Two negatives, four positives; one positive (-0.5) scores below a negative (0.2)
SCORES = torch.tensor([-2.0, 0.2, 0.5, 1.0, 3.0, -0.5])
LABELS = torch.tensor([0, 0, 1, 1, 1, 1])


def test_roc_from_scores_matches_a_hand_worked_example():
    roc = ROCCurve.from_scores(SCORES, LABELS)
    assert roc.fpr[0] == roc.tpr[0] == 0.0 and roc.fpr[-1] == roc.tpr[-1] == 1.0
    # 1 of the 4 * 2 positive-negative pairs is ranked wrongly
    assert roc.auc == pytest.approx(7 / 8)
    # the stored AUC is the area under the stored curve
    assert roc.auc == pytest.approx(
        torch.trapezoid(torch.tensor(roc.tpr), torch.tensor(roc.fpr)).item()
    )


def test_pr_from_scores_matches_a_hand_worked_example():
    pr = PRCurve.from_scores(SCORES, LABELS)
    # Ranked by score: p p p n p n. Precision at each positive: 1, 1, 1, 4/5
    assert pr.average_precision == pytest.approx((1 + 1 + 1 + 0.8) / 4)
    assert pr.recall[0] == 1.0 and pr.recall[-1] == 0.0


def test_confusion_matrix_thresholds_a_binary_score():
    # Logit > 0 is positive: one negative (0.2) above, one positive (-0.5) below
    confusion = ConfusionMatrix.from_scores(SCORES, LABELS)
    assert confusion.matrix == [[1, 1], [1, 3]]  # rows true, columns predicted
    assert confusion.class_names == ["0", "1"]


def test_confusion_matrix_predicts_the_highest_class_score():
    scores = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    confusion = ConfusionMatrix.from_scores(scores, torch.tensor([0, 1]), list("abc"))
    assert confusion.matrix == [[1, 0, 0], [0, 0, 1], [0, 0, 0]]
    assert confusion.class_names == ["a", "b", "c"]


@pytest.mark.parametrize(
    "build",
    [
        lambda: ROCCurve([0.0, 1.0], [0.0], 0.5),
        lambda: ROCCurve([], [], 0.5),
        lambda: PRCurve([1.0], [0.0, 1.0], 0.5),
        lambda: ConfusionMatrix([[1, 0], [0, 1]], ["only one name"]),
        lambda: ConfusionMatrix([[1, 0], [0]], ["a", "b"]),
        lambda: Lines({"x": [0.0, 1.0]}),  # nothing to plot against x
        lambda: Lines({"x": [0.0, 1.0], "y": [1.0]}),
    ],
)
def test_malformed_values_fail_when_built(build):
    with pytest.raises(ValueError):
        build()


def test_the_run_record_stores_the_fields_as_plain_data(tmp_path):
    roc = ROCCurve([0.0, 0.5, 1.0], [0.0, 0.9, 1.0], 0.7)
    with OutputRouter(LocalFileSink(tmp_path)) as sink:
        sink.write(RunEvent("table", "eval/val/roc", roc, step=3))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").open()]
    [event] = [e for e in events if e["kind"] == "table"]
    assert event["value"] == {
        "fpr": [0.0, 0.5, 1.0],
        "tpr": [0.0, 0.9, 1.0],
        "auc": 0.7,
    }
