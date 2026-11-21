from torch import nn
from torchmetrics.classification import MulticlassAccuracy

METRICS = {
    "cross_entropy": nn.CrossEntropyLoss,
    "topk_accuracy": MulticlassAccuracy
}

def build_metric(metric):
    metric_type = metric["type"]

    if metric_type not in METRICS:
        return ValueError(f"{metric_type} is not a defined metric")

    return METRICS[metric_type](
        **metric.get("kwargs", {})
    )