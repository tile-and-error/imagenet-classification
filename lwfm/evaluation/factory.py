from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from lwfm.evaluation.metrics import MetricBinding

METRICS = {"cross_entropy": nn.CrossEntropyLoss, "topk_accuracy": MulticlassAccuracy}


def build_metric(metric):
    metric_type = metric["type"]

    if metric_type not in METRICS:
        return ValueError(f"{metric_type} is not a defined metric")

    return METRICS[metric_type](**metric.get("kwargs", {}))


def build_metric_binding(config):
    metrics = [
        MetricBinding(
            metric_config["name"], build_metric(metric_config), metric_config["inputs"]
        )
        for metric_config in config
    ]

    return metrics
