from torch import nn

from lwfm.loss.losses import LossBinding, LossComposer

LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def build_loss_binding(config) -> LossBinding:  # noqa: ANN001
    loss_type = config["type"]

    if loss_type not in LOSSES:
        raise ValueError(f"{loss_type} is not a defined metric")

    loss = LOSSES[loss_type](**config.get("kwargs", {}))

    return LossBinding(config["name"], loss, config["inputs"], config["weight"])


def build_loss_composer(config) -> LossComposer:  # noqa: ANN001
    loss_bindings = [build_loss_binding(loss_config) for loss_config in config]

    return LossComposer(loss_bindings)
