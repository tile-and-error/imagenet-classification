import torch.nn as nn
import torchvision.models

from lwfm.core.graph import ModelNode
from lwfm.models.preprocessing import build_preprocessing


def build_torchvision_model(config: dict) -> nn.Module:
    model_registry = {"resnet18": torchvision.models.resnet18}

    model_name = config["name"]
    num_classes = config["num_classes"]

    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {config['name']}")

    base_model = model_registry[model_name](
        weights=None, num_classes=num_classes, **config.get("kwargs", {})
    )

    return base_model


MODEL_BUILDERS = {"torchvision": build_torchvision_model}


def build_model(config: dict) -> ModelNode:
    type = config["type"]

    if type not in MODEL_BUILDERS:
        raise ValueError(f"A builder does not exist for type {type}")

    base_model = MODEL_BUILDERS[type](config)

    preprocessing = config["preprocess"]

    if len(preprocessing) == 0:
        model = base_model
    else:
        model_with_preprocessing = build_preprocessing(preprocessing) + [base_model]
        model = nn.Sequential(*model_with_preprocessing)

    model_node = ModelNode(
        name=config["node"],
        module=model,
        inputs=config["inputs"],
        outputs=config["outputs"],
    )

    return model_node
