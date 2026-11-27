from torch import nn
from torchvision.transforms import v2


def build_preprocessing(config) -> list[nn.Module]:
    preprocessing: list[nn.Module] = []

    if "resize" in config:
        preprocessing.append(v2.Resize(tuple(config["resize"]), antialias=True))
    if "normalization" in config:
        preprocessing.append(
            v2.Normalize(
                config["normalization"]["mean"], config["normalization"]["std"]
            )
        )

    return preprocessing
