import random

import numpy as np  # a torchvision dependency, so always installed
import torch


def seed_everything(seed: int) -> None:
    """Seeds Python's, NumPy's and torch's generators (torch's covers every CUDA
    device). Call it before building the model: weight initialisation draws from them.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
