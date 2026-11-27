import platform

import torch
import torchvision


def environment() -> dict[str, str | None]:
    """Versions and machine to record with every run."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
