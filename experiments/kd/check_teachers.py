"""Downloads every KD teacher's weights and checks that each one works.

Run it once, where there is internet access (e.g. the cluster's login node), before
submitting a batch job. From the repository root:

    python experiments/kd/check_teachers.py            # every teacher in parts.TEACHERS
    python experiments/kd/check_teachers.py resnet50   # only these

For each teacher it:
1. builds it, which downloads its weights (and DINOv2's code) into the torch cache, so
   batch jobs load them from disk;
2. checks its output shapes: 1000 ImageNet logits and an embedding of `spec.dim`;
3. scores it on 200 Tiny ImageNet training images from 50 classes that are in
   ImageNet-1k (1000-way top-1, upsampled from 64 px, as in KD). Random weights score
   ~0.1 %, so a score under 5 % means the weights did not load properly.

Exits with an error if any teacher fails, so a job script can run it as a check.
"""

import sys
import time
import traceback
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import to_dtype

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.kd.parts import (  # noqa: E402
    TEACHERS,
    pick_device,
    teacher_model,
    tiny_mapping,
)

ROOT = REPO / "data" / "tiny-imagenet"
MIN_TOP1 = 0.05


def images(
    per_class: int = 4, num_classes: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    # The first images of the first ImageNet-matched folders, with ImageNet labels
    mapping = tiny_mapping()
    folders = [f for f in sorted(mapping) if mapping[f]["imagenet_1k"] is not None]
    x, y = [], []
    for folder in folders[:num_classes]:
        for path in sorted((ROOT / "train" / folder).iterdir())[:per_class]:
            x.append(decode_image(str(path), mode=ImageReadMode.RGB))  # some are grey
            y.append(mapping[folder]["imagenet_1k"])
    return to_dtype(torch.stack(x), torch.float32, scale=True), torch.tensor(y)


@torch.no_grad()
def check(name: str, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> str:
    start = time.perf_counter()
    model = teacher_model(name).eval().to(device)
    outs = [model(chunk.to(device)) for chunk in x.split(50)]
    logits = torch.cat([o["logits"] for o in outs]).cpu()
    embed = torch.cat([o["embed"] for o in outs])
    assert logits.shape == (len(x), 1000), f"logits {tuple(logits.shape)}"
    assert embed.shape == (len(x), TEACHERS[name].dim), f"embed {tuple(embed.shape)}"
    top1 = (logits.argmax(1) == y).float().mean().item()
    assert top1 >= MIN_TOP1, f"top-1 {top1:.1%}: the weights look untrained"
    return (
        f"top-1 {top1:.1%}, embed {embed.shape[1]}, {time.perf_counter() - start:.0f} s"
    )


def main() -> None:
    names = sys.argv[1:] or list(TEACHERS)
    device = pick_device()
    x, y = images()
    print(f"device {device}, {len(x)} images, teachers: {', '.join(names)}")
    failed = []
    for name in names:
        try:
            print(f"OK    {name}: {check(name, x, y, device)}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}:\n{traceback.format_exc()}", flush=True)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")
    print("all teachers work")


if __name__ == "__main__":
    main()
