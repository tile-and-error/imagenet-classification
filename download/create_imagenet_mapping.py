#!/usr/bin/env python3
"""Map Tiny ImageNet class folders (000-199) to ImageNet-1k labels.

    python tiny_imagenet_mapping.py data/tiny-imagenet

Writes <dir>/imagenet_mapping.json: {folder: {wnid, name, imagenet_1k}}, where
imagenet_1k is the ImageNet-1k class index, or null for the 18 Tiny ImageNet classes
that are not in ImageNet-1k.
"""

import argparse
import ast
import json
from pathlib import Path

from timm.data import ImageNetInfo

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
root = parser.parse_args().dir

(info,) = json.loads((root / "dataset_infos.json").read_text()).values()
wnids = info["features"]["label"]["names"]
names = ast.literal_eval((root / "classes.py").read_text().split("=", 1)[1])
imagenet = {w: i for i, w in enumerate(ImageNetInfo().label_names())}

mapping = {
    f"{i:03d}": {"wnid": w, "name": names[w], "imagenet_1k": imagenet.get(w)}
    for i, w in enumerate(wnids)
}
(root / "imagenet_mapping.json").write_text(json.dumps(mapping, indent=2))
print(
    f"{sum(m['imagenet_1k'] is not None for m in mapping.values())}/200 in ImageNet-1k"
)
