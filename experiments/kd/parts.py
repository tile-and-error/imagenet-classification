"""Reusable, swappable parts for knowledge-distillation runs.

Built only from lwfm (graph, losses, evaluator, results) and torch, torchvision and
torchmetrics. Every model returns {"logits": [B, C], "embed": [B, D]}, so any dataset,
student, teacher or loss can be swapped for another without touching the rest:

    DATASETS   where images live, their size and transforms, and how class folders
               map to ImageNet-1k (for slicing the teachers' logits)
    STUDENTS   models trained from scratch; they adapt to the dataset's image size
    TEACHERS   frozen pretrained models
    Setup      one run's choice of student, teacher and losses
"""

import csv
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from itertools import islice
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.regression import CosineSimilarity
from torchvision.io import ImageReadMode, decode_image
from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet34,
    resnet50,
    vit_b_16,
)
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms import v2

from lwfm.core.batch import to_device
from lwfm.core.graph import ExecutionGraph, ModelNode
from lwfm.core.losses import LossBinding, LossComposer
from lwfm.core.types import Batch
from lwfm.evaluation.benchmarks import inference_cost
from lwfm.evaluation.evaluator import Evaluator
from lwfm.evaluation.metrics import CurveBinding, MetricBinding
from lwfm.results.artifacts import save_artifact
from lwfm.results.curves import ConfusionMatrix
from lwfm.results.events import RunEvent
from lwfm.results.plots import PlotSink
from lwfm.results.router import OutputRouter
from lwfm.results.runs import completed_runs, skip_if_completed
from lwfm.results.sinks import ConsoleSink, LocalFileSink, OutputSink
from lwfm.techniques.distillation import DIST, DKD, KD, EmbedCos, EmbedMSE  # noqa: F401
from lwfm.training import Trainer, TrainingResult, seed_everything
from lwfm.training.callbacks import LossWeightSchedule, RestoreBestWeights

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics

# The values each loss reads, in argument order (see lwfm.core.names)
CE = ["student.logits", "targets.class"]
LOGITS = ["student.logits", "teacher.logits"]
EMBED = ["projector.embed", "teacher.embed"]
KD_ONLY_NODES = ("teacher", "projector")  # needed for training, never for predicting
CPU = torch.device("cpu")


def pick_device() -> torch.device:
    """CUDA if there is a GPU, else Apple's MPS, else the CPU."""
    # A local, uncommitted override: a file `.kd-device` at the repository root
    # holding e.g. "cpu" (it is listed in .git/info/exclude, so it is never pushed)
    override = REPO / ".kd-device"
    if override.exists():
        return torch.device(override.read_text().strip())
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return CPU


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------------ datasets


@dataclass(frozen=True)
class DatasetSpec:
    """A dataset downloaded by download/download_data.py: root/{train,test}/<class>/.

    Transforms output float images in [0, 1]; each model normalises its own input, so
    teacher and student can read the same tensor. `train` holds TrainConfig settings
    this dataset needs (e.g. weight decay); never budget fields such as epochs.
    """

    root: Path
    image_size: int  # what the student sees
    train_transform: v2.Transform
    eval_transform: v2.Transform
    imagenet_index: Callable[[Path, str], int | None]  # (root, folder) -> IN-1k index
    class_name: Callable[[Path, str], str]  # (root, folder) -> readable name
    train: dict = field(default_factory=dict)


TINY_MAPPING = REPO / "download/imagenet_mapping.json"  # committed: Tiny -> IN-1k


def tiny_mapping() -> dict:
    # {folder: {wnid, name, imagenet_1k}}; imagenet_1k is null for the 18 classes
    # that are not in ImageNet-1k
    return json.loads(TINY_MAPPING.read_text())


IMAGENET_NAMES = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]  # in index order
TO_FLOAT = v2.ToDtype(torch.float32, scale=True)  # uint8 [0, 255] -> float [0, 1]

DATASETS = {
    "tiny-imagenet": DatasetSpec(
        root=REPO / "data/tiny-imagenet",
        image_size=64,
        train_transform=v2.Compose(
            [
                v2.RandomResizedCrop(64, scale=(0.25, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                TO_FLOAT,
            ]
        ),
        eval_transform=TO_FLOAT,  # images are already 64 px
        imagenet_index=lambda _, c: tiny_mapping()[c]["imagenet_1k"],
        class_name=lambda _, c: tiny_mapping()[c]["name"].split(",")[0],
    ),
    "imagenet-1k": DatasetSpec(
        root=REPO / "data/imagenet-1k",
        image_size=224,
        train_transform=v2.Compose(
            [
                v2.RandomResizedCrop(224, antialias=True),
                v2.RandomHorizontalFlip(),
                TO_FLOAT,
            ]
        ),
        eval_transform=v2.Compose(
            [v2.Resize(256, antialias=True), v2.CenterCrop(224), TO_FLOAT]
        ),
        # Folders 000-999 are already ImageNet-1k's index order
        imagenet_index=lambda _, c: int(c),
        class_name=lambda _, c: IMAGENET_NAMES[int(c)],
        train={"weight_decay": 1e-4},  # the standard ImageNet ResNet setting
    ),
}


def make_split(
    dataset: str,
    keep: list[str] | None = None,
    num_other: int | None = None,
    train_fraction: float = 0.9,
    seed: int = 5504132,
    exist_ok: bool = False,
) -> Path:
    """Writes a split's manifests (train, val, test, class_to_idx, split .csv) and
    returns its folder. The `keep` folders get labels 0..k-1 and every other folder
    shares one "other" label (keep=None: every class keeps its own label)."""
    spec = DATASETS[dataset]
    root = spec.root
    folders = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
    keep = folders if keep is None else keep
    rest = [f for f in folders if f not in keep]
    # num_other=n: only n other folders, drawn from those with an ImageNet-1k match
    # (e.g. 10 + 40 instead of 10 + 190). The teacher's "other" logit merges the
    # classes behind it, and fewer of them bias it less towards "other" (see
    # ToStudentClasses); runs are faster too. The cost: a less varied "other" class
    if num_other is not None:
        matched = [f for f in rest if spec.imagenet_index(root, f) is not None]
        rest = sorted(random.Random(seed).sample(matched, num_other))
    class_to_idx = {c: i for i, c in enumerate(keep)}
    if rest:
        class_to_idx["other"] = len(keep)
    tag = "all" if not rest else f"{len(keep)}+{len(rest)}other"
    pct = round(100 * train_fraction)
    out = root / "splits" / f"{dataset}-{tag}-{pct}-{100 - pct}_{seed}"
    # The name fixes every setting, so an existing split of this name is the same
    # split: reuse it (exist_ok), but never overwrite it
    if out.exists():
        if exist_ok:
            return out
        raise FileExistsError(f"{out} already exists; delete it to regenerate")
    folders = keep + rest  # folders left out of a limited split are never read

    def rows(folder: str, files: list[Path]) -> list[dict]:
        label = class_to_idx.get(folder, class_to_idx.get("other"))
        return [
            {
                "sample_id": f.relative_to(root).with_suffix("").as_posix(),
                "path": f.relative_to(root).as_posix(),
                "label": label,
            }
            for f in files
        ]

    # val is a per-class hold-out from train/; test/ is used as given
    rng = random.Random(seed)  # same split on every run and machine
    splits = {"train": [], "val": [], "test": []}
    for folder in folders:
        files = sorted((root / "train" / folder).iterdir())
        rng.shuffle(files)
        n_train = round(train_fraction * len(files))
        splits["train"] += rows(folder, sorted(files[:n_train]))
        splits["val"] += rows(folder, sorted(files[n_train:]))
        splits["test"] += rows(folder, sorted((root / "test" / folder).iterdir()))

    out.mkdir(parents=True)
    for split, split_rows in splits.items():
        with open(out / f"{split}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "path", "label"])
            writer.writeheader()
            writer.writerows(split_rows)
    with open(out / "class_to_idx.csv", "w", newline="") as f:
        csv.writer(f).writerows([("class", "idx"), *class_to_idx.items()])
    info = {
        "root": root.name,
        "seed": seed,
        "train_fraction": train_fraction,
        "holdout": "per source class, from train/",
        "test": "test/",
        "other_folders": " ".join(rest),
    }
    info |= {f"{split}_size": len(r) for split, r in splits.items()}
    with open(out / "split.csv", "w", newline="") as f:
        csv.writer(f).writerows([("key", "value"), *info.items()])
    return out


class ManifestDataset(Dataset):
    """Manifest rows (sample_id, path, label) as samples in the lwfm batch layout."""

    def __init__(self, root: Path, rows: list[dict[str, str]], transform: v2.Transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        # RGB mode also converts greyscale and CMYK images to 3 channels
        image = decode_image(str(self.root / row["path"]), mode=ImageReadMode.RGB)
        return {
            "inputs": {"image": self.transform(image)},
            "targets": {"class": int(row["label"])},
            "meta": {"sample_id": row["sample_id"]},
        }


class SplitData:
    """One split of one dataset: loaders, class count, class weights and the
    teacher's class indices, all from root/splits/<split>/.

        SplitData("tiny-imagenet", "tiny-70-30_5504132")
        SplitData("imagenet-1k", make_split("imagenet-1k", train_fraction=0.96).name)
    """

    def __init__(self, dataset: str, split: str, workers: int = 8):
        self.spec = DATASETS[dataset]
        self.root, self.workers = self.spec.root, workers
        self.splits = self.root / "splits" / split
        self.name = f"{dataset}/{split}"
        rows = sorted(
            read_csv(self.splits / "class_to_idx.csv"), key=lambda r: int(r["idx"])
        )
        self.classes = [r["class"] for r in rows]  # e.g. "000", ..., "009", "other"
        self.num_classes = len(self.classes)
        self.has_other = self.classes[-1] == "other"
        self._rows: dict[str, list] = {}

    def rows(self, split: str) -> list[dict[str, str]]:
        """A split's manifest rows: sample_id, path and label."""
        if split not in self._rows:  # read each manifest once
            self._rows[split] = read_csv(self.splits / f"{split}.csv")
        return self._rows[split]

    def loader(
        self,
        split: str,
        batch_size: int = 256,
        *,
        augment: bool = True,
        seed: int = 0,
        device: torch.device = CPU,
    ) -> DataLoader:
        """A shuffled loader over a split, augmented for train unless augment=False."""
        train = split == "train" and augment
        transform = self.spec.train_transform if train else self.spec.eval_transform
        return DataLoader(
            ManifestDataset(self.root, self.rows(split), transform),
            batch_size=batch_size,
            # Eval splits are shuffled too, in a fixed order: that doesn't change a
            # full-split metric, but a capped evaluation (eval_batches) then sees a
            # random mix of classes, not the first few (manifests are sorted by class)
            shuffle=True,
            drop_last=train,
            generator=torch.Generator().manual_seed(seed),  # same order for a seed
            num_workers=self.workers,
            pin_memory=device.type == "cuda",
        )

    def class_weights(self) -> Tensor:
        """w_c = N / (C * n_c): balances an imbalanced split in the CE loss
        (all ones for a balanced split)."""
        labels = torch.tensor([int(r["label"]) for r in self.rows("train")])
        counts = torch.bincount(labels, minlength=self.num_classes)
        return counts.sum() / (self.num_classes * counts.float())

    def teacher_classes(self) -> list[int]:
        """ImageNet-1k index of each named class, in label order ("other" excluded)."""
        named = [c for c in self.classes if c != "other"]
        index = [self.spec.imagenet_index(self.root, c) for c in named]
        missing = [c for c, i in zip(named, index, strict=True) if i is None]
        assert not missing, f"classes {missing} have no ImageNet-1k equivalent"
        return index

    def other_folders(self) -> list[str]:
        """The source folders behind "other", read from the train manifest's paths
        (train/<folder>/<file>), so this works for any split."""
        if not self.has_other:
            return []
        label = str(self.num_classes - 1)
        return sorted(
            {r["path"].split("/")[1] for r in self.rows("train") if r["label"] == label}
        )

    def teacher_other_classes(self) -> list[int]:
        """ImageNet-1k indices of the classes behind "other". Folders with no
        ImageNet-1k match (18 of Tiny ImageNet's 200) have no teacher logit to
        merge, so they are skipped."""
        index = [self.spec.imagenet_index(self.root, f) for f in self.other_folders()]
        return [i for i in index if i is not None]

    def class_names(self) -> list[str]:
        return [
            "other" if c == "other" else self.spec.class_name(self.root, c)
            for c in self.classes
        ]


# -------------------------------------------------------------------------- models


class ResNetWithEmbed(nn.Module):
    """A torchvision ResNet that also returns its pooled feature."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        n = self.net  # torchvision's ResNet.forward, stopping before fc
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        embed = torch.flatten(n.avgpool(x), 1)
        return {"logits": n.fc(embed), "embed": embed}


class ViTWithEmbed(nn.Module):
    """A torchvision VisionTransformer that also returns its class token."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        n = self.net  # torchvision's VisionTransformer.forward, stopping before heads
        x = n._process_input(x)
        x = torch.cat([n.class_token.expand(len(x), -1, -1), x], dim=1)
        embed = n.encoder(x)[:, 0]
        return {"logits": n.heads(embed), "embed": embed}


class DinoV2WithEmbed(nn.Module):
    """DINOv2 with its linear head (torch.hub `dinov2_vitb14_lc`)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # As the hub wrapper's forward: the head reads the class tokens of the last
        # `layers` blocks (4 for _lc) and the last block's mean patch token, so its
        # input is (4 + 1) * 768 = 3840. The last class token is the embedding.
        out = self.net.backbone.get_intermediate_layers(
            x, n=self.net.layers, return_class_token=True
        )
        cls = [c for _, c in out]
        head_in = torch.cat([*cls, out[-1][0].mean(1)], dim=1)
        return {"logits": self.net.linear_head(head_in), "embed": cls[-1]}


class ToStudentClasses(nn.Module):
    """Maps a teacher's 1000 ImageNet logits to the student's classes: the `keep`
    logits in label order, then one "other" logit merging the `other` classes,
    logsumexp(z[other]) - shift. Logits of classes in neither list are dropped."""

    def __init__(
        self,
        keep: list[int],
        other: list[int] | None = None,
        shift: float | None = None,
    ):
        super().__init__()
        assert len(set(keep)) == len(keep), "two classes share a teacher class"
        assert not set(keep) & set(other or []), "a class is both kept and other"
        self.has_other = bool(other)
        # Why the shift: merging n classes makes "other" n times likelier than a kept
        # class before the image is seen. When the teacher is unsure (e.g. 64 px
        # images upsampled to 224 px) that prior wins, and it calls most kept-class
        # images "other". Subtracting log n removes the bias (the standard
        # logit-adjustment correction) but pulls "other" images towards the kept
        # classes, so tune_other_shift picks a shift between on val, by macro-F1.
        # ResNet-50, Tiny 10 + 40 val: macro-F1 0.58 at 0, 0.69 at 2.3, 0.57 at 3.7.
        if self.has_other and shift is None:
            shift = math.log(len(other))  # the balanced prior
        self.shift = shift if self.has_other else 0.0
        self.register_buffer("keep", torch.tensor(keep), persistent=False)
        self.register_buffer("other", torch.tensor(other or []), persistent=False)

    def forward(self, out: dict[str, Tensor]) -> dict[str, Tensor]:
        z = out["logits"]
        logits = z[:, self.keep]
        if self.has_other:
            other = torch.logsumexp(z[:, self.other], dim=1, keepdim=True) - self.shift
            logits = torch.cat([logits, other], dim=1)
        return out | {"logits": logits}


def resnet18_student(num_classes: int, image_size: int) -> nn.Module:
    """ResNet-18 from scratch. At 64 px the standard stem (7x7 stride-2 conv +
    max-pool) would shrink the image to 16 px before layer1, so it becomes a 3x3
    stride-1 conv with no pool. At 224 px it is the standard ResNet-18."""
    net = resnet18(num_classes=num_classes)
    if image_size <= 64:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
    return ResNetWithEmbed(net)


def vit_ti_student(num_classes: int, image_size: int) -> nn.Module:
    """DeiT-Tiny-shaped ViT from scratch (12 layers, 3 heads, width 192, 5.5 M
    parameters). Patch 8 at 64 px and patch 16 at 224 px, giving 8x8 and 14x14
    tokens."""
    return ViTWithEmbed(
        VisionTransformer(
            image_size=image_size,
            patch_size=8 if image_size <= 64 else 16,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            num_classes=num_classes,
        )
    )


def weights(w, pretrained: bool):
    return w if pretrained else None


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model. Students: build(num_classes, image_size); teachers:
    build(pretrained). Both return {"logits", "embed"}. `train` holds TrainConfig
    settings the model needs (e.g. a ViT student trains with AdamW)."""

    build: Callable[..., nn.Module]
    dim: int  # embed size
    input_size: int | None = None  # teachers: their native input size
    mean: tuple[float, ...] = MEAN
    std: tuple[float, ...] = STD
    train: dict = field(default_factory=dict)


STUDENTS = {
    "resnet18": ModelSpec(resnet18_student, dim=512),
    "vit_ti": ModelSpec(  # ViTs train poorly with SGD: AdamW, warmup and clipping
        vit_ti_student,
        dim=192,
        train={
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 0.05,
            "warmup_epochs": 10,
            "grad_clip": 1.0,
        },
    ),
}

# Teachers: weights download on first use, then load from the torch cache
TEACHERS = {
    "resnet50": ModelSpec(  # IN-1k top-1 80.9
        lambda p: ResNetWithEmbed(
            resnet50(weights=weights(ResNet50_Weights.IMAGENET1K_V2, p))
        ),
        dim=2048,
        input_size=224,
    ),
    "resnet34": ModelSpec(  # 73.3
        lambda p: ResNetWithEmbed(
            resnet34(weights=weights(ResNet34_Weights.IMAGENET1K_V1, p))
        ),
        dim=512,
        input_size=224,
    ),
    "vit_b16": ModelSpec(  # 81.1, supervised on IN-1k
        lambda p: ViTWithEmbed(
            vit_b_16(weights=weights(ViT_B_16_Weights.IMAGENET1K_V1, p))
        ),
        dim=768,
        input_size=224,
    ),
    "vit_b16_swag": ModelSpec(  # 81.9, weakly supervised pre-training, linear head
        lambda p: ViTWithEmbed(
            vit_b_16(weights=weights(ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1, p))
        ),
        dim=768,
        input_size=224,
    ),
    "dinov2_b14": ModelSpec(  # 84.5, self-supervised; code + weights from GitHub
        lambda p: DinoV2WithEmbed(
            # ":main" pins the branch, so a cached copy loads without internet access
            torch.hub.load(
                "facebookresearch/dinov2:main", "dinov2_vitb14_lc", pretrained=p
            )
        ),
        dim=768,
        input_size=224,
    ),
}


def student_node(name: str, data: SplitData) -> ModelNode:
    """The student as a graph node: its normalisation, then the model."""
    spec = STUDENTS[name]
    model = spec.build(data.num_classes, data.spec.image_size)
    return ModelNode(
        "student",
        nn.Sequential(v2.Normalize(spec.mean, spec.std), model),
        inputs="inputs.image",
        outputs=["logits", "embed"],
    )


def teacher_model(name: str, pretrained: bool = True) -> nn.Sequential:
    """A teacher as trained: resize to its input size, normalise, 1000-way output."""
    spec = TEACHERS[name]
    return nn.Sequential(
        v2.Resize((spec.input_size, spec.input_size), antialias=True),
        v2.Normalize(spec.mean, spec.std),
        spec.build(pretrained),
    )


def to_student_classes(data: SplitData, other_shift: float | None) -> ToStudentClasses:
    other = data.teacher_other_classes() if data.has_other else None
    return ToStudentClasses(data.teacher_classes(), other, other_shift)


def teacher_node(
    name: str,
    data: SplitData,
    pretrained: bool = True,
    other_shift: float | None = None,
    head: str | None = None,
) -> ModelNode:
    """The frozen teacher as a graph node, reading the student's own augmented image
    (so both see the same view). Its class logits come from its mapped ImageNet
    logits, or from a head trained on its embedding (head="linear" or "mlp", E0c)."""
    last = (
        TaskHead(load_teacher_head(head, name, data, pretrained))
        if head
        else to_student_classes(data, other_shift)
    )
    return ModelNode(
        "teacher",
        nn.Sequential(*teacher_model(name, pretrained), last),
        inputs="inputs.image",
        outputs=["logits", "embed"],
        frozen=True,  # eval mode, no gradients, and graph.train() can't undo it
    )


@torch.inference_mode()
def tune_other_shift(
    teacher: str,
    data: SplitData,
    device: torch.device,
    *,
    pretrained: bool = True,
    eval_batches: int | None = None,
    batch_size: int = 256,
    steps: int = 13,
) -> list[dict]:
    """Scores the teacher on val at `steps` shifts from 0 to 1.5 log n (n = classes
    behind "other"), from a single pass of the teacher. Returns one row per shift."""
    assert data.has_other, "only splits with an 'other' class have a shift to tune"
    model = teacher_model(teacher, pretrained).eval().to(device)
    logits, labels = [], []
    for batch in islice(data.loader("val", batch_size, device=device), eval_batches):
        # Only the images go to the device; the labels stay on the CPU, where the
        # metrics below run
        image = batch["inputs"]["image"].to(device)
        logits.append(model(image)["logits"].float().cpu())
        labels.append(batch["targets"]["class"])
    z, y = torch.cat(logits), torch.cat(labels)
    k, is_other = data.num_classes, y == data.num_classes - 1
    max_shift = 1.5 * math.log(len(data.teacher_other_classes()))
    rows = []
    for i in range(steps):
        shift = max_shift * i / (steps - 1)
        p = to_student_classes(data, shift)({"logits": z})["logits"].softmax(1)
        rows.append(
            {
                "shift": round(shift, 3),
                "bal_acc": MulticlassAccuracy(k, average="macro")(p, y).item(),
                "macro_f1": MulticlassF1Score(k, average="macro")(p, y).item(),
                "p_other_on_other": p[is_other, -1].mean().item(),
            }
        )
    return rows


# ------------------------------------------------------------- teacher heads (E0c)
# Instead of mapping the teacher's ImageNet logits to the student's classes, train a
# small head for those classes on the frozen teacher's embedding. It learns the task
# from its labels, including "other", and adapts to upsampled 64 px images.

HEADS = {
    # The features were trained for a linear head, so start there
    "linear": lambda d, k: nn.Linear(d, k),
    # One hidden layer: can undo some distortion from upsampling, but overfits sooner
    "mlp": lambda d, k: nn.Sequential(
        nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, k)
    ),
}


class TaskHead(nn.Module):
    """The teacher's last step when it has a trained head: logits from its embed."""

    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, out: dict[str, Tensor]) -> dict[str, Tensor]:
        return out | {"logits": self.head(out["embed"])}


def head_path(kind: str, teacher: str, data: SplitData, pretrained: bool) -> Path:
    return (
        cache_dir(pretrained)
        / f"head_{kind}_{teacher}_{data.name.replace('/', '_')}.pt"
    )


def fit_teacher_head(
    teacher: str,
    data: SplitData,
    device: torch.device,
    kind: str,
    *,
    pretrained: bool = True,
    max_batches: int | None = None,
    batch_size: int = 256,
    weight_decays: tuple[float, ...] = (0.01, 0.1, 1.0),
    steps: int = 2000,
    eval_every: int = 50,
) -> list[dict]:
    """Trains a `kind` head ("linear" or "mlp") on the teacher's cached embeddings,
    once per weight decay, and saves the best for teacher_node(head=kind). Returns
    one row per weight decay. A head already saved is reused."""
    path = head_path(kind, teacher, data, pretrained)
    if path.exists():
        return torch.load(path, weights_only=False)["rows"]

    def embed(split):
        return teacher_embeddings(
            teacher,
            data,
            split,
            device,
            pretrained=pretrained,
            max_batches=max_batches,
            batch_size=batch_size,
        )

    (h_train, y_train), (h_val, y_val) = embed("train"), embed("val")
    k = data.num_classes
    reads = ["head.logits", "targets.class"]
    ce = nn.CrossEntropyLoss(weight=data.class_weights())
    bal_acc = MulticlassAccuracy(k, average="macro")

    def full_batch(h: Tensor, y: Tensor) -> list[Batch]:
        # The whole split as one batch: each epoch is one full-batch step
        return [{"inputs": {"embed": h}, "targets": {"class": y}}]

    # Full-batch AdamW at lr 1e-3. Its weight decay shrinks weights by lr * wd per
    # step, so wd needs values this large to matter over 2000 steps
    rows, best = [], None
    for wd in weight_decays:
        seed_everything(0)
        head = HEADS[kind](h_train.shape[1], k)
        node = ModelNode("head", head, inputs="inputs.embed", outputs="logits")
        # Early stopping on val macro-F1: the head ends with its best step's weights
        result = Trainer(
            device=CPU, max_epochs=steps, val_every_n_steps=eval_every
        ).fit(
            ExecutionGraph([node]),
            full_batch(h_train, y_train),
            LossComposer([LossBinding("ce", ce, reads)]),
            torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd),
            val_loader=full_batch(h_val, y_val),
            metrics=[
                MetricBinding("macro_f1", MulticlassF1Score(k, average="macro"), reads)
            ],
            callbacks=[RestoreBestWeights("head", "macro_f1")],
        )
        chosen_step = result.best("macro_f1")
        best_f1, best_step = chosen_step["macro_f1"], int(chosen_step["step"])
        head.eval()
        with torch.no_grad():
            p_train, p_val = head(h_train).softmax(1), head(h_val).softmax(1)
        rows.append(
            {
                "head": kind,
                "weight_decay": wd,
                "best_step": best_step,
                "val_bal_acc": bal_acc(p_val, y_val).item(),
                "val_macro_f1": best_f1,
                "train_confidence": p_train.max(1).values.mean().item(),
                "val_confidence": p_val.max(1).values.mean().item(),
            }
        )
        if best is None or best_f1 > best[0]:
            best = (best_f1, head)
    chosen = max(rows, key=lambda r: r["val_macro_f1"])
    for row in rows:
        row["chosen"] = row is chosen
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": kind,
            "dim": h_train.shape[1],
            "num_classes": k,
            "state_dict": best[1].state_dict(),
            "rows": rows,
        },
        path,
    )
    return rows


def load_teacher_head(
    kind: str, teacher: str, data: SplitData, pretrained: bool = True
) -> nn.Module:
    """A head saved by fit_teacher_head, in eval mode."""
    path = head_path(kind, teacher, data, pretrained)
    if not path.exists():
        raise FileNotFoundError(
            f"no {kind} head for {teacher}: run fit_teacher_head (E0c)"
        )
    saved = torch.load(path, weights_only=False)
    head = HEADS[kind](saved["dim"], saved["num_classes"])
    head.load_state_dict(saved["state_dict"])
    return head.eval()


def projector_node(student: str, teacher: str) -> ModelNode:
    """Student embed -> teacher embed space. Training only: never saved or costed."""
    d_s, d_t = STUDENTS[student].dim, TEACHERS[teacher].dim
    return ModelNode(
        "projector",
        nn.Sequential(nn.Linear(d_s, d_t), nn.LayerNorm(d_t)),
        inputs="student.embed",
        outputs="embed",
    )


# -------------------------------------------------------------------------- losses
# KD, DKD, DIST, EmbedCos and EmbedMSE come from lwfm.techniques.distillation (and are
# re-exported here for the notebooks); which values each one reads is our choice.


def loss_reads(loss: nn.Module) -> list[str]:
    """The values a distillation loss reads, in its forward's argument order."""
    if isinstance(loss, EmbedCos | EmbedMSE):
        return EMBED
    needs_labels = isinstance(loss, DKD) or (isinstance(loss, KD) and loss.weighted)
    return [*LOGITS, "targets.class"] if needs_labels else LOGITS


def cache_dir(pretrained: bool) -> Path:
    # Caches from random weights (smoke tests) are kept apart from real ones
    return RUNS / "cache" if pretrained else RUNS / "smoke" / "cache"


@torch.no_grad()
def teacher_embeddings(
    teacher: str,
    data: SplitData,
    split: str,
    device: torch.device,
    *,
    pretrained: bool = True,
    max_batches: int | None = None,
    batch_size: int = 256,
) -> tuple[Tensor, Tensor]:
    """teacher.embed and labels for every un-augmented image of a split, from one
    pass of the frozen teacher. Saved to the cache, so later calls are instant."""
    tag = f"{teacher}_{data.name.replace('/', '_')}_{split}"
    path = cache_dir(pretrained) / f"embed_{tag}.pt"
    if path.exists():
        h, y = torch.load(path, weights_only=True)
        return h.float(), y
    model = teacher_model(teacher, pretrained).eval().to(device)
    hs, ys = [], []
    for batch in islice(data.loader(split, batch_size, augment=False), max_batches):
        hs.append(model(batch["inputs"]["image"].to(device))["embed"].cpu())
        ys.append(batch["targets"]["class"])  # labels stay on the CPU
    h, y = torch.cat(hs), torch.cat(ys)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((h.half(), y), path)  # float16 halves the file
    return h, y


def teacher_embed_stats(
    teacher: str,
    data: SplitData,
    device: torch.device,
    pretrained: bool = True,
    max_batches: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-dimension mean and std of teacher.embed over the un-augmented train set,
    for EmbedMSE."""
    h, _ = teacher_embeddings(
        teacher, data, "train", device, pretrained=pretrained, max_batches=max_batches
    )
    h = h.double()
    return h.mean(0).float(), h.std(0).clamp_min(1e-6).float()


def loss_hparams(loss: nn.Module) -> dict:
    """The plain-number settings of a loss (T, alpha, beta, ...), for the manifest."""
    return {
        k: v
        for k, v in vars(loss).items()
        if not k.startswith("_")
        and k != "training"
        and isinstance(v, int | float | str)
    }


# ------------------------------------------------------------------ metrics, cost


def class_metrics(node: str, num_classes: int) -> list[MetricBinding]:
    """Balanced accuracy (macro recall) is the selection metric: on an imbalanced
    split, top-1 rewards always predicting the majority class. On a balanced split
    it equals top-1. Top-5 is added when there are many classes."""
    reads = [f"{node}.logits", "targets.class"]
    metrics = [
        ("bal_acc", MulticlassAccuracy(num_classes, average="macro")),
        ("macro_f1", MulticlassF1Score(num_classes, average="macro")),
        ("top1", MulticlassAccuracy(num_classes, average="micro")),
    ]
    if num_classes >= 100:
        metrics.append(
            ("top5", MulticlassAccuracy(num_classes, top_k=5, average="micro"))
        )
    return [MetricBinding(name, metric, reads) for name, metric in metrics]


def embed_cos_metric() -> MetricBinding:
    return MetricBinding("embed_cos", CosineSimilarity(reduction="mean"), EMBED)


def confusion_curve(node: str, data: SplitData) -> CurveBinding:
    """The confusion matrix of `node`'s predictions, with the class names: logged as
    the table eval/<split>/confusion and drawn by PlotSink. Per-class recall is each
    row's diagonal over the row's sum."""
    names = data.class_names()
    return CurveBinding(
        "confusion",
        lambda scores, labels: ConfusionMatrix.from_scores(scores, labels, names),
        [f"{node}.logits", "targets.class"],
    )


def one_image(data: SplitData, device: torch.device) -> Tensor:
    """A random image at the dataset's size, for measuring a model's cost."""
    size = data.spec.image_size
    return torch.rand(1, 3, size, size, device=device)


# ------------------------------------------------------------- setups and training


@dataclass(frozen=True)
class TrainConfig:
    """One run's training settings. The budgets below (MAIN, FAST, LONG) are these."""

    epochs: int = 100
    batch_size: int = 256
    eval_batch_size: int = 512
    optimizer: str = "sgd"  # "sgd" (momentum, Nesterov) or "adamw"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    grad_clip: float | None = None  # max gradient norm
    schedule: str = "cosine"  # "cosine" (linear warmup, then cosine) or "one_cycle"
    warmup_epochs: int = 5
    ramp_epochs: int = 20  # DKD weight rises 0 -> 1 over these epochs
    probe_epochs: int = 30  # linear probe, for label-free setups
    max_steps: int | None = None  # per epoch; for smoke tests
    eval_batches: int | None = None  # for smoke tests


# Budgets. A run's final settings are: budget, then the dataset's `train`, then the
# student's `train`, then any `overrides` passed to run() (e.g. SMOKE).
MAIN = TrainConfig()
FAST = TrainConfig(epochs=25, schedule="one_cycle")  # for sweeps: ranks settings only
LONG = TrainConfig(epochs=300)  # KD can keep improving after CE has plateaued
SMOKE = {
    "epochs": 1,
    "batch_size": 8,
    "eval_batch_size": 8,
    "max_steps": 2,
    "eval_batches": 1,
    "probe_epochs": 1,
}

# A local trial run (e.g. on a laptop CPU): the real pipeline, with real weights and
# data, but far less training. Shorter budgets, plus overrides that cap each epoch at
# 30 steps of 64 images and evaluate on 4 x 256 shuffled images. Its numbers show the
# pipeline works and give a rough first look; they are not results.
LOCAL_FAST = replace(FAST, epochs=3)
LOCAL_MAIN = replace(MAIN, epochs=6, warmup_epochs=1)
LOCAL_LONG = replace(LONG, epochs=10, warmup_epochs=1)
LOCAL = {
    "batch_size": 64,
    "max_steps": 30,
    "eval_batch_size": 256,
    "eval_batches": 4,
    "probe_epochs": 3,
    "ramp_epochs": 2,  # DKD must reach full weight within the short budgets
}


@dataclass(frozen=True)
class Setup:
    """What one run trains: a student, optionally a teacher, and its losses."""

    student: str = "resnet18"
    teacher: str = "resnet50"
    # Loss factories, e.g. partial(KD, T=4): a logit loss and/or an embedding loss
    logit: Callable[[], nn.Module] | None = None
    embed: Callable[[], nn.Module] | None = None
    alpha: float | None = None  # (1 - alpha) * CE + alpha * logit loss; None: all 1
    ce: bool = True  # False (with an embed loss): label-free, scored by a probe
    other_shift: float | None = None  # see ToStudentClasses; None: log n
    teacher_head: str | None = None  # "linear"/"mlp": an E0c head, not the logits
    weighted_kd: bool = False  # weight KD by class, as the CE is

    @property
    def needs_teacher(self) -> bool:
        return self.logit is not None or self.embed is not None

    @property
    def needs_probe(self) -> bool:
        return not self.ce

    def build(self, data: SplitData, pretrained: bool = True):
        """The graph nodes and loss bindings this setup trains with."""
        nodes = [student_node(self.student, data)]
        losses = []
        if self.ce:
            ce = nn.CrossEntropyLoss(weight=data.class_weights())
            w = 1.0 if self.alpha is None else round(1 - self.alpha, 6)
            losses.append(LossBinding("ce", ce, CE, weight=w))
        if self.needs_teacher:
            nodes.append(
                teacher_node(
                    self.teacher, data, pretrained, self.other_shift, self.teacher_head
                )
            )
        if self.logit:
            # weighted_kd passes the CE's class weights (only KD accepts them)
            extra = {"weight": data.class_weights()} if self.weighted_kd else {}
            loss = self.logit(**extra)
            w = 1.0 if self.alpha is None else self.alpha
            name = type(loss).__name__
            losses.append(LossBinding(name, loss, loss_reads(loss), weight=w))
        if self.embed:
            nodes.append(projector_node(self.student, self.teacher))
            loss = self.embed()
            losses.append(LossBinding(type(loss).__name__, loss, loss_reads(loss)))
        return nodes, losses

    def train_config(
        self, budget: TrainConfig, data: SplitData, overrides: dict | None = None
    ) -> TrainConfig:
        """The budget with the dataset's, the student's, then any overrides applied."""
        settings = data.spec.train | STUDENTS[self.student].train | (overrides or {})
        return replace(budget, **settings)


def make_optimizer(cfg: TrainConfig, params) -> torch.optim.Optimizer:
    """AdamW, or SGD with Nesterov momentum, as the config says."""
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(
        params,
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=True,
        weight_decay=cfg.weight_decay,
    )


def make_scheduler(cfg: TrainConfig, optimizer, steps_per_epoch: int):
    """One-cycle, or linear warmup then cosine decay. Both step once per batch."""
    total = cfg.epochs * steps_per_epoch
    if cfg.schedule == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, 2 * cfg.lr, total)
    warmup = min(cfg.warmup_epochs * steps_per_epoch, total - 1)
    if warmup == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total - warmup),
        ],
        milestones=[warmup],
    )


def epoch_ramp(target: float, ramp_epochs: int, steps_per_epoch: int):
    """A loss weight rising linearly from 0 to `target` over `ramp_epochs`, held for
    each epoch, as a function of the optimizer step (for LossWeightSchedule)."""
    return lambda step: target * min(1.0, step // steps_per_epoch / ramp_epochs)


def fit(
    graph: ExecutionGraph,
    composer: LossComposer,
    data: SplitData,
    cfg: TrainConfig,
    *,
    seed: int,
    device: torch.device,
    sink: OutputSink,
    metrics: list[MetricBinding],
    curves: list[CurveBinding] = (),
) -> TrainingResult:
    """Trains with lwfm's Trainer: one optimizer step per batch, the LR schedule
    stepped every step, DKD's weight ramped in over cfg.ramp_epochs, and validation
    after every epoch (curves at the last one only)."""
    train = data.loader("train", cfg.batch_size, seed=seed, device=device)
    val = data.loader("val", cfg.eval_batch_size, device=device)
    steps_per_epoch = min(len(train), cfg.max_steps or len(train))
    # Only trainable parameters: the frozen teacher's are left out
    optimizer = make_optimizer(cfg, [p for p in graph.parameters() if p.requires_grad])
    # DKD starts at weight 0 and reaches its full weight after cfg.ramp_epochs
    ramps = [
        LossWeightSchedule(
            b.name, epoch_ramp(b.weight, cfg.ramp_epochs, steps_per_epoch)
        )
        for b in composer.loss_bindings
        if isinstance(b.loss, DKD)
    ]
    trainer = Trainer(
        device=device,
        max_epochs=cfg.epochs,
        gradient_clip_norm=cfg.grad_clip or None,
        log_every_n_steps=1 if cfg.max_steps else 50,
        limit_train_batches=cfg.max_steps,
        limit_val_batches=cfg.eval_batches,
    )
    return trainer.fit(
        graph,
        train,
        composer,
        optimizer,
        val_loader=val if metrics else None,
        metrics=metrics,
        curves=list(curves),
        scheduler=make_scheduler(cfg, optimizer, steps_per_epoch),
        scheduler_interval="step",
        callbacks=ramps,
        output_sink=sink,
    )


def teacher_variant(node: ModelNode) -> dict:
    """How a teacher node makes its logits, for the manifest: a trained head, or the
    mapped ImageNet logits with their "other" shift."""
    last = node.module[-1]
    if isinstance(last, TaskHead):
        head = last.head
        return {"teacher_head": "linear" if isinstance(head, nn.Linear) else "mlp"}
    return {"teacher_head": None, "other_shift": last.shift}


def run(
    name: str,
    setup: Setup,
    data: SplitData,
    budget: TrainConfig,
    *,
    seed: int = 0,
    device: torch.device | None = None,
    pretrained: bool = True,
    overrides: dict | None = None,
    runs: Path = RUNS,
) -> Path:
    """One recorded run in runs/<name>/: trains, tests once, and saves the student
    and its cost. A completed run is skipped, so rerunning an experiment resumes it."""
    device = device or pick_device()
    run_dir = runs / name
    if skip_if_completed(run_dir):
        print(f"skip {run_dir}: already completed")
        return run_dir

    cfg = setup.train_config(budget, data, overrides)
    model = setup.student + (
        f" + teacher {setup.teacher}" if setup.needs_teacher else ""
    )
    print(f"\n>>> train {name}: {model}, {cfg.epochs} epochs, seed {seed}")
    start = time.perf_counter()
    # Build the graph and losses, and check the shapes before training
    seed_everything(seed)  # fixes the student's initial weights
    nodes, losses = setup.build(data, pretrained)
    graph = ExecutionGraph(nodes).to(device)
    shapes = graph.preflight(to_device(next(iter(data.loader("val", 8))), device))
    composer = LossComposer(losses)
    manifest = {  # what the run trained, recorded in manifest.json
        "name": name,
        "seed": seed,
        "student": setup.student,
        "teacher": setup.teacher if setup.needs_teacher else None,
        **(teacher_variant(graph.nodes["teacher"]) if setup.needs_teacher else {}),
        "losses": {
            b.name: {"weight": b.weight, **loss_hparams(b.loss)} for b in losses
        },
        "train": asdict(cfg),
        "data": data.name,
        "shapes": shapes,
    }

    predict = "student"
    # The deployable part, for the final test: no teacher, no projector
    test_graph = ExecutionGraph([n for n in nodes if n.name not in KD_ONLY_NODES])
    metrics = class_metrics(predict, data.num_classes) if setup.ce else []
    if "projector" in graph.nodes:
        metrics.append(embed_cos_metric())
    sinks = LocalFileSink(run_dir, manifest), ConsoleSink(), PlotSink(run_dir / "plots")
    with OutputRouter(*sinks) as sink:
        fit(
            graph,
            composer,
            data,
            cfg,
            seed=seed,
            device=device,
            sink=sink,
            metrics=metrics,
            curves=[confusion_curve(predict, data)] if setup.ce else [],
        )
        student = graph.deployable_model("student")

        if setup.needs_probe:  # label-free: score the embedding with a linear probe
            predict = "probe"
            test_graph, probe = train_probe(
                student, setup, data, cfg, seed, device, sink
            )
            save_weights(run_dir, "probe.pt", probe, sink)

        # Final test: once, after every choice was made on val
        test = data.loader("test", cfg.eval_batch_size, device=device)
        Evaluator().evaluate(
            test_graph,
            islice(test, cfg.eval_batches),
            class_metrics(predict, data.num_classes),
            curve_bindings=[confusion_curve(predict, data)],
            device=device,
            split_name="test",
            output_sink=sink,
        )
        save_weights(run_dir, "student.pt", student, sink)
        for key, value in inference_cost(student, one_image(data, device)).items():
            sink.write(RunEvent("scalar", f"cost/{key}", value))
    print_done(name, run_dir, start)
    return run_dir


def train_probe(
    student: nn.Module,
    setup: Setup,
    data: SplitData,
    cfg: TrainConfig,
    seed: int,
    device: torch.device,
    sink: OutputSink,
) -> tuple[ExecutionGraph, nn.Module]:
    """For a label-free run: trains a linear probe on the frozen student's embedding
    with weighted CE. Returns the graph to test (student, then probe) and the probe."""
    frozen = ModelNode("student", student, "inputs.image", "embed", frozen=True)
    probe = nn.Linear(STUDENTS[setup.student].dim, data.num_classes)
    graph = ExecutionGraph(
        [frozen, ModelNode("probe", probe, "student.embed", "logits")]
    )
    ce = nn.CrossEntropyLoss(weight=data.class_weights())
    probe_cfg = replace(  # a plain schedule, whatever the student trained with
        cfg,
        epochs=cfg.probe_epochs,
        optimizer="sgd",
        lr=0.1,
        schedule="cosine",
        warmup_epochs=0,
        grad_clip=None,
    )
    fit(
        graph,
        LossComposer([LossBinding("ce", ce, ["probe.logits", CE[1]])]),
        data,
        probe_cfg,
        seed=seed,
        device=device,
        sink=sink,
        metrics=class_metrics("probe", data.num_classes),
        curves=[confusion_curve("probe", data)],
    )
    return graph, probe


def save_weights(run_dir: Path, file: str, model: nn.Module, sink: OutputSink) -> None:
    """Saves `model`'s weights as run_dir/<file> and records it as an artifact
    (e.g. student_weights)."""
    name = f"{Path(file).stem}_weights"
    write = partial(torch.save, model.state_dict())
    save_artifact(run_dir, file, write, sink, name=name)


def print_done(name: str, run_dir: Path, start: float) -> None:
    """Prints a finished run's headline numbers and how long it took."""
    summary = json.loads((run_dir / "summary.json").read_text())
    keys = ("eval/val/bal_acc", "eval/test/bal_acc", "eval/test/macro_f1")
    scores = ", ".join(
        f"{k.removeprefix('eval/')} {summary[k]:.4f}" for k in keys if k in summary
    )
    minutes = (time.perf_counter() - start) / 60
    print(f"<<< done {name}: {scores} ({minutes:.1f} min)", flush=True)


def evaluate_teacher(
    name: str,
    teacher: str,
    data: SplitData,
    *,
    device: torch.device | None = None,
    pretrained: bool = True,
    other_shift: float | None = None,
    teacher_head: str | None = None,
    overrides: dict | None = None,
    runs: Path = RUNS,
) -> Path:
    """Scores a teacher alone on val and test (no training), with each split's
    confusion matrix (eval/<split>/confusion) and the teacher's cost at the
    student's input size (so including any upsampling)."""
    device = device or pick_device()
    run_dir = runs / name
    if skip_if_completed(run_dir):
        print(f"skip {run_dir}: already completed")
        return run_dir
    cfg = replace(MAIN, **(overrides or {}))  # only the eval batch settings are used
    print(f"\n>>> evaluate {name}: teacher {teacher} alone, no training")
    start = time.perf_counter()
    node = teacher_node(teacher, data, pretrained, other_shift, teacher_head)
    graph = ExecutionGraph([node]).to(device)
    shapes = graph.preflight(to_device(next(iter(data.loader("val", 8))), device))
    manifest = {
        "name": name,
        "teacher": teacher,
        **teacher_variant(node),
        "data": data.name,
        "shapes": shapes,
    }
    metrics = class_metrics("teacher", data.num_classes)

    sinks = LocalFileSink(run_dir, manifest), ConsoleSink(), PlotSink(run_dir / "plots")
    with OutputRouter(*sinks) as sink:
        for split in ("val", "test"):
            loader = data.loader(split, cfg.eval_batch_size, device=device)
            Evaluator().evaluate(
                graph,
                islice(loader, cfg.eval_batches),
                metrics,
                curve_bindings=[confusion_curve("teacher", data)],
                device=device,
                split_name=split,
                output_sink=sink,
            )
        for key, value in inference_cost(node.module, one_image(data, device)).items():
            sink.write(RunEvent("scalar", f"cost/{key}", value))
    print_done(name, run_dir, start)
    return run_dir


# ------------------------------------------------------------------------- results


def results(pattern: str = "**", runs: Path = RUNS) -> list[dict]:
    """manifest | summary of every completed run matching runs/<pattern>."""
    return completed_runs(runs, pattern)


def best(
    pattern: str, k: int = 1, metric: str = "eval/val/bal_acc", runs: Path = RUNS
) -> list[str]:
    """Names (last path part, seed removed) of the k best runs on a val metric."""
    rows = sorted(results(pattern, runs), key=lambda r: r[metric], reverse=True)
    return [Path(r["run"]).name.rsplit("_s", 1)[0] for r in rows[:k]]


@dataclass
class Experiment:
    """A named set of setups: sweep all on the fast budget, then confirm the best
    `top_k` with 3 seeds on the main budget (and optionally once on the long one).
    Runs go to runs/<name>/<sweep|main|long>/<setup>_s<seed>/."""

    name: str
    setups: dict[str, Setup]
    top_k: int = 1
    long_check: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)
    run_kwargs: dict = field(default_factory=dict)

    def sweep(self, data: SplitData, fast: TrainConfig = FAST) -> None:
        """One run of every setup on the fast budget, to rank them."""
        for key, setup in self.setups.items():
            run(f"{self.name}/sweep/{key}_s0", setup, data, fast, **self.run_kwargs)

    def confirm(
        self, data: SplitData, main: TrainConfig = MAIN, long: TrainConfig = LONG
    ) -> list[str]:
        """Runs the best `top_k` sweep setups on the main budget (every seed) and,
        with long_check, once on the long one. Returns their names, best first."""
        keys = best(f"{self.name}/sweep/*", self.top_k, runs=self.runs)
        for key in keys:
            for seed in self.seeds:
                run(
                    f"{self.name}/main/{key}_s{seed}",
                    self.setups[key],
                    data,
                    main,
                    seed=seed,
                    **self.run_kwargs,
                )
            if self.long_check:
                run(
                    f"{self.name}/long/{key}_s0",
                    self.setups[key],
                    data,
                    long,
                    **self.run_kwargs,
                )
        return keys

    @property
    def runs(self) -> Path:  # the runs folder this experiment writes to
        return self.run_kwargs.get("runs", RUNS)
