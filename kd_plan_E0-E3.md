# KD on Tiny ImageNet: Implementation Plan (E0–E3)

Teacher **ResNet-50** → student **ResNet-18**, on the notebook's Tiny ImageNet split:
10 classes plus "other", at 64 px. Each experiment changes one thing. Build steps 1–6 in
order, then run E0–E3.

The snippets in steps 2–5 were run on CPU against the current `lwfm` code with random
data and untrained weights (torch 2.12, torchvision 0.27, timm 1.0.27). Step 1's
transforms and `teacher_classes` were run on the real data. `run()` and the DINOv2
wrapper are untested: the splits haven't been generated, and the DINOv2 weights aren't
downloaded.

## What exists and what to build

| Piece                                                         | Status                     | Where                                                   |
| ------------------------------------------------------------- | -------------------------- | ------------------------------------------------------- |
| Graph, frozen teacher node (`ModelNode(..., frozen=True)`)    | **Done**                   | `lwfm/core/graph.py`                                    |
| Weighted loss sum, `total = Σ wᵢ·lossᵢ`                       | **Done**                   | `lwfm/core/losses.py`                                   |
| Evaluator (any torchmetrics metric)                           | **Done**                   | `lwfm/evaluation/`                                      |
| Run logging (`manifest.json`, `events.jsonl`, `summary.json`) | **Done**                   | `lwfm/results/`                                         |
| Tiny ImageNet download + class → ImageNet-1k map              | **Done**                   | `download/`, `data/tiny-imagenet/imagenet_mapping.json` |
| 10 + "other" train/val/test manifests                         | **Done** (one bug, step 1) | `examples/imagenet-splits.ipynb`                        |
| Dataset class                                                 | Notebook only              | `ManifestDataset` in the same notebook                  |
| Models, KD losses, Trainer, cost metrics                      | **Missing**                | steps 2–5                                               |

There is **no training loop and no baseline yet**. KD needs no special hook in the loop:
a teacher is just a second graph node, and each loss is one more `LossBinding`.

## The data

Labels `0`–`9` are Tiny ImageNet folders `000`–`009` (goldfish, fire salamander,
bullfrog, tailed frog, alligator, boa, trilobite, scorpion, black widow, tarantula).
Label `10` ("other") is the other 190 folders.

| Split | Source                             | Per kept class | "other" | Total  |
| ----- | ---------------------------------- | -------------- | ------- | ------ |
| train | 70 % of `train/`, per source class | 350            | 66,500  | 70,000 |
| val   | the other 30 %                     | 150            | 28,500  | 30,000 |
| test  | `test/` (official Tiny val)        | 50             | 9,500   | 10,000 |

## Decisions

- **Imbalance.** "Other" is 95 % of every split, so a model that always predicts
  "other" scores 95 % top-1. The two settings below handle it.
  - **Training:** class-weighted CE, with `wᶜ = N / (11·nᶜ)`: 18.2 for each kept class,
    0.096 for "other". KD terms stay unweighted, because the teacher's view of "other"
    images is useful signal too. Each image is still seen once per epoch.
  - **Selection and reporting:** **balanced accuracy** (macro recall; chance and
    always-"other" both score 1/11 = 9.1 %) and macro-F1. Micro top-1 is logged for
    reference only, and top-5 is dropped.
- **Teacher "other" logit.** The teacher predicts 1000 ImageNet classes. Keep the 10 that
  match, and merge the other 990 into one logit with `logsumexp`. The teacher's
  probability for "other" is then exactly 1 − Σ p(kept) from its full softmax (checked).
- **Splits.** Every choice (T, α, best loss, schedule) is made on `val`. `test` is used
  **only** for the final table.
- **Normalisation lives in the model**, not the dataset (PLAN.md §3.5). The dataset
  yields `[0, 1]` images, and each model's `nn.Sequential` starts with its own
  `v2.Normalize`. Both nodes read `inputs.image`, so teacher and student see the same
  augmented view with no extra code.
- **Schedule:** see the next section.

## Training schedule

Every run in one comparison must use the same schedule; otherwise the comparison
measures schedule length, not KD. The options:

| Schedule            | Settings                                                                                  | Good for                                                                                                                 | Drawback                                              |
| ------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **Warmup + cosine** | SGD lr 0.1, momentum 0.9, Nesterov, wd 5e-4, batch 256; 5 warmup epochs, then cosine to 0 | Main runs; almost no tuning                                                                                              | You fix the length up front                           |
| **One-cycle**       | `OneCycleLR`, peak lr 0.2, 25 epochs                                                      | Fast sweeps (E1's grid, E2's first pass)                                                                                 | Rankings at a short budget may not hold at a long one |
| **Step decay**      | lr 0.1, ÷10 at 30/60/90 of 100 epochs, wd 1e-4                                            | Comparing directly with published ImageNet KD results                                                                    | A little worse than cosine for the same budget        |
| **AdamW + cosine**  | lr 1e-3 at batch 256, wd 0.05, 5–10 warmup epochs                                         | ViT students (DeiT-Ti, ViT-S in E5); they train poorly with SGD                                                          | Not comparable with the SGD baseline                  |
| **Long**            | Main schedule at ~3× the epochs                                                           | KD keeps improving long after CE training has plateaued, when teacher and student see the same view (Beyer et al., 2022) | Costly; best configuration only                       |

**Use:**

1. **Sweeps:** one-cycle, 25 epochs, 1 seed. These runs only rank settings.
2. **Main results** (baseline + the best of each experiment, 3 seeds): warmup + cosine,
   100 epochs.
3. **One long check:** the baseline and the best KD run at 300 epochs. If the gap
   between them grows, report both budgets.

There are only 3,500 kept-class training images, so the student overfits them
quickly. Watch `eval/val/bal_acc` per epoch. If it peaks early, shorten the main
schedule for **every** run, rather than picking a best epoch per run.

## Step 1: Data — `examples/imagenet-splits.ipynb`, `lwfm/data/modules.py`

The notebook already writes the right split. Fix one bug before running it:
`splits_file_name`'s second string has no `f` prefix, so the folder is literally named
`...{1 - train_split_fraction}_{SEED}`. Replace the function body with:

```python
# e.g. data/tiny-imagenet/splits/tiny10-0.7_5504132
return ROOT / "splits" / f"tiny10-{train_split_fraction}_{SEED}"
```

Move the notebook's dataset into `lwfm/data/modules.py`, adding a transform argument:

```python
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

# Per-sample augmentation on the CPU. The output is a float image in [0, 1];
# each model normalises its own input (step 2).
TRAIN = v2.Compose(
    [
        v2.RandomResizedCrop(64, scale=(0.25, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
    ]
)
EVAL = v2.ToDtype(torch.float32, scale=True)  # deterministic; images are already 64 px


class ManifestDataset(Dataset):
    """One manifest CSV (sample_id, path, label) -> samples in the lwfm batch layout."""

    def __init__(self, root: Path, manifest: Path, transform: v2.Transform):
        self.root, self.transform = Path(root), transform
        with open(manifest, newline="") as f:
            self.rows = list(csv.DictReader(f))  # every value is a string

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        # RGB mode also turns Tiny ImageNet's few greyscale images into 3 channels
        image = decode_image(str(self.root / row["path"]), mode=ImageReadMode.RGB)
        return {
            "inputs": {"image": self.transform(image)},  # [3, 64, 64] float
            "targets": {"class": int(row["label"])},  # 0-9 kept, 10 other
            "meta": {"sample_id": row["sample_id"]},
        }


def teacher_classes(root: Path, class_to_idx_csv: Path) -> list[int]:
    """ImageNet-1k index of each kept label, in label order: [1, 25, 30, ..., 76].

    Read from the split's own class_to_idx.csv, so it can never disagree with the
    labels the student is trained on. "other" must be the last label.
    """
    mapping = json.loads((root / "imagenet_mapping.json").read_text())
    with open(class_to_idx_csv, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["idx"]))
    assert rows[-1]["class"] == "other", "expected 'other' as the last label"
    return [mapping[r["class"]]["imagenet_1k"] for r in rows[:-1]]


def class_weights(manifest: Path, num_classes: int) -> torch.Tensor:
    """w_c = N / (C * n_c), for nn.CrossEntropyLoss(weight=...)."""
    with open(manifest, newline="") as f:
        labels = torch.tensor([int(r["label"]) for r in csv.DictReader(f)])
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts)  # kept 18.2, other 0.096
```

## Step 2: Models — `lwfm/models/adapters.py`, `lwfm/models/factory.py`

Every model returns `{"logits": [B, C], "embed": [B, D]}`, where `embed` is the pooled
feature before the classifier. The wrappers reuse the model's own layers and leave its
weights unchanged (checked: the logits equal `model(x)`). `ModelNode` then exposes both
outputs as `<node>.logits` and `<node>.embed`.

```python
# adapters.py
class ResNetWithEmbed(nn.Module):
    """Any torchvision ResNet, returning its pooled feature as well as its logits."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        n = self.net
        # Same steps as torchvision's ResNet.forward, stopping before fc
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
        embed = torch.flatten(n.avgpool(x), 1)  # [B, 512] or [B, 2048]
        return {"logits": n.fc(embed), "embed": embed}


class TimmWithEmbed(nn.Module):
    """Any timm classifier (DeiT, ViT, ...), returning the pre-logits feature too."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        tokens = self.net.forward_features(x)  # [B, N, D]
        embed = self.net.forward_head(tokens, pre_logits=True)  # pooled, [B, D]
        return {"logits": self.net.get_classifier()(embed), "embed": embed}


class MergeOther(nn.Module):
    """Teacher logits 1000 -> 11: the 10 kept ImageNet classes, then "other".

    logsumexp merges every remaining class into one logit, so after softmax
    p(other) = 1 - sum(p(kept)), exactly as in the teacher's full 1000-way softmax.
    """

    def __init__(self, keep: list[int], num_teacher_classes: int = 1000):
        super().__init__()
        assert len(set(keep)) == len(keep), "two labels share an ImageNet class"
        rest = torch.ones(num_teacher_classes, dtype=torch.bool)
        rest[keep] = False  # True for the 990 classes that become "other"
        # Buffers move with .to(device); persistent=False keeps them out of state_dict
        self.register_buffer("keep", torch.tensor(keep), persistent=False)
        self.register_buffer("rest", rest, persistent=False)

    def forward(self, out: dict[str, Tensor]) -> dict[str, Tensor]:
        z = out["logits"]
        other = torch.logsumexp(z[:, self.rest], dim=1, keepdim=True)  # [B, 1]
        # Same order as the labels: kept classes 0-9, then other = 10
        return out | {"logits": torch.cat([z[:, self.keep], other], dim=1)}


def projector(d_s: int, d_t: int) -> nn.Module:
    """Maps the student embedding into the teacher's space (512 -> 2048).
    Training only: it is its own graph node, so it's never in the saved student."""
    return nn.Sequential(nn.Linear(d_s, d_t), nn.LayerNorm(d_t))
```

```python
# factory.py
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics


def resnet18_64px(num_classes: int) -> nn.Module:
    """ResNet-18 for 64 px input. The standard stem (7x7 stride-2 conv + max-pool)
    shrinks 64 px to 16 px before layer1, so use a 3x3 stride-1 conv and no pool."""
    net = resnet18(num_classes=num_classes)  # random init: students train from scratch
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    return ResNetWithEmbed(net)


def student_node(num_classes: int) -> ModelNode:
    return ModelNode(
        "student",
        nn.Sequential(v2.Normalize(MEAN, STD), resnet18_64px(num_classes)),
        inputs="inputs.image",  # the augmented [0, 1] batch from the loader
        outputs=["logits", "embed"],  # -> student.logits, student.embed
    )


def teacher_node(keep: list[int]) -> ModelNode:
    net = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )  # downloads once, then cached
    return ModelNode(
        "teacher",
        nn.Sequential(
            # Reads the same augmented image as the student and upsamples it to the
            # teacher's native size. Same result as F.interpolate(bilinear) (checked)
            v2.Resize((224, 224), antialias=True),
            v2.Normalize(MEAN, STD),
            ResNetWithEmbed(net),
            MergeOther(keep),  # teacher.logits: [B, 11]; teacher.embed: [B, 2048]
        ),
        inputs="inputs.image",
        outputs=["logits", "embed"],
        # frozen: requires_grad_(False), eval(), runs under no_grad, and
        # graph.train() can't switch it back to train mode (BatchNorm stays fixed)
        frozen=True,
    )
```

E0 teachers. To swap one in, change the model and the `Normalize` values in `teacher_node`:

| Teacher   | Build                                                                              | Wrapper           | Normalize            | D    | IN-1k top-1 |
| --------- | ---------------------------------------------------------------------------------- | ----------------- | -------------------- | ---- | ----------- |
| ResNet-50 | `resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`                                 | `ResNetWithEmbed` | ImageNet             | 2048 | 80.9        |
| ResNet-34 | `resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)`                                 | `ResNetWithEmbed` | ImageNet             | 512  | 73.3        |
| DeiT-B    | `timm.create_model("deit_base_patch16_224.fb_in1k", pretrained=True)`              | `TimmWithEmbed`   | ImageNet             | 768  | 81.8        |
| ViT-B/16  | `timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)` | `TimmWithEmbed`   | **mean = std = 0.5** | 768  | 85.1        |
| DINOv2-B  | `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")`                    | see below         | ImageNet             | 768  | 84.5        |

For a timm model, read the normalisation from `model.pretrained_cfg["mean"]` and
`["std"]` instead of assuming ImageNet's values. The DINOv2 weights aren't cached
locally, so check these attribute names when you first load it:

```python
class DinoV2WithEmbed(nn.Module):  # net = the dinov2_vitb14_lc hub model
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        f = self.net.backbone.forward_features(x)
        embed = f["x_norm_clstoken"]  # [B, 768]
        # The _lc linear head reads the CLS token concatenated with the mean patch token
        head_in = torch.cat([embed, f["x_norm_patchtokens"].mean(1)], dim=1)
        return {"logits": self.net.linear_head(head_in), "embed": embed}
```

## Step 3: KD losses — `lwfm/techniques/distillation.py`

Notation: `z_s`, `z_t` are student and teacher logits, `p = softmax(z / T)`. Each loss
is an `nn.Module`. Its `LossBinding` names what is passed to `forward`, so swapping a
loss is a config change only.

```python
class KD(nn.Module):
    """Hinton KD: T^2 * KL(p_t || p_s). Binding inputs: LOGITS."""

    def __init__(self, T: float = 4.0):
        super().__init__()
        self.T = T

    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:
        T = self.T
        # kl_div takes log-probabilities for the student and probabilities for the
        # teacher. T^2 keeps gradient size roughly independent of T.
        return T**2 * F.kl_div(
            F.log_softmax(z_s / T, 1), F.softmax(z_t / T, 1), reduction="batchmean"
        )


class DKD(nn.Module):
    """Decoupled KD: T^2 * (alpha * TCKD + beta * NCKD).
    Binding inputs: LOGITS + ["targets.class"]."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, T: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.T = alpha, beta, T

    def forward(self, z_s: Tensor, z_t: Tensor, y: Tensor) -> Tensor:
        T = self.T
        gt = F.one_hot(y, z_s.shape[1]).bool()  # True at each sample's target class
        p_s, p_t = F.softmax(z_s / T, 1), F.softmax(z_t / T, 1)
        # TCKD: KL between the binary distributions [p(target), p(all others)]
        b_s = torch.stack([p_s[gt], 1 - p_s[gt]], 1).clamp_min(1e-12)  # avoid log(0)
        b_t = torch.stack([p_t[gt], 1 - p_t[gt]], 1)
        tckd = F.kl_div(b_s.log(), b_t, reduction="batchmean")
        # NCKD: KL over the non-target classes only. Subtracting 1e4 from the target
        # logit gives it ~0 probability, so the softmax renormalises over the rest
        nckd = F.kl_div(
            F.log_softmax(z_s / T - 1e4 * gt, 1),
            F.softmax(z_t / T - 1e4 * gt, 1),
            reduction="batchmean",
        )
        return T**2 * (self.alpha * tckd + self.beta * nckd)


def pearson(a: Tensor, b: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    """Pearson correlation along `dim` (the cosine of the mean-centred vectors)."""
    a, b = a - a.mean(dim, keepdim=True), b - b.mean(dim, keepdim=True)
    return (a * b).sum(dim) / (a.norm(dim=dim) * b.norm(dim=dim) + eps)


class DIST(nn.Module):
    """Matches correlations rather than exact probabilities. Binding inputs: LOGITS."""

    def __init__(self, beta: float = 1.0, gamma: float = 1.0, T: float = 1.0):
        super().__init__()
        self.beta, self.gamma, self.T = beta, gamma, T

    def forward(self, z_s: Tensor, z_t: Tensor) -> Tensor:
        p_s, p_t = F.softmax(z_s / self.T, 1), F.softmax(z_t / self.T, 1)
        inter = 1 - pearson(p_s, p_t, dim=1).mean()  # each sample, across classes
        intra = 1 - pearson(p_s, p_t, dim=0).mean()  # each class, across the batch
        return self.T**2 * (self.beta * inter + self.gamma * intra)


class EmbedCos(nn.Module):
    """1 - cos(P(h_s), h_t): matches direction only. Binding inputs: EMBED."""

    def forward(self, h_s: Tensor, h_t: Tensor) -> Tensor:
        return 1 - F.cosine_similarity(h_s, h_t, dim=1).mean()


class EmbedMSE(nn.Module):
    """MSE to the standardised teacher embedding. Binding inputs: EMBED."""

    def __init__(self, mean: Tensor, std: Tensor):  # from teacher_embed_stats below
        super().__init__()
        # Buffers: moved by LossComposer.to(device), never trained
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, h_s: Tensor, h_t: Tensor) -> Tensor:
        return F.mse_loss(h_s, (h_t - self.mean) / self.std)


@torch.no_grad()
def teacher_embed_stats(teacher_graph, loader, device) -> tuple[Tensor, Tensor]:
    """Per-dimension mean and std of teacher.embed over the train set.
    Run once, with the EVAL transform, before an EmbedMSE run."""
    total = total_sq = count = 0
    for batch in loader:
        h = teacher_graph(to_device(batch, device))["teacher.embed"].double()
        total, total_sq, count = (
            total + h.sum(0),
            total_sq + (h**2).sum(0),
            count + len(h),
        )
    mean = total / count
    std = (total_sq / count - mean**2).clamp_min(1e-12).sqrt()  # Var = E[h²] - E[h]²
    return mean.float(), std.float()
```

**Sanity checks** (checked on random logits): with identical student and teacher logits,
KD, DKD and DIST are ≈ 0. DIST gives ~1e-3, not exactly 0, because of `eps`. After
`backward()`, every teacher parameter has `grad is None`.

## Step 4: Trainer — `lwfm/training/trainer.py`

This is the PLAN.md §3.6 interface, plus `metric_bindings`, a scheduler stepped per
epoch or per batch (one-cycle needs per batch), and a per-epoch time (needed for the E2
table). It is the notebook's own training loop (`building_blocks.ipynb` §6), moved into
a class.

```python
class Callback(ABC):
    """Code that runs at fixed points in training, e.g. to change a loss weight."""

    @abstractmethod
    def on_epoch_start(self, epoch: int) -> None: ...


class LinearRamp(Callback):
    """Raises a loss weight linearly from 0 to its set value over `epochs` (for DKD).
    Works because LossComposer reads binding.weight afresh on every call."""

    def __init__(self, binding: LossBinding, epochs: int):
        self.binding, self.target, self.epochs = binding, binding.weight, epochs

    def on_epoch_start(self, epoch: int) -> None:
        self.binding.weight = self.target * min(1.0, epoch / self.epochs)


@dataclass(frozen=True)
class TrainingResult:
    global_step: int


class Trainer:
    def fit(
        self,
        graph: ExecutionGraph,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        loss_composer: LossComposer,
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        device: torch.device,
        metric_bindings: Sequence[MetricBinding] = (),  # scored on val each epoch
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        scheduler_interval: Literal["epoch", "step"] = "epoch",
        callbacks: Sequence[Callback] = (),
        output_sink: OutputSink | None = None,
        max_steps: int | None = None,  # stop early, for smoke tests
        log_every: int = 50,  # steps between loss events
    ) -> TrainingResult:
        graph.to(device)
        loss_composer.to(device)  # moves loss buffers: CE class weights, EmbedMSE stats
        step = 0
        for epoch in range(epochs):
            for callback in callbacks:
                callback.on_epoch_start(epoch)
            graph.train()  # frozen nodes (the teacher) stay in eval mode
            start = time.perf_counter()

            for batch in train_loader:
                batch = to_device(batch, device)
                # One forward pass runs every node (student, teacher, projector);
                # the composer reads the outputs it needs by name
                total, terms = loss_composer(batch, graph(batch))
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                if scheduler and scheduler_interval == "step":
                    scheduler.step()
                step += 1

                if output_sink and step % log_every == 0:
                    # terms holds each loss by binding name, already detached
                    for name, value in (terms | {"total": total.detach()}).items():
                        output_sink.write(
                            RunEvent(
                                "scalar",
                                f"train/loss/{name}",
                                value,
                                stage="fit",
                                step=step,
                            )
                        )
                if max_steps and step >= max_steps:
                    return TrainingResult(step)

            if scheduler and scheduler_interval == "epoch":
                scheduler.step()
            if output_sink:
                seconds = time.perf_counter() - start
                output_sink.write(
                    RunEvent(
                        "scalar", "train/epoch_seconds", seconds, stage="fit", step=step
                    )
                )
            if val_loader is not None and metric_bindings:
                # Writes eval/val/<metric>; the evaluator puts the graph back in train mode
                Evaluator().evaluate(
                    graph,
                    val_loader,
                    metric_bindings,
                    device=device,
                    output_sink=output_sink,
                    training_step=step,
                )
        return TrainingResult(step)
```

The two schedules, built per step (`steps = epochs * len(train_loader)`):

```python
def warmup_cosine(optimizer, steps: int, warmup: int):
    # Linear ramp from 1 % of the lr over `warmup` steps, then cosine decay to 0
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps - warmup),
        ],
        milestones=[warmup],
    )


def one_cycle(optimizer, steps: int, peak_lr: float = 0.2):
    # Rises to peak_lr over the first 30 % of steps, then anneals to ~0
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=peak_lr, total_steps=steps
    )
```

## Step 5: Cost metrics — `lwfm/evaluation/benchmarks.py`

Measure `graph.deployable_model("student")` only. It contains no teacher and no
projector, so the "exclude the projector" rule holds automatically.

```python
def inference_cost(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 64, 64),
    device: torch.device = torch.device("cpu"),
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    model = model.eval().to(device)
    x = torch.rand(input_shape, device=device)

    # Size = bytes of the saved weights (not a training checkpoint)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)

    # GPU work is asynchronous, so wait for it before and after each timed call
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    times = []
    with torch.inference_mode():
        for i in range(warmup + repeats):
            sync()
            start = time.perf_counter()
            model(x)
            sync()
            if i >= warmup:  # the first calls are slow (allocation, kernel selection)
                times.append(time.perf_counter() - start)
    times.sort()

    flops = FlopCountAnalysis(model, x).unsupported_ops_warnings(False)
    return {
        "params": sum(p.numel() for p in model.parameters()),
        "size_mb": buffer.getbuffer().nbytes / 1e6,
        "gflops": flops.total() / 1e9,  # fvcore counts one multiply-add as one FLOP
        "latency_ms_median": 1e3 * statistics.median(times),
        "latency_ms_p95": 1e3 * times[int(0.95 * (len(times) - 1))],
    }
```

For reference, the 64 px ResNet-18 with 11 classes has 11.17 M parameters, 44.8 MB and
2.23 GFLOPs.

## Step 6: One run — `scripts/kd_tiny.py`

Every run in E0–E3 is this function. Only `nodes`, `losses` and the schedule change.

```python
ROOT = Path("data/tiny-imagenet")
SPLITS = ROOT / "splits/tiny10-0.7_5504132"
K = 11  # 10 kept classes + other

# The names each loss reads, in argument order
CE = ["student.logits", "targets.class"]
LOGITS = ["student.logits", "teacher.logits"]
EMBED = ["projector.embed", "teacher.embed"]


def class_metrics(node: str) -> list[MetricBinding]:
    """Balanced accuracy (the selection metric), macro-F1, and micro top-1."""
    reads = [f"{node}.logits", "targets.class"]
    return [
        MetricBinding(
            "bal_acc", MulticlassAccuracy(num_classes=K, average="macro"), reads
        ),
        MetricBinding(
            "macro_f1", MulticlassF1Score(num_classes=K, average="macro"), reads
        ),
        MetricBinding(
            "top1", MulticlassAccuracy(num_classes=K, average="micro"), reads
        ),  # ~95 % for always-"other"; for reference only
    ]


def weighted_ce(weight: float = 1.0) -> LossBinding:
    w = class_weights(SPLITS / "train.csv", K)  # kept 18.2, other 0.096
    return LossBinding("ce", nn.CrossEntropyLoss(weight=w), CE, weight=weight)


def run(
    name,
    nodes,
    losses,
    hparams,
    *,
    seed,
    epochs=100,
    schedule="cosine",
    device=torch.device("cuda"),
):
    # --- data: the seed fixes the init, the shuffle order and the augmentation
    torch.manual_seed(seed)
    loader = partial(DataLoader, num_workers=8, pin_memory=True)
    train = loader(
        ManifestDataset(ROOT, SPLITS / "train.csv", TRAIN),
        batch_size=256,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val = loader(ManifestDataset(ROOT, SPLITS / "val.csv", EVAL), batch_size=512)
    test = loader(ManifestDataset(ROOT, SPLITS / "test.csv", EVAL), batch_size=512)

    # --- model: build the graph, then check every output shape on one real batch
    graph = ExecutionGraph(nodes).to(device)
    print(graph.preflight(to_device(next(iter(val)), device)))
    # e.g. {'student.logits': (512, 11), ..., 'teacher.embed': (512, 2048)}

    # --- optimisation: the teacher is frozen, so only the student (and the
    # projector, if present) have requires_grad=True
    composer = LossComposer(losses)
    params = [p for p in graph.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4
    )
    steps = epochs * len(train)
    scheduler = (
        warmup_cosine(optimizer, steps, warmup=5 * len(train))
        if schedule == "cosine"
        else one_cycle(optimizer, steps)
    )
    ramps = [LinearRamp(b, epochs=20) for b in losses if isinstance(b.loss, DKD)]

    # --- validation metrics
    metrics = class_metrics("student")
    if "projector" in graph.nodes:  # E3 check: cos(P(h_s), h_t) should rise
        metrics.append(
            MetricBinding("embed_cos", CosineSimilarity(reduction="mean"), EMBED)
        )

    # --- run: everything below is recorded in runs/<name>/
    run_dir = Path("runs") / name
    manifest = hparams | {
        "seed": seed,
        "epochs": epochs,
        "schedule": schedule,
        "losses": [b.name for b in losses],
        "weights": [b.weight for b in losses],
    }
    with OutputRouter(LocalFileSink(run_dir, manifest), ConsoleSink()) as sink:
        Trainer().fit(
            graph,
            train,
            val,
            composer,
            optimizer,
            epochs=epochs,
            device=device,
            metric_bindings=metrics,
            scheduler=scheduler,
            scheduler_interval="step",
            callbacks=ramps,
            output_sink=sink,
        )
        # Final test, once, after all choices are made on val
        Evaluator().evaluate(
            graph,
            test,
            class_metrics("student"),
            device=device,
            split_name="test",
            output_sink=sink,
        )
        # Save the student alone: no teacher, no projector
        student = graph.deployable_model("student")
        torch.save(student.state_dict(), run_dir / "student.pt")
        sink.write(RunEvent("artifact", "student_weights", run_dir / "student.pt"))
```

`hparams` holds `T`, `α`, `β` and `γ`, so they land in `manifest.json`. The final numbers
go to `summary.json` as `eval/test/bal_acc`, and the per-epoch curve is in
`events.jsonl` under `eval/val/bal_acc`.

## E0: Baselines

1. **Teachers on the split.** Score each teacher in the table on its own, as a one-node
   graph:

   ```python
   keep = teacher_classes(ROOT, SPLITS / "class_to_idx.csv")  # [1, 25, ..., 76]
   teacher = teacher_node(keep)
   graph = ExecutionGraph([teacher]).to(device)
   print(graph.preflight(sample))  # teacher.logits (B, 11), teacher.embed (B, D)
   Evaluator().evaluate(
       graph, test, class_metrics("teacher"), device=device, split_name="test"
   )
   inference_cost(teacher.module, (1, 3, 64, 64), device)  # includes the upsample
   ```

   Also look at the teacher's per-class recall, using
   `MulticlassRecall(num_classes=K, average="none")` directly rather than through the
   `Evaluator`, which needs single numbers. A weak "other" recall means the teacher
   often confuses "other" images with the 10 kept classes, and KD will pass that on.

2. **ResNet-18 baseline, 3 seeds.** This is the reference for every later row:

   ```python
   for seed in (0, 1, 2):
       run(f"baseline_s{seed}", [student_node(K)], [weighted_ce()], {}, seed=seed)
   ```

   Also run seed 0 for 300 epochs (`run("baseline_long_s0", ..., epochs=300)`), for the
   long-schedule check.

**Check:** teacher balanced accuracy is far above 9.1 %. It is optimistic, because Tiny
ImageNet's images come from ImageNet's training set, which the teachers were trained on.

## E1: Logit KD, ResNet-50 → ResNet-18

Sweep with the fast schedule:

```python
for T in (1, 2, 4):
    for alpha in (0.5, 0.9):
        run(
            f"kd_T{T}_a{alpha}_s0",
            [student_node(K), teacher_node(keep)],
            # total = (1 - alpha) * weighted CE + alpha * KD
            [
                weighted_ce(weight=1 - alpha),
                LossBinding("kd", KD(T), LOGITS, weight=alpha),
            ],
            {"T": T, "alpha": alpha},
            seed=0,
            epochs=25,
            schedule="one_cycle",
        )
```

Pick the best setting on `eval/val/bal_acc`. Re-run it with seeds 0–2 on the main
schedule (`epochs=100, schedule="cosine"`), plus seed 0 at 300 epochs.
**Checks:** α = 0 matches the baseline within seed noise, and you have a plot of val
balanced accuracy against epoch for the baseline and the best KD run.

## E2: Swap the logit loss

Keep the E1 graph, and keep `weighted_ce()` at weight 1. Replace only the KD binding:

```python
LossBinding("dkd", DKD(alpha=0.5, beta=0.5), LOGITS + ["targets.class"])  # also beta=8
LossBinding("dist", DIST(beta=1, gamma=1), LOGITS)
```

`run` ramps DKD from 0 to its full weight over 20 epochs. First run 1 seed each on the
fast schedule, then 3 seeds on the main schedule for the best two. With only 11
classes, DKD's non-target term and DIST's per-class term have little to work with, so
gains over KD may be small. **Output:** a table of loss, hyperparameters, test balanced
accuracy and macro-F1 (mean ± std), and `train/epoch_seconds`.

## E3: Embedding KD

Add the projector node. It trains with the student, and it's dropped when saving:

```python
nodes = [
    student_node(K),
    teacher_node(keep),
    # reads student.embed [B, 512], writes projector.embed [B, 2048]
    ModelNode(
        "projector", projector(512, 2048), inputs="student.embed", outputs="embed"
    ),
]
```

| Run | Losses                                                             |
| --- | ------------------------------------------------------------------ |
| (a) | `weighted_ce()` + `LossBinding("cos", EmbedCos(), EMBED)`          |
| (b) | `weighted_ce()` + `LossBinding("mse", EmbedMSE(mean, std), EMBED)` |
| (c) | best E2 loss + EmbedCos                                            |
| (d) | EmbedCos only (no CE), then a linear probe (below)                 |

For (b), get `mean, std` first with
`teacher_embed_stats(ExecutionGraph([teacher_node(keep)]).to(device), loader, device)`,
using a train-set loader with the `EVAL` transform.

For the (d) linear probe, freeze the trained student and train only an 11-way head, for
30 epochs:

```python
# Load the saved student; this also checks it loads without any projector weights
student = nn.Sequential(v2.Normalize(MEAN, STD), resnet18_64px(K))
student.load_state_dict(torch.load("runs/e3d_s0/student.pt", weights_only=True))

nodes = [
    ModelNode(
        "student", student, inputs="inputs.image", outputs="embed", frozen=True
    ),  # the embedding is fixed
    ModelNode("probe", nn.Linear(512, K), inputs="student.embed", outputs="logits"),
]
w = class_weights(SPLITS / "train.csv", K)
losses = [
    LossBinding("ce", nn.CrossEntropyLoss(weight=w), ["probe.logits", "targets.class"])
]
# Train with Trainer as in run(), scoring class_metrics("probe")
```

First run 1 seed each on the fast schedule, then 3 seeds on the main schedule for the
best. **Checks:**

- `eval/val/embed_cos` rises during training.
- `student.pt` loads strictly into a plain `resnet18_64px`, with no projector keys
  (checked).
- Cost is identical across all ResNet-18 rows, since the model is the same.

## Final output

```python
# One row per run: its settings (manifest.json) and final numbers (summary.json)
rows = [
    json.loads((d / "manifest.json").read_text())
    | json.loads((d / "summary.json").read_text())
    | {"run": d.name}
    for d in Path("runs").iterdir()
    if (d / "summary.json").exists()
]
table = pd.DataFrame(rows)
table["config"] = table["run"].str.replace(r"_s\d+$", "", regex=True)  # drop the seed
table.groupby(["config", "epochs"])[["eval/test/bal_acc", "eval/test/macro_f1"]].agg(
    ["mean", "std", "count"]
)
```

Add one plot of val balanced accuracy against epoch, from the `eval/val/bal_acc` lines in
each `events.jsonl`, and one paragraph per experiment describing any deviations from
this plan.

## Later experiments (do not build now)

- **E4/E5 (other teachers and students):** only `teacher_node`/`student_node` change.
  Planned students are MobileNetV2 (D 1280), DeiT-Ti (`img_size=64, patch_size=8`,
  D 192, AdamW schedule) and ViT-S.
- **E6:** E3(d) with a DINOv2-B/L teacher, using the embedding only.
- **E7 (cached teacher):** a node that reads `meta.sample_id` and returns the stored
  `logits` and `embed`. That is about 1.5 MB of logits and 287 MB of embeddings for
  the 70k training images in fp16. Caching loses the shared augmented view, so compare
  it with the live teacher.
- **E8 (spatial KD, for segmentation):** reshape ViT tokens from B×N×D to B×D×√N×√N, add
  a 1×1 conv adapter node, and resize to the teacher's grid.
- **E9 (ImageNet-1k):** R34 → R18 as a sanity check (plain KD ≈ 70.7 %), plus the best
  2–3 configurations. All 1000 classes match, so drop `MergeOther`.
- **Pathology teachers** (no class head, so only EmbedCos and EmbedMSE apply), all via
  `timm` from gated HF repos: UNI `hf-hub:MahmoodLab/uni` (D 1024), H-optimus-0
  `hf-hub:bioptimus/H-optimus-0` (D 1536), Virchow2 `hf-hub:paige-ai/Virchow2` (D 2560:
  CLS token concatenated with the mean patch token). The kept-classes + "other" setup
  here matches the class imbalance expected in WSIs.

## Suggested library developments, most useful first

Steps 1–6 still leave about 150 lines of setup per experiment family, and most of it
repeats. Each item below removes a chunk of that repetition while keeping every part
swappable. The APIs are proposals: names follow PLAN.md where it already names them,
and nothing here exists yet.

### 1. `Trainer` (PLAN.md §3.6)

**What it removes:** the hand-written loop: device moves, train mode, backward,
scheduler, logging and validation. The biggest saving, and everything below builds on it.

```python
Trainer().fit(
    graph,
    train,
    val,
    composer,
    optimizer,
    epochs=100,
    device=device,
    metric_bindings=class_metrics("student"),
    scheduler=scheduler,
    scheduler_interval="step",
    callbacks=[LinearRamp(dkd, epochs=20)],
    output_sink=sink,
)
```

Promote step 4's version as it is, with tests: the overfit check, the frozen teacher
staying in eval, `max_steps`, both scheduler intervals, and the baseline reproduced
with `losses=[CE]`.

### 2. Model presets with a fixed output contract (`lwfm/models/factory.py`)

**What it removes:** wrapper classes, stem surgery, normalisation constants, upsampling,
the "other" merge and `frozen=True`, written again for every teacher. One call builds a
ready node, and every preset returns `logits` and `embed`.

```python
student = models.student("resnet18", num_classes=11, image_size=64)  # 3x3 stem
teacher = models.teacher(
    "timm:vit_base_patch16_224.augreg2_in21k_ft_in1k",  # or "torchvision:resnet50"
    input_size=224,  # adds the Resize; Normalize is read from the weights' config
    classes=keep,
    other=True,  # adds MergeOther; classes=None keeps all 1000
)  # always frozen, always outputs ["logits", "embed"]
```

Swapping a teacher in E4/E5 becomes a change to one string.

### 3. `DataModule` for manifests (PLAN.md §3.5)

**What it removes:** three `DataLoader` calls, seeding the generator, transform choice,
class weights and the teacher class lookup.

```python
data = ImageFolderDataModule(
    ROOT, SPLITS, train_transform=TRAIN, eval_transform=EVAL, batch_size=256, seed=seed
)
data.train_dataloader()
data.val_dataloader()
data.test_dataloader()
data.num_classes  # 11, from class_to_idx.csv
data.class_weights()  # for imbalanced splits like this one
data.teacher_classes()  # ImageNet indices, always matching the labels
```

### 4. `Experiment`: one call per run

**What it removes:** the whole of step 6. Seeding, the run directory, sinks, the
manifest, preflight, the optimizer over trainable parameters only, the final test
evaluation and saving the deployable student all become defaults.

```python
Experiment(
    name="kd_T4_a0.9",
    data=data,
    nodes=[student, teacher],
    losses=[
        LossBinding(
            "ce", nn.CrossEntropyLoss(weight=data.class_weights()), CE, weight=0.1
        ),
        LossBinding("kd", KD(T=4), LOGITS, weight=0.9),
    ],
    metrics=class_metrics("student"),
    optimizer=partial(torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=5e-4),
    schedule=WarmupCosine(warmup_epochs=5),  # or OneCycle(peak_lr=0.2)
    epochs=100,
    deploy="student",  # what is saved and benchmarked
).run(seeds=(0, 1, 2))  # runs/kd_T4_a0.9_s{0,1,2}/
```

The hyperparameters go into `manifest.json` automatically, read from each loss module's
constructor arguments.

### 5. Reusable KD bundles (`lwfm/techniques/`)

**What it removes:** choosing names and weights for the bindings, and adding projector
nodes by hand. A bundle returns the nodes and bindings it needs, so E1–E3 become one
line each.

```python
kd = LogitKD(KD(T=4), alpha=0.9)  # adds (1-α)·CE + α·KD
emb = EmbedKD(EmbedCos(), student_dim=512, teacher_dim=2048)  # adds projector node
Experiment(..., nodes=[student, teacher], techniques=[kd, emb]).run()
```

### 6. Sweeps and results tables (`lwfm/analysis/`)

**What it removes:** the nested loops, the `rows = [...]` merge and the plotting code.

```python
sweep(
    base_experiment,
    T=[1, 2, 4],
    alpha=[0.5, 0.9],
    seeds=[0],
    epochs=25,
    schedule=OneCycle(),
)  # 6 fast runs
table = compare(
    "runs/", group_by=["losses", "T", "alpha"], metric="eval/test/bal_acc"
)  # mean ± std per config
plot_curves("runs/", metric="eval/val/bal_acc", runs=["baseline", "kd_T4_a0.9"])
```

### 7. `BenchmarkRunner` (PLAN.md §3.8)

**What it removes:** remembering which module to time and how. It always measures
`graph.deployable_model(deploy)`, never the teacher, and `Experiment` calls it at the end
of each run, so cost lands in `summary.json` next to accuracy.

```python
BenchmarkRunner([ParameterCount(), ModelSize(), ForwardLatency(batch_size=1)]).run(
    graph.deployable_model("student"), input_shape=(1, 3, 64, 64), device=device
)
```

### 8. Cached teacher node (needed for E7 and for pathology at scale)

**What it removes:** re-running a large frozen teacher every epoch. One pass writes the
outputs, keyed by `sample_id`, and a node with the same output names replaces the live
teacher, so the losses don't change.

```python
cache_teacher(teacher, data.train_dataloader(augment=False), "cache/r50_tiny10.pt")
teacher = CachedTeacher("cache/r50_tiny10.pt", outputs=["logits", "embed"])
```
