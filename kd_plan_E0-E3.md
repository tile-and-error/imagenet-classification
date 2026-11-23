# KD: Plan and Implementation (E0–E3)

A frozen teacher (**ResNet-50**, ImageNet-1k) → two students trained from scratch,
**ResNet-18** (CNN) and **ViT-Tiny** (transformer). The first trial uses Tiny ImageNet
at 64 px: 10 named classes plus one "other" class, built from 40 further classes (or all
190 to confirm results). The same code runs on ImageNet-1k by changing two lines. Each
experiment changes one thing.

**Implemented**, using only `lwfm` and torch/torchvision/torchmetrics:

| File | What it holds |
|---|---|
| [`experiments/kd/kd_e0_e3.ipynb`](experiments/kd/kd_e0_e3.ipynb) | E0–E3, results table and curves, with explanations. Three modes: `SMOKE` (wiring check, minutes), `LOCAL_TRIAL` (default without a CUDA GPU: the real pipeline with short budgets, ~a day on a CPU) and full runs (default on a CUDA GPU) |
| [`experiments/kd/parts.py`](experiments/kd/parts.py) | Reusable parts: dataset and model registries, `make_split`, KD losses, metrics, cost, training loop, `Setup`, `run`, `Experiment` |

## Summary: what was added and how KD works now

**Added** (nothing in `lwfm` itself changed):

- `experiments/kd/parts.py`: every reusable part, in one file.
- `experiments/kd/kd_e0_e3.ipynb`: the experiments, with explanations.
- `runs/` in `.gitignore`: run outputs (checkpoints, logs) stay out of git.
- `download/imagenet_mapping.json` (committed): the Tiny folder → ImageNet-1k class
  map that the teacher needs, so no setup step is needed for it.

**How one KD run works**, from the bottom up:

1. **Data.** `SplitData(DATASET, SPLIT)` reads a split folder of manifest CSVs. It
   gives loaders whose batches are lwfm dicts: `{"inputs": {"image"}, "targets":
   {"class"}, "meta": {"sample_id"}}`. Images are augmented on the CPU and returned
   in [0, 1]; the dataset's entry in `DATASETS` chooses the transforms.
2. **Graph.** `Setup(...).build(data)` returns lwfm `ModelNode`s:
   - `student`: from `STUDENTS`, built for the image size.
   - `teacher`: frozen. It reads the **same** `inputs.image`, resizes it to 224 px,
     normalises it, and maps its 1000 logits to the student's classes. Named classes
     keep their logits. The classes behind "other" are merged into one logit, minus a
     prior correction `other_shift` (see *The teacher's "other" logit*). All other
     ImageNet classes are dropped.
   - `projector`: only for embedding losses.

   Every model outputs `logits` and `embed`. `ExecutionGraph` runs them in order in one
   forward pass.
3. **Losses.** `Setup` also returns `LossBinding`s: weighted CE, plus a KD loss (`KD`,
   `DKD`, `DIST`, `EmbedCos` or `EmbedMSE`). Each loss names the outputs it reads in
   `reads`. lwfm's `LossComposer` sums them with their weights.
4. **Training.** `fit()` is a plain torch loop: forward the graph, compose the loss,
   backward, step.
   - Only the student and projector have gradients, because the teacher is frozen.
   - The optimiser (SGD or AdamW) and schedule (warmup + cosine, or one-cycle) come
     from the layered `TrainConfig`: budget → dataset → student → overrides.
   - Every epoch, lwfm's `Evaluator` scores the student on val.
5. **A run.** `run(name, setup, data, budget)` does steps 2–4 inside lwfm's
   `OutputRouter`. It then:
   - evaluates once on test
   - saves `graph.deployable_model("student")`, which has no teacher and no projector
   - measures the student's cost
   - writes `runs/<data>/<name>/` (`manifest.json`, `events.jsonl`, `summary.json`,
     `student.pt`)

   A completed run is skipped next time, so the notebook resumes where it stopped.
6. **An experiment.** `Experiment(name, {key: Setup})` holds several setups:
   - `.sweep()` runs each one on the `FAST` budget.
   - `.confirm()` takes the best on val balanced accuracy and reruns it with 3 seeds on
     `MAIN`, plus once on `LONG` if asked.
7. **Results.** `results()` / `best()` read the run folders. The notebook turns them into
   a mean ± std table (`summary.csv`) and val curves.

**In the notebook,** E0a chooses each teacher's `other_shift` on val and scores the
teachers (inference only). E0b trains the CE-only baselines. E0c trains linear and MLP
heads on the frozen teacher, compares every teacher variant (first alone, then by the
student it produces), and picks the one used for E1–E3.

**Run folders** (under `runs/<dataset>/<split>/`):

- `teachers/<teacher>/<variant>/`: a teacher scored alone
- `<teacher variant>/<experiment>/<student>/<stage>/<setup>_s<seed>/`: a training run

The teacher variant is `no-teacher`, or e.g. `resnet50-linear`. Completed runs are
skipped by name, so naming the teacher keeps runs made with different teachers from
being reused by mistake. E1–E3 are each one `Experiment` per student. A different student,
teacher, loss or dataset is one argument, or one registry entry; nothing else changes.

## The teacher's "other" logit, and the split size

**Problem.** The zero-shot ResNet-50, on 64 px images upsampled to 224 px, called almost
every named-class image "other". On the Tiny test set:

| | Value |
|---|---|
| 1000-way top-1 on named-class images | **25.8 %** (top-5 51.8 %), against 80.9 % at native 224 px |
| Mean merged p("other") on named-class images, all 990 non-named classes merged | 0.87 |
| Teacher balanced accuracy over the 11 classes | 0.149 |

It had two causes:

1. **Upsampling.** ImageNet models lose most of their accuracy on 64 px images.
   *Still open*; see *Remaining issue* below.
2. **Merging many classes into "other".** A softmax treats every class as equally
   likely in advance, so merging $n$ classes makes "other" $n$ times likelier than a
   named class before the image is seen. When the teacher is unsure, that prior wins.
   *Fixed*, as follows.

**Fix 1: a prior correction (`ToStudentClasses`).**
`z_other = logsumexp(z[classes behind "other"]) − b`. This is the standard
label-shift / logit-adjustment correction.

- `b = log n` gives "other" the same prior as each named class. That maximises
  balanced accuracy, but it pulls "other" images towards the named classes.
- So E0a picks `b` **on val by macro-F1** (`tune_other_shift`), once per teacher, from a
  single teacher pass. Every KD run then uses it.

**Fix 2: merge only the split's own classes, and limit how many there are
(`make_split(num_other=...)`).**

- The teacher's softmax is restricted to the named classes plus the ImageNet classes
  that actually make up "other"; the rest are dropped.
- With fewer "other" classes there is less to correct, and the corrected teacher
  pulls fewer "other" images towards the named classes.

Measured on val (ResNet-50; the shift is the one that maximises macro-F1):

| Split | No correction: bal. acc / macro-F1 | Tuned shift | bal. acc / macro-F1 | p(other) on "other" images |
|---|---|---|---|---|
| **10 + 40** (default) | 0.51 / 0.58 | 2.3 | **0.69 / 0.69** | 0.65 |
| 10 + 190 | 0.37 / 0.44 | 2.6 | 0.61 / 0.61 | 0.77 |

At `b = log n`, balanced accuracy is 0.71 and 0.69, but macro-F1 falls to 0.57 and 0.31.
That is why `b` is tuned rather than fixed.

**Which split to use:**

| | 10 + 40 (default) | 10 + 190 |
|---|---|---|
| Teacher (tuned) | cleaner: macro-F1 0.69 | macro-F1 0.61 |
| Train images | 17,500: runs ~4× faster | 70,000 |
| "Other" share | 80 % | 95 %, closer to the imbalance expected in WSIs |
| Variety in "other" | 40 classes (test contains only those) | 190 classes |

Iterate and sweep on 10 + 40; confirm the final results on 10 + 190 by changing `SPLIT`.
The 40 classes are drawn with a fixed seed from those with an ImageNet-1k match, so the
teacher has a logit for every "other" image. They are listed in the split's
`split.csv`. Pathology foundation models have no class head, so fix 2 doesn't carry
over there. The prior correction does, wherever a teacher has a class head.

**Upsampling (cause 1)** is handled by E0c below: heads trained on the frozen teacher
reach ~0.81 balanced accuracy, against 0.69 for the corrected logits. On ImageNet-1k
neither cause applies: images are native 224 px, and there is no "other" class.

## E0c: heads trained on the frozen teacher

**Why.** E0a's teacher maps its ImageNet logits to our classes and corrects the
"other" prior. A head trained on the frozen teacher's embedding, with our labels, does
better on two counts:

- It learns "other" as an ordinary class.
- It learns what upsampled 64 px images look like in the teacher's feature space.

Pathology foundation models need the same step, since they have no class head.

**How (`fit_teacher_head`):**

1. One pass of the frozen backbone over the un-augmented train and val images. The
   embeddings are cached in `runs/cache/`.
2. A head is trained on the cache: class-weighted CE, full-batch AdamW, weight decay
   ∈ {0.01, 0.1, 1}, up to 2000 steps with early stopping. All choices are made by
   val macro-F1.
3. The chosen head is saved. `Setup(teacher_head="linear" | "mlp")` uses it; the
   teacher's `embed` (and so E3) is unchanged.

**Results** (ResNet-50, 10 + 40 split, val):

| Teacher variant | bal. acc | macro-F1 | Confidence: train / val |
|---|---|---|---|
| Mapped ImageNet logits, no correction | 0.51 | 0.58 | — |
| Mapped ImageNet logits, tuned shift (E0a) | 0.69 | 0.69 | — |
| **Linear head** | 0.81 | 0.79 | 0.98 / 0.96 |
| **MLP head** | 0.81 | **0.82** | **1.00** / 0.99 |

- **Both heads beat the corrected logits by a wide margin**, about +0.12 balanced
  accuracy.
- **The MLP has slightly better macro-F1, but is near-certain on train images** (mean
  top probability 1.00). Its soft targets on those images are then close to one-hot,
  which is what weakens KD. The linear head is a little less certain.

That's why the choice isn't made at teacher level. The notebook runs each variant with
the same KD setting (T = 4, α = 0.9, `FAST` budget, both students), and the variant
whose students score best on val teaches E1–E3.

## How it fits the library

| Need | Provided by |
|---|---|
| Student + frozen teacher + projector reading the same image | `lwfm` `ExecutionGraph` of `ModelNode`s; the teacher has `frozen=True` |
| Weighted sum of losses, each reading named outputs | `lwfm` `LossBinding` + `LossComposer`; each KD loss lists its inputs in `reads` |
| Validation and test metrics | `lwfm` `Evaluator` + `MetricBinding` (torchmetrics) |
| Run records (`manifest.json`, `events.jsonl`, `summary.json`) | `lwfm` `LocalFileSink` + `ConsoleSink` via `OutputRouter` |
| Saved student without teacher or projector | `graph.deployable_model("student")` |
| Training loop, optimisers, schedules, DKD ramp | `parts.fit`, plain torch: `lwfm` has no `Trainer` yet |
| Data | `parts.DATASETS`, `SplitData`, `make_split`, over manifest CSVs |
| Cost (parameters, size, FLOPs, latency) | `parts.inference_cost`; FLOPs from `torch.utils.flop_counter` |

## Swapping parts

| To change | Edit |
|---|---|
| Dataset | `DATASET, SPLIT` in the notebook. Each entry in `parts.DATASETS` holds its root, image size, transforms, folder → ImageNet-index map, class names and any training settings it needs |
| Student | `STUDENTS_TO_RUN`. A new student is one entry in `parts.STUDENTS`: `build(num_classes, image_size)`, its embed size, and any training settings it needs |
| Teacher | `TEACHER`, or `Setup(teacher=...)`. A new teacher is one entry in `parts.TEACHERS` |
| Loss | an `nn.Module` with a `reads` list, passed as `Setup(logit=...)` or `Setup(embed=...)` |
| Experiment | a dict of named `Setup`s in an `Experiment` |

```python
Setup("resnet18")                                         # baseline: weighted CE only
Setup("resnet18", logit=partial(KD, T=4), alpha=0.9)      # 0.1·CE + 0.9·KD
Setup("resnet18", logit=partial(KD, T=4), alpha=0.9,
      weighted_kd=True)                                   # ... KD weighted by class
Setup("vit_ti", logit=partial(DKD, beta=8.0))             # CE + DKD (ramped)
Setup("resnet18", embed=EmbedCos)                         # CE + cosine to teacher embed
Setup("vit_ti", embed=EmbedCos, ce=False)                 # label-free, then a probe
Setup("resnet18", teacher="vit_b16", logit=KD)            # another teacher
```

**Moving to ImageNet-1k:** run `download/download_data.py imagenet-1k data/imagenet-1k`,
then `make_split("imagenet-1k", train_fraction=0.96)`, then set
`DATASET, SPLIT = "imagenet-1k", "imagenet-1k-all-96-4_5504132"`. These then change
automatically:

- images are 224 px, with ImageNet augmentation (eval: resize 256, centre-crop 224)
- the ResNet-18 stem is standard, and ViT-Tiny uses patch 16
- the teacher's 1000 logits are used unchanged, with no "other" class
- class weights are 1
- top-5 is added to the metrics
- weight decay is 1e-4

A run's final settings are layered: budget → dataset → student → smoke overrides. Each
run's `manifest.json` records the result.

## Data (Tiny trial)

Both splits use 70 % of each class's `train/` images for training and 30 % for val. Test
is `test/`, the official Tiny val set.

| Split | Folder (in `data/tiny-imagenet/splits/`) | Per named class (train/val/test) | "other" (train/val/test) |
|---|---|---|---|
| 10 + 40 | `tiny-imagenet-10+40other-70-30_5504132` (`make_split`) | 350 / 150 / 50 | 14,000 / 6,000 / 2,000 |
| 10 + 190 | `tiny-70-30_5504132` (`examples/imagenet-splits.ipynb`) | 350 / 150 / 50 | 66,500 / 28,500 / 9,500 |

The class → ImageNet-1k map is `download/imagenet_mapping.json`, read by
`parts.tiny_mapping()`.

## Decisions

- **Two students.** ResNet-18 (11.2 M parameters) and ViT-Tiny (DeiT-Tiny shape,
  5.5 M). Each runs every experiment, with its own best settings. The ViT is expected
  to trail from scratch on ~70k images, since it lacks a CNN's built-in locality, and
  DeiT showed that ViTs gain the most from a CNN teacher.
- **Imbalance.** "Other" is 80 % (10 + 40) or 95 % (10 + 190) of the data.
  - Training uses class-weighted CE, `wᶜ = N / (C·nᶜ)`.
  - KD terms are unweighted by default. Plain KD is then dominated by "other" (80 % of
    the images and of the teacher's probability mass): in the debug run, KD students at
    α = 0.9 first learned to always predict "other", while their weighted CE rose.
    `Setup(weighted_kd=True)` weights each image's KL by its class weight,
    `T²·Σᵢ w_yᵢ KLᵢ / Σᵢ w_yᵢ`, so each class gets an equal share, as in the CE. E1 tests
    it against plain KD at every (T, α). Only `KD` supports it so far.
  - Runs are chosen and reported on **balanced accuracy** (macro recall; chance is
    9.1 %) and macro-F1. Top-5 is added for ≥ 100 classes.
- **Teacher "other" logit.** Prior-corrected, with the shift chosen on val by
  macro-F1 (see above).
- **Same view.** Both nodes read `inputs.image`. The teacher resizes it to 224 px and
  normalises it itself, so teacher and student always see the same crop and flip.
- **Splits.** Every choice is made on `val`. `test` is evaluated once per run, at the
  end.
- **Validation cost.** Per-epoch metrics run on a student-only graph. `embed_cos` needs
  the teacher, so it runs every 10 epochs.
- **Resumable runs.** A completed run folder is skipped. An interrupted or failed one
  is deleted and run again, since a run can't be resumed partway. A folder still marked
  `running` raises an error, in case another notebook is using it.
- **Device.** `pick_device()` picks CUDA, then MPS, then the CPU. A local, uncommitted
  file `.kd-device` (e.g. `cpu`) overrides it.
  - **⚠ To fix:** lwfm's `to_device` copies to MPS with `non_blocking=True`. When the
    CPU batch is freed before the copy runs, the data arrives corrupted, which breaks
    both training and evaluation on a Mac.
  - Until it's fixed, `.kd-device` holds `cpu` on the Mac. CUDA is believed
    unaffected.

## Training schedule

| Budget | Settings | Used for |
|---|---|---|
| `FAST` | 25 epochs, one-cycle (peak 2× lr) | Sweeps: ranking settings, 1 seed |
| `MAIN` | 100 epochs, 5 warmup epochs then cosine | Reported results, 3 seeds |
| `LONG` | 300 epochs, same shape | Checks whether KD's gain grows with training time |

**Local trial** (`LOCAL_TRIAL`, on automatically without a CUDA GPU):

- The budgets become 3 / 6 / 10 epochs, each capped at 30 steps of 64 images.
- Evaluation uses 1,024 images, drawn in a fixed shuffled order.
- One seed; the DKD ramp takes 2 epochs.
- Runs go to `runs/local/`, so they never mix with full runs.

It shows that the real pipeline works end to end, and gives a first look. It does not
give reportable results. Measured on an M4 CPU, a full-size epoch takes ~21 min
(baseline) to ~59 min (KD with a ResNet-50 teacher), which is why the full budgets need
a GPU.

| Student | Optimiser |
|---|---|
| `resnet18` | SGD, lr 0.1, momentum 0.9 (Nesterov), wd 5e-4 (1e-4 on ImageNet-1k), batch 256 |
| `vit_ti` | AdamW, lr 1e-3, wd 0.05, 10 warmup epochs, gradient clipping at 1.0 |

On Tiny, watch `eval/val/bal_acc`. If it peaks early, shorten the schedule for
**every** run of that student, rather than picking a best epoch per run. ViT-Tiny
would also benefit from stronger augmentation (RandAugment, mixup/cutmix), which isn't
included yet.

## Teachers (E0a)

All come from torchvision, except DINOv2 (`torch.hub`). timm isn't torch, so
torchvision's two ViT-B/16 weights replace DeiT-B and the timm ViT.

| Name | Weights | IN-1k top-1 | D |
|---|---|---|---|
| `resnet50` | `ResNet50_Weights.IMAGENET1K_V2` | 80.9 | 2048 |
| `resnet34` | `ResNet34_Weights.IMAGENET1K_V1` | 73.3 | 512 |
| `vit_b16` | `ViT_B_16_Weights.IMAGENET1K_V1` (supervised, like DeiT-B) | 81.1 | 768 |
| `vit_b16_swag` | `ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1` (weakly supervised) | 81.9 | 768 |
| `dinov2_b14` | `torch.hub` `facebookresearch/dinov2`, `dinov2_vitb14_lc` | 84.5 | 768 |

The DINOv2 wrapper's attribute names haven't been checked against the real model yet
(smoke mode skips it).

## Experiments (run per student)

| | Runs | Selection |
|---|---|---|
| **E0a** | Each teacher alone: choose its `other_shift` on val, then val + test metrics, per-class test recall, teacher cost. **Inference only** | shift: best val macro-F1 |
| **E0b** | Baseline, weighted CE, no teacher: `FAST` ×1, `MAIN` ×3 seeds, `LONG` ×1 | — |
| **E0c** | Linear and MLP heads on the frozen teacher; each teacher variant (mapped logits, linear, mlp) scored alone, then used for KD (T = 4, α = 0.9, `FAST`) with each student | variant: best mean student val bal. acc |
| **E1** | KD sweep, T ∈ {1, 2, 4} × α ∈ {0.5, 0.9} × {plain, class-weighted KD}, total = (1−α)·CE + α·KD | best on val → `MAIN` ×3 + `LONG` ×1 |
| **E2** | Best E1 KD vs DKD (β = 0.5, β = 8; ramp over 20 epochs) vs DIST, each + CE | top 2 → `MAIN` ×3 |
| **E3** | CE + EmbedCos; CE + EmbedMSE; best E2 + EmbedCos; EmbedCos only + linear probe | best → `MAIN` ×3 |

**Checks** (in the notebook):

- *Sanity:* with no shift, the class mapping equals a softmax over just the split's
  classes, with "other" as the sum of its classes.
  Identical logits give KD, DKD and DIST ≈ 0. The teacher gets no gradients and the
  student does.
- *E3:* `eval/val/embed_cos` rises, and `student.pt` loads strictly, with no projector
  weights.
- *All:* cost is identical within each student.

**Verified:**

- The notebook (with `SMOKE = True` swapped in on a copy) runs end to end in 114 s on
  the CPU, E0c included, on the 10 + 40 split.
- The heads were trained on real ResNet-50 embeddings (the table in *E0c*).
- `make_split`, and full runs of both students with every setup type, work on a fake
  224 px dataset with ImageNet-1k settings.
- The real ResNet-50 has been evaluated on the Tiny test set, and its shift tuned on
  both splits' val sets (the tables above).

## Later experiments (not built)

- **E4/E5:** more teachers and students: add registry entries. Planned students are
  MobileNetV2 and ViT-S.
- **E6:** `Setup(student, teacher="dinov2_b14", embed=EmbedCos, ce=False)`.
- **E7:** cached teacher: a node reading `meta.sample_id` and returning the stored
  outputs. Caching loses the shared augmented view, so compare it with the live
  teacher.
- **E8:** spatial KD for segmentation: ViT tokens → grid, plus a 1×1 conv adapter node.
- **E9:** ImageNet-1k: switch `DATASET`/`SPLIT`; R34 → R18 plain KD should reach about
  70.7 %.
- **Pathology teachers** (UNI, H-optimus-0, Virchow2): timm models from gated HF repos,
  so add them to `TEACHERS` when timm is allowed. They have no class head, so they
  need the E0c-style probe for logit KD.

## Suggested library developments, most useful first

`parts.py` is a working prototype of each item below. Promoting it into `lwfm`, with
tests, makes it reusable beyond this experiment.

1. **`Trainer`** ← `parts.fit`, `make_optimizer`, `make_scheduler`: the loop, optimisers,
   per-step schedules, clipping, logging, validation and loss-weight ramps.
2. **Model registry with an output contract** ← `parts.STUDENTS`, `TEACHERS`,
   `ModelSpec`, `teacher_node`. Students adapt to the image size; teachers come with
   their resize, normalisation and class mapping.
3. **Dataset registry + `DataModule`** (PLAN.md §3.5) ← `parts.DATASETS`, `SplitData`,
   `make_split`.
4. **`Experiment` / run orchestration** ← `parts.run`, `Experiment`, `completed`:
   layered settings, seeding, run folders, preflight, final test, saving the
   deployable model, cost, resuming, sweep → confirm.
5. **KD technique bundles** ← `parts.Setup` and the losses' `reads`.
6. **Results and analysis** ← `parts.results`, `best` and the notebook's table/plot
   cells.
7. **`BenchmarkRunner`** (PLAN.md §3.8) ← `parts.inference_cost`.
8. **Teacher heads** ← `parts.teacher_embeddings`, `fit_teacher_head`, `TaskHead`:
   E0c, and pathology teachers without a class head. Also a cached teacher node, for E7.
