# Experiment and Study Specification

## Purpose

`lwfm` joins ordinary user Python objects into reproducible experiments. YAML
describes wiring, scalar values, stages, and search candidates. It is not an
object-construction language.

The user passes model factories, loss callables, metrics, a data module, and
training objects directly to `Experiment`. This follows `examples/building_blocks.ipynb`
and `examples/trainer.ipynb`: the user constructs normal PyTorch components and
`lwfm` joins them with `ModelNode`, `ExecutionGraph`, `LossComposer`,
`MetricBinding`, and `Trainer`. There is no catalog or string-to-object lookup.

There are two YAML formats:

1. A **run specification** describes one concrete run.
2. A **study specification** expands a base run into concrete runs and selects
   using validation results.

## Python boundary

The caller owns all executable behaviour. Fixed implementation choices should
be partial applications in Python; YAML contains only experiment values.

```python
from functools import partial

import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

experiment = Experiment(
    config,
    data_module=data_module,
    model_factories={
        "student": partial(make_student, num_classes=1000),
        "teacher": partial(make_teacher, checkpoint=teacher_checkpoint),
    },
    losses={
        "cross_entropy": nn.CrossEntropyLoss(),
        "distillation": distillation_loss,
    },
    metrics={
        "top1": MulticlassAccuracy(num_classes=1000, top_k=1),
        "top5": MulticlassAccuracy(num_classes=1000, top_k=5),
    },
    trainer_factory=partial(Trainer, device="cuda"),
    optimizer_factory=partial(torch.optim.SGD, momentum=0.9),
    callbacks=[checkpoint_callback],
)
```

The exact `Experiment` constructor may instead receive a small parts dataclass,
but these are explicit Python arguments, not IDs in YAML. Stateful objects
(models, metrics, optimizers, schedulers, callbacks, and module losses) must be
fresh for every run.

For a study, the user supplies a normal function which makes fresh objects for
one resolved configuration:

```python
def make_experiment(config: ExperimentConfig) -> Experiment:
    return Experiment(
        config,
        data_module=make_data_module(),
        model_factories={"student": partial(make_student, num_classes=1000)},
        losses={"cross_entropy": nn.CrossEntropyLoss()},
        metrics={"top1": MulticlassAccuracy(num_classes=1000)},
        trainer_factory=partial(Trainer, device=config.runtime.device),
        optimizer_factory=partial(torch.optim.SGD, momentum=0.9),
    )
```

## Graph construction

`Experiment` builds the graph through a static method. The caller constructs
fresh modules; the method applies the declarative node wiring:

```python
@staticmethod
def build_graph(config, modules, preflight_sample=None) -> ExecutionGraph:
    nodes = [
        ModelNode(
            name=name,
            module=modules[name],
            inputs=spec.inputs,
            outputs=spec.outputs,
            frozen=spec.frozen,
        )
        for name, spec in config.models.items()
    ]
    return ExecutionGraph(nodes, preflight_sample=preflight_sample)
```

For a simple graph, `Experiment` can construct `modules` by calling each
`model_factory[name](**config.models[name].params)`. More specialised graphs
can use a user `graph_factory(config, preflight_sample)` and call this static
method after establishing relationships between modules. `Experiment.build_graph`
still owns `ModelNode` and `ExecutionGraph` construction and validation.

## Idiomatic custom training: spare EMA student

Do not represent callbacks as names in YAML. A callback is executable,
stateful Python behaviour, so create it in a `fit_factory` called after the
graph and run directory exist:

```python
def make_fit(config, graph, run_dir) -> FitPlan:
    return FitPlan(
        trainer=Trainer(device=config.runtime.device, max_epochs=config.training.max_epochs),
        optimizer=torch.optim.SGD(
            (p for p in graph.parameters() if p.requires_grad),
            lr=config.training.params["learning_rate"],
            momentum=0.9,
        ),
        callbacks=[
            EMAUpdate("student", "student_ema", momentum=0.999),
            ModelCheckpoint(run_dir, node="student_ema"),
        ],
    )
```

`FitPlan` is a small library dataclass containing a `Trainer`, optimizer,
optional scheduler/interval, and callbacks; `FitStage` passes it to
`Trainer.fit`. It avoids a growing YAML callback vocabulary.

An EMA is also a graph-construction relationship: `student_ema` starts as a
frozen copy of `student`. A user graph factory expresses that directly:

```python
def make_kd_graph(config, preflight_sample):
    student = make_student(**config.models["student"].params)
    modules = {
        "student": student,
        "student_ema": copy.deepcopy(student),
        "teacher": make_teacher(**config.models["teacher"].params),
    }
    return Experiment.build_graph(config, modules, preflight_sample)
```

YAML declares that `student_ema` exists, is frozen, and is deployable. Expose
`training.params.ema_momentum` only if it is a hyperparameter worth recording
or searching; otherwise the fixed momentum belongs in `make_fit`.

## Run specification

```yaml
schema_version: 1
name: kd_resnet18
seed: 42

runtime:
  device: cuda
  precision: fp32

models:
  student:
    inputs: [inputs.image]
    outputs: [logits]
    frozen: false
    params:
      width_multiplier: 1.0
      dropout: 0.1
  teacher:
    inputs: [inputs.image]
    outputs: [logits]
    frozen: true
    params: {}

losses:
  cross_entropy:
    enabled: true
    inputs: [student.logits, targets.class]
    weight: 1.0
  distillation:
    enabled: true
    inputs: [student.logits, teacher.logits]
    weight: 0.5

metrics:
  top1:
    inputs: [student.logits, targets.class]
  top5:
    inputs: [student.logits, targets.class]

training:
  max_epochs: 20
  params:
    learning_rate: 0.01

deployed_nodes:
  student:
    inputs: [inputs.image]
    outputs: [logits]

outputs:
  root: run_outputs
  sinks: [local, console]

recipe:
  stages:
    - name: fit
      type: fit
      train_split: train
      validation_split: val
      outputs: {state: training_state, student: weights.student}
    - name: select
      type: select_checkpoint
      checkpoint: fit.state
      monitor: val.top1
      goal: max
      outputs: {model: selected.weights}
    - name: evaluate_val
      type: evaluate
      weights: select.model
      split: val
      metrics: [top1, top5]
```

| Section | Maps to |
| --- | --- |
| `models` | `Experiment.build_graph`, `ModelNode`, `ExecutionGraph` |
| `losses` | `LossBinding`, `LossComposer` |
| `metrics` | `MetricBinding`, `Evaluator` |
| `training` | supplied trainer, optimizer, scheduler, and callback factories |
| `deployed_nodes` | `ExecutionGraph.deployable_model` |
| `recipe` | `Recipe` and typed `Stage` instances |
| `outputs` | `OutputRouter` and output sinks |

Unknown keys are errors. Defaults are materialised into `config.resolved.yaml`.

### Models

Every key under `models` must have a corresponding supplied factory. `inputs`,
`outputs`, and `frozen` map directly to `ModelNode`. Short outputs are qualified
with the node name: `student.outputs: [logits]` writes `student.logits`.

`models.<node>.params` is the home for model hyperparameters. It is a
JSON-shaped mapping passed unchanged to that node's factory. `lwfm` neither
interprets its keys nor inspects constructor signatures. Values such as width,
dropout, patch size, adapter rank, or an application-specific checkpoint choice
can therefore be recorded and searched, while a partial factory keeps fixed
implementation choices in Python.

`frozen: true` means no parameter gradients and evaluation mode during graph
training. Every resolved configuration undergoes normal graph and preflight
validation, so invalid shape-changing parameter combinations fail early.

### Losses, metrics, training, and data

Loss and metric names select entries in the supplied `losses` and `metrics`
mappings. YAML defines only their input binding and loss weight.

- `enabled: false` means no `LossBinding` is created or evaluated.
- `enabled: true` is the default.
- `weight` is finite and non-negative. A zero-weight enabled loss is evaluated
  and logged; use `enabled: false` to deactivate it completely.

`training.params` is an opaque JSON-shaped mapping provided to user training
factories. For example, a partial `SGD(momentum=0.9)` may receive
`learning_rate` from it. Loop controls owned by `lwfm`, such as `max_epochs`,
`max_steps`, validation cadence, accumulation, clipping, and precision, are
ordinary `training` fields.

The caller supplies the data module; there is no `data:` construction section.
Stages request `data.train`, `data.val`, `data.test`, or another declared split.
The data module owns splits, preprocessing, batching, workers, and provenance.

## Recipe and artifacts

Stages run in order and have unique names. A recipe must not represent
`Experiment`'s constructed graph or its supplied data module as pretend
artifacts. Their origins are already explicit at the Python boundary.

Each stage type instead has a small typed schema:

- `fit` always uses the graph freshly built by `Experiment`; `train_split` and
  `validation_split` name split capabilities of the supplied data module.
- `evaluate` and `benchmark` use `split` for a data-module split and `weights`
  for a stage-produced model artifact.
- Only `<stage>.<output>` is an artifact reference. It always identifies an
  output declared by an earlier stage.

Thus `checkpoint: fit.state` plainly means “the `state` artifact produced by
the `fit` stage”, while `split: val` plainly means “ask the supplied data module
for its validation split”. There is no `initial.graph` or `data.train` mini
language to learn.

| Stage | Inputs | Outputs |
| --- | --- | --- |
| `fit` | built graph; `train_split`; optional `validation_split` | resumable state and deployable weights |
| `select_checkpoint` | `checkpoint: <stage>.<output>` | selected weights |
| `evaluate` | `weights: <stage>.<output>` and `split` | measurements/events |
| `benchmark` | `weights: <stage>.<output>` | measurements/events |
| `transform` / `export` later | model/weights | transformed/exported artifact |

A resumable training state is never a deployment artifact. Benchmark/export
stages must name a node in `deployed_nodes`, preventing accidental teacher or
training-projector measurement.

## Study specification

A study is separate from a run. It loads `base`, expands a strategy into
patches, validates each resulting run, then calls `make_experiment` for every
trial.

```yaml
schema_version: 1
name: kd_search
base: ../experiments/kd_base.yaml

strategy:
  type: product
  factors:
    - type: values
      choices:
        - id: ce_only
          overrides:
            losses.distillation.enabled: false
        - id: kd
          overrides:
            losses.distillation.enabled: true
            losses.distillation.weight: 0.5
    - type: grid
      axes:
        - path: models.student.params.dropout
          values: [0.0, 0.1]
        - path: training.params.learning_rate
          values: [0.003, 0.01]

replicates:
  seed: [101, 202, 303]

selection:
  split: val
  metric: top1
  goal: max
  aggregate: mean

budget:
  max_runs: 24
  max_concurrent: 3
```

### Strategies

| Type | Shape | Meaning |
| --- | --- | --- |
| `values` | `choices: [{id, overrides}, ...]` | A named fixed set of correlated values. |
| `grid` | `axes: [{path, values}, ...]` | Cartesian product of independent value lists. |
| `product` | `factors: [<strategy>, ...]` | Cartesian product of child strategy results. |

Use `values` for coupled ablations; it replaces a tuple/zip format. Use `grid`
only when every combination is meaningful. Use `product` to combine fixed
ablations with independent grids. The example has `2 × 2 × 2 × 3 = 24` trials.

A list-valued candidate is one value in an axis:

```yaml
type: grid
axes:
  - path: models.student.params.layer_widths
    values: [[32, 64, 128], [64, 128, 256]]
```

`replicates` is separate because seeds repeat a scientific candidate; they are
not hyperparameters. The selector aggregates replicates before comparing
candidates.

Study validation rejects unknown paths, conflicting values for the same product
path, empty grids, duplicate choice IDs, patches that replace Python objects or
whole stages, and test-set selection. Each trial records its patch, base hash,
resolved configuration, study ID, and trial ID.

## Execution order

1. Safely parse and strictly validate YAML.
2. Apply defaults and any study patch; write resolved YAML.
3. Call `make_experiment` to create fresh user objects.
4. Build/preflight the graph and validate data split, loss, metric, and artifact
   references.
5. Create the run manifest and execute the recipe.

The manifest records resolved configuration, data fingerprint, environment,
source revision, events, and artifact hashes.

## Researcher walkthrough: distilled text recognition

This example asks a simple research question: **which student architecture,
tile layout, and distillation-loss weighting gives the best character-list
predictions for handwritten text on varied backgrounds?** A CNN is compared at
several centred-text tile layouts against a ViT-based student. Exact model and
loss implementations are intentionally not part of the configuration format.

- Define the outcome before writing a config. The primary quality measure might
  be character error rate (lower is better); exact-sequence accuracy and
  per-character accuracy are useful secondary measures. Record latency and
  model size only as tie-breakers or additional constraints.

- Supply one project data module. It yields batches such as
  `inputs.image`, `targets.characters`, and `meta.sample_id`; it owns all
  decoding, background augmentation, padding/collation, and train/validation/
  test splits. Split by the right unit (for example writer or document), so the
  same writer/background combination cannot leak between training and test.
  The run config only says `train_split: train` or `split: val`.

- Write one project-level component builder, once. It is ordinary Python, not a
  YAML plugin selected per run. It accepts the resolved run config and creates
  fresh modules, metrics, optimizer, scheduler, callbacks, and data module.
  The runner always calls this same builder for every run and study trial. Its
  minimal interface is:

  ```python
  def make_experiment(config: ExperimentConfig) -> Experiment: ...
  ```

  Internally it calls `Experiment.build_graph(config, modules, sample)` and
  creates the fit plan. This is the irreducible project-code boundary: a generic
  library cannot construct an arbitrary CNN, ViT, sequence loss, or EMA policy
  without user code. It is not functionality injected by an individual YAML
  file, and it does not require a catalog.

- Put only experiment values in the base run YAML. The builder interprets the
  model `params` that it has documented; all fixed details remain Python
  partials. A representative base configuration is:

  ```yaml
  models:
    student:
      inputs: [inputs.image]
      outputs: [character_logits]
      frozen: false
      params:
        architecture: cnn
        tile_layout: centred_3x2
    teacher:
      inputs: [inputs.image]
      outputs: [character_logits]
      frozen: true
      params: {}

  losses:
    character_ce:
      inputs: [student.character_logits, targets.characters]
      weight: 1.0
    distillation:
      enabled: true
      inputs: [student.character_logits, teacher.character_logits]
      weight: 0.5

  metrics:
    character_error_rate:
      inputs: [student.character_logits, targets.characters]
    exact_sequence_accuracy:
      inputs: [student.character_logits, targets.characters]

  training:
    max_epochs: 30
    params:
      learning_rate: 0.001
  ```

- Keep the baseline fair. Use the same data splits, augmentation policy,
  stopping rule, validation schedule, seed replicas, output vocabulary, and
  training budget for every student. If CNN and ViT input representations differ
  materially, record that as part of the candidate rather than quietly changing
  it.

- Use the recipe for the fixed workflow: fit with the supplied train/validation
  splits, select a checkpoint by validation character error rate, evaluate that
  checkpoint on validation, and benchmark the declared deployable student. Do
  not evaluate the final test split inside candidate selection.

- Express mutually dependent choices as named fixed candidates. For example,
  `ce_only` disables distillation, while `logit_kd` enables it with a particular
  weight. This avoids meaningless combinations such as a disabled term with a
  supposedly selected KD recipe.

- Express independent choices as a grid. The following study compares CNN tile
  layouts and a ViT, then tests two loss weights and two learning rates. It is
  deliberately small enough to understand and budget:

  ```yaml
  strategy:
    type: product
    factors:
      - type: values
        choices:
          - id: cnn_small_tiles
            overrides:
              models.student.params.architecture: cnn
              models.student.params.tile_layout: centred_3x2
          - id: cnn_large_tiles
            overrides:
              models.student.params.architecture: cnn
              models.student.params.tile_layout: centred_2x1
          - id: vit
            overrides:
              models.student.params.architecture: vit
              models.student.params.tile_layout: full_image
      - type: grid
        axes:
          - path: losses.distillation.weight
            values: [0.25, 0.5]
          - path: training.params.learning_rate
            values: [0.0003, 0.001]

  replicates:
    seed: [101, 202, 303]

  selection:
    split: val
    metric: character_error_rate
    goal: min
    aggregate: mean
  ```

- Read the study output as an optimal **set of values**, not merely a best
  checkpoint: architecture, tile layout, KD enabled/disabled state, KD weight,
  learning rate, and all unchanged base values. The study aggregates validation
  results across seeds and emits the winning resolved configuration.

- Finally, freeze that selected configuration, train/report it according to the
  project protocol, and evaluate once on the untouched test split. The test
  result confirms the selected hyperparameters; it must not feed back into the
  study.
