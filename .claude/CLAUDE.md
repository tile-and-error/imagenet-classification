# ImageNet Classification + Segmentation

A distillation pipeline for two tasks, image classification and semantic segmentation,
on ImageNet. It is the lead-up to the CS407 project, which runs the same two tasks on
whole-slide pathology images (WSIs); here the method is proven on standard natural images
first, so the move to WSIs is a change of data and teachers rather than of design.

## The pipeline

Knowledge distillation: a large, capable model (the teacher) supervises a smaller, cheaper
one (the student), so the student approaches the teacher's quality at a fraction of the
inference cost. The stages connect as follows.

- **Data inputs** supply labelled images for a task.
- **Preprocessing** turns them into model-ready batches (resizing, cropping, augmentation),
  identically for teacher and student so both see the same input.
- **Teacher(s)** are pretrained and frozen; they only produce the targets the student
  learns from. A run may use more than one teacher at once, and the student learns from
  their combined signal.
- **Student** is the model being trained — the artefact the project ultimately cares about.
- **Loss functions** measure how far the student's output is from the teacher's (and, where
  labels are used, from ground truth). With several teachers, the losses are combined.
- **Metrics** score task quality against ground truth; **benchmarks** package a dataset,
  metric and protocol into a repeatable evaluation.

Every stage is a swappable component chosen per experiment, so a new teacher, student,
loss, dataset or benchmark is added without touching the rest.

### Torch-Native Throughout

Every component is built on standard PyTorch primitives and nothing else — models are ordinary torch modules, data flows as tensors through standard loaders, and losses and metrics operate on those tensors directly. No component wraps the model in a bespoke framework or a non-torch abstraction. This keeps the pipeline portable: any component can be trained, profiled, scripted, or exported through the normal torch tooling, and a model proven here drops into the later WSI setting (and eventually onto constrained hardware) without being unwound from a framework first.

## Style

Concise, mathematical, specific. Check claims rather than build on them. `ruff` owns
formatting and lint (see `pyproject.toml`). Keep the code self-documenting, don't add docstrings
when you don't need to.
