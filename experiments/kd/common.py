"""What the E0-E3 scripts share.

- Each stage (e0.py ... e3.py) is its own job. It saves its choices (e.g. the best
  KD setting) in runs/.../decisions.json, and later stages rebuild setups from them.
- Completed runs are skipped, so a stopped job can just be submitted again.
- Everything a stage reports goes through one lwfm OutputRouter, into its own run
  folder runs/.../report/<stage>/: printed, recorded (plus a CSV per table) and
  plotted.
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import pandas as pd
import torch

from experiments.kd.parts import (
    FAST,
    LOCAL,
    LOCAL_FAST,
    LOCAL_LONG,
    LOCAL_MAIN,
    LONG,
    MAIN,
    REPO,
    RUNS,
    SMOKE,
    Setup,
    SplitData,
    TrainConfig,
    make_split,
    pick_device,
    results,
)
from lwfm.results.artifacts import save_artifact
from lwfm.results.curves import Lines
from lwfm.results.events import RunEvent
from lwfm.results.plots import PlotSink
from lwfm.results.router import OutputRouter
from lwfm.results.runs import run_status
from lwfm.results.sinks import ConsoleSink, LocalFileSink, OutputSink, write_text_atomic
from lwfm.techniques.distillation import DIST, DKD, KD

# The choices, as in the notebook's setup cell
DATASET = "tiny-imagenet"
NAMED = [f"{i:03d}" for i in range(10)]  # Tiny ImageNet folders 000-009
NUM_OTHER = 40  # 10 + 40 split; None for 10 + 190 (all other classes)
STUDENTS_TO_RUN = ["resnet18", "vit_ti"]  # a CNN and a ViT
TEACHER = "resnet50"

SHOWN = [  # columns of the run tables
    "run",
    "student",
    "teacher_head",
    "other_shift",
    "eval/val/bal_acc",
    "eval/val/macro_f1",
    "eval/test/bal_acc",
    "eval/test/macro_f1",
    "eval/test/top1",
    "eval/test/top5",
    "train/epoch_seconds",
    "cost/params_m",
    "cost/gflops",
    "cost/latency_ms_median",
]
START = time.time()  # for the elapsed time in each section header
# Print wide result tables in full rather than truncated
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)


@dataclass
class Ctx:
    """What every stage needs: the data, where results go, the budgets, and the
    stage's report sink."""

    stage: str  # "E0" ... "E3": labels every report event
    mode: str  # "smoke", "local" or "full"
    device: torch.device
    data: SplitData
    runs: Path  # every run of this mode and split
    report: Path  # this stage's report: an lwfm run folder
    sink: OutputSink  # where the stage reports to (console, files, plots)
    kw: dict  # passed to every run(): device, pretrained, overrides, runs
    eval_cfg: TrainConfig  # batch settings for evaluation-only steps
    fast: TrainConfig  # the training budgets: FAST ranks settings in sweeps,
    main: TrainConfig  # MAIN gives the reported numbers,
    long: TrainConfig  # LONG checks whether KD keeps improving
    seeds: tuple[int, ...]  # seeds for the MAIN runs

    @property
    def smoke(self) -> bool:
        return self.mode == "smoke"


# ---------------------------------------------------------------------- reporting


@dataclass(frozen=True)
class Frame:
    """A results table, sent as a `table` event: printed in full by ConsoleSink (its
    str), stored as its title and columns by LocalFileSink."""

    title: str
    columns: dict[str, list]

    @classmethod
    def of(cls, df: pd.DataFrame, title: str) -> "Frame":
        # Named row labels (a pivot's or group-by's) become columns; plain row
        # numbers are dropped
        if any(name is not None for name in df.index.names):
            df = df.reset_index()
        return cls(title, {flat_name(c): df[c].tolist() for c in df.columns})

    def __str__(self) -> str:
        df = pd.DataFrame(self.columns)
        body = df.round(4).to_string(index=False) if len(df) else "(no completed runs)"
        return f"\n{self.title}\n{body}"


def flat_name(column: object) -> str:
    # A two-level column as one name: ("eval/val/bal_acc", "mean"), ("stage", "")
    if isinstance(column, tuple):
        return " / ".join(str(part) for part in column if part != "")
    return str(column)


def note(ctx: Ctx, name: str, value: object) -> None:
    """Reports a message (a note, a setting, a decision) through the stage's sinks."""
    ctx.sink.write(RunEvent("message", name, value, stage=ctx.stage))


def section(ctx: Ctx, title: str) -> None:
    """Reports the start of a section, with the time and the minutes so far."""
    minutes = (time.time() - START) / 60
    note(
        ctx, "section", f"{title}   [{time.strftime('%H:%M:%S')}, {minutes:.0f} min in]"
    )


def table(ctx: Ctx, name: str, df: pd.DataFrame, title: str) -> None:
    """Reports a table, and saves it as <name>.csv in the stage's report."""
    ctx.sink.write(RunEvent("table", name, Frame.of(df, title), stage=ctx.stage))
    save_artifact(ctx.report, f"{name}.csv", df.to_csv, ctx.sink, name=f"csv/{name}")


def show(ctx: Ctx, pattern: str, sort: str = "eval/val/bal_acc") -> pd.DataFrame:
    # One row per completed run matching runs/<pattern>, best first
    df = pd.DataFrame(results(pattern, ctx.runs))
    if df.empty:
        return df
    return df[[c for c in SHOWN if c in df]].sort_values(sort, ascending=False)


def logged(ctx: Ctx, run_name: str, name: str) -> list:
    # Every value logged under `name` in one run, in order (e.g. a per-epoch curve)
    lines = (ctx.runs / run_name / "events.jsonl").read_text().splitlines()
    return [e["value"] for e in map(json.loads, lines) if e["name"] == name]


# ------------------------------------------------------------------------ setup


def parse_mode(stage_name: str) -> tuple[str, torch.device]:
    """The --mode from the command line ("auto" resolved) and the device to use."""
    parser = argparse.ArgumentParser(description=f"Knowledge distillation {stage_name}")
    parser.add_argument(
        "--mode",
        choices=["auto", "smoke", "local", "full"],
        default="auto",
        help="auto: full on a CUDA GPU, local otherwise; smoke: a quick wiring check",
    )
    mode = parser.parse_args().mode
    device = pick_device()
    if mode == "auto":
        mode = "full" if device.type == "cuda" else "local"
    if mode == "full" and device.type != "cuda":
        sys.exit(f"--mode full needs a CUDA GPU, but the device is {device}")
    return mode, device


@contextmanager
def stage(name: str, run_patterns: list[str]) -> Generator[Ctx]:
    """Sets up a stage and opens its report, for `with stage("E1", ...) as ctx:`.
    `run_patterns` are the stage's own run folders under runs/ (for restarts).
    Leaving the `with` records whether the stage completed or failed."""
    mode, device = parse_mode(name)

    # The data: our split of Tiny ImageNet, and one loader worker per CPU the job
    # was given, less one for the main process (sched_getaffinity follows Slurm's
    # --cpus-per-task; macOS doesn't have it)
    split = make_split(
        DATASET, keep=NAMED, num_other=NUM_OTHER, train_fraction=0.7, exist_ok=True
    ).name
    cpus = len(os.sched_affinity(0)) if hasattr(os, "sched_getaffinity") else 8
    workers = 0 if mode == "smoke" else max(1, cpus - 1)
    data = SplitData(DATASET, split, workers=workers)

    # Folders. The report is rebuilt from the runs every time the stage runs
    # (LocalFileSink won't write into an existing run folder)
    root = RUNS if mode == "full" else RUNS / mode  # modes never mix their runs
    runs = root / data.name
    report = runs / "report" / name.lower()
    shutil.rmtree(report, ignore_errors=True)

    # Budgets: shorter in local mode; smoke and local also shrink every run
    budgets = (
        (LOCAL_FAST, LOCAL_MAIN, LOCAL_LONG) if mode == "local" else (FAST, MAIN, LONG)
    )
    overrides = {"smoke": SMOKE, "local": LOCAL}.get(mode)
    seeds = (0, 1, 2) if mode == "full" else (0,)
    gpu = f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""
    epochs = [replace(b, **(overrides or {})).epochs for b in budgets]  # as trained
    settings = {  # recorded as the report's manifest, and printed
        "stage": name,
        "mode": mode,
        "device": f"{device}{gpu}",
        "workers": workers,
        "split": split,
        "students": STUDENTS_TO_RUN,
        "teacher": TEACHER,
        "epochs (FAST, MAIN, LONG)": epochs,
        "seeds": list(seeds),
        "runs": str(runs.relative_to(REPO)),
    }
    # The report: printed, recorded in files, and its curves drawn
    sinks = LocalFileSink(report, settings), ConsoleSink(), PlotSink(report / "plots")
    with OutputRouter(*sinks) as sink:
        ctx = Ctx(
            stage=name,
            mode=mode,
            device=device,
            data=data,
            runs=runs,
            report=report,
            sink=sink,
            kw=dict(
                device=device,
                pretrained=mode != "smoke",
                overrides=overrides,
                runs=runs,
            ),
            eval_cfg=replace(MAIN, **(overrides or {})),
            fast=budgets[0],
            main=budgets[1],
            long=budgets[2],
            seeds=seeds,
        )
        section(ctx, f"{name} SETUP")
        note(ctx, "settings", settings)
        restart_unfinished(ctx, run_patterns)
        yield ctx
        section(ctx, f"{name} DONE")
        note(ctx, "report", str(report.relative_to(REPO)))


def restart_unfinished(ctx: Ctx, run_patterns: list[str]) -> None:
    # A job killed by its time limit or scancel leaves its last run marked
    # "running", which would stop this job. Delete such runs so they start again.
    # Only this stage's runs are touched, so different stages can run at once.
    for pattern in run_patterns:
        for status in ctx.runs.glob(f"{pattern}/**/status.json"):
            if run_status(status.parent) == "running":
                note(ctx, "restart", str(status.parent.relative_to(ctx.runs)))
                shutil.rmtree(status.parent)


# ------------------------------------------------- hand-off between the stages


def save_decision(ctx: Ctx, key: str, value: object) -> None:
    """Records one stage's choice in runs/.../decisions.json, for later stages."""
    path = ctx.runs / "decisions.json"
    decisions = json.loads(path.read_text()) if path.exists() else {}
    decisions[key] = value
    write_text_atomic(path, json.dumps(decisions, indent=2) + "\n")
    note(ctx, f"decision/{key}", value)


def load_decision(ctx: Ctx, key: str, made_by: str) -> object:
    """A choice an earlier stage saved; exits with a clear message if it hasn't run."""
    path = ctx.runs / "decisions.json"
    decisions = json.loads(path.read_text()) if path.exists() else {}
    if key not in decisions:
        sys.exit(f"{key} isn't in {path.relative_to(REPO)}: run {made_by} first")
    note(ctx, f"decision/{key}", decisions[key])
    return decisions[key]


def teacher_setup(variant: str, shift: float | None) -> partial:
    """Setup with the KD teacher filled in, as one of the E0c variants: "logits"
    (ImageNet logits mapped to our classes, with the tuned "other" shift) or a head
    ("linear", "mlp")."""
    args = {"other_shift": shift} if variant == "logits" else {"teacher_head": variant}
    return partial(Setup, teacher=TEACHER, **args)


def chosen_teacher(ctx: Ctx) -> tuple[str, partial]:
    """The teacher variant E0c chose, as its run-name tag and a Setup factory."""
    tag = str(load_decision(ctx, "teacher_variant", "e0.py"))
    shifts = load_decision(ctx, "shifts", "e0.py")
    assert isinstance(shifts, dict)
    return tag, teacher_setup(tag.removeprefix(f"{TEACHER}-"), shifts[TEACHER])


def e1_setups(S: partial, student: str) -> dict[str, Setup]:
    """E1's grid: logit KD at T x alpha, plain and class-weighted. Keys name the
    setting, e.g. "wkd_T4_a0.9" is class-weighted KD at T = 4, alpha = 0.9."""
    return {
        f"{'w' if weighted else ''}kd_T{T}_a{alpha}": S(
            student, logit=partial(KD, T=T), alpha=alpha, weighted_kd=weighted
        )
        for T in (1, 2, 4)
        for alpha in (0.5, 0.9)
        for weighted in (False, True)
    }


def e2_setups(S: partial, student: str, e1_best: str) -> dict[str, Setup]:
    """E2's candidates: the best E1 setting against DKD and DIST."""
    return {
        e1_best: e1_setups(S, student)[e1_best],
        "dkd_b0.5": S(student, logit=partial(DKD, alpha=0.5, beta=0.5)),
        "dkd_b8": S(student, logit=partial(DKD, alpha=0.5, beta=8.0)),
        "dist": S(student, logit=partial(DIST, beta=1.0, gamma=1.0)),
    }


# ---------------------------------------------------------------------- summary


def summary(ctx: Ctx) -> None:
    """Reports every completed training run so far (mean, std and count over seeds),
    and each student's validation curves on the MAIN budget."""
    section(ctx, "RESULTS SO FAR: every training run, mean / std / count over seeds")
    df = pd.DataFrame(results(runs=ctx.runs))
    if df.empty:
        note(ctx, "note", "no completed runs yet")
        return
    df = df[~df["run"].str.startswith("teachers/")]  # training runs only
    parts = df["run"].str.split("/")  # <variant>/<experiment>/<student>/<stage>/<setup>
    df["teacher"], df["experiment"] = parts.str[0], parts.str[1]
    df["stage"] = parts.str[3]
    df["config"] = parts.str[4].str.replace(r"_s\d+$", "", regex=True)
    metrics = [
        "eval/test/bal_acc",
        "eval/test/macro_f1",
        "eval/test/top5",
        "eval/val/bal_acc",
        "train/epoch_seconds",
        "cost/params_m",
        "cost/gflops",
    ]
    out = df.groupby(["stage", "student", "teacher", "experiment", "config"])[
        [m for m in metrics if m in df]
    ].agg(["mean", "std", "count"])
    table(ctx, "summary", out, "All runs (read the main stage for reported numbers)")
    val_curves(ctx, df[df["stage"] == "main"])


def val_curves(ctx: Ctx, main_runs: pd.DataFrame) -> None:
    """Reports one plot per student: val balanced accuracy per epoch for each setup
    on the MAIN budget, averaged over seeds."""
    for student in STUDENTS_TO_RUN:
        curves: dict[str, list[float]] = {}
        setups = main_runs[main_runs["student"] == student]
        for config, group in setups.groupby(["teacher", "experiment", "config"]):
            per_seed = [logged(ctx, r, "eval/val/bal_acc") for r in group["run"]]
            n = min(map(len, per_seed))  # epochs every seed has logged
            mean = [sum(c[i] for c in per_seed) / len(per_seed) for i in range(n)]
            curves["/".join(map(str, config))] = mean
        # Lines needs equal-length columns: cut every setup to the shortest
        n = min(map(len, curves.values()), default=0)
        if n:
            epochs = [float(epoch) for epoch in range(1, n + 1)]
            lines = Lines({"epoch": epochs} | {k: v[:n] for k, v in curves.items()})
            name = f"val_bal_acc_main/{student}"
            ctx.sink.write(RunEvent("table", name, lines, stage=ctx.stage))
