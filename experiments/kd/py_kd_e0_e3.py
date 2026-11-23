"""Knowledge distillation E0-E3 as a script, for batch jobs.

The same runs, in the same order, as kd_e0_e3.ipynb (whose markdown explains each
experiment; kd_plan_E0-E3.md gives the reasoning). From the repository root:

    python -u experiments/kd/py_kd_e0_e3.py                # full on CUDA, else local
    python -u experiments/kd/py_kd_e0_e3.py --mode smoke   # a few-minute wiring check

Everything printed is also saved: every table as CSV and every plot as PNG, in
runs/<mode>/<dataset>/<split>/report/, next to the run folders. A completed run is
skipped, so running it again carries on where a stopped job left off.
"""

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a compute node: plots are saved as PNG
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.kd.parts import (  # noqa: E402
    DIST,
    DKD,
    FAST,
    KD,
    LOCAL,
    LOCAL_FAST,
    LOCAL_LONG,
    LOCAL_MAIN,
    LONG,
    MAIN,
    RUNS,
    SMOKE,
    TEACHERS,
    EmbedCos,
    EmbedMSE,
    Experiment,
    Setup,
    SplitData,
    ToStudentClasses,
    TrainConfig,
    evaluate_teacher,
    fit_teacher_head,
    make_split,
    pick_device,
    results,
    run,
    student_node,
    teacher_embed_stats,
    tune_other_shift,
)
from lwfm.core.graph import ExecutionGraph  # noqa: E402
from lwfm.core.losses import LossComposer  # noqa: E402

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
START = time.time()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)


@dataclass
class Ctx:
    """What every section needs: the data, where results go, and the budgets."""

    mode: str  # "smoke", "local" or "full"
    device: torch.device
    data: SplitData
    runs: Path
    report: Path
    kw: dict  # passed to every run
    eval_cfg: TrainConfig
    fast: TrainConfig
    main: TrainConfig
    long: TrainConfig
    seeds: tuple[int, ...]

    @property
    def smoke(self) -> bool:
        return self.mode == "smoke"


# --------------------------------------------------------------------- printing


def section(title: str) -> None:
    minutes = (time.time() - START) / 60
    bar = "=" * 88
    print(
        f"\n{bar}\n{title}   [{time.strftime('%H:%M:%S')}, {minutes:.0f} min in]\n{bar}"
    )


def table(ctx: Ctx, name: str, df: pd.DataFrame, title: str) -> None:
    # Print a table and save it as report/<name>.csv
    path = ctx.report / f"{name}.csv"
    df.to_csv(path)
    body = df.round(4).to_string() if len(df) else "(no completed runs)"
    print(f"\n--- {title}\n    saved: {path.relative_to(REPO)}\n{body}", flush=True)


def figure(ctx: Ctx, name: str, fig: plt.Figure) -> None:
    path = ctx.report / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--- plot saved: {path.relative_to(REPO)}", flush=True)


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


def setup(mode: str) -> Ctx:
    section("SETUP")
    device = pick_device()
    if mode == "auto":
        mode = "full" if device.type == "cuda" else "local"
    if mode == "full" and device.type != "cuda":
        sys.exit(f"--mode full needs a CUDA GPU, but the device is {device}")

    split = make_split(
        DATASET, keep=NAMED, num_other=NUM_OTHER, train_fraction=0.7, exist_ok=True
    ).name
    data = SplitData(DATASET, split, workers=0 if mode == "smoke" else 8)
    root = RUNS if mode == "full" else RUNS / mode  # modes never mix their runs
    runs = root / data.name
    report = runs / "report"
    report.mkdir(parents=True, exist_ok=True)
    budgets = (
        (LOCAL_FAST, LOCAL_MAIN, LOCAL_LONG) if mode == "local" else (FAST, MAIN, LONG)
    )
    overrides = {"smoke": SMOKE, "local": LOCAL}.get(mode)
    ctx = Ctx(
        mode=mode,
        device=device,
        data=data,
        runs=runs,
        report=report,
        kw=dict(
            device=device, pretrained=mode != "smoke", overrides=overrides, runs=runs
        ),
        eval_cfg=replace(MAIN, **(overrides or {})),
        fast=budgets[0],
        main=budgets[1],
        long=budgets[2],
        seeds=(0, 1, 2) if mode == "full" else (0,),
    )
    gpu = f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""
    print(f"mode:      {mode}")
    print(f"device:    {device}{gpu}")
    print(f"split:     {split}")
    print(f"students:  {', '.join(STUDENTS_TO_RUN)}   teacher: {TEACHER}")
    epochs = [replace(b, **(overrides or {})).epochs for b in budgets]  # as trained
    print(
        f"budgets:   FAST {epochs[0]}, MAIN {epochs[1]}, LONG {epochs[2]} epochs "
        f"(students add their own optimiser settings); seeds {ctx.seeds}"
    )
    print(f"runs:      {runs.relative_to(REPO)}")
    print(f"report:    {report.relative_to(REPO)}  (every table and plot)")
    restart_unfinished(runs)
    return ctx


def restart_unfinished(runs: Path) -> None:
    # A job killed by its time limit or scancel leaves its last run marked
    # "running", which would stop this job. Delete such runs so they start again.
    # (So run one job at a time per mode.)
    for status in runs.glob("**/status.json"):
        if json.loads(status.read_text())["status"] == "running":
            print(f"restart unfinished run from a stopped job: {status.parent}")
            shutil.rmtree(status.parent)


def overview(ctx: Ctx) -> None:
    section("DATA")
    data = ctx.data
    counts = {
        split: pd.Series([int(r["label"]) for r in data.rows(split)])
        .value_counts()
        .sort_index()
        for split in ("train", "val", "test")
    }
    print(
        f"{data.num_classes} classes, {data.spec.image_size} px, "
        f"{len(data.other_folders())} source classes in 'other'"
    )
    df = pd.DataFrame(
        {
            "folder": data.classes,
            "name": data.class_names(),
            **counts,
            "CE weight": data.class_weights().numpy().round(3),
        }
    )
    table(ctx, "data", df, "Images per class and split, with the CE class weights")


def sanity_checks(ctx: Ctx) -> None:
    section("SANITY CHECKS (random weights, seconds)")
    data = ctx.data
    batch = next(iter(data.loader("val", 4)))
    for student in STUDENTS_TO_RUN:
        setup = Setup(student, TEACHER, logit=partial(KD, T=4), alpha=0.9)
        nodes, losses = setup.build(data, pretrained=False)
        graph = ExecutionGraph(nodes)
        print(f"{student} output shapes: {graph.preflight(batch)}")
    for b in losses:
        print(f"loss {b.name}: weight {b.weight}, reads {b.inputs}")

    # 1. With no shift, mapped teacher logits == a softmax over the split's classes
    keep, other = data.teacher_classes(), data.teacher_other_classes()
    z = torch.randn(4, 1000)
    p_split = z[:, keep + other].softmax(1)
    p = ToStudentClasses(keep, other or None, shift=0.0)({"logits": z})["logits"]
    p = p.softmax(1)
    assert torch.allclose(p[:, : len(keep)], p_split[:, : len(keep)], atol=1e-6)
    if other:
        assert torch.allclose(p[:, -1], p_split[:, len(keep) :].sum(1), atol=1e-6)
    # 2. Identical logits -> ~0 loss
    z, y = torch.randn(8, data.num_classes), torch.randint(0, data.num_classes, (8,))
    print(
        f"identical logits: KD {KD()(z, z):.2e}, DKD {DKD()(z, z, y):.2e}, "
        f"DIST {DIST()(z, z):.2e}"
    )
    # 3. Weighted KD with equal weights == plain KD
    z2 = torch.randn(8, data.num_classes)
    equal = torch.ones(data.num_classes)
    assert torch.allclose(KD(weight=equal)(z, z2, y), KD()(z, z2))
    # 4. Gradients reach the student only
    total, _ = LossComposer(losses)(batch, graph(batch))
    total.backward()
    assert all(p.grad is None for p in graph.nodes["teacher"].parameters())
    assert all(p.grad is not None for p in graph.nodes["student"].parameters())
    print("all sanity checks passed")


# ---------------------------------------------------------------- experiments


def e0a(ctx: Ctx) -> dict:
    section("E0a: THE TEACHERS ALONE (inference only)")
    data, shifts = ctx.data, {}
    for teacher in TEACHERS:
        if ctx.smoke and teacher == "dinov2_b14":
            continue  # needs network access even with random weights
        shifts[teacher] = None
        if data.has_other:
            print(f"\n>>> tune the 'other' shift of {teacher} on val")
            rows = tune_other_shift(
                teacher,
                data,
                ctx.device,
                pretrained=not ctx.smoke,
                eval_batches=ctx.eval_cfg.eval_batches,
                batch_size=ctx.eval_cfg.eval_batch_size,
            )
            shifts[teacher] = max(rows, key=lambda r: r["macro_f1"])["shift"]
            print(f"    chosen shift (best val macro-F1): {shifts[teacher]:.3f}")
            table(
                ctx,
                f"e0a_shift_{teacher}",
                pd.DataFrame(rows),
                f"{teacher}: val scores per 'other' shift",
            )
        evaluate_teacher(
            f"teachers/{teacher}/logits",
            teacher,
            data,
            other_shift=shifts[teacher],
            **ctx.kw,
        )
    if data.has_other:  # for E0c: the KD teacher with no correction at all
        evaluate_teacher(
            f"teachers/{TEACHER}/logits-no-shift",
            TEACHER,
            data,
            other_shift=0.0,
            **ctx.kw,
        )
    print(f"\nchosen shifts: {shifts}")
    table(
        ctx,
        "e0a_teachers",
        show(ctx, "teachers/*/logits"),
        "Each teacher alone, with its logits mapped to our classes",
    )

    recall = {
        d.name: logged(ctx, f"teachers/{d.name}/logits", "test/per_class_recall")[0]
        for d in sorted((ctx.runs / "teachers").iterdir())
        if (d / "logits" / "events.jsonl").exists()
    }
    names = dict(zip(data.classes, data.class_names(), strict=True))  # id -> name
    table(
        ctx,
        "e0a_per_class_recall",
        pd.DataFrame(recall).rename(index=names),
        "Test recall per class (rows) for each teacher (columns)",
    )
    return shifts


def e0b(ctx: Ctx) -> None:
    section("E0b: BASELINE STUDENTS (weighted CE, no teacher)")
    for student in STUDENTS_TO_RUN:
        baseline = Setup(student)
        run(
            f"no-teacher/e0/{student}/sweep/baseline_s0",
            baseline,
            ctx.data,
            ctx.fast,
            **ctx.kw,
        )
        for seed in ctx.seeds:
            run(
                f"no-teacher/e0/{student}/main/baseline_s{seed}",
                baseline,
                ctx.data,
                ctx.main,
                seed=seed,
                **ctx.kw,
            )
        run(
            f"no-teacher/e0/{student}/long/baseline_s0",
            baseline,
            ctx.data,
            ctx.long,
            **ctx.kw,
        )
    table(
        ctx,
        "e0b_baselines",
        show(ctx, "no-teacher/e0/*/*/*"),
        "Baselines: every student without a teacher",
    )


def e0c(ctx: Ctx, shifts: dict) -> tuple[str, partial]:
    section("E0c: HEADS ON THE FROZEN TEACHER, AND WHICH VARIANT TEACHES BEST")
    data = ctx.data
    head_kinds = ["linear", "mlp"] if data.has_other else []
    head_rows = []
    for kind in head_kinds:
        print(f"\n>>> fit a {kind} head on the cached {TEACHER} embeddings")
        head_rows += fit_teacher_head(
            TEACHER,
            data,
            ctx.device,
            kind,
            pretrained=not ctx.smoke,
            max_batches=ctx.eval_cfg.eval_batches if ctx.smoke else None,
            batch_size=ctx.eval_cfg.eval_batch_size,
        )
        evaluate_teacher(
            f"teachers/{TEACHER}/{kind}", TEACHER, data, teacher_head=kind, **ctx.kw
        )
    if head_rows:
        table(
            ctx,
            "e0c_heads",
            pd.DataFrame(head_rows),
            "Heads: one row per head and weight decay (train/val confidence "
            "close = not memorised)",
        )
    table(
        ctx,
        "e0c_teacher_variants",
        show(ctx, f"teachers/{TEACHER}/*"),
        f"{TEACHER} variants alone: mapped logits (with and without shift) vs heads",
    )

    variants = {"logits": dict(other_shift=shifts[TEACHER])}
    variants |= {kind: dict(teacher_head=kind) for kind in head_kinds}
    for variant, teacher_args in variants.items():
        for student in STUDENTS_TO_RUN:
            setup = Setup(
                student, TEACHER, logit=partial(KD, T=4), alpha=0.9, **teacher_args
            )
            run(
                f"{TEACHER}-{variant}/e0c/{student}/sweep/kd_T4_a0.9_s0",
                setup,
                data,
                ctx.fast,
                **ctx.kw,
            )

    check = pd.DataFrame(
        results("*/e0c/*/sweep/*", ctx.runs)
        + results("no-teacher/e0/*/sweep/*", ctx.runs)
    )
    check["teacher"] = check["run"].str.split("/").str[0]
    pivot = check.pivot_table(
        index="teacher", columns="student", values="eval/val/bal_acc"
    )
    table(
        ctx,
        "e0c_student_check",
        pivot,
        "Students taught by each teacher variant (KD T=4, α=0.9, FAST): "
        "val balanced accuracy",
    )
    scores = (
        check[check["teacher"] != "no-teacher"]
        .groupby("teacher")["eval/val/bal_acc"]
        .mean()
    )
    tag = scores.idxmax()  # e.g. "resnet50-linear"
    print(f"\nE1-E3 use the teacher variant {tag} (best mean student val bal. acc.)")
    return tag, partial(
        Setup, teacher=TEACHER, **variants[tag.removeprefix(f"{TEACHER}-")]
    )


def e1(ctx: Ctx, tag: str, S: partial) -> tuple[dict, dict]:
    section("E1: LOGIT KD, T x α x {plain, class-weighted}")
    exps = {
        student: Experiment(
            f"{tag}/e1/{student}",
            {
                f"{'w' if weighted else ''}kd_T{T}_a{alpha}": S(
                    student, logit=partial(KD, T=T), alpha=alpha, weighted_kd=weighted
                )
                for T in (1, 2, 4)
                for alpha in (0.5, 0.9)
                for weighted in (False, True)
            },
            long_check=True,
            seeds=ctx.seeds,
            run_kwargs=ctx.kw,
        )
        for student in STUDENTS_TO_RUN
    }
    for exp in exps.values():
        exp.sweep(ctx.data, ctx.fast)
    table(
        ctx,
        "e1_sweep",
        show(ctx, f"{tag}/e1/*/sweep/*"),
        "E1 sweep (FAST, 1 seed): compare with the e0b sweep baselines",
    )

    sweep = pd.DataFrame(results(f"{tag}/e1/*/sweep/*", ctx.runs))
    setting = sweep["run"].str.split("/").str[-1].str.replace(r"_s\d+$", "", regex=True)
    sweep["KD"] = setting.str.startswith("w").map({False: "plain", True: "weighted"})
    sweep["T, α"] = setting.str.removeprefix("w").str.removeprefix("kd_")
    pivot = sweep.pivot_table(
        index="T, α",
        columns=["student", "KD"],
        values=["eval/val/bal_acc", "eval/val/macro_f1"],
    )
    table(
        ctx,
        "e1_plain_vs_weighted",
        pivot,
        "Does class-weighted KD help? Plain vs weighted at each (T, α)",
    )

    best = {s: exp.confirm(ctx.data, ctx.main, ctx.long)[0] for s, exp in exps.items()}
    print(f"\nbest E1 setting per student: {best}")
    table(
        ctx,
        "e1_main",
        show(ctx, f"{tag}/e1/*/main/*"),
        "E1 best setting per student (MAIN, every seed)",
    )
    return exps, best


def e2(ctx: Ctx, tag: str, S: partial, e1_exps: dict, e1_best: dict) -> tuple:
    section("E2: SWAP THE LOGIT LOSS (best KD vs DKD vs DIST)")
    exps = {
        student: Experiment(
            f"{tag}/e2/{student}",
            {
                e1_best[student]: e1_exps[student].setups[e1_best[student]],
                "dkd_b0.5": S(student, logit=partial(DKD, alpha=0.5, beta=0.5)),
                "dkd_b8": S(student, logit=partial(DKD, alpha=0.5, beta=8.0)),
                "dist": S(student, logit=partial(DIST, beta=1.0, gamma=1.0)),
            },
            top_k=2,
            seeds=ctx.seeds,
            run_kwargs=ctx.kw,
        )
        for student in STUDENTS_TO_RUN
    }
    for exp in exps.values():
        exp.sweep(ctx.data, ctx.fast)
    table(ctx, "e2_sweep", show(ctx, f"{tag}/e2/*/sweep/*"), "E2 sweep (FAST, 1 seed)")
    best = {s: exp.confirm(ctx.data, ctx.main, ctx.long) for s, exp in exps.items()}
    print(f"\nbest two E2 losses per student: {best}")
    table(
        ctx,
        "e2_main",
        show(ctx, f"{tag}/e2/*/main/*"),
        "E2 best two losses per student (MAIN, every seed)",
    )
    return exps, best


def e3(ctx: Ctx, tag: str, S: partial, e2_exps: dict, e2_best: dict) -> None:
    section("E3: EMBEDDING KD")
    data = ctx.data
    stats = teacher_embed_stats(
        TEACHER,
        data,
        ctx.device,
        pretrained=not ctx.smoke,
        max_batches=1 if ctx.smoke else None,
    )
    exps = {}
    for student in STUDENTS_TO_RUN:
        logit_best = e2_best[student][0]
        exps[student] = Experiment(
            f"{tag}/e3/{student}",
            {
                "ce_cos": S(student, embed=EmbedCos),
                "ce_mse": S(student, embed=partial(EmbedMSE, *stats)),
                f"{logit_best}+cos": replace(
                    e2_exps[student].setups[logit_best], embed=EmbedCos
                ),
                "cos_only": S(student, embed=EmbedCos, ce=False),  # label-free + probe
            },
            seeds=ctx.seeds,
            run_kwargs=ctx.kw,
        )
        exps[student].sweep(data, ctx.fast)
    table(ctx, "e3_sweep", show(ctx, f"{tag}/e3/*/sweep/*"), "E3 sweep (FAST, 1 seed)")
    best = {s: exp.confirm(data, ctx.main, ctx.long)[0] for s, exp in exps.items()}
    print(f"\nbest E3 setup per student: {best}")
    table(
        ctx,
        "e3_main",
        show(ctx, f"{tag}/e3/*/main/*"),
        "E3 best setup per student (MAIN, every seed)",
    )

    # Checks: the embedding cosine rises, and the saved student needs no projector
    rows = []
    for student in STUDENTS_TO_RUN:
        for d in sorted((ctx.runs / f"{tag}/e3/{student}/sweep").iterdir()):
            cos = logged(
                ctx, f"{tag}/e3/{student}/sweep/{d.name}", "eval/val/embed_cos"
            )
            rows.append(
                {
                    "student": student,
                    "setup": d.name,
                    "embed_cos per 10 epochs": [round(v, 3) for v in cos],
                }
            )
        model = student_node(student, data).module
        saved = next((ctx.runs / f"{tag}/e3/{student}/main").iterdir()) / "student.pt"
        model.load_state_dict(torch.load(saved, weights_only=True))  # strict
        print(f"{student}: the saved student loads without the projector")
    table(
        ctx,
        "e3_embed_cos",
        pd.DataFrame(rows),
        "Val cosine between projected student and teacher embeddings (should rise)",
    )


def summary(ctx: Ctx) -> None:
    section("RESULTS: every training run, mean / std / count over seeds")
    df = pd.DataFrame(results(runs=ctx.runs))
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
    out.to_csv(ctx.runs / "summary.csv")
    table(ctx, "summary", out, "All runs (read the main stage for reported numbers)")

    # Val balanced accuracy per epoch on the MAIN budget, averaged over seeds
    fig, axes = plt.subplots(
        1, len(STUDENTS_TO_RUN), squeeze=False, figsize=(6 * len(STUDENTS_TO_RUN), 4)
    )
    main_runs = df[df["stage"] == "main"]
    for ax, student in zip(axes[0], STUDENTS_TO_RUN, strict=True):
        for config, group in main_runs[main_runs["student"] == student].groupby(
            ["teacher", "experiment", "config"]
        ):
            curves = [logged(ctx, r, "eval/val/bal_acc") for r in group["run"]]
            n = min(map(len, curves))
            mean = [sum(c[i] for c in curves) / len(curves) for i in range(n)]
            ax.plot(mean, label="/".join(config))
        ax.set(title=student, xlabel="epoch", ylabel="val balanced accuracy")
        ax.legend(fontsize=7)
    figure(ctx, "val_curves_main", fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=["auto", "smoke", "local", "full"],
        default="auto",
        help="auto: full on a CUDA GPU, local otherwise",
    )
    ctx = setup(parser.parse_args().mode)
    overview(ctx)
    sanity_checks(ctx)
    shifts = e0a(ctx)
    e0b(ctx)
    tag, S = e0c(ctx, shifts)
    e1_exps, e1_best = e1(ctx, tag, S)
    e2_exps, e2_best = e2(ctx, tag, S, e1_exps, e1_best)
    e3(ctx, tag, S, e2_exps, e2_best)
    summary(ctx)
    section("DONE")
    print(f"tables and plots: {ctx.report.relative_to(REPO)}/")
    for path in sorted(ctx.report.iterdir()):
        print(f"  {path.name}")
    print(f"every run (settings, logs, weights): {ctx.runs.relative_to(REPO)}/")


if __name__ == "__main__":  # also keeps DataLoader worker processes from rerunning it
    main()
