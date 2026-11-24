"""E0: the data, sanity checks, the teachers alone, baseline students, and which
teacher variant teaches best. Run first: E1-E3 use its choices. From the repository
root:

    python -u experiments/kd/e0.py                # full on CUDA, else local
    python -u experiments/kd/e0.py --mode smoke   # a few-minute wiring check

kd_e0_e3.ipynb explains each experiment; kd_plan_E0-E3.md gives the reasoning.
"""

import sys
from functools import partial
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.kd import common, parts  # noqa: E402
from lwfm.core.graph import ExecutionGraph  # noqa: E402
from lwfm.core.losses import LossComposer  # noqa: E402


def overview(ctx: common.Ctx) -> None:
    """Reports the classes, their images per split, and the CE class weights."""
    common.section(ctx, "DATA")
    data = ctx.data
    counts = {
        split: pd.Series([int(r["label"]) for r in data.rows(split)])
        .value_counts()
        .sort_index()
        for split in ("train", "val", "test")
    }
    common.note(
        ctx,
        "data",
        f"{data.num_classes} classes, {data.spec.image_size} px, "
        f"{len(data.other_folders())} source classes in 'other'",
    )
    df = pd.DataFrame(
        {
            "folder": data.classes,
            "name": data.class_names(),
            **counts,
            "CE weight": data.class_weights().numpy().round(3),
        }
    )
    common.table(
        ctx, "data", df, "Images per class and split, with the CE class weights"
    )


def sanity_checks(ctx: common.Ctx) -> None:
    """Checks with random weights, in seconds, before any long run: output shapes,
    the teacher-to-student class mapping, the losses, and that only the student
    gets gradients."""
    common.section(ctx, "SANITY CHECKS (random weights, seconds)")
    data = ctx.data
    batch = next(iter(data.loader("val", 4)))
    for student in common.STUDENTS_TO_RUN:
        setup = parts.Setup(
            student, common.TEACHER, logit=partial(parts.KD, T=4), alpha=0.9
        )
        nodes, losses = setup.build(data, pretrained=False)
        graph = ExecutionGraph(nodes)
        common.note(ctx, f"shapes/{student}", str(graph.preflight(batch)))
    for b in losses:
        common.note(ctx, f"loss/{b.name}", f"weight {b.weight}, reads {b.inputs}")

    # 1. With no shift, mapped teacher logits == a softmax over the split's classes
    keep, other = data.teacher_classes(), data.teacher_other_classes()
    z = torch.randn(4, 1000)
    p_split = z[:, keep + other].softmax(1)
    p = parts.ToStudentClasses(keep, other or None, shift=0.0)({"logits": z})["logits"]
    p = p.softmax(1)
    assert torch.allclose(p[:, : len(keep)], p_split[:, : len(keep)], atol=1e-6)
    if other:
        assert torch.allclose(p[:, -1], p_split[:, len(keep) :].sum(1), atol=1e-6)
    # 2. Identical logits -> ~0 loss
    z, y = torch.randn(8, data.num_classes), torch.randint(0, data.num_classes, (8,))
    common.note(
        ctx,
        "identical logits",
        f"KD {parts.KD()(z, z):.2e}, DKD {parts.DKD()(z, z, y):.2e}, "
        f"DIST {parts.DIST()(z, z):.2e}",
    )
    # 3. Weighted KD with equal weights == plain KD
    z2 = torch.randn(8, data.num_classes)
    equal = torch.ones(data.num_classes)
    assert torch.allclose(parts.KD(weight=equal)(z, z2, y), parts.KD()(z, z2))
    # 4. Gradients reach the student only (the last student's graph and losses)
    total, _ = LossComposer(losses)(batch, graph(batch))
    total.backward()
    assert all(p.grad is None for p in graph.nodes["teacher"].parameters())
    assert all(p.grad is not None for p in graph.nodes["student"].parameters())
    common.note(ctx, "note", "all sanity checks passed")


def test_recall(ctx: common.Ctx, run_name: str) -> list[float]:
    """A run's test recall per class, from its test confusion matrix: each row's
    diagonal (correct) over the row's sum (all images of that class)."""
    tables = common.logged(ctx, run_name, "eval/test/confusion")
    if not tables:  # a run recorded before confusion matrices were logged
        return list(common.logged(ctx, run_name, "test/per_class_recall")[0].values())
    matrix = tables[-1]["matrix"]
    return [row[i] / max(1, sum(row)) for i, row in enumerate(matrix)]


def e0a(ctx: common.Ctx) -> dict:
    """E0a: every teacher alone, with no training. With an "other" class, each
    teacher's "other" shift is tuned on val first. Returns the shift per teacher."""
    common.section(ctx, "E0a: THE TEACHERS ALONE (inference only)")
    data, shifts = ctx.data, {}
    for teacher in parts.TEACHERS:
        if ctx.smoke and teacher == "dinov2_b14":
            continue  # needs network access even with random weights
        shifts[teacher] = None
        if data.has_other:
            common.note(ctx, "note", f"tune the 'other' shift of {teacher} on val")
            rows = parts.tune_other_shift(
                teacher,
                data,
                ctx.device,
                pretrained=not ctx.smoke,
                eval_batches=ctx.eval_cfg.eval_batches,
                batch_size=ctx.eval_cfg.eval_batch_size,
            )
            shifts[teacher] = max(rows, key=lambda r: r["macro_f1"])["shift"]
            common.note(
                ctx, f"shift/{teacher}", f"{shifts[teacher]:.3f} (best val macro-F1)"
            )
            common.table(
                ctx,
                f"e0a_shift_{teacher}",
                pd.DataFrame(rows),
                f"{teacher}: val scores per 'other' shift",
            )
        parts.evaluate_teacher(
            f"teachers/{teacher}/logits",
            teacher,
            data,
            other_shift=shifts[teacher],
            **ctx.kw,
        )
    if data.has_other:  # for E0c: the KD teacher with no correction at all
        parts.evaluate_teacher(
            f"teachers/{common.TEACHER}/logits-no-shift",
            common.TEACHER,
            data,
            other_shift=0.0,
            **ctx.kw,
        )
    common.save_decision(ctx, "shifts", shifts)
    common.table(
        ctx,
        "e0a_teachers",
        common.show(ctx, "teachers/*/logits"),
        "Each teacher alone, with its logits mapped to our classes",
    )

    # Which classes each teacher gets wrong
    recall = {
        d.name: test_recall(ctx, f"teachers/{d.name}/logits")
        for d in sorted((ctx.runs / "teachers").iterdir())
        if (d / "logits" / "events.jsonl").exists()
    }
    common.table(
        ctx,
        "e0a_per_class_recall",
        pd.DataFrame(recall, index=data.class_names()).rename_axis("class"),
        "Test recall per class (rows) for each teacher (columns)",
    )
    return shifts


def e0b(ctx: common.Ctx) -> None:
    """E0b: every student trained with weighted CE and no teacher, on each budget:
    the baselines every KD run is compared with."""
    common.section(ctx, "E0b: BASELINE STUDENTS (weighted CE, no teacher)")
    # Run names are <teacher>/<experiment>/<student>/<budget>/<setup>_s<seed>
    for student in common.STUDENTS_TO_RUN:
        baseline = parts.Setup(student)
        parts.run(
            f"no-teacher/e0/{student}/sweep/baseline_s0",
            baseline,
            ctx.data,
            ctx.fast,
            **ctx.kw,
        )
        for seed in ctx.seeds:
            parts.run(
                f"no-teacher/e0/{student}/main/baseline_s{seed}",
                baseline,
                ctx.data,
                ctx.main,
                seed=seed,
                **ctx.kw,
            )
        parts.run(
            f"no-teacher/e0/{student}/long/baseline_s0",
            baseline,
            ctx.data,
            ctx.long,
            **ctx.kw,
        )
    common.table(
        ctx,
        "e0b_baselines",
        common.show(ctx, "no-teacher/e0/*/*/*"),
        "Baselines: every student without a teacher",
    )


def e0c(ctx: common.Ctx, shifts: dict) -> None:
    """E0c: which version of the KD teacher teaches best: its mapped ImageNet
    logits, or a head trained on its embedding for our classes. Saves the choice
    for E1-E3."""
    common.section(
        ctx, "E0c: HEADS ON THE FROZEN TEACHER, AND WHICH VARIANT TEACHES BEST"
    )
    data = ctx.data
    # 1. Heads on the frozen teacher's embedding (only with an "other" class, where
    # the mapped logits need the shift correction)
    head_kinds = ["linear", "mlp"] if data.has_other else []
    head_rows = []
    for kind in head_kinds:
        common.note(
            ctx, "note", f"fit a {kind} head on the cached {common.TEACHER} embeddings"
        )
        head_rows += parts.fit_teacher_head(
            common.TEACHER,
            data,
            ctx.device,
            kind,
            pretrained=not ctx.smoke,
            max_batches=ctx.eval_cfg.eval_batches if ctx.smoke else None,
            batch_size=ctx.eval_cfg.eval_batch_size,
        )
        parts.evaluate_teacher(
            f"teachers/{common.TEACHER}/{kind}",
            common.TEACHER,
            data,
            teacher_head=kind,
            **ctx.kw,
        )
    if head_rows:
        common.table(
            ctx,
            "e0c_heads",
            pd.DataFrame(head_rows),
            "Heads: one row per head and weight decay (train/val confidence "
            "close = not memorised)",
        )
    common.table(
        ctx,
        "e0c_teacher_variants",
        common.show(ctx, f"teachers/{common.TEACHER}/*"),
        f"{common.TEACHER} variants alone: mapped logits (with and without shift) "
        "vs heads",
    )

    # 2. One short KD run (T = 4, alpha = 0.9) per teacher variant and student
    for variant in ["logits", *head_kinds]:
        S = common.teacher_setup(variant, shifts[common.TEACHER])
        for student in common.STUDENTS_TO_RUN:
            parts.run(
                f"{common.TEACHER}-{variant}/e0c/{student}/sweep/kd_T4_a0.9_s0",
                S(student, logit=partial(parts.KD, T=4), alpha=0.9),
                data,
                ctx.fast,
                **ctx.kw,
            )

    # 3. Compare them with the no-teacher baselines; keep the variant whose students
    # score best on val, on average
    check = pd.DataFrame(
        parts.results("*/e0c/*/sweep/*", ctx.runs)
        + parts.results("no-teacher/e0/*/sweep/*", ctx.runs)
    )
    check["teacher"] = check["run"].str.split("/").str[0]
    pivot = check.pivot_table(
        index="teacher", columns="student", values="eval/val/bal_acc"
    )
    common.table(
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
    common.save_decision(ctx, "teacher_variant", tag)


def main() -> None:
    with common.stage("E0", ["teachers", "no-teacher/e0", "*/e0c"]) as ctx:
        overview(ctx)
        sanity_checks(ctx)
        shifts = e0a(ctx)
        e0b(ctx)
        e0c(ctx, shifts)
        common.summary(ctx)


if __name__ == "__main__":  # also keeps DataLoader worker processes from rerunning it
    main()
