"""E3: embedding KD, alone and added to the best E2 logit loss. Needs e0.py, e1.py
and e2.py. From the repository root:

    python -u experiments/kd/e3.py [--mode smoke|local|full]
"""

import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.kd import common, parts  # noqa: E402


def e3(ctx: common.Ctx) -> None:
    """E3: distilling the teacher's embedding: on its own with CE, added to E2's best
    logit loss, and label-free. The best per student is confirmed on MAIN."""
    common.section(ctx, "E3: EMBEDDING KD")
    data = ctx.data
    tag, S = common.chosen_teacher(ctx)
    e1_best = common.load_decision(ctx, "e1_best", "e1.py")
    e2_best = common.load_decision(ctx, "e2_best", "e2.py")
    assert isinstance(e1_best, dict) and isinstance(e2_best, dict)
    # EmbedMSE standardises the teacher embedding with its train-set mean and std
    stats = parts.teacher_embed_stats(
        common.TEACHER,
        data,
        ctx.device,
        pretrained=not ctx.smoke,
        max_batches=1 if ctx.smoke else None,
    )
    exps = {}
    for student in common.STUDENTS_TO_RUN:
        logit_best = e2_best[student][0]
        logit_setup = common.e2_setups(S, student, e1_best[student])[logit_best]
        exps[student] = parts.Experiment(
            f"{tag}/e3/{student}",
            {
                "ce_cos": S(student, embed=parts.EmbedCos),
                "ce_mse": S(student, embed=partial(parts.EmbedMSE, *stats)),
                f"{logit_best}+cos": replace(logit_setup, embed=parts.EmbedCos),
                # Label-free: no CE; scored afterwards with a linear probe
                "cos_only": S(student, embed=parts.EmbedCos, ce=False),
            },
            seeds=ctx.seeds,
            run_kwargs=ctx.kw,
        )
        exps[student].sweep(data, ctx.fast)
    common.table(
        ctx,
        "e3_sweep",
        common.show(ctx, f"{tag}/e3/*/sweep/*"),
        "E3 sweep (FAST, 1 seed)",
    )
    best = {s: exp.confirm(data, ctx.main, ctx.long)[0] for s, exp in exps.items()}
    common.save_decision(ctx, "e3_best", best)
    common.table(
        ctx,
        "e3_main",
        common.show(ctx, f"{tag}/e3/*/main/*"),
        "E3 best setup per student (MAIN, every seed)",
    )

    # Checks: the embedding cosine rises, and the saved student needs no projector
    rows = []
    for student in common.STUDENTS_TO_RUN:
        for d in sorted((ctx.runs / f"{tag}/e3/{student}/sweep").iterdir()):
            cos = common.logged(
                ctx, f"{tag}/e3/{student}/sweep/{d.name}", "eval/val/embed_cos"
            )
            rows.append(
                {
                    "student": student,
                    "setup": d.name,
                    "embed_cos per epoch": [round(v, 3) for v in cos],
                }
            )
        model = parts.student_node(student, data).module
        saved = next((ctx.runs / f"{tag}/e3/{student}/main").iterdir()) / "student.pt"
        model.load_state_dict(torch.load(saved, weights_only=True))  # strict
        common.note(
            ctx, "note", f"{student}: the saved student loads without the projector"
        )
    common.table(
        ctx,
        "e3_embed_cos",
        pd.DataFrame(rows),
        "Val cosine between projected student and teacher embeddings (should rise)",
    )


def main() -> None:
    with common.stage("E3", ["*/e3"]) as ctx:
        e3(ctx)
        common.summary(ctx)


if __name__ == "__main__":  # also keeps DataLoader worker processes from rerunning it
    main()
