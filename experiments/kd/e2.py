"""E2: swap the logit loss: the best E1 setting vs DKD vs DIST. Needs e0.py and
e1.py. From the repository root:

    python -u experiments/kd/e2.py [--mode smoke|local|full]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.kd import common, parts  # noqa: E402


def e2(ctx: common.Ctx) -> None:
    """E2: is a different logit loss better than E1's best KD? DKD and DIST against
    it; the best two per student are confirmed on MAIN and saved for E3."""
    common.section(ctx, "E2: SWAP THE LOGIT LOSS (best KD vs DKD vs DIST)")
    tag, S = common.chosen_teacher(ctx)
    e1_best = common.load_decision(ctx, "e1_best", "e1.py")
    assert isinstance(e1_best, dict)
    exps = {
        student: parts.Experiment(
            f"{tag}/e2/{student}",
            common.e2_setups(S, student, e1_best[student]),
            top_k=2,
            seeds=ctx.seeds,
            run_kwargs=ctx.kw,
        )
        for student in common.STUDENTS_TO_RUN
    }
    # Sweep each candidate on FAST, then confirm the best two on MAIN
    for exp in exps.values():
        exp.sweep(ctx.data, ctx.fast)
    common.table(
        ctx,
        "e2_sweep",
        common.show(ctx, f"{tag}/e2/*/sweep/*"),
        "E2 sweep (FAST, 1 seed)",
    )
    best = {s: exp.confirm(ctx.data, ctx.main, ctx.long) for s, exp in exps.items()}
    common.save_decision(ctx, "e2_best", best)
    common.table(
        ctx,
        "e2_main",
        common.show(ctx, f"{tag}/e2/*/main/*"),
        "E2 best two losses per student (MAIN, every seed)",
    )


def main() -> None:
    with common.stage("E2", ["*/e2"]) as ctx:
        e2(ctx)
        common.summary(ctx)


if __name__ == "__main__":  # also keeps DataLoader worker processes from rerunning it
    main()
