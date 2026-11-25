"""E1: logit KD, T x alpha x {plain, class-weighted}, with the teacher variant E0
chose. Needs e0.py. From the repository root:

    python -u experiments/kd/e1.py [--mode smoke|local|full]
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.kd import common, parts  # noqa: E402


def e1(ctx: common.Ctx) -> None:
    """E1: logit KD over a grid of T, alpha and plain or class-weighted. Every
    setting gets one short run; the best per student is confirmed on the MAIN
    budget and saved for E2."""
    common.section(ctx, "E1: LOGIT KD, T x α x {plain, class-weighted}")
    tag, S = common.chosen_teacher(ctx)
    exps = {
        student: parts.Experiment(
            f"{tag}/e1/{student}",
            common.e1_setups(S, student),
            long_check=True,
            seeds=ctx.seeds,
            run_kwargs=ctx.kw,
        )
        for student in common.STUDENTS_TO_RUN
    }
    # Sweep: one FAST run per setting, to rank them
    for exp in exps.values():
        exp.sweep(ctx.data, ctx.fast)
    common.table(
        ctx,
        "e1_sweep",
        common.show(ctx, f"{tag}/e1/*/sweep/*"),
        "E1 sweep (FAST, 1 seed): compare with the e0b sweep baselines",
    )

    # Plain vs weighted KD side by side, from the setting names (e.g. wkd_T4_a0.9)
    sweep = pd.DataFrame(parts.results(f"{tag}/e1/*/sweep/*", ctx.runs))
    setting = sweep["run"].str.split("/").str[-1].str.replace(r"_s\d+$", "", regex=True)
    sweep["KD"] = setting.str.startswith("w").map({False: "plain", True: "weighted"})
    sweep["T, α"] = setting.str.removeprefix("w").str.removeprefix("kd_")
    pivot = sweep.pivot_table(
        index="T, α",
        columns=["student", "KD"],
        values=["eval/val/bal_acc", "eval/val/macro_f1"],
    )
    common.table(
        ctx,
        "e1_plain_vs_weighted",
        pivot,
        "Does class-weighted KD help? Plain vs weighted at each (T, α)",
    )

    # Confirm: the best setting per student on MAIN (every seed) and once on LONG
    best = {s: exp.confirm(ctx.data, ctx.main, ctx.long)[0] for s, exp in exps.items()}
    common.save_decision(ctx, "e1_best", best)
    common.table(
        ctx,
        "e1_main",
        common.show(ctx, f"{tag}/e1/*/main/*"),
        "E1 best setting per student (MAIN, every seed)",
    )


def main() -> None:
    with common.stage("E1", ["*/e1"]) as ctx:
        e1(ctx)
        common.summary(ctx)


if __name__ == "__main__":  # also keeps DataLoader worker processes from rerunning it
    main()
