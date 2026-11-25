#!/bin/bash
# Queues the KD stages as separate jobs, each starting only once the one before it
# has finished successfully. From the repository root:
#     bash experiments/kd/submit.sh            # e0 e1 e2 e3
#     bash experiments/kd/submit.sh e2 e3      # just these, e.g. after e1 is done
set -euo pipefail

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(e0 e1 e2 e3)
previous=""
for stage in "${stages[@]}"; do
    dependency=${previous:+--dependency=afterok:$previous}
    previous=$(sbatch --parsable --job-name="kd-$stage" $dependency \
        experiments/kd/kd.sbatch "$stage")
    echo "$stage: job $previous${dependency:+ (after ${dependency#*:})}"
done
