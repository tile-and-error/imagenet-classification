#!/bin/bash
#
# Runs one KD stage (e0, e1, e2 or e3) on one GPU. Submit from the repository root:
#     sbatch experiments/kd/kd.sbatch e0
# E1-E3 use the choices of the stages before them, so run them in order; to queue
# all four at once, each waiting for the last: bash experiments/kd/submit.sh
# Output: joboutput_<job>.out. Tables and plots: runs/<dataset>/<split>/report/.
# If a job stops early, submit it again: finished runs are skipped.
#
#SBATCH --job-name=taelwfm-kd     # Job name for tracking
#SBATCH --partition=eagle  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=6       # Number of CPU threads used by your job
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # Number of GPUs to use
#SBATCH --mem=60000               # 60 GB RAM
#SBATCH --time=2-00:00:00         # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

STAGE=${1:?usage: sbatch experiments/kd/kd.sbatch e0|e1|e2|e3}

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load CUDA

## Execute your program(s) ##
cd /dcs/23/u5507039/cs407/imagenet-classification
source .venv/bin/activate   # the venv made in setup, with requirements.txt installed

# -u: print straight away rather than buffered. --mode full stops at once without a GPU.
python -u "experiments/kd/${STAGE}.py" --mode full
