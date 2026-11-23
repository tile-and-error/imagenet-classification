#!/bin/bash
#
# Runs every KD experiment (E0-E3) on one GPU. Submit from the repository root:
#     sbatch experiments/kd/kd_simple_notebook.sbatch
# Output: joboutput_<job>.out. Tables and plots: runs/<dataset>/<split>/report/.
# If a job stops early, submit it again: finished runs are skipped.
#
#SBATCH --job-name=taelwfm-1    # Job name for tracking
#SBATCH --partition=eagle  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=8        # Number of CPU threads used by your job
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1              # Number of GPUs to use
#SBATCH --mem=60000               # 60 GB RAM
#SBATCH --time=2-00:00:00         # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
module load CUDA

## Execute your program(s) ##
cd /dcs/23/u5507039/cs407/imagenet-classification
source .venv/bin/activate   # the venv made in setup, with requirements.txt installed

# -u: print straight away rather than buffered. --mode full stops at once without a GPU.
python -u experiments/kd/py_kd_e0_e3.py --mode full
