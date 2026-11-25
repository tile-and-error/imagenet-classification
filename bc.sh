#!/bin/bash

# This is the Batch Compute script for DCS
# Move this to DCS large /bc

# run bc experiment.yaml cpu
# OR
# run bc experiment.yaml cpu cpu.yaml
#
# The accepted parameters are: cpu, cpu_multi, gpu, gecko, falcon, eagle

USER=$(whoami)
LARGEPATH="/"
DEVICE=$2
TIMESTAMP=$(ts '%Y%m%d-%H%M%S')
SETUP=$3

EXPERIMENT="$LARGEPATH/configs/experiments/$USER/$1"
cp $1 $EXPERIMENT

if [[ $# < 2 ]]; then
    echo "At least two parameters must be provided, the expiremnt file and system"
    exit 1
fi

if [[ $# > 3 ]]; then
    echo "Too many parameters provided"
    exit 1
fi

if [[ $# -eq 2 ]];
    python3.14 "$LARGEPATH/lwfm/integrations/slurm/generate.py" dcs "$LARGEPATH" default "$DEVICE" "$USER" "$TIMESTAMP" "$EXPERIMENT"
else
    python3.14 "$LARGEPATH/lwfm/integrations/slurm/generate.py" dcs "$LARGEPATH" custom "$DEVICE" "$USER" "$TIMESTAMP" "$SETUP"
fi

BATCHFILE="$LARGEPATH/configs/slurmjobs/${USER}_${TIMESTAMP}.sbatch"

ssh kudu-taught

sbatch $BATCHFILE