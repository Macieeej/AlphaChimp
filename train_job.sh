#!/bin/bash
#SBATCH --job-name=alphachimp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/AlphaChimp"
mkdir -p logs

source "$HOME/miniforge3/bin/activate"
conda activate alphachimp

export PYTHONPATH="$HOME/AlphaChimp:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RES="${RES:-256}"
NGPU="${NGPU:-1}"
CONFIG="$HOME/AlphaChimp/configs/alphachimp/alphachimp_res${RES}.py"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "PWD: $(pwd)"
echo "RES=$RES"
echo "NGPU=$NGPU"
echo "CONFIG=$CONFIG"
echo "PYTHONPATH=$PYTHONPATH"
echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"

test -f "$CONFIG"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_port=25525 \
  tools/train.py \
  "$CONFIG" \
