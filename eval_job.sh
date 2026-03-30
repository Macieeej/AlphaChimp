#!/bin/bash
#SBATCH --job-name=alphachimp-test
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

CONFIG="$HOME/AlphaChimp/configs/alphachimp/alphachimp_res576.py"
CHECKPOINT="$HOME/AlphaChimp/work_dirs/alphachimp/alphachimp_res576.pth"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "PWD: $(pwd)"
echo "CONFIG=$CONFIG"
echo "CHECKPOINT=$CHECKPOINT"

test -f "$CONFIG"
test -f "$CHECKPOINT"

CMD=(
  python tools/test.py
  "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --launcher none
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
