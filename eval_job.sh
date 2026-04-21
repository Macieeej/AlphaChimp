#!/bin/bash
#SBATCH --job-name=alphachimp-test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks-per-node=4
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
CHECKPOINT="$HOME/AlphaChimp/work_dirs/alphachimp_res576/best_mAP_overall_iter_7000.pth"
NGPU=4
PORT=25525

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "PWD: $(pwd)"
echo "CONFIG=$CONFIG"
echo "CHECKPOINT=$CHECKPOINT"
echo "NGPU=$NGPU"

test -f "$CONFIG"
test -f "$CHECKPOINT"

CMD=(
  python -m torch.distributed.launch
  --nnodes=1
  --node_rank=0
  --master_addr=127.0.0.1
  --nproc_per_node="$NGPU"
  --master_port="$PORT"
  tools/test.py
  "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --launcher pytorch
)

printf 'Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
