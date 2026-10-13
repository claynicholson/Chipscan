#!/bin/bash
# Run the full chipscan flow on one scan on MIT ORCD Engaging (free GPU tier).
#   sbatch run_scan.sbatch /orcd/pool/007/clayn/scans/AFT5_07_C1.png AFT5_07_C1 0.814
# Results land in /orcd/pool/007/clayn/runs/<name>/ (findings.json, maps, corrected pyramid).
#SBATCH -J chipscan
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /orcd/pool/007/clayn/logs/chipscan.%j.log
set -euo pipefail
IMAGE="$1"; NAME="${2:-$(basename "${IMAGE%.*}")}"; PX="${3:-0.814}"
POOL=/orcd/pool/007/clayn
module load miniforge
source activate "$POOL/envs/chipscan"
cd "$POOL/chipscan"
nvidia-smi --query-gpu=name,memory.total --format=csv
time chipscan add "$IMAGE" --name "$NAME" --pixel-size "$PX" --runs "$POOL/runs"
