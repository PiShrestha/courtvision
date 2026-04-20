#!/bin/bash
# canary: verify torch + ultralytics work on a V100 via the generic gpu
# partition under the biocomplexity account (different allocation from the
# saturated gpu-a6000 / cs6770_sp26 slot).
#
# excludes udc-an33* nodes per the torch/cuDNN mismatch documented in
# CLAUDE.local.md (those are a6000 nodes anyway, but belt and braces).

#SBATCH --job-name=cv_canary_a100
#SBATCH --output=logs/canary_a100_%j.out
#SBATCH --error=logs/canary_a100_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
# tiny ultralytics sanity: load yolov8n + one forward pass on a zeros tensor
from ultralytics import YOLO
import numpy as np
model = YOLO("yolov8n.pt")
img = np.zeros((480, 640, 3), dtype=np.uint8)
res = model(img, verbose=False)
print("ultralytics load + forward ok: detected", len(res[0].boxes or []), "boxes on zeros image")
PY
