#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:?Set DATASET_ROOT to the converted training dataset}"
REPO_ID="${REPO_ID:?Set REPO_ID to the dataset repo ID used during conversion}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/checkpoints}"
RUN_NAME="${RUN_NAME:-station_lite_act_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUT_DIR="$OUTPUT_ROOT/$RUN_NAME"

[[ -d "$DATASET_ROOT" ]] || { echo "Dataset not found: $DATASET_ROOT" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "Run directory already exists: $OUT_DIR" >&2; exit 2; }
python - <<'PY'
import torch
import lerobot
assert torch.cuda.is_available(), "CUDA is not available"
print("CUDA and LeRobot are ready")
PY

echo "Dataset: $DATASET_ROOT"
echo "Repo ID: $REPO_ID"
echo "Run: $OUT_DIR"
echo "Steps: $STEPS; batch: $BATCH_SIZE"

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.eval_split=0.13 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --batch_size="$BATCH_SIZE" --num_workers=4 --steps="$STEPS" --log_freq=100 \
  --eval_steps=2000 --max_eval_samples=0 --env_eval_freq=0 \
  --save_checkpoint=true --save_freq=2000 --seed=42 --wandb.enable=false \
  --output_dir="$OUT_DIR" --job_name="$RUN_NAME"
