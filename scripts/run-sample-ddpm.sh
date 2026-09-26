#!/usr/bin/env bash
set -euo pipefail

# Sample with DDPM from any directory. Optionally pass --model-path /path/to/checkpoint.pt.
# Architecture defaults match scripts/run.sh; override them below or via CLI.
# Use the existing project environment without syncing dependencies on each run.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

uv run --no-sync --project "$SCRIPT_DIR/.." python "$SCRIPT_DIR/diffusion_sample.py" \
  --model-path "$SCRIPT_DIR/../checkpoints/cifar10-20260925-1/00000768_00010.136794_checkpoint.pt" \
  --image-size 32 \
  --attention-resolutions 16 8 \
  --device cuda \
  --res-blocks 3 \
  --num-heads 4 \
  --timesteps 1000 \
  --model-channels 128 \
  --embedding-channels 512 \
  --seed 114514 \
  --channel-mult 1 2 4 \
  --sample ddpm \
  "$@"
