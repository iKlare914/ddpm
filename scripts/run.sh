#!/usr/bin/env bash
set -euo pipefail

uv run python diffusion_train.py \
  --image-size 32 \
  --epochs 16 \
  --batch-size 64 \
  --attention-resolutions 16 8 \
  --device cuda \
  --res-block 3 \
  --num-heads 4 \
  --timesteps 1000 \
  --log-sample \
  --model-channel 128 \
  --embedding-channel 512 \
  --seed 114514 \
  --channel-mult 1 2 4 \
  --wandb-mode online \
  --save-interval-epoch 4 \
  --save-dir ../checkpoints/cifar10-20260924-1