#!/usr/bin/env bash
# Train BicycleGAN on the prepared sketch|photo pairs.
# Defaults fit a free Kaggle/Colab T4 in roughly 3-4 hours for ~8k pairs.
#   bash scripts/train.sh                     # default
#   EPOCHS=10 bash scripts/train.sh           # quick run to check everything works
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-datasets/ethnicwear}
NAME=${NAME:-ethnicwear_bicycle_gan}
EPOCHS=${EPOCHS:-30}          # epochs at full learning rate, then the same number decaying to 0
python third_party/BicycleGAN/train.py \
  --dataroot "$DATA" \
  --name "$NAME" \
  --model bicycle_gan \
  --direction AtoB \
  --checkpoints_dir checkpoints \
  --load_size 256 --crop_size 256 \
  --input_nc 1 --nz 8 \
  --niter "$EPOCHS" --niter_decay "$EPOCHS" \
  --save_epoch_freq 5 \
  --use_dropout \
  --display_id -1     # no visdom server on Kaggle/Colab; losses still print
