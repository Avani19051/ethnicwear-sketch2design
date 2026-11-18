#!/usr/bin/env bash
# Fetch the original BicycleGAN code (junyanz/BicycleGAN, BSD licence) at a pinned commit.
# It is not copied into this repo; this project imports it.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=40b9d52c27b9831f56c1c7c7a6ddde8bc9149067
if [ ! -d third_party/BicycleGAN ]; then
  git clone https://github.com/junyanz/BicycleGAN third_party/BicycleGAN
fi
git -C third_party/BicycleGAN checkout -q "$COMMIT"
pip install -q -r requirements.txt
echo "BicycleGAN ready at third_party/BicycleGAN ($COMMIT)"
