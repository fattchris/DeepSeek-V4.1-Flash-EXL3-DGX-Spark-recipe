#!/usr/bin/env bash
# Build the TP4 serving image (deepseek-v41-exl3:tp4) on this Spark.
#
# Three layers, each reused if already present:
#   deepseek-v41-exl3:spark        locked base  (Dockerfile.spark, runtime.lock.json)
#   deepseek-v41-exl3:disk-engram  + disk-backed Engram overlay (Dockerfile.disk-engram)
#   deepseek-v41-exl3:tp4          + MoE kernels and every TP4 fix (Dockerfile.tp4)
#
# Run on every Spark (or build once and `docker save | ssh ... docker load`).
# REBUILD=1 forces all three layers.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

BASE_IMAGE_TAG="${BASE_IMAGE_TAG:-deepseek-v41-exl3:spark}"
DISK_IMAGE_TAG="${DISK_ENGRAM_IMAGE:-deepseek-v41-exl3:disk-engram}"
TP4_IMAGE_TAG="${TP4_IMAGE:-deepseek-v41-exl3:tp4}"
P2B_CB="${P2B_CB:-2}"   # 2 = mul1 codebook, which is what the 4.75bpw pack uses

have() { docker image inspect "$1" >/dev/null 2>&1 && ! is_true "${REBUILD:-0}"; }

if have "$BASE_IMAGE_TAG"; then
  echo "reuse $BASE_IMAGE_TAG"
else
  IMAGE="$BASE_IMAGE_TAG" bash "$RECIPE_ROOT/scripts/build_runtime.sh"
fi

if have "$DISK_IMAGE_TAG"; then
  echo "reuse $DISK_IMAGE_TAG"
else
  IMAGE="$BASE_IMAGE_TAG" DISK_ENGRAM_IMAGE="$DISK_IMAGE_TAG" \
    bash "$RECIPE_ROOT/scripts/build_disk_engram_runtime.sh"
fi

docker build \
  -f "$RECIPE_ROOT/Dockerfile.tp4" \
  --build-arg BASE_RUNTIME="$DISK_IMAGE_TAG" \
  --build-arg P2B_CB="$P2B_CB" \
  -t "$TP4_IMAGE_TAG" \
  "$RECIPE_ROOT"

echo
echo "Built $TP4_IMAGE_TAG:"
docker image inspect "$TP4_IMAGE_TAG" --format '  ID={{.Id}}'
