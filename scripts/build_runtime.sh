#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:deepseekv41-flash-0909}"
VLLM_EXL3_REF="${VLLM_EXL3_REF:-3ce1ae08f3e4a9545c58d5ac6456807c702d524a}"
VLLM_EXL3_REPO="${VLLM_EXL3_REPO:-https://github.com/vcruz305/vllm-exl3.git}"
EXLLAMAV3_REF="${EXLLAMAV3_REF:-be57335b087e4f001c5caae061544df3c06ba01e}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"

echo "Building $IMAGE"
echo "  base:        $BASE_IMAGE"
echo "  vllm-exl3:  $VLLM_EXL3_REF"
echo "  vllm-exl3 repo: $VLLM_EXL3_REPO"
echo "  ExLlamaV3:  $EXLLAMAV3_REF"
echo "  CUDA arch:   $TORCH_CUDA_ARCH_LIST"

docker build \
  -f "$RECIPE_ROOT/Dockerfile.spark" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg VLLM_EXL3_REF="$VLLM_EXL3_REF" \
  --build-arg VLLM_EXL3_REPO="$VLLM_EXL3_REPO" \
  --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  -t "$IMAGE" \
  "$RECIPE_ROOT"

echo
echo "Built runtime. Local image identity:"
docker image inspect "$IMAGE" --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}'
