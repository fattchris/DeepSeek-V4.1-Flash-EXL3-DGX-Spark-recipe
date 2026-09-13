#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

LOCK_TOOL="$RECIPE_ROOT/scripts/runtime_lock.py"
python3 "$LOCK_TOOL" validate >/dev/null

LOCK_BASE_IMAGE="$(python3 "$LOCK_TOOL" get base_image.ref)"
LOCK_VLLM_EXL3_REPO="$(python3 "$LOCK_TOOL" get vllm_exl3.repo)"
LOCK_VLLM_EXL3_REF="$(python3 "$LOCK_TOOL" get vllm_exl3.commit)"
LOCK_EXLLAMAV3_REF="$(python3 "$LOCK_TOOL" get exllamav3.commit)"
LOCK_CUDA_ARCH="$(python3 "$LOCK_TOOL" get torch_cuda_arch_list)"

BASE_IMAGE="${BASE_IMAGE:-$LOCK_BASE_IMAGE}"
VLLM_EXL3_REPO="${VLLM_EXL3_REPO:-$LOCK_VLLM_EXL3_REPO}"
VLLM_EXL3_REF="${VLLM_EXL3_REF:-$LOCK_VLLM_EXL3_REF}"
EXLLAMAV3_REF="${EXLLAMAV3_REF:-$LOCK_EXLLAMAV3_REF}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$LOCK_CUDA_ARCH}"

check_lock_override() {
  local name="$1" actual="$2" locked="$3"
  if [[ "$actual" != "$locked" ]]; then
    if ! is_true "${ALLOW_RUNTIME_OVERRIDE:-0}"; then
      echo "ERROR: $name=$actual differs from runtime.lock.json ($locked)." >&2
      echo "Set ALLOW_RUNTIME_OVERRIDE=1 only for an intentional experimental build." >&2
      exit 2
    fi
    echo "WARNING: ALLOW_RUNTIME_OVERRIDE=1; $name differs from runtime.lock.json." >&2
  fi
}

check_lock_override BASE_IMAGE "$BASE_IMAGE" "$LOCK_BASE_IMAGE"
check_lock_override VLLM_EXL3_REPO "$VLLM_EXL3_REPO" "$LOCK_VLLM_EXL3_REPO"
check_lock_override VLLM_EXL3_REF "$VLLM_EXL3_REF" "$LOCK_VLLM_EXL3_REF"
check_lock_override EXLLAMAV3_REF "$EXLLAMAV3_REF" "$LOCK_EXLLAMAV3_REF"
check_lock_override TORCH_CUDA_ARCH_LIST "$TORCH_CUDA_ARCH_LIST" "$LOCK_CUDA_ARCH"

echo "Building $IMAGE from runtime.lock.json"
echo "  base:            $BASE_IMAGE"
echo "  vllm-exl3 repo:  $VLLM_EXL3_REPO"
echo "  vllm-exl3 ref:   $VLLM_EXL3_REF"
echo "  ExLlamaV3 ref:   $EXLLAMAV3_REF"
echo "  CUDA arch:       $TORCH_CUDA_ARCH_LIST"

docker build \
  -f "$RECIPE_ROOT/Dockerfile.spark" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg VLLM_EXL3_REPO="$VLLM_EXL3_REPO" \
  --build-arg VLLM_EXL3_REF="$VLLM_EXL3_REF" \
  --build-arg EXLLAMAV3_REF="$EXLLAMAV3_REF" \
  --build-arg CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  -t "$IMAGE" \
  "$RECIPE_ROOT"

echo
echo "Built runtime. Local image identity:"
docker image inspect "$IMAGE" --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}'
