#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 head|worker" >&2
  exit 2
fi

require_env HEAD_IP
require_env NODE_IP

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' is not present. Run ./scripts/build_runtime.sh first." >&2
  exit 2
fi

# Replace a stale local cluster container on this host.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

DOCKER_ARGS=(
  run -d --rm
  --name "$CONTAINER_NAME"
  --gpus all
  --network host
  --ipc host
  --device "${RDMA_DEVICE:-/dev/infiniband}"
  --ulimit memlock=-1
  --ulimit stack=67108864
  --entrypoint ray
  -v "$RECIPE_ROOT:/recipe:ro"
  -v "$HF_HOME:/root/.cache/huggingface"
  -e HF_HOME=/root/.cache/huggingface
  -e VLLM_HOST_IP="$NODE_IP"
  -e RAY_DEDUP_LOGS=0
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e NCCL_NET=${NCCL_NET:-ib}
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1}
  -e NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-2}
  -e NCCL_IB_ADDR_FAMILY=${NCCL_IB_ADDR_FAMILY:-AF_INET}
  -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
  -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp1s0f1np1}
  -e NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
  -e NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ARGS+=( -e HF_TOKEN="$HF_TOKEN" )
fi

if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR '$MODEL_DIR' does not exist on this node." >&2
    exit 2
  fi
  DOCKER_ARGS+=( -v "$MODEL_DIR:/models" )
fi

if [[ "$ROLE" == "head" ]]; then
  if [[ "$NODE_IP" != "$HEAD_IP" ]]; then
    echo "ERROR: head NODE_IP ($NODE_IP) must equal HEAD_IP ($HEAD_IP)." >&2
    exit 2
  fi
  RAY_ARGS=(start --head --node-ip-address="$NODE_IP" --port="$RAY_PORT" --num-gpus=1 --disable-usage-stats --block)
else
  RAY_ARGS=(start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$NODE_IP" --num-gpus=1 --disable-usage-stats --block)
fi

echo "Starting $ROLE container '$CONTAINER_NAME' on $NODE_IP"
docker "${DOCKER_ARGS[@]}" "$IMAGE" "${RAY_ARGS[@]}"

echo "Container started. Recent logs:"
sleep 2
docker logs --tail 30 "$CONTAINER_NAME" || true
