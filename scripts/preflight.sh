#!/usr/bin/env bash
# Pre-launch check, run on the head Spark once every node's container is up:
#   1. the snapshot is complete (every shard named in the index is on disk);
#   2. an NCCL all-reduce across all nodes works.
# Usage: bash scripts/preflight.sh 2|4
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]]; then
  echo "Usage: $0 2|4" >&2
  exit 2
fi

MODEL="$(resolve_model_for_tp "$TP")"
echo "=== Snapshot: $MODEL ==="
docker exec -i "$CONTAINER_NAME" python3 - "$MODEL" <<'PY'
import json, os, sys
root = sys.argv[1]
idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(idx.values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(root, s))]
cfg = json.load(open(os.path.join(root, "config.json")))
q = cfg.get("quantization_config", {})
print(f"tensors {len(idx)}, shards {len(shards)}, missing {len(missing)}")
print("quantization_config:", json.dumps(q))
if missing:
    sys.exit("ERROR: missing shards: " + ", ".join(missing))
PY

if is_true "${SKIP_NCCL_COLLECTIVE:-0}"; then
  echo "SKIP_NCCL_COLLECTIVE=1: skipping the cross-node check." >&2
  exit 0
fi

echo
echo "=== Cross-node NCCL all-reduce ==="
docker exec -i "$CONTAINER_NAME" python /recipe/scripts/cluster_collective.py \
  --tp "$TP" \
  --megabytes "${COLLECTIVE_MEGABYTES:-16}" \
  --warmup "${COLLECTIVE_WARMUP:-2}" \
  --iterations "${COLLECTIVE_ITERATIONS:-5}" \
  --master-port "${COLLECTIVE_MASTER_PORT:-29557}"
