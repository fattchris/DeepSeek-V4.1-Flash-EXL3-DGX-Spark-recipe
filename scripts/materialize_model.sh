#!/usr/bin/env bash
# Download the model for a topology onto a large local disk. Run on EVERY Spark.
# Usage: bash scripts/materialize_model.sh 2|4 /absolute/path/to/snapshot
#
#   TP4 -> vcruz305/DSV4.1-Flash-EXL3-4.75bpw        (~460 GB)
#   TP2 -> vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw   (~450 GB)
#
# The revision comes from runtime.lock.json; MODEL_REVISION overrides it.
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
DEST="${2:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]] || [[ -z "$DEST" ]]; then
  echo "Usage: $0 2|4 /absolute/path/to/model-snapshot" >&2
  exit 2
fi
case "$DEST" in
  /*) ;;
  *) echo "ERROR: destination must be an absolute path" >&2; exit 2 ;;
esac

# The lock's repo, not MODEL: MODEL is the in-container path used at serve time.
if [[ "$TP" == "4" ]]; then REPO="$MODEL_TP4"; else REPO="$MODEL_TP2"; fi
REVISION="${MODEL_REVISION:-$(python3 "$LOCK_TOOL" get "models.tp$TP.revision")}"
MIN_FREE_GIB="${DOWNLOAD_MIN_FREE_GIB:-500}"

mkdir -p "$DEST"

# Keep Hugging Face/Xet staging on the same large filesystem as the model.
export HF_HOME="${MODEL_HF_HOME:-$(dirname "$DEST")/.hf-dsv41-cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
mkdir -p "$HF_HOME"

FREE_GIB="$(df -PB1G "$DEST" | awk 'NR==2 {print $4}')"
if (( FREE_GIB < MIN_FREE_GIB )); then
  echo "ERROR: $DEST has ${FREE_GIB} GiB free; need ${MIN_FREE_GIB} GiB (override with DOWNLOAD_MIN_FREE_GIB)." >&2
  exit 2
fi

echo "Repo:        $REPO"
echo "Revision:    ${REVISION:-main}"
echo "Destination: $DEST  (${FREE_GIB} GiB free)"
echo

ARGS=("$REPO" --local-dir "$DEST")
[[ -n "$REVISION" ]] && ARGS+=( --revision "$REVISION" )
if command -v hf >/dev/null 2>&1; then
  hf download "${ARGS[@]}"
else
  huggingface-cli download "${ARGS[@]}"
fi

python3 - "$DEST" <<'PY'
import json, os, sys
root = sys.argv[1]
idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
shards = sorted(set(idx.values()))
missing = [s for s in shards if not os.path.isfile(os.path.join(root, s))]
if missing:
    sys.exit("ERROR: download incomplete, missing: " + ", ".join(missing))
print(f"OK: {len(idx)} tensors in {len(shards)} shards")
PY

echo
echo "Set on EVERY Spark:"
echo "  MODEL_DIR=$(dirname "$DEST")"
echo "  MODEL=/models/$(basename "$DEST")"
