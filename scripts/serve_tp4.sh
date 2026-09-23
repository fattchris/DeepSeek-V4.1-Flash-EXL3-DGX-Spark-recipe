#!/usr/bin/env bash
# Serve TP4 (4x Spark, TP4 + EP4) from a local snapshot. Run on the head Spark
# after every node is up (scripts/start_disk_engram_cluster.sh).
#   set -a; . profiles/tp4.env; set +a; bash scripts/serve_tp4.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODEL_RESOLVED="$(resolve_model_for_tp 4)"
if [[ "$MODEL_RESOLVED" != /models/* ]]; then
  cat >&2 <<EOF
ERROR: MODEL must be a local snapshot path inside the container (/models/...).
Got: $MODEL_RESOLVED
Download it on every Spark first:
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-4.75bpw
then set MODEL_DIR=/large/models and MODEL=/models/DSV4.1-Flash-EXL3-4.75bpw.
EOF
  exit 2
fi

export MODEL="$MODEL_RESOLVED"
exec "$SCRIPT_DIR/serve.sh" 4
