#!/usr/bin/env bash
# Serve TP2 (2x Spark, TP2 + EP2 by default) from a local snapshot. Run on the
# head Spark after both nodes are up (scripts/start_disk_engram_cluster.sh).
#   set -a; . profiles/tp2.env; set +a; bash scripts/serve_tp2.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

MODEL_RESOLVED="$(resolve_model_for_tp 2)"
if [[ "$MODEL_RESOLVED" != "/models" && "$MODEL_RESOLVED" != /models/* ]]; then
  cat >&2 <<EOF
ERROR: MODEL must be a local snapshot path inside the container (/models/...).
Got: $MODEL_RESOLVED
Download it on both Sparks first:
  bash scripts/materialize_model.sh 2 /large/models/DSV4.1-Flash-SAGE-EXL3-3.30bpw
then set MODEL_DIR=/large/models and MODEL=/models/DSV4.1-Flash-SAGE-EXL3-3.30bpw.
EOF
  exit 2
fi

export MODEL="$MODEL_RESOLVED"
exec "$SCRIPT_DIR/serve.sh" 2
