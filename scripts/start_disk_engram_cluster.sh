#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE="${DISK_ENGRAM_PROFILE:-$RECIPE_ROOT/profiles/tp4.env}"
ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 head|worker" >&2
  exit 2
fi
if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: disk-Engram profile not found: $PROFILE" >&2
  exit 2
fi

# Preserve caller overrides over profile defaults.
declare -A _CALLER_ENV=()
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
  _CALLER_ENV["$key"]="$value"
done < <(env)

set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a
for key in "${!_CALLER_ENV[@]}"; do
  printf -v "$key" '%s' "${_CALLER_ENV[$key]}"
  export "$key"
done
unset _CALLER_ENV key value

export IMAGE="${IMAGE:-deepseek-v41-exl3:tp4}"
export VLLM_ENGRAM_DISK_BACKED=1
if [[ -z "${VLLM_ENGRAM_MODEL_DIR:-}" ]]; then
  if [[ -n "${MODEL:-}" && "$MODEL" == /* ]]; then
    export VLLM_ENGRAM_MODEL_DIR="$MODEL"
  else
    echo "ERROR: set VLLM_ENGRAM_MODEL_DIR=/models/<snapshot> (or MODEL to that absolute path)." >&2
    exit 2
  fi
fi

exec "$SCRIPT_DIR/start_cluster.sh" "$ROLE"
