#!/bin/bash
# Apply the SM120 sparse-MLA contract fixes inside a running container.
# Idempotent; run once per container after the vllm-exl3 install.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

for p in "$HERE"/patches/*.py; do
    echo "applying $(basename "$p")"
    python3 "$p"
done

# The .cu widening only takes effect if the prebuilt JIT module is evicted.
JIT=/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so
if [ -f "$JIT" ]; then
    rm -f "$JIT"
    echo "cleared prebuilt sparse_mla_sm120 JIT cache (will rebuild from patched source)"
fi

python3 - <<'PY'
import importlib
import vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse  # noqa: F401 compile check
print("flashinfer_sparse imports OK")
PY
echo "sm120-sparse-fix applied"
