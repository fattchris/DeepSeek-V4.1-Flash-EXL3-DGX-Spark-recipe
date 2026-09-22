"""Patch: avoid flashinfer's persistent_topk on SM12x when it would oversubscribe.

At large `max_model_len` the sparse indexer's logits stride grows with context, and
flashinfer's `persistent_topk` derives `ctas_per_group = ceil(stride / max_chunk)`
from it. On GB10 (SM121) the device allows only 99 KiB smem/block, so the launch
needs 90 CTAs where only 48 fit, and the `FilteredTopK` fallback needs >=128 KiB per
block -- which this device cannot provide. Result: engine init fails outright.

    RuntimeError: launch_persistent_topk, topk.cu:138, persistent_topk would
    oversubscribe and the FilteredTopK fallback requires >=128KB smem per block
    (have 101376). total_ctas=90 > num_sms*occupancy=48
    (TopK=512, vec_size=4, ctas_per_group=90, smem=48688).

`sparse_attn_indexer.py` already has a third branch -- `ops.top_k_per_row_decode` --
that does not use the persistent kernel. The selection logic reaches it only when
`use_coherent_topk` and `use_persistent_topk` are both false. `use_persistent_topk`
is currently unconditional on CUDA, so this patch adds a device-capability guard:
on SM121 (GB10), take the per-row path instead.

This is a *fallback selection* change, not a kernel change: `top_k_per_row_decode`
is the established non-persistent implementation and preserves the requested K.

Usage (inside the container, every node):
    python3 persistent_topk_sm12x_fix.py            # install
    python3 persistent_topk_sm12x_fix.py --probe    # show the current selection logic
    python3 persistent_topk_sm12x_fix.py --undo     # restore the backup

RESTART REQUIRED: this patches a file that is imported at engine start.

PATCH ROOT: defaults to /usr/local/lib/python3.12/dist-packages/vllm; override with
VLLM_PKG_ROOT=/path/to/site-packages/vllm.
"""
import ast
import os
import shutil
import sys

ROOT = os.environ.get("VLLM_PKG_ROOT") or "/usr/local/lib/python3.12/dist-packages/vllm"
TARGET = os.path.join(ROOT, "model_executor/layers/sparse_attn_indexer.py")
BAK = ".pre-persistent-topk-sm12x"

MARK = "_PERSISTENT_TOPK_SM12X"

OLD = """        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )"""

NEW = """        use_persistent_topk = (
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            # _PERSISTENT_TOPK_SM12X: flashinfer's persistent_topk derives its CTA
            # count from the logits stride, which grows with context. On SM121 the
            # device allows only 99 KiB smem/block, so at large max_model_len the
            # launch needs more CTAs than fit and the FilteredTopK fallback cannot
            # be satisfied (it needs >=128 KiB/block). Engine init then fails with
            # "persistent_topk would oversubscribe". Take the per-row path instead;
            # it is the established non-persistent implementation and preserves K.
            and not current_platform.is_device_capability(121)
        )"""


def probe() -> int:
    s = open(TARGET).read()
    print("=" * 20, TARGET)
    for i, ln in enumerate(s.split("\n"), 1):
        if "use_persistent_topk" in ln or "use_cooperative_topk" in ln or MARK in ln:
            print(f"{i:5d}  {ln}")
    print()
    print("guard installed:", MARK in s)
    return 0


def install() -> int:
    s = open(TARGET).read()
    if MARK in s:
        print("already installed")
        return 0
    if OLD not in s:
        print("ERROR: selection logic not found; the file has diverged", file=sys.stderr)
        return 2
    new = s.replace(OLD, NEW, 1)
    ast.parse(new)                       # validate BEFORE writing
    if not os.path.exists(TARGET + BAK):
        shutil.copy2(TARGET, TARGET + BAK)
    open(TARGET, "w").write(new)
    print("installed:", MARK)
    print("backup:", TARGET + BAK)
    print("RESTART REQUIRED: the indexer is imported at engine start.")
    return 0


def undo() -> int:
    if os.path.exists(TARGET + BAK):
        shutil.copy2(TARGET + BAK, TARGET)
        os.remove(TARGET + BAK)
        print("restored", TARGET)
    else:
        print("no backup at", TARGET + BAK)
    return 0


if __name__ == "__main__":
    if "--undo" in sys.argv:
        raise SystemExit(undo())
    if "--probe" in sys.argv:
        raise SystemExit(probe())
    raise SystemExit(install())
