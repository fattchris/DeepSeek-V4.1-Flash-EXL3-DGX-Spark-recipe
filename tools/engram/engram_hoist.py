#!/usr/bin/env python3
"""enghoist.py -- hoist the engram hash + disk lookup out of model.forward (DECIDER_V4_HOIST).

Site: DeepseekV41ModelState.prepare_inputs (nvidia/model_state.py). That hook already runs
eagerly once per real step, before the forward / graph replay, and already owns the engram
lookback window. The model forward then only READS Engram.staged_rows (fixed __init__-time
buffers), so FULL_DECODE_ONLY can capture it with plain torch.cuda.graph.

  python3 enghoist.py              # apply (idempotent, backups *.pre_enghoist)
  python3 enghoist.py --undo       # restore backups
  python3 enghoist.py --fields     # only print the runner-field / route greps (no edits)

Runtime knobs (both read in the worker at ModelState init; default needs NO env, so Ray
worker env drift cannot silently disable it):
  VLLM_ENGRAM_HOIST=0        disable (forward does the lookup itself, as before)
  VLLM_ENGRAM_HOIST_CHECK=1  EAGER BOOTS ONLY (it syncs): compare hoisted hashes with the
                             in-forward hashes for the first 32 forwards, print _ENGHOIST CHECK

Run `fullengram.py --undo` first: eager_break_always cannot work under FULL (there is no
BreakableCUDAGraphCapture object on that path) and the hoist makes it unnecessary.
Apply on ALL FOUR nodes (scripts still --rm: fold into the recipe next to the engram key fix).
"""
import os
import re
import shutil
import sys

V = "/usr/local/lib/python3.12/dist-packages/vllm"
MODEL = f"{V}/models/deepseek_v4_1/nvidia/model.py"
STATE = f"{V}/models/deepseek_v4_1/nvidia/model_state.py"
MARK = "_ENGHOIST"


def backup(p):
    b = p + ".pre_enghoist"
    if not os.path.exists(b):
        shutil.copy2(p, b)


def undo():
    for p in (MODEL, STATE):
        b = p + ".pre_enghoist"
        if os.path.exists(b):
            shutil.copy2(b, p)
            os.remove(b)
            print("restored", p)
        else:
            print("no backup for", p)


def fields():
    # Host-only greps: confirm the names the hoist relies on exist in THIS fork.
    ib = f"{V}/v1/worker/gpu/input_batch.py"
    if os.path.exists(ib):
        s = open(ib).read()
        for name in ("input_ids", "positions", "query_start_loc", "idx_mapping"):
            hit = re.search(rf"^\s+{name}\s*:", s, re.M)
            print(f"RUNNER FIELD InputBatch.{name}: {'OK' if hit else 'MISSING'}")
    else:
        print("RUNNER FIELD: input_batch.py not found at", ib)
    mr = f"{V}/v1/worker/gpu/model_runner.py"
    if os.path.exists(mr):
        for i, line in enumerate(open(mr), 1):
            if "model_state.prepare_inputs" in line or "run_fullgraph" in line or "prepare_dummy_inputs" in line:
                print(f"RUNNER ROUTE model_runner.py:{i}: {line.strip()}")
    # prepare_inputs must be called BEFORE run_fullgraph in execute_model (lower line number,
    # same function). If the fork calls it only on the non-FULL branch, stop and report.


STAGE_METHOD = '''
    @torch.inference_mode()
    def stage_engram_rows(  # _ENGHOIST
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        lookback_token_ids: torch.Tensor,
    ) -> None:
        """Eager, outside any capture/replay: hash + fill every Engram.staged_rows.

        V2 runner only (stateless hash: no slot cache, so no block table / slot
        mapping). Must see the same padded input_ids/positions the forward sees.
        """
        if self.engram_hash is None or not self.engram_hash.ensure_cache():
            return  # profile run (KV unbound): forward skips engram too
        image_mask = image_sentinel_mask(input_ids)
        hashes = self.engram_hash(
            input_ids,
            positions,
            query_start_loc,
            image_mask,
            lookback_token_ids,
            image_sentinel_mask(lookback_token_ids),
            None,
            None,
        )
        gathered = gather_engram_hashes(hashes)
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None:
                engram.prepare_embeddings(gathered[:, engram.layer_hash_index])
        if getattr(self, "_engram_hoist_check", False):
            self._engram_hoist_ref = hashes

    def _enghoist_check(self, engram_hashes: torch.Tensor) -> None:  # _ENGHOIST
        # Host-only no-op unless VLLM_ENGRAM_HOIST_CHECK=1 (eager boots only: it syncs).
        if not getattr(self, "_engram_hoist_check", False):
            return
        n = getattr(self, "_engram_hoist_nchk", 0)
        if n >= 32:
            return
        self._engram_hoist_nchk = n + 1
        ref = getattr(self, "_engram_hoist_ref", None)
        self._engram_hoist_ref = None
        if ref is None:
            print(f"_ENGHOIST CHECK n={engram_hashes.shape[0]} nostage (dummy run)", flush=True)
            return
        same = ref.shape == engram_hashes.shape and bool(torch.equal(ref, engram_hashes))
        print(
            f"_ENGHOIST CHECK n={engram_hashes.shape[0]} equal={same} "
            f"ref={tuple(ref.shape)} fwd={tuple(engram_hashes.shape)}",
            flush=True,
        )
'''


def patch_model():
    s = open(MODEL).read()
    if MARK in s:
        print("model.py already patched")
        return
    # (A) forward guard. Exactly one such line (the decoder-layer one is
    # `if self.engram is not None and engram_hashes is not None:`).
    a = "            if engram_hashes is not None:\n"
    assert s.count(a) == 1, f"forward guard anchor count={s.count(a)} (expected 1)"
    s = s.replace(
        a,
        "            _hoisted = getattr(self, \"_engram_hoisted\", False)  # _ENGHOIST\n"
        "            if _hoisted and engram_hashes is not None:\n"
        "                self._enghoist_check(engram_hashes)\n"
        "            if engram_hashes is not None and not _hoisted:\n",
    )
    # (B) method, inserted before the inner model's embed_input_ids (first one after the
    # engram_hash attribute, i.e. inside the class that owns self.layers/self.engram_hash).
    own = s.index("self.engram_hash: NgramHashState | None = None")
    m = s.index("    def embed_input_ids(", own)
    s = s[:m] + STAGE_METHOD.lstrip("\n") + "\n" + s[m:]
    for need in ("image_sentinel_mask", "gather_engram_hashes", "islice"):
        assert need in s, f"model.py lacks {need}"
    backup(MODEL)
    open(MODEL, "w").write(s)
    print("model.py: forward guard + stage_engram_rows")


def patch_state():
    s = open(STATE).read()
    if MARK in s:
        print("model_state.py already patched")
        return
    a0 = "from typing import Any\n"
    assert s.count(a0) == 1, "import anchor"
    s = s.replace(a0, "import os\n" + a0)
    # (C) init: find the module that owns the hash, arm the static skip flag.
    a1 = "                (self.max_num_reqs, depth), -1, dtype=torch.int32, device=device\n            )\n"
    assert s.count(a1) == 1, "init anchor"
    s = s.replace(
        a1,
        a1
        + "        # _ENGHOIST: static mode, NOT per-step state -- capture/warmup run the\n"
        + "        # forward via prepare_dummy_inputs and must skip the lookup too.\n"
        + "        self._engram_host = None\n"
        + "        if self.lookback_token_ids is not None and os.environ.get(\"VLLM_ENGRAM_HOIST\", \"1\") == \"1\":\n"
        + "            for _m in model.modules():\n"
        + "                _h = getattr(_m, \"engram_hash\", None)\n"
        + "                if _h is not None and hasattr(_m, \"stage_engram_rows\") and not _h.use_slot_cache:\n"
        + "                    _m._engram_hoisted = True\n"
        + "                    _m._engram_hoist_check = os.environ.get(\"VLLM_ENGRAM_HOIST_CHECK\", \"0\") == \"1\"\n"
        + "                    self._engram_host = _m\n"
        + "                    break\n"
        + "            print(f\"_ENGHOIST armed={self._engram_host is not None}\", flush=True)\n",
    )
    # (D) per-step stage, right after the lookback window is refreshed (same stream => ordered).
    a2 = "        model_inputs[\"lookback_token_ids\"] = window\n        return model_inputs\n"
    assert s.count(a2) == 1, "prepare_inputs anchor"
    s = s.replace(
        a2,
        "        model_inputs[\"lookback_token_ids\"] = window\n"
        "        if self._engram_host is not None:  # _ENGHOIST\n"
        "            self._engram_host.stage_engram_rows(\n"
        "                input_batch.input_ids,\n"
        "                input_batch.positions,\n"
        "                input_batch.query_start_loc,\n"
        "                window,\n"
        "            )\n"
        "        return model_inputs\n",
    )
    backup(STATE)
    open(STATE, "w").write(s)
    print("model_state.py: arm flag + per-step stage")


if __name__ == "__main__":
    if "--undo" in sys.argv:
        undo()
    elif "--fields" in sys.argv:
        fields()
    else:
        fields()
        patch_model()
        patch_state()
        print("done. Gate 1 = eager boot with VLLM_ENGRAM_HOIST_CHECK=1 (see ANSWER_FABLE_HOIST.md)")
