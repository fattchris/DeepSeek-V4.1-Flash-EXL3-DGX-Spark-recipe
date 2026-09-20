#!/usr/bin/env python3
"""_ENGRAMKEY: fix + probe for the engram split-op registry key collision.

Defect: Engram.prepare_embeddings keys _ENGRAM_OPAQUE_LAYERS by
    getattr(self, "layer_name", None) or getattr(self, "prefix", "engram")
but Engram.__init__ never stores prefix/layer_name, so EVERY layer uses the key
"engram". Eager and capture are set-then-call (self-consistent). Breakable REPLAY
does not re-run prepare_embeddings, only the recorded lambda
    fn(weak_ids, "engram", weak_rows)
so every engram break resolves the LAST registered module (layer 14) and layer 1's
staged_rows get filled from layer 14's table.

Usage (inside the container, every node):
    python3 engramkey.py               # unique key + host-only probes
    python3 engramkey.py --probe-only  # probes only (shows the mismatch live)
    python3 engramkey.py --undo        # restore both files

Probes are pure Python in code that ALREADY runs eagerly at replay (the engram op
body and BreakableCUDAGraphWrapper._replay): no new break, no CUDA call, no sync
(Tensor.data_ptr() is a host read).
"""
import os
import shutil
import sys

PKG = os.environ.get("VLLM_PKG_ROOT") or "/usr/local/lib/python3.12/dist-packages/vllm"
ENGRAM = os.path.join(PKG, "models/deepseek_v4_1/common/engram.py")
BREAK = os.path.join(PKG, "compilation/breakable_cudagraph.py")
BAK = ".pre_engramkey"

args = set(sys.argv[1:])
if "--undo" in args:
    for p in (ENGRAM, BREAK):
        if os.path.exists(p + BAK):
            shutil.copy2(p + BAK, p)
            os.remove(p + BAK)
            print("restored", p)
        else:
            print("no backup for", p)
    sys.exit(0)

probe_only = "--probe-only" in args


def patch(path, pairs, marker):
    s = open(path).read()
    if marker in s:
        print("already:", marker, path)
        return
    for old, new in pairs:
        assert s.count(old) == 1, f"{path}: anchor count {s.count(old)} != 1:\n{old}"
        s = s.replace(old, new, 1)
    compile(s, path, "exec")
    if not os.path.exists(path + BAK):
        shutil.copy2(path, path + BAK)
    open(path, "w").write(s)
    print("patched:", marker, path)


# ---- 0. static verdict (needs no boot) ---------------------------------------
src = open(ENGRAM).read()
stores = [k for k in ("self.prefix =", "self.layer_name =") if k in src]
print("STATIC: Engram stores", stores or "NEITHER prefix nor layer_name",
      "=> registry key is", "per-layer" if stores else '"engram" for every layer (COLLISION)')

# ---- 1. probe in the op body --------------------------------------------------
PROBE = '''_ENGRAMKEY_STATS = {"calls": 0, "mismatch": 0, "logged": 0}


def _engramkey_probe(module, layer_name: str, out: torch.Tensor) -> None:
    # _ENGRAMKEY_PROBE: host-only. `out` must be the resolved module's own staging.
    st = _ENGRAMKEY_STATS
    st["calls"] += 1
    ptr = out.data_ptr()
    owners = [module.staged_rows, *module._extra_staged_rows]
    ok = any(ptr == o.data_ptr() for o in owners)
    if not ok:
        st["mismatch"] += 1
    if st["logged"] < 12 or (not ok and st["logged"] < 48):
        st["logged"] += 1
        logger.info(
            "_ENGRAMKEY call=%d key=%s resolved_hash_index=%s registry=%s out_owner_ok=%s mismatches=%d",
            st["calls"], layer_name, getattr(module, "layer_hash_index", "?"),
            sorted(_ENGRAM_OPAQUE_LAYERS), ok, st["mismatch"],
        )


def _engram_disk_lookup_out_op('''

patch(ENGRAM, [
    ("def _engram_disk_lookup_out_op(", PROBE),
    ("""    module = _ENGRAM_OPAQUE_LAYERS[layer_name]
    module.embed_tokens.lookup(ids, out)""",
     """    module = _ENGRAM_OPAQUE_LAYERS[layer_name]
    _engramkey_probe(module, layer_name, out)
    module.embed_tokens.lookup(ids, out)"""),
], "_ENGRAMKEY_PROBE")

# ---- 2. the fix: unique key per Engram module ---------------------------------
if not probe_only:
    patch(ENGRAM, [
        ("""            name = getattr(self, "layer_name", None) or getattr(self, "prefix", "engram")
""",
         """            # _ENGRAMKEY_FIX: Engram stores neither layer_name nor prefix, so the old
            # key was "engram" for every layer and replay resolved the last one.
            name = f"engram.{self.layer_hash_index}.{id(self):x}"
"""),
    ], "_ENGRAMKEY_FIX")

# ---- 3. per-replay counter in the fork runner (DIAGNOSTIC, optional) ----------
# This probe is instrumentation only -- it does not change behaviour. The
# breakable fork runner may be absent or shaped differently in another build, so
# a failure here must not abort after fix #1 has already been written.
if not os.path.exists(BREAK):
    print("skipped (no breakable runner at %s): _FORKPROBE is diagnostic only" % BREAK)
else:
  try:
    patch(BREAK, [
    ("""        entry.capture.replay()
""",
     """        _fp_before = _forkprobe_engram()
        entry.capture.replay()
        # _FORKPROBE: host-only per-replay accounting (no CUDA calls).
        _FORKPROBE_N[0] += 1
        if _FORKPROBE_N[0] <= 12:
            _fp_after = _forkprobe_engram()
            logger.info(
                "_FORKPROBE replay=%d desc=%s segs=%d graphs=%d breaks=%d addr_ok=%s "
                "engram_calls_delta=%d engram_mismatch_delta=%d",
                _FORKPROBE_N[0], entry.batch_descriptor, len(entry.capture.segments),
                entry.capture.num_graphs, entry.capture.num_eager_breaks,
                self._collect_tensor_addresses(args, kwargs) == entry.input_addresses,
                _fp_after[0] - _fp_before[0], _fp_after[1] - _fp_before[1],
            )
"""),
    ("""def is_breakable_cudagraph_enabled() -> bool:""",
     """_FORKPROBE_N = [0]


def _forkprobe_engram() -> tuple[int, int]:
    try:
        from vllm.models.deepseek_v4_1.common import engram as _eg

        st = _eg._ENGRAMKEY_STATS
        return st["calls"], st["mismatch"]
    except Exception:
        return (-1, -1)


def is_breakable_cudagraph_enabled() -> bool:"""),
  ], "_FORKPROBE")
  except Exception as _exc:
    print("skipped (runner probe failed, diagnostic only): %r" % (_exc,))

print("engramkey done (%s)" % ("probe only" if probe_only else "fix + probe"))
