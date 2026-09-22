#!/usr/bin/env python3
"""engwill.py -- THE patch for DECIDER_V4_WAIT63: engfast2 + parallel cold-row fault-in + work_ms split.

Supersedes engfast2.py (v2/v2.1). Any installed _ENGFAST block (v1, v2, or the sibling SYNCPROOF
session's engfast3.py) is undone first via the same *.pre_engfast backups. Apply ONE of engwill.py /
engfast3.py, never both: same mechanism (POSIX_FADV_WILLNEED before the gather), different flag names.

w63 delta (WAIT63 consult). wait_ms (63 ms) is the previous step's GPU work = NOT a cost, no patch touches
it. work_ms (5.8 + 5.0 ms/step, GPU drained) IS the cost, and it is 10-20x the warm selftest price
(0.24 ms) => the rows are faulting from NVMe one at a time (18 rows x 2 files, ~150 us each, serial).
  - WILLNEED: once the ids are on the host, posix_fadvise(POSIX_FADV_WILLNEED) every row range on the
    table's own fds (self._fd_w / self._fd_s) BEFORE the memmap gather. The kernel submits all reads at
    once (NVMe QD ~36) and the gather then waits for the slowest instead of the sum. Hint only: the
    bytes still come through the SAME memmap gather => parity unchanged by construction.
    warm cost ~36 cheap syscalls (~0.05-0.1 ms/layer). `touch /tmp/engfast.nowill` = live A/B off.
  - `_ENGTIME` line splits work_ms into ids_ms (ids D2H + its sync) / gather_ms (fadvise + memmap
    gather = page faults) / tail_ms (H2D enqueue + trailing sync), prints period_ms (mean time between
    successive lookups of the same table = THIS window's step time; never divide by a banked kstep)
    and majflt (process-wide major faults per lookup over the window).
  - `touch /tmp/engfast.nobarrier` drops the diagnostic barrier sync (the ":467" one). Prediction:
    wait_ms -> ~0, ids_ms -> ~63, period_ms unchanged. That is the falsifier for "the sync costs 63 ms".

--- v2 notes (unchanged) ---

What changes in engram_disk.py fetch_unique_local_rows (appended wrapper, no anchors):
  legacy / step (2 layers, u=18): 36x .item + 72x safetensors get_slice + 72x copy_ + 6 syncs
  fast   / step                 :  2x numpy fancy-index gather from an np.memmap of the SAME bytes
                                   into the SAME pinned staging buffers, 1 sync per layer (the ids D2H)
  - header parsed once: row g of tensor K lives at file offset 8 + N + data_offsets[0] + g*width
    (weight width = dim = 256 B, strides (256,1); scale width = dim/32 = 8 B, strides (8,1); C-order)
  - MADV_RANDOM on both maps (cold fault = 1 page, not a 128 KB read-around)
  - ids D2H into a pinned buffer; inverse H2D from a pinned buffer (legacy: pageable, = a 3rd sync)
  - trailing current_stream().synchronize() KEPT by default (v2.1, 20MS consult: the event guards the
    pinned-buffer REWRITE, but not a consumer on another stream; legacy's sync covered both, and the
    in-serve parity gate compares HOST bytes so it is blind to that hazard).
    `touch /tmp/engfast.nosync` drops it live (event-only; prices the sync, expected <= ~0.2 ms/step).
  - covers PREFILL too (legacy prefill = ~0.1 ms x 6 x prompt_tokens x 2 layers of pure TTFT)
  - parity gate: first 64 decode lookups + first 2 prefill lookups PER TABLE run both paths and
    compare u, staged weight bytes, staged scale bytes, inverse. Any mismatch => legacy pinned for
    that table + `_ENGFAST PARITY FAIL`. Lines: `_ENGFAST PARITY ... equal=True` at n=1 and n=64.
  - `_ENGTIME` per 512 decode lookups: wait_ms (GPU still busy; NOT idle) / work_ms (GPU idle).
    `_ENGPREFILL` for the first 6 prefill-sized lookups per table.
  - live A/B, same boot, no env (Ray env drift cannot touch it):
        docker exec <c> touch /tmp/engfast.off      # legacy (still timed)   -- on ALL FOUR nodes
        docker exec <c> rm -f /tmp/engfast.off      # fast
Second edit (engram.py _lookup_disk): 6-iteration per-head remap loop (18 launches) -> 3 launches.
Skipped with a message if the anchor is not found exactly once.

  python3 engwill.py --selftest [MODEL_DIR]    # ZERO BOOTS, no GPU, read-only. Run this FIRST, inside
                                               # the live container on one node. Proves the offsets
                                               # byte-exact vs get_slice, prints legacy vs memmap
                                               # us/row, AND cold serial vs cold WILLNEED ms per
                                               # 18-row lookup = the price of this patch without a boot.
  python3 engwill.py                           # apply on ALL FOUR nodes (backups *.pre_engfast), reboot
  python3 engwill.py --undo

UNTESTED here (sandbox cannot run python). --selftest is the syntax + layout gate before any boot.
"""
import os
import shutil
import sys

V = os.environ.get("VLLM_PKG_ROOT") or "/usr/local/lib/python3.12/dist-packages/vllm"
DISK = f"{V}/models/deepseek_v4_1/common/engram_disk.py"
ENGRAM = f"{V}/models/deepseek_v4_1/common/engram.py"
MARK = "# --- _ENGFAST"
VER = "_ENGFAST w63"

BLOCK = '''
# --- _ENGFAST begin (_ENGFAST w63, engwill.py; remove with engwill.py --undo) ---
def _engfast_install():
    import json as _json
    import mmap as _mmap
    import struct as _struct
    import time as _time

    import numpy as _np

    T = DiskBackedEngramTable
    if getattr(T, "_engfast", False):
        return
    legacy = T.fetch_unique_local_rows
    OFF = "/tmp/engfast.off"
    NOSYNC = "/tmp/engfast.nosync"
    NOWILL = "/tmp/engfast.nowill"  # w63: skip the WILLNEED prefetch (live A/B)
    NOBARRIER = "/tmp/engfast.nobarrier"  # w63: skip the diagnostic barrier sync (falsifier)
    _fadv = getattr(os, "posix_fadvise", None)
    _WILL = getattr(os, "POSIX_FADV_WILLNEED", None)
    NPAR = 64  # parity-checked decode lookups per table
    NPAR_PF = 2  # parity-checked prefill lookups per table
    WIN = 512  # decode lookups per _ENGTIME line
    DECODE_MAX = 64  # ids per lookup counted as decode (3 rows x 6 heads = 18 at k=2)

    def _rank():
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return os.environ.get("RANK", "?")

    def _mm(path, key, width, dtypes, need_rows):
        with open(path, "rb") as f:
            (n,) = _struct.unpack("<Q", f.read(8))
            meta = _json.loads(f.read(n))[key]
        b, e = meta["data_offsets"]
        shape = [int(x) for x in meta["shape"]]
        if (
            meta["dtype"] not in dtypes
            or len(shape) != 2
            or shape[1] != width
            or e - b != shape[0] * width
            or need_rows > shape[0]
        ):
            raise ValueError(f"engfast: unexpected layout for {key}: {meta} need_rows={need_rows}")
        m = _np.memmap(path, dtype=_np.uint8, mode="r", offset=8 + n + b, shape=(shape[0], width))
        try:
            m._mmap.madvise(_mmap.MADV_RANDOM)
        except Exception:
            pass
        return m, 8 + n + b

    def _majflt():
        try:
            with open("/proc/self/stat") as f:
                return int(f.read().rsplit(")", 1)[1].split()[9])
        except Exception:
            return -1

    def _state(self):
        st = getattr(self, "_ef", None)
        if st is None:
            sw = self.dim // self.block_size
            st = {
                "hw": self._host_w.view(torch.uint8).numpy(),
                "hs": self._host_s.numpy(),
                "ids": torch.empty(4096, dtype=torch.long, pin_memory=True),
                "inv": torch.empty(4096, dtype=torch.long, pin_memory=True),
                "ev": None,
                "npar": 0,
                "npar_pf": 0,
                "npf": 0,
                "bad": False,
                "mode": None,
                "n": 0,
                "u": 0,
                "wait": 0.0,
                "work": 0.0,
                "sw": sw,
                "will": _fadv is not None and _WILL is not None,
                "last": (0.0, 0.0, 0.0),
                "ids_t": 0.0,
                "gath_t": 0.0,
                "tail_t": 0.0,
                "per": 0.0,
                "pn": 0,
                "tprev": None,
                "mf0": _majflt(),
            }
            try:
                st["mw"], st["bw"] = _mm(str(self.weight_file), self.weight_key, self.dim, ("F8_E4M3",), self.vocab_end)
                st["ms"], st["bs"] = _mm(str(self.scale_file), self.scale_key, sw, ("F8_E8M0", "U8"), self.vocab_end)
            except Exception as exc:
                st["bad"] = True
                print(f"_ENGFAST layer={self.layer_id} memmap unavailable -> legacy pinned: {exc!r}", flush=True)
            self._ef = st
        return st

    def _fast(self, st, local_row_ids):
        n = local_row_ids.numel()
        if n > st["ids"].numel():
            cap = 1 << (n - 1).bit_length()
            if st["ev"] is not None:
                st["ev"].synchronize()
            st["ids"] = torch.empty(cap, dtype=torch.long, pin_memory=True)
            st["inv"] = torch.empty(cap, dtype=torch.long, pin_memory=True)
        ids_t = st["ids"][:n]
        _ta = _time.perf_counter()
        ids_t.copy_(local_row_ids.detach().reshape(-1), non_blocking=True)
        torch.cuda.current_stream().synchronize()  # the ONE unavoidable sync: ids live on the GPU
        if st["ev"] is not None:
            st["ev"].synchronize()  # previous H2D out of the pinned buffers has landed (any stream)
        _tb = _time.perf_counter()
        ids = ids_t.numpy()
        valid = ids >= 0
        if not valid.any():
            return legacy(self, local_row_ids)
        uniq, inv_valid = _np.unique(ids[valid], return_inverse=True)  # sorted, like torch.unique
        u = int(uniq.size)
        if u > self.max_stage_rows:
            return legacy(self, local_row_ids)  # raises the same overflow error
        owned = self.vocab_end - self.vocab_start
        ok = uniq < owned
        g = _np.where(ok, uniq + self.vocab_start, 0)
        if st["will"] and not os.path.exists(NOWILL):  # decode AND prefill (cold prefill = serial faults too)
            # w63: kick every row's read NOW (async, all in flight together); the gather below then
            # waits for the slowest page instead of faulting 2*u pages one after another.
            # Hint only -- the bytes still come through the same memmap gather.
            try:
                fw, fs, bw, bs, dw, dsw = self._fd_w, self._fd_s, st["bw"], st["bs"], self.dim, st["sw"]
                for x in g[ok].tolist():
                    _fadv(fw, bw + x * dw, dw, _WILL)
                    _fadv(fs, bs + x * dsw, dsw, _WILL)
            except Exception as exc:
                st["will"] = False
                print(f"_ENGFAST layer={self.layer_id} WILLNEED disabled: {exc!r}", flush=True)
        st["hw"][:u] = st["mw"][g]
        st["hs"][:u] = st["ms"][g]
        if not ok.all():
            st["hw"][:u][~ok] = 0
            st["hs"][:u][~ok] = 0
        _tc = _time.perf_counter()
        inv = st["inv"][:n]
        inv_np = inv.numpy()
        inv_np[:] = -1
        inv_np[valid] = inv_valid.reshape(-1)
        self._gpu_w[:u].copy_(self._host_w[:u], non_blocking=True)
        self._gpu_s[:u].copy_(self._host_s[:u], non_blocking=True)
        inv_gpu = inv.to(device=self.device, non_blocking=True)
        if st["ev"] is None:
            st["ev"] = torch.cuda.Event()
        st["ev"].record()
        if not os.path.exists(NOSYNC):
            # v2.1 (20MS consult): trailing sync KEPT by default = gather-only change, same cross-stream
            # guarantee as legacy. `touch /tmp/engfast.nosync` = event-only (prices the sync, <= ~0.2 ms/step).
            torch.cuda.current_stream().synchronize()
        st["last"] = (_tb - _ta, _tc - _tb, _time.perf_counter() - _tc)
        self.rows_fetched_total += u
        self.lookups_total += 1
        return self._gpu_w[:u], self._gpu_s[:u], inv_gpu

    def _parity(self, st, local_row_ids, res, tag):
        st["ev"].synchronize()
        u = int(res[0].shape[0])
        hw = st["hw"][:u].copy()
        hs = st["hs"][:u].copy()
        inv = res[2].cpu()
        ref = legacy(self, local_row_ids)  # rewrites the same staging buffers, fully synced
        same = (
            int(ref[0].shape[0]) == u
            and bool((st["hw"][:u] == hw).all())
            and bool((st["hs"][:u] == hs).all())
            and bool(torch.equal(ref[2].cpu(), inv))
        )
        key = "npar_pf" if tag == "prefill" else "npar"
        st[key] += 1
        if not same:
            st["bad"] = True
        if not same or tag == "prefill" or st[key] in (1, NPAR):
            verdict = "equal=True" if same else "FAIL -> legacy path pinned"
            print(
                f"_ENGFAST PARITY rank={_rank()} layer={self.layer_id} {tag} n={st[key]} "
                f"ids={local_row_ids.numel()} u={u} {verdict}",
                flush=True,
            )
        return ref

    def fetch(self, local_row_ids):
        n_ids = local_row_ids.numel()
        if n_ids == 0 or self.device.type != "cuda":
            return legacy(self, local_row_ids)
        st = _state(self)
        decode = n_ids <= DECODE_MAX
        t0 = _time.perf_counter()
        if not os.path.exists(NOBARRIER):
            # Diagnostic barrier: splits "GPU still busy" (wait) from "GPU drained, host working" (work).
            # Costs nothing: the ids D2H a few lines later would wait for exactly the same GPU work.
            torch.cuda.current_stream().synchronize()
        t1 = _time.perf_counter()
        mode = "legacy" if (st["bad"] or os.path.exists(OFF)) else "fast"
        st["last"] = (0.0, 0.0, 0.0)
        if mode == "fast":
            res = _fast(self, st, local_row_ids)
            if st["ev"] is not None and (
                (decode and st["npar"] < NPAR) or (not decode and st["npar_pf"] < NPAR_PF)
            ):
                return _parity(self, st, local_row_ids, res, "decode" if decode else "prefill")
        else:
            res = legacy(self, local_row_ids)
        t2 = _time.perf_counter()
        if not decode:
            if st["npf"] < 6:
                st["npf"] += 1
                print(
                    f"_ENGPREFILL rank={_rank()} layer={self.layer_id} mode={mode} ids={n_ids} "
                    f"u={int(res[0].shape[0])} wait_ms={1e3 * (t1 - t0):.2f} work_ms={1e3 * (t2 - t1):.2f}",
                    flush=True,
                )
            return res
        zero = {"n": 0, "u": 0, "wait": 0.0, "work": 0.0, "ids_t": 0.0, "gath_t": 0.0, "tail_t": 0.0, "per": 0.0, "pn": 0}
        if st["mode"] != mode:
            st.update(zero)
            st.update({"mode": mode, "tprev": None, "mf0": _majflt()})
        st["n"] += 1
        st["u"] += int(res[0].shape[0])
        st["wait"] += t1 - t0
        st["work"] += t2 - t1
        st["ids_t"] += st["last"][0]
        st["gath_t"] += st["last"][1]
        st["tail_t"] += st["last"][2]
        if st["tprev"] is not None and t0 - st["tprev"] < 0.5:  # skip gaps between requests
            st["per"] += t0 - st["tprev"]
            st["pn"] += 1
        st["tprev"] = t0
        if st["n"] >= WIN:
            n = st["n"]
            mf = _majflt()
            mfr = (mf - st["mf0"]) / n if (mf >= 0 and st["mf0"] >= 0) else -1.0  # process-wide, both tables
            print(
                f"_ENGTIME rank={_rank()} layer={self.layer_id} mode={mode} lookups={n} "
                f"u_mean={st['u'] / n:.1f} wait_ms={1e3 * st['wait'] / n:.3f} "
                f"work_ms={1e3 * st['work'] / n:.3f} ids_ms={1e3 * st['ids_t'] / n:.3f} "
                f"gather_ms={1e3 * st['gath_t'] / n:.3f} tail_ms={1e3 * st['tail_t'] / n:.3f} "
                f"period_ms={1e3 * st['per'] / max(1, st['pn']):.3f} majflt={mfr:.1f} "
                f"will={int(st['will'] and not os.path.exists(NOWILL))} "
                f"barrier={int(not os.path.exists(NOBARRIER))}",
                flush=True,
            )
            st.update(zero)
            st["mf0"] = mf
        return res

    T.fetch_unique_local_rows = fetch
    T._engfast = True
    print(
        "_ENGFAST w63 installed (touch /tmp/engfast.off = legacy, .nosync = drop trailing sync, "
        ".nowill = no WILLNEED prefetch, .nobarrier = no diagnostic barrier)",
        flush=True,
    )


try:
    _engfast_install()
except Exception as _e:
    print(f"_ENGFAST install failed: {_e!r}", flush=True)
# --- _ENGFAST end ---
'''

HEAD_LOOP = (
    "        for h_i in range(local_heads):\n"
    "            global_h = self.head_start + h_i\n"
    "            if global_h >= self.n_hash_cols:\n"
    "                continue\n"
    "            remapped[:, global_h] = torch.where(\n"
    "                valid[:, h_i],\n"
    "                inv[:, h_i],\n"
    "                torch.full_like(inv[:, h_i], -1),\n"
    "            )\n"
)
HEAD_VEC = (
    "        _nh = max(0, min(local_heads, self.n_hash_cols - self.head_start))  # _ENGFAST\n"
    "        if _nh:\n"
    "            remapped[:, self.head_start : self.head_start + _nh] = torch.where(\n"
    "                valid[:, :_nh], inv[:, :_nh], torch.full_like(inv[:, :_nh], -1)\n"
    "            )\n"
)


def undo():
    for p in (DISK, ENGRAM):
        b = p + ".pre_engfast"
        if os.path.exists(b):
            shutil.copy2(b, p)
            os.remove(b)
            print("restored", p)
        else:
            print("no backup for", p)


def apply():
    if not os.path.exists(DISK):
        sys.exit(f"MISSING {DISK} (disk-engram overlay not installed at the expected path)")
    s = open(DISK).read()
    if VER in s:
        print("already applied (w63):", DISK)
    else:
        if MARK in s:
            print("older/other _ENGFAST block found: undoing it first")
            undo()
            s = open(DISK).read()
            assert MARK not in s, "old block still present after undo (no backup?): restore engram_disk.py by hand"
        for need in (
            "class DiskBackedEngramTable",
            "def fetch_unique_local_rows",
            "self._host_w",
            "self.weight_key",
            "self.vocab_end",
            "self._fd_w",
            "self._fd_s",
        ):
            assert need in s, f"engram_disk.py lacks {need!r}"
        assembled = s.rstrip("\n") + "\n" + BLOCK
        # Compile BEFORE overwriting: a syntax error in the appended block must
        # not leave a broken engram_disk.py in the tree.
        compile(assembled, DISK, "exec")
        shutil.copy2(DISK, DISK + ".pre_engfast")
        open(DISK, "w").write(assembled)
        print("applied:", DISK)
    e = open(ENGRAM).read()
    if "# _ENGFAST" in e:
        print("already applied:", ENGRAM)
    elif e.count(HEAD_LOOP) == 1:
        shutil.copy2(ENGRAM, ENGRAM + ".pre_engfast")
        open(ENGRAM, "w").write(e.replace(HEAD_LOOP, HEAD_VEC))
        compile(open(ENGRAM).read(), ENGRAM, "exec")
        print("applied:", ENGRAM, "(per-head remap loop vectorized)")
    else:
        print(f"SKIPPED {ENGRAM}: head-loop anchor count={e.count(HEAD_LOOP)} (fetch fix still active)")


def selftest(model_dir, n=256):
    """Zero-boot, CPU-only, read-only: byte parity of the memmap view vs get_slice + us/row of both."""
    import json
    import random
    import struct
    import time

    import numpy as np
    import torch
    from safetensors import safe_open

    wmap = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]

    def header(path, key):
        with open(path, "rb") as f:
            (hn,) = struct.unpack("<Q", f.read(8))
            meta = json.loads(f.read(hn))[key]
        return hn, meta

    def legacy_rows(sl_w, sl_s, rows, dim, sw):
        hw = torch.empty(len(rows), dim, dtype=torch.float8_e4m3fn)
        hs = torch.empty(len(rows), sw, dtype=torch.uint8)
        uv = torch.tensor(rows, dtype=torch.long)
        t = time.perf_counter()
        for i in range(len(rows)):
            row = int(uv[i].item())
            w = sl_w[row : row + 1]
            s = sl_s[row : row + 1]
            if s.dtype == torch.float8_e8m0fnu:
                s = s.view(torch.uint8)
            hw[i].copy_(w.view(torch.float8_e4m3fn).reshape(dim))
            hs[i].copy_(s.reshape(sw))
        dt = time.perf_counter() - t
        return hw.view(torch.uint8).numpy(), hs.numpy(), 1e6 * dt / len(rows)

    def mm_rows(mw, ms, rows):
        g = np.asarray(rows, dtype=np.int64)
        t = time.perf_counter()
        a = np.ascontiguousarray(mw[g])
        b = np.ascontiguousarray(ms[g])
        dt = time.perf_counter() - t
        return a, b, 1e6 * dt / len(rows)

    ok_all = True
    for layer in (1, 14):
        wk = f"layers.{layer}.engram.embed.weight"
        sk = f"layers.{layer}.engram.embed.scale"
        wf = os.path.join(model_dir, wmap[wk])
        sf = os.path.join(model_dir, wmap[sk])
        hn_w, mw_meta = header(wf, wk)
        hn_s, ms_meta = header(sf, sk)
        R, dim = (int(x) for x in mw_meta["shape"])
        Rs, sw = (int(x) for x in ms_meta["shape"])
        bw, ew = mw_meta["data_offsets"]
        bs, es = ms_meta["data_offsets"]
        print(
            f"layer {layer}: weight {mw_meta['dtype']} {R}x{dim} file={os.path.basename(wf)} "
            f"abs_off={8 + hn_w + bw} | scale {ms_meta['dtype']} {Rs}x{sw} "
            f"file={os.path.basename(sf)} abs_off={8 + hn_s + bs}"
        )
        assert R == Rs and ew - bw == R * dim and es - bs == Rs * sw, "layout: not dense 2-D row-major"
        mw = np.memmap(wf, dtype=np.uint8, mode="r", offset=8 + hn_w + bw, shape=(R, dim))
        ms = np.memmap(sf, dtype=np.uint8, mode="r", offset=8 + hn_s + bs, shape=(Rs, sw))
        hw_ = safe_open(wf, framework="pt", device="cpu")
        hs_ = safe_open(sf, framework="pt", device="cpu")
        sl_w, sl_s = hw_.get_slice(wk), hs_.get_slice(sk)
        rng = random.Random(1234 + layer)
        set_a = sorted({0, R - 1, *(rng.randrange(R) for _ in range(n))})
        set_b = sorted({1, R - 2, *(rng.randrange(R) for _ in range(n))})
        la_w, la_s, leg_cold = legacy_rows(sl_w, sl_s, set_a, dim, sw)
        # v2.1: warm = 5 passes, report median (and min..max). A single pass on a node that is serving
        # (page cache under pressure) can carry one reclaim stall or re-fault: L14 read 91.8 vs L1 18.2
        # us/row on identical code in the first run of this selftest.
        leg_w5 = sorted(legacy_rows(sl_w, sl_s, set_a, dim, sw)[2] for _ in range(5))
        leg_warm = leg_w5[2]
        ma_w, ma_s, _first = mm_rows(mw, ms, set_a)  # first touch through THIS mapping = minor faults
        mm_w5 = sorted(mm_rows(mw, ms, set_a)[2] for _ in range(5))
        mm_warm = mm_w5[2]
        print(
            f"  warm x5 us/row: legacy min={leg_w5[0]:.1f} med={leg_w5[2]:.1f} max={leg_w5[4]:.1f} | "
            f"memmap first-touch={_first:.2f} min={mm_w5[0]:.2f} med={mm_w5[2]:.2f} max={mm_w5[4]:.2f}"
        )
        mb_w, mb_s, mm_cold = mm_rows(mw, ms, set_b)
        lb_w, lb_s, _ = legacy_rows(sl_w, sl_s, set_b, dim, sw)
        # v2.1: price MADV_RANDOM on a THIRD, untouched row set (what the installed patch actually uses):
        # a cold fault reads 1 page instead of the default read-around window.
        try:
            import mmap as _mmap_mod

            mw_r = np.memmap(wf, dtype=np.uint8, mode="r", offset=8 + hn_w + bw, shape=(R, dim))
            ms_r = np.memmap(sf, dtype=np.uint8, mode="r", offset=8 + hn_s + bs, shape=(Rs, sw))
            mw_r._mmap.madvise(_mmap_mod.MADV_RANDOM)
            ms_r._mmap.madvise(_mmap_mod.MADV_RANDOM)
            set_c = sorted({2, R - 3, *(rng.randrange(R) for _ in range(n))})
            _, _, mm_cold_rand = mm_rows(mw_r, ms_r, set_c)
            print(
                f"  cold us/row: memmap default={mm_cold:.1f} | memmap MADV_RANDOM={mm_cold_rand:.1f} "
                f"=> fully-cold step (36 rows): {36 * mm_cold / 1e3:.1f} ms vs {36 * mm_cold_rand / 1e3:.1f} ms"
            )
            # w63 (WAIT63): the same cold gather in DECODE-SIZED batches of 18 rows, serial faults vs
            # WILLNEED-first. This is the zero-boot price of the prefetch. Gate: will <= 0.35 x serial.
            fdw, fds = os.open(wf, os.O_RDONLY), os.open(sf, os.O_RDONLY)
            base_w, base_s = 8 + hn_w + bw, 8 + hn_s + bs

            def batches(rows, will):
                per = []
                for i in range(0, len(rows) - 17, 18):
                    chunk = rows[i : i + 18]
                    t = time.perf_counter()
                    if will:
                        for x in chunk:
                            os.posix_fadvise(fdw, base_w + x * dim, dim, os.POSIX_FADV_WILLNEED)
                            os.posix_fadvise(fds, base_s + x * sw, sw, os.POSIX_FADV_WILLNEED)
                    g = np.asarray(chunk, dtype=np.int64)
                    a, b = np.ascontiguousarray(mw_r[g]), np.ascontiguousarray(ms_r[g])
                    per.append(1e3 * (time.perf_counter() - t))
                per.sort()
                return per[len(per) // 2], a, b, chunk

            set_d = [rng.randrange(R) for _ in range(18 * 15)]
            set_e = [rng.randrange(R) for _ in range(18 * 15)]
            ser_ms, _, _, _ = batches(set_d, False)
            wil_ms, wa, wb, wchunk = batches(set_e, True)
            warm_ms, _, _, _ = batches(set_e, True)  # same rows again = warm cost incl. the 36 syscalls
            rw, rs, _ = legacy_rows(sl_w, sl_s, wchunk, dim, sw)
            will_same = np.array_equal(rw, wa) and np.array_equal(rs, wb)
            ok_all &= bool(will_same)
            print(
                f"  WILLNEED ms per 18-row lookup (median of 15): cold serial={ser_ms:.2f} | cold willneed={wil_ms:.2f} "
                f"| warm willneed={warm_ms:.3f} | bytes equal={will_same} "
                f"=> {'GO' if wil_ms <= 0.35 * ser_ms else 'NO-GO (prefetch not parallelizing on this fs)'}"
            )
            os.close(fdw)
            os.close(fds)
        except Exception as exc:
            print(f"  MADV_RANDOM cold probe skipped: {exc!r}")
        same = (
            np.array_equal(la_w, ma_w)
            and np.array_equal(la_s, ma_s)
            and np.array_equal(lb_w, mb_w)
            and np.array_equal(lb_s, mb_s)
        )
        ok_all &= bool(same)
        print(
            f"  PARITY rows={len(set_a)}+{len(set_b)} equal={same} nonzero_w={int(ma_w.any())} "
            f"| us/row legacy cold={leg_cold:.1f} warm={leg_warm:.1f} | memmap cold={mm_cold:.1f} warm={mm_warm:.2f}"
        )
        print(
            f"  => item #1 host price at 18 rows/layer: legacy warm {18 * leg_warm / 1e3:.2f} ms/layer, "
            f"memmap warm {18 * mm_warm / 1e3:.3f} ms/layer (x2 layers per step; excludes syncs + launches)"
        )
    print("SELFTEST", "PASS" if ok_all else "FAIL -- do NOT apply")
    return ok_all


if __name__ == "__main__":
    if "--undo" in sys.argv:
        undo()
    elif "--selftest" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("--")]
        md = rest[0] if rest else os.environ.get("NVME_ENGRAM_PATH") or os.environ.get("VLLM_ENGRAM_MODEL_DIR") or "/models/dsv41-orig"
        sys.exit(0 if selftest(md) else 1)
    else:
        apply()
