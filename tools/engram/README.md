# Disk-engram and draft-loader fixes

Four silent defects on the DeepSeek-V4.1-Flash EXL3 serving path. None of them
raises an error: they show up only as wrong token distributions (under CUDA-graph
replay), as slow cold I/O, or as a draft model quietly running on a quarter of
its expert weights.

Apply all four (idempotent, `--undo` supported) inside the serving container on
**every rank**, **before the model loads**:

```bash
python3 tools/engram/engram_key_fix.py          # replay keying          (GPU graphs)
python3 tools/engram/engram_hoist.py            # enables FULL_DECODE_ONLY
python3 tools/engram/engram_io_fix.py           # cold I/O, +27.4% on novel text
python3 tools/engram/draft_ep_filter_fix.py     # draft ran at 25% of its experts
```

Order does not matter between them; all four patch distinct sites, all are safe
to leave installed under eager and under PIECEWISE, and all four must be in place
before the process that loads the model starts (see the restart note below).

### What each fix is worth (4x DGX Spark, TP4/EP4, bs1, k=2)

| # | fix | effect |
|---|---|---|
| 1 | `engram_key_fix` | correctness under replay; without it acceptance collapses to 1.00 |
| 2 | `engram_hoist` | unblocks `FULL_DECODE_ONLY` capture (was a hard CUDA error) |
| 3 | `engram_io_fix` | +27.4% decode on **novel** text; ~0 on warm/repeat text |
| 4 | `draft_ep_filter_fix` | **+9% to +20%** on every domain; best cell 30.21 tok/s |

### Restart required

Every script here patches files on disk. Python binds these symbols at import
time, so applying a script to an **already-running** server changes the file and
prints `installed` while the live process keeps using the old code — no error,
and the fix appears to do nothing. Apply before starting the server, or restart
it afterwards.

### Runtime prerequisites

- `VLLM_ENGRAM_DISK_BACKED=1` — the disk-engram path must be active, otherwise
  fixes 1-3 are inert.
- The model must be served with `cudagraph_mode: FULL_DECODE_ONLY` for fix 2 to
  matter; **no environment variable enables it — it is a serve-yaml setting.**
- `num_speculative_tokens: 2` (k=2) is the measured optimum on this hardware.
- Fix 4 needs the loader + DSpark draft only; no CUDA graphs.
- Fix 3's target (`engram_disk.py`) is part of the disk-engram overlay; fix 4's
  targets (`ep_weight_filter.py`, `dspark.py`) are stock upstream vLLM.

### Patch root

The scripts default to `/usr/local/lib/python3.12/dist-packages/vllm`. Override
with `VLLM_PKG_ROOT=/path/to/site-packages/vllm` if your install differs (the
disk-engram Dockerfile resolves the same root dynamically via
`Path(vllm.__file__).resolve().parent`).

### Environment / flag switches

| switch | effect |
|---|---|
| `VLLM_ENGRAM_HOIST=0` | disable fix 2's hoist (forward does the lookup itself) |
| `VLLM_ENGRAM_HOIST_CHECK=1` | eager-only parity check (syncs; boots slowly) |
| `VLLM_DSPARK_ALLOW_UNLOADED_EXPERTS=1` | downgrade fix 4's fail-closed gate |
| `/tmp/engfast.off` | A/B switch: restore legacy row fetch |
| `/tmp/engfast.nowill` | A/B switch: keep the gather, drop `WILLNEED` |
| `/tmp/epfilter.off` | A/B switch: restore the original EP pre-filter |
| `/tmp/epgate.allow` | A/B switch: allow unloaded draft experts |

All are optional; every fix is active by default with **no environment set**, so
Ray worker env drift cannot silently disable one.

---

## 1. Registry-key collision (`engram_key_fix.py`)

`Engram.prepare_embeddings` registers itself in `_ENGRAM_OPAQUE_LAYERS` under

```python
name = getattr(self, "layer_name", None) or getattr(self, "prefix", "engram")
```

but `Engram.__init__` **never stores `prefix` or `layer_name`** (it takes `prefix`
as an argument, uses it once, and discards it). Both `getattr`s therefore fall to
the default, and **every** Engram layer registers under the literal key `"engram"`.

The config has two Engram layers:

```json
"engram_layer_ids": [1, 14],
"engram_num_embeddings": [384006168, 384016682]
```

Eager and capture are set-then-call, so they are self-consistent. **Replay is not**:
the breakable runner records `fn(weak_ids, "engram", weak_rows)` and resolves the key
at call time, so every replayed step fills **layer 1's** staged rows from **layer
14's** table using layer 1's hash ids. Different tables, different primes/offsets.
The result is finite, normal-magnitude, wrong data injected at layer 1 for every
token of every decode step.

Symptoms: correct under eager, wrong under graphs; no NaN, no assertion, no CUDA
error. Under speculative decoding the target distribution is off, so every draft
token is rejected and acceptance collapses to exactly 1.00.

Fix: key by the stored `layer_hash_index`, which `__init__` does keep:

```python
name = f"engram.{self.layer_hash_index}.{id(self):x}"
```

## 2. Host-dependent lookup inside capture (`engram_hoist.py`)

`FULL_DECODE_ONLY` bypasses the breakable wrapper entirely
(`@eager_break_during_capture` is a no-op when `cudagraph_runtime_mode == FULL`),
so the split-op trick that keeps the disk fetch eager under PIECEWISE does not
apply. The host-side D2H in the lookup then lands inside a plain
`torch.cuda.graph` capture and dies with:

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
```

Fix: stage the hash + lookup **eagerly**, once per real step, in
`DeepseekV41ModelState.prepare_inputs` — which already runs outside capture and
already owns the engram lookback window — and skip it in the model forward behind a
**static** flag (a per-step flag is evaluated at capture time and would be False).

`Engram.staged_rows` is allocated once in `__init__`, sized
`max_num_batched_tokens * dp_size`, and written in place; the fork's own comment
says it exists to *"keep lookup results alive across breakable graph segments"*. So
it is address-stable and the captured graph can read it as a static input.

Verified parity (eager, `VLLM_ENGRAM_HOIST_CHECK=1`):

```
_ENGHOIST CHECK n=2 equal=True ref=(2, 2, 24) fwd=(2, 2, 24)
```

---

## Results

Runtime identity for every number below. These matter as much as the config:

| | |
|---|---|
| hardware | 4x NVIDIA DGX Spark (GB10, SM121), TP4 / EP4 |
| kernel | `vllm_exl3_c` built from the vllm-exl3 PRs below, ABI 4 |
| vllm-exl3 | kernels from PRs #31 (multi-K), #32 (codebook `P2B_CB`), #33 (padded), now merged; `Dockerfile.tp4` builds them from `4c95648` |
| model | DeepSeek-V4.1-Flash EXL3 pack, served from `/models/dsv41-orig` style layout |
| engine config | `cudagraph_mode: FULL_DECODE_ONLY`, `cudagraph_capture_sizes [3,6,9,12]`, `max-model-len 4096`, `enable-prefix-caching: true` |
| spec decode | `method: dspark`, `num_speculative_tokens: 2`, `draft_sample_method: probabilistic`, `rejection_sample_method: block`, `quantization: mxfp4` |
| engram | `VLLM_ENGRAM_DISK_BACKED=1` |
| benchmark | `llm-inference-bench`, sustained 30 s cells, c=1, `max_tokens 512` |

| config | ctx0 | repeat | ctx2048 |
|---|---|---|---|
| eager | 14.70 | 21.20 | 15.62 |
| PIECEWISE | 13.84 | 20.03 | 13.80 |
| **FULL_DECODE_ONLY + hoist** | **18.60** | **26.03-26.34** | **17.77** |

The repeat cell is quoted as a range: it is the same configuration measured on
different boots (26.03 and 26.34 tok/s), and per-run spread of this size is
normal here. Treat ~26 as the figure, not either endpoint.

After fix 4 (the draft EP filter) the multi-domain matrix is:

| domain | ctx0 | ctx2048 | acceptance |
|---|---|---|---|
| code | 27.02 | 27.01 | 1.89 / 1.92 |
| prose | 27.44 | 30.21 | 1.97 / 2.21 |
| structured | 27.73 | 29.70 | 1.99 / 2.14 |

Correct output at every state (`The capital of France is` -> `Paris`, temp 0).

---

## 3. Serial NVMe page faults in the row fetch (`engram_io_fix.py`)

`fetch_unique_local_rows` gathers each unique row with a per-row loop. Measured in-serve with the
`_ENGTIME` instrument: **`work_ms` = 5.8 + 5.0 ms/step** across the two Engram layers, while the GPU is
drained — i.e. ~36 serial page touches per step at ~160 us each (18 rows x weight + scale file), one NVMe
4K random read at a time, QD1.

Fix: `posix_fadvise(POSIX_FADV_WILLNEED)` on every row's byte range (using the table's own fds) **before**
the memmap gather. The kernel allocates the pages and submits all reads at once (NVMe QD ~36), so the
gather waits for the slowest instead of the sum. It is a hint only — the bytes still come through the same
gather, so parity is unchanged by construction.

Measured: `work_ms` **5.76 -> 0.586** (layer 1), **4.98 -> 0.201** (layer 14), `majflt=0`.

**Measured, ABBA cold A/B** (8 never-seen essays, live flag flip on all four nodes, ms/step per request
from `/metrics`, major faults per step from `/proc/vmstat`):

| arm | ms/step | majflt/step (max node) |
|---|---|---|
| cold **ON** (WILLNEED) | **69.9** | 0.3-0.7 |
| cold **OFF** | **89.1** | 62.6-67.5 |
| warm ON vs OFF (control) | 67.8 vs 67.5 | 0.1 |

**delta = 19.1 ms/step = +27.4% decode throughput on novel text**, with major faults present only on the
OFF rows — the mechanism confirmed end to end. Warm/repeat rows are untouched (control agrees within 0.3 ms),
so the repeat bench is unchanged at ~67 ms/step, as expected.

Two traps if you re-measure this: (1) prompts must be **never-seen** — a prompt can only be cold once, and
reusing one makes the A/B read zero; (2) measure **ms/step**, not tok/s — tok/s is contaminated by acceptance
variance (~20% here) which would bury a 13% effect. Also do **not** try to `drop_caches`: `invalidate_mapping_pages`
skips pages mapped into a process, and the rows under test are exactly the mapped ones, so it would leave them
warm and read a false negative.

The same script also replaces the per-row `safetensors` `get_slice` loop with a single memmap fancy-index
gather (byte-exact vs `get_slice`, verified on 258+258 rows including first/last of both tables).

## Diagnosing this class of problem

Two instruments did all the work and neither needs a profiler:

- `nvidia-smi --query-gpu=utilization.gpu --format=csv -lms 500` on every node during one long decode.
  0% at idle, 63-67% under load here — so ~25 ms/step of the 67.9 ms step had no kernel executing.
- `py-spy record --pid <RayWorkerProc pid>` (not the EngineCore pid: that one is ~99% blocked in
  `_wait_for_response` and tells you nothing). 89.2% of worker samples landed in one `synchronize`, with a
  single caller chain back to the engram lookup — which is what pointed at the fetch.

`engram_io_fix.py` carries the `_ENGTIME` line: `wait_ms` (GPU still busy - not idle, not a cost) and
`work_ms` (GPU idle - this is the cost), plus its own `period_ms` denominator so a window's numbers are
never divided by a step time from a different workload.

---

## 4. Draft expert weights dropped by the loader's EP pre-filter (`draft_ep_filter_fix.py`)

The DSpark draft was running with **only 25% of its expert weights loaded**, silently.

The model loader drops expert weights it considers non-local *before* reading them
(`weight_utils.py`, `should_skip_weight(name, local_expert_ids)`), and the window comes from
`model_config.get_num_experts()` = `n_routed_experts` = **384**. The DSpark draft has
**128** experts (`dspark_n_routed_experts`). With EP=4:

| EP rank | filter keeps `mtp.*.experts.E.*.weight` for | draft map owns | owned weights that arrive |
|---|---|---|---|
| 0 | E in [0, 96) | [0, 32) | **32/32** |
| 1 | E in [96, 192) | [32, 64) | **0/32** |
| 2 | E in [192, 288) | [64, 96) | **0/32** |
| 3 | E in [288, 384) | [96, 128) | **0/32** |

Every model-side consumer was special-cased for the draft's expert count; the loader's
filter was not. Because the drop happens before `get_tensor`, the names never reach the
draft's load loop — there is no error, no warning, just missing weights. (This is why
in-loop instrumentation cannot see it: it is upstream of the loop.)

Observed with a post-load probe:

```
LIVECHK TP0_EP0  w13_weight absmax=255.0 nz_rows=32   <- loaded
LIVECHK TP1_EP1  w13_weight absmax=0.0   nz_rows=0    <- empty
LIVECHK TP2_EP2  w13_weight absmax=0.0   nz_rows=0
LIVECHK TP3_EP3  w13_weight absmax=0.0   nz_rows=0
```

Symptom: acceptance is **flat and low across every domain**, and in particular **code is no
faster than prose** — the opposite of what a live expert FFN gives, because the drafter
degrades to attention + shared experts + the Markov bigram head, which has no structural
advantage. Before/after on `llm-inference-bench` (c=1, sustained 30 s cells):

| domain | ctx | before tok/s | after | before acc | after acc |
|---|---|---|---|---|---|
| code | 0 | 24.12 | **27.02** | 1.72 | **1.89** |
| code | 2k | — | **27.01** | — | **1.92** |
| prose | 0 | 25.07 | **27.44** | 1.76 | **1.97** |
| prose | 2k | — | **30.21** | — | **2.21** |
| structured | 0 | 25.02 | **27.73** | 1.81 | **1.99** |
| structured | 2k | — | **29.70** | — | **2.14** |

**+9% to +20%**, best cell 30.21 tok/s.

Fix: an `_EPFILTER` wrapper that bypasses the pre-read skip decision for `mtp.*` names — the
draft's own expert map still drops non-local ids inside `RoutedExperts.weight_loader`, which
is the designed behaviour — plus an `_EPGATE` per-rank accounting check that **fails closed**
if any owned slot is empty.

```bash
python3 tools/engram/draft_ep_filter_fix.py --probe-only   # zero-boot: print the filter + window table
python3 tools/engram/draft_ep_filter_fix.py                # install
python3 tools/engram/draft_ep_filter_fix.py --undo
```

Live A/B without an env var (Ray worker env drifts): `touch /tmp/epfilter.off` restores the
old behaviour. Opt out of the fail-closed gate with `VLLM_DSPARK_ALLOW_UNLOADED_EXPERTS=1`
or `touch /tmp/epgate.allow`.

**Upstream fix** (the proper one, for the loader rather than a patch): size the draft load's
EP filter from `dspark_n_routed_experts`, not `n_routed_experts`.

---

## 5. Large context blocked by flashinfer's persistent topk on SM12x (`persistent_topk_sm12x_fix.py`)

Booting with a large `max-model-len` fails at **engine init**:

```
RuntimeError: launch_persistent_topk, /workspace/csrc/libtorch_stable/topk.cu:138,
persistent_topk would oversubscribe and the FilteredTopK fallback requires >=128KB smem per block
(have 101376). total_ctas=90 > num_sms*occupancy=48
(TopK=512, vec_size=4, ctas_per_group=90, smem=48688).
```

**Why**: flashinfer's `persistent_topk` derives its CTA count from the **logits stride**,
which grows with context:

```
fixed smem = 2,080 B;  max chunk = (49,152 - 2,080)/4 = 11,768 elements
CTAs = ceil(1,048,576 / 11,768) = 90        smem = 2,080 + 11,652*4 = 48,688
```

On **SM121 (GB10) the device allows only 99 KiB smem/block** (`sharedMemPerBlockOptin` = 101,376),
so the `FilteredTopK` fallback (which needs >=128 KiB/block) is unreachable and 90 CTAs do
not fit in 48. This is **not** a memory problem — the KV pool had 1.9x headroom.

**The fix**: `sparse_attn_indexer.py` already has a third branch — `ops.top_k_per_row_decode` —
that does not use the persistent kernel, but `use_persistent_topk` is unconditional on CUDA,
so it is never reached. The patch adds an SM121 guard so the per-row path is selected:

```python
and not current_platform.is_device_capability(121)
```

A *fallback-selection* change, not a kernel change: `top_k_per_row_decode` is the established
non-persistent implementation and preserves the requested K.

```bash
python3 tools/engram/persistent_topk_sm12x_fix.py --probe   # show the selection logic
python3 tools/engram/persistent_topk_sm12x_fix.py           # install
python3 tools/engram/persistent_topk_sm12x_fix.py --undo
```

Restart required (the indexer is imported at engine start). Measured on 4x DGX Spark:

| `max-model-len` | before | after |
|---|---|---|
| 262,144 | boots | boots |
| **1,048,576** | **engine init fails** | **boots: 2,454,802 KV tokens, 2.34x concurrency** |

Retrieval verified at the full context: needle-in-haystack at **998,755 tokens — PASS**
(`finish_reason: stop`). Paris gate 4/4.

**Upstream fix**: the per-row path should be selected automatically when the persistent launch
would oversubscribe a low-smem device, rather than relying on a device-family check.
