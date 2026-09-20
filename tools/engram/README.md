# Disk-engram CUDA-graph fixes

Two independent defects in the disk-backed Engram path, both of which only appear
under **CUDA graph capture/replay** and neither of which shows up eagerly.

Apply both (idempotent, `--undo` supported) inside the serving container on **every
rank**:

```bash
python3 tools/engram/engram_key_fix.py     # correctness under replay
python3 tools/engram/engram_hoist.py       # enables FULL_DECODE_ONLY capture
```

Both are also safe to leave installed under eager and under PIECEWISE.

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

## Results (4x DGX Spark, TP4/EP4, bs1, k=2 spec decode)

| config | ctx0 | repeat | ctx2048 |
|---|---|---|---|
| eager | 14.70 | 21.20 | 15.62 |
| PIECEWISE | 13.84 | 20.03 | 13.80 |
| **FULL_DECODE_ONLY + hoist** | **18.60** | **26.03** | **17.77** |

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
