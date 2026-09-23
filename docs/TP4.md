# TP4 + EP4: four DGX Sparks

This is the preferred DeepSeek-V4.1-Flash EXL3 runtime qualification topology.

**Published artifact:** `vcruz305/DSV4.1-Flash-EXL3-4.75bpw`

## Geometry

- 4 Spark nodes / 4 GB10 GPUs total
- tensor parallel size: 4
- expert parallel: enabled
- 384 routed experts -> **96 whole experts per rank**
- expert matrix: **5120 × 2304**
- top-k: 6

There is no 128-total-expert ExLlamaV3 ceiling.

## Gate 0: physical checkpoint compatibility

The published checkpoint uses heterogeneous K3–K8 **inside routed layers and even across `w1/w2/w3` of one expert**. The pinned `vllm-exl3` now stores exact per-expert/per-projection trellis shapes, so this layout no longer requires a layer-uniform repack.

```bash
ALLOW_SOURCE_ARTIFACT_DOWNLOAD=1 \
  bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-TP4

python3 scripts/validate_pack.py \
  /large/models/DSV4.1-Flash-EXL3-TP4 \
  --topology tp4 \
  --reserve-gib 32
```

Require:

```text
DEPLOYABLE_CURRENT_LOADER=YES
```

That proves the **pack/loader contract**, not full hardware deployment.

The heterogeneous path is correctness-first `LinearEXL3` execution and is not CUDA-graph-qualified yet. First boot must stay eager.

## Gate 1: runtime identity and transport

Every Spark must use the same locked runtime and model revision.

```bash
bash scripts/doctor.sh 4
bash scripts/runtime_identity.sh
bash scripts/preflight.sh 4
bash scripts/cluster_collective.sh 4
```

The collective must pass before model load.

## Gate 2: resident Engram is already a known capacity failure

Corrected resident min-fit testing at **8K / seq1 / eager / text-only / DSpark-off / native-off** drove the GB10 unified-memory pool into the near-full cliff:

```text
RESIDENT_ENGRAM_TP4=CAPACITY_FAIL
~121 GiB used class / ~0.5 GiB MemAvailable class during Engram materialization
```

Do not use resident Engram as the normal qualification gate anymore. `scripts/tp4_min_fit.sh` is retained only for an explicit regression reproduction:

```bash
ALLOW_RESIDENT_ENGRAM_RETEST=1 bash scripts/tp4_min_fit.sh
```

On Spark, pinned-host/UVA Engram does not create physical capacity because CPU and GPU share the same memory pool.

## Gate 3: build and start the disk-Engram runtime

Build the baseline and explicit derivative:

```bash
bash scripts/build_runtime.sh
bash scripts/build_disk_engram_runtime.sh
```

Start every node with the same materialized checkpoint path. The dedicated cluster wrapper propagates disk-Engram and mixed-K prescan environment into the Ray processes before vLLM launches.

Example head:

```bash
IMAGE=deepseek-v41-exl3:disk-engram \
MODEL_DIR=/large/models \
MODEL=/models/DSV4.1-Flash-EXL3-TP4 \
VLLM_ENGRAM_MODEL_DIR=/models/DSV4.1-Flash-EXL3-TP4 \
HEAD_IP=10.0.0.10 NODE_IP=10.0.0.10 \
  bash scripts/start_disk_engram_cluster.sh head
```

Use `worker` on the other three Sparks.

## Gate 4: all-node disk preflight and first load

Arm the OOM guard on every node. The guard is bound to one exact configured container name and uses WARN=24 GiB / ABORT=16 GiB by default.

Then:

```bash
bash scripts/tp4_disk_engram_min_fit.sh --check
bash scripts/tp4_disk_engram_min_fit.sh
```

Before load, the launcher probes all four Ray GPU nodes and requires:

- `VLLM_ENGRAM_DISK_BACKED=1`;
- the same local model/index path;
- disk-Engram overlay import;
- weight-loader Engram skip active;
- mixed-K-capable `vllm-exl3`;
- non-network backing storage.

The first load is locked to:

```text
context:              8192
max seqs:             1
max batched tokens:   1024
text only:            yes
DSpark:               off
native MoE:           off
eager:                on
Engram:               node-local disk-backed
```

## Gate 5: serving correctness

A successful load is not enough. Require:

```bash
bash scripts/smoke_test.sh
```

and exact response:

```text
EXL3 Spark OK
```

Also capture:

- `/v1/models` ready;
- 96 owned experts/rank;
- actual mixed-K dispatch;
- per-rank unified-memory high-water mark;
- full Engram non-residency evidence;
- no dense reconstruction of the routed expert bank.

## After the baseline passes

Only then qualify independently:

1. larger context (32K/64K/128K from measured headroom);
2. DSpark;
3. uniform-K fused/native A/B where eligible;
4. CUDA graphs **only after a mixed-K graph-safe path is implemented/qualified**.

Do not infer 128K viability from the disk-Engram parity tests alone.

## Deployed reference profile (1M context, DSpark on, CUDA graphs on)

Gate 4 above is the *first-boot* gate. A separate, explicitly-qualified deployment currently
serves TP4 with a 1M-token context, DSpark on and decode-only CUDA graphs. Its full launch
surface is checked in so it is reviewable instead of living in `/tmp` on one node:

- [`../profiles/tp4-live-1m.env`](../profiles/tp4-live-1m.env) — launchable profile for `scripts/serve_tp4.sh`
- [`../configs/serve-tp4-live.yaml`](../configs/serve-tp4-live.yaml) — the captured serve config verbatim

What the deployed profile sets beyond the first-boot gate:

| knob | value | why it matters |
|---|---|---|
| `MAX_MODEL_LEN` | `1048576` | 1M-token context |
| `KV_CACHE_MEMORY_BYTES` | `8589934592` (8 GiB) | manual KV budget; **disables** profiling, so `GPU_MEMORY_UTILIZATION` is inert |
| `MAX_NUM_SEQS` | `4` | four concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `4096` | with a draft block, vLLM warns this is below the speculative-scheduling recommendation |
| `BLOCK_SIZE` | `128` | KV block size |
| `PREFIX_CACHING` | `1` | explicit |
| `DSPARK` | `1` | with `NUM_SPECULATIVE_TOKENS=2` and `SPECULATIVE_QUANTIZATION=mxfp4` |
| `EXL3_MOE_KERNEL` | `native` | native kernel while ABI-3 V4.1 native MoE stays OFF (rule 6) |
| `COMPILATION_CONFIG` | `{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12]}` | decode-only graph capture |

Recorded effective values for the captured run (identity: image
`deepseek-v41-exl3:fresh` = `sha256:8c560955…`, `vllm-exl3` @ `814d4fe3…`, `exllamav3` @
`be57335b…`, `TORCH_CUDA_ARCH_LIST=12.1a`, CUDA 13.0.1, TP4+EP4, model path `/models/dsv41-orig`):

```text
kv cache reserved           8.0 GiB (profiling skipped)
GPU KV cache size           2,454,802 tokens
max concurrency @1M req     2.34x
/v1/models max_model_len    1,048,576
```

That run used a **local 429 GB pack** (`/models/dsv41-orig`), not the locked published revision
`vcruz305/DSV4.1-Flash-EXL3-4.75bpw@e971fd55`. Per rule 3, this profile is a reference capture,
**not** a qualification of the locked snapshot, and it does not advance any pin. Treat it as the
starting point for an explicit qualification, and record throughput against the identity above if
numbers are ever published from it (rule 8).

Two operational traps this profile documents:

1. `KV_CACHE_MEMORY_BYTES` and `GPU_MEMORY_UTILIZATION` cannot both be in force. vLLM reserves the
   byte budget and skips memory profiling, so a utilization value set alongside it silently does
   nothing. Choose one mechanism per profile.
2. A stale `VLLM_KV_CACHE_MEMORY_BYTES` environment variable in a launcher is shadowed by the
   `--kv-cache-memory-bytes` argument. The captured deployment carried `17179869184` (16 GiB) in
   the launcher while the served config reserved 8 GiB; delete the environment form when a
   profile uses the flag.
3. The captured launcher exported `VLLM_ENGRAM_DISK_BACKED=1` while the container had **no**
   `VLLM_ENGRAM_MODEL_DIR` and the served config carried only
   `--engram-config {"cpu_offload":false}`. The disk-backed overlay cannot have been reachable in
   that state, and `scripts/serve.sh` fails closed on exactly that combination, so the profile
   leaves the variable out. Which Engram mechanism the deployment intended is an open question for
   the maintainer (rule 9).

See [`DISK_ENGRAM.md`](DISK_ENGRAM.md) for the storage path and overlay details.
