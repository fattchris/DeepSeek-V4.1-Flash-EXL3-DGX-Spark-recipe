# TP4: four DGX Sparks (TP4 + EP4)

**Model:** [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw)
(revision `b0c44df`, 33 shards, ~460 GB)
**Engine:** vLLM (DeepSeek-V4.1 image) + `vllm-exl3` with the multi-K / padded MoE kernels
**Status:** serving

## Performance

Verified by **@fattchris** on 4× DGX Spark (GB10), TP4 + EP4, with `llm-inference-bench`. Every
figure is a 30-second sustained cell:

| Measurement | Result |
|---|---:|
| Single-stream decode, prose | **30.2 tok/s** |
| Single-stream decode, code | **33.4 tok/s** |
| Best warm single stream | 33.37 tok/s |
| Aggregate decode, c = 1 / 2 / 4 / 8 | 27.8 / 43.9 / 57.7 / 61.5 tok/s |
| KV budget | 3.97M tokens |
| Max context | **1,048,576 tokens** (full 1M) |

Settings: DSpark k=2 with MXFP4 draft experts, `FULL_DECODE_ONLY` CUDA graphs, disk-backed Engram.

Per-domain single-stream matrix from the same work ([PR #41](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe/pull/41),
c=1, `max_tokens 512`):

| Domain | ctx 0 | ctx 2048 | Draft acceptance |
|---|---:|---:|---:|
| prose | 27.44 | 30.21 | 1.97 / 2.21 |
| structured | 27.73 | 29.70 | 1.99 / 2.14 |
| code | 27.02 | 27.01 | 1.89 / 1.92 |

At 1M context, a needle-in-a-haystack test passes at 998,755 tokens. Per-fix before/after tables:
[`tools/engram/README.md`](../tools/engram/README.md).

Serving configuration: [`profiles/tp4.env`](../profiles/tp4.env). It sets 1M context,
`max_num_seqs=4`, the DSpark drafter at k=2 with MXFP4 draft experts, decode-only CUDA graphs
(`FULL_DECODE_ONLY`, capture sizes 3/6/9/12), prefix caching and disk-backed Engram. Runtime:
base image `vllm/vllm-openai:deepseekv41-flash-0909`, `vllm-exl3` `814d4fe` + MoE kernels from
`4c95648` (PR #33) with `P2B_CB=2`, ExLlamaV3 `be57335`, CUDA 13.0.1, `TORCH_CUDA_ARCH_LIST=12.1a`.
The serve config captured from the deployment is
[`configs/serve-tp4-live.yaml`](../configs/serve-tp4-live.yaml).

`profiles/tp4.env` reserves **13 GiB** of KV per rank (`KV_CACHE_MEMORY_BYTES=13958643712`),
enough for the 3.97M-token budget. The captured 8 GiB config
([`configs/serve-tp4-live.yaml`](../configs/serve-tp4-live.yaml)) booted with 2,454,802 tokens, and
the budget scales linearly with the byte count.

## TP-MoE (EP1): current best serving configs

An alternative to the EP4 path above: route MoE through tensor parallelism
(`enable-expert-parallel false`, `enable-ep-weight-filter false`,
`MOE_PARALLEL_MODE=tp`) instead of expert parallelism. Requires
**[vllm-exl3 PR #36](https://github.com/vcruz305/vllm-exl3/pull/36)**
(Hadamard-aligned uneven TP MoE, `VLLM_EXL3_MOE_TP_ALIGN=128`) and
**[vllm-exl3 PR #37](https://github.com/vcruz305/vllm-exl3/pull/37)**
(padded-MoE loops bounded by `n_valid`). Also sets `VLLM_EXL3_TRELLIS_ARENA=0`.
Cold load is ~9 minutes.

Measured by **@fattchris** on 4x DGX Spark (GB10), TP4, DSpark, with
`llm-inference-bench` v0.4.29 (20s sustained cells), 2026-09-24.

Two profiles, both in [`profiles/`](../profiles) and
[`configs/`](../configs):

| Profile | Env file | Config | Speculative k | Use case |
|---|---|---|---|---|
| all-round (default) | [`profiles/tp4-tpmoe.env`](../profiles/tp4-tpmoe.env) | [`configs/serve-tp4-tpmoe-k2.yaml`](../configs/serve-tp4-tpmoe-k2.yaml) | 2 | mixed / high-concurrency |
| code-heavy | [`profiles/tp4-tpmoe-k4.env`](../profiles/tp4-tpmoe-k4.env) | [`configs/serve-tp4-tpmoe-k4.yaml`](../configs/serve-tp4-tpmoe-k4.yaml) | 4 | low-concurrency, code-focused |

### k=2 (default): decode tok/s by context x concurrency

| Context | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | c10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 36.5 | 49.4 | 63.1 | 62.7 | 68.7 | 71.5 | 80.4 | 80.1 | 87.1 | 87.2 |
| 16k | 31.2 | 49.4 | 62.3 | 65.1 | 72.5 | 75.3 | 81.4 | 82.3 | 87.4 | 89.7 |
| 32k | 30.9 | 50.1 | 61.9 | 64.0 | 74.5 | 77.5 | 81.4 | 85.1 | 88.3 | 90.4 |
| 64k | 33.7 | 52.0 | 58.3 | 65.0 | 69.8 | 70.2 | 80.6 | 82.4 | 83.2 | 87.1 |
| 128k | 35.4 | 50.4 | 59.3 | 62.8 | 70.5 | 72.1 | 81.7 | 85.9 | 87.0 | 91.0 |

`--coding-peak` (c1, 3 runs): **44.6 tok/s**. The bench's default decode
prompt is prose-like; use `--coding-peak` to measure code workloads.

### k=4 (code-heavy, DSpark's trained max at `dspark_block_size=5`)

| Cell | k=2 | k=4 | Delta |
|---|---:|---:|---:|
| coding-peak (c1) | 44.6 | **50.2** | +13% |
| bench decode, c1 | 36.5 | 32.9 | -9% |
| bench decode, c2 | 49.4 | 45.4 | -8% |
| bench decode, c4 | 62.7 | 53.2 | -15% |
| bench decode, c8 | 80.1 | 71.8 | -10% |
| bench decode, c10 | 87.2 | 71.6 | -18% |

k=4 trades mixed/high-concurrency throughput for peak single/low-concurrency
code decode speed. Its capture sizes are multiples of 5 up to 50
(`PADDED_MAX_T=64`, `NATIVE_MOE_MAX_ROWS=64`) to match `rows = max-num-seqs *
(k+1)`.

### Draft acceptance by content (k=2, mean accept length, max 3.0)

| Content | temp 0 | temp 0.6 | temp 1.0 |
|---|---:|---:|---:|
| code | 2.69 | 2.70 | 2.72 |
| prose | 1.82 | 1.91 | 2.01 |

Code acceptance is stable across temperature; prose acceptance is lower and
more temperature-sensitive. This is also why the bench's default decode
prompt (prose-like) undercounts code throughput — use `--coding-peak` there.

### Config guidance

- `PADDED_MAX_T` / `NATIVE_MOE_MAX_ROWS` should be sized for the largest
  batch (`rows = max-num-seqs * (k+1)`). With PR #37 landed, headroom above
  the real row count is free; without PR #37, set it to the exact row count
  or padded-loop rows beyond `n_valid` do real work.
- History of this line of tuning, single-stream c1: control 28.5 tok/s ->
  `PADDED_MAX_T=4` 31.1 tok/s -> TP-MoE 36.0 tok/s.

## Geometry

- 4 Sparks, one GB10 each; tensor parallel 4, expert parallel on
- 384 routed experts → **96 whole experts per rank**, expert matrix **5120 × 2304**, top-k 6
- Pure MoE TP4 (576-wide local experts, not 128-aligned) stays behind
  `ALLOW_EXPERIMENTAL_TP4_MOE_TP=1`

## Run it

All four Sparks need the same image, the same snapshot at the same path, and working
Spark-to-Spark networking.

**1. Build the TP4 image** (on each Spark, or build once and `docker save | docker load`):

```bash
bash scripts/build_tp4_runtime.sh
```

The build has three layers: the locked base (`deepseek-v41-exl3:spark`), then disk-backed Engram
(`:disk-engram`), then `:tp4`. The `:tp4` layer rebuilds `vllm_exl3_c` with the multi-K and padded
MoE kernels, applies `tools/dspark/*.patch`, and runs every patch script in
`overlays/sm120-sparse-fix/`, `tools/dspark/` and `tools/engram/`. Each script stops the build if its
anchor is missing, so nothing needs patching by hand inside a running container.

**2. Download the model** (on each Spark):

```bash
bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-4.75bpw
```

**3. Start Ray** on every Spark:

```bash
set -a; . profiles/tp4.env; set +a
export MODEL_DIR=/large/models
HEAD_IP=<head-ip> NODE_IP=<head-ip>  bash scripts/start_disk_engram_cluster.sh head
HEAD_IP=<head-ip> NODE_IP=<this-ip>  bash scripts/start_disk_engram_cluster.sh worker   # other three
```

**4. Check and serve** from the head:

```bash
bash scripts/preflight.sh 4      # snapshot complete + NCCL all-reduce across the 4 nodes
bash scripts/serve_tp4.sh
bash scripts/smoke_test.sh       # /v1/models + one deterministic completion
```

`DRY_RUN=1 bash scripts/serve_tp4.sh` prints the exact `vllm serve` command without loading.

Optional: `scripts/oom_guard.sh` / `scripts/watch_oom_guard.sh` run a MemAvailable watchdog on
each host. It stops the recipe container before the unified-memory pool runs out.

## Tuning notes

- **KV budget vs utilization.** With `KV_CACHE_MEMORY_BYTES` set, vLLM reserves that budget and
  skips memory profiling, so `GPU_MEMORY_UTILIZATION` has no effect. Use one or the other.
- **Stale env var.** When `--kv-cache-memory-bytes` is passed, a `VLLM_KV_CACHE_MEMORY_BYTES`
  exported in the launcher is ignored. The original launcher still carried `17179869184` (16 GiB)
  while the served config reserved 8 GiB. The profile uses only the flag.
- **Engram must be disk-backed on Spark.** Resident Engram fills the 128 GB unified pool on every
  rank, even at 8K context. See [`docs/DISK_ENGRAM.md`](../docs/DISK_ENGRAM.md).
- **Drafter width.** `NUM_SPECULATIVE_TOKENS=2` was the fastest setting on 4× GB10. `serve.sh`
  defaults to 5 when a profile does not set it.
- **Max batched tokens.** 4096 is below vLLM's recommendation for speculative scheduling, so vLLM
  warns about it. 4096 is the value that was measured.
- **CUDA graphs.** `FULL_DECODE_ONLY` depends on `tools/engram/engram_hoist.py`. Without it,
  capture fails on a host-side copy. Without `engram_key_fix.py`, graph replay silently reads the
  wrong Engram table and draft acceptance drops to 1.00.

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md). The quickest fallback is
`EAGER=1 DSPARK=0 MAX_MODEL_LEN=8192`. It separates graph and drafter problems from load problems.
