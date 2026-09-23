# TP4: four DGX Sparks (TP4 + EP4)

**Model:** [`vcruz305/DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw)
(revision `b0c44df`, 33 shards, ~460 GB)
**Engine:** vLLM (DeepSeek-V4.1 image) + `vllm-exl3` with the multi-K / padded MoE kernels
**Status:** serving

## Performance

4× DGX Spark (GB10), TP4 + EP4, single stream (c=1), DSpark k=2 with MXFP4 draft experts,
`FULL_DECODE_ONLY` CUDA graphs, disk-backed Engram. Measured with `llm-inference-bench`; each figure
is a 30-second sustained cell with `max_tokens 512` at `max-model-len 4096`:

| Domain | ctx 0 | ctx 2048 | Draft acceptance |
|---|---:|---:|---:|
| prose | 27.44 | **30.21** | 1.97 / 2.21 |
| structured | 27.73 | 29.70 | 1.99 / 2.14 |
| code | 27.02 | 27.01 | 1.89 / 1.92 |

Warm repeat prompt: about 26 tok/s. Eager mode on the same stack: 14.70 tok/s at ctx 0.

**1M context.** With `max-model-len 1048576` the engine boots and reports **2,454,802 tokens** of KV
(8 GiB KV per rank, 2.34 concurrent 1M-token requests). A needle-in-a-haystack test at 998,755
tokens passes. No tok/s figure was recorded at 1M.

No multi-stream (c > 1) numbers have been recorded yet. Per-fix before/after tables are in
[`tools/engram/README.md`](../tools/engram/README.md).

Serving configuration: [`profiles/tp4.env`](../profiles/tp4.env). It sets 1M context,
`max_num_seqs=4`, the DSpark drafter at k=2 with MXFP4 draft experts, decode-only CUDA graphs
(`FULL_DECODE_ONLY`, capture sizes 3/6/9/12), prefix caching and disk-backed Engram. Runtime:
base image `vllm/vllm-openai:deepseekv41-flash-0909`, `vllm-exl3` `814d4fe` + MoE kernels from
`4c95648` (PR #33) with `P2B_CB=2`, ExLlamaV3 `be57335`, CUDA 13.0.1, `TORCH_CUDA_ARCH_LIST=12.1a`.
The serve config captured from the deployment is
[`configs/serve-tp4-live.yaml`](../configs/serve-tp4-live.yaml).

To get more KV, raise `KV_CACHE_MEMORY_BYTES`. KV capacity grows linearly with the byte budget.
This has not been re-measured above 8 GiB.

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
