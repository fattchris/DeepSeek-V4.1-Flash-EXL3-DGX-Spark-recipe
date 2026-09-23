# Disk-backed Engram

On DGX Spark, Engram runs disk-backed. The TP2 and TP4 profiles both enable it.

## Why

A resident Engram table does not fit. At 8K context, one sequence, eager, the TP4 load reached
about 121 GiB used and about 0.5 GiB `MemAvailable` on each node while materializing Engram.
GB10 has one LPDDR5x pool for both CPU and GPU, so `cpu_offload=True` and pinned-host/UVA
placement add no capacity.

## What it does

The overlay in [`overlays/disk-engram/`](../overlays/disk-engram/README.md):

- never allocates the full Engram embedding table as a resident parameter;
- skips `engram.embed.weight/scale` in the weight iterator;
- reads the table from the safetensors files on node-local NVMe;
- stages only the unique rows each lookup needs into bounded pinned-host and GPU buffers;
- reuses vLLM's FP8/ue8m0 Engram dequant;
- is switched on by `VLLM_ENGRAM_DISK_BACKED=1` (or `EngramConfig.disk_backed`).

It refuses network filesystems. The snapshot must sit on local disk, at the same in-container path
on every node, and `VLLM_ENGRAM_MODEL_DIR` must point at it.

For TP4 CUDA graphs, the `tools/engram/` fixes (baked into the `:tp4` image) stage the Engram hash
and lookup outside graph capture. They also key replay per table and avoid cold-page stalls. The
[`tools/engram/README.md`](../tools/engram/README.md) file describes each fix and what it measured.

## Build

```bash
bash scripts/build_runtime.sh              # deepseek-v41-exl3:spark
bash scripts/build_disk_engram_runtime.sh  # deepseek-v41-exl3:disk-engram
bash scripts/build_tp4_runtime.sh          # deepseek-v41-exl3:tp4 (builds the two above if missing)
```

`Dockerfile.disk-engram` copies the overlay files over the locked runtime. The build fails if the
loader does not skip the Engram tensors.

## Start the nodes

`scripts/start_disk_engram_cluster.sh head|worker` starts Ray with `VLLM_ENGRAM_DISK_BACKED=1`,
`VLLM_ENGRAM_MODEL_DIR` and `VLLM_EXL3_MODEL_DIR` already in the container. vLLM workers inherit
them from there. It reads `DISK_ENGRAM_PROFILE`, which defaults to `profiles/tp4.env`.

## OOM guard

`scripts/oom_guard.sh` (or `scripts/watch_oom_guard.sh` on every node) watches `MemAvailable`. At
24 GiB it warns. At 16 GiB it stops the container named in `OOM_GUARD_CONTAINER_NAME`, and only
that container.
