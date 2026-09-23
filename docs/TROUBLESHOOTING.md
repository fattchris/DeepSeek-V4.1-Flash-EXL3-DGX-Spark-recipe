# Troubleshooting

## Settings look wrong

Precedence is: shell environment > profile > `.env` > `runtime.lock.json` defaults. Run
`DRY_RUN=1 bash scripts/serve_tp4.sh` (or `serve_tp2.sh`). It prints the exact `vllm serve` command
and every resolved setting without loading the model.

## Incomplete download

`bash scripts/preflight.sh 4` lists missing shards. Rerun `scripts/materialize_model.sh`; it
resumes. The 4.75bpw pack has **33** shards. A copy downloaded before revision `b0c44df` has 32
and lacks the backbone (attention, shared experts, norms), which produces garbage output. Rerun the
download to fetch shard 33, the new index and `config.json`.

## `64 vs 48` trellis shape error

An old `vllm-exl3` with layer-uniform K allocation. Rebuild the image from the pinned revision:
`bash scripts/build_runtime.sh`.

## Resident Engram hits the memory cliff

Expected. Engram must be disk-backed on Spark. Use a profile from `profiles/`; both set
`VLLM_ENGRAM_DISK_BACKED=1`. If load still starts materializing `engram.embed.weight`, the node is
running the base image instead of `:disk-engram` / `:tp4`, or it was started without
`scripts/start_disk_engram_cluster.sh`.

## One node fails the disk-Engram setup

Common causes:

- the node runs a different image;
- `VLLM_ENGRAM_DISK_BACKED=1` was not set when the Ray container started;
- `VLLM_ENGRAM_MODEL_DIR` differs between nodes, or the snapshot is missing on one of them;
- the snapshot is on NFS/CIFS (disk Engram needs local storage).

Restart every node through `scripts/start_disk_engram_cluster.sh` with the same profile.

## CUDA-graph capture fails, or draft acceptance is exactly 1.00

The node is not running the `:tp4` image. `engram_hoist.py` is what makes `FULL_DECODE_ONLY`
capture possible. Without `engram_key_fix.py`, graph replay reads the wrong Engram table. Rebuild
with `scripts/build_tp4_runtime.sh`. To confirm that load works on its own, fall back to
`EAGER=1 DSPARK=0`.

## `sparse_attn_indexer` fails at long context on GB10

The SM121 top-k fallback needs `tools/engram/persistent_topk_sm12x_fix.py`, which ships in the
`:tp4` image. Without it, the 1M-context profile fails engine init.

## Ray sees fewer GPUs than expected

```bash
bash scripts/cluster_status.sh
```

Each Spark contributes one GPU. Check `HEAD_IP`, the per-node `NODE_IP`, host networking and the
Ray port. `bash scripts/preflight.sh 4` then runs a real NCCL all-reduce.

## RDMA / RoCE

```text
ENABLE_RDMA=auto   # default
ENABLE_RDMA=1      # require RDMA
ENABLE_RDMA=0      # plain TCP
```

`ENABLE_RDMA=0` separates generic Ray/NCCL problems from RoCE tuning.

## An existing container blocks startup

The scripts never remove a running container on their own. Use `REPLACE_CONTAINER=1` when you
mean to replace it, or `bash scripts/stop_cluster.sh`.

## TP4 shows expert width 576

EP is off. TP4 + EP4 keeps whole 5120 × 2304 experts, 96 per rank. The profiles set
`MOE_PARALLEL_MODE=ep`.

## Smoke test returns HTTP 200 but fails

`smoke_test.sh` checks the response content, not only the status code. A fluent but wrong answer
means the load is broken. The usual cause is a snapshot missing shard 33.
