# TP2: two DGX Sparks

**Model:** [`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw)
(revision `e831e9e4`, 31 shards)
**Engine:** vLLM (DeepSeek-V4.1 image) + `vllm-exl3`, disk-backed Engram
**Status:** the loader supports this pack. No serving numbers published yet.

## Geometry

Two MoE layouts are selectable with `MOE_PARALLEL_MODE`:

| Mode | Layout | Local experts | Expert matrix |
|---|---|---:|---:|
| `ep` (default) | TP2 + EP2 | 192 | 5120 × 2304 |
| `tp` (A/B) | pure MoE TP2 | 384 | 5120 × 1152 |

1152 is exactly 9 × 128, so pure MoE TP2 needs no width padding. SGLang's TP4 run needed 576→640
padding, but that penalty does not apply here. Compare the two modes one variable at a time.

The pack mixes K2–K8 per expert and per projection. The pinned `vllm-exl3` keeps each tensor's
exact trellis shape, so no layer-uniform repack is needed.

## Run it

```bash
# on both Sparks
bash scripts/build_runtime.sh
bash scripts/build_disk_engram_runtime.sh
bash scripts/materialize_model.sh 2 /large/models/DSV4.1-Flash-SAGE-EXL3-3.30bpw

set -a; . profiles/tp2.env; set +a
export MODEL_DIR=/large/models DISK_ENGRAM_PROFILE=profiles/tp2.env
HEAD_IP=<head-ip> NODE_IP=<this-ip> bash scripts/start_disk_engram_cluster.sh head   # `worker` on Spark 2

# head only
bash scripts/preflight.sh 2
bash scripts/serve_tp2.sh
bash scripts/smoke_test.sh
```

[`profiles/tp2.env`](../profiles/tp2.env) is a conservative first-load profile: 8K context, one
sequence, eager, drafter off. Raise the context and turn on DSpark once it serves.
`MOE_PARALLEL_MODE=tp bash scripts/serve_tp2.sh` runs the pure-MoE-TP2 A/B.

### Why the profile passes `HF_OVERRIDES_JSON`

This snapshot's `config.json` predates the mixed-format fields the runtime reads
(`non_routed_quantization`, `mtp_experts`, `mtp_experts_start_layer`). The profile passes them at
launch through `--hf-overrides`, so the model files stay unchanged. The values match
[`tp2-e831e9e4-metadata.json`](tp2-e831e9e4-metadata.json), which records the snapshot they were
derived from.

## KV cache

DeepSeek-V4.1 has four persistent global-cache producers (layers 2, 8, 14, 20), and the first three
are pooled 2:1. The logical global KV+index floor is **890 bytes** per original token. Real vLLM
allocation is higher once backend layout, local SWA, padding and workspaces are counted. Size the
context from the KV figure in the boot log, not from the floor.

## Earlier results

[`HISTORICAL_OFFLOAD_FINDINGS.md`](HISTORICAL_OFFLOAD_FINDINGS.md) records an earlier two-Spark run
of this pack. It streamed routed experts and Engram from NVMe and decoded at about 4–6 tok/s. The
mechanisms are documented there. Those numbers predate the current loader.
