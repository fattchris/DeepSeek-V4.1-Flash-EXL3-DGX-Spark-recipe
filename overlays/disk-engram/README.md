# Disk-Engram vLLM overlay

This directory contains the **experimental node-local disk-backed Engram overlay** contributed in recipe PR #2 by **@Blackwellboy** and hardened during mainline integration.

Original contributor commit:

```text
65c160e565fa4f0a01e55f433ab5868207f04401
```

The overlay is derived from the DeepSeek V4.1 vLLM code shipped by the recipe's locked base image:

```text
vllm/vllm-openai:deepseekv41-flash-0909
```

The exact upstream vLLM source commit represented by that image tag is not asserted here because it has not been independently resolved. Do not invent or substitute one. Re-qualify this overlay whenever the base image changes.

## Files

| Path | Role |
|---|---|
| `vllm/config/engram.py` | adds `EngramConfig.disk_backed` |
| `vllm/models/deepseek_v4_1/common/engram.py` | wires disk-backed lookup into V4.1 Engram |
| `vllm/models/deepseek_v4_1/common/engram_disk.py` | node-local bounded row staging |
| `vllm/model_executor/model_loader/weight_utils.py` | skips full Engram tensor materialization and reclaims consumed safetensors page cache |

The mixed-K EXL3 implementation does **not** live in this overlay. It is in `vcruz305/vllm-exl3`, where @Blackwellboy's PR #10 is merged and pinned by `runtime.lock.json`.

## Supported application path

Do not manually overwrite site-packages for normal qualification. Build the reproducible derived image:

```bash
bash scripts/build_runtime.sh
bash scripts/build_disk_engram_runtime.sh
```

This produces:

```text
deepseek-v41-exl3:disk-engram
```

`Dockerfile.disk-engram` applies only the explicit files above to the already locked Spark runtime and performs syntax/import assertions. It also corrects a typo in the PR #2 comparison-only eager loader branch before the image is accepted.

Manual bind mounts are reserved for targeted debugging because they bypass those build assertions.

## Use

Start each Spark with `bash scripts/start_disk_engram_cluster.sh head|worker`, then serve with
`scripts/serve_tp4.sh` or `scripts/serve_tp2.sh`. See `docs/DISK_ENGRAM.md`.

## License / provenance

The copied/adapted vLLM files retain the upstream Apache-2.0 license and copyright notices. New recipe integration/orchestration remains under this repository's license. See `THIRD_PARTY_NOTICES.md` for attribution details.
