# DeepSeek-V4.1-Flash EXL3 on DGX Spark

Serving recipes for **DeepSeek-V4.1-Flash** in EXL3 on NVIDIA DGX Spark (GB10). There is one
folder per cluster size:

| Sparks | Folder | Model | Engine | Status |
|---|---|---|---|---|
| **4** | [`four-spark-tp4/`](four-spark-tp4/README.md) | [`DSV4.1-Flash-EXL3-4.75bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-EXL3-4.75bpw) | vLLM + `vllm-exl3` | **Serving** |
| **2** | [`two-spark-tp2/`](two-spark-tp2/README.md) | [`DSV4.1-Flash-SAGE-EXL3-3.30bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw) | vLLM + `vllm-exl3` | Loader ready; not benchmarked |
| **1** | [`one-spark-tp1/`](one-spark-tp1/README.md) | [`DSV4.1-Flash-SAGE-EXL3-1.59bpw`](https://huggingface.co/vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw) | native ExLlamaV3 | **Serving** |

## Performance

### Four Sparks (TP4 + EP4)

Single stream, measured with `llm-inference-bench` (30-second sustained cells):

| Measurement | Result |
|---|---:|
| Decode, prose (ctx 2048) | **30.21 tok/s** |
| Decode, structured (ctx 2048) | 29.70 tok/s |
| Decode, code (ctx 0 / 2048) | 27.02 / 27.01 tok/s |
| Max context | **1,048,576 tokens** (boots; 2,454,802 KV tokens; needle test passes at 998,755) |

The decode figures were measured at `max-model-len 4096`, with DSpark k=2 (MXFP4 draft experts),
decode-only CUDA graphs and disk-backed Engram. No multi-stream numbers have been recorded yet.
Serving profile: [`profiles/tp4.env`](profiles/tp4.env). Runtime: `vllm-exl3` `814d4fe` plus the multi-K /
padded MoE kernels from `4c95648`, ExLlamaV3 `be57335`, CUDA 13.0.1, `sm_121a`.
[`four-spark-tp4/`](four-spark-tp4/README.md) has the full runtime identity and the KV budget.

### One Spark (TP1)

Native ExLlamaV3 from the `vcruz305/exllamav3` fork, with the EXL3 attention/MTP overlay:

| Measurement | Result |
|---|---:|
| Decode, no drafter | **15.13 – 15.22 tok/s** |
| Decode, DSpark drafter, fresh prompt | **17.53 median**, 19.82 mean |
| Decode, DSpark drafter, repeat prompt | 20.11 – 24.67 tok/s |
| Prefill, chunk 4096 | 254 – 261 tok/s |

Methodology, runtime identity and negative results:
[`one-spark-tp1/BENCHMARKS.md`](one-spark-tp1/BENCHMARKS.md).

## Quick start: four Sparks

On every Spark:

```bash
git clone https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe.git
cd DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe

bash scripts/build_tp4_runtime.sh
bash scripts/materialize_model.sh 4 /large/models/DSV4.1-Flash-EXL3-4.75bpw

set -a; . profiles/tp4.env; set +a
export MODEL_DIR=/large/models
HEAD_IP=<head-ip> NODE_IP=<this-ip> bash scripts/start_disk_engram_cluster.sh head   # `worker` on the other three
```

On the head only:

```bash
bash scripts/preflight.sh 4
bash scripts/serve_tp4.sh
bash scripts/smoke_test.sh
```

This serves an OpenAI-compatible API on port 8000 under the model name `deepseek-v41-exl3`.
Details and tuning are in [`four-spark-tp4/README.md`](four-spark-tp4/README.md).

For one Spark, follow [`one-spark-tp1/README.md`](one-spark-tp1/README.md). It runs a different
engine: native ExLlamaV3, with no Docker and no Ray.

## How the pieces fit

- **vLLM** owns the DeepSeek-V4.1 graph: attention, compressed sparse attention, Engram, routing
  and the DSpark drafter.
- **[`vllm-exl3`](https://github.com/vcruz305/vllm-exl3)** plugs EXL3 routed-expert storage and
  kernels into vLLM. It supports per-expert mixed K (K2–K8), including different K across
  `w1/w2/w3` of one expert.
- **ExLlamaV3** supplies the EXL3 kernels. On one Spark it also serves the whole model.
- **Engram** is disk-backed on Spark. The embedding table stays on local NVMe and each step stages
  only the rows it needs. A resident Engram table does not fit in the 128 GB unified pool.

| Layout | Local experts | Expert matrix |
|---|---:|---:|
| TP4 + EP4 | 96 | 5120 × 2304 |
| TP2 + EP2 | 192 | 5120 × 2304 |
| pure MoE TP2 (A/B) | 384 | 5120 × 1152 |
| TP1 | 384 | 5120 × 2304 |

## Runtime

[`runtime.lock.json`](runtime.lock.json) pins the vLLM base image, `vllm-exl3`, ExLlamaV3, the
TP4 MoE-kernel ref, the CUDA arch and the model revisions. The build scripts read it, so you do
not pass any of these by hand.

| Image | Built by | Adds |
|---|---|---|
| `deepseek-v41-exl3:spark` | `scripts/build_runtime.sh` | vLLM DeepSeek-V4.1 image + `vllm-exl3` + ExLlamaV3 |
| `deepseek-v41-exl3:disk-engram` | `scripts/build_disk_engram_runtime.sh` | disk-backed Engram overlay |
| `deepseek-v41-exl3:tp4` | `scripts/build_tp4_runtime.sh` | multi-K/padded MoE kernels, DSpark and Engram fixes, SM121 sparse-attention fix |

## Repository layout

```text
four-spark-tp4/     TP4 guide and measured numbers
two-spark-tp2/      TP2 guide, metadata override, earlier offload findings
one-spark-tp1/      native ExLlamaV3 guide, benchmarks, TabbyAPI config, launcher
profiles/           tp4.env (measured), tp2.env
configs/            serve config captured from the measured TP4 deployment
scripts/            build, download, cluster start/stop, preflight, serve, smoke test, OOM guard
tools/engram/       Engram fixes: CUDA-graph hoist, replay keying, cold I/O (baked into :tp4)
tools/dspark/       DSpark drafter fixes and MoE-kernel patches (baked into :tp4)
overlays/           vLLM source overlays: disk Engram, SM121 sparse attention, DSpark-in-checkpoint, H2D prefetch
docs/               disk Engram, compatibility, troubleshooting, SGLang notes
tests/              CPU-only contract tests (run in CI)
```

## Credits

- **@fattchris**: TP4 serving, including the Engram replay/graph/cold-I/O fixes, the SM121
  sparse-attention fix, the DSpark config and the deployed profile. Also the `vllm-exl3` multi-K,
  codebook, padded-MoE and draft-loader PRs and the GB10 build fixes.
- **@Blackwellboy**: per-expert mixed-K loading in `vllm-exl3` and the disk-backed Engram path.
- **@joeynyc**: the SAGE 3.30 two-Spark work.
- **@tiggerite**: the single-layer ExLlamaV3 / `vllm-exl3` Docker build.
- **turboderp**: [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and the EXL3 format.
- **DeepSeek**: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash).

Third-party code and licenses: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

Recipe code is **AGPL-3.0-only**. Model weights, vLLM, ExLlamaV3, CUDA components and container
layers keep their own licenses.
