# AGENTS.md

Guidance for automated agents and contributors working in this recipe repository.

## Scope

This repo is a **serving recipe**, not the EXL3 quantizer and not a fork of DeepSeek/vLLM. Keep model architecture changes in upstream/fork repos and EXL3 plugin changes in `vcruz305/vllm-exl3`.

## Non-negotiable rules

1. Preserve both TP2 and TP4 recipes. Do not optimize one by silently breaking the other.
2. Never replace the dedicated DeepSeek V4.1 base image with stock pip vLLM. The V4.1 architecture is image-pinned.
3. Do not silently advance `vllm-exl3`, ExLlamaV3, the base image, CUDA flags or a model revision. Update `runtime.lock.json` and the matching README together.
4. TP4+EP4 is the correctness-first path: 96 whole main experts/rank at 5120 x 2304.
5. TP2+EP2 is experimental: 192 whole main experts/rank and a much tighter memory budget.
6. The ABI-3 V4.1 native MoE path must remain opt-in until hardware evidence supports changing that policy.
7. Do not claim skipped CUDA tests as passes.
8. Do not publish throughput numbers without recording exact runtime identity, model revision, topology, context, batch, speculative policy and actual EXL3 backend.
9. Keep Engram variants explicit. A disk-backed or node-local Engram patch is a separate experimental variable, not a hidden part of the baseline.
10. Preserve third-party attribution. If code is copied/adapted, add exact source URLs/commits/files and licensing to `THIRD_PARTY_NOTICES.md`.

## Checks before merging

At minimum:

```bash
bash -n scripts/*.sh
python -m pytest -q tests
python -m json.tool configs/quantization_config.example.json >/dev/null
```

For runtime changes, build the image on a DGX Spark and run:

```bash
bash scripts/preflight.sh 4
bash scripts/smoke_test.sh
```

## Benchmark discipline

Keep these separate:

- kernel microbenchmark;
- full-model decode speed;
- prefill throughput / TTFT;
- aggregate multi-request throughput;
- DSpark acceptance and speedup;
- long-context memory behavior.

Do not combine warm prefix-cache hits with cold TTFT measurements.
