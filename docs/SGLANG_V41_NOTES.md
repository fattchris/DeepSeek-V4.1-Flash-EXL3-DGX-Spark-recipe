# SGLang DeepSeek V4.1 notes

Reference reviewed 2026-09-12:
https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization?v=4

This recipe does **not** copy SGLang kernels. The article served only as an
architecture/performance reference.

## What carried over

1. **V4.1 cache accounting differs from the old V4 43-layer MLA formula.** V4.1 has four
   persistent global-cache producers (layers 2, 8, 14, 20), and the first three are pooled 2:1.
   The logical global KV+index floor is 890 bytes per original token. Real allocation is higher,
   so read the KV size from the boot log.
2. **Check which kernel actually runs.** The article's biggest plain-decode gain came from reaching
   the intended Blackwell MXFP8 path after fixing scale layout. The same kind of issue showed up
   here: the DSpark draft needs its MXFP4/MXFP8 quant overrides (`tools/dspark/`) to reach the
   fast path.
3. **MoE TP is worth an A/B against EP.** SGLang reduced rank-wait skew with MoE TP4 after
   padding 576→640. For two Sparks, 2304/2 = 1152 is already 128-aligned, so pure MoE TP2 is
   available as `MOE_PARALLEL_MODE=tp` with no padding.
4. **Speculative verify is its own kernel regime.** On TP4 the multi-K and padded MoE kernels
   (`vllm-exl3` #31–#33) handle heterogeneous-K verify batches, and CUDA graphs cover them.
5. **GB300 results do not transfer to Spark as-is.** Their topology and throughput figures are
   hypotheses to test on GB10, not expected Spark numbers.

## Status of the follow-ups

- Grouped/fused heterogeneous mixed-K MoE dispatch: **done** (`p2b_fused_moe_mk`, in the `:tp4` image).
- CUDA graphs over mixed-K: **done** (`p2b_fused_moe_padded` + `FULL_DECODE_ONLY`, TP4).
- Pure MoE TP4 with 576→640 padding: not implemented.
- EP2 vs pure-MoE-TP2 A/B on real hardware: not measured yet.
