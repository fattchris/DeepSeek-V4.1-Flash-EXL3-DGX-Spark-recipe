# DSpark spec-decode enablement for the EXL3 pack (4x DGX Spark, verified)

Measured on 4x GB10 TP4+EP4: **16.78 / 15.62 tok/s @ conc 1 (ctx 0/2048)** vs 13.13
non-spec baseline (+28%), Paris quality gate 4/4 at temp 0, mean acceptance 2.0-2.6
at k=5. The draft (mtp.*) stages are MXFP4; attention/dense are MXFP8 (F8_E4M3 +
E8M0 on a plain [N/32, K/32] grid).

Spec config (serve yaml):
  speculative-config:
    method: dspark
    num_speculative_tokens: 5
    draft_sample_method: probabilistic
    rejection_sample_method: block
    quantization: mxfp4        # the draft quant override — REQUIRED

Patch scripts (idempotent, run in container, in order):
  1. dq.py        - spec_decode/dspark/utils.py: _DRAFT_QUANT_OVERRIDE
  2. mkdq8.py     - dspark.py: _DQ8 MXFP8 dequant-at-load (draft attn -> bf16)
  3. fixswa.py    - flashinfer_sparse.py: DSpark non-causal SWA rows pass at
                    native width 192 (TOPK=192 template clamps by topk_length);
                    fixes the SM120 position-128 device assert

Stage-0 probes (no boot): s0_tp4.py, qprobe.py

Patch files (not scripts — apply to a vllm-exl3 checkout, not inside a container):

  exl3_native_multik.patch, exl3_padded.patch
    Python-side wiring for the native MoE entry points. They apply cleanly to
    vllm-exl3 814d4fe (`git apply --check` rc=0), but the kernels they call are
    shipped by the **still-open** vllm-exl3 PRs:

      #31  multi-K fused MoE   (p2b_fused_moe_mk)
      #32  codebook P2B_CB     (build-time codebook; required for mul1 packs)
      #33  padded MoE         (p2b_fused_moe_padded, the CUDA-graph path)

    Build the extension from PR #33 (it contains #31 and #32) — see
    tools/multik/README.md for the build flags, which matter:
    TORCH_CUDA_ARCH_LIST=12.1a, NVCC_APPEND_FLAGS, and -DP2B_CB=<1|2>.

Store-scripts note: an earlier revision of this directory carried a full 4,729-line
copy of vllm-exl3's `src/vllm_exl3/exl3.py`. Nothing referenced it, it was not the
pinned revision, and it duplicated ~70% of this PR's bytes; it has been removed.
Use the patches above against your own vllm-exl3 checkout instead.
