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
