# SM120 (GB10) sparse-MLA contract fixes

Patches that make the DeepSeek V4.1 Flash sparse-MLA path dispatchable on
DGX Spark (SM121/GB10) with the 4.75bpw pack at TP4+EP4. Without these,
warmup dies in sequence at:

1. `SM120 sparse-MLA has no decode kernel for this shape: ... topk=1152`
   (vision-padded SWA rows reach the kernel; decode/prefill kernels are
   compiled for TK = window_size only)
2. `Assertion error (deepgemm attention.hpp:262): block_kv == 32 or 64`
   (indexer/compressed specs advertise cache_config.block_size; pages must
   hold exactly 64 states)
3. `ValueError: No common block size for 64` (adverts veto 64)
4. `Unsupported sparse-MLA prefill configuration: ... extra_page_block_size=128`
   (dual-cache prefill dispatcher only instantiated PBSX ∈ {64, 2})

## What each patch does

- `_causal_swa_window` (flashinfer_sparse.py): narrow vision-padded SWA rows
  (window + max_image_tokens) to `window_size` with a contiguous slice and an
  on-device length assert, applied at BOTH `_forward_decode` and
  `_forward_prefill`. Never clamps lengths — discarding valid candidates
  would silently corrupt attention.
- Indexer + compressed-KV specs: `block_size = 64 * compress_ratio`
  (64 states/page satisfies deepgemm {32,64}, prefill-dual {64,2}, and the
  decode page contract simultaneously).
- Adverts: `DeepseekV4IndexerBackend`/`DeepseekV4SparseMLABackend`
  `[64 if fam90 else 128]` → `[MultipleOf(64)]`; CUTLASS `[128]` →
  `[MultipleOf(64)]`; SWA `get_preferred_block_size → 256` override removed.
- SWA cache literal 32 → 64 (the decode kernel's page contract).
- flashinfer `sparse_mla_sm120_prefill.cu`: widen `dispatch_dsv4_dual`
  `extra_page_block_size` to {2, 32, 64, 128} (address math only). Requires
  clearing the prebuilt JIT cache
  (`flashinfer_jit_cache/jit_cache/sparse_mla_sm120/`) so the module
  rebuilds from source.

## Apply

Run `apply.sh` inside each container after the exl3 plugin install (order
with respect to other overlays does not matter; idempotent).

## Verified

4× DGX Spark (GB10), TP4+EP4, vllm-exl3 @ 814d4fe, after the backbone fix
(recipe issue #13): warmup passes all dispatch; 11.3 tok/s decode,
1.1k tok/s prefill, TTFT 1.3s.
