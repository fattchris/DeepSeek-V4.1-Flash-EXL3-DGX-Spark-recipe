import sys

p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dspark/utils.py"
s = open(p).read()
if "_DRAFT_QUANT_OVERRIDE" in s:
    print("already"); sys.exit(0)

a = """    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)"""
b = """    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)
    # _DRAFT_QUANT_OVERRIDE: speculative-config 'quantization' must win over the
    # checkpoint's global block (pack config.json says exl3 for the main experts;
    # the mtp.* draft stages are MXFP4 w{1,2,3}.{weight,scale}).
    _q = getattr(speculative_config, "quantization", None)
    if _q and draft_model_config.quantization != _q:
        from vllm.model_executor.layers.quantization import get_quant_config as _gqc
        draft_model_config.quantization = _q
        draft_vllm_config.quant_config = _gqc(draft_model_config, vllm_config.load_config)
        logger.info("DSpark draft quant override: %s -> %s (%s)",
                    "checkpoint", _q, type(draft_vllm_config.quant_config).__name__)"""
assert a in s, "quant override anchor"
s = s.replace(a, b, 1)
compile(s, p, "exec")
open(p, "w").write(s)
print("draft quant override installed")
