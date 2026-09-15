#!/usr/bin/env python3
"""Preflight the experimental one-GPU / large-host-RAM V4.1 EXL3 path.

This checks runtime capabilities only. It does not allocate the model or prove
that EXL3 kernels are correct/fast over mapped host memory.
"""
from __future__ import annotations

import json
import sys


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    import torch
    import vllm
    import vllm_exl3
    from vllm.config import EngramConfig, UVAOffloadConfig
    from vllm.utils.platform_utils import is_uva_available

    if not torch.cuda.is_available():
        fail("CUDA is not available")

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    capability = tuple(int(x) for x in torch.cuda.get_device_capability(device))
    arch_list = list(torch.cuda.get_arch_list())

    if capability != (12, 0):
        fail(
            "This preflight is for the sm_120 Blackwell qualification path; "
            f"detected capability={capability} ({name})"
        )
    if not is_uva_available():
        fail("vLLM reports UVA unavailable on this host")

    engram = EngramConfig()
    offload = UVAOffloadConfig()
    if not hasattr(engram, "cpu_offload"):
        fail("This vLLM revision lacks EngramConfig.cpu_offload")
    if not hasattr(offload, "cpu_offload_params"):
        fail("This vLLM revision lacks selective UVA cpu_offload_params")

    # Import the class the way vLLM's model registry does: the package export.
    # nvidia/model.py only defines DeepseekV41LLMForCausalLM; the registered
    # DeepseekV41ForCausalLM lives in nvidia/vl_model.py and is re-exported
    # from the package.
    try:
        from vllm.models.deepseek_v4_1 import DeepseekV41ForCausalLM
    except Exception as exc:
        fail(f"DeepSeek-V4.1 NVIDIA model import failed: {type(exc).__name__}: {exc}")
    if DeepseekV41ForCausalLM is None:
        fail("DeepseekV41ForCausalLM import returned no class")

    vllm_exl3.register()
    diagnostics = vllm_exl3.runtime_diagnostics()
    uva = diagnostics.get("uva_expert_offload", {})
    if not uva.get("guard_installed"):
        fail("vllm-exl3 UVA expert placement guard is not installed")

    result = {
        "status": "capability-preflight-pass",
        "qualification": "not an end-to-end model pass",
        "gpu": name,
        "compute_capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": arch_list,
        "vllm": getattr(vllm, "__version__", "unknown"),
        "vllm_uva_available": True,
        "vllm_engram_cpu_offload": True,
        "vllm_selective_weight_uva": True,
        "vllm_deepseek_v41_nvidia_model": True,
        "vllm_exl3_uva_guard": uva,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
