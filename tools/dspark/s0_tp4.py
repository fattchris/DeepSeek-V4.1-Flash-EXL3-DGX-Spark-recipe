# stage-0 variant with the EXACT serve args (TP4+EP4 via Ray path parity)
import json
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.utils import get_model_architecture

ea = EngineArgs(
    model="/models/dsv41-orig", quantization="exl3", tokenizer_mode="deepseek_v41",
    tensor_parallel_size=1, enable_expert_parallel=True, trust_remote_code=True,
    max_model_len=4096, enforce_eager=True, language_model_only=True,
    enable_prefix_caching=True, block_size=128, gpu_memory_utilization=0.91,
    speculative_config=json.loads('{"method":"dspark","num_speculative_tokens":5,'
        '"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'),
)
vc = ea.create_engine_config()
t, d = vc.model_config, vc.speculative_config.draft_model_config
print("TARGET quant =", t.quantization, "arch =", t.architectures)
print("DRAFT  quant =", d.quantization, "arch =", d.architectures, "n_predict =", getattr(d.hf_config, "n_predict", None))
print("DRAFT  resolves to =", get_model_architecture(d)[0].__name__)
print("use_dspark =", vc.speculative_config.use_dspark())
