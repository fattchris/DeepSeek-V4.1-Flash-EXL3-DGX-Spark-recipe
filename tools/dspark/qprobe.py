import json
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.weight_utils import get_quant_config
from dataclasses import replace as _r

ea = EngineArgs(
    model="/models/dsv41-orig", quantization="exl3", tokenizer_mode="deepseek_v41",
    tensor_parallel_size=1, enable_expert_parallel=True, trust_remote_code=True,
    max_model_len=4096, enforce_eager=True, language_model_only=True,
    speculative_config=json.loads('{"method":"dspark","num_speculative_tokens":5,'
        '"draft_sample_method":"probabilistic","rejection_sample_method":"block","quantization":"mxfp4"}'),
)
vc = ea.create_engine_config()
d = vc.speculative_config.draft_model_config
d.quantization = "mxfp4"
qc = get_quant_config(d, vc.load_config)
print("draft quant_config class:", type(qc).__name__)
