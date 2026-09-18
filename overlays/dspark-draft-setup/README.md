# DSpark draft-delegation setup (config fixes + WIP notes)

Enabling spec-decode on the 4.75bpw pack requires pack config.json additions
the pack does not ship. Verified fixes (each by boot):

1. `quantization_config.non_routed_quantization` = {deepseek_v4_fp8, dynamic,
   [32,32]} — must live INSIDE the quant block (Exl3Config.from_config reads it
   there). Top-level placement breaks the dense scale mapper with
   `KeyError: layers.0.attn.fused_wqa_wkv.weight_scale_inv`.
2. text_config gains (from the official base config): num_nextn_predict_layers=3,
   dspark_block_size=5, dspark_noise_token_id=128799,
   dspark_target_layer_ids=[37,38,39], dspark_markov_rank=256,
   dspark_n_routed_experts=128, dspark_num_experts_per_tok=3.
3. The plugin Exl3Config needs the weight_block_size property (PR#16 lineage)
   reading non_routed_quantization, or mixed-quant delegation misroutes.

## Known remaining blocker
The draft model still resolves to the target class at draft get_model and
streams the full checkpoint (w13_mul1 / wqa_scale KeyErrors ~40s in).
Attribution requires an offline stage-0 draft-class probe before further
boots. Also: skip mtp.*.ffn.gate.bias_vl in the draft loader
(language-model-only does not register it).
