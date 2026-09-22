import sys

p = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/dspark.py"
s = open(p).read()
if "_dq8_scale_bases" in s:
    print("already"); sys.exit(0)

# 0) bias_vl skip becomes conditional (Fable change 3)
a0 = """        if rest.endswith(".ffn.gate.bias_vl"):  # _B3_BIASVL: vision-only bias; not registered under language-model-only
            return None"""
b0 = """        if rest.endswith(  # _B3_BIASVL: vision-only bias; skip only when unregistered
            ".ffn.gate.bias_vl"
        ) and not getattr(self, "_dq8_biasvl_registered", False):
            return None"""
assert a0 in s, "bias_vl anchor"
s = s.replace(a0, b0, 1)

# A) streaming scale cache + biasvl flag (no list(weights) — decider change)
a = "        for name, loaded_weight in weights:"
b = """        # _DQ8 (decider-reviewed: Astra + Fable converge): draft dense shards are
        # MXFP8 — F8_E4M3 weights with E8M0 scales on a plain [N/32, K/32] row-major
        # grid — while the model registers plain bf16 params. Dequantize at load.
        # Decision is dtype-based (covers attn, main_proj, shared experts), the
        # scale cache streams (scales sort before weights and are tiny).
        self._dq8_biasvl_registered = any(
            k.endswith("ffn.gate.bias_vl") for k in params_dict
        )
        _scale_cache: dict[str, torch.Tensor] = {}
        _late: list[tuple[str, torch.Tensor]] = []
        _dq8_scale_bases = {
            n[: -len(".weight_scale")]
            for n in params_dict
            if n.endswith(".weight_scale")
        }
        for name, loaded_weight in weights:"""
assert a in s, "loop anchor"
s = s.replace(a, b, 1)

# B) scale entries: registered -> existing rename flow; unregistered -> cache
a2 = """                    if _alt_name in params_dict:
                        name = _alt_name"""
b2 = """                    if _alt_name in params_dict:
                        name = _alt_name
            # _DQ8: match the POST-RENAME suffix (.scale -> linear_scale_name above)
            if (
                name.endswith(".weight_scale") or name.endswith(".weight_scale_inv")
            ) and ".experts." not in name:
                _base = name.rsplit(".weight_scale", 1)[0] if name.endswith(
                    ".weight_scale"
                ) else name.rsplit(".weight_scale_inv", 1)[0]
                if _base not in _dq8_scale_bases and (
                    f"{_base}.weight_scale_inv" not in params_dict
                    and f"{_base}.weight_scale" not in params_dict
                ):
                    _scale_cache[_base] = loaded_weight
                    continue  # _DQ8: consumed with its weight below"""
assert a2 in s, "alias anchor"
s = s.replace(a2, b2, 1)

# C) dtype-based dequant before placement (Astra Q2: placement stays with the loader;
#    fused param is disable_tp/replicated, ordinary loaders slice TP)
a3 = """            if ".experts." in name:"""
b3 = """            # _DQ8: any F8_E4M3 dense shard whose scale is cached dequantizes to
            # bf16 here; loaders/shard placement proceed unchanged afterwards.
            if loaded_weight.dtype == torch.float8_e4m3fn:
                _base = name[: -len(".weight")] if name.endswith(".weight") else ""
                _sc = _scale_cache.get(_base)
                if _sc is not None:
                    _sf = _sc.to(torch.float32)
                    _br = max(1, loaded_weight.shape[0] // _sf.shape[0])
                    _bc = max(1, loaded_weight.shape[1] // _sf.shape[1])
                    _sfx = _sf.repeat_interleave(_br, 0).repeat_interleave(_bc, 1)
                    if _sfx.shape != loaded_weight.shape:
                        _sfx = _sfx[: loaded_weight.shape[0], : loaded_weight.shape[1]]
                    loaded_weight = (loaded_weight.to(torch.float32) * _sfx).to(
                        torch.bfloat16
                    )
                elif (
                    _base
                    and _base not in _dq8_scale_bases
                    and f"{_base}.weight_scale_inv" not in params_dict
                ):
                    _late.append((name, loaded_weight))  # _DQ8: scale not yet seen
                    continue
            if ".experts." in name:"""
assert a3 in s, "expert anchor"
s = s.replace(a3, b3, 1)

# D) flush late arrivals (defensive; scales sort first so normally empty)
a4 = """        if self.model.confidence_head is not None and not loaded_confidence_head:"""
b4 = """        for name, loaded_weight in _late:  # _DQ8 flush: scales seen after weights
            _base = name[: -len(".weight")]
            _sc = _scale_cache.get(_base)
            if _sc is None:
                logger.error("DQ8: no scale for %s", name)
                continue
            _sf = _sc.to(torch.float32)
            _br = max(1, loaded_weight.shape[0] // _sf.shape[0])
            _bc = max(1, loaded_weight.shape[1] // _sf.shape[1])
            _sfx = _sf.repeat_interleave(_br, 0).repeat_interleave(_bc, 1)
            if _sfx.shape != loaded_weight.shape:
                _sfx = _sfx[: loaded_weight.shape[0], : loaded_weight.shape[1]]
            _w = (loaded_weight.to(torch.float32) * _sfx).to(torch.bfloat16)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, _w)
            loaded_params.add(name)
        if self.model.confidence_head is not None and not loaded_confidence_head:"""
assert a4 in s, "flush anchor"
s = s.replace(a4, b4, 1)

compile(s, p, "exec")
open(p, "w").write(s)
print("dspark _DQ8 v2 (decider-reviewed) installed")
