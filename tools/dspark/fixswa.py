import sys

# Fix 1: DSpark non-causal passthrough (Astra corrected-A + Fable patch 09, merged consensus)
p1 = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py"
s = open(p1).read()
if "_DSPARK_NONCAUSAL_PASSTHROUGH" not in s:
    a = """        swa_indices, swa_lens = self._causal_swa_window(swa_indices, swa_lens)
        q = self._prepare_query(q, output)"""
    b = """        # _DSPARK_NONCAUSAL_PASSTHROUGH (deciders Astra+Fable consensus): DSpark
        # draft rows are get_dspark_swa_index_width(window, k) wide (192 for
        # 128+5) and hold up to window+k valid slots BY DESIGN (the block's own
        # tokens ride the SWA row). The SM120 decode kernel has a TOPK=192
        # template for 16/32/64/128 heads and clamps by topk_length; only the
        # dual-cache prefill kernels are TK=window-only and the draft never
        # prefills. So: causal vision-padded rows narrow; dspark rows pass
        # through untouched at native width.
        _w = int(self.window_size)
        _width = int(swa_indices.shape[-1])
        _vw = _w + int(getattr(self, "max_image_tokens", 0) or 0)
        if _width == _w or (_width == _vw and _vw != _w):
            swa_indices, swa_lens = self._causal_swa_window(swa_indices, swa_lens)
        else:
            swa_indices = swa_indices.contiguous()  # FFI: stride(0) == width
            swa_lens = swa_lens.contiguous()
        q = self._prepare_query(q, output)"""
    assert a in s, "decode narrow anchor"
    s = s.replace(a, b, 1)
    compile(s, p1, "exec")
    open(p1, "w").write(s)
    print("fix1 passthrough installed")
else:
    print("fix1 already")

# Fix 2: _DQ8 shared-expert w2 scale base must match the post-rename weight name
p2 = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/dspark.py"
s = open(p2).read()
if "_DQ8_W2BASE" not in s:
    a = """                if (
                    _base not in _dq8_scale_bases
                    and f"{_base}.weight_scale_inv" not in params_dict
                    and f"{_base}.weight_scale" not in params_dict
                ):
                    _scale_cache[_base] = loaded_weight"""
    b = """                # _DQ8_W2BASE: the loader renames shared_experts.w2 ->
                # down_proj AFTER this block but BEFORE the dequant lookup;
                # cache under the post-rename base so C finds it (Fable).
                _base = _base.replace(
                    ".shared_experts.w2", ".shared_experts.down_proj"
                )
                if (
                    _base not in _dq8_scale_bases
                    and f"{_base}.weight_scale_inv" not in params_dict
                    and f"{_base}.weight_scale" not in params_dict
                ):
                    _scale_cache[_base] = loaded_weight"""
    assert a in s, "w2 base anchor"
    s = s.replace(a, b, 1)
    compile(s, p2, "exec")
    open(p2, "w").write(s)
    print("fix2 w2 base installed")
else:
    print("fix2 already")
