import sys

# ---- 1. vLLM flashinfer_sparse.py: _causal_swa_window helper + both call sites ----
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py"
s = open(p).read()
if "_causal_swa_window" not in s:
    # Helper method: insert before _forward_prefill def
    anchor = "    def _forward_prefill("
    helper = '''    def _causal_swa_window(self, indices, lengths):
        """Narrow vision-padded SWA rows to window_size (deciders' consensus).

        Rows are left-aligned: <= window_size valid slots then -1; columns
        window_size..window_size+max_image_tokens are vision padding (-1 for
        text). The dual-cache prefill kernels exist for TOPK == window_size
        only; the FFI requires stride(0) == width, hence .contiguous().
        Lengths are asserted (never clamped): discarding valid candidates
        would silently corrupt attention.
        """
        w = int(self.window_size)
        if int(indices.shape[-1]) == w:
            return indices, lengths
        if int(indices.shape[-1]) < w:
            raise RuntimeError(
                f"SM120 text SWA requires width >= window_size={w}, "
                f"got {indices.shape[-1]}"
            )
        torch._assert_async(
            ((lengths >= 0) & (lengths <= w)).all(),
            "SM120 text SWA cannot discard valid image/noncausal candidates",
        )
        return indices[..., :w].contiguous(), lengths.contiguous()

'''
    assert anchor in s, "prefill anchor missing"
    s = s.replace(anchor, helper + anchor, 1)

    # Prefill call site (chunk loop)
    a = """            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]"""
    b = a + """
            swa_indices_chunk, swa_lens_chunk = self._causal_swa_window(
                swa_indices_chunk, swa_lens_chunk
            )"""
    assert a in s, "prefill chunk anchor missing"
    s = s.replace(a, b, 1)

    # Decode call site: the decode path builds swa_indices_chunk the same way
    # (it appears once more outside the prefill loop)
    occurrences = s.count(a)
    if occurrences == 1:
        # decode uses differently-shaped slicing; find its chunk build
        da = """            swa_indices_chunk = swa_metadata.prefill_swa_indices[
                query_start:query_end
            ]"""
        if da in s:
            s = s.replace(da, da + """
            swa_indices_chunk, swa_lens_chunk = self._causal_swa_window(
                swa_indices_chunk, swa_lens_chunk
            )""", 1)
        else:
            print("WARN: decode chunk anchor not found; decode may already be narrow")
    compile(s, p, "exec")
    open(p, "w").write(s)
    print("flashinfer_sparse.py: helper + prefill site OK")
else:
    print("flashinfer_sparse.py: already patched")

# ---- 2. flashinfer _sparse_mla_sm120.py: remove the generic clamp entirely ----
p2 = "/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_sparse_mla_sm120.py"
s2 = open(p2).read()
start = s2.find("        # _V41_TOPK_CLAMP")
if start != -1:
    end_marker = "                topk = _sup_t"
    end = s2.find(end_marker, start)
    assert end != -1, "clamp end missing"
    end += len(end_marker)
    s2 = s2[:start] + "        # No width clamp here. The vLLM SM120 backend narrows SWA rows to\n        # window_size before the call; silently picking a smaller compiled\n        # specialization would drop valid keys (and truncated extra_indices,\n        # i.e. indexer top-k)." + s2[end:]
    compile(s2, p2, "exec")
    open(p2, "w").write(s2)
    print("_sparse_mla_sm120.py: generic clamp removed")
else:
    print("_sparse_mla_sm120.py: no clamp found (ok)")
