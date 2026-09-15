import sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py"
s = open(p).read()
if "_NARROW_DECODE" not in s:
    a = """        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None"""
    b = """        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        # _NARROW_DECODE: vision-padded rows (window+1024) must narrow to the
        # causal text window before the kernel; decode TK must equal
        # window_size (consensus patch, both paths).
        swa_indices, swa_lens = self._causal_swa_window(swa_indices, swa_lens)"""
    assert a in s, "decode anchor missing"
    s = s.replace(a, b, 1)
    compile(s, p, "exec")
    open(p, "w").write(s)
    print("decode narrow installed")
else:
    print("already")
