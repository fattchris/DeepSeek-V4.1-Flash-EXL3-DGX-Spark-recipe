import sys

# 1. The .cu widening (both dispatch branches)
p = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu"
s = open(p).read()
if "extra_page_block_size == 128 ||" not in s:
    a1 = """  if (topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
      topk_extra % BI == 0 && (extra_page_block_size == 64 || extra_page_block_size == 2)) {"""
    b1 = """  if (topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
      topk_extra % BI == 0 &&
      (extra_page_block_size == 128 || extra_page_block_size == 64 ||
       extra_page_block_size == 32 || extra_page_block_size == 2)) {"""
    assert a1 in s, "fulltile gate anchor"
    s = s.replace(a1, b1, 1)

    a2 = """    if (extra_page_block_size == 64) {
      DISPATCH_FULLTILE_BY_NH_PBSX(64);
    } else {
      DISPATCH_FULLTILE_BY_NH_PBSX(2);
    }"""
    b2 = """    if (extra_page_block_size == 128) {
      DISPATCH_FULLTILE_BY_NH_PBSX(128);
    } else if (extra_page_block_size == 64) {
      DISPATCH_FULLTILE_BY_NH_PBSX(64);
    } else if (extra_page_block_size == 32) {
      DISPATCH_FULLTILE_BY_NH_PBSX(32);
    } else {
      DISPATCH_FULLTILE_BY_NH_PBSX(2);
    }"""
    assert a2 in s, "fulltile dispatch anchor"
    s = s.replace(a2, b2, 1)

    a3 = """  if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  }"""
    b3 = """  if (extra_page_block_size == 128) {
    DISPATCH_BY_NH_PBSX(128);
  } else if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
  } else if (extra_page_block_size == 32) {
    DISPATCH_BY_NH_PBSX(32);
  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  }"""
    assert a3 in s, "CM dispatch anchor"
    s = s.replace(a3, b3, 1)
    open(p, "w").write(s)
    print(".cu widened to {2,32,64,128}")
else:
    print(".cu already widened")

# 2. Fable's post-construct assert in flashinfer_sparse.py
p2 = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py"
s2 = open(p2).read()
if "_EXTRAPBS_ASSERT" not in s2:
    a = """            extra_kv_paged = self._as_sparse_cache(compressed_k_cache)"""
    b = a + """
            # _EXTRAPBS_ASSERT (decider consensus): tensor rows must be a
            # non-zero multiple of the Python-computed rows; strict equality
            # would reject legitimate ratio-2 128-row pages after widening.
            _rows = attn_metadata.block_size // self.compress_ratio
            assert attn_metadata.block_size % self.compress_ratio == 0, (
                f"block_size={attn_metadata.block_size} not a multiple of "
                f"compress_ratio={self.compress_ratio}")
            assert _rows > 0 and extra_kv_paged.shape[1] % _rows == 0, (
                f"extra page rows {extra_kv_paged.shape[1]} not a multiple "
                f"of computed rows {_rows}")"""
    cnt = s2.count(a)
    assert cnt >= 1, "extra_kv_paged anchor"
    s2 = s2.replace(a, b)  # all sites (decode + prefill)
    compile(s2, p2, "exec")
    open(p2, "w").write(s2)
    print(f"assert installed at {cnt} site(s)")
else:
    print("assert already")
