import sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/attention.py"
s = open(p).read()
if "_INDEXER64" not in s:
    # DeepseekV4IndexerCache.get_kv_cache_spec at :991-996 — replace its block_size expr
    anchor = "            block_size=self.cache_config.block_size,"
    idx = s.find("def get_kv_cache_spec", s.find("class DeepseekV4IndexerCache"))
    assert idx != -1, "indexer class/spec not found"
    seg_start = s.find(anchor, idx)
    assert seg_start != -1 and seg_start < idx + 2000, "indexer block_size anchor not found"
    s = s[:seg_start] + "            block_size=64 * self.compress_ratio,  # _INDEXER64: 64 states/page for deepgemm+prefill-dual contracts" + s[seg_start+len(anchor):]
    compile(s, p, "exec")
    open(p, "w").write(s)
    print("indexer spec -> 64*compress_ratio")
else:
    print("already")
