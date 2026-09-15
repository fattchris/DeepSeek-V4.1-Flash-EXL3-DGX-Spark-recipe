import sys

# 1. sparse_swa.py: remove the 256 preference (MultipleOf(32) advert stays; 64 satisfies)
p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py"
s = open(p).read()
if "return 256" in s:
    a = """    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 256

"""
    assert a in s, "swa pref anchor"
    s = s.replace(a, "", 1)
    compile(s, p, "exec"); open(p, "w").write(s)
    print("sparse_swa: 256 preference removed")
else:
    print("sparse_swa: already clean")

# 2. cutlass_mla.py: [128] -> [64] won't work; consensus says MultipleOf(64)
p2 = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/cutlass_mla.py"
s2 = open(p2).read()
if "return [128]" in s2:
    s2 = s2.replace("        return [128]", "        return [MultipleOf(64)]", 1)
    compile(s2, p2, "exec"); open(p2, "w").write(s2)
    print("cutlass_mla: [128] -> MultipleOf(64)")
else:
    print("cutlass_mla: already ok")
