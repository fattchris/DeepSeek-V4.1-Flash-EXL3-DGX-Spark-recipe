import sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"
s = open(p).read()
a = "        return [64 if current_platform.is_device_capability_family(90) else 128]"
b = "        from vllm.v1.attention.backend import MultipleOf as _MO64\n        return [_MO64(64)]  # accept 64 (SM121 sparse pages) and multiples"
if a in s:
    s = s.replace(a, b, 1)
    compile(s, p, "exec"); open(p, "w").write(s)
    print("indexer advert -> MultipleOf(64)")
elif "_MO64(64)" in s:
    print("already")
else:
    print("ANCHOR MISSING"); sys.exit(1)
