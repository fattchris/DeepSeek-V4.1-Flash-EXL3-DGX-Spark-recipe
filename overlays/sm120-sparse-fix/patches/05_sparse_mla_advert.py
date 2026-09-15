import sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/sparse_mla.py"
s = open(p).read()
a = "        return [64 if current_platform.is_device_capability_family(90) else 128]"
b = "        return [MultipleOf(64)]  # accept 64 (SM121 sparse pages) and multiples"
if a in s:
    s = s.replace(a, b, 1)
    compile(s, p, "exec"); open(p, "w").write(s)
    print("sparse_mla advert -> MultipleOf(64)")
elif b in s:
    print("already")
else:
    print("ANCHOR MISSING"); sys.exit(1)
