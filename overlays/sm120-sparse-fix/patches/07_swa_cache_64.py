import sys
p = "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/attention.py"
s = open(p).read()
a = """            backend_cls=self.swa_backend_cls,
            block_size=32,
        )"""
b = """            backend_cls=self.swa_backend_cls,
            block_size=64,
        )"""
if a in s:
    s = s.replace(a, b, 1)
    compile(s, p, "exec"); open(p, "w").write(s)
    print("SWA cache block 32 -> 64")
elif b in s:
    print("already 64")
else:
    print("ANCHOR MISSING"); sys.exit(1)
