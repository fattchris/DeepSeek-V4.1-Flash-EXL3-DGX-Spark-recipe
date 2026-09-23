# SAGE 3.30 on two Sparks: working mechanisms and remaining work

The exact `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` checkpoint generated
responses on two GB10 nodes using NVMe for both routed experts and Engram.
Weights were not requantized. This establishes a working offload path, but
decode remains roughly 4–6 tokens/sec in the recorded test. It is not a
usable-speed claim or qualification of this repository's current defaults.

The implementation used plugin commit
`8f4517e` (full historical SHA in the decode receipt). Current mainline already contains
newer mixed-K loading and disk-Engram work. Preserve those contributions;
the findings here identify additional mechanisms and integration boundaries.
This PR changes documentation only, with no pin or deployment changes.

## What got the model responding

| Obstacle | Working approach | Integration boundary |
| --- | --- | --- |
| Gapped safetensors payloads rejected by the parser | Copy tensors into a contiguous serving container while preserving names, dtypes, shapes and every tensor byte; hash readback | Container repair does not fix a runtime's mixed-K allocation or quantization metadata |
| Different K in gate/up/down and across experts | Use each projection's physical trellis size and original MUL1 marker, without padding every tensor to K8 | Current upstream already has a mixed-K correctness path; do not replace it with our older loader |
| Expert weights alone exceed the admitted cache budget | 80 GiB fixed expert arena/rank, byte-based per-layer quotas and NVMe reads for misses | Requires a complete per-rank expert catalog and a loader that never allocates the full expert bank first |
| Staging and file-cache pressure during loading | Bounded native-weight streaming, explicit release of consumed file pages, aligned expert records, whole-record verification and bounded staging | File-cache release is local to opened model files; no global cache flush |
| Engram tables consume the same RAM pool as CUDA | Keep original tables on local disk, fetch owned rows, retain native FP8/E8M0 conversion and native hashing/collectives | The historical adapter differs from the current disk-Engram overlay; transplant row caching at its row-read boundary |
| Serial miss I/O stalls the caller | At most two asynchronous read/hash tickets, including completed and leased buffers | CPU readers make no CUDA calls; upload and CUDA lifetime remain the consumer's responsibility |
| Repeated Engram reads | 131072-row/table cache, 8192-row request cap, four readers, one pending batch | Cache bytes and Python overhead both count toward host RAM; this remains an eager path |

The portable shard tool and experimental cache components are submitted
separately to `vcruz305/vllm-exl3`. They do not install the complete historical
model loader or replace this recipe's launch scripts. The remaining adapter
work must check expert ownership, all required tensor loads, native-weight
delegation and failure propagation across ranks before serving is enabled.

## Recorded full-model result

The compact [decode receipt](evidence/sage330/decode-baseline.json) records
all six decode measurements, prompt bodies, generation settings, source
receipt digest and relevant runtime flags. Image IDs identify local builds;
they are not publicly pullable images. Hostnames, network addresses, private
mounts, raw process environments and checkpoint tensors are omitted.

| Runtime setting | Recorded value |
| --- | --- |
| Checkpoint revision | `e831e9e4d6bfeafa6d630848296417b1393404a3` |
| Hardware/topology | Two GB10 nodes; TP2+EP2, 192 whole main experts/rank; DP1/PP1 |
| vLLM / Torch / CUDA | `0.1.dev20904+g179dd0fa9` / `2.13.0+cu130` / 13.0 |
| ExLlamaV3 | `be57335b087e4f001c5caae061544df3c06ba01e` |
| Routed backend | Custom heterogeneous packed MUL1 kernel for thin experts; grouped packed GEMM for fat experts; CPU-controlled routes/cache misses |
| Context capacity / concurrency / prefill chunk | 32768 / 1 / 512 tokens |
| Expert arena / KV budget | 80 GiB / 4 GiB per rank |
| Attention cache | `fp8`, block size 64; native compressed/shared KV ownership retained |
| Graphs / DSpark / prefix cache / decoder-tail approximation | All off |
| Memory + memory-swap cgroup ceilings | Both 104 GiB per rank |
| Runtime guard | At least 8 GiB host RAM available/rank; zero swap |

After a fresh deployment and eight quality warmup cases, the run alternated
three coding/writing pairs. Requests used temperature 0, seed 17 and thinking
off. There was no separate decode warmup. Timing used streamed token arrival
at the client; decode excludes the first token and is not end-to-end request
throughput. These short prompts ran in a 32K-capacity server; the table is
**not** a 32K-input decode benchmark.

| Workload | Generated tokens/request | Decode median (tokens/sec) | TTFT range (seconds) |
| --- | --- | --- | --- |
| Coding | 135 | 4.400 | 1.944–3.406 |
| Writing | 512 | 5.681 | 2.835–4.406 |

All eight warmup gates passed; they cover a small instruction/arithmetic/code/
writing/tool suite, not broad model accuracy. All six measurements completed,
identity matched before and after, and the workers were stopped. Historical
4K/32K/64K/128K probes passed under an earlier 80 GiB, 128-token-prefill profile;
they do not extend this later profile's qualification or establish a maximum
context limit. Higher contexts need a separate fresh memory and quality run.

The [Engram correctness receipt](evidence/sage330/engram-correctness.json)
records original-table checks on both ranks for layers 1 and 14, including
8192-row requests, bit-exact native results and cache-capacity churn. It is
component evidence, not proof of graph capture or distributed serving speed.

## What did not solve speed

- Moving an 80 GiB cache from CUDA allocation to a system-memory mapping passed
  its later correctness checks but regressed both full-model workloads in the
  matched comparison. It is not promoted as a performance fix.
- A pinned 80 GiB shared-cache attempt failed the zero-swap guard during full
  startup. Component allocation success did not qualify full-model loading.
- Decoder-tail truncation reduced one repetitive prompt's prefill time but
  is approximate and lacked general-quality qualification. It stays excluded.
- Resident-path graph fixtures passed component checks. Full-model CUDA
  graphs remain blocked by CPU routing/miss handling and Engram staging.
- DSpark tensors were inventoried, but draft loading, verification batches,
  acceptance and full-model speedup were not qualified.

## Next integration work

1. Keep the exact 3.30 checkpoint and bring the bounded store/cache interfaces
   into the current loader, with complete ownership and load-completeness
   checks. A missing owned expert must never be treated as an unowned route.
2. Keep resident routing and launch metadata on GPU; handle actual misses
   through a bounded protocol with stable slot lifetimes. A faster component
   is not a full-model speed claim.
3. Make Engram staging compatible with capture while preserving native IDs,
   ownership, dequantization, history, collectives and gating.
4. Qualify original DSpark weights and actual accepted tokens per verification
   pass. Budget its memory and batch shapes explicitly.
5. Measure long-context KV ownership, indexer workspaces and actual host/cgroup
   use. Logical bytes/token estimates alone do not establish capacity.

No TP4 runtime qualification, C2 serving, graph-enabled deployment, DSpark
speedup, or 256K–1M context claim is made by this handoff.
