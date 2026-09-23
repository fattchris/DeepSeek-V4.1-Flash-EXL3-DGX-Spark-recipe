#!/usr/bin/env python3
"""Contract tests for the measured TP4 profile (profiles/tp4.env).

Text/structure checks only: no GPU, no Spark, no network. They keep the
profile, the captured serve config, serve.sh and Dockerfile.tp4 in agreement.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SERVE = ROOT / "scripts" / "serve.sh"
PROFILE = ROOT / "profiles" / "tp4.env"
CAPTURED = ROOT / "configs" / "serve-tp4-live.yaml"
TP4_DOC = ROOT / "four-spark-tp4" / "README.md"


def _profile_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in PROFILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _yaml_top_level(text: str) -> dict[str, object]:
    """Minimal indent-aware mapping parser: top-level keys, nested one level."""
    result: dict[str, object] = {}
    current: dict[str, object] | None = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indented = raw.startswith((" ", "\t"))
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if indented:
            if current is None:
                continue
            current[key] = value
            continue
        if value:
            result[key] = value
            current = None
        else:
            nested: dict[str, object] = {}
            result[key] = nested
            current = nested
    return result


class DeployedProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.serve = SERVE.read_text(encoding="utf-8")
        self.profile = _profile_env()
        self.yaml = _yaml_top_level(CAPTURED.read_text(encoding="utf-8"))

    def test_serve_keeps_the_locked_defaults(self) -> None:
        # The new knobs must be opt-in: unset means the previous behavior.
        self.assertIn('NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"', self.serve)
        self.assertIn('KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"', self.serve)
        self.assertIn('BLOCK_SIZE="${BLOCK_SIZE:-}"', self.serve)
        self.assertIn('PREFIX_CACHING="${PREFIX_CACHING:-}"', self.serve)
        self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-$LOCK_MAX_MODEL_LEN}"', self.serve)

    def test_serve_forwards_every_deployed_knob(self) -> None:
        for needle in (
            '--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"',
            '--block-size "$BLOCK_SIZE"',
            '--enable-prefix-caching',
            '--no-enable-prefix-caching',
            '--compilation-config "$COMPILATION_CONFIG"',
            '--speculative-config "$SPEC_JSON"',
            'VLLM_EXL3_MOE_KERNEL="$EXL3_MOE_KERNEL"',
        ):
            self.assertIn(needle, self.serve, needle)

    def test_serve_validates_the_new_inputs(self) -> None:
        # Fail closed on malformed values instead of handing them to vLLM.
        self.assertIn("KV_CACHE_MEMORY_BYTES must be an integer byte count", self.serve)
        self.assertIn("BLOCK_SIZE must be an integer token count", self.serve)
        self.assertIn("PREFIX_CACHING must be a boolean 0/1 value", self.serve)
        self.assertIn("NUM_SPECULATIVE_TOKENS must be an integer", self.serve)
        self.assertIn("COMPILATION_CONFIG is not valid JSON object data.", self.serve)

    def test_serve_forwards_the_exl3_execution_tuning(self) -> None:
        for key in (
            "VLLM_EXL3_ALLOW_SHAPE_MISMATCH",
            "VLLM_EXL3_TRELLIS_ARENA",
            "VLLM_EXL3_PADDED_MAX_T",
            "VLLM_EXL3_PADDED_MAX_K",
            "VLLM_EXL3_NATIVE_MOE_MAX_ROWS",
            "VLLM_EXL3_MADV_AFTER_H2D",
            "VLLM_EXL3_PREFETCH",
            "VLLM_EXL3_PAD_SO_CB",
        ):
            self.assertIn(key, self.serve, key)

    def test_profile_and_captured_config_agree(self) -> None:
        cases = (
            ("max-model-len", "MAX_MODEL_LEN"),
            ("kv-cache-memory-bytes", "KV_CACHE_MEMORY_BYTES"),
            ("max-num-seqs", "MAX_NUM_SEQS"),
            ("max-num-batched-tokens", "MAX_NUM_BATCHED_TOKENS"),
            ("block-size", "BLOCK_SIZE"),
            ("gpu-memory-utilization", "GPU_MEMORY_UTILIZATION"),
        )
        for yaml_key, env_key in cases:
            self.assertIn(yaml_key, self.yaml)
            self.assertEqual(
                str(self.yaml[yaml_key]), self.profile[env_key],
                f"{yaml_key} in {CAPTURED.name} must match {env_key} in {PROFILE.name}",
            )

    def test_profile_context_limit_is_one_million(self) -> None:
        self.assertEqual(self.profile["MAX_MODEL_LEN"], "1048576")
        self.assertEqual(str(self.yaml["max-model-len"]), "1048576")

    def test_profile_speculative_block_matches_capture(self) -> None:
        spec = self.yaml["speculative-config"]
        self.assertIsInstance(spec, dict)
        assert isinstance(spec, dict)
        self.assertEqual(spec["method"], "dspark")
        self.assertEqual(
            spec["num_speculative_tokens"], self.profile["NUM_SPECULATIVE_TOKENS"]
        )
        self.assertEqual(spec["quantization"], self.profile["SPECULATIVE_QUANTIZATION"])

    def test_profile_compilation_config_matches_capture(self) -> None:
        comp = self.yaml["compilation-config"]
        self.assertIsInstance(comp, dict)
        assert isinstance(comp, dict)
        declared = json.loads(self.profile["COMPILATION_CONFIG"])
        self.assertEqual(declared["cudagraph_mode"], comp["cudagraph_mode"])
        sizes = [int(n) for n in re.findall(r"\d+", str(comp["cudagraph_capture_sizes"]))]
        self.assertEqual(declared["cudagraph_capture_sizes"], sizes)

    def test_profile_engram_and_prefix_caching_match_capture(self) -> None:
        engram = self.yaml["engram-config"]
        self.assertIsInstance(engram, dict)
        assert isinstance(engram, dict)
        self.assertEqual(engram["cpu_offload"], "false")
        self.assertIn('"cpu_offload":false', self.profile["EXTRA_VLLM_ARGS"])
        self.assertEqual(self.yaml["enable-prefix-caching"], "true")
        self.assertEqual(self.profile["PREFIX_CACHING"], "1")

    def test_profile_does_not_carry_the_shadowed_kv_env_var(self) -> None:
        # Trap documented in four-spark-tp4/README.md: a VLLM_KV_CACHE_MEMORY_BYTES launcher
        # variable is shadowed by --kv-cache-memory-bytes. The profile must not
        # reintroduce it.
        self.assertNotIn("VLLM_KV_CACHE_MEMORY_BYTES", self.profile)

    def test_capture_records_runtime_identity(self) -> None:
        text = CAPTURED.read_text(encoding="utf-8")
        for needle in (
            "sha256:8c560955",
            "814d4fe38082cddd838b45418c7d13a95395a36a",
            "be57335b087e4f001c5caae061544df3c06ba01e",
            "TORCH_CUDA_ARCH_LIST=12.1a",
        ):
            self.assertIn(needle, text, needle)
        # The captured run used a locally repaired pack; say where that shard lives now.
        self.assertIn("fix_missing_backbone.py", text)
        self.assertIn("b0c44df", text)

    def test_profile_uses_disk_backed_engram_with_explicit_dir(self) -> None:
        self.assertEqual(self.profile["VLLM_ENGRAM_DISK_BACKED"], "1")
        self.assertEqual(self.profile["VLLM_ENGRAM_MODEL_DIR"], self.profile["MODEL"])
        self.assertEqual(self.profile["IMAGE"], "deepseek-v41-exl3:tp4")

    def test_tp4_image_bakes_in_every_fix(self) -> None:
        text = (ROOT / "Dockerfile.tp4").read_text(encoding="utf-8")
        lock = json.loads((ROOT / "runtime.lock.json").read_text(encoding="utf-8"))
        kern = lock["tp4_kernels"]
        self.assertIn(f"ARG VLLM_EXL3_KERNEL_REF={kern['commit']}", text)
        self.assertIn(f"ARG P2B_CB={kern['p2b_cb']}", text)
        for path in [
            "overlays/sm120-sparse-fix/apply.sh",
            *sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "tools" / "engram").glob("*.py")),
            "tools/dspark/dq.py",
            "tools/dspark/mkdq8.py",
            "tools/dspark/fixswa.py",
            "tools/dspark/exl3_native_multik.patch",
            "tools/dspark/exl3_padded.patch",
        ]:
            name = path.split("/", 1)[1].replace("\\", "/")
            self.assertIn(name, text, f"Dockerfile.tp4 does not apply {path}")

    def test_tp4_docs_publish_the_profile_and_the_traps(self) -> None:
        docs = TP4_DOC.read_text(encoding="utf-8")
        self.assertIn("profiles/tp4.env", docs)
        self.assertIn("configs/serve-tp4-live.yaml", docs)
        self.assertIn("2,454,802 tokens", docs)
        self.assertIn("skips memory profiling", docs)
        self.assertIn("17179869184", docs)


if __name__ == "__main__":
    unittest.main()
