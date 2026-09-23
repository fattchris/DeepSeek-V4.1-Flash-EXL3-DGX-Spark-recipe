#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import re
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

import sys
sys.path.insert(0, str(SCRIPTS))
from runtime_lock import load_lock  # noqa: E402


class RecipeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = load_lock(ROOT / "runtime.lock.json")

    def test_dockerfile_defaults_match_lock(self) -> None:
        dockerfile = (ROOT / "Dockerfile.spark").read_text()

        def arg(name: str) -> str:
            match = re.search(rf"^ARG {re.escape(name)}=(.+)$", dockerfile, re.M)
            self.assertIsNotNone(match, f"missing Docker ARG {name}")
            return match.group(1).strip()

        self.assertEqual(arg("BASE_IMAGE"), self.lock["base_image"]["ref"])
        self.assertEqual(arg("VLLM_EXL3_REPO"), self.lock["vllm_exl3"]["repo"])
        self.assertEqual(arg("VLLM_EXL3_REF"), self.lock["vllm_exl3"]["commit"])
        self.assertEqual(arg("EXLLAMAV3_REF"), self.lock["exllamav3"]["commit"])
        self.assertEqual(arg("CUDA_ARCH_LIST"), self.lock["torch_cuda_arch_list"])
        self.assertIn("physical_fused_k_guard_installed", dockerfile)
        self.assertIn('fused_k_source"] == "physical_trellis_geometry"', dockerfile)

    def test_mixed_k_contract(self) -> None:
        caps = self.lock["capabilities"]
        self.assertEqual(caps["accepted_exl3_config_k"], [2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(caps["tensor_level_mixed_k_within_layer"])
        self.assertEqual(caps["routed_allocation_scope"], "per_expert_exact_trellis_shapes")
        self.assertEqual(caps["heterogeneous_mixed_k_dispatch"], "linear_exl3_python_loop")
        self.assertEqual(caps["uniform_k_fused_k_source"], "physical_trellis_geometry")
        self.assertFalse(caps["heterogeneous_mixed_k_cudagraph_qualified"])
        self.assertEqual(caps["mixed_k_first_boot"], "eager")

    def test_no_stale_runtime_pins(self) -> None:
        stale = [
            "8f4517e80416466fa4a3ad2eb28685021" + "d39e95f",
            "21fa627a3933d80de2d1030e732354d8" + "c3cd761e",
            "ee8c2c171bbe0d036a3accb24a76af5" + "a95506748",
            "5666d1b4a55ef2237797eaee60cbb042" + "e933f375",
        ]
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix not in {".md", ".py", ".sh", ".json", ".yml", ".yaml", ""}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if path == Path(__file__):
                continue
            if path.relative_to(ROOT).as_posix() == "two-spark-tp2/evidence/sage330/decode-baseline.json":
                # This one receipt describes an old measured deployment, not a
                # selectable runtime pin. Exempt only its historical identity
                # field; continue checking the rest of this file and all others.
                receipt = json.loads(text)
                self.assertTrue(receipt["scope"].startswith("Historical full-model decode;"))
                self.assertEqual(receipt["source_run_id"], "decode-baseline-r1")
                self.assertEqual(receipt["runtime"].pop("plugin_commit"), stale[0])
                text = json.dumps(receipt)
            for value in stale:
                if value in text:
                    offenders.append(f"{path.relative_to(ROOT)}: stale plugin SHA {value}")
        self.assertEqual(offenders, [], "\n".join(offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)

