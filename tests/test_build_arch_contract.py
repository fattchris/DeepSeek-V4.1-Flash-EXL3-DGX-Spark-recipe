from __future__ import annotations

import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BuildArchContractTests(unittest.TestCase):
    """The CUDA build target must survive base images that already export
    TORCH_CUDA_ARCH_LIST (e.g. vllm/vllm-openai). Docker lets a base ENV shadow
    an ARG of the same name, which silently compiled ExLlamaV3 for the base
    image's seven-arch list on an sm_120 host and failed in ptxas (sm_75)."""

    def test_dockerfile_does_not_reuse_env_name_as_build_arg(self) -> None:
        text = (ROOT / "Dockerfile.spark").read_text()
        self.assertIsNone(re.search(r"^ARG TORCH_CUDA_ARCH_LIST\b", text, re.M))
        self.assertIsNotNone(re.search(r"^ARG CUDA_ARCH_LIST=", text, re.M))
        self.assertIn("TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST}", text)

    def test_dockerfile_asserts_effective_arch(self) -> None:
        text = (ROOT / "Dockerfile.spark").read_text()
        self.assertIn('test "$TORCH_CUDA_ARCH_LIST" = "$CUDA_ARCH_LIST"', text)

    def test_build_script_passes_renamed_build_arg(self) -> None:
        text = (ROOT / "scripts/build_runtime.sh").read_text()
        self.assertIn('--build-arg CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"', text)
        self.assertNotIn("--build-arg TORCH_CUDA_ARCH_LIST=", text)


if __name__ == "__main__":
    unittest.main()
