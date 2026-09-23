from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DiskEngramContractTests(unittest.TestCase):
    def test_derived_image_enforces_safe_loader(self) -> None:
        text = (ROOT / "Dockerfile.disk-engram").read_text()
        self.assertIn(
            "disk-backed Engram requires the per-tensor safetensors iterator",
            text,
        )
        self.assertIn('safetensors_load_strategy in {"eager", "torchao", "prefetch"}', text)
        self.assertIn(
            "should_skip_engram_embed_tensor('layers.1.engram.embed.weight') is True",
            text,
        )

    def test_disk_env_reaches_ray_and_vllm(self) -> None:
        cluster = (ROOT / "scripts/start_cluster.sh").read_text()
        serve = (ROOT / "scripts/serve.sh").read_text()
        for key in (
            "VLLM_ENGRAM_DISK_BACKED",
            "VLLM_ENGRAM_MODEL_DIR",
            "VLLM_EXL3_MODEL_DIR",
        ):
            self.assertIn(key, cluster)
            self.assertIn(key, serve)

    def test_disk_cluster_wrapper_defaults_to_tp4_profile(self) -> None:
        text = (ROOT / "scripts/start_disk_engram_cluster.sh").read_text()
        self.assertIn("profiles/tp4.env", text)
        self.assertIn("deepseek-v41-exl3:tp4", text)

    def test_oom_guard_targets_one_exact_container(self) -> None:
        guard = (ROOT / "scripts/oom_guard.sh").read_text()
        wrapper = (ROOT / "scripts/watch_oom_guard.sh").read_text()
        self.assertIn("OOM_GUARD_CONTAINER_NAME", guard)
        self.assertIn('grep -Fx "$TARGET_CONTAINER"', guard)
        self.assertNotIn("grep -E '^(dsv41|deepseek-v41)'", guard)
        self.assertIn("OOM_GUARD_CONTAINER_NAME", wrapper)


if __name__ == "__main__":
    unittest.main(verbosity=2)
