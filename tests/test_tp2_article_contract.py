from pathlib import Path
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


def _env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value.strip("'")
    return out


class Tp2ContractTests(unittest.TestCase):
    def test_serve_supports_guarded_tp2_moe_tp_ab(self) -> None:
        text = (ROOT / "scripts" / "serve.sh").read_text()
        assert 'MOE_PARALLEL_MODE="${MOE_PARALLEL_MODE:-ep}"' in text
        assert '--enable-expert-parallel --enable-ep-weight-filter' in text
        assert 'ALLOW_EXPERIMENTAL_TP4_MOE_TP' in text
        assert '576->640' in text


    def test_tp2_profile_is_a_conservative_disk_engram_first_load(self) -> None:
        env = _env(ROOT / "profiles" / "tp2.env")
        assert env["VLLM_ENGRAM_DISK_BACKED"] == "1"
        assert env["MOE_PARALLEL_MODE"] == "ep"
        assert env["MAX_MODEL_LEN"] == "8192"
        assert env["DSPARK"] == "0"
        assert env["EAGER"] == "1"
        assert env["MODEL"] == env["VLLM_ENGRAM_MODEL_DIR"]


    def test_tp2_profile_override_matches_the_checked_in_attestation(self) -> None:
        env = _env(ROOT / "profiles" / "tp2.env")
        att = json.loads((ROOT / "two-spark-tp2" / "tp2-e831e9e4-metadata.json").read_text())
        assert json.loads(env["HF_OVERRIDES_JSON"]) == att["runtime_hf_overrides"]


    def test_tp2_docs_keep_the_geometry_facts(self) -> None:
        docs = (ROOT / "two-spark-tp2" / "README.md").read_text(encoding="utf-8")
        stale = "mixed K2-K8 is incompatible with the pinned layer-uniform loader"
        assert stale not in docs
        assert "5120 × 1152" in docs
        assert "890 bytes" in docs


if __name__ == "__main__":
    unittest.main(verbosity=2)
