"""Check that the assembly includes match the checked-in generators."""
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
STDOUT_PAIRS = {
    "gen_glm_mla_gemm_vopd.py": "src/glm/prefill/glm_pf_mla_gemm_vopd.inc",
    "gen_hc_wide2_output_vopd.py": "src/hip_hc_wide2_output_vopd.inc",
    "gen_hc_wide2_vopd.py": "src/hip_hc_wide2_vopd.inc",
    "gen_q8_rows_lane_vopd.py": "src/hip_q8_rows_lane_vopd.inc",
    "gen_q8_rows_vopd.py": "src/hip_q8_rows_vopd.inc",
    "gen_selected_exact_mla_scores_vopd.py": "src/attn_core/selected_exact_mla_scores_vopd.inc",
}


def main():
    python = [sys.executable, "-I", "-B"]
    failed = []
    for generator, target in STDOUT_PAIRS.items():
        expected = (ROOT / target).read_text(encoding="utf-8")
        actual = subprocess.check_output(python + [str(ROOT / "tools" / generator)], text=True, encoding="utf-8")
        if actual != expected:
            failed.append(target)

    with tempfile.TemporaryDirectory() as tmp:
        output = pathlib.Path(tmp) / "ds41_qk.inc"
        subprocess.check_output(python + [str(ROOT / "tools/gen_ds41_qk_asm.py"), str(output)])
        target = "src/hip_ds41_attention_qk.inc"
        if output.read_text(encoding="utf-8") != (ROOT / target).read_text(encoding="utf-8"):
            failed.append(target)

    for target in failed:
        print(f"{target}: generated assembly differs from its source", file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
