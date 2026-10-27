"""Admission tests for capture provenance, using a fake local compiler, never a GPU."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
MODULE_SPEC = importlib.util.spec_from_file_location("ptx_verifier_capture", LAB / "capture.py")
capture = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(capture)


PTX = """// complete module: unrelated entry is deliberately retained
.version 8.7
.target sm_120
.address_size 64
.visible .entry unrelated_kernel() { ret; }
.visible .entry _ZN5strix4ds414cuda4attn20carry_restore_kernelEPKfPfPjmj()
{ ret; }
"""


@unittest.skipUnless(sys.platform == "linux", "the capture profile is Linux/GCC only")
class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ptx-capture-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.tu = self.source / capture.SOURCES["carry_restore_kernel"]
        self.tu.parent.mkdir(parents=True)
        self.tu.write_text("// intact complete translation unit\n", encoding="utf-8")
        self.header = self.source / "include/strix/capture_test.hpp"
        self.header.parent.mkdir(parents=True)
        self.header.write_text("// original project dependency\n", encoding="utf-8")
        (self.source / "CMakeLists.txt").write_text("# declared build source\n", encoding="utf-8")
        for command in (["init", "-q"], ["add", "."],
                        ["-c", "user.name=Capture Test", "-c", "user.email=capture@example.invalid",
                         "commit", "-qm", "fixture", "--no-verify"]):
            subprocess.run(["git", "-C", str(self.source), *command], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.revision = subprocess.check_output(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"], text=True).strip()
        manifest = json.loads((LAB / "cases.json").read_text(encoding="utf-8"))
        for case in manifest["cases"]:
            case["source_revision"] = self.revision
        self.manifest = self.root / "cases.json"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.toolkit = self.root / "cuda"
        for directory in ("bin", "nvvm/bin", "nvvm/libdevice"):
            (self.toolkit / directory).mkdir(parents=True, exist_ok=True)
        for name in ("bin/nvcc.profile", "bin/cudafe++", "nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc"):
            (self.toolkit / name).write_text("fake toolchain component\n", encoding="utf-8")
        self.host = self.root / "g++"
        self.host.write_text(
            "#!" + sys.executable + "\nimport sys\n"
            "print(" + repr(sys.executable) + " if '-print-prog-name=cc1plus' in sys.argv else 'fake GCC')\n",
            encoding="utf-8")
        self.host.chmod(0o755)
        self.nvcc = self.toolkit / "bin/nvcc"
        self.write_compiler()
        self.args = argparse.Namespace(
            source_root=str(self.source), kernel="carry_restore_kernel", nvcc=str(self.nvcc),
            nvcc_sha256=capture.file_hash(self.nvcc), host_cxx=str(self.host),
            output_dir=str(self.root / "capture"), timeout_seconds=10,
        )
        # Real compiler settings in the surrounding test environment should not
        # enter the fake-tool fixture or weaken explicit injection tests.
        self.environment = patch.dict(os.environ, {name: "" for name in capture.INJECTED_ENV})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def write_compiler(self, behavior="success"):
        script = (
            "#!" + sys.executable + "\n"
            "from pathlib import Path\nimport sys, time\n"
            "behavior = " + repr(behavior) + "\n"
            "args = sys.argv[1:]\n"
            "if '--version' in args:\n    print('fake nvcc 12.8'); raise SystemExit(0)\n"
            "source = Path(next(a for a in args if a.endswith('.cu')))\n"
            "header = source.parents[2] / 'include/strix/capture_test.hpp'\n"
            "def value(prefix): return next(a[len(prefix):] for a in args if a.startswith(prefix))\n"
            "deps = [source, header]\n"
            "if '--ptx' in args:\n"
            "    if behavior == 'fail':\n        print('compiler failure evidence', file=sys.stderr); raise SystemExit(9)\n"
            "    if behavior == 'timeout': time.sleep(30)\n"
            "    if behavior == 'mutate': header.write_text('// changed during compile')\n"
            "    if behavior == 'dependency-change': deps = [source]\n"
            "    ptx = " + repr(PTX) + "\n"
            "    if behavior == 'ambiguous': ptx += '.entry carry_restore_kernel() { ret; }\\n'\n"
            "    Path(value('--output-file=')).write_text(ptx)\n"
            "def escaped(p):\n"
            "    return str(p).replace('\\\\', '\\\\\\\\').replace('$', '$$').replace(' ', '\\\\ ').replace('#', '\\\\#').replace(':', '\\\\:')\n"
            "Path(value('--dependency-output=')).write_text('module.ptx : ' + ' '.join(escaped(p) for p in deps) + '\\n')\n"
        )
        self.nvcc.write_text(script, encoding="utf-8")
        self.nvcc.chmod(0o755)

    def run_capture(self, behavior=None):
        if behavior is not None:
            self.write_compiler(behavior)
            self.args.nvcc_sha256 = capture.file_hash(self.nvcc)
        return capture.capture(self.args, self.manifest)

    def assert_failed_receipt(self, receipt, phrase):
        self.assertEqual(receipt["outcome"], "ERROR", receipt)
        self.assertIn(phrase, receipt["reason"])
        self.assertFalse(receipt["source_files_match_declared_revision"])
        self.assertFalse(receipt["native_qualified"])
        self.assertFalse(receipt["model_qualified"])
        self.assertFalse(receipt["bitwise_verified"])
        self.assertEqual(receipt["production_build_equivalence"], "not_attested")
        self.assertTrue((Path(self.args.output_dir) / "capture.json").is_file())

    def test_capture_keeps_whole_module_and_records_fixed_profile_and_provenance(self):
        receipt = self.run_capture()
        self.assertEqual(receipt["outcome"], "CAPTURED", receipt)
        self.assertTrue(receipt["source_files_match_declared_revision"])
        self.assertEqual(receipt["source_revision"], self.revision)
        self.assertEqual({p["path"] for p in receipt["source_files"]},
                         {"CMakeLists.txt", str(self.tu.relative_to(self.source)),
                          "include/strix/capture_test.hpp"})
        self.assertEqual((Path(self.args.output_dir) / "module.ptx").read_text(), PTX)
        self.assertEqual(receipt["ptx"]["sha256"], hashlib.sha256(PTX.encode()).hexdigest())
        self.assertEqual(receipt["ptx"]["entry_count"], 2)
        command = next(step["argv"] for step in receipt["commands"] if step["label"] == "compile")
        self.assertIn(str(self.tu), command)
        self.assertIn("--ptx", command)
        for flag in capture.PROFILE_FLAGS:
            self.assertIn(flag, command)
        self.assertNotIn("--kernels-used", " ".join(command))
        self.assertFalse(receipt["native_qualified"])
        self.assertFalse(receipt["model_qualified"])
        self.assertFalse(receipt["bitwise_verified"])
        self.assertEqual(receipt["production_build_equivalence"], "not_attested")

    def test_modified_translation_unit_is_rejected_before_compiler_launch(self):
        self.tu.write_text("// modified source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.run_capture()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_modified_cmake_is_rejected_before_compiler_launch(self):
        (self.source / "CMakeLists.txt").write_text("# changed flags\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source differs"):
            self.run_capture()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_modified_project_header_blocks_ptx_compilation(self):
        self.header.write_text("// modified header\n", encoding="utf-8")
        receipt = self.run_capture()
        self.assert_failed_receipt(receipt, "source differs")
        self.assertNotIn("compile", [step["label"] for step in receipt["commands"]])

    def test_hash_mismatch_does_not_execute_compiler(self):
        self.args.nvcc_sha256 = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 does not match"):
            self.run_capture()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_injected_compiler_environment_is_rejected(self):
        for name in ("NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS", "CPATH", "GCC_EXEC_PREFIX"):
            with self.subTest(name=name), patch.dict(os.environ, {name: "injected"}):
                with self.assertRaisesRegex(ValueError, name):
                    self.run_capture()
        self.assertFalse(Path(self.args.output_dir).exists())

    def test_new_output_directory_must_be_outside_checkout(self):
        self.args.output_dir = str(self.source / "capture")
        with self.assertRaisesRegex(ValueError, "outside"):
            self.run_capture()

    def test_existing_output_is_not_overwritten(self):
        output = Path(self.args.output_dir)
        output.mkdir()
        (output / "keep.txt").write_text("preserved", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must be new"):
            self.run_capture()
        self.assertEqual((output / "keep.txt").read_text(), "preserved")

    def test_dependency_mutation_during_compile_is_not_attested(self):
        self.assert_failed_receipt(self.run_capture("mutate"), "changed during compilation")

    def test_changed_compiled_dependency_set_is_not_attested(self):
        self.assert_failed_receipt(self.run_capture("dependency-change"), "dependency closure differs")

    def test_compile_failure_keeps_stderr_and_failed_status(self):
        receipt = self.run_capture("fail")
        self.assert_failed_receipt(receipt, "compiler step failed: compile")
        self.assertEqual(receipt["commands"][-1]["return_code"], 9)
        self.assertIn("compiler failure evidence",
                      (Path(self.args.output_dir) / "compile.stderr.txt").read_text())

    def test_ambiguous_kernel_symbol_is_not_attested(self):
        self.assert_failed_receipt(self.run_capture("ambiguous"), "exactly one compiled entry")

    def test_timeout_cannot_become_a_successful_capture(self):
        self.args.timeout_seconds = 1
        receipt = self.run_capture("timeout")
        self.assert_failed_receipt(receipt, "compiler step failed")
        self.assertTrue(receipt["commands"][-1]["timed_out"])


class MetadataTests(unittest.TestCase):
    def test_comment_cannot_supply_a_kernel_symbol(self):
        ptx = PTX.replace(".visible .entry _ZN", "// .visible .entry _ZN")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            capture.discover_symbol(ptx, "carry_restore_kernel")

    def test_wrong_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "target"):
            capture.discover_symbol(PTX.replace("sm_120", "sm_100"), "carry_restore_kernel")

    def test_make_dependencies_keep_escaped_paths_and_continuations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, header = root / "source.cu", root / "space #$.hpp"
            source.touch()
            header.touch()
            result = capture.parse_dependencies(
                "module.ptx: source.cu \\\n space\\ \\#$$.hpp\n", root)
            self.assertEqual(set(result), {source, header})

    def test_dependency_rule_cannot_expand_variables(self):
        with self.assertRaisesRegex(ValueError, "Make expansion"):
            capture.parse_dependencies("module.ptx: $(INJECTED)\n", Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
