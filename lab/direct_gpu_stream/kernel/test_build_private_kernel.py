"""Host preparation/receipt checks. No kernel build, install or GPU calls."""
from __future__ import annotations

import importlib.util
import io
import contextlib
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build_private_kernel", HERE / "build_private_kernel.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def valid_config() -> str:
    return "\n".join([f"{key}={value}" for key, value in build.REQUIRED_CONFIG.items()] +
                     ['CONFIG_LOCALVERSION="-strix-gpu-host-test"', '# CONFIG_LOCALVERSION_AUTO is not set',
                      '# CONFIG_MODULE_SIG_ALL is not set']) + "\n"


class PrivateKernelPreparationTests(unittest.TestCase):
    def test_config_requires_exact_supported_full_build(self):
        build.validate_config(valid_config())
        for key in build.REQUIRED_CONFIG:
            with self.subTest(key=key), self.assertRaises(ValueError):
                wrong = "y" if build.REQUIRED_CONFIG[key] == "n" else "n"
                build.validate_config(valid_config().replace(f"{key}={build.REQUIRED_CONFIG[key]}", f"{key}={wrong}"))

    def test_no_stock_or_automatic_release_and_no_duplicate_values(self):
        for text in (valid_config().replace("-strix-gpu-host-test", ""),
                     valid_config().replace("-strix-gpu-host-test", "-strix-gpu-host test"),
                     valid_config() + "CONFIG_MODULES=n\n",
                     valid_config() + "CONFIG_LOCALVERSION_AUTO=y\n",
                     valid_config().replace('# CONFIG_MODULE_SIG_ALL is not set', 'CONFIG_MODULE_SIG_ALL=y')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                build.validate_config(text)

    def test_private_workspace_rejects_project_and_nested_git(self):
        for path in (build.REPO, build.REPO / "lab/private-artifacts"):
            with self.assertRaises(ValueError):
                build.private_path(path)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".git").mkdir()
            with self.assertRaises(ValueError):
                build.private_path(root / "nested/build")

    def archive(self, root: Path, entries: list[tuple[str, str | None]]) -> Path:
        path = root / "archive.tar.gz"
        with tarfile.open(path, "w:gz") as output:
            for name, target in entries:
                info = tarfile.TarInfo(name)
                if target is not None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = target
                    output.addfile(info)
                else:
                    data = b"source"
                    info.size = len(data)
                    output.addfile(info, io.BytesIO(data))
        return path

    def test_archive_allows_only_internal_relative_symlinks(self):
        prefix = f"linux-{build.PIN}"
        with tempfile.TemporaryDirectory() as temporary:
            path = self.archive(Path(temporary), [(f"{prefix}/scripts/source", None),
                                                   (f"{prefix}/include/link", "../scripts/source")])
            build.validate_archive(path)

    def test_archive_traversal_symlink_and_duplicate_rejections(self):
        prefix = f"linux-{build.PIN}"
        cases = [[("/outside", None)], [(f"{prefix}/../../outside", None)],
                 [("other-tree/source", None)], [(f"{prefix}/link", "/outside")],
                 [(f"{prefix}/link", "../outside")],
                 [(f"{prefix}/link", "../outside"), (f"{prefix}/link/file", None)],
                 [(f"{prefix}/link", "scripts"), (f"{prefix}/link/file", None)],
                 [(f"{prefix}/source", None), (f"{prefix}/source", None)]]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    build.validate_archive(self.archive(Path(temporary), entries))

    def test_cli_rejects_hash_drift_existing_workspace_and_relative_workspace_before_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, [(f"linux-{build.PIN}/source", None)])
            base = root / "base.config"
            base.write_text(valid_config())
            existing = root / "already-built"
            existing.mkdir()
            good = ["build_private_kernel.py", "--archive", str(archive), "--base-config", str(base),
                    "--base-config-sha256", build.sha256(base), "--workspace", str(root / "new-build")]
            for kind in ("archive_hash", "config_hash", "existing", "relative", "invalid_config"):
                argv = list(good)
                archive_hash = build.sha256(archive)
                if kind == "archive_hash":
                    archive_hash = "0" * 64
                if kind == "config_hash":
                    argv[argv.index("--base-config-sha256") + 1] = "0" * 64
                if kind == "existing":
                    argv[argv.index("--workspace") + 1] = str(existing)
                if kind == "relative":
                    argv[argv.index("--workspace") + 1] = "relative-build"
                base.write_text(valid_config() if kind != "invalid_config" else "CONFIG_MODULES=n\n")
                # Bind the changed content so invalid_config specifically tests semantic rejection.
                if kind == "invalid_config":
                    argv[argv.index("--base-config-sha256") + 1] = build.sha256(base)
                with self.subTest(kind=kind), patch.object(build.sys, "platform", "linux"), \
                     patch.object(build.sys, "argv", argv), patch.object(build, "ARCHIVE_SHA256", archive_hash), \
                     patch.object(build, "run") as runner, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as stopped:
                        build.main()
                    self.assertEqual(stopped.exception.code, 1)
                    runner.assert_not_called()

    def test_actual_input_hash_detects_same_path_content_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.config"
            path.write_text(valid_config())
            bound = build.sha256(path)
            path.write_text(valid_config().replace("-strix-gpu-host-test", "-strix-gpu-other"))
            self.assertNotEqual(build.sha256(path), bound)

    def test_tree_identity_detects_content_path_and_symlink_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source"
            path.write_bytes(b"first")
            first = build.tree_identity(root)
            path.write_bytes(b"second")
            self.assertNotEqual(first, build.tree_identity(root))
            path.write_bytes(b"first")
            self.assertEqual(first, build.tree_identity(root))
            path.rename(root / "renamed")
            self.assertNotEqual(first, build.tree_identity(root))

    def test_exact_gnu_note_and_dependency_receipt_fields(self):
        self.assertEqual(build.gnu_build_id("GNU Build ID: 0123456789ABCDEF"), "0123456789abcdef")
        for text in ("", "Build ID: abc", "Build ID: 0123456789abcdef\nBuild ID: 0123456789abcdef"):
            with self.assertRaises(ValueError):
                build.gnu_build_id(text)
        self.assertEqual(build.module_dependencies("thunderbolt,crc-itu-t\n"), ["thunderbolt", "crc_itu_t"])
        self.assertEqual(build.module_dependencies("\n"), [])
        for text in ("thunderbolt,thunderbolt", "../../module", "$(module)"):
            with self.assertRaises(ValueError):
                build.module_dependencies(text)

    def test_multicall_tools_keep_the_invoked_symlink_basename(self):
        with patch.object(build.shutil, "which", return_value="/usr/sbin/depmod"):
            self.assertEqual(build.tool_path("depmod"), str(Path("/usr/sbin/depmod").absolute()))
        with patch.object(build.shutil, "which", return_value=None):
            with self.assertRaises(ValueError):
                build.tool_path("depmod")

    def test_actual_metadata_capture_retains_only_bounded_output(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            text, record = build.capture_metadata([sys.executable, "-c", "print('metadata')"], root / "small.log", 2, dict(os.environ))
            self.assertEqual(text.strip(), "metadata")
            self.assertEqual(record["exit_code"], 0)
            with self.assertRaises(ValueError):
                build.capture_metadata([sys.executable, "-c", "print('x'*65536)"], root / "large.log", 2, dict(os.environ))

    @unittest.skipUnless(sys.platform == "linux", "actual process group cleanup uses Linux")
    def test_timeout_terminates_actual_child_process_group(self):
        import os
        import subprocess
        import time
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "pid"
            child_code = "import time; time.sleep(60)"
            code = ("import pathlib,subprocess,sys,time; "
                    f"c=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
                    f"pathlib.Path({str(pid_path)!r}).write_text(str(c.pid)); time.sleep(60)")
            with self.assertRaises(subprocess.TimeoutExpired):
                build.run([sys.executable, "-c", code], root / "timeout.log", 1, dict(os.environ))
            pid = int(pid_path.read_text())
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = Path(f"/proc/{pid}/stat")
                if not status.exists() or status.read_text().split()[2] == "Z":
                    break
                time.sleep(0.01)
            else:
                self.fail("descendant remained running after timeout")


if __name__ == "__main__":
    unittest.main()
