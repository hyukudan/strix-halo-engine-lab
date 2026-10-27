"""Capture complete, unchanged engine translation units with a local CUDA compiler.

This Linux/GCC standalone profile is separate from the engine build. It records
inputs and PTX; it never claims production binary equivalence or qualification.
No compiler, dependency, source file, or verifier is downloaded by this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from runner import DEFAULT_MANIFEST, SHA256, file_hash, load_manifest


PROFILE = "linux-gcc-sm120-cxx20-standalone-release-v1"
SOURCES = {
    "carry_restore_kernel": "src/cuda/ds41_attention.cu",
    "hc_expand_kernel": "src/cuda/ds41_core_norm.cu",
}
PROFILE_FLAGS = (
    "--std=c++20", "--gpu-architecture=compute_120", "--machine=64",
    "-O3", "-DNDEBUG", "--fmad=false", "-prec-div=true",
    "-prec-sqrt=true", "-ftz=false",
    "-Xcompiler=-ffp-contract=off,-Wall,-Wextra",
)
INJECTED_ENV = (
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN", "CPATH",
    "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
    "COMPILER_PATH", "GCC_EXEC_PREFIX", "LIBRARY_PATH", "LD_PRELOAD",
    "LD_LIBRARY_PATH", "CFLAGS", "CXXFLAGS", "CUDAFLAGS", "NVCCFLAGS",
)
MAX_METADATA_BYTES = 16 * 1024 * 1024


def clean_environment(nvcc, host_cxx, temporary_directory):
    injected = [name for name in INJECTED_ENV if os.environ.get(name)]
    if injected:
        raise ValueError("remove injected compiler environment variables: " + ", ".join(injected))
    # The subprocess environment is deliberately small; no caller options or
    # include paths are inherited. PATH remains explicit in the receipt.
    return {
        "PATH": os.pathsep.join(dict.fromkeys(
            (str(nvcc.parent), str(host_cxx.parent), "/usr/bin", "/bin"))),
        "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
        "TMPDIR": str(temporary_directory),
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
    }


def executable(path, name):
    result = Path(path).resolve(strict=True)
    if not result.is_file() or not os.access(result, os.X_OK):
        raise ValueError(name + " must be an executable file")
    return result


def select_source(kernel, manifest_path=DEFAULT_MANIFEST):
    manifest = load_manifest(manifest_path)
    cases = [case for case in manifest["cases"] if case["source_kernel"] == kernel]
    if kernel not in SOURCES or not cases:
        raise ValueError("kernel is outside the capture allowlist")
    identities = {(case["source_path"], case["source_revision"]) for case in cases}
    if len(identities) != 1 or next(iter(identities))[0] != SOURCES[kernel]:
        raise ValueError("cases disagree on the allowed source identity")
    source, revision = next(iter(identities))
    return source, revision, [case["id"] for case in cases]


def local_git(git, root, args, environment):
    result = subprocess.run(
        [str(git), "--no-replace-objects", "-C", str(root), *args],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, check=False,
    )
    if result.returncode:
        raise ValueError("local git could not read the declared source revision")
    if len(result.stdout) > MAX_METADATA_BYTES:
        raise ValueError("oversized git source object")
    return result.stdout


def check_source_revision(git, root, revision, paths, environment):
    """Compare actual bytes, including each project header, with git objects."""
    result = []
    for path in sorted(set(paths), key=str):
        path = Path(path)
        resolved = path.resolve(strict=True)
        if not path.is_relative_to(root) or not resolved.is_relative_to(root):
            raise ValueError("project input escapes the source checkout")
        relative = path.relative_to(root).as_posix()
        expected = local_git(git, root, ["show", revision + ":" + relative], environment)
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError("source differs from declared revision: " + relative)
        result.append({"path": relative, "sha256": file_hash(path)})
    return result


def parse_dependencies(text, root):
    """Read nvcc's single Make rule, including escaped spaces and continuations."""
    text = text.replace("\\\r\n", "").replace("\\\n", "")
    prefix = re.match(r"\Amodule\.ptx[ \t]*:", text)
    if prefix is None:
        raise ValueError("unexpected dependency target")
    tokens, token, index = [], [], prefix.end()
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 1
            if index >= len(text):
                raise ValueError("unterminated dependency escape")
            token.append(text[index])
        elif char == "$":
            if text[index:index + 2] != "$$":
                raise ValueError("unsupported Make expansion in dependencies")
            token.append("$")
            index += 1
        elif char.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        elif char in "#:\x00":
            raise ValueError("unexpected dependency syntax")
        else:
            token.append(char)
        index += 1
    if token:
        tokens.append("".join(token))
    if not tokens or len(tokens) > 10000:
        raise ValueError("empty or oversized dependency set")
    paths = set()
    for token in tokens:
        path = Path(token)
        path = Path(os.path.abspath(path if path.is_absolute() else root / path))
        if not path.is_file():
            raise ValueError("missing compiler dependency: " + str(path))
        if path.is_relative_to(root) and not path.resolve().is_relative_to(root):
            raise ValueError("project dependency escapes the source checkout")
        paths.add(path)
    return sorted(paths, key=str)


def fingerprint(paths):
    return [{"path": str(path), "resolved_path": str(path.resolve(strict=True)),
             "sha256": file_hash(path), "bytes": path.stat().st_size}
            for path in sorted(set(paths), key=str)]


def toolkit_files(nvcc, host_cxx, frontend):
    root = nvcc.parent.parent
    required = [nvcc, host_cxx, frontend, root / "bin/nvcc.profile",
                root / "bin/cudafe++", root / "nvvm/bin/cicc"]
    libdevice = list((root / "nvvm/libdevice").glob("*.bc"))
    if not libdevice or any(not path.is_file() for path in required):
        raise ValueError("capture requires a complete CUDA toolkit layout with libdevice")
    return required + libdevice + [
        path for path in (root / "nvvm/lib64").glob("*.so*") if path.is_file()
    ]


def discover_symbol(ptx, kernel):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.DOTALL)
    entries = re.findall(
        r"^\s*(?:\.(?:visible|weak)\s+)?\.entry\s+([A-Za-z_$][A-Za-z0-9_$.]*)\s*\(",
        text, re.MULTILINE,
    )
    marker = str(len(kernel)) + kernel
    selected = [entry for entry in entries
                if entry == kernel or (entry.startswith("_Z") and marker in entry)]
    if len(selected) != 1:
        raise ValueError("expected exactly one compiled entry for " + kernel)
    targets = re.findall(r"^\s*\.target\s+([^\n]+)", text, re.MULTILINE)
    versions = re.findall(r"^\s*\.version\s+([0-9]+\.[0-9]+)\s*$", text, re.MULTILINE)
    if len(targets) != 1 or targets[0].strip() != "sm_120" or len(versions) != 1:
        raise ValueError("unexpected PTX target or version declaration")
    return {"symbol": selected[0], "entry_count": len(entries),
            "target": "sm_120", "ptx_version": versions[0]}


def run_logged(command, label, directory, source_root, environment, deadline, commands):
    record = {"label": label, "argv": [str(part) for part in command],
              "cwd": str(source_root), "return_code": None, "timed_out": False}
    commands.append(record)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("capture time budget exhausted")
    stdout_path, stderr_path = directory / (label + ".stdout.txt"), directory / (label + ".stderr.txt")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(command, cwd=source_root, env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record["return_code"] = process.returncode
    record["logs"] = fingerprint([stdout_path, stderr_path])
    if record["timed_out"] or process.returncode != 0:
        raise ValueError("compiler step failed: " + label)
    return stdout_path


def bounded_text(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES:
        raise ValueError("oversized capture metadata")
    return data.decode("utf-8")


def capture(args, manifest_path=DEFAULT_MANIFEST):
    if sys.platform != "linux":
        raise ValueError("this capture profile supports Linux with GCC only")
    if not math.isfinite(args.timeout_seconds) or not 0.1 <= args.timeout_seconds <= 600:
        raise ValueError("timeout must be between 0.1 and 600 seconds")
    if not SHA256.fullmatch(args.nvcc_sha256):
        raise ValueError("nvcc SHA256 must be 64 lowercase hexadecimal characters")
    source_root = Path(args.source_root).resolve(strict=True)
    nvcc, host_cxx = executable(args.nvcc, "nvcc"), executable(args.host_cxx, "host compiler")
    if file_hash(nvcc) != args.nvcc_sha256:
        raise ValueError("nvcc SHA256 does not match the supplied executable")
    output = Path(args.output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory must be new")
    output = output.parent.resolve(strict=True) / output.name
    checkout = Path(__file__).resolve().parents[2]
    if output.is_relative_to(source_root) or output.is_relative_to(checkout):
        raise ValueError("capture output must be outside source and pilot checkouts")
    environment = clean_environment(nvcc, host_cxx, output / "tmp")
    git_name = shutil.which("git", path=environment["PATH"])
    if not git_name:
        raise ValueError("git is required for local source validation")
    git = Path(git_name).resolve()
    top = local_git(git, source_root, ["rev-parse", "--show-toplevel"], environment).decode().strip()
    if Path(top).resolve() != source_root:
        raise ValueError("source root must be the Git checkout root")
    # Also reject another Git worktree as an artifact destination.
    probe = subprocess.run([str(git), "-C", str(output.parent), "rev-parse", "--show-toplevel"],
                           env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=30, check=False)
    if probe.returncode == 0:
        raise ValueError("capture output must be outside Git worktrees")
    source, revision, case_ids = select_source(args.kernel, manifest_path)
    source_path = source_root / source
    base_paths = [source_path, source_root / "CMakeLists.txt"]
    check_source_revision(git, source_root, revision, base_paths, environment)
    manifest_hash = file_hash(manifest_path)
    script_hash = file_hash(Path(__file__))
    output.mkdir(mode=0o700)
    (output / "tmp").mkdir(mode=0o700)
    receipt = {
        "schema": "strix/ptx-capture-v1", "profile": PROFILE, "outcome": "ERROR",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_kind": "standalone_complete_translation_unit",
        "source_root": str(source_root), "source_path": source,
        "source_revision": revision, "source_kernel": args.kernel, "case_ids": case_ids,
        "source_files_match_declared_revision": False,
        "production_build_equivalence": "not_attested",
        "toolchain_attestation": "recorded_files_only",
        "native_qualified": False, "model_qualified": False, "bitwise_verified": False,
        "manifest_sha256": manifest_hash, "capture_script_sha256": script_hash,
        "environment": environment, "commands": [],
    }
    deadline = time.monotonic() + args.timeout_seconds
    try:
        def run(command, label):
            return run_logged(command, label, output, source_root, environment,
                              deadline, receipt["commands"])
        receipt["nvcc_version"] = bounded_text(run([str(nvcc), "--version"], "nvcc-version"))
        receipt["host_cxx_version"] = bounded_text(run([str(host_cxx), "--version"], "host-version"))
        frontend_name = bounded_text(run([str(host_cxx), "-print-prog-name=cc1plus"], "host-frontend")).strip()
        frontend = executable(frontend_name, "GCC cc1plus frontend")
        tool_paths = toolkit_files(nvcc, host_cxx, frontend)
        tools_before = fingerprint(tool_paths)
        if next(item for item in tools_before if item["path"] == str(nvcc))["sha256"] != args.nvcc_sha256:
            raise ValueError("nvcc changed before compilation")
        receipt["toolchain_files"] = tools_before
        common = [str(nvcc), *PROFILE_FLAGS, "--compiler-bindir=" + str(host_cxx),
                  "--include-path=" + str(source_root / "include")]
        dep_before = output / "dependencies-before.d"
        run([*common, "--generate-dependencies", "--dependency-target-name=module.ptx",
             "--dependency-output=" + str(dep_before), str(source_path)], "dependencies-before")
        dependencies = parse_dependencies(bounded_text(dep_before), source_root)
        if source_path not in dependencies:
            raise ValueError("dependency closure omitted the translation unit")
        project_paths = sorted(set(base_paths + [p for p in dependencies if p.is_relative_to(source_root)]))
        source_files = check_source_revision(git, source_root, revision, project_paths, environment)
        before = fingerprint(dependencies + base_paths)
        receipt["source_files"] = source_files
        receipt["dependencies"] = before
        ptx_path, dep_built = output / "module.ptx", output / "dependencies-built.d"
        run([*common, "--ptx", "--generate-dependencies-with-compile",
             "--dependency-target-name=module.ptx", "--dependency-output=" + str(dep_built),
             str(source_path), "--output-file=" + str(ptx_path)], "compile")
        if parse_dependencies(bounded_text(dep_built), source_root) != dependencies:
            raise ValueError("compiled dependency closure differs from the checked closure")
        if fingerprint(dependencies + base_paths) != before:
            raise ValueError("source or dependency changed during compilation")
        if fingerprint(tool_paths) != tools_before:
            raise ValueError("compiler toolchain changed during compilation")
        if (file_hash(manifest_path) != manifest_hash
                or file_hash(Path(__file__)) != script_hash):
            raise ValueError("capture script or catalog changed during compilation")
        if check_source_revision(git, source_root, revision, project_paths, environment) != source_files:
            raise ValueError("project inputs changed after source validation")
        ptx_info = discover_symbol(bounded_text(ptx_path), args.kernel)
        receipt.update(
            outcome="CAPTURED", source_files_match_declared_revision=True,
            ptx={**fingerprint([ptx_path])[0], **ptx_info},
            reason="complete translation unit captured under the recorded standalone profile",
        )
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["reason"] = str(error)
    finally:
        # Only temporary compiler files are removed; complete PTX and all logs stay private.
        shutil.rmtree(output / "tmp")
        with (output / "capture.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--kernel", choices=sorted(SOURCES), required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--nvcc-sha256", required=True)
    parser.add_argument("--host-cxx", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="new directory, with an existing parent outside Git worktrees")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    try:
        receipt = capture(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"outcome": "ERROR", "reason": str(error)}))
        return 3
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["outcome"] == "CAPTURED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
