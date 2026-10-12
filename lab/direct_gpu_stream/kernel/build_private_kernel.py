#!/usr/bin/env python3
"""Build a pinned full kernel and stage its modules privately; never install/load."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

PIN = "72d3fcf802c45d00b300f25b848a93c3a2bd7c7e"
ARCHIVE_SHA256 = "f265c0eeb342d2f697c0ede885dcca5292e89bd6100ea3265fed409c589e537d"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REQUIRED_CONFIG = {
    "CONFIG_64BIT": "y", "CONFIG_X86_64": "y", "CONFIG_MODULES": "y",
    "CONFIG_USB4": "m", "CONFIG_USB4_CONFIGFS": "m", "CONFIG_USB4_STREAM": "m",
    "CONFIG_USB4_GPU_STREAM": "y", "CONFIG_DMA_SHARED_BUFFER": "y",
    "CONFIG_MODULE_COMPRESS": "n",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def private_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == REPO or REPO in resolved.parents:
        raise ValueError("build inputs and artifacts must remain outside the project")
    if any((parent / ".git").exists() for parent in (resolved, *resolved.parents)):
        raise ValueError("build inputs and artifacts must remain outside Git")
    return resolved


def tool_path(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise ValueError(f"required build tool unavailable: {name}")
    # kmod selects depmod/modinfo by argv[0]; resolving its symlink changes the API.
    return str(Path(executable).absolute())


def config_values(text: str) -> dict[str, str]:
    result = {}
    for line in text.splitlines():
        if line.startswith("CONFIG_") and "=" in line:
            key, value = line.split("=", 1)
            if key in result:
                raise ValueError(f"duplicate configuration key: {key}")
            result[key] = value
        elif line.startswith("# CONFIG_") and line.endswith(" is not set"):
            key = line[2:-11]
            if key in result:
                raise ValueError(f"duplicate configuration key: {key}")
            result[key] = "n"
    return result


def validate_config(text: str) -> None:
    values = config_values(text)
    for key, expected in REQUIRED_CONFIG.items():
        if values.get(key) != expected:
            raise ValueError(f"required configuration: {key}={expected}")
    version = values.get("CONFIG_LOCALVERSION", "")
    if not re.fullmatch(r'"-strix-gpu-[a-z0-9][a-z0-9.-]{0,47}"', version):
        raise ValueError("CONFIG_LOCALVERSION must be a unique -strix-gpu- suffix")
    for key in ("CONFIG_LOCALVERSION_AUTO", "CONFIG_MODULE_SIG_ALL"):
        if values.get(key, "n") != "n":
            raise ValueError(f"unsupported automatic version/signing: {key}")


def tree_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            record = [relative, "symlink", os.readlink(path)]
        elif path.is_file():
            record = [relative, "file", sha256(path)]
        else:
            continue
        digest.update(json.dumps(record, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def gnu_build_id(text: str) -> str:
    values = re.findall(r"Build ID:\s*([0-9a-fA-F]+)", text)
    if len(values) != 1 or len(values[0]) % 2 or len(values[0]) < 16:
        raise ValueError("missing or ambiguous GNU build ID")
    return values[0].lower()


def validate_archive(path: Path) -> None:
    """Allow only a single pinned tree and links that remain inside that tree."""
    prefix = f"linux-{PIN}"
    seen, links = set(), set()
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        for index, member in enumerate(archive):
            name = PurePosixPath(member.name)
            if index >= 500000 or name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != prefix:
                raise ValueError("archive entry escapes the pinned source tree")
            if str(name) in seen or any(str(parent) in links for parent in name.parents):
                raise ValueError("duplicate archive entry or traversal through a symlink")
            seen.add(str(name))
            total += member.size
            if total > 4 * 1024 ** 3 or not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError("unsupported archive entry type/size")
            if member.issym():
                target = PurePosixPath(member.linkname)
                if target.is_absolute():
                    raise ValueError("absolute archive symlink")
                parts = list(name.parent.parts)
                for part in target.parts:
                    if part == "..":
                        if len(parts) <= 1:
                            raise ValueError("archive symlink escapes the pinned source tree")
                        parts.pop()
                    elif part != ".":
                        parts.append(part)
                links.add(str(name))


def module_dependencies(text: str) -> list[str]:
    values = [item.strip().replace("-", "_") for item in text.strip().split(",") if item.strip()]
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", item) for item in values) or len(set(values)) != len(values):
        raise ValueError("malformed module dependency list")
    return values


def run(command: list[str], log: Path, timeout: int, env: dict[str, str]) -> dict:
    started = time.monotonic()
    with log.open("wb") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
        try:
            code = child.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            raise
    if code:
        raise RuntimeError(f"command failed ({code}); see {log}")
    return {"argv": command, "exit_code": code, "elapsed_seconds": round(time.monotonic() - started, 3),
            "log": str(log), "log_sha256": sha256(log)}


def capture_metadata(command: list[str], log: Path, timeout: int, env: dict[str, str]) -> tuple[str, dict]:
    record = run(command, log, min(timeout, 60), env)
    if log.stat().st_size > 64 * 1024:
        raise ValueError("module metadata exceeds retained 64 KiB limit")
    return log.read_text(encoding="utf-8"), record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--base-config-sha256", required=True)
    parser.add_argument("--workspace", required=True, type=Path, help="new private absolute directory")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--stage-timeout-seconds", type=int, default=7200)
    parser.add_argument("--private-host-deps", type=Path, help="optional private include/lib libelf directory")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this workflow requires Linux; it never installs or loads a module")
    try:
        if not 1 <= args.jobs <= 64 or not 60 <= args.stage_timeout_seconds <= 14400:
            raise ValueError("jobs/timeout outside supported bounds")
        if not args.workspace.is_absolute():
            raise ValueError("workspace must be absolute")
        archive, base, workspace = map(private_path, (args.archive, args.base_config, args.workspace))
        if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(workspace)):
            raise ValueError("workspace path contains unsupported build-shell metacharacters")
        if workspace.exists():
            raise ValueError("workspace must not exist; builds never reuse an ambiguous base")
        if sha256(archive) != ARCHIVE_SHA256:
            raise ValueError("archive does not match pinned upstream SHA256")
        validate_archive(archive)
        if not re.fullmatch(r"[0-9a-f]{64}", args.base_config_sha256) or sha256(base) != args.base_config_sha256:
            raise ValueError("base configuration SHA256 mismatch")
        validate_config(base.read_text())
        tools = {}
        for name in ("tar", "make", "gcc", "ld", "depmod", "modinfo", "readelf"):
            tools[name] = tool_path(name)
        host_arguments = []
        host_records = {}
        if args.private_host_deps:
            host = private_path(args.private_host_deps)
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(host)):
                raise ValueError("private dependency path contains unsupported compiler metacharacters")
            for name in ("include/libelf.h", "include/gelf.h", "lib/libelf.so"):
                if not (host / name).is_file():
                    raise ValueError(f"private dependency missing: {name}")
            host_arguments = [f"HOSTCFLAGS=-I{host}/include",
                              f"HOSTLDFLAGS=-L{host}/lib -Wl,-rpath,{host}/lib"]
            host_records = {str(path): sha256(path) for path in sorted(host.rglob("*")) if path.is_file()}
        recipe_paths = [HERE / name for name in ("build_private_kernel.py", "apply_pinned_kernel.py", "stream.patch",
                        "Kconfig.patch", "gpu_stream.c", "gpu_stream.h", "gpu_stream_logic.h")]
        recipe_paths.append(REPO / "include/strix/linux_direct_gpu_stream_uapi.h")
        recipe_records = {str(path.relative_to(REPO)): sha256(path) for path in recipe_paths}
        tool_records = {name: {"path": path, "sha256": sha256(Path(path))} for name, path in tools.items()}
        workspace.mkdir(parents=True)
        logs = workspace / "logs"
        logs.mkdir()
        commands = []
        # A closed build must not inherit make targets/options, compiler injection,
        # configuration overrides, or Python import hooks from the invoking shell.
        env = {key: value for key, value in os.environ.items()
               if key in ("PATH", "HOME", "TMPDIR", "TMP", "TEMP")}
        env.update({"LC_ALL": "C", "KBUILD_BUILD_USER": "strix", "KBUILD_BUILD_HOST": "private-build",
                    "KBUILD_BUILD_TIMESTAMP": "2026-10-12 00:00:00 +0000", "KBUILD_BUILD_VERSION": "1"})
        def invoke(argv: list[str], name: str) -> None:
            commands.append(run(argv, logs / f"{name}.log", args.stage_timeout_seconds, env))
        invoke([tools["tar"], "-xzf", str(archive), "-C", str(workspace)], "extract")
        source = workspace / f"linux-{PIN}"
        invoke([sys.executable, str(HERE / "apply_pinned_kernel.py"), "--source", str(source)], "apply")
        source_identity = tree_identity(source)
        build, stage = workspace / "build", workspace / "stage"
        build.mkdir()
        shutil.copyfile(base, build / ".config")
        make = [tools["make"], "-C", str(source), f"O={build}", "ARCH=x86_64",
                f"CC={tools['gcc']}", f"HOSTCC={tools['gcc']}", f"LD={tools['ld']}", *host_arguments]
        invoke([*make, "olddefconfig"], "configuration")
        validate_config((build / ".config").read_text())
        invoke([*make, f"-j{args.jobs}", "W=1", "KCFLAGS=-Werror", "bzImage", "modules"], "full-build")
        release = (build / "include/config/kernel.release").read_text().strip()
        if not re.fullmatch(r"[A-Za-z0-9.+_-]+", release) or "-strix-gpu-" not in release:
            raise ValueError("unexpected kernel release")
        invoke([*make, "modules_install", f"INSTALL_MOD_PATH={stage}", f"DEPMOD={tools['depmod']}"], "private-stage")
        invoke([tools["depmod"], "-b", str(stage), "-e", "-E", str(build / "Module.symvers"), release], "dependency-check")
        # depmod prints unresolved symbols but can return success: diagnostics are fail closed.
        dep_log = (logs / "dependency-check.log").read_text()
        if "WARNING:" in dep_log or "ERROR:" in dep_log:
            raise ValueError("depmod reported unresolved symbols or malformed staged modules")
        module_root = stage / "lib/modules" / release
        modules = sorted(module_root.rglob("*.ko"))
        names = {path.stem.replace("-", "_"): path for path in modules}
        if len(names) != len(modules) or not {"thunderbolt", "thunderbolt_stream"} <= names.keys():
            raise ValueError("missing or duplicate required stream/controller modules")
        records = {}
        for name, path in names.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError("unsupported module name")
            def metadata(argv: list[str], field: str) -> str:
                text, record = capture_metadata(argv, logs / f"{name}-{field}.log", args.stage_timeout_seconds, env)
                commands.append(record)
                return text
            vermagic = metadata([tools["modinfo"], "-F", "vermagic", str(path)], "vermagic").strip()
            if not vermagic or vermagic.split()[0] != release:
                raise ValueError(f"wrong module vermagic: {name}")
            deps = module_dependencies(metadata([tools["modinfo"], "-F", "depends", str(path)], "depends"))
            if any(dep not in names for dep in deps):
                raise ValueError(f"missing required dependency module: {name}")
            note = metadata([tools["readelf"], "-n", str(path)], "notes")
            records[name] = {"path": str(path.relative_to(stage)), "sha256": sha256(path),
                             "vermagic": vermagic, "gnu_build_id": gnu_build_id(note), "depends": deps}
        if tree_identity(source) != source_identity or sha256(archive) != ARCHIVE_SHA256 or sha256(base) != args.base_config_sha256:
            raise ValueError("build inputs changed during construction")
        if any(sha256(REPO / path) != digest for path, digest in recipe_records.items()) or \
           any(sha256(Path(path)) != digest for path, digest in host_records.items()) or \
           any(sha256(Path(record["path"])) != record["sha256"] for record in tool_records.values()):
            raise ValueError("recipe/tool/dependency inputs changed during construction")
        kernel_stage = stage / "boot"
        kernel_stage.mkdir()
        artifacts = {"bzImage": build / "arch/x86/boot/bzImage", "vmlinux": build / "vmlinux",
                     "System.map": build / "System.map", "config": build / ".config", "Module.symvers": build / "Module.symvers"}
        for name, path in artifacts.items():
            shutil.copyfile(path, kernel_stage / f"{name}-{release}")
        receipt = {"schema": "strix-private-full-kernel-build-v1", "upstream_commit": PIN,
                   "archive_sha256": ARCHIVE_SHA256, "source_tree_sha256": source_identity,
                   "base_config_sha256": args.base_config_sha256, "final_config_sha256": sha256(build / ".config"),
                   "kernel_release": release, "modules": records,
                   "kernel_artifacts": {name: {"path": str((kernel_stage / f"{name}-{release}").relative_to(stage)), "sha256": sha256(path)} for name, path in artifacts.items()},
                   "tools": tool_records, "recipe_sources_sha256": recipe_records, "private_host_dependencies_sha256": host_records,
                   "build_environment": env,
                   "commands": commands, "qualified": False, "boot_compatibility_verified": False,
                   "installed": False, "loaded": False, "hardware_executed": False,
                   "scope": "full pinned build and private staging only; requires target boot/config/signature and native qualification"}
        (workspace / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Private kernel/module stage: {stage}; release {release}; no install/load/qualification")
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Build stopped: {error}\n")


if __name__ == "__main__":
    main()
