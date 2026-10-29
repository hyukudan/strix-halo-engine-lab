"""RTX session qualification; never invoked by CPU tests."""
import argparse
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import shutil
import shlex
import re
import struct
import subprocess
import threading
import time

if not __debug__:
    raise RuntimeError("qualification requires Python assertions (no -O/PYTHONOPTIMIZE)")

def gate_device():
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert device and (device.isdecimal() or device.startswith("GPU-")) and "," not in device, \
        "set CUDA_VISIBLE_DEVICES to one GPU ordinal or UUID"
    return device

PARTS = ("tokens.u32", "logits.f32", "attention.bin", "drafter.bin", "frontier.bin")
ORDINARY_PARTS = ("tokens.u32", "greedy.u32", "logits.f32", "attention.bin", "drafter.bin")
ORDINARY_OUTPUT_CONTRACT = {
    "stdout_protocol_result_bytes": "identical to the labelled baseline",
    "default_session_policy": "on for rtx-tp1-decode-rows; explicit off controls retain their policy reason",
    "stderr_comparison": "not compared; worker log combines stdout and stderr",
    "baseline_evidence": "native qualification records and drafter trace, excluding timings and operational fields",
    "drafter_state_evidence": "candidate-internal B == A == L == parked raw bytes",
}


class WindowsJob:
    """Atomic job assignment at CreateProcess; no uncontained runnable child.

    Windows 10's JOB_LIST attribute attaches the child during creation. It is
    also suspended until Budget approves execution under its short state lock.
    The watchdog can terminate the job while process creation/return is blocked.
    """
    def __init__(self):
        import ctypes
        from ctypes import wintypes as w
        self.ctypes, self.w = ctypes, w
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([w.LPVOID, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "InitializeProcThreadAttributeList": ([w.LPVOID, w.DWORD, w.DWORD, ctypes.POINTER(ctypes.c_size_t)], w.BOOL),
            "UpdateProcThreadAttribute": ([w.LPVOID, w.DWORD, ctypes.c_size_t, w.LPVOID, ctypes.c_size_t, w.LPVOID, w.LPVOID], w.BOOL),
            "DeleteProcThreadAttributeList": ([w.LPVOID], None),
            "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, w.LPVOID, w.LPVOID, w.BOOL, w.DWORD, w.LPVOID, w.LPCWSTR, w.LPVOID, w.LPVOID], w.BOOL),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, including pointer-sized counters.
        class Basic(ctypes.Structure):
            _fields_ = [("times", ctypes.c_int64 * 2), ("flags", w.DWORD),
                        ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                        ("active", w.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class Limits(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6),
                        ("memory", ctypes.c_size_t * 4)]
        self.handle = self.api.CreateJobObjectW(None, None)
        self.check(self.handle)
        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self.check(self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def check(self, result):
        if not result:
            raise self.ctypes.WinError(self.ctypes.get_last_error())
        return result

    def terminate(self):
        self.check(self.api.TerminateJobObject(self.handle, 124))

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None

    def start(self, cmd, stdout=None, stderr=None, env=None):
        import msvcrt
        c, w = self.ctypes, self.w
        class Startup(c.Structure):
            _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR),
                        ("title", w.LPWSTR), ("geometry", w.DWORD * 7),
                        ("flags", w.DWORD), ("show", w.WORD), ("reserved_size", w.WORD),
                        ("reserved_ptr", w.LPVOID), ("stdin", w.HANDLE),
                        ("stdout", w.HANDLE), ("stderr", w.HANDLE), ("attributes", w.LPVOID)]
        class ProcessInfo(c.Structure):
            _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE),
                        ("pid", w.DWORD), ("tid", w.DWORD)]
        si, pi = Startup(), ProcessInfo()
        si.cb = c.sizeof(si)
        fds = []
        attrs = None
        initialized = False
        try:
            count = 1
            if stdout is not None:
                assert stderr == subprocess.STDOUT
                fds = [os.open(os.devnull, os.O_RDONLY)]
                fds.append(os.dup(stdout.fileno()))
                for fd in fds:
                    os.set_inheritable(fd, True)
                handles = (w.HANDLE * 2)(*(msvcrt.get_osfhandle(fd) for fd in fds))
                si.flags = 0x100  # STARTF_USESTDHANDLES
                si.stdin, si.stdout, si.stderr = handles[0], handles[1], handles[1]
                count += 1
            else:
                assert stderr is None
            size = c.c_size_t()
            self.api.InitializeProcThreadAttributeList(None, count, 0, c.byref(size))
            attrs = c.create_string_buffer(size.value)
            self.check(self.api.InitializeProcThreadAttributeList(attrs, count, 0, c.byref(size)))
            initialized = True
            si.attributes = c.addressof(attrs)
            jobs = (w.HANDLE * 1)(self.handle)
            self.check(self.api.UpdateProcThreadAttribute(attrs, 0, 0x2000D, jobs, c.sizeof(jobs), None, None))
            if stdout is not None:
                self.check(self.api.UpdateProcThreadAttribute(attrs, 0, 0x20002, handles, c.sizeof(handles), None, None))
            command = c.create_unicode_buffer(subprocess.list2cmdline(cmd))
            environment = None if env is None else c.create_unicode_buffer(
                "\0".join(f"{k}={v}" for k, v in sorted(env.items(), key=lambda x: x[0].upper())) + "\0")
            # EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED. Unsupported job
            # attributes fail closed; never fall back to an uncontained launch.
            self.check(self.api.CreateProcessW(None, command, None, None, bool(fds),
                       0x80404, environment, None, c.byref(si), c.byref(pi)))
            return WindowsProcess(self, pi.process, pi.thread, pi.pid)
        finally:
            if initialized:
                self.api.DeleteProcThreadAttributeList(attrs)
            for fd in fds:
                os.close(fd)


class WindowsProcess:
    """Minimal process interface used by Worker; owns both native handles."""
    def __init__(self, job, handle, thread, pid):
        import _winapi
        self.native = _winapi
        self.job, self.handle, self.thread, self.pid = job, handle, thread, pid
        self.returncode = None

    def resume(self):
        result = self.job.api.ResumeThread(self.thread)
        if result == 0xFFFFFFFF:
            raise self.job.ctypes.WinError(self.job.ctypes.get_last_error())
        self.job.api.CloseHandle(self.thread)
        self.thread = None

    def poll(self):
        if self.returncode is None and self.native.WaitForSingleObject(self.handle, 0) == 0:
            self.returncode = self.native.GetExitCodeProcess(self.handle)
        return self.returncode

    def wait(self, timeout):
        if self.native.WaitForSingleObject(self.handle, max(0, int(timeout * 1000))) != 0:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        return self.poll()

    def kill(self):
        if self.poll() is None:
            self.native.TerminateProcess(self.handle, 124)

    def __del__(self):
        for handle in (self.thread, self.handle):
            if handle:
                self.job.api.CloseHandle(handle)


class Budget:
    """One wall deadline, including input I/O, hashing, loads, comparisons and exits.

    The watchdog terminates the worker job two seconds before the deadline; it
    exits the driver at the deadline even if a filesystem operation is stuck.
    """
    def __init__(self, seconds=420):
        assert 2 < seconds <= 420
        self.end = time.monotonic() + seconds
        self.job = WindowsJob() if os.name == "nt" else None
        self.workers = set()
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.expired = False
        self.watchdog = threading.Thread(target=self.watch, daemon=True)
        self.watchdog.start()

    def remaining(self):
        left = self.end - time.monotonic()
        if self.expired or left <= 2:
            raise TimeoutError("phase deadline (worker cleanup reserve reached)")
        return left - 2

    def start(self, cmd, **kwargs):
        self.remaining()
        if self.job is None:
            raise RuntimeError("RTX qualification requires Windows job containment")
        # Native creation atomically assigns the suspended child to the job.
        # No state lock spans this potentially blocking call.
        proc = self.job.start(cmd, **kwargs)
        with self.lock:
            if self.expired or time.monotonic() >= self.end - 2:
                proc.kill()
                proc.wait(timeout=max(.01, self.end - time.monotonic()))
                raise TimeoutError("phase deadline during worker startup")
            self.workers.add(proc)
            try:
                proc.resume()
            except BaseException:
                proc.kill()
                proc.wait(timeout=max(.01, self.end - time.monotonic()))
                self.workers.discard(proc)
                raise
            return proc

    def register(self, proc):
        # CPU-test registration; production startup uses contained start().
        with self.lock:
            self.workers.add(proc)
            if self.expired:
                proc.kill()
                proc.wait(timeout=max(.01, self.end - time.monotonic()))
                self.workers.discard(proc)
                raise TimeoutError("phase deadline")

    def forget(self, proc):
        with self.lock:
            self.workers.discard(proc)

    def kill_workers(self):
        with self.lock:
            self.expired = True
            if self.job is not None:
                self.job.terminate()
            workers = list(self.workers)
            for proc in workers:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
        for proc in workers:
            proc.wait(timeout=max(.01, self.end - time.monotonic()))

    def watch(self):
        if not self.done.wait(max(0, self.end - time.monotonic() - 2)):
            try:
                self.kill_workers()
            finally:
                # A cleanup exception must not disable the absolute deadline.
                if not self.done.wait(max(0, self.end - time.monotonic())):
                    print("FAIL: whole-run deadline; worker job terminated", flush=True)
                    os._exit(124)

    def reap_workers(self):
        self.kill_workers()

    def close(self, stop_watchdog=True):
        try:
            self.reap_workers()
        finally:
            if stop_watchdog:
                self.done.set()
                self.watchdog.join(timeout=max(.01, self.end - time.monotonic()))
            if self.job is not None:
                self.job.close()
                self.job = None


def ids(path):
    out = [int(x) for x in Path(path).read_text().split()]
    assert all(0 <= x < 129280 for x in out), "bad token fixture"
    return out


def seed_tokens(path):
    b = Path(path).read_bytes()
    assert b[:8] == b"DS41ATST"
    version, pos, pending, layers = struct.unpack_from("<4I", b, 8)
    assert version == 1 and layers == 40
    at = 24
    for _ in range(layers):
        at += 5
        for width in (1, 1, 1, 4, 4):
            n, = struct.unpack_from("<Q", b, at)
            at += 8 + n * width
    n, = struct.unpack_from("<Q", b, at)
    assert n == pos
    return list(struct.unpack_from("<" + "I" * n, b, at + 8))


class Worker:
    def __init__(self, args, out, sessions=True):
        args.budget.remaining()
        self.budget = args.budget
        self.out = out
        out.mkdir(parents=True)
        self.spool = out / "spool"
        self.seq = 0
        self.results = []
        self.stall_seconds = getattr(args, "stall_seconds", 20)
        self.window_seconds = getattr(args, "window_ms", 64) / 1000
        self.load_seconds = getattr(args, "load_seconds", 74)
        self.hash_seconds = getattr(args, "expected_hash_seconds", 90)
        self.log_offset = 0
        self.log_partial = b""
        self.last_progress = None
        self.log = (out / "worker.log").open("wb")
        cmd = [args.exe, args.model, "--receipt", args.receipt, "--import-state", args.seed,
               "--draft", args.draft, "--draft-device", "0", "--draft-vocab", "full", "--arena-gib", str(args.arena_gib),
               "--max-rows", "6", "--max-positions", str(args.max_positions),
               "--phases", "chats0", "--seqs-file", str(args.fixture), "--chats", "0",
               "--chat-prompt", "1", "--chat-steps", "0",
               "--controller-prior", "none", "--controller-costs", "none",
               "--residency-save", "none", "--residency-seed", "none",
               "--serve-idle", "1800", "--serve", str(self.spool), "--qual"]
        cmd += getattr(args, "extra_flags", [])
        if sessions == "off":
            cmd += ["--session-cache", "off"]
        elif sessions == "explicit":
            cmd += ["--session-cache", "on", "--session-disk", "off",
                    "--session-ram-bytes", str(args.host_mib * 1024 * 1024)]
        elif sessions == "on":
            cmd += ["--session-cache", "on"]
        elif sessions == "auto":
            cmd += ["--session-dir", str(args.out / "auto-tier"), "--session-max-parked", "1"]
        elif sessions == "auto-starved":
            # Default policy with a session RAM budget below the admission reserve.
            # Setting 0's prefetch options are launch options here: a name-only request carries only
            # prompt/observer fields, and the effective configuration equals the setting-0 reference.
            cmd += ["--session-dir", str(args.out / "starved-tier"), "--session-ram-bytes", "1",
                    "--prefetch-layer", str(SETTINGS[0]["prefetch_layer"]), "--env", SETTINGS[0]["env"]]
        elif sessions is True:
            cmd += ["--serve-session-mib", str(args.host_mib)]
        self.proc = None
        try:
            environment = os.environ.copy()
            if sessions in ("auto", "auto-starved"):
                environment.pop("STRIX_SESSION_CACHE", None)
                environment.pop("STRIX_SESSION_DISK", None)
            self.proc = self.budget.start(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=environment)
            self.wait(self.spool / "ready.json", seconds=self.load_seconds + 15)
            if sessions == "auto":
                self.default_config = default_policy(json.loads((self.spool / "ready.json").read_text()))
        except BaseException:
            self.abort()
            raise

    def read_progress(self, request):
        # Separate reader: worker owns the writer handle, including on Windows.
        with (self.out / "worker.log").open("rb") as stream:
            stream.seek(self.log_offset)
            data = stream.read()
            self.log_offset = stream.tell()
        lines = (self.log_partial + data).split(b"\n")
        self.log_partial = lines.pop()
        advanced = False
        for line in lines:
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            if event.get("event") == "session_progress" and event.get("request") == request:
                self.last_progress = event
                print(json.dumps(event), flush=True)
                advanced = True
        return advanced

    def wait(self, path, request=None, stall=None, seconds=None):
        started = last_live = time.monotonic()
        while not path.exists():
            now = time.monotonic()
            if request is not None and self.read_progress(request):
                last_live = now
            if ((stall is not None and now-last_live > stall) or
                    (seconds is not None and now-started > seconds)):
                raise TimeoutError(f"worker stall/deadline waiting for {path}; last progress: {self.last_progress}")
            self.budget.remaining()
            if self.proc.poll() is not None:
                raise RuntimeError(f"worker exited {self.proc.returncode}; see {self.out}")
            time.sleep(min(.05, self.budget.remaining()))
        self.budget.remaining()

    def req(self, expected=0, **fields):
        self.seq += 1
        name = f"{self.seq:04d}"
        vals = []
        for k, v in fields.items():
            vals.append("--" + k.replace("_", "-"))
            if v is not True:
                vals.append(str(v))
        self.budget.remaining()
        p = self.spool / "queue" / (name + ".tmp")
        p.write_text("\n".join(vals) + "\n")
        p.rename(p.with_suffix(".req"))
        result = self.spool / "done" / (name + ".json")
        if "session" in fields or "session_op" in fields:
            self.last_progress = None
            # Completion has its own bound as well as the no-progress bound.
            tokens = sum(len(Path(fields[key]).read_text().split()) for key in
                         ("session_tokens", "session_control_prefix", "session_history") if key in fields)
            tokens += fields.get("chat_prompt", 0)
            estimate = math.ceil(tokens / 6) * self.window_seconds
            seconds = 30 + self.hash_seconds + 2 * estimate + fields.get("session_steps", fields.get("chat_steps", 0)) * .2
            self.hash_seconds = 0
            self.wait(result, request=name, stall=self.stall_seconds, seconds=seconds)
        else:
            self.wait(result)
        r = json.loads(result.read_text())
        assert r["rc"] == expected and not r["fatal"], r
        self.results.append(r)
        return r

    def quit(self):
        self.req(quit=True)
        assert self.proc.wait(timeout=self.budget.remaining()) == 0
        self.budget.forget(self.proc)
        self.log.close()
        (self.out / "results.json").write_text(json.dumps(self.results, indent=2))

    def abort(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=max(.01, self.budget.end - time.monotonic()))
            self.budget.forget(self.proc)
        self.log.close()


def equal_dumps(a, b, parts=PARTS, generated=64):
    result = {}
    for part in parts:
        x = Path(str(a) + "." + part).read_bytes()
        y = Path(str(b) + "." + part).read_bytes()
        assert x == y, f"BITWISE FAILURE: {part}: {a} / {b}"
        if part == "tokens.u32":
            assert len(x) > 0 and len(x) % 4 == 0
            if generated is not None:
                assert len(x) == generated * 4
        if part == "logits.f32":
            assert len(x) > 0 and len(x) % (129280 * 4) == 0
            if generated is not None:
                assert len(x) == generated * 129280 * 4
        result[part] = {"bytes": len(x), "sha256": hashlib.sha256(x).hexdigest()}
    return result


# Different phase/window, prompt, prefetch, and environment settings. Every
# request carries a raw observer, which performs no intermediate capture/import.
SETTINGS = (
    dict(phases="chats0", chat_prompt=1, chat_steps=8, prefetch_layer=0,
         env="STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH=0"),
    # The worker canonical key is import-state|seqs-file|chat-prompt|chat.
    # Prompt 1's earlier plain vector cannot qualify prompt 3. Run a matching
    # plain arm in this request, including when setting 1 is the first request.
    dict(phases="chats0,schats0f3", chat_prompt=3, chat_steps=11, prefetch_layer=1,
         env="STRIX_DS41_REPLICA_WINDOW_ENGRAM_PREFETCH=1"),
)


def ordinary(w, args, label, setting):
    dest = w.out / label
    trace = Path(str(dest) + ".draft.jsonl")
    w.req(qual=True, chat_logits=True, qual_dump=dest, draft_trace=trace, **SETTINGS[setting])
    receipt = Path(str(dest) + ".draft.receipt.json")
    draft_receipt(trace, receipt, speculative=setting == 1)
    reference = Path(str(args.references[setting]) + ".draft.receipt.json")
    assert reference.read_bytes() == receipt.read_bytes(), f"drafter trace mismatch: {label}"
    equal_dumps(args.references[setting], dest, ORDINARY_PARTS, generated=None)


def negatives(w, args, mode):
    checkpoint = args.out / mode / "checkpoint"
    for kind in ("corrupt", "truncated", "incompatible"):
        dest = args.out / mode / kind
        shutil.copytree(checkpoint, dest)
        p = dest / ("manifest.bin" if kind == "incompatible" else "attention.bin")
        b = bytearray(p.read_bytes())
        if kind == "incompatible":
            # Change the compatibility identity and reseal the manifest; exercise
            # the same CPU refusal as a flag-mismatched worker without loading one.
            n, = struct.unpack_from("<Q", b)
            compat_at = 8 + n + 8
            b[compat_at] = ord("0") if b[compat_at] != ord("0") else ord("1")
            b[-64:] = hashlib.sha256(b[:-72]).hexdigest().encode("ascii")
        elif kind == "corrupt":
            b[len(b) // 2] ^= 1
        else:
            del b[len(b) // 2:]
        p.write_bytes(b)
        r = w.req(expected=2, session=kind, session_op="resume", session_path=dest)
        assert r["error"]["code"] == ("session_incompatible" if kind == "incompatible" else "session_corrupt"), r
        assert r["status"] == (409 if kind == "incompatible" else 422), r
        expected = "compatibility mismatch" if kind == "incompatible" else ("SHA-256", "truncated", "budget")
        assert any(s in r["error"]["message"] for s in ([expected] if isinstance(expected, str) else expected)), r
    r = w.req(expected=2, session="overflow", session_op="create", session_tokens=args.overflow)
    assert "token file" in r["error"]["message"], r


def first_worker(args, w, mode):
    ordinary(w, args, "consecutive-x", 0)
    ordinary(w, args, "consecutive-y", 1)
    ordinary(w, args, "reset-x", 0)
    for mode in (mode,):
        out = args.out / mode
        out.mkdir()
        turn = dict(session_dspark=mode, session_fixed=3 if mode == "on" else 0)
        # L runs P and S with the exact two chunk boundaries used by A, in a
        # single request. No capture/checkpoint/import at the boundary P.
        w.req(session="L", session_op="create", session_control_prefix=args.prompt_tail,
              session_tokens=args.suffix, session_steps=64, session_dump=out / "L", **turn)
        w.req(session="L", session_op="delete")
        w.req(session="A", session_op="create", session_tokens=args.prompt_tail, **turn)
        w.req(session="A", session_op="checkpoint", session_durability="ram")
        w.req(session="A", session_op="checkpoint", session_durability="disk", session_path=out / "checkpoint")
        w.req(session="A", session_tokens=args.suffix, session_steps=64, session_dump=out / "A", **turn)
        equal_dumps(out / "A", out / "L")
        w.req(session="ParkA", session_op="resume", session_path=out / "checkpoint")
        w.req(session="ParkA", session_op="resume")
        w.req(session="ParkA", session_op="park")
        w.req(session="B", session_op="create", session_tokens=args.suffix, session_steps=3, **turn)
        w.req(session="B", session_op="park")
        # Ordinary requests reset app/runtime state between parked sessions.
        ordinary(w, args, "interleaved-" + mode + "-y", 1)
        ordinary(w, args, "interleaved-" + mode + "-x", 0)
        w.req(session="ParkA", session_op="resume")
        w.req(session="ParkA", session_tokens=args.suffix, session_steps=64, session_dump=out / "parked", **turn)
        equal_dumps(out / "A", out / "parked")
        for name in ("A", "ParkA", "B"):
            w.req(session=name, session_op="delete")
        w.req(session_op="list")
        w.req(session_op="stats")


def restarted_worker(args, w, mode):
    report = {}
    for mode in (mode,):
        out = args.out / mode
        w.req(session="A", session_op="resume", session_path=out / "checkpoint")
        w.req(session="A", session_tokens=args.suffix, session_steps=64, session_dump=out / "B",
              session_dspark=mode, session_fixed=3 if mode == "on" else 0)
        report[mode] = {"A_L": equal_dumps(out / "A", out / "L"),
                        "A_B": equal_dumps(out / "A", out / "B"),
                        "A_parked": equal_dumps(out / "A", out / "parked")}
        negatives(w, args, mode)
        ordinary(w, args, "after-restore-" + mode, 0)
        w.req(session="A", session_op="delete")
    return report


DISK_PARTS = ("manifest.bin", "attention.bin", "drafter.bin", "frontier.bin")
AUTO_TAIL_TOKENS = 192


def default_policy(ready):
    config = ready["session_config"]
    assert ready["env"]["STRIX_SESSION_CACHE"] is None, "automatic worker inherited a cache override"
    assert config["session_cache"] is True and config["session_explicit_opt_in"] is False, "cache not enabled by default"
    assert config["session_route"] == "rtx-tp1-decode-rows", "unqualified automatic route"
    assert not config["session_off_reason"], config
    return config


def auto_name(mode):
    return "D-" + mode


def auto_saved(args, mode):
    return args.out / "auto" / mode


def disk_evidence(path):
    return {part: {"bytes": (path / part).stat().st_size, "sha256": file_sha(path / part)}
            for part in DISK_PARTS}


def packed(value):
    value = value.encode() if isinstance(value, str) else value
    return struct.pack("<Q", len(value)) + value


class Record:
    def __init__(self, data):
        self.data, self.at = data, 0

    def integer(self):
        assert self.at + 8 <= len(self.data), "truncated record"
        value, = struct.unpack_from("<Q", self.data, self.at)
        self.at += 8
        return value

    def blob(self):
        size = self.integer()
        assert self.at + size <= len(self.data), "truncated record"
        value = self.data[self.at:self.at + size]
        self.at += size
        return value

    def text(self):
        return self.blob().decode("ascii")

    def end(self):
        prefix = self.data[:self.at]
        assert self.text() == hashlib.sha256(prefix).hexdigest(), "record SHA-256 mismatch"
        assert self.at == len(self.data), "trailing record bytes"


def managed_pointer(tier, name):
    r = Record((tier / (name + ".current")).read_bytes())
    assert r.text() == "strix-rtx-session-pointer-v1"
    assert r.text() == name
    directory, lineage, generation, manifest = r.text(), r.text(), r.integer(), r.text()
    assert re.fullmatch(re.escape(name) + r"\.g[0-9]+", directory), "invalid generation directory"
    r.end()
    return dict(directory=directory, lineage=lineage, generation=generation, manifest=manifest)


def managed_evidence(tier, name):
    pointer = managed_pointer(tier, name)
    path = tier / pointer["directory"]
    evidence = disk_evidence(path)
    assert evidence["manifest.bin"]["sha256"] == pointer["manifest"], "manifest SHA-256 mismatch"
    r = Record((path / "manifest.bin").read_bytes())
    assert r.text() == "strix-rtx-session-v2"
    compat, lineage, generation = r.text(), r.text(), r.integer()
    assert lineage == pointer["lineage"] and generation == pointer["generation"]
    assert r.integer() == 3
    for part in DISK_PARTS[1:]:
        assert r.text() == part and r.integer() == 0 and r.text() == "replicated" and r.integer() == 0
        assert r.integer() == evidence[part]["bytes"] and r.text() == evidence[part]["sha256"]
    r.end()
    return dict(pointer=pointer, compat=compat, sections=evidence)


def generated_dump(raw, dest, prompt_rows, mode, steps=64):
    # Ordinary observers include prompt and rejected verification rows. Select
    # exactly the committed generated rows, using native acceptance records.
    windows = [json.loads(line) for line in Path(str(raw) + ".events.jsonl").read_text().splitlines()]
    windows = [r for r in windows if r.get("event") == "window" and r.get("phase", "").startswith("schats")]
    selected, cursor = [], prompt_rows
    if mode == "on":
        for r in windows:
            kept = r["accepted"] + 1
            assert 0 < kept <= r["rows"] <= 6
            selected.extend(range(cursor, cursor + kept))
            cursor += r["rows"]
        assert len(selected) == steps, "missing generated-window evidence"
    else:
        selected = list(range(prompt_rows, prompt_rows + steps))
        cursor += steps
    for part, width in (("tokens.u32", 4), ("logits.f32", 129280 * 4)):
        source = Path(str(raw) + "." + part)
        assert source.stat().st_size == cursor * width, "unexpected ordinary raw rows"
        with source.open("rb") as src, Path(str(dest) + "." + part).open("wb") as dst:
            for row in selected:
                src.seek(row * width)
                data = src.read(width)
                assert len(data) == width
                dst.write(data)
    for part in ("attention.bin", "drafter.bin"):
        shutil.copyfile(Path(str(raw) + "." + part), Path(str(dest) + "." + part))


def full_prompt(w, out, label, prompt, mode, name=None, steps=64):
    contents = "session\t" + ",".join(map(str, prompt)) + "\n"
    fixture = out / ("prompt-" + hashlib.sha256(contents.encode()).hexdigest() + ".tsv")
    fixture.write_text(contents)
    dest = out / label
    raw = out / (label + "-raw")
    fields = dict(phases="schats0f3" if mode == "on" else "chats0", seqs_file=fixture,
                  chats=0, chat_prompt=len(prompt), chat_steps=steps, qual=True,
                  chat_logits=True, qual_dump=raw, json=Path(str(raw) + ".events.jsonl"),
                  draft_trace=Path(str(raw) + ".trace.jsonl"))
    if name is not None:
        fields["session"] = name
    assert not any(k.startswith("session_") for k in fields), "automatic request is not name-only"
    if mode == "on":
        # Canonical references belong to this worker, keyed by fixture path,
        # prompt length and chat index. Reuse the path for equal full prompts,
        # including name-only recomputations, without adding session controls.
        if not hasattr(w, "full_prompt_references"):
            w.full_prompt_references = set()
        key = (str(fixture), len(prompt), fields["chats"], steps)
        if key not in w.full_prompt_references:
            w.req(phases="chats0", seqs_file=fixture, chats=fields["chats"],
                  chat_prompt=len(prompt), chat_steps=steps, qual=True)
            w.full_prompt_references.add(key)
    response = w.req(**fields)
    result = response.get("session_result")
    if name is not None:
        assert result is not None, "default name-only admission was not executed"
    if result is None or result["recomputed"]:
        generated_dump(raw, dest, len(prompt), mode, steps)
    else:
        for part in PARTS[:-1]:
            shutil.copyfile(Path(str(raw) + "." + part), Path(str(dest) + "." + part))
    if result is not None:
        shutil.copyfile(Path(str(raw) + ".frontier.bin"), Path(str(dest) + ".frontier.bin"))
    return result


def generated_ids(base):
    data = Path(str(base) + ".tokens.u32").read_bytes()
    return list(struct.unpack("<" + "I" * (len(data) // 4), data))


def launch_seed_probe(args, w, out, prompt):
    # The ordinary oracle is a valid alternate import with a later position.
    # Do not issue a launch-import request between this replacement and create.
    alternate = Path(str(out / "L-latest") + ".attention.bin")
    w.req(import_state=alternate, phases="chats0", seqs_file=args.fixture,
          chats=0, chat_prompt=1, chat_steps=0, qual=True)
    tail, history = out / "seed-tail.ids", out / "seed-history.ids"
    tail.write_text("\n".join(map(str, prompt)))
    history.write_text("\n".join(map(str, seed_tokens(args.seed) + prompt)))
    name = "launch-seed"
    w.req(session=name, session_op="create", session_tokens=tail, session_steps=64,
          session_dump=out / "seed-create")
    report = {"create": equal_dumps(out / "seed-create", out / "L-latest", PARTS[:-1])}
    # The saved history also has 64 generated IDs, so this shorter full history
    # must fall back to the launch seed rather than continue the saved session.
    response = w.req(session=name, session_history=history, session_steps=64,
                     session_dump=out / "seed-history")
    assert not response["session_result"]["history_hit"], "seed probe did not exercise history fallback"
    report["history_fallback"] = equal_dumps(out / "seed-history", out / "L-latest", PARTS[:-1])
    w.req(session=name, session_op="delete")
    return report


def automatic_save(args, w, mode):
    tier = args.out / "auto-tier"
    def tier_inventory():
        return None if not tier.exists() else sorted(
            (str(p.relative_to(tier)), p.stat().st_size, p.stat().st_mtime_ns)
            for p in tier.rglob("*") if p.is_file())
    before = tier_inventory()
    report = {"default_config": w.default_config,
              "ordinary": ordinary_phase(args, w, 0, baseline=False)}
    assert tier_inventory() == before, "ordinary-only worker touched the session tier"
    prompt, suffix = ids(args.auto_prompt_tail), ids(args.suffix)
    assert prompt and suffix
    out = auto_saved(args, mode)
    out.mkdir(parents=True)
    name, durable = "A-" + mode, auto_name(mode)
    # All oracles are unnamed ordinary requests, never parked or saved.
    full_prompt(w, out, "L-latest", prompt + suffix, mode)
    continuation = prompt + suffix + generated_ids(out / "L-latest") + suffix
    full_prompt(w, out, "L-cont", continuation, mode)
    if mode == "off":
        report["launch_seed"] = launch_seed_probe(args, w, out, prompt + suffix)
    first = full_prompt(w, out, "A-first", prompt, mode, name, steps=0)
    assert first["recomputed"] and not first["history_hit"], first
    # Park A without exceeding one parked slot, then require a real hit.
    full_prompt(w, out, "B-first", [1], mode, "B-" + mode, steps=0)
    latest = full_prompt(w, out, "A-latest", prompt + suffix, mode, name)
    assert latest["history_hit"] and not latest["recomputed"], latest
    report[mode] = {"A_L_latest": equal_dumps(out / "A-latest", out / "L-latest", PARTS[:-1]),
                    "history_hit": latest["history_hit"]}
    assert not (tier / (name + ".current")).exists(), "RAM admission wrote disk"
    # A non-prefix full prompt must recompute exactly as an ordinary request.
    miss = [(prompt[0] + 1) % 129280] + prompt[1:]
    full_prompt(w, out, "miss-L", miss, mode)
    missed = full_prompt(w, out, "miss-A", miss, mode, name)
    assert missed["recomputed"] and not missed["history_hit"], missed
    report[mode]["non_prefix"] = equal_dumps(out / "miss-A", out / "miss-L", PARTS[:-1])
    longer = miss + generated_ids(out / "miss-A") + suffix
    # Without eviction this would be a history hit. Two other names evict A.
    full_prompt(w, out, "B-next", [2], mode, "B-" + mode, steps=0)
    full_prompt(w, out, "C-first", [3], mode, "C-" + mode, steps=0)
    inventory = w.req(session_op="list")["session_result"]["sessions"]
    assert not any(s["id"] == name for s in inventory), "A was not evicted"
    assert not (tier / (name + ".current")).exists(), "RAM eviction wrote disk"
    full_prompt(w, out, "evicted-L", longer, mode)
    evicted = full_prompt(w, out, "evicted-A", longer, mode, name)
    assert evicted["recomputed"] and not evicted["history_hit"], evicted
    report[mode]["ram_eviction"] = equal_dumps(out / "evicted-A", out / "evicted-L", PARTS[:-1])
    # Only the explicit checkpoint makes D durable.
    full_prompt(w, out, "D-first", prompt, mode, durable, steps=0)
    saved = full_prompt(w, out, "D-latest", prompt + suffix, mode, durable)
    assert saved["history_hit"] and saved["generation"] == 2, saved
    equal_dumps(out / "D-latest", out / "L-latest", PARTS[:-1])
    equal_dumps(out / "D-latest", out / "A-latest")
    w.req(session=durable, session_op="checkpoint", session_durability="disk")
    evidence = managed_evidence(tier, durable)
    assert evidence["pointer"]["generation"] == saved["generation"]
    assert evidence["pointer"]["lineage"] == saved["lineage"]
    assert evidence["compat"] == saved["compat"]
    for part, key in (("attention.bin", "attention_sha256"), ("drafter.bin", "drafter_sha256"),
                      ("frontier.bin", "frontier_sha256")):
        assert evidence["sections"][part]["sha256"] == saved[key], "checkpoint is not latest committed state"
        assert file_sha(Path(str(out / "D-latest") + "." + part)) == saved[key]
    (out / "latest.json").write_text(json.dumps(evidence, sort_keys=True))
    report[mode]["latest_disk"] = evidence
    return report


def verify_automatic_restore(args):
    # Read-only: saved explicit checkpoints must survive worker exit unchanged.
    for mode in ("off", "on"):
        expected = json.loads((auto_saved(args, mode) / "latest.json").read_text())
        assert managed_evidence(args.out / "auto-tier", auto_name(mode)) == expected, "managed state changed after save"


def corrupt_copy(tier, source, name):
    pointer = managed_pointer(tier, source)
    directory = name + ".g1"
    dest = tier / directory
    shutil.copytree(tier / pointer["directory"], dest)
    manifest = dest / "manifest.bin"
    data = manifest.read_bytes()
    manifest.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    record = b"".join(packed(v) for v in ("strix-rtx-session-pointer-v1", name, directory, pointer["lineage"]))
    record += struct.pack("<Q", pointer["generation"]) + packed(pointer["manifest"])
    (tier / (name + ".current")).write_bytes(record + packed(hashlib.sha256(record).hexdigest()))
    return dest


def automatic_restore(args, w):
    report = {"default_config": w.default_config}
    for mode in ("off", "on"):
        out, name = auto_saved(args, mode), auto_name(mode)
        tier = args.out / "auto-tier"
        bad_name = "corrupt-" + mode
        bad = corrupt_copy(tier, name, bad_name)
        before, good_before = disk_evidence(bad), managed_evidence(tier, name)
        stats = w.req(session_op="stats")["session_result"]
        r = w.req(expected=2, session=bad_name, session_op="restore", session_durability="disk")
        assert r["status"] == 422 and r["error"]["code"] == "session_corrupt", r
        assert "manifest SHA-256 mismatch" in r["error"]["message"], r
        assert disk_evidence(bad) == before and managed_evidence(tier, name) == good_before
        after = w.req(session_op="stats")["session_result"]
        for field in ("ram_bytes", "parked", "active", "evicted", "disk_bytes"):
            assert stats[field] == after[field], f"refusal changed {field}"
        # Explicit restore by managed name, after the previous owner's lock release.
        w.req(session=name, session_op="restore", session_durability="disk")
        w.req(session=name, session_tokens=args.suffix, session_steps=64, session_dump=out / "D-cont",
              session_dspark=mode, session_fixed=3 if mode == "on" else 0)
        report[mode] = {"D_cont_L_D_cont": equal_dumps(out / "D-cont", out / "L-cont", PARTS[:-1]),
                        "latest_disk": good_before, "corrupt_manifest_refusal": r["error"]}
        unknown = w.req(expected=2, session="unknown-" + mode, session_op="turn", session_tokens=args.suffix)
        assert unknown["status"] == 404 and unknown["error"]["code"] == "session_not_found", unknown
        report[mode]["unknown_suffix"] = unknown["error"]
    return report


def automatic_admission(args, w):
    # Automatic admission fails (RAM budget below the reserve): a name-only request runs as the
    # ordinary request it names and must equal the ordinary baseline reference byte for byte.
    dest = w.out / "starved-x"
    trace = Path(str(dest) + ".draft.jsonl")
    fields = {k: v for k, v in SETTINGS[0].items() if k not in ("prefetch_layer", "env")}
    r = w.req(session="starved", qual=True, chat_logits=True, qual_dump=dest, draft_trace=trace, **fields)
    assert r["rc"] == 0 and "session_result" not in r, r
    receipt = Path(str(dest) + ".draft.receipt.json")
    draft_receipt(trace, receipt, speculative=False)
    assert Path(str(args.references[0]) + ".draft.receipt.json").read_bytes() == receipt.read_bytes(), "drafter trace mismatch"
    equal_dumps(args.references[0], dest, ORDINARY_PARTS, generated=None)
    log = (w.out / "worker.log").read_text(errors="replace")
    assert "running it without a session" in log, "fallback line missing"
    return {"name_only_admission_failure": "ordinary run == baseline reference"}


def run_worker(args, name, action, sessions=True):
    model_dir = Path(args.model).parent
    before = set(model_dir.iterdir())
    try:
        w = Worker(args, args.out / name, sessions=sessions)
        try:
            result = action(w)
            w.quit()
            return result
        except BaseException:
            w.abort()
            raise
    finally:
        # Check after worker exit, including failed requests/startup. In
        # particular the model's automatic .residency must never be created.
        created = set(model_dir.iterdir()) - before
        assert not created, f"worker created files next to model: {sorted(map(str, created))}"


PHASES = tuple(f"{arm}-{setting}" for arm in ("baseline", "candidate") for setting in (0, 1)) + (
    "save-off", "restore-off", "save-on", "restore-on", "auto-save-target", "auto-save-dspark", "auto-restore",
    "auto-admission")


def prerequisites(phase):
    if phase.startswith("auto-save-"):
        return list(PHASES[:8])
    if phase == "auto-restore":
        return ["auto-save-target", "auto-save-dspark"]
    if phase.startswith("candidate-"):
        return ["baseline-0", "baseline-1"] + (["candidate-0"] if phase == "candidate-1" else [])
    if phase.startswith("save-"):
        return ["candidate-0", "candidate-1"]
    if phase.startswith("restore-"):
        return ["save-" + phase.split("-")[1]]
    return []


def qualification_receipt(events, dest):
    # Native --qual fields; omit timings, process IDs and output paths.
    keys = ("event", "phase", "k", "pos", "rows", "fed", "argmax", "logits_fnv",
            "fed_rows", "toks", "fnvs", "accepted", "key", "name", "position",
            "next_token", "digest", "chat", "len", "fnv")
    records = []
    for line in Path(events).read_text().splitlines():
        record = json.loads(line)
        if record.get("event") in ("prompt_window", "window", "chat_end", "chat_hash") or "logits_fnv" in record:
            records.append({key: record[key] for key in keys if key in record})
    assert any(r["event"] == "chat_end" and "digest" in r for r in records), "missing final state evidence"
    assert any(r.get("fnvs") or r.get("logits_fnv") for r in records), "missing logit evidence"
    data = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
    Path(dest).write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def draft_receipt(trace, dest, speculative):
    # Compare native --qual --draft-trace exact float bits, all
    # five proposed IDs (including unverified proposals), and accepted counts.
    # Decimal confidences and timing fields cannot establish bit equality.
    keys = ("phase", "pos", "t", "k", "rows", "a", "d", "cb", "g")
    records = []
    for line in Path(trace).read_text().splitlines():
        r = json.loads(line)
        assert all(key in r for key in keys), "incomplete drafter trace"
        assert len(r["d"]) == len(r["cb"]) == 5, "missing drafter proposals/confidence bits"
        assert all(isinstance(v, str) and len(v) == 8 and
                   all(c in "0123456789abcdef" for c in v) for v in r["cb"]), "invalid confidence bits"
        assert all(type(v) is int and 0 <= v < 129280 for v in r["d"]), "invalid draft IDs"
        assert type(r["a"]) is int and 0 <= r["a"] <= r["k"] <= 5, "invalid accepted count"
        records.append({key: r[key] for key in keys})
    assert bool(records) == speculative, "missing/unexpected drafter trace evidence"
    data = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
    Path(dest).write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "windows": len(records)}


def ordinary_phase(args, w, setting, baseline):
    # Switch-OFF / defaults-ON unnamed requests: changed settings, then exact reset.
    report = {}
    for label, index in (("reference", setting), ("next", 1 - setting), ("reset", setting)):
        dest = w.out / label
        events = Path(str(dest) + ".events.jsonl")
        trace = Path(str(dest) + ".draft.jsonl")
        fields = {} if baseline else dict(qual_dump=dest)
        w.req(qual=True, chat_logits=True, json=events, draft_trace=trace, **fields, **SETTINGS[index])
        receipt = Path(str(dest) + ".receipt.json")
        report[label] = qualification_receipt(events, receipt)
        draft_dest = Path(str(dest) + ".draft.receipt.json")
        report[label]["drafter_trace"] = draft_receipt(trace, draft_dest, speculative=index == 1)
        if not baseline:
            baseline_receipt = args.out / f"baseline-{setting}" / (label + ".receipt.json")
            assert baseline_receipt.read_bytes() == receipt.read_bytes(), f"baseline receipt mismatch: {label}"
            baseline_draft = args.out / f"baseline-{setting}" / (label + ".draft.receipt.json")
            assert baseline_draft.read_bytes() == draft_dest.read_bytes(), f"baseline drafter trace mismatch: {label}"
            report[label]["raw"] = equal_dumps(dest, dest, ORDINARY_PARTS, generated=None)
    if not baseline:
        equal_dumps(args.references[setting], w.out / "reset", ORDINARY_PARTS, generated=None)
    return report


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def baseline_build_info(exe, revision):
    """Inspect embedded public JSON without executing the baseline binary."""
    assert revision.strip(), "empty baseline revision label"
    found = []
    with Path(exe).open("rb") as stream:
        if not os.fstat(stream.fileno()).st_size:
            return {"verification": "unavailable"}
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            for match in re.finditer(rb'\{\s*"format"\s*:\s*"strix-build"', data):
                try:
                    info, _ = json.JSONDecoder().raw_decode(
                        data[match.start():match.start() + 65536].decode("ascii", errors="replace"))
                except ValueError:
                    continue
                sha = info.get("git_sha")
                if info.get("status") == "available" and sha:
                    assert re.fullmatch(r"[0-9a-fA-F]{7,40}", revision) and \
                        sha.lower().startswith(revision.lower()), "baseline revision differs from embedded build info"
                    found.append(sha)
    return {"verification": "verified", "git_sha": sorted(set(found))} if found else {"verification": "unavailable"}


def extra_flags(value):
    """Shared window budget overrides; cannot override hermetic/qualification flags."""
    flags = shlex.split(value)
    if len(flags) % 2:
        raise argparse.ArgumentTypeError("extra flags require option/value pairs")
    seen = set()
    for name, value in zip(flags[::2], flags[1::2]):
        if name not in ("--host-margin-gib", "--max-pinned-gib") or name in seen:
            raise argparse.ArgumentTypeError("extra flags allow unique --host-margin-gib / --max-pinned-gib only")
        try:
            number = float(value)
        except ValueError:
            number = float("nan")
        if not math.isfinite(number) or number < 0:
            raise argparse.ArgumentTypeError(f"invalid budget for {name}")
        seen.add(name)
    return flags


def phase_allowances(phase, window_ms=64, prompt_tokens=6144):
    # Two independent P producers (L then A), six admitted rows per window.
    # 64 ms = rounded 37.3 ms R6 base + ~33 cold misses at .8 ms/miss.
    # Includes matching plain regression arms; these are planning estimates.
    prompt = 2 * math.ceil(prompt_tokens / 6) * window_ms / 1000
    if phase.startswith("auto-save-"):
        # Thirteen full-prompt requests per mode. Target also probes an alternate
        # import, explicit create and history fallback; DSpark adds eight plain
        # references (four decode 64 rows, four establish zero-step prompts).
        target = phase == "auto-save-target"
        producers = 16 if target else 21
        return (producers * math.ceil((min(prompt_tokens, AUTO_TAIL_TOKENS) + 128) / 6) * window_ms / 1000,
                135 if target else 100 + 4 * 64 * window_ms / 1000)
    if phase == "auto-restore":
        return 5, 50
    return (prompt, 85) if phase.startswith("save-") else (5, 50 if phase.startswith("restore-") else 40)


def phase_deadline(expected, explicit=None):
    deadline = min(420, math.ceil(expected + max(10, expected * .08))) if explicit is None else explicit
    assert math.isfinite(deadline) and expected < deadline <= 420, "one phase must fit <=420 s including load and cleanup"
    return deadline


RECEIPT_PUBLICATION = "fsync-atomic-final-history-default-v7"


def accepted_receipt(path):
    record = json.loads(path.read_text())
    assert record.get("pass") is True, f"failed prerequisite {path.stem}"
    assert record.get("publication") == RECEIPT_PUBLICATION, f"unfinished prerequisite {path.stem}"
    return record


def publish_receipts(budget, started, seconds, phase_receipt, out, result, identity):
    # gate.json is an index only: qualification requires every final phase receipt.
    # Complete all other filesystem operations before publishing this phase.
    others = [phase for phase in PHASES if out / (phase + ".json") != phase_receipt]
    if all((out / (phase + ".json")).is_file() for phase in others):
        for phase in others:
            accepted_receipt(out / (phase + ".json"))
        with (out / "gate.json").open("w") as stream:
            json.dump(dict(phases=list(PHASES), run_identity=identity,
                           requires_final_receipts=True), stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    result["elapsed_seconds"] = time.monotonic() - started
    result["publication"] = RECEIPT_PUBLICATION
    pending = phase_receipt.with_suffix(".json.tmp")
    with pending.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    assert time.monotonic() < budget.end and time.monotonic() - started < seconds, "receipt publication exceeded phase deadline"
    # Commit point: nothing may follow this atomic rename except returning.
    return os.replace(pending, phase_receipt)


def main():
    started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=PHASES)
    for name in ("exe", "baseline-exe", "model", "receipt", "seed", "draft", "prompt", "suffix", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--baseline-revision", required=True,
                   help="baseline source label; checked against embedded build identity when available")
    p.add_argument("--draft-vocab", choices=("full",), default="full",
                   help="full draft vocabulary is required and forced on every worker arm")
    p.add_argument("--arena-gib", type=float, default=80)
    p.add_argument("--extra-flags", type=extra_flags, default=os.environ.get("SESSGATE_EXTRA", ""),
                   help='shared worker budgets, e.g. "--host-margin-gib 1 --max-pinned-gib 70" (or SESSGATE_EXTRA)')
    p.add_argument("--max-positions", type=int, default=16384)
    p.add_argument("--host-mib", type=int, default=512)
    p.add_argument("--deadline-seconds", type=float, help="override calibrated phase deadline (maximum 420 s)")
    p.add_argument("--window-ms", type=float, default=64, help="measured cold R6 window cost for timing estimates")
    p.add_argument("--load-seconds", type=float, default=74)
    p.add_argument("--stall-seconds", type=float, default=20, help="session no-progress deadline")
    p.add_argument("--expected-hash-seconds", type=float,
                   help="hash allowance (90 s; 5 s for automatic phases after warmed prerequisite phases)")
    p.add_argument("--expected-prompt-seconds", type=float)
    p.add_argument("--expected-decode-seconds", type=float)
    args = p.parse_args()
    assert all(math.isfinite(x) and x > 0 for x in (args.window_ms, args.load_seconds, args.stall_seconds))
    if args.expected_hash_seconds is None:
        args.expected_hash_seconds = 5 if args.phase.startswith("auto-") else 90
    prompt_seconds, decode_seconds = phase_allowances(args.phase, args.window_ms)
    if args.expected_prompt_seconds is None:
        args.expected_prompt_seconds = prompt_seconds
    if args.expected_decode_seconds is None:
        args.expected_decode_seconds = decode_seconds
    allowances = (args.expected_hash_seconds, args.expected_prompt_seconds, args.expected_decode_seconds)
    expected = args.load_seconds + sum(allowances) + 30
    assert all(math.isfinite(x) and x >= 0 for x in allowances)
    args.deadline_seconds = phase_deadline(expected, 420 if args.phase.startswith("auto-") and args.deadline_seconds is None else args.deadline_seconds)
    print(f"Phase {args.phase}: expected {expected:g} s (one load <={args.load_seconds:g} s, hashing {allowances[0]:g} s, "
          f"prompt {allowances[1]:g} s, decode/regression {allowances[2]:g} s, I/O/cleanup 30 s); "
          f"absolute deadline {args.deadline_seconds:g} s; session stall {args.stall_seconds:g} s; "
          f"R6 window model {args.window_ms:g} ms. Estimates require local calibration.", flush=True)
    args.budget = Budget(args.deadline_seconds - (time.monotonic() - started))
    try:
        device = gate_device()
        repo = Path(__file__).resolve().parents[1]
        args.out = Path(args.out).resolve()
        assert args.out != repo and repo not in args.out.parents, "qualification artifacts must stay outside Git"
        args.out.mkdir(parents=True, exist_ok=True)
        phase_receipt = args.out / (args.phase + ".json")
        assert not phase_receipt.exists() and not (args.out / args.phase).exists(), "phase already attempted; use a fresh run on failure"
        for phase in prerequisites(args.phase):
            receipt = args.out / (phase + ".json")
            assert receipt.is_file(), f"missing prerequisite {phase}"
            accepted_receipt(receipt)
        for name in ("exe", "baseline_exe", "model", "receipt", "seed", "draft", "prompt", "suffix"):
            setattr(args, name, str(Path(getattr(args, name)).resolve(strict=True)))
        identity = {name: getattr(args, name) for name in
                    ("exe", "baseline_exe", "baseline_revision", "model", "receipt", "seed", "draft", "prompt", "suffix",
                     "arena_gib", "extra_flags", "max_positions", "host_mib")}
        identity["cuda_visible_devices"] = device
        identity["gate_sha256"] = file_sha(__file__)
        identity["binary_sha256"] = file_sha(args.exe)
        identity["baseline_binary_sha256"] = file_sha(args.baseline_exe)
        identity["baseline_build_info"] = baseline_build_info(args.baseline_exe, args.baseline_revision)
        identity["draft_vocab"] = "full"  # forced by Worker for every arm
        identity["auto_tail_token_cap"] = AUTO_TAIL_TOKENS
        for name in ("receipt", "seed", "prompt", "suffix"):
            identity[name + "_sha256"] = file_sha(getattr(args, name))
        config = args.out / "run.json"
        if config.exists():
            assert json.loads(config.read_text()) == identity, "run inputs/binaries changed between phases"
        else:
            assert args.phase.startswith("baseline-"), "run must begin with baseline evidence"
            config.write_text(json.dumps(identity, indent=2))
        prompt, seed = ids(args.prompt), seed_tokens(args.seed)
        assert len(prompt) == 8192 and prompt[:len(seed)] == seed, "P must extend exact seed history to 8192 IDs"
        assert len(seed) <= 8192 and ids(args.suffix), "nonempty suffix required"
        args.prompt_tail = args.out / "prompt-tail.ids"
        args.prompt_tail.write_text("\n".join(map(str, prompt[len(seed):])))
        args.auto_prompt_tail = args.out / "auto-prompt-tail.ids"
        args.auto_prompt_tail.write_text("\n".join(map(str, prompt[len(seed):][:AUTO_TAIL_TOKENS])))
        args.overflow = args.out / "overflow.ids"
        args.overflow.write_text("1 18446744073709551616")
        args.fixture = args.out / "launch.tsv"
        args.fixture.write_text("session\t1,2,3\n")
        args.references = [args.out / f"candidate-{i}" / "reference" for i in range(2)]
        # Verify all on-disk prerequisites before uploading a model.
        required = []
        if args.phase.startswith("candidate-"):
            for i in range(2):
                for label in ("reference", "next", "reset"):
                    required.append(args.out / f"baseline-{i}" / (label + ".receipt.json"))
                    required.append(args.out / f"baseline-{i}" / (label + ".draft.receipt.json"))
            if args.phase == "candidate-1":
                for label in ("reference", "next", "reset"):
                    required += [Path(str(args.out / "candidate-0" / label) + "." + part) for part in ORDINARY_PARTS]
        elif args.phase.startswith(("save-", "restore-", "auto-")):
            for ref in args.references:
                required += [Path(str(ref) + "." + part) for part in ORDINARY_PARTS]
                required.append(Path(str(ref) + ".draft.receipt.json"))
            if args.phase.startswith("restore-"):
                mode = args.phase.split("-")[1]
                required += [args.out / mode / "checkpoint" / part for part in
                             ("manifest.bin", "attention.bin", "drafter.bin", "frontier.bin")]
                for label in ("A", "L", "parked"):
                    required += [Path(str(args.out / mode / label) + "." + part) for part in PARTS]
            if args.phase == "auto-restore":
                for mode in ("off", "on"):
                    required.append(auto_saved(args, mode) / "latest.json")
                    for label in ("A-latest", "L-latest", "L-cont"):
                        required += [Path(str(auto_saved(args, mode) / label) + "." + part) for part in PARTS[:-1]]
        for path in required:
            assert path.is_file(), f"missing prerequisite artifact {path}"
        report = {}
        if args.phase.startswith("auto-save-"):
            mode = "off" if args.phase == "auto-save-target" else "on"
            report = run_worker(args, args.phase, lambda w: automatic_save(args, w, mode), sessions="auto")
        elif args.phase == "auto-restore":
            verify_automatic_restore(args)
            report = run_worker(args, args.phase, lambda w: automatic_restore(args, w), sessions="auto")
        elif args.phase == "auto-admission":
            report = run_worker(args, args.phase, lambda w: automatic_admission(args, w), sessions="auto-starved")
        elif args.phase.startswith(("baseline-", "candidate-")):
            setting = int(args.phase[-1])
            baseline = args.phase.startswith("baseline-")
            if baseline:
                args.exe = args.baseline_exe
            report = run_worker(args, args.phase, lambda w: ordinary_phase(args, w, setting, baseline), sessions=False if baseline else ("off" if setting == 0 else "on"))
            if args.phase == "candidate-1":
                for i in range(2):
                    equal_dumps(args.references[1 - i], args.out / f"candidate-{i}" / "next",
                                ORDINARY_PARTS, generated=None)
        elif args.phase.startswith("save-"):
            mode = args.phase.split("-")[1]
            run_worker(args, args.phase, lambda w: first_worker(args, w, mode), sessions="explicit")
        else:
            mode = args.phase.split("-")[1]
            report = run_worker(args, args.phase, lambda w: restarted_worker(args, w, mode), sessions="explicit")
        args.budget.remaining()
        result = dict(phase=args.phase, **{"pass": True}, run_identity=identity,
                      modes=report, worker_launches=1, expected_seconds=expected,
                      ordinary_output_contract=ORDINARY_OUTPUT_CONTRACT)
        # Finish cleanup and output before the final receipt commit. The daemon
        # watchdog remains armed until this command returns and the driver exits.
        args.budget.close(stop_watchdog=False)
        print(f"Publishing success: {args.phase}; receipts {args.out}", flush=True)
        return publish_receipts(args.budget, started, args.deadline_seconds, phase_receipt,
                                args.out, result, identity)
    except BaseException:
        args.budget.close()
        raise


if __name__ == "__main__":
    main()
