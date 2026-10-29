"""CPU checks for the qualification driver; no CUDA or model fixtures."""
import importlib.util
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("gate", Path(__file__).resolve().parents[1] / "tools/rtx-session-gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateTests(unittest.TestCase):
    def test_baseline_revision_checks_embedded_identity_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "baseline.exe"
            for contents in (b"", b"MZ-no-build-info", b'{"format":"strix-build","status":"unavailable"}'):
                exe.write_bytes(contents)
                self.assertEqual(gate.baseline_build_info(exe, "label-only"), {"verification": "unavailable"})
            sha = "abcdef01" + "2" * 32
            record = gate.json.dumps(dict(format="strix-build", status="available", git_sha=sha)).encode()
            exe.write_bytes(b"MZ\0" + record + b"\0trailing binary")
            for revision in (sha, sha[:8], sha[:8].upper()):
                self.assertEqual(gate.baseline_build_info(exe, revision)["git_sha"], [sha])
            for revision in ("12345678", "abcd", "label-only"):
                with self.assertRaisesRegex(AssertionError, "embedded build info"):
                    gate.baseline_build_info(exe, revision)
            with self.assertRaisesRegex(AssertionError, "empty baseline"):
                gate.baseline_build_info(exe, " ")

    def test_name_only_history_and_explicit_durable_restart(self):
        # Exercise the real phase driver and pointer/manifest parser. Only the
        # numerical producer is modeled; no child, model or GPU is used here.
        class AutoWorker:
            def __init__(worker, root, fault=None):
                worker.root, worker.ram, worker.calls = root, {}, []
                worker.references = {}
                worker.fault, worker.active, worker.evicted = fault, "", 0
                worker.imported_history = []
                worker.default_config = {"session_cache": True, "session_explicit_opt_in": False,
                    "session_route": "rtx-tp1-decode-rows", "session_off_reason": ""}

            def dumps(worker, snapshot, output=()):
                state = gate.json.dumps({k: snapshot[k] for k in ("history", "mode")}, sort_keys=True).encode()
                return {"attention.bin": state, "drafter.bin": state + b"drafter",
                        "frontier.bin": state + b"frontier", "logits.f32": gate.json.dumps(list(output)).encode(),
                        "tokens.u32": gate.struct.pack("<" + "I" * len(output), *output)}

            def checkpoint(worker, name, snapshot):
                tier = worker.root / "auto-tier"
                snapshot = dict(snapshot)
                if worker.fault == "stale-checkpoint":
                    snapshot["history"] = [99]
                path = tier / (name + ".g" + str(snapshot["generation"]))
                path.mkdir(parents=True, exist_ok=True)
                manifest = gate.packed("strix-rtx-session-v2") + gate.packed("compat")
                manifest += gate.packed(snapshot["lineage"]) + gate.struct.pack("<QQ", snapshot["generation"], 3)
                for part in gate.DISK_PARTS[1:]:
                    data = worker.dumps(snapshot)[part]
                    (path / part).write_bytes(data)
                    manifest += gate.packed(part) + gate.struct.pack("<Q", 0) + gate.packed("replicated")
                    manifest += gate.struct.pack("<QQ", 0, len(data)) + gate.packed(gate.hashlib.sha256(data).hexdigest())
                manifest += gate.packed(gate.hashlib.sha256(manifest).hexdigest())
                (path / "manifest.bin").write_bytes(manifest)
                record = b"".join(gate.packed(v) for v in ("strix-rtx-session-pointer-v1", name, path.name, snapshot["lineage"]))
                record += gate.struct.pack("<Q", snapshot["generation"]) + gate.packed(gate.hashlib.sha256(manifest).hexdigest())
                (tier / (name + ".current")).write_bytes(record + gate.packed(gate.hashlib.sha256(record).hexdigest()))

            def req(worker, expected=0, **f):
                worker.calls.append(f)
                name, op = f.get("session"), f.get("session_op")
                tier = worker.root / "auto-tier"
                if "import_state" in f:
                    worker.imported_history = gate.json.loads(Path(f["import_state"]).read_bytes())["history"]
                    return {}
                if op == "delete":
                    worker.ram.pop(name, None)
                    return {}
                if op == "list":
                    return {"session_result": {"sessions": [dict(id=n) for n in worker.ram]}}
                if op == "stats":
                    return {"session_result": dict(ram_bytes=len(worker.ram), parked=len(worker.ram), active=worker.active,
                        evicted=worker.evicted, disk_bytes=sum(p.stat().st_size for p in tier.rglob("*") if p.is_file()))}
                if op == "checkpoint":
                    worker.checkpoint(name, worker.ram[name])
                    return {}
                if op == "restore":
                    try:
                        evidence = gate.managed_evidence(tier, name)
                    except AssertionError:
                        assert expected == 2
                        return dict(status=422, error=dict(code="session_corrupt", message="checkpoint manifest SHA-256 mismatch"))
                    old = gate.json.loads((tier / evidence["pointer"]["directory"] / "attention.bin").read_bytes())
                    old.update(generation=evidence["pointer"]["generation"], lineage=evidence["pointer"]["lineage"])
                    if worker.fault == "wrong-restore":
                        old["history"] = [99]
                    worker.ram[name] = old
                    return {}
                if "session_tokens" in f and name not in worker.ram and op != "create":
                    assert expected == 2
                    return dict(status=404, error=dict(code="session_not_found"))
                old = worker.ram.get(name)
                if "seqs_file" in f:
                    prompt = [int(v) for v in Path(f["seqs_file"]).read_text().split("\t")[1].strip().split(",")]
                    mode, steps = ("on" if f["phases"].startswith("schats") else "off"), f["chat_steps"]
                    key = (str(f["seqs_file"]), f["chat_prompt"], f["chats"])
                    if mode == "on":
                        assert worker.references.get(key, -1) >= steps, "missing matching plain reference"
                    elif name is None:
                        worker.references[key] = steps
                        if "qual_dump" not in f:
                            return {}
                    hit = old is not None and prompt[:len(old["history"])] == old["history"]
                    generation = old["generation"] + 1 if hit else 1
                    if worker.fault == "disabled-auto":
                        name = None
                    if worker.fault == "ignore-hit":
                        hit = False
                    history = prompt
                    dump = f["qual_dump"]
                    Path(str(dump) + ".events.jsonl").write_text("")
                else:
                    mode, steps = f.get("session_dspark", "off"), f["session_steps"]
                    if op == "create":
                        seed = worker.imported_history if worker.fault == "mutable-create-seed" else []
                        history = seed + gate.ids(f["session_tokens"])
                    elif "session_history" in f:
                        history = gate.ids(f["session_history"])
                        if worker.fault == "mutable-history-seed":
                            history = worker.imported_history + history
                    else:
                        history = old["history"] + gate.ids(f["session_tokens"])
                    generation, hit, dump = (old["generation"] + 1 if old else 1), False, f["session_dump"]
                output = [(history[-1] + i + 1) % 129280 for i in range(steps)]
                snapshot = dict(history=history + output, mode=mode, generation=generation,
                    lineage=old["lineage"] if old and hit else gate.hashlib.md5(str(name).encode()).hexdigest())
                if name is not None:
                    if name not in worker.ram and len(worker.ram) >= 2:
                        victim = next(iter(worker.ram))
                        if worker.fault == "automatic-spill":
                            worker.checkpoint(victim, worker.ram[victim])
                        del worker.ram[victim]
                        worker.evicted += 1
                    worker.ram.pop(name, None)
                    worker.ram[name], worker.active = snapshot, name
                else:
                    worker.active = ""
                dumps = worker.dumps(snapshot, output)
                for part, data in dumps.items():
                    Path(str(dump) + "." + part).write_bytes(data)
                if name is None:
                    return {}
                result = dict(generation=generation, lineage=snapshot["lineage"], compat="compat",
                              history_hit=hit, recomputed=not hit)
                for part, key in (("attention.bin", "attention_sha256"), ("drafter.bin", "drafter_sha256"), ("frontier.bin", "frontier_sha256")):
                    result[key] = gate.hashlib.sha256(dumps[part]).hexdigest()
                return {"session_result": result}

        def normalize(raw, dest, *unused):
            for part in gate.PARTS[:-1]:
                gate.shutil.copyfile(Path(str(raw) + "." + part), Path(str(dest) + "." + part))

        def compare(a, b, parts=gate.PARTS):
            for part in parts:
                self.assertEqual(Path(str(a) + "." + part).read_bytes(), Path(str(b) + "." + part).read_bytes(), part)
            return {part: "equal" for part in parts}

        for fault in (None, "disabled-auto", "ignore-hit", "automatic-spill", "stale-checkpoint", "wrong-restore",
                      "mutable-create-seed", "mutable-history-seed"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(out=Path(tmp), auto_prompt_tail=Path(tmp) / "prompt", suffix=Path(tmp) / "suffix",
                                       fixture=Path(tmp) / "launch.tsv", seed=Path(tmp) / "seed")
                args.auto_prompt_tail.write_text("4 5 6")
                args.suffix.write_text("7 8")
                saved = AutoWorker(args.out, None if fault == "wrong-restore" else fault)
                with mock.patch.object(gate, "ordinary_phase", return_value={}) as ordinary, \
                     mock.patch.object(gate, "seed_tokens", return_value=[]), \
                     mock.patch.object(gate, "generated_dump", side_effect=normalize), \
                     mock.patch.object(gate, "equal_dumps", side_effect=compare):
                    if fault and fault != "wrong-restore":
                        with self.assertRaises(AssertionError):
                            gate.automatic_save(args, saved, "off")
                        continue
                    gate.automatic_save(args, saved, "off")
                    ordinary.assert_called_once_with(args, saved, 0, baseline=False)
                    calls = saved.calls[:]
                    # The second save has a fresh default worker and empty RAM.
                    saved = AutoWorker(args.out)
                    gate.automatic_save(args, saved, "on")
                    self.assertEqual(ordinary.call_count, 2)
                    calls += saved.calls
                    named = [c for c in calls if "session" in c and "seqs_file" in c]
                    self.assertTrue(named)
                    self.assertTrue(all(not any(k.startswith("session_") for k in c) for c in named))
                    self.assertFalse(any(c.get("session", "").startswith("L") for c in calls))
                    replacements = [i for i, c in enumerate(calls) if "import_state" in c]
                    self.assertEqual(len(replacements), 1)
                    self.assertEqual(calls[replacements[0] + 1]["session_op"], "create")
                    self.assertIn("session_history", calls[replacements[0] + 2])
                    references = [c for c in calls if "seqs_file" in c and "qual_dump" not in c and "import_state" not in c]
                    self.assertEqual(len(references), 8)  # accounted for in phase_allowances
                    checkpoints = [c for c in calls if c.get("session_op") == "checkpoint"]
                    self.assertEqual([c["session"] for c in checkpoints], [gate.auto_name(m) for m in ("off", "on")])
                    gate.verify_automatic_restore(args)
                    restarted = AutoWorker(args.out, fault)
                    if fault:
                        with self.assertRaises(AssertionError):
                            gate.automatic_restore(args, restarted)
                    else:
                        report = gate.automatic_restore(args, restarted)
                        self.assertEqual(set(report), {"off", "on", "default_config"})
                        restores = [c for c in restarted.calls if c.get("session_op") == "restore"]
                        self.assertEqual(len(restores), 4)
                        self.assertTrue(all("session_path" not in c for c in restores))
                        self.assertTrue(all(c["session_durability"] == "disk" for c in restores))
                        for mode in ("off", "on"):
                            self.assertEqual(report[mode]["unknown_suffix"]["code"], "session_not_found")

    def test_worker_session_resources_are_wired_lazily(self):
        source = (Path(gate.__file__).parents[1] / "apps/cuda_ds41_replica.cu").read_text()
        startup, requests = source.split("const auto session_preflight=", 1)
        self.assertNotIn("session_disk.open(", startup)
        self.assertNotIn("default_disk_budget(", source)
        self.assertIn("std::unique_ptr<SessionResources> session_resources", startup)
        self.assertIn("std::optional<SessionRequest> session_request", startup)
        self.assertNotIn("sc::Names session_names", startup)
        self.assertNotIn("on_evict", source)
        self.assertIn("LaunchSeed<decltype(imported0)> session_seed_holder(session_config.enabled,state_file,", startup)
        replace = startup.split("session_seed_holder.before_replace(imported0,session_config.enabled,state_file);", 1)[1]
        self.assertLess(replace.index("imported0 ="), replace.index("imported_path ="))
        self.assertNotIn("seed_reference", source)
        before_initialize, initialize = startup.split("const auto session_initialize=", 1)
        self.assertNotIn("GetEnvironmentStringsW", source)
        self.assertNotIn("session_launch_env", source)
        self.assertNotIn("compat_environment_record()", before_initialize)
        self.assertIn("sc::compat_environment_record()", initialize)
        self.assertLess(initialize.index("=launch_runtime"), initialize.index("apply_env()"))
        self.assertLess(initialize.index("apply_env()"), initialize.index("sc::compat_environment_record()"))
        session_dispatch = requests.split("session_used=true;", 1)[1].split("} else {", 1)[0]
        self.assertLess(session_dispatch.index("=launch_runtime"), session_dispatch.index("apply_env()"))
        self.assertLess(session_dispatch.index("apply_env()"), session_dispatch.index("session_preflight()"))
        self.assertIn("session_resources->disk.ensure_open()", requests)
        self.assertNotIn("session_resources->names.turn(c.id)", requests)
        self.assertIn("sc::prepare_ram_command(c,session_resources->ram,session_resources->names", requests)
        preflight = requests.split("const auto session_execute=", 1)[0]
        self.assertIn("const auto& session_seed=session_seed_holder.materialize(imported0)", preflight)
        self.assertNotIn("disk.contains(c.id)", preflight)
        self.assertNotIn("disk.advanced", source)
        self.assertEqual(preflight.count("sc::restore("), 1)
        self.assertIn('if(c.op=="restore")', preflight)
        self.assertIn("session_error.value_or(sc::protocol_error(error))", requests)

    def test_full_prompt_references_use_same_worker_fixture_prompt_and_chat(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gate, "generated_dump"):
            out = Path(tmp)
            workers = [SimpleNamespace(req=mock.Mock(return_value={})) for _ in range(2)]
            first, restarted = workers
            gate.full_prompt(first, out, "first", [4, 5], "on", steps=0)
            gate.full_prompt(first, out, "decode", [4, 5], "on", steps=64)
            gate.full_prompt(first, out, "same", [4, 5], "on", steps=64)
            gate.full_prompt(first, out, "different", [7, 8], "on", steps=64)
            gate.full_prompt(restarted, out, "restart", [4, 5], "on", steps=64)
            for worker in workers:
                references = {}
                for call in worker.req.call_args_list:
                    f = call.kwargs
                    key = (str(f["seqs_file"]), f["chat_prompt"], f["chats"])
                    if f["phases"] == "chats0":
                        references[key] = f["chat_steps"]
                        self.assertNotIn("session", f)
                    else:
                        self.assertEqual(f["phases"], "schats0f3")
                        self.assertGreaterEqual(references.get(key, -1), f["chat_steps"])
            calls = [c.kwargs for c in first.req.call_args_list]
            refs = [c for c in calls if c["phases"] == "chats0"]
            self.assertEqual(len(refs), 3)  # zero rows, full decode, different prompt
            self.assertEqual(refs[0]["seqs_file"], refs[1]["seqs_file"])
            self.assertNotEqual(refs[1]["seqs_file"], refs[2]["seqs_file"])
            self.assertEqual(len(list(out.glob("prompt-*.tsv"))), 2)
            self.assertEqual(restarted.req.call_args_list[0].kwargs["phases"], "chats0")

    def test_single_device_is_caller_selected(self):
        for value in ("0", "GPU-test-device"):
            with mock.patch.dict(gate.os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                self.assertEqual(gate.gate_device(), value)
        for value in ("", "-1", "0,1"):
            with mock.patch.dict(gate.os.environ, {"CUDA_VISIBLE_DEVICES": value}):
                with self.assertRaises(AssertionError):
                    gate.gate_device()

    def test_worker_cache_modes_use_distinct_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = mock.Mock()
            args = SimpleNamespace(budget=budget, exe="worker", model="m", receipt="r", seed="s",
                draft="d", arena_gib=74.8, max_positions=16384, fixture="f", host_mib=512,
                out=Path(tmp), extra_flags=gate.extra_flags("--host-margin-gib 1 --max-pinned-gib 70"))
            for mode in (False, "off", "default", "on", "explicit", "auto", True):
                ready = {"session_config": {"session_cache": True, "session_explicit_opt_in": False,
                    "session_route": "rtx-tp1-decode-rows", "session_off_reason": ""}, "env": {"STRIX_SESSION_CACHE": None}}
                with mock.patch.dict(gate.os.environ, {"STRIX_SESSION_CACHE": "0"}), \
                     mock.patch.object(gate.Worker, "wait"), mock.patch.object(Path, "read_text", return_value=gate.json.dumps(ready)):
                    worker = gate.Worker(args, Path(tmp) / str(mode), sessions=mode)
                try:
                    cmd = budget.start.call_args.args[0]
                    for name, value in (("--draft-vocab", "full"), ("--residency-save", "none"), ("--residency-seed", "none"),
                                        ("--arena-gib", "74.8"), ("--host-margin-gib", "1"),
                                        ("--max-pinned-gib", "70")):
                        self.assertEqual(cmd.count(name), 1)
                        self.assertEqual(cmd[cmd.index(name) + 1], value)
                    self.assertEqual("--session-cache" in cmd, mode in ("off", "on", "explicit"))
                    self.assertEqual("--serve-session-mib" in cmd, mode is True)
                    if mode == "explicit":
                        self.assertEqual(cmd[cmd.index("--session-cache") + 1], "on")
                        self.assertEqual(cmd[cmd.index("--session-disk") + 1], "off")
                        self.assertEqual(cmd[cmd.index("--session-ram-bytes") + 1], str(512 * 1024**2))
                    if mode == "on":
                        self.assertEqual(cmd[cmd.index("--session-cache") + 1], "on")
                    if mode == "off":
                        self.assertEqual(cmd[cmd.index("--session-cache") + 1], "off")
                    if mode == "auto":
                        self.assertNotIn("STRIX_SESSION_CACHE", budget.start.call_args.kwargs["env"])
                        options = {v for v in cmd if v.startswith("--session-")}
                        self.assertEqual(options, {"--session-dir", "--session-max-parked"})
                        self.assertEqual(cmd[cmd.index("--session-dir") + 1], str(Path(tmp) / "auto-tier"))
                        self.assertEqual(cmd[cmd.index("--session-max-parked") + 1], "1")
                finally:
                    worker.log.close()

    def test_automatic_default_report_refuses_opt_in_disabled_and_wrong_route(self):
        ready = dict(session_config=dict(session_cache=True, session_explicit_opt_in=False,
            session_route="rtx-tp1-decode-rows", session_off_reason=""), env={"STRIX_SESSION_CACHE": None})
        self.assertEqual(gate.default_policy(ready), ready["session_config"])
        for field, value in (("session_cache", False), ("session_explicit_opt_in", True),
                             ("session_route", "unknown"), ("session_off_reason", "baseline")):
            changed = dict(ready, session_config=dict(ready["session_config"], **{field: value}))
            with self.assertRaises(AssertionError):
                gate.default_policy(changed)
        with self.assertRaises(AssertionError):
            gate.default_policy(dict(ready, env={"STRIX_SESSION_CACHE": "0"}))

    @unittest.skipUnless(gate.os.name == "nt", "Windows job containment")
    def test_child_environment_really_unsets_inherited_cache(self):
        budget = gate.Budget(10)
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.dict(gate.os.environ, {"STRIX_SESSION_CACHE": "0"}):
                dest = Path(tmp) / "environment.json"
                env = gate.os.environ.copy()
                env.pop("STRIX_SESSION_CACHE")
                env["STRIX_SESSION_ENV_TEST"] = "unicode-\u03bb"
                proc = budget.start([sys.executable, "-c",
                    "import json, os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps([os.environ.get('STRIX_SESSION_CACHE'), os.environ['STRIX_SESSION_ENV_TEST']]))",
                    str(dest)], env=env)
                self.assertEqual(proc.wait(2), 0)
                budget.forget(proc)
                self.assertEqual(gate.json.loads(dest.read_text()), [None, "unicode-\u03bb"])
        finally:
            budget.close()

    def test_generated_dump_selects_committed_full_vocab_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, dest = Path(tmp) / "raw", Path(tmp) / "generated"
            width = 129280 * 4
            # Two prompt rows, a rejected row in window 1, then final decode row.
            Path(str(raw) + ".tokens.u32").write_bytes(gate.struct.pack("<6I", 1, 2, 3, 4, 99, 5))
            Path(str(raw) + ".logits.f32").write_bytes(b"".join(bytes([i]) * width for i in range(6)))
            Path(str(raw) + ".events.jsonl").write_text(
                gate.json.dumps(dict(event="window", phase="schats0f3", rows=3, accepted=1)) + "\n" +
                gate.json.dumps(dict(event="window", phase="schats0f3", rows=1, accepted=0)) + "\n")
            for part in ("attention.bin", "drafter.bin"):
                Path(str(raw) + "." + part).write_bytes(b"state")
            gate.generated_dump(raw, dest, 2, "on", steps=3)
            self.assertEqual(gate.generated_ids(dest), [3, 4, 5])
            self.assertEqual(Path(str(dest) + ".logits.f32").read_bytes(),
                             bytes([2]) * width + bytes([3]) * width + bytes([5]) * width)
            with self.assertRaises(AssertionError):
                gate.generated_dump(raw, dest, 2, "on", steps=4)

    def test_spec_request_requires_same_prompt_reference_in_both_orders(self):
        class CanonWorker:
            # Native worker_canon key: import state, seqs file, prompt, chat.
            def __init__(self, out):
                self.out, self.canon = out, {}

            def req(self, **fields):
                key = (fields.get("import_state", "seed"), fields.get("seqs_file", "fixture"),
                       fields["chat_prompt"], "0")
                plain = "chats0" in fields["phases"].split(",")
                if "schats0f3" in fields["phases"].split(","):
                    assert plain or key in self.canon, "plain_found=false"
                if plain:
                    self.canon[key] = fields["chat_steps"] + 1

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp))
            worker = CanonWorker(Path(tmp))
            worker.req(**gate.SETTINGS[0])
            # Reproduce the original baseline failure: earlier plain chat 0,
            # same import/file, different prompt length. Chat steps is not a key.
            with self.assertRaisesRegex(AssertionError, "plain_found=false"):
                worker.req(**dict(gate.SETTINGS[1], phases="schats0f3"))
            for setting in (0, 1):
                worker = CanonWorker(Path(tmp))
                with mock.patch.object(gate, "qualification_receipt", return_value={}), \
                     mock.patch.object(gate, "draft_receipt", return_value={}):
                    gate.ordinary_phase(args, worker, setting, baseline=True)
                self.assertEqual(set(k[2] for k in worker.canon), {1, 3})

    def test_extra_flags_reject_overrides_and_invalid_budgets(self):
        self.assertEqual(gate.extra_flags(""), [])
        for value in ("--residency-save x", "--session-cache on", "--arena-gib 74.8",
                      "--host-margin-gib", "--max-pinned-gib nan", "--max-pinned-gib inf",
                      "--host-margin-gib -1", "--host-margin-gib 1 --host-margin-gib 2"):
            with self.assertRaises(argparse.ArgumentTypeError):
                gate.extra_flags(value)

    def test_model_directory_guard_after_success_failure_and_startup_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models"
            model_dir.mkdir()
            model = model_dir / "model.gguf"
            model.touch()
            residency = Path(str(model) + ".residency")
            args = SimpleNamespace(model=str(model), out=root / "out")
            for fail in (False, True, "startup"):
                worker = mock.Mock()
                worker.quit.side_effect = lambda: residency.touch()
                worker.abort.side_effect = lambda: residency.touch()
                def action(w):
                    if fail:
                        raise RuntimeError("request failed")
                def start(*a, **kw):
                    if fail == "startup":
                        residency.touch()
                        raise RuntimeError("startup failed")
                    return worker
                with mock.patch.object(gate, "Worker", side_effect=start):
                    with self.assertRaisesRegex(AssertionError, "next to model"):
                        gate.run_worker(args, "phase", action)
                residency.unlink()
            # Existing neighboring files are allowed; clean phases pass.
            residency.touch()
            with mock.patch.object(gate, "Worker"):
                gate.run_worker(args, "phase", lambda w: "ok")

    def test_revised_phase_budgets_fit_window(self):
        for phase in gate.PHASES:
            expected = 74 + (5 if phase.startswith("auto-") else 90) + sum(gate.phase_allowances(phase)) + 30
            self.assertAlmostEqual(expected, 410.072 if phase.startswith("save-") else
                             299.296 if phase == "auto-save-target" else
                             297.96 if phase == "auto-save-dspark" else 164 if phase == "auto-restore" else
                             154 if phase == "auto-admission" else
                             249 if phase.startswith("restore-") else 239)
            self.assertLess(expected, 420)
            self.assertLessEqual(gate.phase_deadline(expected), 420)

        for phase in ("auto-save-target", "auto-save-dspark"):
            for load in (75, 100):
                expected = load + 5 + sum(gate.phase_allowances(phase)) + 30
                self.assertLessEqual(expected, 330)
                self.assertEqual(gate.phase_deadline(expected, 420), 420)

    def test_deadlines_scale_with_window_cost_and_reject_oversized_phase(self):
        fast = 194 + sum(gate.phase_allowances("save-off", 32))
        slow = 194 + sum(gate.phase_allowances("save-off", 64))
        self.assertGreater(slow, fast)
        self.assertGreater(gate.phase_deadline(slow), gate.phase_deadline(fast))
        self.assertEqual(gate.phase_deadline(slow), 420)
        self.assertLess(gate.phase_deadline(249), 300)
        for expected, explicit in ((420, None), (200, 199), (200, 421), (200, float("nan"))):
            with self.assertRaises(AssertionError):
                gate.phase_deadline(expected, explicit)

    def progress_worker(self, out):
        worker = gate.Worker.__new__(gate.Worker)
        worker.out = out
        worker.log_offset = 0
        worker.log_partial = b""
        worker.last_progress = None
        worker.proc = mock.Mock()
        worker.proc.poll.return_value = None
        worker.budget = mock.Mock()
        worker.budget.remaining.return_value = 100
        return worker

    def test_invalid_session_tokens_reach_worker_for_negative_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker = self.progress_worker(out)
            worker.spool = out / "spool"
            (worker.spool / "queue").mkdir(parents=True)
            (worker.spool / "done").mkdir()
            worker.seq, worker.results = 0, []
            worker.window_seconds, worker.stall_seconds, worker.hash_seconds = .064, 20, 90
            tokens = out / "overflow.ids"
            tokens.write_text("1 18446744073709551616")
            result = worker.spool / "done" / "0001.json"
            result.write_text('{"rc": 2, "fatal": false}')
            with mock.patch.object(worker, "wait") as wait:
                worker.req(expected=2, session="overflow", session_tokens=tokens)
            self.assertEqual(wait.call_args.kwargs["stall"], 20)
            self.assertGreater(wait.call_args.kwargs["seconds"], 120)

    def test_progress_reader_handles_partial_lines_and_filters_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker = self.progress_worker(out)
            log = out / "worker.log"
            record = dict(event="session_progress", request="0004", stage="prompt", done=96, total=6144)
            line = gate.json.dumps(record)
            log.write_text('noise\n' + gate.json.dumps(dict(record, request="0003")) + '\n' + line[:20])
            self.assertFalse(worker.read_progress("0004"))
            with log.open("a") as stream:
                stream.write(line[20:] + '\n')
            self.assertTrue(worker.read_progress("0004"))
            self.assertEqual(worker.last_progress, record)
            self.assertFalse(worker.read_progress("0004"))

    def test_stall_reports_last_completed_progress_before_global_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker = self.progress_worker(out)
            worker.last_progress = {"stage": "prompt", "done": 96}
            with mock.patch.object(worker, "read_progress", return_value=False), \
                 mock.patch.object(gate.time, "monotonic", side_effect=[0, 21]):
                with self.assertRaisesRegex(TimeoutError, "last progress:.*96"):
                    worker.wait(out / "missing", request="0004", stall=20, seconds=200)
            worker.budget.remaining.assert_not_called()

    def test_stalled_request_reaps_worker_via_phase_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker = self.progress_worker(out)
            worker.log = (out / "worker.log").open("wb")
            worker.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            worker.budget.end = time.monotonic()+5
            worker.last_progress = {"stage": "capture", "done": 0}
            args = SimpleNamespace(model=str(out / "model"), out=out)
            try:
                with mock.patch.object(gate, "Worker", return_value=worker):
                    with self.assertRaisesRegex(TimeoutError, "last progress:.*capture"):
                        gate.run_worker(args, "phase", lambda w:
                            w.wait(out / "missing", request="0004", stall=.05, seconds=3))
                self.assertIsNotNone(worker.proc.poll())
                self.assertTrue(worker.log.closed)
                worker.budget.forget.assert_called_once_with(worker.proc)
            finally:
                if worker.proc.poll() is None:
                    worker.proc.kill()
                    worker.proc.wait(2)
                worker.log.close()

    def test_progress_resets_idle_timer_but_cannot_extend_request_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            worker = self.progress_worker(out)
            with mock.patch.object(worker, "read_progress", return_value=True), \
                 mock.patch.object(gate.time, "monotonic", side_effect=[0, 19, 38, 57]), \
                 mock.patch.object(gate.time, "sleep"):
                with self.assertRaises(TimeoutError):
                    worker.wait(out / "missing", request="0004", stall=20, seconds=50)

    def test_deadline_kills_worker_without_wait_call(self):
        budget = gate.Budget(3)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        budget.register(proc)
        try:
            # Model a blocked driver: only the watchdog can clean up the worker.
            proc.wait(timeout=2.5)
            self.assertIsNotNone(proc.returncode)
            with self.assertRaises(TimeoutError):
                budget.remaining()
        finally:
            budget.close()
        self.assertFalse(budget.watchdog.is_alive())

    def test_close_reaps_worker_on_exception(self):
        budget = gate.Budget(10)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        budget.register(proc)
        budget.close()
        self.assertIsNotNone(proc.poll())

    @unittest.skipUnless(gate.os.name == "nt", "Windows job containment")
    def test_start_crossing_deadline_is_reaped(self):
        budget = gate.Budget(10)
        real_start = budget.job.start
        children = []
        def crossing(*args, **kwargs):
            proc = real_start(*args, **kwargs)
            children.append(proc)
            budget.end = time.monotonic() + 1
            return proc
        try:
            with mock.patch.object(budget.job, "start", side_effect=crossing):
                with self.assertRaises(TimeoutError):
                    budget.start([sys.executable, "-c", "import time; time.sleep(30)"])
            self.assertIsNotNone(children[0].returncode)
            self.assertFalse(budget.workers)
        finally:
            budget.close()

    @unittest.skipUnless(gate.os.name == "nt", "Windows job containment")
    def test_contained_launch_redirects_output_and_closes_job(self):
        budget = gate.Budget(10)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.log"
            try:
                with path.open("wb") as log:
                    proc = budget.start([sys.executable, "-c",
                        "import sys; print('stdout'); print('stderr', file=sys.stderr)"],
                        stdout=log, stderr=subprocess.STDOUT)
                    self.assertEqual(proc.wait(2), 0)
                    budget.forget(proc)
                self.assertIn("stdout", path.read_text())
                self.assertIn("stderr", path.read_text())
                sleeper = budget.start([sys.executable, "-c", "import time; time.sleep(30)"])
                # Kill-on-close protects even an unregistered running process.
                budget.forget(sleeper)
                budget.close()
                self.assertIsNotNone(sleeper.wait(2))
            finally:
                if not budget.done.is_set():
                    budget.close()

    @unittest.skipUnless(gate.os.name == "nt", "Windows job containment")
    def test_delayed_startup_across_actual_deadline(self):
        budget = gate.Budget(3)
        real_start = budget.job.start
        created, release = threading.Event(), threading.Event()
        children, errors = [], []
        def delayed(*args, **kwargs):
            proc = real_start(*args, **kwargs)
            children.append(proc)
            created.set()
            release.wait(6)
            return proc
        def launch():
            try:
                budget.start([sys.executable, "-c", "import time; time.sleep(30)"])
            except BaseException as error:
                errors.append(error)
        # Keep this test runner alive while observing the real hard-deadline
        # callback. The worker must already be dead with start() still blocked.
        with mock.patch.object(budget.job, "start", side_effect=delayed), \
             mock.patch.object(gate.os, "_exit") as hard_exit:
            thread = threading.Thread(target=launch)
            thread.start()
            try:
                self.assertTrue(created.wait(.8))
                time.sleep(max(0, budget.end - time.monotonic() + .1))
                self.assertTrue(thread.is_alive())
                self.assertIsNotNone(children[0].poll())
                hard_exit.assert_called_once_with(124)
            finally:
                release.set()
                thread.join(2)
                budget.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TimeoutError)
        self.assertFalse(budget.workers)

    def test_receipt_commit_is_last_and_requires_final_version(self):
        budget = gate.Budget(10)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            receipt = out / (gate.PHASES[-1] + ".json")
            for phase in gate.PHASES[:-1]:
                (out / (phase + ".json")).write_text(gate.json.dumps(
                    {"pass": True, "publication": gate.RECEIPT_PUBLICATION}))
            real_fsync, real_replace = gate.os.fsync, gate.os.replace
            operations = []
            def sync(fd):
                self.assertFalse(receipt.exists())
                operations.append("fsync")
                return real_fsync(fd)
            def commit(src, dst):
                self.assertEqual(operations, ["fsync", "fsync"])
                self.assertFalse(receipt.exists())
                operations.append("rename")
                return real_replace(src, dst)
            try:
                budget.close(stop_watchdog=False)
                with mock.patch.object(gate.os, "fsync", side_effect=sync), \
                     mock.patch.object(gate.os, "replace", side_effect=commit):
                    gate.publish_receipts(budget, time.monotonic(), 10, receipt,
                                          out, {"pass": True}, {})
                self.assertEqual(operations[-1], "rename")
                self.assertTrue(gate.accepted_receipt(receipt)["pass"])
                receipt.write_text('{"pass": true}')
                with self.assertRaisesRegex(AssertionError, "unfinished prerequisite"):
                    gate.accepted_receipt(receipt)
            finally:
                budget.close()

    def test_receipt_deadline_checked_after_fsync(self):
        budget = gate.Budget(10)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            receipt = out / "baseline-0.json"
            try:
                def late_sync(fd):
                    budget.end = time.monotonic() - .01
                with mock.patch.object(gate.os, "fsync", side_effect=late_sync):
                    with self.assertRaisesRegex(AssertionError, "receipt publication"):
                        gate.publish_receipts(budget, time.monotonic(), 10, receipt,
                                              out, {"pass": True}, {})
                self.assertFalse(receipt.exists())
                with self.assertRaises(FileNotFoundError):
                    gate.accepted_receipt(receipt)
            finally:
                budget.close()

    def test_blocked_formerly_post_publication_op_leaves_no_accepted_receipt(self):
        # gate.json used to be opened after pass:true was already visible.
        # A real hard exit must now leave no prerequisite that can be accepted.
        script = '''
import importlib.util
import sys
import time
from pathlib import Path
from unittest import mock
s = importlib.util.spec_from_file_location("gate", sys.argv[1])
g = importlib.util.module_from_spec(s)
s.loader.exec_module(g)
b = g.Budget(3)
out = Path(sys.argv[2])
for phase in g.PHASES[:-1]:
    (out / (phase + ".json")).write_text(g.json.dumps(
        {"pass": True, "publication": g.RECEIPT_PUBLICATION}))
real_open = Path.open
def blocked_open(path, *args, **kwargs):
    if path.name == "gate.json":
        time.sleep(30)
    return real_open(path, *args, **kwargs)
b.close(stop_watchdog=False)
with mock.patch.object(Path, "open", new=blocked_open):
    g.publish_receipts(b, time.monotonic(), 3, out / (g.PHASES[-1] + ".json"),
                       out, {"pass": True}, {})
'''
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, "-c", script, gate.__file__, tmp],
                                    capture_output=True, timeout=6)
            self.assertEqual(result.returncode, 124, result.stderr)
            self.assertIn(b"whole-run deadline", result.stdout)
            with self.assertRaises(FileNotFoundError):
                gate.accepted_receipt(Path(tmp) / (gate.PHASES[-1] + ".json"))

    @unittest.skipUnless(gate.os.name == "nt", "Windows job containment")
    def test_failed_start_closes_log(self):
        budget = gate.Budget(10)
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(budget=budget, exe="missing", model="m", receipt="r", seed="s",
                draft="d", arena_gib=80, max_positions=16384, fixture="f", host_mib=512)
            try:
                with mock.patch.object(budget.job, "start", side_effect=OSError("start failed")):
                    with self.assertRaises(OSError):
                        gate.Worker(args, Path(tmp) / "worker")
                self.assertFalse(budget.workers)
                # Windows refuses removing an open log handle.
                (Path(tmp) / "worker" / "worker.log").unlink()
            finally:
                budget.close()

    def test_phases_require_prior_save_and_baseline(self):
        self.assertEqual(gate.PHASES, ("baseline-0", "baseline-1", "candidate-0", "candidate-1",
            "save-off", "restore-off", "save-on", "restore-on", "auto-save-target", "auto-save-dspark", "auto-restore", "auto-admission"))
        for phase in ("auto-save-target", "auto-save-dspark"):
            self.assertEqual(gate.prerequisites(phase), list(gate.PHASES[:8]))
        self.assertEqual(gate.prerequisites("auto-restore"), ["auto-save-target", "auto-save-dspark"])
        for mode in ("off", "on", "explicit"):
            self.assertEqual(gate.prerequisites("restore-" + mode), ["save-" + mode])
            self.assertEqual(gate.prerequisites("save-" + mode), ["candidate-0", "candidate-1"])
        self.assertEqual(gate.prerequisites("candidate-0"), ["baseline-0", "baseline-1"])
        with self.assertRaises(AssertionError):
            gate.Budget(421)

    def test_baseline_receipts_ignore_timing_but_reject_numeric_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, dest = Path(tmp) / "events.jsonl", Path(tmp) / "receipt.json"
            def write(fnv, elapsed):
                events.write_text(gate.json.dumps(dict(event="chats0", pos=10, fed=1,
                    argmax=2, logits_fnv=fnv, wall=elapsed)) + "\n" +
                    gate.json.dumps(dict(event="chat_end", digest="abc", toks=[1, 2])) + "\n")
                gate.qualification_receipt(events, dest)
                return dest.read_bytes()
            reference = write("0123", 1)
            self.assertEqual(reference, write("0123", 99))
            self.assertNotEqual(reference, write("0124", 1))
            events.write_text('{}\n')
            with self.assertRaises(AssertionError):
                gate.qualification_receipt(events, dest)

    @staticmethod
    def draft_record():
        return dict(phase="schats0f3", pos=17, t=7, k=3, rows=4, a=0,
                    d=[1, 2, 3, 4, 5], cb=["3f000000"] * 5, g=[9, 8, 7, 6],
                    c=[.5] * 5, draft_ms=1)

    def test_draft_receipt_requires_bits_ids_and_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace, dest = Path(tmp) / "trace", Path(tmp) / "receipt"
            def write(record):
                trace.write_text(gate.json.dumps(record) + "\n")
                gate.draft_receipt(trace, dest, speculative=True)
                return dest.read_bytes()
            record = self.draft_record()
            reference = write(record)
            self.assertEqual(reference, write(dict(record, c=[.1] * 5, draft_ms=99)))
            for key, value in (("cb", ["3f000000"] * 4 + ["3f000001"]),
                               ("d", [1, 2, 3, 4, 6]), ("a", 1)):
                self.assertNotEqual(reference, write(dict(record, **{key: value})))
            for key in ("cb", "d", "a"):
                incomplete = dict(record)
                del incomplete[key]
                with self.assertRaisesRegex(AssertionError, "incomplete"):
                    write(incomplete)
            for bits in (["3f000000"] * 4, ["not bits"] * 5):
                with self.assertRaises(AssertionError):
                    write(dict(record, cb=bits))
            trace.write_text("")
            with self.assertRaisesRegex(AssertionError, "missing/unexpected"):
                gate.draft_receipt(trace, dest, speculative=True)
            self.assertEqual(gate.draft_receipt(trace, dest, speculative=False)["windows"], 0)

    def test_baseline_candidate_arms_reject_drafter_changes(self):
        class TraceWorker:
            def __init__(self, out, change=None):
                self.out, self.change, self.calls = out, change, []
                out.mkdir(parents=True)

            def req(worker, **fields):
                worker.calls.append(fields)
                Path(fields["json"]).write_text(gate.json.dumps(
                    dict(event="window", logits_fnv="0123")) + "\n" +
                    gate.json.dumps(dict(event="chat_end", digest="same")) + "\n")
                speculative = "schats0f3" in fields["phases"]
                record = GateTests.draft_record()
                if worker.change:
                    record.update(worker.change)
                Path(fields["draft_trace"]).write_text(gate.json.dumps(record) + "\n" if speculative else "")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(out=out, references=[out / f"candidate-{i}" / "reference" for i in (0, 1)])
            for setting in (0, 1):
                baseline = TraceWorker(out / f"baseline-{setting}")
                gate.ordinary_phase(args, baseline, setting, baseline=True)
                self.assertTrue(all("draft_trace" in c and c["qual"] for c in baseline.calls))
                for n, change in enumerate((None, {"cb": ["3f000001"] * 5},
                                           {"d": [1, 2, 3, 4, 6]}, {"a": 1})):
                    candidate = TraceWorker(out / f"candidate-{setting}-{n}", change)
                    with mock.patch.object(gate, "equal_dumps", return_value={}):
                        if change is None:
                            report = gate.ordinary_phase(args, candidate, setting, baseline=False)
                            self.assertIn("drafter_trace", report["reference"])
                        else:
                            with self.assertRaisesRegex(AssertionError, "baseline drafter trace mismatch"):
                                gate.ordinary_phase(args, candidate, setting, baseline=False)

    def test_receipt_distinguishes_output_contract_and_evidence(self):
        contract = gate.ORDINARY_OUTPUT_CONTRACT
        self.assertEqual(contract["stdout_protocol_result_bytes"], "identical to the labelled baseline")
        self.assertIn("on for rtx-tp1-decode-rows", contract["default_session_policy"])
        self.assertIn("not compared", contract["stderr_comparison"])
        self.assertIn("candidate-internal", contract["drafter_state_evidence"])

    def test_restore_only_runs_selected_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), suffix="suffix")
            worker = mock.Mock()
            for mode in ("off", "on", "explicit"):
                worker.reset_mock()
                with mock.patch.object(gate, "equal_dumps", return_value={}), \
                     mock.patch.object(gate, "negatives"), mock.patch.object(gate, "ordinary"):
                    report = gate.restarted_worker(args, worker, mode)
                self.assertEqual(list(report), [mode])
                restores = [c for c in worker.req.call_args_list if c.kwargs.get("session_op") == "resume"]
                self.assertEqual(len(restores), 1)
                self.assertEqual(restores[0].kwargs["session_path"], Path(tmp) / mode / "checkpoint")

    def test_save_checkpoint_is_not_republished_by_later_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=Path(tmp), suffix="suffix", prompt_tail="prompt")
            worker = mock.Mock()
            with mock.patch.object(gate, "equal_dumps", return_value={}), \
                 mock.patch.object(gate, "ordinary"):
                gate.first_worker(args, worker, "off")
            calls = [c.kwargs for c in worker.req.call_args_list]
            saves = [c for c in calls if c.get("session_durability") == "disk"]
            self.assertEqual(len(saves), 1)
            self.assertEqual(saves[0]["session_path"], Path(tmp) / "off" / "checkpoint")
            index = calls.index(saves[0])
            self.assertTrue(any(c.get("session") == "A" and c.get("session_dump") for c in calls[index+1:]))

    def test_byte_compare_rejects_non_argmax_logit_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            for base in (a, b):
                Path(str(base) + ".tokens.u32").write_bytes(bytes(64 * 4))
                Path(str(base) + ".logits.f32").write_bytes(bytes(64 * 129280 * 4))
                for part in gate.PARTS[2:]:
                    Path(str(base) + "." + part).write_bytes(b"state")
            gate.equal_dumps(a, b)
            path = Path(str(b) + ".logits.f32")
            with path.open("r+b") as out:
                out.seek((37 * 129280 + 100001) * 4)
                out.write(b"\x01")
            with self.assertRaisesRegex(AssertionError, "logits.f32"):
                gate.equal_dumps(a, b)

    def test_final_canonical_state_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            Path(str(a) + ".frontier.bin").write_bytes(b"state-A")
            Path(str(b) + ".frontier.bin").write_bytes(b"state-B")
            with self.assertRaisesRegex(AssertionError, "frontier.bin"):
                gate.equal_dumps(a, b, ("frontier.bin",))


if __name__ == "__main__":
    unittest.main()
