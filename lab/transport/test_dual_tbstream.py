import contextlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import dual_tbstream as dual


class Clock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def pause(self, duration):
        self.value += duration


class ProcessFixture:
    def __init__(self):
        self.processes = []
        self.files = []
        self.signals = []

    def popen(self, command, **kwargs):
        self.files += [kwargs["stdout"], kwargs["stderr"]]
        if command == ["launch-fails"]:
            raise OSError("second launch failed")
        fixture = self

        class Process:
            pid = 1000 + len(fixture.processes)
            returncode = None
            waits = 0
            published = False

            def poll(self):
                if not self.published:
                    if command == ["done"]:
                        kwargs["stdout"].write(b'{"measured_seconds":1.0}')
                        self.returncode = 0
                    elif command == ["fail"]:
                        self.returncode = 7
                    elif command == ["bad-json"]:
                        kwargs["stdout"].write(b"not-json")
                        self.returncode = 0
                    elif command == ["large"]:
                        kwargs["stdout"].write(b"x" * (dual.OUTPUT_LIMIT + 1))
                    kwargs["stdout"].flush()
                    self.published = True
                return self.returncode

            def wait(self, timeout):
                self.waits += 1
                if self.returncode is None:
                    raise subprocess.TimeoutExpired(command, timeout)
                return self.returncode

        process = Process()
        self.processes.append(process)
        self.asserted_new_session = kwargs["start_new_session"]
        return process

    def kill(self, pid, sig):
        self.signals.append((pid, sig))
        for process in self.processes:
            if process.pid == pid and sig == getattr(signal, "SIGKILL", 9) and process.returncode is None:
                process.returncode = -9

    def run(self, commands, clock=None, timeout=0.03):
        clock = clock or Clock()
        return dual._run_lanes(commands, timeout_seconds=timeout, popen=self.popen,
                               now=clock.now, pause=clock.pause, kill_group=self.kill)


class DualRunnerTests(unittest.TestCase):
    def assert_cleaned(self, fixture):
        self.assertTrue(all(file.closed for file in fixture.files))
        self.assertTrue(all(process.returncode is not None and process.waits for process in fixture.processes))
        for process in fixture.processes:
            self.assertIn((process.pid, getattr(signal, "SIGKILL", 9)), fixture.signals)

    def test_success_reaps_and_stops_groups_after_leaders_exit(self):
        fixture = ProcessFixture()
        outputs, wall = fixture.run([["done"], ["done"]])
        self.assertEqual([output["measured_seconds"] for output in outputs], [1, 1])
        self.assertEqual(wall, 0)
        self.assertTrue(fixture.asserted_new_session)
        self.assert_cleaned(fixture)

    def test_timeout_failure_output_and_second_launch_cleanup(self):
        for commands, expected in [([["hang"], ["hang"]], TimeoutError),
                                   ([["hang"], ["fail"]], RuntimeError),
                                   ([["hang"], ["large"]], RuntimeError),
                                   ([["done"], ["bad-json"]], ValueError),
                                   ([["hang"], ["launch-fails"]], OSError)]:
            with self.subTest(commands=commands):
                fixture = ProcessFixture()
                with self.assertRaises(expected):
                    fixture.run(commands)
                self.assert_cleaned(fixture)

    def test_clock_regression_and_late_success_do_not_pass(self):
        for late in (False, True):
            fixture = ProcessFixture()
            calls = 0

            def now():
                nonlocal calls
                calls += 1
                return (1 if late else -1) if calls >= 4 else 0

            with self.assertRaises((TimeoutError, RuntimeError)):
                dual._run_lanes([["done"], ["done"]], timeout_seconds=0.1,
                                popen=fixture.popen, now=now, pause=lambda _: None,
                                kill_group=fixture.kill)
            self.assert_cleaned(fixture)

    def test_invalid_deadline_and_git_temp_reject_before_launch(self):
        for timeout in (0, -1, float("nan"), float("inf"), 3601):
            fixture = ProcessFixture()
            with self.assertRaises(ValueError):
                fixture.run([["done"], ["done"]], timeout=timeout)
            self.assertFalse(fixture.processes)
        with tempfile.TemporaryDirectory() as root:
            path = pathlib.Path(root)
            (path / ".git").mkdir()
            (path / "tmp").mkdir()
            fixture = ProcessFixture()
            with mock.patch.object(dual.tempfile, "gettempdir", return_value=str(path / "tmp")):
                with self.assertRaises(ValueError):
                    fixture.run([["done"], ["done"]])
            self.assertFalse(fixture.processes)

    def test_output_declares_unknown_overlap_unqualified_estimate(self):
        args = dual.parser().parse_args(["client", "--left-device", "/dev/a", "--right-device", "/dev/b"])
        with mock.patch.object(dual, "_run_lanes", return_value=([{"measured_seconds": 1}, {"measured_seconds": 2}], 4)), contextlib.redirect_stdout(io.StringIO()) as capture:
            self.assertEqual(dual.launch_clients(args), 0)
        report = json.loads(capture.getvalue())
        self.assertEqual(report["measured_overlap"], "UNKNOWN")
        self.assertFalse(report["common_measurement_start_barrier"])
        self.assertFalse(report["aggregate_performance_qualified"])
        self.assertEqual(report["aggregate_duration_estimate_seconds"], 2)
        self.assertNotIn("qualified_seconds", report)

    @unittest.skipUnless(os.name == "posix", "native baselines require POSIX process groups")
    def test_actual_local_process_timeout_kills_and_reaps_no_network(self):
        children = []

        def popen(command, **kwargs):
            process = subprocess.Popen(command, **kwargs)
            children.append(process)
            return process

        with self.assertRaises(TimeoutError):
            dual._run_lanes([[sys.executable, "-c", "import time; time.sleep(60)"] for _ in range(2)],
                            timeout_seconds=0.1, popen=popen)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(process.returncode is not None for process in children))


if __name__ == "__main__":
    unittest.main()
