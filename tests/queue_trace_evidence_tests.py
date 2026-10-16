import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("queue_trace_evidence", Path(__file__).resolve().parents[1] / "tools/qualification/queue_trace_evidence.py")
EVIDENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVIDENCE)


def span(ticket, queue, begin, end, position=0):
    return dict(ticket=ticket, queue=queue, request_id=7, session_epoch=9,
                execution_identity=11, position=position, step_id=ticket, layer=0,
                logical_rows=1, launcher_returned=True, gpu_event_completed=True,
                timing_known=True, begin_ns=begin, end_ns=end)


class QueueTraceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.json"
        self.trace = dict(profiling_only=True, scope="launcher-queue-spans-including-gaps-and-waits",
                          common_origin="same-device-HIP-timing-event", clock_domain=1, device_binding=2,
                          timing_resolution_approx_ns=1000, dropped=0, capture_skipped=0,
                          dependency_dropped=0, api_errors=0, observer_failed=False,
                          all_recorded_events_completed=True,
                          spans=[span(1,1,0,10), span(2,2,5,15), span(3,1,2,8), span(4,1,20,25,1)])

    def collect(self):
        self.path.write_text(json.dumps(dict(device_queue_trace=self.trace)))
        return EVIDENCE.collect(self.path)

    def test_overlapping_and_nested_spans_do_not_double_count_one_queue(self):
        receipt = self.collect()
        self.assertEqual(receipt["observed"]["observed_queue_span_union_ns"], 20)
        self.assertEqual(receipt["observed"]["observed_distinct_queue_span_overlap_ns"], 5)
        self.assertEqual(receipt["observed"]["observed_envelope_without_recorded_span_ns"], 5)
        self.assertEqual(receipt["observed"]["per_queue_union_ns"],
                         [dict(queue=1,union_ns=15), dict(queue=2,union_ns=10)])
        self.assertEqual(len(receipt["per_frontier"]), 2)
        self.assertEqual(receipt["per_frontier"][0]["observed"]["observed_queue_span_union_ns"], 15)
        self.assertTrue(receipt["all_reported_spans_timed_and_returned"])
        for key in ("kernel_busy_ns", "overlap_of_kernels_ns", "device_critical_path_ns",
                    "native_execution_independently_verified", "qualification"):
            self.assertIsNone(receipt[key])

    def test_partial_dropped_and_error_traces_never_claim_complete_coverage(self):
        for key in ("dropped", "capture_skipped", "dependency_dropped", "api_errors"):
            with self.subTest(key=key):
                self.trace[key] = 1
                self.assertFalse(self.collect()["all_reported_spans_timed_and_returned"])
                self.trace[key] = 0
        row = self.trace["spans"][0]
        row.update(timing_known=False,gpu_event_completed=False,begin_ns=None,end_ns=None)
        receipt = self.collect()
        self.assertEqual(receipt["unknown_timing_spans"], 1)
        self.assertFalse(receipt["all_reported_spans_timed_and_returned"])
        self.trace["spans"] = []
        self.assertFalse(self.collect()["all_reported_spans_timed_and_returned"])

    def test_zero_length_touching_and_disjoint_frontiers(self):
        self.trace["spans"] = [span(1,1,0,5),span(2,2,5,10),span(3,2,7,7)]
        receipt = self.collect()
        self.assertEqual(receipt["observed"]["observed_queue_span_union_ns"], 10)
        self.assertEqual(receipt["observed"]["observed_distinct_queue_span_overlap_ns"], 0)

    def test_invalid_metadata_timestamps_and_duplicate_tickets_reject(self):
        initial = copy.deepcopy(self.trace)
        for mutate in (lambda t: t.update(clock_domain=0),
                       lambda t: t.update(profiling_only=1),
                       lambda t: t.update(common_origin="host-clock"),
                       lambda t: t["spans"][1].update(ticket=1),
                       lambda t: t["spans"][0].update(begin_ns=20),
                       lambda t: t["spans"][0].update(begin_ns=True),
                       lambda t: t["spans"][0].update(gpu_event_completed=False),
                       lambda t: t["spans"][0].update(timing_known=False)):
            self.trace = copy.deepcopy(initial)
            mutate(self.trace)
            with self.assertRaises(ValueError):
                self.collect()
        self.path.write_text('{"device_queue_trace":{},"device_queue_trace":{}}')
        with self.assertRaises(ValueError):
            EVIDENCE.collect(self.path)

    def test_output_is_exclusive_and_partial_trace_has_distinct_exit(self):
        self.collect()
        output = Path(self.temp.name) / "receipt.json"
        args = ["--input",str(self.path),"--output",str(output)]
        self.assertEqual(EVIDENCE.main(args),0)
        original = output.read_bytes()
        with self.assertRaises(FileExistsError):
            EVIDENCE.main(args)
        self.assertEqual(output.read_bytes(),original)
        self.trace["dropped"]=1
        self.collect()
        self.assertEqual(EVIDENCE.main(args[:-1]+[str(Path(self.temp.name)/"partial.json")]),3)


if __name__ == "__main__":
    unittest.main()
